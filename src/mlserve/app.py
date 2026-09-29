"""The serving API.

Run: ``python tasks.py serve``  (or ``make serve``)

Three design decisions worth stating, because each one is a thing that is
usually got wrong:

**Liveness and readiness are different endpoints.** ``/healthz`` reports only
that the process is up, and never touches the model or the database. ``/readyz``
answers whether this instance can serve traffic, and fails when the default
runtime cannot be loaded. Wiring the model check into ``/healthz`` is the classic
mistake: a bad deploy then makes the orchestrator kill a process that was
healthy, and the restart storm hides the real cause.

**Request logging happens after the response.** Persisting a row is done through
a background task, so database latency is not charged to the caller and does not
contaminate the p99 measured by the benchmark.

**Batch size is capped, not truncated.** A batch larger than ``max_batch`` is
rejected with 413 instead of being silently split or clipped, because client
code that sends 1000 rows and receives 64 predictions without an error is worse
than one that gets an explicit failure.
"""

from __future__ import annotations

import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from mlserve import __version__, db, registry
from mlserve.config import ARTIFACTS, ROOT, load_config
from mlserve.schema import (
    BatchPredictRequest,
    HealthResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

logger = logging.getLogger("mlserve")

SERVING_CFG = load_config("serving")
DEFAULT_RUNTIME: str = SERVING_CFG.get("default_runtime", "onnx-fp32")
MAX_BATCH: int = int(SERVING_CFG.get("max_batch", 64))
LOG_CFG = SERVING_CFG.get("request_log", {})
LOG_ENABLED: bool = (
    bool(LOG_CFG.get("enabled", True)) and os.environ.get("MLSERVE_REQUEST_LOG", "1") != "0"
)
LOG_SAMPLE_RATE: float = float(LOG_CFG.get("sample_rate", 1.0))
# Row cap enforced by the drift cron job via db.prune, not by the write path.
LOG_MAX_ROWS: int = int(LOG_CFG.get("max_rows", 50_000))

# Latency buckets are tuned to a CPU-served toy model: sub-millisecond through
# to the point where something is clearly wrong.
_LATENCY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

REQUESTS = Counter(
    "mlserve_requests_total",
    "Total HTTP requests handled.",
    ["endpoint", "runtime", "status"],
)
LATENCY = Histogram(
    "mlserve_request_duration_seconds",
    "Wall-clock latency of the inference call, excluding persistence.",
    ["endpoint", "runtime"],
    buckets=_LATENCY_BUCKETS,
)
BATCH_SIZE = Histogram(
    "mlserve_batch_size",
    "Samples per inference call.",
    ["runtime"],
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
VALIDATION_ERRORS = Counter(
    "mlserve_validation_errors_total",
    "Requests rejected by request-contract validation.",
    ["endpoint"],
)
INFERENCE_ERRORS = Counter(
    "mlserve_inference_errors_total",
    "Inference calls that raised.",
    ["runtime"],
)
PREDICTIONS = Counter(
    "mlserve_predictions_total",
    "Predictions emitted, by predicted class.",
    ["runtime", "predicted_class"],
)
MODEL_LOADED = Gauge(
    "mlserve_model_loaded",
    "1 when the runtime is loaded in this process.",
    ["runtime"],
)
MODEL_SIZE_BYTES = Gauge(
    "mlserve_model_size_bytes",
    "Size of the runtime artifact on disk.",
    ["runtime"],
)
DB_WRITE_ERRORS = Counter(
    "mlserve_db_write_errors_total",
    "Failed attempts to persist a served request.",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    for row in registry.describe():
        MODEL_LOADED.labels(runtime=str(row["runtime"])).set(0)
        MODEL_SIZE_BYTES.labels(runtime=str(row["runtime"])).set(float(row["size_bytes"]))
    logger.info(
        "mlserve %s up: default_runtime=%s max_batch=%d request_log=%s",
        __version__,
        DEFAULT_RUNTIME,
        MAX_BATCH,
        LOG_ENABLED,
    )
    yield


app = FastAPI(
    title="mlserve-ops",
    version=__version__,
    summary="A small model service used to compare serving runtimes under a reproducible harness.",
    lifespan=lifespan,
)

if SERVING_CFG.get("cors_origins"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(SERVING_CFG["cors_origins"]),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


def _resolve_runtime(runtime: str | None) -> str:
    name = runtime or DEFAULT_RUNTIME
    try:
        registry.load(name)
    except KeyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    MODEL_LOADED.labels(runtime=name).set(1)
    return name


def _json_safe(value):
    """Coerce a validation-error payload into something JSON can actually encode.

    Two things break naive serialisation here:

    1.  NaN and Infinity are not valid JSON. Pydantic echoes the offending input
        back inside the error, so a NaN in the request body made the *error
        response* unserialisable and turned an intended 422 into a 500.
    2.  Pydantic v2 puts the original exception object in each error's ``ctx``
        (``{'error': ValueError(...)}``), which is also not serialisable.

    Anything that is not a JSON primitive falls back to ``str``, so an unusual
    error shape degrades to a readable message instead of a 500.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return repr(value) if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError) -> JSONResponse:
    VALIDATION_ERRORS.labels(endpoint=request.url.path).inc()
    return JSONResponse(status_code=422, content=_json_safe({"detail": exc.errors()}))


def _persist(
    *,
    request_id: str,
    runtime: str,
    features: list[float],
    predicted_class: int,
    confidence: float,
    latency_ms: float,
    batch_size: int,
) -> None:
    """Background task: write one row.

    There is no pruning check here on purpose. An earlier version counted rows
    after every insert to enforce a row cap, which made writes quadratic in table
    size and stalled the HTTP benchmark at a few thousand rows. Pruning now lives
    in ``db.prune`` and runs from the drift cron job, off the user-facing path.
    """
    try:
        db.record_request(
            request_id=request_id,
            runtime=runtime,
            features=features,
            predicted_class=predicted_class,
            confidence=confidence,
            latency_ms=latency_ms,
            batch_size=batch_size,
        )
    except Exception:  # noqa: BLE001 - a logging failure must never fail a prediction
        DB_WRITE_ERRORS.inc()
        logger.exception("failed to persist request %s", request_id)


def _infer(runtime: str, x: np.ndarray, endpoint: str) -> tuple[np.ndarray, float]:
    """Run inference and record latency. Returns (logits, latency_ms)."""
    started = time.perf_counter()
    try:
        logits = registry.load(runtime).predict(x)
    except Exception:
        INFERENCE_ERRORS.labels(runtime=runtime).inc()
        raise
    latency_ms = (time.perf_counter() - started) * 1000.0
    LATENCY.labels(endpoint=endpoint, runtime=runtime).observe(latency_ms / 1000.0)
    BATCH_SIZE.labels(runtime=runtime).observe(len(x))
    return logits, latency_ms


def _to_predictions(logits: np.ndarray) -> list[Prediction]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    out: list[Prediction] = []
    for row in probs:
        cls = int(row.argmax())
        out.append(
            Prediction(
                predicted_class=cls,
                confidence=float(row[cls]),
                probabilities=[float(v) for v in row],
            )
        )
    return out


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    """Liveness. Deliberately checks nothing but this process."""
    return HealthResponse(status="ok", version=__version__)


@app.get("/readyz", response_model=ReadyResponse, tags=["ops"])
def readyz(response: Response) -> ReadyResponse:
    """Readiness. Fails when the default runtime cannot be loaded."""
    present = registry.available_runtimes()
    missing = [name for name in registry.RUNTIME_BY_NAME if name not in present]
    try:
        registry.load(DEFAULT_RUNTIME)
        MODEL_LOADED.labels(runtime=DEFAULT_RUNTIME).set(1)
        ready = True
    except Exception as exc:  # noqa: BLE001 - any load failure means not ready
        logger.error("readiness failed for %s: %s", DEFAULT_RUNTIME, exc)
        ready = False

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        status="ready" if ready else "not ready",
        runtimes_ready=present,
        runtimes_missing=missing,
        default_runtime=DEFAULT_RUNTIME,
    )


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models", tags=["ops"])
def models() -> dict[str, object]:
    """Inventory of the three runtimes: present, loaded, size, cold-load time."""
    return {
        "default_runtime": DEFAULT_RUNTIME,
        "max_batch": MAX_BATCH,
        "runtimes": registry.describe(),
    }


@app.get("/stats", tags=["ops"])
def stats() -> dict[str, object]:
    """Row count in the request log. Cheap proof the persistence path works."""
    return {"request_log_enabled": LOG_ENABLED, "requests_persisted": db.count_requests()}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(
    payload: PredictRequest,
    background: BackgroundTasks,
    runtime: str | None = Query(
        default=None, description=f"one of {sorted(registry.RUNTIME_BY_NAME)}"
    ),
) -> PredictResponse:
    name = _resolve_runtime(runtime)
    x = np.asarray(payload.features, dtype=np.float32).reshape(1, 1, 8, 8)
    logits, latency_ms = _infer(name, x, "/predict")
    predictions = _to_predictions(logits)

    first = predictions[0]
    PREDICTIONS.labels(runtime=name, predicted_class=str(first.predicted_class)).inc()
    REQUESTS.labels(endpoint="/predict", runtime=name, status="200").inc()

    if LOG_ENABLED:
        background.add_task(
            _persist,
            request_id=str(uuid.uuid4()),
            runtime=name,
            features=payload.features,
            predicted_class=first.predicted_class,
            confidence=first.confidence,
            latency_ms=latency_ms,
            batch_size=1,
        )

    return PredictResponse(
        predictions=predictions, runtime=name, batch_size=1, latency_ms=latency_ms
    )


@app.post("/predict/batch", response_model=PredictResponse, tags=["inference"])
def predict_batch(
    payload: BatchPredictRequest,
    background: BackgroundTasks,
    runtime: str | None = Query(
        default=None, description=f"one of {sorted(registry.RUNTIME_BY_NAME)}"
    ),
) -> PredictResponse:
    name = _resolve_runtime(runtime)
    n = len(payload.features)
    if n > MAX_BATCH:
        VALIDATION_ERRORS.labels(endpoint="/predict/batch").inc()
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"batch of {n} exceeds max_batch={MAX_BATCH}",
        )

    x = np.asarray(payload.features, dtype=np.float32).reshape(-1, 1, 8, 8)
    logits, latency_ms = _infer(name, x, "/predict/batch")
    predictions = _to_predictions(logits)

    for pred in predictions:
        PREDICTIONS.labels(runtime=name, predicted_class=str(pred.predicted_class)).inc()
    REQUESTS.labels(endpoint="/predict/batch", runtime=name, status="200").inc()

    if LOG_ENABLED:
        for sample, pred in zip(payload.features, predictions, strict=True):
            background.add_task(
                _persist,
                request_id=str(uuid.uuid4()),
                runtime=name,
                features=sample,
                predicted_class=pred.predicted_class,
                confidence=pred.confidence,
                latency_ms=latency_ms / n,
                batch_size=n,
            )

    return PredictResponse(
        predictions=predictions, runtime=name, batch_size=n, latency_ms=latency_ms
    )


@app.get("/", tags=["ops"])
def index() -> dict[str, object]:
    return {
        "service": "mlserve-ops",
        "version": __version__,
        "docs": "/docs",
        "endpoints": [
            "/healthz",
            "/readyz",
            "/metrics",
            "/models",
            "/stats",
            "/predict",
            "/predict/batch",
        ],
        "config_dir": str((ROOT / "configs").relative_to(ROOT)),
        "artifacts": str(ARTIFACTS.name),
    }
