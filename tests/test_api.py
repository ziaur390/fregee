"""API tests.

Covers the liveness/readiness split, batch-size ceiling, validation handling,
the metrics surface, and that persistence actually happens. The batch-ceiling
test and the readiness test are the two that protect real behaviour: one stops a
single caller from allocating an unbounded tensor, the other stops a bad deploy
from being reported as healthy.
"""

from __future__ import annotations

import pytest

from mlserve import db
from mlserve.app import MAX_BATCH
from mlserve.schema import FEATURE_COUNT

GOOD = [0.5] * FEATURE_COUNT


# ------------------------------------------------------------------ ops


def test_healthz_is_unconditional(client) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["version"]


def test_readyz_reports_ready_and_lists_runtimes(client) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["runtimes_missing"] == []
    assert set(body["runtimes_ready"]) == {"torchscript", "onnx-fp32", "onnx-int8"}


def test_readyz_returns_503_when_the_model_is_missing(client, monkeypatch) -> None:
    """The behaviour that makes readiness worth having."""
    from mlserve import registry as reg

    def boom(*_args, **_kwargs):
        raise FileNotFoundError("model.ts not found")

    monkeypatch.setattr(reg, "load", boom)
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not ready"


def test_healthz_stays_ok_when_the_model_is_missing(client, monkeypatch) -> None:
    """Liveness must not depend on the model, or the orchestrator restart-loops."""
    from mlserve import registry as reg

    def boom(*_args, **_kwargs):
        raise FileNotFoundError("model.ts not found")

    monkeypatch.setattr(reg, "load", boom)
    assert client.get("/healthz").status_code == 200


def test_models_inventory(client) -> None:
    body = client.get("/models").json()
    assert body["default_runtime"] == "onnx-fp32"
    assert body["max_batch"] == MAX_BATCH
    assert len(body["runtimes"]) == 3


def test_index_lists_the_surface(client) -> None:
    body = client.get("/").json()
    assert "/predict" in body["endpoints"]
    assert "/readyz" in body["endpoints"]


# ------------------------------------------------------------------ inference


def test_predict_default_runtime(client, sample_row) -> None:
    body = client.post("/predict", json={"features": sample_row}).json()
    assert body["runtime"] == "onnx-fp32"
    assert body["batch_size"] == 1
    assert len(body["predictions"]) == 1
    assert 0 <= body["predictions"][0]["predicted_class"] <= 9
    assert 0.0 <= body["predictions"][0]["confidence"] <= 1.0
    assert len(body["predictions"][0]["probabilities"]) == 10


def test_probabilities_sum_to_one(client, sample_row) -> None:
    body = client.post("/predict", json={"features": sample_row}).json()
    assert sum(body["predictions"][0]["probabilities"]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("runtime", ["torchscript", "onnx-fp32", "onnx-int8"])
def test_predict_every_runtime(client, sample_row, runtime: str) -> None:
    response = client.post(f"/predict?runtime={runtime}", json={"features": sample_row})
    assert response.status_code == 200
    assert response.json()["runtime"] == runtime


def test_runtimes_return_the_same_class_for_one_sample(client, sample_row) -> None:
    classes = {
        client.post(f"/predict?runtime={name}", json={"features": sample_row}).json()[
            "predictions"
        ][0]["predicted_class"]
        for name in ("torchscript", "onnx-fp32", "onnx-int8")
    }
    assert len(classes) == 1, f"runtimes disagree: {classes}"


def test_predict_unknown_runtime_is_400(client, sample_row) -> None:
    response = client.post("/predict?runtime=nonexistent", json={"features": sample_row})
    assert response.status_code == 400


def test_batch_predict(client, sample_row) -> None:
    rows = [sample_row, [0.1] * FEATURE_COUNT, [1.0] * FEATURE_COUNT]
    body = client.post("/predict/batch", json={"features": rows}).json()
    assert body["batch_size"] == 3
    assert len(body["predictions"]) == 3


def test_batch_predict_at_exactly_max_batch(client, sample_row) -> None:
    body = client.post("/predict/batch", json={"features": [sample_row] * MAX_BATCH}).json()
    assert body["batch_size"] == MAX_BATCH


def test_batch_larger_than_max_is_413(client, sample_row) -> None:
    """Rejected, not silently truncated. A caller must know rows were dropped."""
    response = client.post("/predict/batch", json={"features": [sample_row] * (MAX_BATCH + 1)})
    assert response.status_code == 413
    assert "exceeds max_batch" in response.json()["detail"]


def test_batch_of_one_matches_single_predict(client, sample_row) -> None:
    single = client.post("/predict", json={"features": sample_row}).json()
    batch = client.post("/predict/batch", json={"features": [sample_row]}).json()
    assert single["predictions"][0]["predicted_class"] == batch["predictions"][0]["predicted_class"]


# ------------------------------------------------------------------ validation failures


@pytest.mark.parametrize(
    "payload",
    [
        {"features": [0.5] * 10},
        {"features": [0.5] * 65},
        {"features": []},
        {"features": [2.0] * FEATURE_COUNT},
        {"features": [-1.0] * FEATURE_COUNT},
        {"features": [None] * FEATURE_COUNT},
        {},
    ],
)
def test_malformed_requests_are_422(client, payload) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_nan_is_rejected_over_http(client) -> None:
    """NaN survives a conv layer and still yields a confident argmax, so it must
    be refused at the edge.

    Sent as raw JSON text because Python's json module refuses to *serialise* NaN,
    so ``json=`` would raise client-side and never reach the server. NaN is not
    valid JSON at all, which is exactly why it needs a test: the parser accepted
    it and the error response then failed to serialise, turning a 422 into a 500.
    """
    values = [0.5] * FEATURE_COUNT
    values[0] = "NaN"
    body = '{"features": [' + ", ".join(str(v) for v in values) + "]}"
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, f"expected 422, got {response.status_code}"
    assert "nan" in response.text.lower(), "the offending value should be echoed back"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_literals_never_produce_a_500(client, literal: str) -> None:
    """A malformed request is a client error, never a server error."""
    values = [0.5] * FEATURE_COUNT
    values[0] = literal
    body = '{"features": [' + ", ".join(str(v) for v in values) + "]}"
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, f"{literal} produced {response.status_code}"


# ------------------------------------------------------------------ metrics + persistence


def test_metrics_exposes_the_expected_series(client, sample_row) -> None:
    client.post("/predict", json={"features": sample_row})
    text = client.get("/metrics").text
    for metric in (
        "mlserve_requests_total",
        "mlserve_request_duration_seconds",
        "mlserve_batch_size",
        "mlserve_predictions_total",
        "mlserve_model_size_bytes",
    ):
        assert metric in text, f"{metric} missing from /metrics"


def test_metrics_counters_move(client, sample_row) -> None:
    client.get("/metrics")  # ensure the registry is initialised
    client.post("/predict", json={"features": sample_row})
    client.post("/predict/batch", json={"features": [sample_row] * 4})
    text = client.get("/metrics").text
    assert 'mlserve_requests_total{endpoint="/predict",runtime="onnx-fp32",status="200"}' in text


def test_requests_are_persisted(client, sample_row) -> None:
    before = db.count_requests()
    client.post("/predict/batch", json={"features": [sample_row] * 3})
    assert db.count_requests() == before + 3


def test_stats_endpoint_reflects_the_log(client, sample_row) -> None:
    client.post("/predict", json={"features": sample_row})
    body = client.get("/stats").json()
    assert body["requests_persisted"] >= 1


def test_persistence_failure_does_not_fail_the_prediction(client, sample_row, monkeypatch) -> None:
    """A logging outage must not take down inference."""
    from mlserve import db as db_mod

    monkeypatch.setattr(
        db_mod, "record_request", lambda **_kw: (_ for _ in ()).throw(RuntimeError("db down"))
    )
    response = client.post("/predict", json={"features": sample_row})
    assert response.status_code == 200


def test_malformed_batch_does_not_count_as_a_validation_error(client, sample_row) -> None:
    """The counter must distinguish 'malformed' from 'rejected for size'.

    Asserts the real property - the number does not move - rather than looking for
    a zero series, because a Prometheus counter with no observations has no series
    to find and an absence assertion would pass for the wrong reason.
    """
    import re

    def batch_errors() -> float:
        text = client.get("/metrics").text
        match = re.search(
            r'mlserve_validation_errors_total\{endpoint="/predict/batch"\} ([0-9.]+)', text
        )
        return float(match.group(1)) if match else 0.0

    client.post("/predict/batch", json={"features": [sample_row] * (MAX_BATCH + 1)})
    after_rejection = batch_errors()
    assert after_rejection >= 1, "the oversize rejection should have been counted"

    client.post("/predict/batch", json={"features": [sample_row]})
    assert batch_errors() == after_rejection, "a valid batch moved the error counter"
