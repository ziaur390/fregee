"""Runtime registry.

Every runtime exposes the same interface - ``predict(x) -> logits`` - so the
benchmark can treat TorchScript, ONNX fp32 and ONNX int8 interchangeably. The
comparison is only meaningful because all three are exported from one set of
trained weights (see :mod:`mlserve.export`).

Loading is cached per runtime name. The cache is what makes ``cold_start_ms``
and ``warm_start_ms`` different numbers, and that difference is itself reported
in the benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from mlserve.config import ARTIFACTS, RUNTIME_BY_NAME
from mlserve.model import INPUT_SHAPE


class Predictor(Protocol):
    """The only interface the rest of the codebase depends on."""

    name: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits for a batch of shape (N, 1, 8, 8)."""


def _validate(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:  # single sample without a batch axis
        x = x[None, ...]
    if x.ndim != 4 or x.shape[1:] != INPUT_SHAPE:
        raise ValueError(f"expected shape (N, {INPUT_SHAPE}), got {x.shape}")
    return x


@dataclass
class TorchScriptPredictor:
    name: str
    module: object

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        x = _validate(x)
        with torch.no_grad():
            out = self.module(torch.from_numpy(x))
        return out.numpy()


@dataclass
class OnnxPredictor:
    name: str
    session: object
    input_name: str = field(default="input")

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = _validate(x)
        return self.session.run(None, {self.input_name: x})[0]


_cache: dict[str, Predictor] = {}
_load_times: dict[str, float] = {}


def artifact_path(runtime_name: str) -> Path:
    spec = RUNTIME_BY_NAME.get(runtime_name)
    if spec is None:
        known = ", ".join(sorted(RUNTIME_BY_NAME))
        raise KeyError(f"unknown runtime {runtime_name!r}; known: {known}")
    return ARTIFACTS / spec.file


def artifact_size_bytes(runtime_name: str) -> int:
    path = artifact_path(runtime_name)
    return path.stat().st_size if path.exists() else 0


def load(runtime_name: str) -> Predictor:
    """Load (or return the cached) predictor for ``runtime_name``."""
    if runtime_name in _cache:
        return _cache[runtime_name]

    # artifact_path validates the name and raises the readable KeyError, so it
    # runs before the dict lookup that would otherwise raise a bare 'key'.
    path = artifact_path(runtime_name)
    spec = RUNTIME_BY_NAME[runtime_name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python tasks.py export` (or `make export`) first."
        )

    started = time.perf_counter()
    if spec.kind == "torchscript":
        import torch

        module = torch.jit.load(str(path))
        module.eval()
        predictor: Predictor = TorchScriptPredictor(name=runtime_name, module=module)
    elif spec.kind == "onnx":
        import onnxruntime as ort

        # One thread per session keeps latency comparable with the single-threaded
        # torch configuration set in config.set_seed.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        predictor = OnnxPredictor(
            name=runtime_name, session=session, input_name=session.get_inputs()[0].name
        )
    else:  # pragma: no cover - guarded by the config table
        raise ValueError(f"unsupported runtime kind: {spec.kind}")

    _load_times[runtime_name] = (time.perf_counter() - started) * 1000.0
    _cache[runtime_name] = predictor
    return predictor


def load_time_ms(runtime_name: str) -> float:
    """Milliseconds the initial load took. 0.0 if never loaded."""
    return _load_times.get(runtime_name, 0.0)


def is_loaded(runtime_name: str) -> bool:
    return runtime_name in _cache


def loaded_runtimes() -> list[str]:
    return sorted(_cache)


def clear_cache() -> None:
    """Drop cached predictors. Used by tests and by cold-start measurement."""
    _cache.clear()
    _load_times.clear()


def available_runtimes() -> list[str]:
    """Runtime names whose artifact exists on disk, in a stable order."""
    return sorted(name for name in RUNTIME_BY_NAME if artifact_path(name).exists())


def describe() -> list[dict[str, object]]:
    """Inventory for the ``/models`` endpoint and the report."""
    rows: list[dict[str, object]] = []
    for name, spec in RUNTIME_BY_NAME.items():
        rows.append(
            {
                "runtime": name,
                "kind": spec.kind,
                "file": spec.file,
                "present": artifact_path(name).exists(),
                "loaded": is_loaded(name),
                "size_bytes": artifact_size_bytes(name),
                "load_time_ms": round(load_time_ms(name), 3),
            }
        )
    return rows
