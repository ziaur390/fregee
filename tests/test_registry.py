"""Runtime registry tests.

The registry is the seam the whole benchmark depends on: if a runtime can be
loaded but disagrees with the others, every latency number becomes meaningless.
So the parity assertions here are load-bearing, not decorative.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlserve import registry
from mlserve.config import RUNTIME_BY_NAME
from mlserve.model import load_digits_split

ALL_RUNTIMES = sorted(RUNTIME_BY_NAME)


@pytest.fixture(scope="module")
def test_batch() -> tuple[np.ndarray, np.ndarray]:
    split = load_digits_split(seed=1337)
    return split.x_test[:64], split.y_test[:64]


def test_all_three_runtimes_are_available() -> None:
    assert registry.available_runtimes() == ALL_RUNTIMES


@pytest.mark.parametrize("name", ALL_RUNTIMES)
def test_runtime_loads_and_predicts(name: str) -> None:
    split = load_digits_split(seed=1337)
    logits = registry.load(name).predict(split.x_test[:8])
    assert logits.shape == (8, 10), f"{name} returned {logits.shape}"


@pytest.mark.parametrize("name", ALL_RUNTIMES)
def test_runtime_handles_batch_size_one(name: str) -> None:
    """A dynamic batch axis that only works above size 1 is a broken export."""
    split = load_digits_split(seed=1337)
    assert registry.load(name).predict(split.x_test[:1]).shape == (1, 10)


def test_dynamic_batch_axis_is_real() -> None:
    """Batch sizes the export was not traced with must still work."""
    split = load_digits_split(seed=1337)
    for size in (1, 3, 17, 64, 129):
        for name in ALL_RUNTIMES:
            assert registry.load(name).predict(split.x_test[:size]).shape == (size, 10)


def test_runtimes_agree_on_labels(test_batch) -> None:
    """The parity claim. fp32 must be exact; int8 is allowed a small slack."""
    x, _ = test_batch
    reference = registry.load("torchscript").predict(x).argmax(axis=1)
    for name in ALL_RUNTIMES:
        labels = registry.load(name).predict(x).argmax(axis=1)
        disagreement = float(np.mean(labels != reference))
        tolerance = 0.02 if name == "onnx-int8" else 0.0
        assert disagreement <= tolerance, f"{name} disagrees on {disagreement:.2%} of samples"


def test_fp32_onnx_matches_torchscript_numerically(test_batch) -> None:
    """Same maths, different engine - logits should agree to float32 precision."""
    x, _ = test_batch
    a = registry.load("torchscript").predict(x)
    b = registry.load("onnx-fp32").predict(x)
    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)


def test_int8_is_smaller_than_fp32() -> None:
    """If quantisation produced no size win, the int8 row in the report is a lie."""
    assert registry.artifact_size_bytes("onnx-int8") < registry.artifact_size_bytes("onnx-fp32")


def test_load_is_cached() -> None:
    registry.load("onnx-fp32")
    first = registry.load_time_ms("onnx-fp32")
    registry.load("onnx-fp32")
    assert registry.load_time_ms("onnx-fp32") == first
    assert registry.is_loaded("onnx-fp32")


def test_clear_cache_forces_a_reload() -> None:
    registry.load("onnx-fp32")
    registry.clear_cache()
    assert not registry.is_loaded("onnx-fp32")
    registry.load("onnx-fp32")
    assert registry.is_loaded("onnx-fp32")


def test_unknown_runtime_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="unknown runtime"):
        registry.load("pytorch-mobile")


def test_shape_validation_rejects_bad_input() -> None:
    predictor = registry.load("onnx-fp32")
    with pytest.raises(ValueError, match="expected shape"):
        predictor.predict(np.zeros((4, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="expected shape"):
        predictor.predict(np.zeros((4, 1, 8, 9), dtype=np.float32))


def test_unbatched_input_is_promoted() -> None:
    """A single (1, 8, 8) sample should be accepted, not rejected."""
    predictor = registry.load("onnx-fp32")
    assert predictor.predict(np.zeros((1, 8, 8), dtype=np.float32)).shape == (1, 10)


def test_describe_reports_every_runtime() -> None:
    rows = registry.describe()
    assert {r["runtime"] for r in rows} == set(ALL_RUNTIMES)
    assert all(r["present"] for r in rows)
    assert all(r["size_bytes"] > 0 for r in rows)
