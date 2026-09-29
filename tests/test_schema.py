"""Request-contract tests.

These are the trust-boundary tests. Each one corresponds to a way malformed
input could otherwise reach the model: wrong length, non-finite values, values
outside the trained range, and unexpected fields.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlserve.schema import FEATURE_COUNT, BatchPredictRequest, PredictRequest

GOOD = [0.5] * FEATURE_COUNT


def test_accepts_a_well_formed_sample() -> None:
    req = PredictRequest(features=GOOD)
    assert len(req.features) == FEATURE_COUNT
    assert req.features[0] == 0.5


@pytest.mark.parametrize("bad_len", [0, 1, 63, 65, 128])
def test_rejects_wrong_feature_count(bad_len: int) -> None:
    with pytest.raises(ValidationError, match="expected 64 features"):
        PredictRequest(features=[0.5] * bad_len)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_values(bad: float) -> None:
    """NaN survives a conv layer and still produces a confident argmax."""
    payload = [0.5] * FEATURE_COUNT
    payload[7] = bad
    with pytest.raises(ValidationError):
        PredictRequest(features=payload)


@pytest.mark.parametrize("bad", [-0.01, 1.01, 16.0, 255.0])
def test_rejects_values_outside_the_trained_range(bad: float) -> None:
    """Raw 0-16 pixels must be rejected, not silently served wrong answers."""
    payload = [0.5] * FEATURE_COUNT
    payload[3] = bad
    with pytest.raises(ValidationError, match="outside trained range"):
        PredictRequest(features=payload)


def test_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(features=GOOD, unexpected="value")


def test_rejects_strings_that_look_numeric() -> None:
    payload = [0.5] * FEATURE_COUNT
    payload[0] = "0.5"
    with pytest.raises(ValidationError):
        PredictRequest(features=payload)


def test_accepts_boundary_values() -> None:
    payload = [0.0] * FEATURE_COUNT
    payload[0] = 1.0
    assert PredictRequest(features=payload).features[0] == 1.0


def test_batch_rejects_empty_list() -> None:
    with pytest.raises(ValidationError, match="at least one sample"):
        BatchPredictRequest(features=[])


def test_batch_reports_the_offending_row() -> None:
    with pytest.raises(ValidationError, match=r"features\[1\]"):
        BatchPredictRequest(features=[GOOD, [0.5] * 10])


def test_batch_accepts_mixed_valid_rows() -> None:
    req = BatchPredictRequest(features=[GOOD, [0.1] * FEATURE_COUNT, [1.0] * FEATURE_COUNT])
    assert len(req.features) == 3
