"""Request and response contracts.

This is the trust boundary. Everything that arrives from the network is validated
here before it reaches the model, and the validation is deliberately strict:

* exactly 64 features (8x8), because a shorter vector would otherwise be
  broadcast silently by numpy into a wrong-shaped batch;
* every value finite, because NaN propagates through a conv layer and produces a
  confident-looking but meaningless argmax;
* every value in [0, 1], because that is the range the model was trained on.
  Accepting out-of-range input would let a caller feed raw 0-16 pixels and get
  plausible answers from a model operating far outside its training support.
"""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

FEATURE_COUNT = 64
FEATURE_MIN = 0.0
FEATURE_MAX = 1.0

Feature = Annotated[float, Field(strict=True)]


def _check_feature(value: float, where: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{where}: value must be finite, got {value!r}")
    if not (FEATURE_MIN <= value <= FEATURE_MAX):
        raise ValueError(
            f"{where}: value {value} outside trained range "
            f"[{FEATURE_MIN}, {FEATURE_MAX}]. Normalise pixels by dividing by 16.0."
        )
    return float(value)


class PredictRequest(BaseModel):
    """One sample: 64 pixel values in [0, 1]."""

    model_config = ConfigDict(extra="forbid")

    features: list[Feature]

    @field_validator("features")
    @classmethod
    def _validate_features(cls, value: list[float]) -> list[float]:
        if len(value) != FEATURE_COUNT:
            raise ValueError(f"expected {FEATURE_COUNT} features, got {len(value)}")
        return [_check_feature(v, f"features[{i}]") for i, v in enumerate(value)]


class BatchPredictRequest(BaseModel):
    """1..max_batch samples. The upper bound is enforced by the route, not here."""

    model_config = ConfigDict(extra="forbid")

    features: list[list[Feature]]

    @field_validator("features")
    @classmethod
    def _validate_features(cls, value: list[list[float]]) -> list[list[float]]:
        if not value:
            raise ValueError("batch must contain at least one sample")
        out: list[list[float]] = []
        for row, sample in enumerate(value):
            if len(sample) != FEATURE_COUNT:
                raise ValueError(
                    f"features[{row}]: expected {FEATURE_COUNT} values, got {len(sample)}"
                )
            out.append([_check_feature(v, f"features[{row}][{i}]") for i, v in enumerate(sample)])
        return out


class Prediction(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: list[float]


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    runtime: str
    batch_size: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    runtimes_ready: list[str]
    runtimes_missing: list[str]
    default_runtime: str
