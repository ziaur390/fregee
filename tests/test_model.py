"""Model, split and reproducibility tests.

The reproducibility test is the important one: if the same seed stops producing
the same split, every number in results/REPORT.md becomes untraceable.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlserve.config import load_config
from mlserve.model import DigitCNN, build_model, load_digits_split, parameter_count

CFG = load_config("model")


def test_split_sizes_sum_to_dataset() -> None:
    split = load_digits_split(seed=42)
    total = sum(split.sizes.values())
    assert total == 1797, f"expected the full digits dataset, got {total}"


def test_split_is_reproducible_for_a_seed() -> None:
    a = load_digits_split(seed=7)
    b = load_digits_split(seed=7)
    np.testing.assert_array_equal(a.y_train, b.y_train)
    np.testing.assert_array_equal(a.x_test, b.x_test)


def test_split_changes_with_the_seed() -> None:
    a = load_digits_split(seed=7)
    b = load_digits_split(seed=8)
    assert not np.array_equal(a.y_test, b.y_test), "different seeds produced identical test sets"


def test_split_is_stratified() -> None:
    """Every class must appear in the test split, otherwise macro F1 is unstable."""
    split = load_digits_split(seed=1)
    assert set(np.unique(split.y_test)) == set(range(10))


def test_normalisation_uses_training_range() -> None:
    """Pixels arrive as 0-16 and must be scaled into 0-1."""
    split = load_digits_split(seed=1)
    for name, x in (("train", split.x_train), ("val", split.x_val), ("test", split.x_test)):
        assert x.min() >= 0.0, f"{name} has negative values"
        assert x.max() <= 1.0, f"{name} exceeds the trained range"


def test_input_shape_matches_the_contract() -> None:
    split = load_digits_split(seed=1)
    assert split.x_train.shape[1:] == (1, 8, 8)
    assert split.x_train.reshape(len(split.x_train), -1).shape[1] == 64


def test_model_forward_shape_and_dtype() -> None:
    import torch

    net = build_model(CFG)
    out = net(torch.zeros(4, 1, 8, 8))
    assert out.shape == (4, 10)
    assert out.dtype == torch.float32


def test_model_is_seed_reproducible() -> None:
    """Same seed must give identical initial weights, or training is not replayable."""
    import torch

    from mlserve.config import set_seed

    set_seed(99)
    a = build_model(CFG)
    set_seed(99)
    b = build_model(CFG)
    for (n1, p1), (n2, p2) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert n1 == n2
        assert torch.equal(p1, p2), f"weights differ at {n1}"


@pytest.mark.slow
def test_training_reaches_a_usable_accuracy() -> None:
    from mlserve.train import train_model

    _, metrics = train_model(CFG)
    assert metrics["test"]["accuracy"] > 0.9, metrics["test"]
    assert metrics["test"]["macro_f1"] > 0.9, metrics["test"]


def test_parameter_count_is_small_enough_for_ci() -> None:
    """Guard against someone swapping in a model that makes CI minutes explode."""
    assert parameter_count(DigitCNN()) < 500_000
