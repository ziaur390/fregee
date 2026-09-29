"""Tests for the statistics layer.

These matter more than the rest of the suite. Every number in the report passes
through this module, so a bug here is not a bug in a test helper - it is a wrong
result published in a document that claims to be reproducible.

The tests are written against properties that must hold regardless of the data
(the estimate lies inside its own interval, a paired comparison of identical
samples is exactly 1.0), not against hard-coded values that would only pin the
current implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlserve.bench.stats import (
    bootstrap_ci,
    bootstrap_ratio_ci,
    cohens_d,
    mean,
    observations_needed,
    percentile,
    summarize,
    tail_is_underpowered,
    wilson_interval,
)

# ------------------------------------------------------------------ basics


def test_percentile_matches_numpy() -> None:
    data = list(range(1, 101))
    assert percentile(data, 50) == pytest.approx(np.percentile(data, 50))
    assert percentile(data, 99) == pytest.approx(np.percentile(data, 99))


def test_percentile_of_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)


def test_mean_of_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        mean([])


# ------------------------------------------------------------------ bootstrap CI


def test_interval_contains_the_point_estimate() -> None:
    values = np.random.default_rng(0).normal(10, 2, size=200)
    interval = bootstrap_ci(values, np.mean, seed=1)
    assert interval.low <= interval.point <= interval.high


def test_interval_is_reproducible_for_a_seed() -> None:
    """Two runs of the same report must not disagree because of RNG drift."""
    values = np.random.default_rng(2).normal(5, 1, size=150)
    first = bootstrap_ci(values, np.mean, seed=99)
    second = bootstrap_ci(values, np.mean, seed=99)
    assert (first.low, first.high) == (second.low, second.high)


def test_different_seeds_move_the_interval_slightly() -> None:
    values = np.random.default_rng(3).normal(5, 3, size=200)
    a = bootstrap_ci(values, np.mean, seed=1)
    b = bootstrap_ci(values, np.mean, seed=2)
    assert a.low != b.low
    # Both must still bracket the same point estimate.
    assert a.point == b.point


def test_interval_narrows_as_n_grows() -> None:
    """The property that makes reporting n alongside every interval worthwhile."""
    rng = np.random.default_rng(4)
    small = bootstrap_ci(rng.normal(0, 1, size=50), np.mean, seed=0)
    large = bootstrap_ci(rng.normal(0, 1, size=5000), np.mean, seed=0)
    assert large.width < small.width


def test_single_observation_gives_a_degenerate_interval() -> None:
    interval = bootstrap_ci([2.5], np.mean, seed=0)
    assert (interval.point, interval.low, interval.high) == (2.5, 2.5, 2.5)
    assert interval.n == 1


def test_bootstrap_of_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([], np.mean)


def test_skewed_data_interval_is_asymmetric() -> None:
    """Why a percentile bootstrap rather than a normal approximation.

    Latency is right-skewed. A symmetric interval would put its lower bound in
    physically impossible territory and understate the upper one.
    """
    rng = np.random.default_rng(5)
    skewed = rng.lognormal(mean=0.0, sigma=1.2, size=500)
    interval = bootstrap_ci(skewed, np.mean, seed=0)
    below = interval.point - interval.low
    above = interval.high - interval.point
    assert above > below, "expected a right-skewed interval for lognormal data"


# ------------------------------------------------------------------ ratio CI


def test_ratio_of_identical_samples_is_exactly_one() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    interval = bootstrap_ratio_ci(values, values, seed=0)
    assert interval.point == pytest.approx(1.0)
    assert interval.low == pytest.approx(1.0)
    assert interval.high == pytest.approx(1.0)


def test_ratio_detects_a_real_doubling() -> None:
    base = list(np.linspace(10.0, 12.0, 200))
    twice = [2 * v for v in base]
    interval = bootstrap_ratio_ci(twice, base, seed=0)
    assert interval.point == pytest.approx(2.0)
    assert interval.low > 1.0, "a consistent 2x should exclude parity"


def test_paired_bootstrap_is_narrower_than_independent() -> None:
    """The reason this function resamples index positions, not the two sides.

    Two measurements from the same cell share a machine, a thermal state and a
    load level. Treating them as independent widens the interval enough to turn a
    real difference into 'no measurable difference'.
    """
    rng = np.random.default_rng(7)
    # Same underlying values, plus independent measurement noise on each side.
    shared = rng.normal(10, 1.0, size=300)
    a = shared + rng.normal(0, 0.3, size=300)
    b = shared + rng.normal(0, 0.3, size=300)

    paired = bootstrap_ratio_ci(a, b, seed=0)

    # Independently resampled comparison, for contrast.
    idx_a = rng.integers(0, a.size, size=(2000, a.size))
    idx_b = rng.integers(0, b.size, size=(2000, b.size))
    independent = np.percentile(a[idx_a].mean(axis=1) / b[idx_b].mean(axis=1), [2.5, 97.5])
    independent_width = float(independent[1] - independent[0])

    assert paired.width < independent_width


def test_ratio_bootstrap_handles_unequal_lengths() -> None:
    interval = bootstrap_ratio_ci([1.0, 2.0, 3.0], [1.0, 2.0], seed=0)
    assert math.isfinite(interval.point)


def test_ratio_of_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ratio_ci([], [1.0])


# ------------------------------------------------------------------ Wilson interval


def test_wilson_stays_inside_zero_and_one_at_the_extremes() -> None:
    """The exact failure mode of a normal approximation near p=1.

    270/270 correct is 1.0, where the normal interval gives a bound above 1.0 -
    which is not a probability. This is why Wilson is used for accuracy.
    """
    perfect = wilson_interval(270, 270)
    assert perfect.high <= 1.0
    assert perfect.low < 1.0, "a perfect score still has a non-degenerate lower bound"

    none = wilson_interval(0, 270)
    assert none.low >= 0.0
    assert none.high > 0.0


def test_wilson_contains_the_naive_proportion() -> None:
    interval = wilson_interval(263, 270)
    assert interval.low <= 263 / 270 <= interval.high


def test_wilson_narrows_with_more_samples() -> None:
    small = wilson_interval(95, 100)
    large = wilson_interval(9500, 10000)
    assert large.width < small.width


def test_wilson_requires_a_positive_total() -> None:
    with pytest.raises(ValueError, match="total"):
        wilson_interval(0, 0)


# ------------------------------------------------------------------ effect size


def test_cohens_d_sign_follows_the_direction() -> None:
    low = [1.0, 1.1, 0.9, 1.05, 0.95]
    high = [3.0, 3.1, 2.9, 3.05, 2.95]
    assert cohens_d(high, low) > 0
    assert cohens_d(low, high) < 0


def test_cohens_d_handles_zero_variance() -> None:
    """A runtime that returns identical timings must not raise a divide-by-zero."""
    assert cohens_d([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.0


def test_cohens_d_needs_two_points_per_group() -> None:
    assert math.isnan(cohens_d([1.0], [2.0, 3.0]))


# ------------------------------------------------------------------ tail power


def test_tail_power_thresholds_match_the_documented_table() -> None:
    """p95 needs 200 observations at the 10-in-tail standard, p99 needs 1000."""
    assert observations_needed(50, 10) == 20
    assert observations_needed(90, 10) == 100
    assert observations_needed(95, 10) == 200
    assert observations_needed(99, 10) == 1000


def test_tail_power_flags_a_thin_tail() -> None:
    assert tail_is_underpowered(100, 99, 10), "100 observations cannot support a p99"
    assert not tail_is_underpowered(1000, 99, 10)
    assert not tail_is_underpowered(200, 95, 10)
    assert tail_is_underpowered(199, 95, 10)


def test_one_observation_in_tail_is_not_enough() -> None:
    """The rule this replaced. At n=100 the p99 is the sample maximum wearing a
    percentile's name, so the standard is 10-in-tail, not 1."""
    assert tail_is_underpowered(100, 99, min_tail_observations=10)
    assert not tail_is_underpowered(100, 99, min_tail_observations=1)


def test_tail_power_rejects_impossible_percentiles() -> None:
    for bad in (0, 100, -5, 150):
        with pytest.raises(ValueError, match="percentile"):
            tail_is_underpowered(100, bad)


# ------------------------------------------------------------------ summarize


def test_summarize_returns_all_four_statistics() -> None:
    values = list(np.random.default_rng(11).normal(20, 5, size=300))
    result = summarize(values, seed=0)
    assert set(result) == {"mean", "p50", "p95", "p99"}
    for interval in result.values():
        assert interval.low <= interval.point <= interval.high
        assert interval.n == 300


def test_summarize_percentiles_are_ordered() -> None:
    """p50 <= p95 <= p99 must hold, or the table is internally inconsistent."""
    values = list(np.random.default_rng(12).exponential(3.0, size=500))
    result = summarize(values, seed=0)
    assert result["p50"].point <= result["p95"].point <= result["p99"].point
