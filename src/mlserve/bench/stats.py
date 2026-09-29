"""Benchmark statistics.

Every number this repository reports comes with an interval from here. That is a
deliberate constraint, not decoration: a single p99 from a single run is a
sample, and reporting it as a fact is the most common way a benchmark misleads -
including the person who ran it.

Two interval strategies, chosen for the question being asked:

* :func:`bootstrap_ci` resamples the raw observations. Used for latency
  percentiles, where the distribution is skewed and a normal approximation is
  wrong.
* :func:`bootstrap_ratio_ci` resamples *pairs*, preserving the fact that two
  measurements in the same cell share a machine, a thermal state and a load
  level. Comparing two independently-bootstrapped intervals is the classic
  mistake that makes a real difference look like noise, or the reverse.

The number of observations matters and is reported alongside every interval: a
p99 estimated from 128 samples is a tail guess, not a tail measurement, and the
report says so where it applies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval on both sides."""

    point: float
    low: float
    high: float
    n: int
    confidence: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.point:.4g} [{self.low:.4g}, {self.high:.4g}] (n={self.n})"


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` in [0, 100]."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("percentile of an empty sequence")
    return float(np.percentile(arr, q))


def mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("mean of an empty sequence")
    return float(arr.mean())


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for ``statistic`` over ``values``.

    A percentile bootstrap rather than a normal approximation because latency is
    right-skewed: the mean sits above the median and the tail is long, so a
    symmetric interval would place a lower bound somewhere it cannot physically
    be and understate the upper one.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap of an empty sequence")
    if arr.size == 1:
        value = float(statistic(arr))
        return Interval(value, value, value, 1, 1.0 - alpha)

    rng = np.random.default_rng(seed)
    n = arr.size
    draws = rng.integers(0, n, size=(resamples, n))
    samples = np.array([statistic(arr[draw]) for draw in draws], dtype=float)

    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(statistic(arr)), float(low), float(high), n, 1.0 - alpha)


def bootstrap_ratio_ci(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI for ``mean(numerator) / mean(denominator)``.

    Resamples index *positions* and applies the same index to both sequences, so
    the resampled pairs are drawn together. This is what makes the interval
    reflect the paired structure of the design. Bootstrap the two independently
    and the interval comes out roughly twice as wide, which is how a genuine
    speedup ends up being reported as "not significant".
    """
    a = np.asarray(numerator, dtype=float)
    b = np.asarray(denominator, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("ratio bootstrap needs non-empty sequences")
    if a.size != b.size:
        # Fall back to independent resampling but keep the sample size explicit.
        rng = np.random.default_rng(seed)
        a_draws = a[rng.integers(0, a.size, size=(resamples, a.size))]
        b_draws = b[rng.integers(0, b.size, size=(resamples, b.size))]
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, a.size, size=(resamples, a.size))
        a_draws = a[idx]
        b_draws = b[idx]

    ratios = a_draws.mean(axis=1) / b_draws.mean(axis=1)
    point = float(a.mean() / b.mean())
    low, high = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(point, float(low), float(high), min(a.size, b.size), 1.0 - alpha)


def wilson_interval(successes: int, total: int, *, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a proportion.

    Used for accuracy rather than a normal approximation because accuracy here
    sits near 0.97, where the normal interval produces bounds above 1.0 and is
    simply wrong. Wilson stays inside [0, 1] at the extremes.
    """
    if total <= 0:
        raise ValueError("wilson_interval needs total > 0")
    # stdlib inverse-normal rather than scipy: this is the only place a z value is
    # needed, and a benchmark harness should not carry a dependency for one number.
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin), total, 1 - alpha)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised mean difference between two independent groups.

    Reported next to the raw ratio for one reason: a ratio says how much faster,
    this says how consistent that is relative to the spread. A 1.3x speedup with
    d=0.1 is a coin flip, not a result.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    pooled_var = ((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1)) / (
        x.size + y.size - 2
    )
    if pooled_var <= 0:
        return 0.0
    return float((x.mean() - y.mean()) / np.sqrt(pooled_var))


def summarize(
    values: Sequence[float], *, resamples: int = 2000, alpha: float = 0.05, seed: int = 0
):
    """The four statistics reported for every cell, each with a CI."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": bootstrap_ci(arr, np.mean, resamples=resamples, alpha=alpha, seed=seed),
        "p50": bootstrap_ci(
            arr, lambda v: np.percentile(v, 50), resamples=resamples, alpha=alpha, seed=seed
        ),
        "p95": bootstrap_ci(
            arr, lambda v: np.percentile(v, 95), resamples=resamples, alpha=alpha, seed=seed
        ),
        "p99": bootstrap_ci(
            arr, lambda v: np.percentile(v, 99), resamples=resamples, alpha=alpha, seed=seed
        ),
    }


def tail_is_underpowered(
    n_observations: int, percentile: float, min_tail_observations: int = 10
) -> bool:
    """True when a percentile cannot be estimated from this many observations.

    The test is how many observations are expected to land *in the tail* at or
    above the percentile being reported::

        expected_tail = n * (1 - percentile/100)

    and the rule is that you want at least ``min_tail_observations`` of them.

    Why 10 rather than the usual "at least 1": with one observation in the tail,
    the reported p99 is literally that observation, so it is the sample maximum
    wearing a percentile's name. Ten is the smallest count at which the estimate
    has any spread to it.

    At the 10-observation standard:

    ============  ================  ==========================
    percentile    observations      note
    ============  ================  ==========================
    p50           20                trivially satisfied
    p90           100
    p95           200
    p99           1000              out of reach for a quick run
    ============  ================  ==========================

    So a table can legitimately report a p50 and mean from a few hundred calls
    while the p95 and p99 columns in the same table are not measurements at all.
    That is the reason this function is called per percentile instead of once per
    row: collapsing them into a single "tail ok" flag hides exactly the
    distinction that matters.
    """
    if not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    return n_observations * (1 - percentile / 100.0) < min_tail_observations


def observations_needed(percentile: float, min_tail_observations: int = 10) -> int:
    """How many observations a given percentile needs to be worth reporting.

    Used by the report to state the shortfall rather than only flag it, so a
    reader knows whether it is 20 calls short or 20,000.

    Written as ``ceil(tail * 100 / (100 - p))`` rather than the equivalent-looking
    ``ceil(tail / (1 - p/100))`` because the second form is wrong on exact inputs:
    0.1 is not representable in binary, so ``10 / 0.1`` evaluates to
    100.00000000000001 and ``ceil`` returns 101 for the p90 case, which the report
    would then print as a requirement. Integer arithmetic keeps the documented
    thresholds exact.
    """
    if not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    numerator = min_tail_observations * 100
    denominator = 100 - percentile
    return int(np.ceil(numerator / denominator))
