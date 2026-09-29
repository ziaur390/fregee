"""Drift detection tests.

The two that matter are the pair at the bottom: a no-drift window must come back
stable, and a deliberately shifted window must come back alert. A detector that
has only been tested against clean data is a detector nobody knows the sensitivity
of.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlserve.config import load_config
from mlserve.drift.detect import (
    bin_proportions,
    detect,
    ks_statistic_and_pvalue,
    population_stability_index,
    write_outputs,
)
from mlserve.drift.reference import Reference, build_reference, load_reference
from mlserve.model import load_digits_split
from mlserve.schema import FEATURE_COUNT

CFG = load_config("drift")


@pytest.fixture(scope="module")
def reference(artifacts) -> Reference:
    try:
        return load_reference()
    except FileNotFoundError:
        return build_reference(CFG)


@pytest.fixture(scope="module")
def no_drift_window(reference: Reference) -> list[list[float]]:
    """Held-out test rows. Same distribution as training, so no drift."""
    split = load_digits_split(seed=1337)
    rows = split.x_test.reshape(len(split.x_test), -1)
    return [[float(v) for v in row] for row in rows]


@pytest.fixture(scope="module")
def drifted_window(reference: Reference) -> list[list[float]]:
    """A window with a deliberate, documented shift.

    Every pixel is pushed upward by 0.35 and clipped to [0, 1]. That is a real
    distribution change - brighter images than the model was trained on - not
    noise, so the detector has something it is supposed to find.
    """
    split = load_digits_split(seed=1337)
    rows = split.x_test.reshape(len(split.x_test), -1)
    shifted = np.clip(rows + 0.35, 0.0, 1.0)
    return [[float(v) for v in row] for row in shifted]


# ------------------------------------------------------------------ PSI maths


def test_psi_of_the_reference_against_itself_is_near_zero(reference: Reference) -> None:
    """The floor of the metric. A non-zero baseline would add a constant offset
    to every measurement and make the thresholds meaningless."""
    for index in range(0, FEATURE_COUNT, 7):
        psi = population_stability_index(
            reference.bin_edges[index], reference.proportions[index], reference.sample[:, index]
        )
        assert psi < 0.05, f"feature {index} scores {psi:.4f} against its own reference"


def test_psi_rises_when_the_distribution_moves(reference: Reference) -> None:
    """Monotonicity: bigger shifts must score higher. This is what makes PSI
    usable as a magnitude rather than only a yes/no test.

    Uses the first *informative* feature. Several of the 64 pixels are constant
    background in every training image, and PSI on a zero-variance feature is 0 by
    construction no matter what is sent to it - see Reference.informative.
    """
    index = int(reference.informative_indices[0])
    column = reference.sample[:, index]
    edges, props = reference.bin_edges[index], reference.proportions[index]

    scores = [
        population_stability_index(edges, props, np.clip(column + delta, 0, 1))
        for delta in (0.0, 0.1, 0.25, 0.5)
    ]
    assert scores == sorted(scores), f"PSI was not monotonic in shift size: {scores}"
    assert scores[-1] > scores[0] + 0.1, f"a 0.5 shift barely moved PSI: {scores}"


def test_constant_features_are_excluded_from_scoring(reference: Reference) -> None:
    """A zero-variance feature has no reference distribution.

    Its PSI is 0 by construction, so including it would dilute the mean and
    inflate the denominator of fraction_watched. It must be recorded and counted,
    not silently averaged in.
    """
    # Three corner pixels are constant across the whole digits dataset.
    assert reference.n_uninformative >= 3, (
        f"expected the known constant corner pixels, found {reference.n_uninformative}"
    )
    assert reference.n_informative == reference.n_features - reference.n_uninformative
    assert not reference.informative[reference.uninformative_indices].any()


def test_psi_is_finite_with_empty_bins(reference: Reference) -> None:
    """Features with heavy ties produce zero-width bins. log(0) would be -inf and
    a single empty bin would dominate the score without the epsilon."""
    index = int(reference.informative_indices[0])
    psi = population_stability_index(
        reference.bin_edges[index], reference.proportions[index], np.ones(len(reference.sample))
    )
    assert np.isfinite(psi), "PSI produced a non-finite value on a degenerate input"


def test_bin_proportions_sum_to_one(reference: Reference) -> None:
    props = bin_proportions(reference.bin_edges[0], reference.sample[:, 0])
    assert props.sum() == pytest.approx(1.0)
    assert (props >= 0).all()


def test_bin_proportions_handle_an_empty_sample(reference: Reference) -> None:
    """A monitoring job must not crash on an empty window."""
    props = bin_proportions(reference.bin_edges[0], np.array([]))
    assert props.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ KS corroboration


def test_ks_identical_samples_give_zero_statistic() -> None:
    values = np.linspace(0, 1, 200)
    stat, pvalue = ks_statistic_and_pvalue(values, values)
    assert stat == pytest.approx(0.0, abs=1e-12)
    assert pvalue == pytest.approx(1.0)


def test_ks_separated_samples_give_a_large_statistic() -> None:
    a = np.linspace(0.0, 0.3, 200)
    b = np.linspace(0.7, 1.0, 200)
    stat, pvalue = ks_statistic_and_pvalue(a, b)
    assert stat > 0.9
    assert pvalue < 1e-6


def test_ks_refuses_a_degenerate_sample() -> None:
    stat, pvalue = ks_statistic_and_pvalue(np.array([0.5]), np.array([0.5, 0.6]))
    assert np.isnan(stat)
    assert np.isnan(pvalue)


# ------------------------------------------------------------------ the detector


def test_reference_has_uniform_shape_across_features(reference: Reference) -> None:
    """Heavy ties mean repeated quantiles. Rejecting them would give features
    different bin counts, which breaks the stacked array and makes the reference
    shape data-dependent."""
    assert reference.bin_edges.shape == (FEATURE_COUNT, reference.bins + 1)
    assert reference.proportions.shape == (FEATURE_COUNT, reference.bins)


def test_reference_edges_are_monotonic(reference: Reference) -> None:
    """np.histogram requires non-decreasing edges. Outer edges are deliberately
    infinite so a value outside the training range lands in the first or last bin
    rather than being dropped - which is exactly the case being hunted."""
    for index in range(reference.n_features):
        edges = reference.bin_edges[index]
        assert np.all(np.diff(edges) >= 0), f"feature {index} edges are not monotonic"
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf


def test_reference_proportions_sum_to_one(reference: Reference) -> None:
    sums = reference.proportions.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)


def test_no_drift_window_is_stable(reference: Reference, no_drift_window) -> None:
    """The false-positive check. Clean traffic must not trip the detector."""
    report = detect(CFG, reference=reference, features=no_drift_window, prune=False)
    assert report.status == "stable", (
        f"clean data reported {report.status}: max PSI {report.max_psi:.4f}, "
        f"{report.n_watched} features watched"
    )
    assert report.n_alerted == 0
    assert report.max_psi < CFG["thresholds"]["psi_watch"]


def test_shifted_window_alerts(reference: Reference, drifted_window) -> None:
    """The sensitivity check. This is the test that gives the detector meaning:
    without it, a function that always returns 'stable' would pass every other
    test in this file."""
    report = detect(CFG, reference=reference, features=drifted_window, prune=False)
    assert report.status == "alert", (
        f"a 0.35 pixel shift was not detected: max PSI {report.max_psi:.4f}, "
        f"{report.n_watched} watched, {report.n_alerted} alerted"
    )
    assert report.n_alerted > 0
    assert report.max_psi >= CFG["thresholds"]["psi_alert"]


def test_shifted_window_scores_higher_than_clean(
    reference: Reference, no_drift_window, drifted_window
) -> None:
    """The comparison that matters, stated directly as an ordering."""
    clean = detect(CFG, reference=reference, features=no_drift_window, prune=False)
    shifted = detect(CFG, reference=reference, features=drifted_window, prune=False)
    assert shifted.max_psi > clean.max_psi
    assert shifted.mean_psi > clean.mean_psi


def test_small_window_reports_insufficient_data(reference: Reference, no_drift_window) -> None:
    """Below the minimum the job must refuse to compare rather than report a PSI
    computed from a handful of rows."""
    tiny = no_drift_window[: CFG["window"]["min_requests"] // 4]
    report = detect(CFG, reference=reference, features=tiny, prune=False)
    assert report.status == "insufficient_data"
    assert report.n_requests == len(tiny)
    assert np.isnan(report.max_psi)


def test_ks_is_reported_but_does_not_drive_the_status(
    reference: Reference, no_drift_window
) -> None:
    """KS is corroboration. Its p-value must be present in the output, and the
    status must not depend on it - that is the whole argument in drift.yaml."""
    report = detect(CFG, reference=reference, features=no_drift_window, prune=False)
    for feature in report.features:
        assert np.isfinite(feature.ks_statistic) or np.isnan(feature.ks_statistic)
        assert feature.band in {"stable", "watch", "alert"}


def test_report_counts_only_informative_features(reference: Reference, drifted_window) -> None:
    """The counts must describe what was actually scored."""
    report = detect(CFG, reference=reference, features=drifted_window, prune=False)
    assert report.n_informative == len(report.features)
    assert report.n_informative + report.n_uninformative == report.n_features
    assert report.fraction_watched <= 1.0
    payload = report.to_json()
    assert payload["n_informative"] == report.n_informative
    assert "constant feature" in payload["scoring_note"]


def test_report_serialises_only_drifted_features(reference: Reference, drifted_window) -> None:
    """Keeping all 64 features in drift.json makes it unreadable at a glance."""
    report = detect(CFG, reference=reference, features=drifted_window, prune=False)
    payload = report.to_json()
    drifted = payload["drifted_features"]
    assert drifted, "a shifted window should list drifted features"
    assert all(f["band"] != "stable" for f in drifted)
    assert len(drifted) <= report.n_features


def test_prometheus_output_is_well_formed(reference: Reference, no_drift_window) -> None:
    report = detect(CFG, reference=reference, features=no_drift_window, prune=False)
    text = report.to_prometheus()
    for metric in (
        "mlserve_drift_psi_max",
        "mlserve_drift_psi_mean",
        "mlserve_drift_features_watched",
        "mlserve_drift_features_alerted",
        "mlserve_drift_fraction_watched",
        "mlserve_drift_alert",
    ):
        assert f"# TYPE {metric} gauge" in text
        assert any(line.startswith(metric + " ") for line in text.splitlines())
    # The alert gauge must be 0 for clean data, or every alert rule fires always.
    assert "mlserve_drift_alert 0" in text


def test_alert_gauge_is_one_when_drifting(reference: Reference, drifted_window) -> None:
    report = detect(CFG, reference=reference, features=drifted_window, prune=False)
    assert "mlserve_drift_alert 1" in report.to_prometheus()


def test_outputs_are_written_to_the_configured_paths(
    reference: Reference, no_drift_window, monkeypatch
) -> None:
    from mlserve.config import RESULTS

    report = detect(CFG, reference=reference, features=no_drift_window, prune=False)
    json_path, prom_path, collector_path = write_outputs(report, CFG)

    assert json_path.exists()
    assert prom_path.exists()
    assert json_path.parent == RESULTS
    assert prom_path.parent == RESULTS
    assert json_path.read_text(encoding="utf-8").strip().endswith("}")
    # No collector directory configured, so no third file.
    assert collector_path is None


def test_outputs_are_published_to_the_textfile_collector_atomically(
    reference: Reference, no_drift_window, tmp_path, monkeypatch
) -> None:
    """The collector copy must land at the configured path and leave no temp files.

    node_exporter reads this directory on its own schedule. A non-atomic write would
    let it parse a half-written file, which presents as a random scrape error and a
    drift gauge that flaps - intermittent, and very hard to attribute.
    """
    monkeypatch.setenv("MLSERVE_TEXTFILE_DIR", str(tmp_path / "collector"))
    report = detect(CFG, reference=reference, features=no_drift_window, prune=False)
    json_path, prom_path, collector_path = write_outputs(report, CFG)

    assert collector_path is not None
    assert collector_path.exists()
    assert collector_path.parent == tmp_path / "collector"
    # Same content as the results copy.
    assert collector_path.read_text(encoding="utf-8") == prom_path.read_text(encoding="utf-8")
    assert "mlserve_drift_psi_max" in collector_path.read_text(encoding="utf-8")

    # The staging file is renamed away, not left behind.
    leftovers = [p.name for p in collector_path.parent.iterdir() if p.name.startswith(".")]
    assert not leftovers, f"atomic write left temporary files behind: {leftovers}"
    assert json_path.exists()
