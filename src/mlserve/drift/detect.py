"""Drift detection job.

Run: ``python tasks.py drift``  (or ``make drift``)

Compares the distribution of recently served request features against the
training reference and writes ``results/drift.json`` plus a Prometheus textfile.

Why PSI decides and KS only corroborates is argued in ``configs/drift.yaml``. In
short: KS is a hypothesis test, and with a few thousand observations across 64
features it reports significance for shifts far too small to affect the model,
while 64 uncorrected p-values produce about three false alarms per run by
construction. A detector that cries wolf gets muted. PSI is a magnitude with
conventional, sample-size-independent thresholds, so a PSI of 0.3 means the same
thing on a small window as on a large one.

This module also prunes the request log. Row-cap enforcement lives here rather
than on the write path because counting rows after every insert made writes
quadratic in table size - which is what stalled the HTTP benchmark.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlserve import db
from mlserve.config import RESULTS, ROOT, ensure_dirs, load_config, read_json, set_seed, write_json
from mlserve.drift.reference import Reference, flatten, load_reference

#: PSI adds a small epsilon to every proportion before taking the log, because
#: log(0) is -inf and a single empty bin would otherwise dominate the score.
#: 1e-6 is small enough not to distort a real shift and large enough to keep the
#: arithmetic finite.
PSI_EPSILON = 1e-6


@dataclass(frozen=True)
class FeatureDrift:
    index: int
    psi: float
    ks_statistic: float
    ks_pvalue: float
    watched: bool
    alerted: bool

    @property
    def band(self) -> str:
        if self.alerted:
            return "alert"
        return "watch" if self.watched else "stable"


@dataclass(frozen=True)
class DriftReport:
    status: str
    n_requests: int
    n_features: int
    n_informative: int
    n_uninformative: int
    uninformative_features: list[int]
    bins: int
    max_psi: float
    mean_psi: float
    n_watched: int
    n_alerted: int
    fraction_watched: float
    features: list[FeatureDrift]
    thresholds: dict
    pruned_rows: int
    reference_samples: int

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "n_requests": self.n_requests,
            "n_features": self.n_features,
            "n_informative": self.n_informative,
            "n_uninformative": self.n_uninformative,
            "uninformative_features": self.uninformative_features,
            "bins": self.bins,
            "max_psi": round(self.max_psi, 6),
            "mean_psi": round(self.mean_psi, 6),
            "n_watched": self.n_watched,
            "n_alerted": self.n_alerted,
            "fraction_watched": round(self.fraction_watched, 6),
            "reference_samples": self.reference_samples,
            "pruned_rows": self.pruned_rows,
            "thresholds": self.thresholds,
            "scoring_note": (
                "PSI is computed only over features that vary in the training split. "
                "A constant feature has zero variance, so it has no distribution to "
                "drift from and its PSI would be 0 regardless of the input - "
                "including it would dilute every average and inflate the denominator "
                "of fraction_watched."
            ),
            # Only the drifted features, so the file stays readable at 64 features.
            "drifted_features": [
                {
                    "index": f.index,
                    "psi": round(f.psi, 6),
                    "band": f.band,
                    "ks_statistic": round(f.ks_statistic, 6),
                    "ks_pvalue": round(f.ks_pvalue, 8),
                }
                for f in self.features
                if f.band != "stable"
            ],
        }

    def to_prometheus(self) -> str:
        """Textfile-collector format so drift alerts use the same stack as the API.

        Two deliberate choices:

        1.  ``mlserve_drift_measured`` distinguishes "the window was too small to
            compare" from "the window was compared and nothing moved". Without it,
            a job that has never run successfully looks identical to a healthy one -
            which is the failure mode `MLServeDriftDetectorStale` exists to catch, and
            a dashboard cannot catch it at all.
        2.  The PSI gauges are omitted rather than set to NaN when nothing was
            measured. A published NaN is not a value and it renders as a gap in a
            dashboard; omitting the series is honest, and the alert rules evaluate
            false either way, so an unmeasured window cannot page anyone.
        """
        measured = self.status != "insufficient_data"
        lines = [
            "# HELP mlserve_drift_measured 1 when a comparison was actually made, 0 when the window was too small.",
            "# TYPE mlserve_drift_measured gauge",
            f"mlserve_drift_measured {1 if measured else 0}",
            "# HELP mlserve_drift_requests Requests in the comparison window.",
            "# TYPE mlserve_drift_requests gauge",
            f"mlserve_drift_requests {self.n_requests}",
            "# HELP mlserve_drift_features_watched Features at or above psi_watch.",
            "# TYPE mlserve_drift_features_watched gauge",
            f"mlserve_drift_features_watched {self.n_watched}",
            "# HELP mlserve_drift_features_alerted Features at or above psi_alert.",
            "# TYPE mlserve_drift_features_alerted gauge",
            f"mlserve_drift_features_alerted {self.n_alerted}",
            "# HELP mlserve_drift_fraction_watched Share of scored features watched or worse.",
            "# TYPE mlserve_drift_fraction_watched gauge",
            f"mlserve_drift_fraction_watched {self.fraction_watched:.6f}",
            "# HELP mlserve_drift_alert 1 when the window trips the alert policy.",
            "# TYPE mlserve_drift_alert gauge",
            f"mlserve_drift_alert {1 if self.status == 'alert' else 0}",
        ]
        if measured:
            lines += [
                "# HELP mlserve_drift_psi_max Highest per-feature PSI across the window.",
                "# TYPE mlserve_drift_psi_max gauge",
                f"mlserve_drift_psi_max {self.max_psi:.6f}",
                "# HELP mlserve_drift_psi_mean Mean per-feature PSI across the window.",
                "# TYPE mlserve_drift_psi_mean gauge",
                f"mlserve_drift_psi_mean {self.mean_psi:.6f}",
            ]
        lines += [
            "# HELP mlserve_drift_last_run_unixtime Unix time of the last successful run.",
            "# TYPE mlserve_drift_last_run_unixtime gauge",
            f"mlserve_drift_last_run_unixtime {int(_now())}",
        ]
        return "\n".join(lines) + "\n"


def _now() -> float:
    import time

    return time.time()


# --------------------------------------------------------------------------- maths


def bin_proportions(edges: np.ndarray, column: np.ndarray) -> np.ndarray:
    """Share of ``column`` in each bin defined by ``edges``."""
    counts, _ = np.histogram(column, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.full(len(edges) - 1, 1.0 / (len(edges) - 1))
    return counts / total


def population_stability_index(
    edges: np.ndarray, reference: np.ndarray, sample: np.ndarray
) -> float:
    """PSI between a reference distribution and a sample, over fixed bin edges.

    The same edges are used for both sides. Re-deriving edges per window - a
    common mistake - makes a shift partly invisible, because the bins move with
    the data being measured. Edges come from the reference and stay fixed.
    """
    actual = bin_proportions(edges, sample)
    expected = np.clip(reference, PSI_EPSILON, None)
    observed = np.clip(actual, PSI_EPSILON, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def ks_statistic_and_pvalue(
    reference_sample: np.ndarray, sample: np.ndarray
) -> tuple[float, float]:
    """Two-sample KS statistic and p-value.

    Uses scipy rather than a hand-rolled implementation. The statistic itself is
    a few lines, but the p-value has known edge cases (small samples, ties,
    extreme statistic values) that scipy handles correctly and a quick
    reimplementation would not. Reimplementing a statistical test to avoid a
    dependency that is already installed is not laziness, it is risk.

    Recorded for corroboration only - PSI drives the alert, for the reasons
    argued in configs/drift.yaml.
    """
    from scipy.stats import ks_2samp

    if reference_sample.size < 2 or sample.size < 2:
        return float("nan"), float("nan")
    result = ks_2samp(reference_sample, sample)
    return float(result.statistic), float(result.pvalue)


# --------------------------------------------------------------------------- report


def detect(
    cfg: dict | None = None,
    *,
    reference: Reference | None = None,
    features: list[list[float]] | None = None,
    prune: bool = True,
) -> DriftReport:
    """Run one drift comparison.

    ``features`` may be supplied directly, which is how the tests drive both the
    no-drift and the shifted case without needing a populated database.
    """
    cfg = cfg or load_config("drift")
    reference = reference or load_reference()

    window_cfg = cfg["window"]
    thresholds = cfg["thresholds"]

    rows = features
    if rows is None:
        rows = db.recent_features(
            limit=int(window_cfg["max_requests"]),
            within_minutes=int(window_cfg["within_minutes"]),
        )

    pruned = 0
    if prune and rows is not features:
        pruned = db.prune(int(cfg["maintenance"]["prune_to_max_rows"]))

    if rows is None or len(rows) < int(window_cfg["min_requests"]):
        return DriftReport(
            status="insufficient_data",
            n_requests=0 if rows is None else len(rows),
            n_features=reference.n_features,
            n_informative=reference.n_informative,
            n_uninformative=reference.n_uninformative,
            uninformative_features=reference.uninformative_indices.tolist(),
            bins=reference.bins,
            max_psi=float("nan"),
            mean_psi=float("nan"),
            n_watched=0,
            n_alerted=0,
            fraction_watched=0.0,
            features=[],
            thresholds=thresholds,
            pruned_rows=pruned,
            reference_samples=reference.n_samples,
        )

    sample_matrix = flatten(np.asarray(rows, dtype=np.float64))

    drifts: list[FeatureDrift] = []
    # Only informative features are scored. See Reference.informative and the
    # scoring_note in the JSON output: a zero-variance feature has no reference
    # distribution, so its PSI is 0 by construction and would dilute the average.
    for index in reference.informative_indices:
        column = sample_matrix[:, index]
        psi = population_stability_index(
            reference.bin_edges[index], reference.proportions[index], column
        )
        ks_stat, ks_p = ks_statistic_and_pvalue(reference.sample[:, index], column)
        drifts.append(
            FeatureDrift(
                index=index,
                psi=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
                watched=psi >= float(thresholds["psi_watch"]),
                alerted=psi >= float(thresholds["psi_alert"]),
            )
        )

    psis = np.array([d.psi for d in drifts], dtype=float)
    n_watched = sum(1 for d in drifts if d.watched)
    n_alerted = sum(1 for d in drifts if d.alerted)
    scored = len(drifts) or 1
    fraction_watched = n_watched / scored

    # Decision policy. A single feature moving is noise; a fifth of them moving
    # together is a change in the input distribution.
    if n_alerted > 0 or fraction_watched > float(thresholds["max_fraction_watched"]):
        status = "alert"
    elif n_watched > 0:
        status = "watch"
    else:
        status = "stable"

    return DriftReport(
        status=status,
        n_requests=len(rows),
        n_features=reference.n_features,
        n_informative=reference.n_informative,
        n_uninformative=reference.n_uninformative,
        uninformative_features=reference.uninformative_indices.tolist(),
        bins=reference.bins,
        max_psi=float(psis.max()) if psis.size else float("nan"),
        mean_psi=float(psis.mean()) if psis.size else float("nan"),
        n_watched=n_watched,
        n_alerted=n_alerted,
        fraction_watched=fraction_watched,
        features=drifts,
        thresholds=thresholds,
        pruned_rows=pruned,
        reference_samples=reference.n_samples,
    )


def resolve_output_path(configured: str) -> Path:
    """Resolve a configured output path under :data:`mlserve.config.RESULTS`.

    The config reads better with a leading ``results/`` (it is a path a human
    looks at), but the directory must come from ``RESULTS`` so it honours
    ``MLSERVE_RESULTS``. Joining the configured path onto ``ROOT`` instead is what
    broke the container: inside the image ``ROOT`` is ``/app``, which WORKDIR creates
    as root, so the non-root service user could not create ``/app/results`` and the
    drift job died with ``PermissionError``.

    A prefix equal to the results directory name is stripped, so both
    ``drift.json`` and ``results/drift.json`` resolve to the same place.
    """
    relative = Path(configured)
    if relative.parts and relative.parts[0] == RESULTS.name:
        relative = Path(*relative.parts[1:])
    return RESULTS / relative


def write_outputs(report: DriftReport, cfg: dict) -> tuple[Path, Path, Path | None]:
    """Write the drift outputs.

    Two destinations for the Prometheus file when a collector directory is
    configured:

    * under ``RESULTS``, so it travels with the benchmark artefacts; and
    * into ``MLSERVE_TEXTFILE_DIR``, where node_exporter's textfile collector reads
      it, if that variable is set.

    The collector copy is written with ``os.replace`` on a temp file in the same
    directory. A non-atomic write would let the collector read a half-written file,
    which surfaces as a random parse error in Prometheus and a drift gauge that
    flaps. This also removes the need for a separate sidecar container copying the
    file across volumes.
    """
    ensure_dirs()
    json_path = resolve_output_path(cfg["outputs"]["json"])
    prom_path = resolve_output_path(cfg["outputs"]["prometheus_textfile"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    prom_path.parent.mkdir(parents=True, exist_ok=True)

    payload = report.to_json()
    payload["reference_meta"] = (
        read_json(ROOT / "artifacts" / "reference_meta.json")
        if (ROOT / "artifacts" / "reference_meta.json").exists()
        else {}
    )
    write_json(json_path, payload)
    prom_text = report.to_prometheus()
    prom_path.write_text(prom_text, encoding="utf-8")

    collector_path: Path | None = None
    collector_dir = os.environ.get("MLSERVE_TEXTFILE_DIR")
    if collector_dir:
        target_dir = Path(collector_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        collector_path = target_dir / prom_path.name
        # Write to a sibling temp file then rename, so the collector never observes
        # a partially written file. os.replace is atomic within a filesystem.
        staging = target_dir / f".{prom_path.name}.tmp"
        staging.write_text(prom_text, encoding="utf-8")
        os.replace(staging, collector_path)

    return json_path, prom_path, collector_path


def main() -> int:
    cfg = load_config("drift")
    set_seed(int(load_config("model")["seed"]))
    ensure_dirs()

    try:
        report = detect(cfg)
    except FileNotFoundError as exc:
        print(f"cannot run drift detection: {exc}")
        return 1

    json_path, prom_path, collector_path = write_outputs(report, cfg)

    print(f"drift status    {report.status.upper()}")
    print(f"requests        {report.n_requests} (reference n={report.reference_samples})")
    if report.status != "insufficient_data":
        print(f"max PSI         {report.max_psi:.4f}")
        print(f"mean PSI        {report.mean_psi:.4f}")
        print(
            f"features        {report.n_watched} watched, {report.n_alerted} alerted "
            f"of {report.n_informative} scored "
            f"({report.n_uninformative} constant features excluded)"
        )
        drifted = [f for f in report.features if f.band != "stable"]
        if drifted:
            for feature in drifted[:8]:
                print(
                    f"  feature[{feature.index:2d}]  psi={feature.psi:6.3f}  "
                    f"{feature.band:<6}  ks_p={feature.ks_pvalue:.2e}"
                )
        else:
            print("  no feature exceeded the watch threshold")
    else:
        print(
            f"  window has fewer than {cfg['window']['min_requests']} requests; no comparison made"
        )
    if report.pruned_rows:
        print(f"pruned          {report.pruned_rows} old rows")

    print(f"wrote           {_display(json_path)}")
    print(f"wrote           {_display(prom_path)}")
    if collector_path is not None:
        print(f"wrote           {collector_path} (node_exporter textfile collector)")

    # Exit non-zero on alert so this works as a cron/CI gate as well as a report.
    return 2 if report.status == "alert" else 0


def _display(path: Path) -> str:
    """Path relative to the repo when it is under it, absolute otherwise.

    ``relative_to(ROOT)`` raises when the results directory is redirected outside
    the repo by MLSERVE_RESULTS, which is exactly what the container does.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
