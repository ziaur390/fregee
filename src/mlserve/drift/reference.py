"""Build and load the reference distribution the drift job compares against.

Run: ``python tasks.py reference``  (or ``make reference``)

Writes ``artifacts/reference.npz``: per feature, the quantile bin edges and the
training proportions that fall in each bin.

Two choices here are load-bearing.

**Quantile bins, not equal-width.** The features are normalised pixel intensities
of 8x8 digit images, so the overwhelming majority of values sit near zero
(background). Equal-width bins would place roughly 90% of the training data in
the first bin of every feature, which makes PSI nearly blind: a shift moving mass
from bin 1 to bin 2 would barely register, and a shift moving mass *within* bin 1
would not register at all. Quantile edges guarantee each reference bin holds a
comparable share of training data, so every bin contributes comparably to the
score.

**Reference from the training split only.** Using the full dataset would fold the
test distribution into the baseline. The detector exists to notice when
production diverges from what the model was fitted on, and that is the training
split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlserve.config import ARTIFACTS, load_config, set_seed, write_json
from mlserve.model import load_digits_split
from mlserve.schema import FEATURE_COUNT

REFERENCE_PATH = ARTIFACTS / "reference.npz"

#: Number of training rows kept verbatim for the KS comparison.
#: PSI works from bins, but KS needs an actual sample, and reconstructing one
#: from bin proportions would compare a synthetic approximation against the real
#: window. 500 real rows is exact, costs ~250 KB compressed, and is a large
#: enough reference that the KS statistic is stable.
KS_SAMPLE_ROWS = 500


@dataclass(frozen=True)
class Reference:
    """Per-feature bin edges, training proportions per bin, and a raw sample.

    ``informative`` marks features that actually vary in the training split. Three
    of the 64 digit pixels are constant background in every training image, and a
    feature with zero variance has no distribution for production to drift away
    from - its PSI is 0 by construction and it would drag every average down while
    contributing nothing. Those features are recorded and reported, then excluded
    from the score rather than silently inflating the denominator.
    """

    bin_edges: np.ndarray  # (FEATURE_COUNT, bins + 1)
    proportions: np.ndarray  # (FEATURE_COUNT, bins)
    sample: np.ndarray  # (KS_SAMPLE_ROWS, FEATURE_COUNT) real training rows
    n_samples: int
    bins: int
    informative: np.ndarray  # (FEATURE_COUNT,) bool

    @property
    def n_features(self) -> int:
        return int(self.bin_edges.shape[0])

    @property
    def informative_indices(self) -> np.ndarray:
        return np.flatnonzero(self.informative)

    @property
    def uninformative_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.informative)

    @property
    def n_informative(self) -> int:
        return int(self.informative.sum())

    @property
    def n_uninformative(self) -> int:
        return self.n_features - self.n_informative


def flatten(rows: np.ndarray) -> np.ndarray:
    """(N, 1, 8, 8) -> (N, 64)."""
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), -1)


def _quantile_edges(column: np.ndarray, bins: int) -> np.ndarray:
    """Bin edges from quantiles, with the ends pushed out to +/-inf.

    The infinite outer edges matter: a production value outside the training
    range must land in the first or last bin rather than being dropped or raising.
    A value the model never saw during training is exactly the case this detector
    exists to catch, so silently discarding it would defeat the purpose.

    Note there is deliberately no ``np.unique`` on the interior edges. It looks
    like it would help, but most of these features are 8x8 digit pixels where the
    overwhelming majority of values are background, so many quantiles return the
    same number and deduplicating them yields a *different number of edges per
    feature*. That breaks the stacked array and, worse, makes the reference shape
    depend on the data. Repeated adjacent edges produce zero-width bins, which
    numpy accepts, and which are handled below by the epsilon in the PSI.
    """
    quantiles = np.linspace(0, 100, bins + 1)[1:-1]
    inner = np.percentile(column, quantiles)
    return np.concatenate(([-np.inf], inner, [np.inf]))


def build_reference(cfg: dict) -> Reference:
    """Build the reference from the training split and persist it."""
    seed = int(cfg["reference"].get("seed", load_config("model")["seed"]))
    set_seed(seed)

    model_cfg = load_config("model")
    split = load_digits_split(
        seed=int(model_cfg["seed"]),
        val_size=float(model_cfg["data"]["val_size"]),
        test_size=float(model_cfg["data"]["test_size"]),
    )
    train = flatten(split.x_train)

    bins = int(cfg["reference"]["bins"])
    edges = np.stack([_quantile_edges(train[:, i], bins) for i in range(FEATURE_COUNT)])

    # Proportions computed with the same edge set that will be used at detection
    # time, so a no-drift window scores exactly 0 rather than a small non-zero
    # value caused by re-binning differences.
    proportions = np.stack(
        [_bin_proportions(edges[i], train[:, i], bins) for i in range(FEATURE_COUNT)]
    )

    rng = np.random.default_rng(seed)
    sample_rows = train[rng.choice(len(train), size=min(KS_SAMPLE_ROWS, len(train)), replace=False)]

    # A feature with one unique training value has zero variance: no distribution,
    # nothing to drift from, and PSI = 0 no matter what production sends.
    informative = np.array([len(np.unique(train[:, i])) > 1 for i in range(FEATURE_COUNT)])

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        REFERENCE_PATH,
        bin_edges=edges,
        proportions=proportions,
        sample=sample_rows,
        informative=informative,
        n_samples=np.array([len(train)]),
        bins=np.array([bins]),
    )
    return Reference(edges, proportions, sample_rows, len(train), bins, informative)


def _bin_proportions(edges: np.ndarray, column: np.ndarray, bins: int) -> np.ndarray:
    counts, _ = np.histogram(column, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.full(bins, 1.0 / bins)
    return counts / total


def load_reference() -> Reference:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"{REFERENCE_PATH} is missing. Run `python tasks.py reference` first."
        )
    with np.load(REFERENCE_PATH) as data:
        return Reference(
            bin_edges=data["bin_edges"],
            proportions=data["proportions"],
            sample=data["sample"],
            n_samples=int(data["n_samples"][0]),
            bins=int(data["bins"][0]),
            informative=data["informative"],
        )


def main() -> int:
    cfg = load_config("drift")
    reference = build_reference(cfg)
    write_json(
        ARTIFACTS / "reference_meta.json",
        {
            "n_samples": reference.n_samples,
            "ks_sample_rows": len(reference.sample),
            "bins": reference.bins,
            "n_features": reference.n_features,
            "n_informative": reference.n_informative,
            "uninformative_features": np.flatnonzero(~reference.informative).tolist(),
            "binning": cfg["reference"]["binning"],
            "source": cfg["reference"]["source"],
        },
    )
    print(f"reference built from {reference.n_samples} training samples")
    print(
        f"  features    {reference.n_features} ({reference.n_informative} informative, "
        f"{reference.n_uninformative} constant and excluded)"
    )
    print(f"  bins        {reference.bins} (quantile)")
    print(f"  wrote       {REFERENCE_PATH.relative_to(ARTIFACTS.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
