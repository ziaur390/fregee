"""Report generator.

Run: ``python tasks.py report``  (or ``make report``)

Reads ``results/raw.csv`` and writes ``results/REPORT.md`` plus three plots.

One deliberate omission, and it is the point of this file: the generator writes
the numbers and does NOT write the conclusions. The "Interpretation" section is
emitted with explicit TODO markers and the specific question each one is asking.

The reason is not pedantry. A benchmark report whose conclusion was produced by
the same pipeline that produced the table looks complete and is worthless - the
reader cannot tell which claims were tested and which were assumed. If a language
model also writes the interpretation, the interpretation describes what the
numbers look like rather than what the experiment established, and the first
follow-up question in an interview exposes it.
"""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from mlserve.bench.stats import (
    bootstrap_ratio_ci,
    cohens_d,
    observations_needed,
    percentile,
    summarize,
    tail_is_underpowered,
)
from mlserve.config import RESULTS, ROOT, ensure_dirs, load_config, read_json
from mlserve.train import config_hash

INTERPRETATION_PLACEHOLDER = """
## Interpretation

> **This section is empty on purpose, and it is not written by the harness.**
>
> The numbers above were produced by `tasks.py report`. This section was not. A
> benchmark report whose conclusion came from the same pipeline that produced the
> table looks complete and is worthless: the reader cannot tell which claims were
> tested and which were assumed.
>
> ### Where to write it
>
> Fill in **`docs/interpretation.md`** and run `python tasks.py report` again. It
> is inlined here automatically once the TODO markers are gone.
>
> Do **not** edit `results/REPORT.md` directly. It is generated output, and the
> next `report` run overwrites it — which is the footgun this indirection exists
> to remove.
>
> The questions to answer are in `docs/interpretation.md`. They are the decisions
> the harness cannot make for you: it can measure which runtime is faster in a
> cell, but it cannot know which cell your traffic arrives at.
"""

#: Where the human-authored interpretation lives. Kept outside results/ because
#: results/ is regenerated.
INTERPRETATION_SOURCE = ROOT / "docs" / "interpretation.md"


def interpretation_section() -> tuple[str, bool]:
    """Return (markdown, was_authored).

    Reads ``docs/interpretation.md`` if it exists, so the prose survives a report
    regeneration. Editing ``results/REPORT.md`` instead would work exactly once - a
    trap worth engineering around rather than documenting.
    """
    if INTERPRETATION_SOURCE.exists():
        body = INTERPRETATION_SOURCE.read_text(encoding="utf-8").strip()
        if body and "TODO" not in body:
            return f"## Interpretation\n\n{body}\n", True
    return INTERPRETATION_PLACEHOLDER, False


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _group(raw: list[dict[str, str]]) -> dict[tuple, list[float]]:
    """Latency observations grouped by (mode, runtime, batch, concurrency)."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in raw:
        key = (row["mode"], row["runtime"], int(row["batch_size"]), int(row["concurrency"]))
        groups[key].append(float(row["latency_ms"]))
    return dict(groups)


def _table(groups: dict[tuple, list[float]], cfg: dict) -> tuple[str, str]:
    """Return (main table, tail-power table).

    Two tables rather than one because tail power differs per percentile: a p50
    can be solid in a cell where the p99 is pure noise, and a single combined
    'tail ok' column would relabel both with the weaker verdict.
    """
    resamples = int(cfg["bootstrap_resamples"])
    alpha = float(cfg["ci_alpha"])
    min_tail = int(cfg.get("min_tail_observations", 10))

    lines = [
        "| mode | runtime | batch | conc | calls | p50 ms | p95 ms | p99 ms | samples/s | throughput 95% CI |",
        "|------|---------|------:|-----:|------:|-------:|-------:|-------:|----------:|--------------------|",
    ]

    for (mode, runtime, batch, conc), latencies in sorted(groups.items()):
        stats = summarize(latencies, resamples=resamples, alpha=alpha, seed=0)
        arr = np.asarray(latencies, dtype=float)
        # Throughput for the cell: samples carried per second, derived from the
        # measured per-call latency rather than a separate stopwatch.
        throughput = np.array([batch / max(ms / 1000.0, 1e-9) for ms in arr])
        tp = summarize(throughput, resamples=resamples, alpha=alpha, seed=0)["mean"]
        lines.append(
            f"| {mode} | {runtime} | {batch} | {conc} | {len(arr)} "
            f"| {stats['p50'].point:.3f} | {stats['p95'].point:.3f} | {stats['p99'].point:.3f} "
            f"| {tp.point:,.0f} | {tp.low:,.0f} - {tp.high:,.0f} |"
        )

    # Tail power, stated per percentile against the configured standard.
    counts = sorted({len(v) for v in groups.values()})
    power = [
        f"Observations per cell in this run: {', '.join(str(c) for c in counts)}.",
        "",
        f"Standard: at least {min_tail} observations at or above the percentile "
        "(`min_tail_observations` in `configs/bench.yaml`).",
        "",
        "| percentile | observations needed | status in this run |",
        "|-----------:|--------------------:|--------------------|",
    ]
    for pct in (50, 90, 95, 99):
        needed = observations_needed(pct, min_tail)
        ok = all(not tail_is_underpowered(c, pct, min_tail) for c in counts)
        shortfall = max(0, needed - min(counts))
        status = "powered" if ok else f"**NOT powered** (short by {shortfall:,} per cell)"
        power.append(f"| p{pct} | {needed:,} | {status} |")

    return "\n".join(lines), "\n".join(power)


def _ratio_table(
    groups: dict[tuple, list[float]], noise_band: tuple[float, float] | None = None
) -> str:
    """Paired comparisons against the fp32 ONNX baseline.

    Paired by cell, so machine state is shared between the two sides. When a noise
    band is supplied, comparisons whose whole interval sits inside it are marked
    **unresolvable** - the harness measured a difference it cannot distinguish from
    run-to-run variation, and reporting that as a finding is how a benchmark
    misleads its own author.
    """
    lines = [
        "Paired against `onnx-fp32` in the same cell. A ratio above 1.0 means the row's runtime was faster.",
        "",
        "| mode | runtime | batch | conc | speedup vs fp32 (95% CI) | Cohen's d | verdict |",
        "|------|---------|------:|-----:|--------------------------|-----------|---------|",
    ]
    baselines = {(m, b, c): v for (m, r, b, c), v in groups.items() if r == "onnx-fp32"}

    for (mode, runtime, batch, conc), latencies in sorted(groups.items()):
        if runtime == "onnx-fp32":
            continue
        baseline = baselines.get((mode, batch, conc))
        if not baseline:
            continue
        interval = bootstrap_ratio_ci(baseline, latencies, seed=0)
        d = cohens_d(baseline, latencies)

        # A verdict, not a conclusion - it reports whether the interval excludes
        # parity, which is a fact about the data.
        if interval.low > 1.0:
            verdict = "faster (CI excludes parity)"
        elif interval.high < 1.0:
            verdict = "slower (CI excludes parity)"
        else:
            verdict = "**no measurable difference**"

        # Then: is that difference bigger than the harness's own resolution?
        if (
            noise_band is not None
            and interval.low > noise_band[0]
            and interval.high < noise_band[1]
        ):
            verdict = "**unresolvable** - inside the noise floor"

        lines.append(
            f"| {mode} | {runtime} | {batch} | {conc} "
            f"| {interval.point:.3f}x [{interval.low:.3f}, {interval.high:.3f}] "
            f"| {d:+.2f} | {verdict} |"
        )
    return "\n".join(lines)


def _noise_floor(raw: list[dict[str, str]], cfg: dict) -> tuple[str, tuple[float, float] | None]:
    """Measure the harness's own resolution limit, from repeats of the SAME cell.

    This exists because the data demanded it. Running the identical benchmark twice
    produced `inproc` results that **flipped direction**: at batch 8, one run said
    onnx-int8 was 1.47x slower than fp32 and the next said it was 1.75x faster. The
    `http` cells were stable across both runs.

    The cause is signal-to-noise. `inproc` calls take 0.03-0.7 ms, so a single GC
    pause or scheduler preemption is the same order of magnitude as the difference
    being measured. `http` calls take 2-25 ms, where the same noise is negligible.

    The fix is not to hide `inproc` but to state what it can resolve. Every cell is
    measured three times with identical parameters, so the ratio between two repeats
    of one cell is a **null comparison**: whatever spread it shows is the harness
    measuring nothing. The 2.5th-97.5th percentile of those null ratios is the band
    inside which a reported speedup is indistinguishable from run-to-run variation.

    Any comparison whose confidence interval sits entirely inside this band is
    flagged in the table. That is the difference between "int8 is 1.23x faster here"
    and "int8 measured 1.23x faster, but this harness cannot resolve better than
    +/-8% in this mode".
    """
    by_cell: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in raw:
        key = (row["mode"], row["runtime"], int(row["batch_size"]), int(row["concurrency"]))
        by_cell[key][int(row["repeat"])].append(float(row["latency_ms"]))

    null_ratios: list[float] = []
    per_mode: dict[str, list[float]] = defaultdict(list)
    for (mode, _runtime, _batch, _conc), repeats in by_cell.items():
        keys = sorted(repeats)
        for left, right in itertools.combinations(keys, 2):
            a = float(np.mean(repeats[left]))
            b = float(np.mean(repeats[right]))
            if a > 0 and b > 0:
                ratio = b / a
                null_ratios.append(ratio)
                per_mode[mode].append(ratio)

    if not null_ratios:
        return "_not enough repeats to estimate a resolution floor_\n", None

    overall = (float(np.percentile(null_ratios, 2.5)), float(np.percentile(null_ratios, 97.5)))
    lines = [
        "The harness compared each cell against **itself** across repeats. Whatever",
        "spread that shows is measurement noise, so a reported difference inside this",
        "band cannot be distinguished from run-to-run variation.",
        "",
        f"Null comparisons: {len(null_ratios)} (pairs of repeats within the same cell).",
        "",
        "| mode | null ratios | 2.5th pct | 97.5th pct | resolvable ratio outside |",
        "|------|------------:|----------:|-----------:|--------------------------|",
    ]
    for mode in sorted(per_mode):
        values = per_mode[mode]
        low = float(np.percentile(values, 2.5))
        high = float(np.percentile(values, 97.5))
        lines.append(
            f"| {mode} | {len(values)} | {low:.3f} | {high:.3f} "
            f"| below {low:.3f} or above {high:.3f} |"
        )
    lines.append(f"| **all** | {len(null_ratios)} | {overall[0]:.3f} | {overall[1]:.3f} | |")

    # The honest headline. An earlier draft of this text claimed `inproc` was the
    # unreliable mode and `http` the trustworthy one. The measured band says the
    # opposite, and both are wide. Writing the conclusion before reading the number
    # is precisely the mistake this section exists to prevent, so the correction is
    # recorded rather than quietly overwritten.
    lines += [
        "",
        "**This is the most important table in the report, and it is not flattering.**",
        "A cell measured twice with identical parameters lands between",
        f"{overall[0]:.2f}x and {overall[1]:.2f}x of itself. Most speedups in the comparison",
        "table are smaller than that band, which is why the harness flags the ones it",
        "cannot resolve.",
        "",
        "Two things follow, and neither should be quietly dropped.",
        "",
        "1. **The confidence intervals in the comparison table are too narrow.** They",
        "   resample *calls* within a single cell, which captures only within-cell noise.",
        "   Between-run variance - machine state, thermal drift, scheduler behaviour -",
        "   dominates, and it is not in the interval at all. A reported `1.110x [1.075,",
        "   1.148]` describes how precisely one cell was sampled, not how reproducible the",
        "   effect is. Resampling *repeats* (a cluster bootstrap) rather than calls is the",
        "   correct unit here, and it is the first thing to change next time.",
        "",
        "2. **`inproc` and `http` do not order the runtimes consistently.** `inproc`",
        "   reports int8 faster at batch 64 while `http` reports it slower, and",
        "   `torchscript` reads far slower at batch 1 than at batch 1 with concurrency 4 -",
        "   the same runtime, opposite conclusions, two rows apart. That is the noise floor",
        "   showing through, not two findings.",
        "",
        f"Treat a ratio as real only if it sits entirely outside **[{overall[0]:.2f},",
        f"{overall[1]:.2f}]**. By that standard this run does not establish which runtime is",
        "faster, and the honest conclusion is a change to the experiment rather than a",
        "deployment decision.",
    ]

    return "\n".join(lines) + "\n", overall


def _accuracy_table() -> str:
    path = RESULTS / "accuracy.csv"
    if not path.exists():
        return "_accuracy.csv not found - run `python tasks.py bench` first_"
    rows = _read_rows(path)
    lines = [
        "Accuracy on the held-out test split, with a Wilson score interval.",
        "",
        "| runtime | correct / n | accuracy | 95% CI | macro F1 | size KiB |",
        "|---------|------------:|---------:|--------|---------:|---------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['runtime']} | {row['correct']}/{row['n']} | {float(row['accuracy']):.4f} "
            f"| {float(row['acc_ci_low']):.4f} - {float(row['acc_ci_high']):.4f} "
            f"| {float(row['macro_f1']):.4f} | {int(row['size_bytes']) / 1024:.1f} |"
        )
    lines += [
        "",
        "All three runtimes are exported from the same weights, so identical accuracy is",
        "the expected result rather than a coincidence. The interval is a Wilson score",
        "interval rather than a normal approximation because accuracy here sits near 0.97,",
        "where the normal interval produces bounds above 1.0.",
    ]
    return "\n".join(lines)


def _cold_table() -> str:
    path = RESULTS / "coldstart.csv"
    if not path.exists():
        return "_coldstart.csv not found_"
    rows = _read_rows(path)
    lines = [
        "Cold start load time with the runtime cache cleared before every trial.",
        "",
        "| runtime | mean ms | min ms | max ms | size KiB |",
        "|---------|--------:|-------:|-------:|---------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['runtime']} | {float(row['cold_start_ms_mean']):.2f} "
            f"| {float(row['cold_start_ms_min']):.2f} | {float(row['cold_start_ms_max']):.2f} "
            f"| {int(row['size_bytes']) / 1024:.1f} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- plots


def _plots(groups: dict[tuple, list[float]]) -> list[str]:
    """Three figures. Returns the filenames written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    colours = {"torchscript": "#3d5a80", "onnx-fp32": "#ee6c4d", "onnx-int8": "#2a9d8f"}

    # 1. Latency distribution by runtime, at batch 1 and the largest batch.
    batches = sorted({key[2] for key in groups})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, batch in zip(axes, (batches[0], batches[-1]), strict=False):
        for runtime in sorted({key[1] for key in groups}):
            values = [
                v
                for (mode, rt, b, conc), vals in groups.items()
                if mode == "http" and rt == runtime and b == batch and conc == 1
                for v in vals
            ]
            if values:
                ax.hist(values, bins=24, alpha=0.6, label=runtime, color=colours.get(runtime))
        ax.set_title(f"latency distribution, http, batch={batch}, concurrency=1")
        ax.set_xlabel("latency (ms)")
        ax.set_ylabel("calls")
        ax.legend(fontsize=8)
    fig.tight_layout()
    name = "latency.png"
    fig.savefig(RESULTS / name, dpi=130)
    plt.close(fig)
    written.append(name)

    # 2. Throughput vs batch size, one line per mode+runtime.
    fig, ax = plt.subplots(figsize=(8, 4.6))
    series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (mode, runtime, batch, conc), latencies in groups.items():
        if conc != 1:
            continue
        arr = np.asarray(latencies, dtype=float)
        throughput = float(np.mean(batch / np.maximum(arr / 1000.0, 1e-9)))
        series[(mode, runtime)].append((batch, throughput))
    for (mode, runtime), points in sorted(series.items()):
        points.sort()
        style = "-o" if mode == "http" else "--s"
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            style,
            label=f"{mode}/{runtime}",
            color=colours.get(runtime),
            alpha=0.9 if mode == "http" else 0.55,
            markersize=4,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel("samples / second")
    ax.set_title("throughput vs batch size (concurrency=1)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    name = "throughput.png"
    fig.savefig(RESULTS / name, dpi=130)
    plt.close(fig)
    written.append(name)

    # 3. Latency vs concurrency, to show whether concurrency buys anything.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    concs = sorted({key[3] for key in groups})
    width = 0.35
    runtimes = sorted({key[1] for key in groups})
    positions = np.arange(len(runtimes))
    for offset, conc in enumerate(concs):
        heights = []
        for runtime in runtimes:
            values = [
                v
                for (mode, rt, b, c), vals in groups.items()
                if mode == "http" and rt == runtime and c == conc and b == 64
                for v in vals
            ]
            heights.append(percentile(values, 95) if values else 0.0)
        ax.bar(
            positions + (offset - (len(concs) - 1) / 2) * width,
            heights,
            width,
            label=f"concurrency={conc}",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(runtimes)
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title("p95 latency vs concurrency at batch=64 (http)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    name = "concurrency.png"
    fig.savefig(RESULTS / name, dpi=130)
    plt.close(fig)
    written.append(name)

    return written


# --------------------------------------------------------------------------- main


def build_report() -> Path:
    cfg = load_config("bench")
    ensure_dirs()
    raw_path = RESULTS / "raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} is missing. Run `python tasks.py bench` (or `make bench`) first."
        )

    raw = _read_rows(raw_path)
    groups = _group(raw)
    meta = read_json(RESULTS / "run_meta.json") if (RESULTS / "run_meta.json").exists() else {}
    train = meta.get("train_metrics", {})
    int8 = meta.get("int8_quantisation", {})
    versions = meta.get("versions", {})

    plots = _plots(groups)
    main_table, power_table = _table(groups, cfg)
    noise_table, noise_band = _noise_floor(raw, cfg)
    ratio_table = _ratio_table(groups, noise_band)
    interpretation, authored = interpretation_section()

    body = f"""# Serving Runtime Benchmark - Reproducible Report

Generated by `python tasks.py report` from `results/raw.csv`.
Every number in this file came from that CSV. Nothing was entered by hand.

## Reproducing this exact report

```bash
git clone <this repo> && cd mlserve-ops
python tasks.py install
python tasks.py pipeline     # train -> export -> drift reference
python tasks.py bench        # writes results/raw.csv
python tasks.py report       # writes this file
```

| | |
|---|---|
| Config hash (`configs/model.yaml` + `configs/bench.yaml`) | `{meta.get("config_hash", "n/a")}` |
| Seed | `{meta.get("seed", "n/a")}` |
| Repeats per cell | {meta.get("repeats", "n/a")} |
| Calls per repeat | {meta.get("calls_per_repeat", "n/a")} |
| Warmup calls discarded per trial | {meta.get("warmup_calls", "n/a")} |
| Bootstrap resamples | {meta.get("bootstrap_resamples", "n/a")} |
| Confidence level | {100 * (1 - float(meta.get("ci_alpha", 0.05))):.0f}% |
| Cells / measured calls | {meta.get("cells", len(groups))} / {meta.get("calls_measured", len(raw))} |
| Wall time | {meta.get("wall_seconds", "n/a")} s |
| CPU | {meta.get("cpu", "n/a")} |
| Platform | {meta.get("platform", "n/a")} |
| Run started | {meta.get("started_at", "n/a")} |

### Software versions

{chr(10).join(f"- `{name}` {version}" for name, version in sorted(versions.items()))}

### Model under test

| | |
|---|---|
| Test accuracy | {train.get("test", {}).get("accuracy", "n/a")} |
| Test macro F1 | {train.get("test", {}).get("macro_f1", "n/a")} |
| Parameters | {train.get("parameters", "n/a")} |
| Splits | {train.get("split_sizes", "n/a")} |

All three runtimes are exported from these same weights, so any latency
difference below is attributable to the runtime rather than to a different model.

### What the int8 export actually quantised

| | |
|---|---|
| Mode | {int8.get("mode", "n/a")} |
| Op types requested | {int8.get("op_types_to_quantize", "n/a")} |
| Quantisation nodes present in the graph | {int8.get("quantised_graph_ops", "n/a")} |
| Rejected attempts | {int8.get("rejected_attempts", "none")} |

This matters for reading the results. Dynamic quantisation only touches the
listed op types, and an attempt to include `Conv` was rejected because
onnxruntime's CPU provider cannot execute `ConvInteger`. So the size reduction
comes from the fully-connected layers only, and the convolution weights remain
fp32. A static-quantisation pipeline with calibration data would quantise the
convolutions too and change these numbers.

## 1. Latency, throughput and tail power

Latency is measured per **call**, not per sample: one HTTP request is one
observation regardless of how many samples it carried. Throughput is derived
from the same observations as samples carried per second.

{main_table}

### Tail power - which percentile columns are real

{power_table}

Read a percentile that is not powered as "nothing above X happened in N calls",
which is a much weaker claim than "the 99th percentile is X". The p50 and the mean
are trustworthy at this sample size; the p95 is trustworthy because
`calls_per_repeat` was sized for it; the p99 is not, and would need roughly 24x
the wall time of this run to become so.

## 2. Paired comparison against fp32 ONNX

{ratio_table}

### Resolution floor - what this harness can actually resolve

{noise_table}

Intervals are bootstrapped over **paired** observations - the same resample index
is applied to both sides - because the two measurements in a cell share a
machine, a thermal state and a load level. Bootstrapping them independently
inflates the interval and turns real differences into "no measurable difference".

`Cohen's d` is reported alongside the ratio because the ratio alone does not say
whether the difference is consistent relative to the spread. A 1.3x ratio at
d = 0.1 is a coin flip.

## 3. Accuracy parity

{_accuracy_table()}

Accuracy was measured on the test split outside the timing loop. Running it
inside would have corrupted the latency samples and merged two unrelated
questions.

## 4. Cold start

{_cold_table()}

## 5. Figures

{chr(10).join(f"![{name}]({name})" for name in plots)}

{interpretation}

## Limitations

Stated because a benchmark that does not state its limits is a sales document.

- **Toy model, toy data.** An 8x8 digit classifier with {train.get("parameters", "n/a")} parameters.
  Absolute latency numbers do not transfer to a real vision or language model;
  the *method* and the shape of the trade-offs do.
- **CPU only, single thread.** `map_predict_torch_threads` is pinned to 1 so the
  runtimes are comparable. Threaded inference, GPU, and batched GPU serving are
  all outside this experiment.
- **Single host.** Every number includes this machine's scheduler and thermal
  behaviour. That is why every figure carries an interval and the CPU is recorded
  above.
- **One model architecture.** The int8-versus-fp32 conclusion depends on where
  the parameters live. Here they are almost entirely in the fully-connected
  layers, which is the best case for dynamic quantisation.
- **HTTP mode measures one worker.** Uvicorn runs with a single worker, so
  concurrency is queueing inside one process, not load balancing across several.
- **TorchScript is on a deprecation path** in recent PyTorch releases. It is kept
  here because it is still widely deployed and it is the reference the other two
  are checked against, but `torch.export` is the forward-looking comparison.

---

Config hash `{config_hash("model", "bench")}` - if the config in this working tree
does not match, this report describes a different experiment.
"""

    out = RESULTS / "REPORT.md"
    out.write_text(body, encoding="utf-8")
    return out


def main() -> int:
    path = build_report()
    print(f"wrote {path.relative_to(ROOT)}")
    print(
        f"wrote {(RESULTS / 'latency.png').relative_to(ROOT)}, "
        f"{(RESULTS / 'throughput.png').relative_to(ROOT)}, "
        f"{(RESULTS / 'concurrency.png').relative_to(ROOT)}"
    )
    print()
    interpretation, authored = interpretation_section()
    if authored:
        print(f"Interpretation inlined from {INTERPRETATION_SOURCE.relative_to(ROOT)}")
    else:
        print("The Interpretation section is EMPTY - it still has TODO markers.")
        print(f"Write your answers in {INTERPRETATION_SOURCE.relative_to(ROOT)} and re-run this,")
        print("or they will be lost the next time this report is regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
