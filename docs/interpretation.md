# Interpretation

> **DRAFT — this is not finished.**
>
> I wrote this from the measured numbers so you would have something to react to
> rather than a blank page, but **you need to rewrite it in your own words before
> showing it to anyone.** Every claim below is traceable to `results/REPORT.md`, so
> the work is checking each one and saying it your way.
>
> If an interviewer asks "what did *you* conclude?", a section you did not write is
> worse than no section at all. This box is inlined into `results/REPORT.md` along
> with the text, so it is visible to anyone reading the report until you delete it —
> which is the point. **Delete this box when you have rewritten the section.**

---

## Headline: this run does not establish which runtime is faster

I ran the full 48-cell matrix and the honest result is that **the harness cannot
resolve the differences it measures.** The report contains a table that says so,
and it is the most important table in the report:

| mode | null ratios | 2.5th pct | 97.5th pct |
|------|------------:|----------:|-----------:|
| http | 72 | 0.511 | 1.653 |
| inproc | 72 | 0.718 | 1.576 |
| **all** | 144 | **0.580** | **1.708** |

That band comes from comparing **each cell against itself**: the same
configuration, measured three times, and the ratio between two repetitions of it.
Whatever spread that shows is the measurement device, not the phenomenon. A cell
measured twice with identical parameters lands between 0.58x and 1.71x of itself.

Every speedup I set out to measure is smaller than that. Nineteen of the 48 paired
comparisons are flagged **unresolvable** for exactly this reason.

**So the honest answer to "which runtime would you deploy?" is: this experiment
cannot tell me, and reporting a winner from it would be reporting noise.**

---

## What the data does say, and how much to trust each part

### int8 accuracy is unaffected

| runtime | accuracy | 95% CI | size |
|---|---:|---|---:|
| torchscript | 0.9741 | 0.9475 – 0.9874 | 550.4 KiB |
| onnx-fp32 | 0.9741 | 0.9475 – 0.9874 | 535.0 KiB |
| onnx-int8 | 0.9741 | 0.9475 – 0.9874 | **151.5 KiB** |

All three are exported from the same weights, so identical accuracy is the
expected result rather than a lucky one. The interval is wide (270 test samples)
but it is the same interval for all three, so the *comparison* is solid even
though the absolute value is not precise.

This is the one claim I would defend without reservation: **int8 costs nothing
measurable in accuracy and is 3.5x smaller.**

What the size buys is cold start and memory, and there the picture is mixed:

| runtime | cold start mean | min | max |
|---|---:|---:|---:|
| torchscript | 13.26 ms | 11.37 | 16.80 |
| onnx-fp32 | 3.36 ms | 3.18 | 3.49 |
| onnx-int8 | 3.69 ms | 2.89 | 4.65 |

int8 is 3.5x smaller than fp32 but cold-starts in the *same* time. The smaller
file does not translate into faster loading here, which is worth knowing before
arguing that quantisation speeds up autoscaling.

### `inproc` effect sizes are not believable

The `inproc` ratios include `0.201x [0.194, 0.208]` with Cohen's d of **-4.65**.
That says one runtime is five times slower than another while executing the same
arithmetic on the same weights with the same thread count. An effect that large in
a comparison that should be near parity is not a finding — it is a sign the
measurement mode is broken.

`inproc` calls take 30–700 microseconds. At that scale, one garbage-collection
pause, one scheduler preemption, or one Python bytecode loop iteration is the same
order of magnitude as the thing being measured. The mode is measuring the
interpreter's mood as much as the runtime.

**`http` is the mode I would trust, because it was the only one that agreed with
itself across two independent runs.** I verified that by running the identical
benchmark twice and comparing directions: every `http` cell kept its sign, while
`inproc` flipped at batch 8 and batch 32.

### The confidence intervals are wrong, and that is my main correction

The comparison table reports things like `1.110x [1.075, 1.148]`. That interval is
far narrower than the 0.58–1.71 band, and the discrepancy is a methodology error:

> the interval resamples **calls within a single cell**, so it captures only
> within-cell noise. The dominant source of variance is **between runs** — machine
> state, thermal drift, whatever else the machine is doing — and it is not in the
> interval at all.

A ±4% interval on a measurement whose run-to-run spread is ±70% is not a precise
result. It is a precise statement about one biased sample.

**The fix is a cluster bootstrap**: resample *repeats*, not calls. With three
repeats per cell that gives a coarse but honest interval. That is the first change
I would make.

---

## Which rows I would not trust

1. **Every `inproc` row.** Not because the mode is useless, but because at
   microsecond latencies this harness has no resolution. Its null band is 86% wide.
2. **The whole p99 column.** At 210 observations per cell, a p99 needs 1,000 to
   have ten samples in its tail. The report flags this automatically.
3. **Any ratio inside 0.58–1.71.** Flagged in the table.
4. **Both concurrency-4 columns.** They are the widest in the null comparison,
   which makes sense — with more requests in flight, scheduling is more of the
   measurement.

---

## What I would change next time, in priority order

1. **Cluster bootstrap over repeats.** Fixes the interval, which is the actual
   bug. Everything else is secondary to this.
2. **Randomise cell execution order.** Cells currently run in a fixed sequence
   (mode → runtime → batch → concurrency), so runtime N is always measured in a
   slightly different machine state than runtime N-1. That is a confound I did not
   control for.
3. **More repeats, fewer calls per repeat.** Three repeats of 210 calls is the
   wrong shape: it buys a precise within-cell estimate and a poor between-run one.
   Ten repeats of 60 calls would be a better trade.
4. **Report the null band first.** A benchmark should state its own resolution
   before showing any result.
5. **Run on an idle machine.** Two runs of the same configuration gave different
   answers while Terraform and WSL were running in the background. That is not the
   harness's fault, but it is the harness's problem to detect and report.

---

## The one thing I would say in an interview

The experiment failed to produce a deployment recommendation, and I know why:
the harness's own run-to-run variation is larger than the effect it was built to
measure. I found that by comparing each configuration against itself, which is a
control I added *after* seeing results that disagreed with each other.

The skill on display is not "I benchmarked three runtimes". It is "I caught my own
benchmark lying, worked out the mechanism, and said so in the report instead of
picking the number I liked."
