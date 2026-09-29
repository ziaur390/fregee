# Interpretation

This file is inlined into `results/REPORT.md` by `python tasks.py report`.

**Edit this file, not `results/REPORT.md`.** The report is generated output; the
next `tasks.py report` overwrites it, and anything typed there is lost.

Delete the placeholder text below and answer the questions. Each one is a decision
the harness cannot make for you — it can measure, but it cannot tell you which
operating point your traffic actually arrives at.

---

<!-- TODO: delete from here and write your answers -->

## Which runtime, and at what batch size?

Name the cell you would deploy. A 1.35x throughput win at batch 64 is irrelevant
if real traffic arrives at batch 1.

## What does the int8 result actually show?

Compare its latency interval *and* its accuracy interval against fp32. If the
interval straddles parity, say so and do not claim a speedup. If it is faster with
no measurable accuracy loss, say what the 3.5x size reduction buys you.

## Where is the bottleneck — the runtime or the API?

Compare each `inproc` row against its `http` counterpart. A gap that appears only
in `http` is API overhead, not inference cost.

## Did concurrency help, and did you expect it to?

Every runtime here is single-threaded, so the prediction is "no, it queues".
Say whether the data agreed, and if not, what that implies.

## Which rows should not be trusted?

Check the tail-power table. A p99 from fewer than 1,000 observations is not a tail
measurement.

## What would you change next time?

One concrete change, justified by something the current data could not answer.
