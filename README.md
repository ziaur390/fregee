# mlserve-ops

A model service that provisions its own host, compares three serving runtimes
under a seed-controlled experiment, and reports every number with a confidence
interval.

Built to answer four questions with evidence rather than assertion:

| Question | Where the answer lives | What it proves |
|---|---|---|
| Can you run a Linux host? | `ansible/` | Users, SSH policy, firewall, systemd hardening, log rotation, timers |
| Can you operate a service? | `Dockerfile`, `docker-compose.yml`, `monitoring/` | Health-gated startup, alert rules that fire, backup with a **verified** restore |
| Can you serve a model? | `src/mlserve/app.py`, `registry.py` | Batched inference, three runtimes, SLO-grade metrics, drift detection |
| Can you design an experiment? | `configs/bench.yaml`, `src/mlserve/bench/` | A matrix with a stated rationale, bootstrap intervals, paired comparisons, tail-power reporting |

---

## The one-command demo

```bash
git clone https://github.com/ziaur390/fregee.git && cd fregee
python tasks.py up        # train -> export -> bring up api, prometheus, alertmanager, grafana
python tasks.py smoke     # assert every endpoint and the prometheus target actually answer
python tasks.py bench     # run the experiment matrix
python tasks.py report    # generate results/REPORT.md and the plots
```

On Linux or in CI, `make up` / `make test` / `make bench` are thin shims over the
same commands. Everything works on Windows too, which is why the runner is
`tasks.py` and not a Makefile.

> **`results/REPORT.md` ends with an empty "Interpretation" section.**
> That is intentional. The harness writes the numbers; the conclusions are yours
> to draw. See [Design decisions](#design-decisions) below for why, and for the
> specific questions that section asks.

---

## Architecture

```
                          ┌──────────────────────────────────────────┐
   terraform/envs/*  ───► │  host provisioning (terraform)           │
                          │  VM + firewall + cloud-init              │
                          └───────────────────┬──────────────────────┘
                                              │ prints the ansible command
                                              ▼
   ansible/playbook.yml ─► ┌──────────────────────────────────────────┐
     base    users         │  host configuration (ansible, idempotent)│
     firewall docker       │  ssh keys-only · ufw deny-by-default ·   │
     logrotate deploy      │  docker w/ log caps · systemd hardening  │
                          └───────────────────┬──────────────────────┘
                                              │ systemd units + timers
                                              ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  mlserve-api.service  ──  FastAPI  ──┬─ /predict        (single)          │
   │  User=mlserve, ProtectSystem=strict  ├─ /predict/batch  (capped at 64)    │
   │                                      ├─ /healthz        (liveness)        │
   │                                      ├─ /readyz         (model loaded?)   │
   │                                      ├─ /metrics        (prometheus)      │
   │                                      └─ /models /stats                    │
   │                                                                          │
   │  registry.py ──► torchscript ─┐                                          │
   │                  onnx-fp32    ├── one set of trained weights              │
   │                  onnx-int8    ┘   (that is what makes the bench fair)     │
   │                                                                          │
   │  request log ──► SQLite (request_id PK, idempotent restores)             │
   └───────────────┬──────────────────────────────┬───────────────────────────┘
                   │                              │
                   ▼                              ▼
   ┌───────────────────────────┐   ┌──────────────────────────────────────────┐
   │ mlserve-drift.timer       │   │ mlserve-backup.timer                     │
   │ PSI per feature (KS corrob)│  │ create archive -> restore to scratch DB  │
   │ writes results/drift.json  │  │ -> assert row-count + checksum parity     │
   │ + textfile for prometheus  │  │ FAILS LOUDLY if the restore disagrees     │
   └───────────┬───────────────┘   └──────────────────────────────────────────┘
               │ textfile
               ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │ prometheus ──► alertmanager ──► alert-sink (webhook logger, zero cost)    │
   │ grafana   ──► provisioned dashboard, no manual setup                     │
   └──────────────────────────────────────────────────────────────────────────┘
```

---

## Results

Reproduce with `python tasks.py pipeline && python tasks.py bench && python tasks.py report`.
Full tables, intervals and plots: **`results/REPORT.md`**.

### Model under test

| | |
|---|---|
| Task | 8x8 digit classification (`sklearn.datasets.load_digits`) |
| Parameters | 136,586 |
| Splits | 1257 train / 270 val / 270 test |
| Test accuracy | 0.9741 |
| Test macro F1 | 0.9740 |

Chosen for one reason: it trains on CPU in seconds, so the whole benchmark runs
free in CI and reproduces on any laptop. The absolute latency numbers below do not
transfer to a real vision model. The *method* does.

### Runtime export — same weights, three runtimes

| Runtime | Test accuracy | Disagreement vs TorchScript | Size | Cold load |
|---|---:|---:|---:|---:|
| torchscript | 0.9741 | 0.00% | 550.5 KiB | 8.67 ms |
| onnx-fp32 | 0.9741 | 0.00% | 535.0 KiB | 2.41 ms |
| **onnx-int8** | 0.9741 | 0.00% | **151.5 KiB** | 2.51 ms |

The parity check is an assertion, not a warning: a silently broken export would
make every latency comparison meaningless, so `tasks.py export` fails the build if
a runtime disagrees beyond tolerance.

**What the int8 export actually did.** `quantize_dynamic` was asked to quantise
`MatMul`, `Gemm` **and** `Conv`. That produced `ConvInteger` nodes, which
onnxruntime's CPU provider cannot execute — the file quantises fine and then
refuses to load. The export now *validates each candidate by running it* and falls
back to `MatMul`+`Gemm`. So the 3.5x size reduction comes entirely from the
fully-connected layers; convolution weights are still fp32. A static-quantisation
pipeline with calibration data would quantise those too and change these numbers.

### Served latency (HTTP, concurrency 1)

| Batch | Runtime | p50 ms | p95 ms | samples/s |
|---:|---|---:|---:|---:|
| 1 | torchscript | 2.660 | 3.625 | 365 |
| 1 | onnx-fp32 | 2.084 | 2.801 | 468 |
| 1 | **onnx-int8** | **1.971** | **2.559** | **497** |
| 64 | torchscript | 8.957 | 10.6 | 7,050 |
| 64 | onnx-fp32 | 7.976 | 9.615 | 7,926 |
| 64 | **onnx-int8** | **7.611** | **10.062** | **8,118** |

### The finding worth reading

Paired against fp32 ONNX in the same cell, with 95% intervals from paired
bootstrap resampling:

| Batch | Conc | int8 speedup vs fp32 | Verdict |
|---:|---:|---|---|
| 1 | 1 | **1.110x [1.075, 1.148]** | faster, CI excludes parity |
| 8 | 1 | 0.966x [0.935, 1.003] | no measurable difference |
| 32 | 1 | **1.081x [1.053, 1.110]** | faster |
| 64 | 1 | 0.998x [0.976, 1.021] | no measurable difference |
| **64** | **4** | **0.918x [0.883, 0.955]** | **slower** |

int8 is **not uniformly faster**. It wins at small batches — where per-call
overhead dominates and its smaller weights fit cache better — and *loses* at the
largest batch under concurrency, where the arithmetic dominates and the
quantise/dequantise steps are pure overhead.

TorchScript, meanwhile, is slower than fp32 ONNX in every cell measured
(0.80x–0.97x). Same weights, same maths, different engine.

**These two results are why the Interpretation section is empty.** The numbers say
*what happened*. They do not say which runtime to deploy, and that depends on the
batch-size and concurrency mix of real traffic — which the harness cannot know.

---

## Reproduction and verification

- **[VERIFICATION.md](VERIFICATION.md)** — the record of what was actually run, with
  real output: 248 tests, the benchmark run, the backup-and-restore drill, the
  containerised stack, and the alerting path exercised end to end. It also lists
  what could *not* be checked on the build machine, and the seventeen defects found
  and fixed during verification.

```bash
python tasks.py verify       # lint, tests, pipeline, bench, report, seed, drift, backup+restore
python tasks.py verify-all   # the above plus the container stack and smoke test
```

After a fresh clone, `artifacts/` and `results/` do not exist — they are generated
and gitignored. `python tasks.py pipeline` rebuilds them in about a minute.

---

## Design decisions

The four pieces below are the ones a language model should not write for you.
They are judgement calls about the experiment, not code generation, and they are
what a reviewer will probe.

The generator deliberately emits `results/REPORT.md` with an **empty
Interpretation section** and explicit TODO markers. If the same pipeline that
produced the table also produced the conclusion, a reader cannot tell which
claims were tested and which were assumed — and the first follow-up question
exposes it.

### 1. Why this experiment matrix?

`configs/bench.yaml` sweeps **runtime × batch size × concurrency × mode**, and
fixes the number of *calls* per cell rather than the number of samples:

- latency is a per-call distribution, so observation counts must be identical
  across cells or the p99 at batch 64 is estimated from a different sample size
  than the p99 at batch 1;
- throughput is a rate, so it normalises correctly even though a batch-64 call
  carries 64x the samples of a batch-1 call.

Batch size is swept because it changes the answer to "which runtime is fastest" —
demonstrated above. `inproc` and `http` are measured separately because a gap that
appears only in `http` is API overhead, not inference cost.

### 2. How should the int8 result be read?

Answered in the report's Interpretation section — but the harness gives you what
you need: the speedup interval, Cohen's d, and the accuracy Wilson interval. If the
speedup interval straddles 1.0, there is no measurable difference and saying
"int8 is faster" is wrong.

### 3. Why PSI, and where does the threshold sit?

Argued at the top of `configs/drift.yaml`. Summary: KS is a hypothesis test, and
with thousands of observations across 64 features it reports significance for
shifts too small to change model behaviour — while 64 uncorrected p-values expect
about three false positives per run. A detector that cries wolf gets muted, and a
muted detector is worse than none. PSI is a magnitude with conventional,
sample-size-independent thresholds, so 0.3 means the same thing on a 500-request
window as on a 50,000-request one. **PSI decides; KS corroborates.**

Also recorded: 3 of the 64 pixels are *constant* background in every training
image. A zero-variance feature has no distribution to drift from, so it is
excluded and counted rather than silently averaged in.

### 4. What was broken on purpose, and what fired?

See `docs/runbook.md`. The short version: during development the HTTP benchmark
stalled at ~12 minutes, which turned out to be two real defects — an
`O(n²)` row-count check on the write path, and `init_db()` running
`CREATE TABLE` on every insert. Both are fixed and both are now guarded by tests.

---

## Quickstart

### Local stack (Docker)

```bash
python tasks.py up        # builds the image, runs init, starts api + monitoring
python tasks.py smoke     # asserts health, inference, rejection paths, prometheus
open http://localhost:3000      # grafana, admin/admin
open http://localhost:8000/docs # api
python tasks.py down
```

### Local without Docker

```bash
python tasks.py install
python tasks.py pipeline   # train -> export -> drift reference
python tasks.py test       # unit tests, no docker needed
python tasks.py serve      # http://localhost:8000/docs
```

### Check everything works

```bash
python tasks.py verify       # lint, tests, pipeline, bench, report, seed, drift, backup+restore
python tasks.py verify-all   # the above plus the container stack and the smoke test
```

`verify` seeds the request log before the backup drill. On a fresh clone the log is
empty and the restore verifier **refuses to pass** — "a restore that verifies no
data is not a restore" — so seeding is what makes the drill meaningful rather than
ceremonial.

### A real VM (free)

```bash
cd terraform/envs/local && terraform init && terraform apply
# terraform prints the exact ansible-playbook command to run next
```

---

## Repository layout

```
configs/            model.yaml · serving.yaml · bench.yaml · drift.yaml
src/mlserve/
  model.py train.py export.py registry.py     the model and its three runtimes
  app.py schema.py server.py db.py            the service
  bench/runner.py stats.py report.py          the experiment
  drift/reference.py detect.py                PSI + KS
  ops/backup.py backup_cli.py                 verified backup and restore
monitoring/         prometheus · alertmanager · grafana · alert_sink.py
ansible/            playbook + 6 roles (base users firewall docker logrotate deploy)
terraform/          modules/host + envs/{local,free-tier}
tests/              200+ tests, including the failure paths of every guard
docs/               architecture.md · runbook.md
tasks.py            cross-platform task runner (stdlib only)
```

---

## Reproducibility

Every number in `results/` came from a specific configuration, and the config hash
is recorded in both `run_meta.json` and `REPORT.md`. If the working tree's config
does not match the recorded hash, the report describes a different experiment.

Guarantees, each with a test behind it:

- **Seeded splits.** Same seed, byte-identical train/val/test partition.
- **Per-cell seeds from a digest, not `hash()`.** Python randomises string hashing
  per process, so the built-in `hash` would make two runs of the same command draw
  different inputs and look like timing noise. `sha256` is used instead.
- **TorchScript, ONNX fp32 and ONNX int8 exported from the same `model.pt`**, with
  an enforced parity assertion.
- **Bootstrap intervals everywhere**, with the observation count printed next to
  each one.
- **Paired resampling for comparisons**, because two measurements in a cell share
  a machine and a thermal state.
- **Tail power reported per percentile.** At 210 observations per cell the p50 and
  p95 are powered and the p99 is not; the report says so rather than printing a
  number that is really the sample maximum.

---

## Limitations

Stated because a benchmark that does not state its limits is a sales document.

- **Toy model, toy data.** 136,586 parameters on 8x8 greyscale digits. Absolute
  latency does not transfer to a real vision or language model.
- **CPU only, single-threaded.** Pinned to one thread so the runtimes are
  comparable. Threaded inference and GPU serving are outside this experiment.
- **One host.** Every number includes this machine's scheduler and thermals, which
  is why every figure carries an interval and the CPU is recorded.
- **The int8 conclusion is architecture-dependent.** Here almost all parameters
  sit in the fully-connected layers, which is the best case for dynamic
  quantisation.
- **HTTP mode is one uvicorn worker.** Concurrency is queueing inside one process,
  not load balancing across several.
- **TorchScript is on a deprecation path** in recent PyTorch. Kept because it is
  still widely deployed and it is the reference the other two are checked against,
  but `torch.export` is the forward-looking comparison.
- **SQLite only** in the compose stack. The verified-restore drill does not yet
  handle Postgres, and shipping a database the backup cannot verify would be worse
  than not offering it. The migration is documented in `docker-compose.yml`.

---

## Licence

Apache 2.0. See `LICENSE`.
