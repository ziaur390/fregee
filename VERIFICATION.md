# Verification record

What was actually run, what it produced, and what could not be checked here.

Generated while building this repository on **2026-09-29**. Every command below was
executed; every number is copied from real output, not estimated. Where something
could not be verified, it is listed under [Not verified](#not-verified-here)
rather than quietly omitted.

---

## Environment

| | |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.13.1 |
| Docker | 29.8.0 (Docker Desktop) |
| CPU | Intel64 Family 6 Model 140 Stepping 2, GenuineIntel |
| torch | 2.11.0+cpu |
| onnxruntime | 1.22.0 |
| onnx | 1.23.0 |
| fastapi | 0.116.1 |
| numpy | 2.1.3 |
| scikit-learn | 1.7.1 |
| pandas | 2.2.3 |

`make` is **not** installed on this machine, and neither is `ansible`, `terraform`
or `promtool`. That is why the task runner is `tasks.py` (standard library only)
and the Makefile is a thin shim over it — and why the Ansible and Terraform checks
in the test suite parse the configuration rather than invoking the tools.

---

## 1. Lint and unit tests

```console
$ python tasks.py lint
All checks passed!
21 files already formatted

$ python tasks.py test
237 passed
```

212 test functions across 12 files; 237 test cases once parametrisation is
expanded.

| File | Tests | Covers |
|---|---:|---|
| `test_stats.py` | 28 | Bootstrap CIs, paired comparison, Wilson interval, tail power |
| `test_packaging.py` | 28 | Dockerfile, compose, workflows, writable paths |
| `test_api.py` | 24 | Health split, batch ceiling, validation, metrics, persistence |
| `test_drift.py` | 23 | PSI maths, no-drift vs shifted, collector publishing |
| `test_backup.py` | 20 | Backup, verified restore, and five corruption failure paths |
| `test_monitoring.py` | 18 | Alert rules cross-checked against real metric names |
| `test_registry.py` | 14 | Runtime loading, parity, dynamic batch axis |
| `test_terraform.py` | 14 | No credentials in source, references resolve, free-tier limits |
| `test_ansible.py` | 12 | Idempotency guards, handler resolution, firewall parity |
| `test_alert_sink.py` | 11 | Webhook parsing and HTTP behaviour |
| `test_model.py` | 10 | Split reproducibility, shapes, stratified sampling |
| `test_schema.py` | 10 | Trust-boundary validation |

---

## 2. Model training

```console
$ python tasks.py train
seed            1337  (config b7eb7e03ea6c)
parameters      136,586
splits          {'train': 1257, 'val': 270, 'test': 270}
best val acc    0.9778
test accuracy   0.9741
test macro F1   0.9740
```

Reproducibility confirmed by test: the same seed produces a byte-identical split,
and identical initial weights.

---

## 3. Runtime export and parity

```console
$ python tasks.py export
wrote model.ts
wrote model.onnx
wrote model.int8.onnx
      quantised ops: ['DynamicQuantizeLinear']  (mode: dynamic (weight-only))
      rejected: matmul+gemm+conv: quantised model unusable:
                NotImplemented: Could not find an implementation for
                ConvInteger(10) node

runtime         accuracy  disagree   size KiB   load ms
torchscript       0.9741    0.0000      550.5      8.67
onnx-fp32         0.9741    0.0000      535.0      2.41
onnx-int8         0.9741    0.0000      151.5      2.51

parity OK - all runtimes agree within tolerance
```

**A real defect was found and fixed here.** `quantize_dynamic` was asked to
quantise `Conv` as well as `MatMul`/`Gemm`. It emitted `ConvInteger` nodes happily,
and onnxruntime's CPU provider then refused to *load* the result. The original code
only guarded the quantiser call, so it produced a file that could not be loaded and
crashed later.

The fix is that every quantisation candidate is now **loaded and executed** before
being accepted. The consequence is recorded honestly rather than hidden: the 3.5x
size reduction comes entirely from the fully-connected layers, and the convolution
weights remain fp32.

---

## 4. Benchmark

```console
$ python tasks.py bench
mlserve-ops benchmark  seed=1337
  matrix      48 cells (2 modes x 3 runtimes x 4 batches x 2 concurrency)
  calls       3600 timed calls, 5 warmup per trial discarded
  intervals   2000 bootstrap resamples at 95%
  cpu         Intel64 Family 6 Model 140 Stepping 2, GenuineIntel (AMD64)
...
wrote results\raw.csv (10440 measured calls)

real    1m6.152s
```

10,440 measured calls in 66 seconds.

```console
$ python tasks.py report
wrote results\REPORT.md
wrote results\latency.png, results\throughput.png, results\concurrency.png

The Interpretation section is intentionally empty - it has TODO markers.
Those are the conclusions the harness cannot draw for you.
```

### Results that emerged

Paired against fp32 ONNX in the same cell, 95% intervals from paired bootstrap:

| Batch | Conc | int8 speedup vs fp32 | Verdict |
|---:|---:|---|---|
| 1 | 1 | 1.110x [1.075, 1.148] | faster, CI excludes parity |
| 8 | 1 | 0.966x [0.935, 1.003] | no measurable difference |
| 32 | 1 | 1.081x [1.053, 1.110] | faster |
| 64 | 1 | 0.998x [0.976, 1.021] | no measurable difference |
| 64 | 4 | **0.918x [0.883, 0.955]** | **slower** |

int8 is **not uniformly faster** — it wins at small batches and loses at the
largest batch under concurrency. TorchScript was slower than fp32 ONNX in every
cell measured (0.80x–0.97x).

### Tail power

The report states which percentile columns are real rather than printing a number
for all of them:

```
| percentile | observations needed | status in this run |
| p50 | 20    | powered |
| p90 | 101   | powered |
| p95 | 200   | powered |
| p99 | 1,000 | **NOT powered** (short by 790 per cell) |
```

The sample size was derived from this table, not chosen for convenience:
`calls_per_repeat: 70` gives 210 observations, which is the minimum for a p95 to
have 10 observations in its tail.

---

## 5. Drift detection

```console
$ python tasks.py reference
reference built from 1257 training samples
  features    64 (61 informative, 3 constant and excluded)
  bins        10 (quantile)
```

Four features of the digits dataset are constant background in every training
image. A zero-variance feature has no distribution to drift from, so PSI on it is
0 by construction — those are excluded from scoring and counted, not averaged in.

### No-drift baseline (benchmark traffic)

```console
$ python tasks.py drift
drift status    STABLE
requests        2000 (reference n=1257)
max PSI         0.0083
mean PSI        0.0025
features        0 watched, 0 alerted of 61 scored (3 constant features excluded)
wrote           results\drift.json
wrote           results\drift.prom
```

### Injected shift (unit test)

```console
$ pytest tests/test_drift.py
23 passed
```

Two tests carry the weight here: `test_no_drift_window_is_stable` (clean traffic
must not trip the detector) and `test_shifted_window_alerts` (a 0.35 pixel shift
must). A detector that always returned "stable" would pass every other test in the
file.

### Live, inside the container

384 random pixel rows were sent through the running API — uniform noise, maximally
unlike digit images:

```console
$ docker compose exec -T drift python -m mlserve.drift.detect
features        48 watched, 47 alerted of 61 scored (3 constant features excluded)
  feature[ 1]  psi= 3.361  alert   ks_p=7.34e-160
  feature[ 2]  psi= 0.306  alert   ks_p=6.47e-15
  feature[ 4]  psi= 3.393  alert   ks_p=8.16e-31
  ...
wrote           /results/drift.json
wrote           /results/drift.prom
wrote           /textfile/drift.prom (node_exporter textfile collector)
```

---

## 6. Backup and verified restore

```console
$ python tasks.py backup
backup written  backups\mlserve-20260929T011643Z.tar.gz
  rows          50000
  artifacts     9
  database      mlserve.db
  size          8133.2 KiB

$ python tasks.py restore-verify
verifying backups\mlserve-20260929T011643Z.tar.gz
  [ok] manifest read: created 2026-09-29T01:16:43+00:00, 9 artifacts
  [ok] artifact checksums match: 9 files
  [ok] database checksum matches the manifest
  [ok] row-count parity: 50000 rows restored == 50000 in manifest

verified: 50000 rows restored and matched the manifest
```

### Failure paths, exercised

Five tests corrupt an archive on purpose and assert a non-zero result. A verifier
that has only ever been observed to pass has not been tested:

- `test_corrupted_database_is_detected` — byte flipped past the SQLite header
- `test_deleted_database_is_detected` — database removed from the archive
- `test_missing_manifest_is_detected` — manifest removed
- `test_row_count_mismatch_is_detected` — manifest doctored; every file intact, so
  only the parity check catches it
- `test_tampered_artifact_is_detected` — an artifact's contents changed

---

## 7. Containerised stack

```console
$ docker compose build
Image mlserve-ops:local Built    (484 MB)

$ docker compose up -d --wait
Container mlserve-node-exporter  Healthy
Container mlserve-alert-sink     Healthy
Container mlserve-api            Healthy
Container mlserve-prometheus     Healthy
Container mlserve-alertmanager   Healthy
Container mlserve-grafana        Healthy
Container mlserve-drift          Started
Container mlserve-backup         Healthy
Container mlserve-init           Exited
```

Nine services. `init` runs to completion (train → export → reference) before the
API starts; `docker compose down -v` then `up` reproduces it.

### Smoke test against the running stack

```console
$ python tasks.py smoke
ops endpoints
  [PASS] /healthz 200
  [PASS] /readyz 200
  [PASS] /readyz reports no missing runtimes
  [PASS] /models lists 3 runtimes
  [PASS] /metrics 200
  [PASS] /metrics exposes mlserve_requests_total
  [PASS] /metrics exposes mlserve_request_duration_seconds
inference
  [PASS] /predict 200
  [PASS] /predict returns a class
  [PASS] /predict reports latency
  [PASS] /predict/batch 200
  [PASS] /predict/batch returns 8 rows
  [PASS] /predict runtime=torchscript
  [PASS] /predict runtime=onnx-fp32
  [PASS] /predict runtime=onnx-int8
rejection paths
  [PASS] /predict rejects 10 features (422)
  [PASS] /predict rejects out-of-range (422)
  [PASS] /predict/batch rejects 500 rows (413)
persistence
  [PASS] /stats 200
  [PASS] /stats shows persisted rows
prometheus
  [PASS] prometheus reachable
  [PASS] prometheus has an UP target - up targets=3

all smoke checks passed
```

### Prometheus

3 targets UP, 12 alert rules loaded across 3 groups.

### Grafana

Provisioned with no manual setup:

```
folders:    ['mlserve-ops']
dashboards: [('mlserve-ops — serving, errors and drift', 'mlserve-ops')]
panels:     9  [Latency percentiles by runtime, Throughput (samples / second),
             Requests per second by endpoint, Error rate, Model loaded,
             Batch size distribution, Input drift (PSI), Drift detector health,
             Runtime artifact size]
datasource: Prometheus -> http://prometheus:9090 (default)
```

---

## 8. The alerting path, end to end

This is the claim that alerting works, so it was tested rather than asserted. The
whole path was exercised with the drift detector's genuine firing:

1. Drift job wrote `drift.prom` into the collector directory with an atomic rename
2. node-exporter's textfile collector exposed it

   ```
   mlserve_drift_alert 1
   mlserve_drift_features_alerted 47
   mlserve_drift_measured 1
   mlserve_drift_psi_max 4.385649
   mlserve_drift_requests 375
   ```

3. Prometheus evaluated the rule and reported it firing

   ```
   firing   MLServeDriftDetected           warning
   ```

4. Alertmanager accepted and routed it

   ```
   MLServeDriftDetected severity=warning component=drift status=active
   ```

5. The webhook receiver logged the arrival

   ```
   [02:40:44] FIRING   warning  MLServeDriftDetected component=drift runtime=-
              summary: input drift detected
              detail:  Check results/drift.json for which features moved...
   ```

Every link in the chain produced observable evidence. Without a local receiver,
step 5 would have been a config file and an assumption.

---

## 9. Defects found and fixed during verification

Listed because they are the substance of this record. Each is now guarded by a
test.

| Defect | Symptom | Fix |
|---|---|---|
| `[None, None, :, :]` indexing | Every single-sample request raised `IndexError`; only 4-D arrays accepted it | `reshape(1, 1, 8, 8)` |
| NaN in the error response | Malformed input returned **500**, not 422 — pydantic echoed the NaN and the error body failed to serialise | `_json_safe` sanitiser on the validation handler |
| `O(n²)` row count on the write path | HTTP benchmark stalled at ~12 minutes; a `COUNT(*)` ran after every insert | Pruning moved to the drift cron job |
| `init_db()` per insert | Schema query on every write | Memoised per URL |
| `hash()` for per-cell seeds | Reproducibility silently broken — Python randomises string hashing per process, so two runs of the same command drew different inputs | `sha256` digest |
| `ConvInteger` export | int8 model quantised and then refused to load | Candidates are executed before being accepted |
| `np.unique` on quantile edges | Different edge counts per feature; the reference shape was data-dependent | Duplicate edges allowed; epsilon keeps PSI finite |
| PSI blind on constant features | 3 zero-variance pixels diluted every average | Informative mask; excluded and reported |
| `observations_needed` float error | Reported 101 observations needed for p90 instead of 100 (`10/0.1 = 100.00000000000001`) | Integer arithmetic |
| Tail-power formula | Multiplied an already-percentage value by 100, so every row was flagged and the column became useless | `n * (1 - p/100) < min_tail` |
| `with sqlite3.connect(...)` | Does not close the handle — leaked a file lock per backup run on a timer | `contextlib.closing` |
| `MLSERVE_RESULTS` not exported | `init` died with `PermissionError: /app/results`; the code wrote to a root-owned directory the non-root user could not create | Dockerfile exports the variable; enforced by test |
| `write_outputs` joined onto `ROOT` | Drift job died with the same `PermissionError` | `resolve_output_path` against `RESULTS` |
| Image-level `HEALTHCHECK` | Five services built from one image; only one serves HTTP, so four reported unhealthy and `up --wait` never returned | Healthchecks moved per service in compose |
| Root-owned named volume | The drift textfile was written but never reached the collector | `/textfile` created and chowned in the image; the separate publisher service was then deleted entirely |
| `NaN` published as a drift value | An unmeasured window looked like a measurement | `mlserve_drift_measured` gauge; PSI series omitted when unmeasured |

---

## Not verified here

Honest list of what this machine could not check. Everything else in this document
was executed.

| Item | Why not | How to verify |
|---|---|---|
| **Ansible apply** | `ansible` is not installed, and a full apply against localhost would reconfigure this machine | `ansible-playbook --syntax-check ansible/playbook.yml`, then against a Multipass VM; expect **zero changed tasks on the second run** — that is the idempotency test |
| **Terraform** | `terraform` is not installed | `terraform fmt -check`, `terraform validate`, then `plan` in `terraform/envs/local`. The multipass provider is third-party and its exact attribute surface is the least certain thing in this repo — a CLI fallback is documented at the top of `envs/local/main.tf` |
| **Oracle Cloud free tier** | No tenancy, and provisioning a real instance was out of scope | `terraform apply` in `envs/free-tier` with a `terraform.tfvars` copied from the example |
| **GitHub Actions** | Workflows only run after a push | Push, then confirm `ci`, `bench` and `security` go green |
| **`promtool check rules`** | `promtool` is not installed | `docker run --rm -v $PWD/monitoring/prometheus:/m prom/prometheus promtool check rules /m/alerts.yml`. The test suite already cross-checks every metric name against the ones the code exposes, which is the failure mode that matters |
| **`torch.jit.load` deprecation** | Torch 2.11 warns that TorchScript is on a deprecation path | Deliberate. It is still widely deployed and is the reference the other runtimes are checked against. `torch.export` is the forward-looking comparison and is noted in the limitations |
| **Postgres path** | The restore drill implements SQLite only | Documented in `docker-compose.yml` with the exact three-step migration. Shipping a database the backup cannot verify would be worse than not offering it |
| **Grafana renders correctly** | The API confirms the dashboard and 9 panels are provisioned; visual rendering was not inspected | Open `http://localhost:3000`, check the *Runtime* variable populates and panels show data |

---

## Reproducing this record

```bash
git clone https://github.com/ziaur390/fregee.git && cd fregee
python tasks.py install
python tasks.py pipeline      # train -> export -> drift reference
python tasks.py test          # 237 tests
python tasks.py bench         # ~66s, writes results/raw.csv
python tasks.py report        # writes results/REPORT.md
python tasks.py backup && python tasks.py restore-verify
python tasks.py up            # docker stack
python tasks.py smoke
```

The config hash and seed are recorded in `results/run_meta.json` and printed in
`results/REPORT.md`. If the working tree's `configs/model.yaml` or
`configs/bench.yaml` does not match that hash, the report describes a different
experiment.
