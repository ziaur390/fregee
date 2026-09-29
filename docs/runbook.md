# Runbook

One section per alert in `monitoring/prometheus/alerts.yml`. Each states what the
alert actually means, the triage order, and what to do.

The triage orders are ordered deliberately. Most of them start by ruling out a
*measurement* problem before an *infrastructure* problem, because a broken metric
is far more common than a broken service and chasing a phantom outage is how an
on-call rotation burns out.

---

## MLServeModelNotLoaded

**Severity** critical · **Fires after** 1m

`mlserve_model_loaded{runtime}` is 0. Loading is lazy, so a 0 on an idle instance
just means nothing has been served yet — which is why the alert requires one
minute of sustained zero rather than firing on first observation.

1. `curl -s localhost:8000/readyz` — does it return 503, and which runtimes does it
   list as missing?
2. Check whether this is **every** replica or one. Every replica means the
   artefact is missing; one replica means that process is unhealthy.
3. If artefacts are missing, the deploy volume is at fault:
   `ls -la /var/lib/mlserve/artifacts` and compare against `/models`.
4. If one process is unhealthy, read its log — a corrupt `model.ts` or a
   `model.onnx` that fails `onnx.checker` produces a load exception at first use.
5. Rebuild with `python tasks.py pipeline`, then restart.

**Do not** restart repeatedly. If the artefact is missing, a restart loop just
buries the cause. The inhibition rules already suppress the latency and error
alerts this causes, so you are not being flooded and can take the time.

---

## MLServeP99LatencyBreach / MLServeP99LatencyCritical

**Severity** warning (>50 ms) / critical (>250 ms)

### Triage in this order

**1. Did the traffic shape change?** This is the most common cause and the least
likely to be a real problem. A migration from batch 1 to batch 64 raises latency by
design — see the benchmark table in the README. Check the *Batch size distribution*
panel in Grafana before anything else. If mass moved right, close the alert.

**2. Did the runtime change?** `/models` shows which runtimes are loaded, and the
latency panel is labelled by runtime. Serving `torchscript` instead of
`onnx-fp32` accounts for a 1.25x latency increase on its own — measured, not
assumed.

**3. Is it concurrency?** Compare the p50 against the p99. Queueing shows up as a
wide gap with a stable p50: requests are waiting, not running slower. Check the
`concurrency` axis in `results/raw.csv` for what this service does at the current
level.

**4. Is the host saturated?** `MLServeP99LatencyCritical` with a rising p50 across
*all* runtimes usually means CPU throttling or memory pressure, not the model.
Check `systemctl status mlserve-api` for `MemoryMax` hits.

**5. Only then** suspect a code regression. Compare against a fresh
`python tasks.py bench`, which reproduces the whole matrix on this host.

---

## MLServeErrorRateHigh

**Severity** warning · 5xx share above 2% for 5m

2% is the threshold because below it a single misbehaving client sending malformed
payloads can trip the alert, and above it the failures are systemic.

1. `curl -s localhost:8000/metrics | grep inference_errors` — is it inference
   failing, or something else?
2. Check the Grafana *Error rate* panel for which runtime is failing. A single
   runtime failing usually means a bad artefact for that runtime only — the other
   two will be serving normally.
3. `docker compose logs api --tail 200` for the traceback.

Note that **validation errors are not 5xx**. A client sending wrong-shaped payloads
gets 422 and appears on `MLServeValidationErrorSpike`, not here. If your 5xx rate
is up with no validation errors, the fault is server-side.

---

## MLServeInferenceErrors

**Severity** warning · any increase in 10m

Any inference exception is a defect, not load, so there is no rate threshold. A
single one is worth knowing about.

Common causes, in order of likelihood seen here:

- A tensor returned non-contiguous from the batch reshape. The app reshapes to
  `(-1, 1, 8, 8)` — an earlier version used `[None, None, :, :]`, which works on a
  4-D array and raises `IndexError` on a 1-D one, so every single-sample request
  failed. That bug is now covered by `tests/test_api.py`.
- An ONNX session that loaded but cannot execute an op. `ConvInteger` is the known
  case: `quantize_dynamic` emits it happily and onnxruntime CPU then refuses to
  load it.

---

## MLServeDbWriteFailures

**Severity** warning · any increase in 10m

**This one is easy to miss and it matters.** Predictions still succeed when
persistence fails, which is deliberate — a logging outage must not take down
inference. The cost is that the request log can silently lose rows.

Why it matters beyond the log: the **drift detector reads this table**. A sustained
write failure blinds drift detection too, and the drift gauge will simply freeze at
its last value, which looks exactly like a stable service.

1. Check disk space first: `df -h`. A full disk is the usual cause.
2. `sqlite3 /var/lib/mlserve/mlserve.db "PRAGMA integrity_check;"` — the database
   itself may be corrupt.
3. Confirm the row cap is being enforced. `db.prune` runs from the drift job, not
   from the write path (counting rows on every insert made writes quadratic and
   stalled the benchmark), so a stalled drift job means the log grows unbounded.

---

## MLServeDriftDetected

**Severity** warning · `mlserve_drift_alert == 1`

**Drift is not automatically a fault.** Inputs changing is normal. What drift means
is that the model is now being asked about a distribution it was not fitted on, and
its accuracy on that traffic is unmeasured.

1. `cat results/drift.json` — it lists the drifted features with their PSI, band and
   KS statistic.
2. Ask **which** features moved. Pixel-level features are hard to interpret
   directly; map the index back with `idx // 8, idx % 8` for row/column, or
   visualise a mean image from the recent window against the reference.
3. Ask whether this is **real traffic change or an upstream bug**. A preprocessing
   change that forgets the divide-by-16 normalisation produces a massive, obvious
   PSI on nearly every feature and is a bug, not drift. A slow change on a few
   features is more likely genuine.
4. If genuine: the model needs retraining on a window that includes the new
   distribution. The reference `artifacts/reference.npz` should be rebuilt from the
   *new* training data, not from production traffic — production traffic is
   unlabelled.

`MLServeDriftDetectorStale` is the more dangerous alert, because it means you
cannot trust this one at all.

---

## MLServeDriftDetectorStale

**Severity** warning · no run in 2 hours

The drift metrics are only as fresh as the last write to the textfile. A stalled
job leaves the last value in place forever, so **a dead drift detector and a stable
service look identical on every dashboard**. That is the whole reason this alert
exists.

1. `systemctl list-timers mlserve-drift.timer` — is the timer scheduled?
2. `systemctl status mlserve-drift.service` — check `LastTriggerUSec` and the exit
   code. Remember exit code **2 means an alert was found**, not a crash;
   `SuccessExitStatus=2` is configured for that reason.
3. `journalctl -u mlserve-drift -n 50` for the actual error.
4. Confirm the textfile is reaching the collector:
   `ls -la /var/lib/node_exporter/textfile/drift.prom`. If it exists but the metric
   is stale, the publisher's atomic rename is failing or node-exporter cannot read
   the directory.

---

## MLServeValidationErrorSpike

**Severity** info · more than 1 rejection/second for 10m

Some rejection is healthy — the API rejects wrong-length vectors, non-finite
values, and pixels outside the trained range on purpose. Sustained rejection means
**a caller changed its payload shape and is now failing on its own side**, usually
without noticing.

1. `curl -s localhost:8000/metrics | grep validation_errors_total` — which
   endpoint?
2. Reproduce the shape from the 422 body. The error names the field and index.
3. The most likely culprit is a client sending **raw 0–16 pixels** instead of
   normalised 0–1. That is rejected by design: the model was trained on normalised
   input, and serving raw pixels would produce confident nonsense rather than an
   error.

---

## MLServeTailPowerLow

**Severity** info · fewer than 200 requests in the comparison window

This one is about the *experiment*, not the service. Below 200 observations, a p95
cannot be estimated to the standard this repository holds itself to
(`min_tail_observations: 10` in `configs/bench.yaml`), and the drift job reports
`insufficient_data` rather than a score.

Usual cause is simply low traffic. Either wait, or lower
`configs/bench.yaml`'s `calls_per_repeat` and record in the report that the p95 is
no longer powered — **do not** lower the standard to make the warning go away.
Reporting an unpowered percentile as if it were measured is the failure this alert
exists to prevent.

---

## Things that are not alerts but will page you anyway

### The whole stack is up but `/metrics` is empty

Prometheus scraped before the API was ready. `depends_on: service_healthy` should
prevent this; if it happens, check `docker compose ps` for a container stuck in
`starting`.

### Grafana shows "No data" on every panel

Datasource URL. It must be `http://prometheus:9090` (the compose service name), not
`localhost:9090`. Provisioning sets this, so a manual change is the usual cause.

### `docker compose up` hangs on `init`

`init` runs train → export → reference. Export performs a **parity assertion** and
exits non-zero if a runtime disagrees with TorchScript beyond tolerance. That is
intentional: a silently broken export would make every benchmark number
meaningless. Read the init logs — it names the runtime and the disagreement rate.
