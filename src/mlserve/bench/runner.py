"""Benchmark runner.

Run: ``python tasks.py bench``  (or ``make bench``)

Executes the matrix in ``configs/bench.yaml`` and writes ``results/raw.csv``,
``results/accuracy.csv``, ``results/coldstart.csv`` and ``results/run_meta.json``.

Design notes that matter for whether the numbers mean anything:

* **Warmup is discarded.** The first call through a TorchScript module and the
  first call through an ONNX session both do lazy graph work. Including them
  measures initialisation and labels it inference cost.

* **Leaf measurement is the call, not the sample.** One HTTP request is one
  latency observation regardless of how many samples rode in it. Per-sample
  latency would make a batch-64 request look 64x slower than it is.

* **inproc and http are separate modes.** inproc isolates the runtime; http
  includes JSON parsing, validation and ASGI overhead. A difference that appears
  in http but not in inproc is an API effect, not a runtime effect - and if only
  one mode were measured, that would be invisible.

* **The HTTP server is started and stopped by this script.** `make bench` needs
  no manual setup, which is what makes the result reproducible rather than
  merely repeatable on one person's machine.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mlserve import registry
from mlserve.bench.stats import wilson_interval
from mlserve.config import (
    RESULTS,
    ROOT,
    ensure_dirs,
    load_config,
    read_json,
    set_seed,
    write_json,
)
from mlserve.model import load_digits_split
from mlserve.train import config_hash


@dataclass
class CallSample:
    """One measured call. One row in raw.csv."""

    mode: str
    runtime: str
    batch_size: int
    concurrency: int
    repeat: int
    call_index: int
    n_samples: int
    latency_ms: float
    samples_per_second: float


@dataclass
class Cell:
    """A configuration being measured, and the calls observed for it."""

    mode: str
    runtime: str
    batch_size: int
    concurrency: int

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.mode, self.runtime, self.batch_size, self.concurrency)

    @property
    def label(self) -> str:
        return f"{self.mode}/{self.runtime}/b{self.batch_size}/c{self.concurrency}"


# --------------------------------------------------------------------------- helpers


def _cpu_description() -> str:
    """Best-effort CPU string. Reported so a timing difference can be attributed."""
    machine = platform.machine()
    cpu = ""
    try:
        if sys.platform == "linux":
            with Path("/proc/cpuinfo").open(encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
        elif sys.platform == "darwin":
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        elif sys.platform == "win32":
            cpu = os.environ.get("PROCESSOR_IDENTIFIER", "") or platform.processor() or "unknown"
    except Exception:  # noqa: BLE001 - this is descriptive metadata, never fatal
        cpu = "unknown"
    return f"{cpu or 'unknown'} ({machine})"


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for module in ("numpy", "pandas", "torch", "onnxruntime", "onnx", "sklearn", "fastapi"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            versions[module] = "missing"
    return versions


def _cell_seed(base_seed: int, key: tuple, repeat: int) -> int:
    """Deterministic per-cell seed.

    Deliberately NOT ``hash((key, repeat))``. Python randomises str and bytes
    hashing per process unless PYTHONHASHSEED is set before the interpreter
    starts, so the built-in hash makes the input sequence differ between two runs
    of the same command. That would quietly invalidate the reproducibility claim
    this repository is built around, and it would look like ordinary run-to-run
    timing noise rather than a bug. sha256 is stable across processes and
    versions.
    """
    digest = hashlib.sha256(f"{key}|{repeat}".encode()).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % 100_000


def _free_port(preferred: int) -> int:
    """Return ``preferred`` if it is free, otherwise an ephemeral port."""
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])


@contextlib.contextmanager
def serve(cfg: dict) -> Iterator[str]:
    """Start the API in a subprocess and yield its base URL.

    Started fresh per bench run so the numbers are not affected by a long-running
    process that has different memory layout or caches.
    """
    http_cfg = cfg["http"]
    port = _free_port(int(http_cfg["port"]))
    env = dict(os.environ)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    # Keep the server single-threaded and quiet so it is not the variable.
    env["MLSERVE_TORCH_THREADS"] = env.get("MLSERVE_TORCH_THREADS", "1")
    env["MLSERVE_PORT"] = str(port)
    env["MLSERVE_ACCESS_LOG"] = "0"
    # Request logging OFF for the duration of the benchmark. Persistence happens
    # in a background task, which Starlette completes before the keep-alive
    # connection is reused - so 64 sequential INSERTs per batch-64 request would
    # show up in the next request's latency. This experiment measures inference
    # serving; persistence is a separate subsystem whose cost is not measured
    # anywhere here, and leaving it on would silently blend the two.
    env["MLSERVE_REQUEST_LOG"] = "0"

    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "mlserve.server"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    import httpx

    deadline = time.time() + float(http_cfg["startup_timeout_s"])
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            err = (proc.stderr.read() or b"").decode(errors="replace")
            raise RuntimeError(f"api exited during startup (code {proc.returncode}):\n{err}")
        try:
            if httpx.get(f"{base}/readyz", timeout=2.0).status_code == 200:
                ready = True
                break
        except Exception:  # noqa: BLE001 - still starting
            time.sleep(0.25)

    if not ready:
        proc.terminate()
        raise RuntimeError(
            f"api did not become ready within {http_cfg['startup_timeout_s']}s at {base}"
        )

    try:
        yield base
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:  # pragma: no cover - stubborn child
            proc.kill()


# --------------------------------------------------------------------------- measurement


def _sample_pool(seed: int) -> np.ndarray:
    """Features used for timing.

    Drawn from the whole dataset rather than the test split: latency does not
    depend on labels, and a larger pool means consecutive trials see different
    inputs instead of replaying the same 270 rows.
    """
    from sklearn.datasets import load_digits

    digits = load_digits()
    x = digits.images.astype(np.float32)[:, None, :, :] / 16.0
    rng = np.random.default_rng(seed)
    return x[rng.permutation(len(x))]


def _batched(pool: np.ndarray, calls: int, batch_size: int, seed: int) -> list[np.ndarray]:
    """``calls`` consecutive windows of ``batch_size`` rows, wrapping if needed."""
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, len(pool) - batch_size)))
    out: list[np.ndarray] = []
    for i in range(calls):
        offset = (start + i * batch_size) % max(1, len(pool) - batch_size)
        out.append(pool[offset : offset + batch_size])
    return out


def measure_inproc(cell: Cell, batches: list[np.ndarray], warmup: int) -> list[CallSample]:
    """Time registry calls directly. Only the runtime is in the measurement."""
    predictor = registry.load(cell.runtime)

    for batch in batches[:warmup]:
        predictor.predict(batch)

    samples: list[CallSample] = []
    for index, batch in enumerate(batches):
        started = time.perf_counter()
        predictor.predict(batch)
        latency_ms = (time.perf_counter() - started) * 1000.0
        samples.append(
            CallSample(
                mode=cell.mode,
                runtime=cell.runtime,
                batch_size=cell.batch_size,
                concurrency=cell.concurrency,
                repeat=0,  # set by the caller
                call_index=index,
                n_samples=len(batch),
                latency_ms=latency_ms,
                samples_per_second=len(batch) / max(latency_ms / 1000.0, 1e-9),
            )
        )
    return samples


def measure_http(
    cell: Cell, batches: list[np.ndarray], warmup: int, base_url: str, timeout: float
) -> list[CallSample]:
    """Time real HTTP round-trips through the API.

    A single reused client, because measuring TCP and TLS setup on every call
    would be measuring connection establishment under the name of inference.
    """
    import httpx

    url = f"{base_url}/predict/batch?runtime={cell.runtime}"
    payloads = [{"features": batch.reshape(len(batch), -1).tolist()} for batch in batches]

    samples: list[CallSample] = []
    lock_index = [0]

    with httpx.Client(
        timeout=timeout, limits=httpx.Limits(max_keepalive_connections=cell.concurrency + 2)
    ) as client:
        for payload in payloads[:warmup]:
            client.post(url, json=payload)

        def one(payload: dict) -> CallSample:
            started = time.perf_counter()
            response = client.post(url, json=payload)
            latency_ms = (time.perf_counter() - started) * 1000.0
            response.raise_for_status()
            n = len(payload["features"])
            with_lock = lock_index[0]
            lock_index[0] += 1
            return CallSample(
                mode=cell.mode,
                runtime=cell.runtime,
                batch_size=cell.batch_size,
                concurrency=cell.concurrency,
                repeat=0,
                call_index=with_lock,
                n_samples=n,
                latency_ms=latency_ms,
                samples_per_second=n / max(latency_ms / 1000.0, 1e-9),
            )

        if cell.concurrency == 1:
            samples = [one(payload) for payload in payloads[warmup:]]
        else:
            with ThreadPoolExecutor(max_workers=cell.concurrency) as pool:
                samples = list(pool.map(one, payloads[warmup:]))

    samples.sort(key=lambda s: s.call_index)
    return samples


# --------------------------------------------------------------------------- accuracy


def measure_accuracy(cfg: dict, runtimes: list[str]) -> list[dict[str, object]]:
    """Accuracy parity across runtimes, with a Wilson interval.

    Separate from timing on purpose. Running this inside the timing loop would
    both corrupt the latency samples and confuse two different questions.
    """
    from sklearn.metrics import f1_score

    seed = int(cfg["seed"])
    split = load_digits_split(seed=seed, val_size=0.15, test_size=0.15)
    rows: list[dict[str, object]] = []

    registry.clear_cache()
    for name in runtimes:
        predictor = registry.load(name)
        started = time.perf_counter()
        logits = predictor.predict(split.x_test)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        labels = logits.argmax(axis=1)
        correct = int((labels == split.y_test).sum())
        total = int(len(split.y_test))
        interval = wilson_interval(correct, total, alpha=float(cfg["ci_alpha"]))
        rows.append(
            {
                "runtime": name,
                "n": total,
                "correct": correct,
                "accuracy": interval.point,
                "acc_ci_low": interval.low,
                "acc_ci_high": interval.high,
                "macro_f1": float(f1_score(split.y_test, labels, average="macro")),
                "size_bytes": registry.artifact_size_bytes(name),
                "inference_ms_total": round(elapsed_ms, 3),
            }
        )
    return rows


def measure_cold_start(runtimes: list[str], repeats: int = 5) -> list[dict[str, object]]:
    """Load time and RSS delta, with the cache cleared before every trial."""
    rows: list[dict[str, object]] = []
    for name in runtimes:
        times: list[float] = []
        for _ in range(repeats):
            registry.clear_cache()
            started = time.perf_counter()
            registry.load(name)
            times.append((time.perf_counter() - started) * 1000.0)
        arr = np.asarray(times, dtype=float)
        rows.append(
            {
                "runtime": name,
                "repeats": repeats,
                "cold_start_ms_mean": float(arr.mean()),
                "cold_start_ms_min": float(arr.min()),
                "cold_start_ms_max": float(arr.max()),
                "size_bytes": registry.artifact_size_bytes(name),
            }
        )
    return rows


# --------------------------------------------------------------------------- driver


def build_matrix(cfg: dict) -> list[Cell]:
    return [
        Cell(mode=mode, runtime=runtime, batch_size=batch, concurrency=conc)
        for mode in cfg["modes"]
        for runtime in cfg["runtimes"]
        for batch in cfg["batch_sizes"]
        for conc in cfg["concurrency"]
    ]


def run(cfg: dict, *, modes: list[str] | None = None, quiet: bool = False) -> Path:
    """Execute the matrix. Returns the path to raw.csv."""
    ensure_dirs()
    seed = int(cfg["seed"])
    cells = build_matrix(cfg)
    if modes:
        cells = [cell for cell in cells if cell.mode in modes]

    pool = _sample_pool(seed)
    calls = int(cfg["calls_per_repeat"])
    warmup = int(cfg["warmup_calls"])
    repeats = int(cfg["repeats"])

    all_samples: list[CallSample] = []
    started_all = time.perf_counter()

    http_ctx = (
        serve(cfg) if any(cell.mode == "http" for cell in cells) else contextlib.nullcontext(None)
    )
    with http_ctx as base_url:
        for cell in cells:
            cell_started = time.perf_counter()
            for repeat in range(repeats):
                # Seed per (cell, repeat) so the input sequence is reproducible
                # while still varying across trials.
                cell_seed = _cell_seed(seed, cell.key, repeat)
                batches = _batched(pool, calls + warmup, cell.batch_size, cell_seed)

                if cell.mode == "inproc":
                    samples = measure_inproc(cell, batches, warmup)
                elif cell.mode == "http":
                    assert base_url is not None
                    samples = measure_http(
                        cell, batches, warmup, base_url, float(cfg["http"]["request_timeout_s"])
                    )
                else:  # pragma: no cover - config guarded
                    raise ValueError(f"unknown mode: {cell.mode}")

                for sample in samples:
                    sample.repeat = repeat
                all_samples.extend(samples)

            if not quiet:
                latencies = [
                    s.latency_ms
                    for s in all_samples
                    if (s.mode, s.runtime, s.batch_size, s.concurrency) == cell.key
                ]
                p50 = float(np.percentile(latencies, 50)) if latencies else float("nan")
                elapsed = time.perf_counter() - cell_started
                print(
                    f"  {cell.label:<44} p50={p50:8.3f} ms  n={len(latencies):4d}  ({elapsed:5.1f}s)",
                    flush=True,
                )

    raw_path = RESULTS / "raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(all_samples[0]).keys()))
        writer.writeheader()
        for sample in all_samples:
            writer.writerow(asdict(sample))

    accuracy_rows = (
        measure_accuracy(cfg, list(cfg["runtimes"]))
        if cfg.get("accuracy", {}).get("enabled", True)
        else []
    )
    if accuracy_rows:
        with (RESULTS / "accuracy.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(accuracy_rows[0].keys()))
            writer.writeheader()
            writer.writerows(accuracy_rows)

    cold_rows = measure_cold_start(list(cfg["runtimes"]))
    with (RESULTS / "coldstart.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cold_rows)

    train_metrics_path = ROOT / "artifacts" / "train_metrics.json"
    train_metrics = read_json(train_metrics_path) if train_metrics_path.exists() else {}
    int8_meta_path = ROOT / "artifacts" / "model.int8.meta.json"
    int8_meta = read_json(int8_meta_path) if int8_meta_path.exists() else {}

    write_json(
        RESULTS / "run_meta.json",
        {
            "config_hash": config_hash("model", "bench"),
            "seed": seed,
            "repeats": repeats,
            "calls_per_repeat": calls,
            "warmup_calls": warmup,
            "bootstrap_resamples": cfg["bootstrap_resamples"],
            "ci_alpha": cfg["ci_alpha"],
            "cells": len(cells),
            "calls_measured": len(all_samples),
            "wall_seconds": round(time.perf_counter() - started_all, 2),
            "cpu": _cpu_description(),
            "platform": platform.platform(),
            "versions": _versions(),
            "train_metrics": {k: v for k, v in train_metrics.items() if k != "history"},
            "int8_quantisation": int8_meta,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    if not quiet:
        print(f"\nwrote {raw_path.relative_to(ROOT)} ({len(all_samples)} measured calls)")
        print(f"wrote {(RESULTS / 'run_meta.json').relative_to(ROOT)}")
    return raw_path


def main() -> int:
    cfg = load_config("bench")
    ensure_dirs()
    set_seed(int(cfg["seed"]))

    # Environment overrides so CI can run a cheap smoke matrix and a local run can
    # go deeper, without editing the config that a report is pinned to.
    if os.environ.get("MLSERVE_BENCH_CALLS"):
        cfg["calls_per_repeat"] = int(os.environ["MLSERVE_BENCH_CALLS"])
    if os.environ.get("MLSERVE_BENCH_REPEATS"):
        cfg["repeats"] = int(os.environ["MLSERVE_BENCH_REPEATS"])
    if os.environ.get("MLSERVE_BENCH_MODES"):
        cfg["modes"] = os.environ["MLSERVE_BENCH_MODES"].split(",")

    cells = build_matrix(cfg)
    total_calls = len(cells) * int(cfg["calls_per_repeat"]) * int(cfg["repeats"])
    print(f"mlserve-ops benchmark  seed={cfg['seed']}")
    print(
        f"  matrix      {len(cells)} cells "
        f"({len(cfg['modes'])} modes x {len(cfg['runtimes'])} runtimes x "
        f"{len(cfg['batch_sizes'])} batches x {len(cfg['concurrency'])} concurrency)"
    )
    print(
        f"  calls       {total_calls} timed calls, {cfg['warmup_calls']} warmup per trial discarded"
    )
    print(
        f"  intervals   {cfg['bootstrap_resamples']} bootstrap resamples at "
        f"{100 * (1 - cfg['ci_alpha']):.0f}%"
    )
    print(f"  cpu         {_cpu_description()}")
    print()

    run(cfg)
    print("\nnext: python tasks.py report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
