#!/usr/bin/env python3
"""Task runner for mlserve-ops. Standard library only, so it works on Windows
without ``make`` installed.

Usage:
    python tasks.py <target> [target ...]
    python tasks.py            # lists every target

The Makefile in this repo is a thin shim over this file, so Linux and CI use
``make <target>`` and both paths run identical commands.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def _env() -> dict[str, str]:
    """Environment for every child process.

    PYTHONPATH points at ``src`` so the package is importable without an
    editable install, which keeps ``git clone && python tasks.py test`` working
    on a fresh machine.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC}{os.pathsep}{existing}" if existing else str(SRC)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Deterministic-ish numeric stack; torch threads fight for cores under bench.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run(*cmd: str, check: bool = True) -> int:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=_env(), check=False)
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {printable}")
    return proc.returncode


def py(*args: str, check: bool = True) -> int:
    return run(sys.executable, *args, check=check)


def mod(module: str, *args: str, check: bool = True) -> int:
    return py("-m", module, *args, check=check)


def docker(*args: str, check: bool = True) -> int:
    if shutil.which("docker") is None:
        raise SystemExit("docker not found on PATH")
    return run("docker", *args, check=check)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"removed {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- targets


def t_install() -> None:
    """Install runtime + dev dependencies."""
    py("-m", "pip", "install", "-e", ".[dev,postgres]")


def t_lint() -> None:
    """Ruff lint + format check."""
    mod("ruff", "check", "src", "tests", "tasks.py")
    mod("ruff", "format", "--check", "src", "tests", "tasks.py")


def t_format() -> None:
    """Apply ruff formatting."""
    mod("ruff", "format", "src", "tests", "tasks.py")
    mod("ruff", "check", "--fix", "src", "tests", "tasks.py")


def t_train() -> None:
    """Train the model and write artifacts/model.pt + train_metrics.json."""
    mod("mlserve.train")


def t_export() -> None:
    """Export the trained model to TorchScript, ONNX fp32 and ONNX int8."""
    mod("mlserve.export")


def t_reference() -> None:
    """Build the drift reference distribution from the training split."""
    mod("mlserve.drift.reference")


def t_serve() -> None:
    """Run the API with autoreload for local development."""
    mod("uvicorn", "mlserve.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload")


def t_test() -> None:
    """Unit tests, excluding anything that needs docker."""
    mod("pytest", "tests", "-m", "not docker", "-q")


def t_test_all() -> None:
    """Every test including docker-dependent ones."""
    mod("pytest", "tests", "-q")


def t_cov() -> None:
    """Unit tests with a coverage report."""
    mod("pytest", "tests", "-m", "not docker", "--cov=mlserve", "--cov-report=term-missing")


def t_bench() -> None:
    """Run the benchmark matrix and write results/raw.csv."""
    mod("mlserve.bench.runner")


def t_report() -> None:
    """Render results/REPORT.md plus the plots from results/raw.csv."""
    mod("mlserve.bench.report")


def t_drift() -> None:
    """Run the drift detection job once.

    Exit code 2 means the drift policy tripped. That is a finding, not a failure,
    so it must not break a `verify` chain that runs drift as one of its steps.
    """
    code = mod("mlserve.drift.detect", check=False)
    if code == 2:
        print("drift alert raised (exit 2) - this is a finding, not a failure")
    elif code != 0:
        raise SystemExit(f"drift job failed with exit code {code}")


def t_pipeline() -> None:
    """train -> export -> reference. The full offline build."""
    t_train()
    t_export()
    t_reference()


def t_verify() -> None:
    """End-to-end verification: lint, tests, pipeline, bench, drift, backup."""
    t_lint()
    t_test()
    t_pipeline()
    t_bench()
    t_report()
    t_drift()
    t_backup()
    t_restore_verify()


def t_verify_all() -> None:
    """`verify` plus the container stack and the smoke test.

    Requires Docker and takes several minutes. Run `verify` first - it is the fast
    signal, and this is the slow one.
    """
    t_verify()
    t_up()
    t_smoke()
    t_down()


def t_backup() -> None:
    """Write a timestamped backup archive with checksums."""
    mod("mlserve.ops.backup_cli", "create")


def t_restore_verify() -> None:
    """Restore the newest backup into a scratch DB and assert parity."""
    mod("mlserve.ops.backup_cli", "verify")


def t_backup_list() -> None:
    """List existing backups."""
    mod("mlserve.ops.backup_cli", "list")


def t_up() -> None:
    """Bring up the full stack (api, prometheus, alertmanager, grafana, postgres)."""
    docker("compose", "up", "-d", "--build")
    print("\nAPI      http://localhost:8000/docs")
    print("Metrics  http://localhost:8000/metrics")
    print("Grafana  http://localhost:3000  (admin / admin by default, change it)")


def t_down() -> None:
    """Tear the stack down, keeping volumes."""
    docker("compose", "down")


def t_nuke() -> None:
    """Tear the stack down and delete volumes. Destroys local run data."""
    docker("compose", "down", "-v", "--remove-orphans")


def t_logs() -> None:
    """Follow logs from the stack."""
    docker("compose", "logs", "-f", "--tail", "100")


def t_smoke() -> None:
    """Assert the running stack actually answers on its health and metric endpoints."""
    mod("tests.smoke")


def t_clean() -> None:
    """Remove generated artifacts, results and caches."""
    for path in ("artifacts", "results", "backups", ".pytest_cache", ".ruff_cache"):
        clean_dir(ROOT / path)
    for cache in ROOT.rglob("__pycache__"):
        if ".git" not in cache.parts:
            shutil.rmtree(cache, ignore_errors=True)
    print("clean")


TARGETS = {
    "install": t_install,
    "lint": t_lint,
    "format": t_format,
    "train": t_train,
    "export": t_export,
    "reference": t_reference,
    "serve": t_serve,
    "test": t_test,
    "test-all": t_test_all,
    "cov": t_cov,
    "bench": t_bench,
    "report": t_report,
    "drift": t_drift,
    "pipeline": t_pipeline,
    "verify": t_verify,
    "verify-all": t_verify_all,
    "backup": t_backup,
    "restore-verify": t_restore_verify,
    "backup-list": t_backup_list,
    "up": t_up,
    "down": t_down,
    "nuke": t_nuke,
    "logs": t_logs,
    "smoke": t_smoke,
    "clean": t_clean,
}


def usage() -> None:
    print(__doc__)
    print("targets:")
    width = max(len(name) for name in TARGETS)
    for name, fn in TARGETS.items():
        summary = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        print(f"  {name.ljust(width)}  {summary}")


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        usage()
        return 0
    for name in argv:
        fn = TARGETS.get(name)
        if fn is None:
            print(f"unknown target: {name}\n", file=sys.stderr)
            usage()
            return 2
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
