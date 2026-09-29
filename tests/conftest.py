"""Shared test fixtures.

The artifact fixture trains and exports once per session if the artifacts are
missing, so ``pytest`` works from a clean clone with no prior build step. That
matters more than it looks: a test suite that only passes after a manual
prerequisite is a test suite CI cannot run.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Point persistence at a temp database before anything imports mlserve.db.
_TMP = Path(tempfile.mkdtemp(prefix="mlserve-test-"))
os.environ.setdefault("MLSERVE_DATABASE_URL", f"sqlite:///{(_TMP / 'test.db').as_posix()}")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from mlserve import registry  # noqa: E402
from mlserve.config import ARTIFACTS, RUNTIME_BY_NAME  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def artifacts() -> dict[str, Path]:
    """Guarantee model.pt, model.ts, model.onnx and model.int8.onnx all exist."""
    needed = [ARTIFACTS / "model.pt"] + [ARTIFACTS / spec.file for spec in RUNTIME_BY_NAME.values()]
    if not all(p.exists() for p in needed):
        from mlserve.export import main as export_main
        from mlserve.train import main as train_main

        assert train_main() == 0, "training failed"
        assert export_main() == 0, "export or parity check failed"

    missing = [p.name for p in needed if not p.exists()]
    assert not missing, f"artifacts still missing after build: {missing}"
    return {p.name: p for p in needed}


@pytest.fixture(autouse=True)
def clean_state():
    """Fresh request log and unloaded runtimes around every test."""
    from mlserve import db

    db.truncate()
    registry.clear_cache()
    yield
    db.truncate()
    registry.clear_cache()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from mlserve.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_row() -> list[float]:
    """One real test-set sample, normalised the way the API expects."""
    from mlserve.model import load_digits_split

    split = load_digits_split(seed=1337)
    return [float(v) for v in split.x_test[0].reshape(-1)]
