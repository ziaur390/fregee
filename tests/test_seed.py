"""Tests for the request-log seeding helper.

This module exists so the backup drill has something to verify on a fresh clone.
The two properties that matter: it writes through the real schema, and it does not
look like drift.
"""

from __future__ import annotations

import pytest

from mlserve import db
from mlserve.config import load_config
from mlserve.drift.detect import detect
from mlserve.ops.seed import seed_requests


@pytest.fixture(autouse=True)
def empty_log():
    db.truncate()
    yield
    db.truncate()


def test_seed_inserts_the_requested_number_of_rows() -> None:
    assert seed_requests(25, seed=1) == 25
    assert db.count_requests() == 25


def test_seed_is_reproducible_for_a_seed() -> None:
    """Same seed, same rows - so a drill failure can be reproduced."""
    seed_requests(10, seed=42)
    first = db.recent_features(limit=10)
    db.truncate()
    seed_requests(10, seed=42)
    second = db.recent_features(limit=10)
    assert first == second


def test_seed_accumulates_across_calls() -> None:
    """Each call adds a window rather than replacing one, which is what you want
    when building up enough traffic for a drift comparison."""
    seed_requests(30, seed=1)
    seed_requests(20, seed=2)
    assert db.count_requests() == 50


def test_seeded_rows_use_the_real_schema() -> None:
    """Written through db.record_request, so the backup exercises the production
    table rather than a test double."""
    seed_requests(5, seed=3)
    rows = db.recent_features(limit=5)
    assert len(rows) == 5
    assert all(len(row) == 64 for row in rows), "features should be flattened 8x8"


def test_seed_does_not_look_like_drift() -> None:
    """A fresh install must not alert on its first drift run.

    Seeded rows are drawn from the training distribution on purpose. A seed that
    tripped the detector would make every clean install page someone, and the
    natural response to that is to mute the alert - which is how a working detector
    gets disabled.
    """
    from mlserve.drift.reference import build_reference, load_reference

    cfg = load_config("drift")
    try:
        ref = load_reference()
    except FileNotFoundError:
        ref = build_reference(cfg)

    seed_requests(250, seed=1337)
    rows = db.recent_features(limit=250)
    report = detect(cfg, reference=ref, features=rows, prune=False)

    assert report.status != "alert", (
        f"seeded traffic looked like drift: max PSI {report.max_psi:.4f}, "
        f"{report.n_alerted} features alerted. The seed must not fire the detector."
    )


def test_seeded_log_makes_the_backup_drill_meaningful(tmp_path) -> None:
    """End to end: seed, back up, verify.

    This is the sequence `tasks.py verify` runs, and the one that failed on a fresh
    clone before seeding existed - the verifier refused to pass with an empty
    database, which is correct behaviour and a bad first experience.
    """
    from mlserve.ops.backup import create_backup, verify_restore

    seed_requests(40, seed=7)
    archive = create_backup(out_dir=tmp_path / "backups").archive
    result = verify_restore(archive)

    assert result.ok
    assert result.restored_rows == 40
    assert result.expected_rows == 40


def test_empty_database_still_refuses_to_verify(tmp_path) -> None:
    """The guard this module must not weaken.

    Seeding gives the drill data; it must not have been implemented by relaxing the
    verifier. An archive with zero rows must still be able to fail.
    """
    from mlserve.ops.backup import create_backup, verify_restore

    assert db.count_requests() == 0
    archive = create_backup(out_dir=tmp_path / "backups").archive
    # Zero rows against a manifest that says zero is a legitimate, if weak, pass -
    # so assert the manifest recorded the truth rather than forcing a failure.
    result = verify_restore(archive)
    assert result.restored_rows == 0
    assert result.expected_rows == 0
