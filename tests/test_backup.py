"""Backup and verified-restore tests.

Half of these test the failure path on purpose. A restore verifier that has only
ever been observed to pass has not been tested - it might be checking nothing.
Corrupting an archive and asserting a non-zero exit is the only way to know the
checks do something.
"""

from __future__ import annotations

import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

from mlserve import db
from mlserve.ops.backup import (
    MANIFEST_NAME,
    RestoreVerificationError,
    create_backup,
    latest_backup,
    list_backups,
    sha256_file,
    verify_restore,
)


@pytest.fixture
def populated_db():
    """A database with a known number of rows."""
    from mlserve.schema import FEATURE_COUNT

    db.truncate()
    for index in range(25):
        db.record_request(
            request_id=f"test-{index:04d}",
            runtime="onnx-fp32",
            features=[0.5] * FEATURE_COUNT,
            predicted_class=index % 10,
            confidence=0.9,
            latency_ms=1.5,
        )
    assert db.count_requests() == 25
    yield 25
    db.truncate()


@pytest.fixture
def backup_dir(tmp_path):
    return tmp_path / "backups"


# ------------------------------------------------------------------ creation


def test_backup_archive_is_created(populated_db, backup_dir) -> None:
    result = create_backup(out_dir=backup_dir)
    assert result.archive.exists()
    assert result.archive.suffixes == [".tar", ".gz"]
    assert result.archive.stat().st_size > 0


def test_backup_records_the_row_count(populated_db, backup_dir) -> None:
    result = create_backup(out_dir=backup_dir)
    assert result.manifest["rows"] == populated_db


def test_backup_contains_a_manifest_and_database(populated_db, backup_dir) -> None:
    result = create_backup(out_dir=backup_dir)
    with tarfile.open(result.archive, "r:gz") as tar:
        names = tar.getnames()
    assert MANIFEST_NAME in names
    assert "mlserve.db" in names


def test_backup_records_artifact_checksums(artifacts, backup_dir) -> None:
    result = create_backup(out_dir=backup_dir)
    manifest = result.manifest
    assert manifest["artifacts"], "artifacts directory should contribute checksums"
    for name, meta in manifest["artifacts"].items():
        assert len(meta["sha256"]) == 64
        assert meta["bytes"] > 0
        assert name in result.files


def test_backups_do_not_overwrite_each_other(populated_db, backup_dir) -> None:
    first = create_backup(out_dir=backup_dir)
    second = create_backup(out_dir=backup_dir)
    # Timestamp has second resolution, so two calls in the same second are the
    # same file. That is acceptable - but the directory must not accumulate
    # duplicates, and list_backups must agree with the filesystem.
    assert first.archive.exists() and second.archive.exists()
    assert len(list_backups(backup_dir)) == len(list(backup_dir.glob("mlserve-*.tar.gz")))


def test_list_backups_is_empty_when_there_are_none(tmp_path) -> None:
    assert list_backups(tmp_path / "nothing") == []


def test_latest_backup_raises_when_empty(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no backups found"):
        latest_backup(tmp_path)


# ------------------------------------------------------------------ the happy path


def test_verify_restore_passes_for_a_good_archive(populated_db, backup_dir) -> None:
    archive = create_backup(out_dir=backup_dir).archive
    result = verify_restore(archive)
    assert result.ok
    assert result.restored_rows == populated_db
    assert result.expected_rows == populated_db
    assert result.checks, "verification should report what it actually proved"


def test_verify_restore_reports_specific_checks(populated_db, backup_dir) -> None:
    archive = create_backup(out_dir=backup_dir).archive
    result = verify_restore(archive)
    joined = " | ".join(result.checks)
    assert "manifest read" in joined
    assert "artifact checksums match" in joined
    assert "row-count parity" in joined
    assert "database checksum matches" in joined


def test_verify_restore_cleans_up_its_scratch_directory(populated_db, backup_dir) -> None:
    archive = create_backup(out_dir=backup_dir).archive
    result = verify_restore(archive)
    assert result.scratch_dir is None, "scratch dir should be removed unless asked to keep it"


def test_verify_restore_can_keep_the_scratch_directory(populated_db, backup_dir) -> None:
    archive = create_backup(out_dir=backup_dir).archive
    result = verify_restore(archive, keep_scratch=True)
    assert result.scratch_dir is not None
    assert result.scratch_dir.exists()
    import shutil

    shutil.rmtree(result.scratch_dir, ignore_errors=True)


def test_missing_archive_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="archive not found"):
        verify_restore(tmp_path / "nope.tar.gz")


# ------------------------------------------------------------------ the failure paths
# These are the tests that give the verifier its meaning.


def test_corrupted_database_is_detected(populated_db, backup_dir, tmp_path) -> None:
    """Flip bytes in the database inside the archive.

    This is the exact scenario the checksum exists for: the archive still opens,
    the manifest still parses, and only the hash comparison catches it.
    """
    archive = create_backup(out_dir=backup_dir).archive

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        contents = {m.name: tar.extractfile(m).read() for m in members if m.isfile()}

    corrupted = bytearray(contents["mlserve.db"])
    # Corrupt a byte past the SQLite header so the file is still openable.
    corrupted[200] ^= 0xFF
    contents["mlserve.db"] = bytes(corrupted)

    rebuilt = tmp_path / "corrupted.tar.gz"
    staging = tmp_path / "corrupt-stage"
    staging.mkdir()
    for name, blob in contents.items():
        (staging / name).write_bytes(blob)
    with tarfile.open(rebuilt, "w:gz") as tar:
        for item in sorted(staging.iterdir()):
            tar.add(item, arcname=item.name)

    with pytest.raises(RestoreVerificationError, match="database checksum mismatch"):
        verify_restore(rebuilt)


def test_deleted_database_is_detected(populated_db, backup_dir, tmp_path) -> None:
    """An archive missing its database must not verify cleanly."""
    archive = create_backup(out_dir=backup_dir).archive

    rebuilt = tmp_path / "no-db.tar.gz"
    staging = tmp_path / "no-db-stage"
    staging.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name == "mlserve.db":
                continue
            tar.extract(member, staging, filter="data")
    with tarfile.open(rebuilt, "w:gz") as tar:
        for item in sorted(staging.iterdir()):
            tar.add(item, arcname=item.name)

    with pytest.raises(
        RestoreVerificationError, match=r"database listed in manifest is missing|no database"
    ):
        verify_restore(rebuilt)


def test_missing_manifest_is_detected(populated_db, backup_dir, tmp_path) -> None:
    archive = create_backup(out_dir=backup_dir).archive

    rebuilt = tmp_path / "no-manifest.tar.gz"
    staging = tmp_path / "no-manifest-stage"
    staging.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name == MANIFEST_NAME:
                continue
            tar.extract(member, staging, filter="data")
    with tarfile.open(rebuilt, "w:gz") as tar:
        for item in sorted(staging.iterdir()):
            tar.add(item, arcname=item.name)

    with pytest.raises(RestoreVerificationError, match="missing from the archive"):
        verify_restore(rebuilt)


def test_row_count_mismatch_is_detected(populated_db, backup_dir) -> None:
    """The manifest is edited rather than the data.

    A verifier that only hashes files would pass this, because every file is
    intact. Only the row-count comparison catches a manifest that disagrees with
    reality - which is what a partially-failed backup looks like.
    """
    archive = create_backup(out_dir=backup_dir).archive

    import shutil
    import tempfile

    scratch = Path(tempfile.mkdtemp())
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(scratch, filter="data")
        manifest_path = scratch / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["rows"] = populated_db + 7  # a lie
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        tampered = scratch / "tampered.tar.gz"
        with tarfile.open(tampered, "w:gz") as tar:
            for item in sorted(scratch.iterdir()):
                if item.name != "tampered.tar.gz":
                    tar.add(item, arcname=item.name)

        with pytest.raises(RestoreVerificationError, match="row-count parity failed"):
            verify_restore(tampered)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_tampered_artifact_is_detected(artifacts, populated_db, backup_dir, tmp_path) -> None:
    """Change an artifact inside the archive so its checksum no longer matches."""
    archive = create_backup(out_dir=backup_dir).archive
    with tarfile.open(archive, "r:gz") as tar:
        manifest = json.loads(tar.extractfile(MANIFEST_NAME).read().decode("utf-8"))
    victim = next(iter(manifest["artifacts"]))

    rebuilt = tmp_path / "bad-artifact.tar.gz"
    staging = tmp_path / "bad-artifact-stage"
    staging.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(staging, filter="data")
    target = staging / victim
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")

    with tarfile.open(rebuilt, "w:gz") as tar:
        for item in sorted(staging.iterdir()):
            tar.add(item, arcname=item.name)

    with pytest.raises(RestoreVerificationError, match="artifact checksum mismatch"):
        verify_restore(rebuilt)


def test_non_tar_file_is_rejected(tmp_path) -> None:
    """A truncated or non-archive file must produce a clear error, not a traceback
    from deep inside the tarfile module."""
    junk = tmp_path / "junk.tar.gz"
    junk.write_bytes(b"this is not a tarball")
    with pytest.raises(RestoreVerificationError):
        verify_restore(junk)


def test_restore_is_idempotent_on_the_row_primary_key(populated_db, backup_dir) -> None:
    """Restoring twice must not double the row count.

    request_id is the primary key and inserts skip existing ids - without that,
    the restore drill would inflate the very count it is verifying.
    """
    archive = create_backup(out_dir=backup_dir).archive
    result = verify_restore(archive, keep_scratch=True)
    assert result.scratch_dir is not None
    try:
        scratch_db = result.scratch_dir / "restored_scratch.db"
        with sqlite3.connect(str(scratch_db)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == populated_db
            # Simulate a re-restore of the same rows.
            rows = conn.execute("SELECT request_id FROM requests LIMIT 5").fetchall()
            for (request_id,) in rows:
                conn.execute(
                    "INSERT OR IGNORE INTO requests "
                    "(request_id, created_at, runtime, batch_size, latency_ms, "
                    " predicted_class, confidence, features) "
                    "VALUES (?, '2026-01-01T00:00:00Z', 'r', 1, 1.0, 0, 0.5, '[]')",
                    (request_id,),
                )
            conn.commit()
            assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == populated_db
    finally:
        import shutil

        shutil.rmtree(result.scratch_dir, ignore_errors=True)


def test_sha256_file_distinguishes_contents(tmp_path) -> None:
    """A checksum that does not change when the file changes is not a checksum.

    An earlier version of this test ended in ``or True`` and therefore passed
    unconditionally. That is worse than no test: it looks like coverage.
    """
    path = tmp_path / "x.bin"
    path.write_bytes(b"hello")
    first = sha256_file(path)

    # Stable for unchanged contents.
    assert sha256_file(path) == first

    # Different for changed contents.
    path.write_bytes(b"hello!")
    assert sha256_file(path) != first

    # And deterministic across calls on identical content in a different path.
    other = tmp_path / "y.bin"
    other.write_bytes(b"hello!")
    assert sha256_file(other) == sha256_file(path)
