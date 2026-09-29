"""Backup with verified restore.

This is the SysAdmin credibility piece, and the point of it is the word
*verified*.

Writing an archive is easy and tells you nothing. A backup that has never been
restored is a hypothesis, not a control - and the failure is silent until the day
you need it. So ``verify_restore`` does not check that the archive exists. It
extracts it, restores the database into a scratch location, and asserts that the
row count and every checksum match what the manifest recorded. It raises on any
mismatch, and the test suite exercises the failure path by corrupting an archive,
because a verifier that has only ever been seen to pass has not been tested.

Implemented in Python rather than shell. The original plan called for
``scripts/*.sh``, but shell here would be untestable on this development machine
and untested code in a backup path is worse than no code. The Ansible role runs
this on a systemd timer instead; a shell wrapper would add nothing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tarfile
import tempfile
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from mlserve import db
from mlserve.config import ARTIFACTS, ROOT

BACKUP_DIR = ROOT / "backups"
MANIFEST_NAME = "manifest.json"
CHUNK = 1024 * 1024


class RestoreVerificationError(RuntimeError):
    """Raised when a restored backup does not match its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_path_from_url(url: str) -> Path | None:
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        return Path(url.removeprefix("sqlite:///"))
    return None


@dataclass
class BackupResult:
    archive: Path
    manifest: dict
    files: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    archive: Path
    ok: bool
    restored_rows: int
    expected_rows: int
    checks: list[str] = field(default_factory=list)
    scratch_dir: Path | None = None


def create_backup(
    *, db_url: str | None = None, artifacts_dir: Path | None = None, out_dir: Path | None = None
) -> BackupResult:
    """Write a timestamped archive containing the database and an artifact manifest.

    The database is copied with SQLite's own backup API rather than a file copy.
    ``shutil.copy`` on a live SQLite file can capture a torn write - the page
    cache and any in-flight transaction are not part of the file yet, so the copy
    can be internally inconsistent while looking perfectly fine.

    ``db.init_db`` runs first so a fresh checkout yields a valid (empty) database
    instead of none. An empty log means the drill verifies 0 == 0, which is honest
    but weak - run ``python tasks.py seed`` first to give it something real.
    """
    artifacts_dir = artifacts_dir or ARTIFACTS
    out_dir = out_dir or BACKUP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    url = db_url or db.database_url()

    # Create the schema if it does not exist, so a fresh checkout produces a valid
    # database rather than no database at all. Without this the first backup on a
    # clean clone has no database to record, and the restore verifier refuses -
    # correctly, but with a message that reads like a code fault.
    db.init_db(url)

    created = datetime.now(UTC)
    stamp = created.strftime("%Y%m%dT%H%M%SZ")

    with tempfile.TemporaryDirectory(prefix="mlserve-backup-") as tmp:
        staging = Path(tmp)
        manifest: dict = {
            "created_at": created.isoformat(timespec="seconds"),
            "database_url_scheme": url.split(":", 1)[0],
            "database": None,
            "rows": 0,
            "artifacts": {},
        }

        sqlite_path = _sqlite_path_from_url(url)
        if sqlite_path is not None and sqlite_path.exists():
            destination = staging / "mlserve.db"
            # contextlib.closing, not `with sqlite3.connect(...)`. A Connection's
            # own context manager commits or rolls back the transaction and does
            # NOT close the handle. Leaving it open locks the file on Windows and
            # fails the temporary-directory cleanup - and in production it leaks
            # one handle per backup run on a systemd timer, so the failure
            # compounds quietly until the disk fills with locked temp files.
            source_conn = sqlite3.connect(str(sqlite_path))
            try:
                with closing(sqlite3.connect(str(destination))) as target_conn:
                    source_conn.backup(target_conn)
            finally:
                source_conn.close()

            with closing(sqlite3.connect(str(destination))) as conn:
                row_count = (
                    int(conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0])
                    if _table_exists(conn, "requests")
                    else 0
                )
            manifest["database"] = "mlserve.db"
            manifest["rows"] = row_count
            manifest["database_sha256"] = sha256_file(destination)
            manifest["database_bytes"] = destination.stat().st_size
        else:
            # Non-SQLite backend, or a SQLite file that does not exist yet.
            # Recorded explicitly so a verified restore cannot silently pass by
            # checking nothing.
            manifest["database"] = None
            manifest["note"] = (
                "no SQLite database file found; for Postgres use pg_dump in the "
                "Ansible role and extend verify_restore to use pg_restore"
            )

        for artifact in sorted(artifacts_dir.glob("*")):
            if artifact.is_file() and artifact.suffix not in {".db", ".sqlite"}:
                manifest["artifacts"][artifact.name] = {
                    "sha256": sha256_file(artifact),
                    "bytes": artifact.stat().st_size,
                }
                shutil.copy2(artifact, staging / artifact.name)

        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        archive = out_dir / f"mlserve-{stamp}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for item in sorted(staging.iterdir()):
                tar.add(item, arcname=item.name)

    return BackupResult(
        archive=archive,
        manifest=manifest,
        files=sorted(
            [
                *manifest["artifacts"],
                *([manifest["database"]] if manifest["database"] else []),
                MANIFEST_NAME,
            ]
        ),
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def verify_restore(
    archive: Path, *, scratch_root: Path | None = None, keep_scratch: bool = False
) -> VerificationResult:
    """Extract, restore into a scratch location, and assert parity with the manifest.

    Every check appends to ``checks`` so a caller can print what was actually
    proven. Any mismatch raises :class:`RestoreVerificationError` - a verifier
    that returns a boolean gets ignored, and a backup nobody checks is not a
    backup.
    """
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")

    checks: list[str] = []
    scratch = Path(tempfile.mkdtemp(prefix="mlserve-restore-", dir=scratch_root))

    try:
        try:
            with tarfile.open(archive, "r:gz") as tar:
                # filter="data" rejects absolute paths and traversal outside the
                # destination. Extracting an untrusted tarball without it is a
                # path-traversal hole.
                tar.extractall(scratch, filter="data")
        except tarfile.TarError as exc:
            raise RestoreVerificationError(f"archive is not a readable tar.gz: {exc}") from exc

        manifest_path = scratch / MANIFEST_NAME
        if not manifest_path.exists():
            raise RestoreVerificationError(f"{MANIFEST_NAME} missing from the archive")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks.append(
            f"manifest read: created {manifest.get('created_at')}, {len(manifest.get('artifacts', {}))} artifacts"
        )

        # --- artifact checksums
        for name, meta in sorted(manifest.get("artifacts", {}).items()):
            restored = scratch / name
            if not restored.exists():
                raise RestoreVerificationError(f"artifact listed in manifest is missing: {name}")
            actual = sha256_file(restored)
            if actual != meta["sha256"]:
                raise RestoreVerificationError(
                    f"artifact checksum mismatch for {name}: manifest {meta['sha256'][:12]}, restored {actual[:12]}"
                )
        if manifest.get("artifacts"):
            checks.append(f"artifact checksums match: {len(manifest['artifacts'])} files")

        # --- database restore into a scratch database and row-count parity
        expected_rows = int(manifest.get("rows", 0))
        restored_rows = 0
        db_name = manifest.get("database")
        if db_name:
            restored_db = scratch / db_name
            if not restored_db.exists():
                raise RestoreVerificationError(f"database listed in manifest is missing: {db_name}")

            actual_db_hash = sha256_file(restored_db)
            if actual_db_hash != manifest.get("database_sha256"):
                raise RestoreVerificationError(
                    "database checksum mismatch: the archive is corrupted or was modified"
                )
            checks.append("database checksum matches the manifest")

            # Restore into a separate scratch database rather than reusing the
            # extracted file, so this exercises an actual open-read-query path.
            scratch_db = scratch / "restored_scratch.db"
            source = sqlite3.connect(str(restored_db))
            try:
                with closing(sqlite3.connect(str(scratch_db))) as target:
                    source.backup(target)
            finally:
                source.close()

            with closing(sqlite3.connect(str(scratch_db))) as conn:
                if _table_exists(conn, "requests"):
                    restored_rows = int(conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0])
                else:
                    raise RestoreVerificationError(
                        "restored database has no 'requests' table; nothing was actually restorable"
                    )
            if restored_rows != expected_rows:
                raise RestoreVerificationError(
                    f"row-count parity failed: manifest says {expected_rows}, restored database has {restored_rows}"
                )
            checks.append(
                f"row-count parity: {restored_rows} rows restored == {expected_rows} in manifest"
            )
        else:
            # No database in the backup. Say so loudly rather than reporting a
            # clean verification that checked nothing.
            checks.append("WARNING: manifest contains no database - parity was not verified")
            raise RestoreVerificationError(
                "manifest contains no database; a restore that verifies no data is not a restore"
            )

        return VerificationResult(
            archive=archive,
            ok=True,
            restored_rows=restored_rows,
            expected_rows=expected_rows,
            checks=checks,
            scratch_dir=scratch if keep_scratch else None,
        )
    finally:
        if not keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def latest_backup(out_dir: Path | None = None) -> Path:
    out_dir = out_dir or BACKUP_DIR
    archives = sorted(out_dir.glob("mlserve-*.tar.gz"))
    if not archives:
        raise FileNotFoundError(
            f"no backups found in {out_dir}. Run `python tasks.py backup` first."
        )
    return archives[-1]


def list_backups(out_dir: Path | None = None) -> list[dict]:
    out_dir = out_dir or BACKUP_DIR
    rows: list[dict] = []
    for archive in sorted(out_dir.glob("mlserve-*.tar.gz")):
        rows.append({"archive": archive.name, "bytes": archive.stat().st_size})
    return rows
