"""CLI for the backup and restore-verification drill."""

from __future__ import annotations

import argparse
import sys

from mlserve.ops.backup import (
    RestoreVerificationError,
    create_backup,
    latest_backup,
    list_backups,
    verify_restore,
)


def cmd_backup(_args: argparse.Namespace) -> int:
    result = create_backup()
    print(f"backup written  {result.archive}")
    print(f"  rows          {result.manifest['rows']}")
    print(f"  artifacts     {len(result.manifest['artifacts'])}")
    print(f"  database      {result.manifest.get('database') or '(none - see manifest note)'}")
    print(f"  size          {result.archive.stat().st_size / 1024:.1f} KiB")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    archive = args.archive or latest_backup()
    print(f"verifying {archive}")
    try:
        result = verify_restore(archive)
    except (RestoreVerificationError, FileNotFoundError) as exc:
        print(f"\nRESTORE VERIFICATION FAILED\n  {exc}", file=sys.stderr)
        return 1

    for check in result.checks:
        print(f"  [ok] {check}")
    print(f"\nverified: {result.restored_rows} rows restored and matched the manifest")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    rows = list_backups()
    if not rows:
        print("no backups found")
        return 0
    print(f"{'archive':<40} {'KiB':>10}")
    for row in rows:
        print(f"{row['archive']:<40} {row['bytes'] / 1024:>10.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlserve.ops.backup", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("create", help="write a timestamped backup archive").set_defaults(
        func=cmd_backup
    )

    verify = sub.add_parser("verify", help="restore the newest backup and assert parity")
    verify.add_argument("archive", nargs="?", help="specific archive; defaults to the newest")
    verify.set_defaults(func=cmd_verify)

    sub.add_parser("list", help="list existing backups").set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
