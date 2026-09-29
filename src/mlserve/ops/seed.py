"""Seed the request log with synthetic traffic.

Run: ``python tasks.py seed [count]``  or  ``python -m mlserve.ops.seed 500``

Why this exists: on a fresh clone there is no database, so the backup drill has
nothing to verify. The verifier correctly refuses to pass in that case — "a restore
that verifies no data is not a restore" — but that makes `tasks.py verify` fail on a
clean checkout, which is a bad first experience and, worse, teaches people to ignore
the failure.

The right fix is not to weaken the verifier. It is to give it something real to
check, which is what this module does: it writes rows into the same table the API
writes to, through the same code path, so the backup and restore exercise the
production schema rather than a test double.

Rows are drawn from the training distribution, so a seeded log does **not** trip the
drift detector. That matters: a seed that looked like drift would make every fresh
install alert on first run.
"""

from __future__ import annotations

import argparse
import sys
import uuid

import numpy as np

from mlserve import db
from mlserve.config import load_config, set_seed
from mlserve.model import load_digits_split

DEFAULT_COUNT = 500


def seed_requests(
    count: int = DEFAULT_COUNT,
    *,
    url: str | None = None,
    runtime: str = "onnx-fp32",
    seed: int | None = None,
) -> int:
    """Insert ``count`` synthetic requests drawn from the training distribution.

    Returns the number of rows actually inserted. Idempotent across calls only in
    the sense that each row gets a fresh uuid — calling it twice doubles the log,
    which is what you want for building up a window.
    """
    cfg = load_config("model")
    actual_seed = int(cfg["seed"] if seed is None else seed)
    set_seed(actual_seed)

    split = load_digits_split(
        seed=int(cfg["seed"]),
        val_size=float(cfg["data"]["val_size"]),
        test_size=float(cfg["data"]["test_size"]),
    )
    features = split.x_train.reshape(len(split.x_train), -1)

    db.init_db(url)
    rng = np.random.default_rng(actual_seed)

    inserted = 0
    for _ in range(count):
        row = features[int(rng.integers(0, len(features)))]
        db.record_request(
            request_id=str(uuid.uuid4()),
            runtime=runtime,
            features=[float(v) for v in row],
            predicted_class=int(rng.integers(0, 10)),
            confidence=float(rng.uniform(0.6, 0.99)),
            latency_ms=float(rng.uniform(0.5, 4.0)),
            batch_size=int(rng.choice([1, 8, 32, 64])),
            url=url,
        )
        inserted += 1

    return inserted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlserve.ops.seed", description=__doc__)
    parser.add_argument(
        "count",
        nargs="?",
        type=int,
        default=DEFAULT_COUNT,
        help=f"rows to insert (default {DEFAULT_COUNT})",
    )
    parser.add_argument("--runtime", default="onnx-fp32", help="runtime label to record")
    parser.add_argument("--seed", type=int, default=None, help="override the model config seed")
    args = parser.parse_args(argv)

    if args.count < 1:
        print("count must be at least 1", file=sys.stderr)
        return 2

    before = db.count_requests()
    inserted = seed_requests(args.count, runtime=args.runtime, seed=args.seed)
    after = db.count_requests()

    print(f"seeded   {inserted} requests (runtime={args.runtime})")
    print(f"log now  {after} rows (was {before})")
    print()
    print("next: python tasks.py backup && python tasks.py restore-verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
