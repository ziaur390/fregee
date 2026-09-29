"""Uvicorn entrypoint for container and systemd use.

Kept separate from :mod:`mlserve.app` so the app module stays importable by
tests without dragging in a server runtime.
"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "mlserve.app:app",
        host=os.environ.get("MLSERVE_HOST", "0.0.0.0"),  # noqa: S104 - container contract
        port=int(os.environ.get("MLSERVE_PORT", "8000")),
        log_level=os.environ.get("MLSERVE_LOG_LEVEL", "info"),
        # Workers stay at 1: the benchmark measures per-runtime overhead, and
        # multiple workers would make latency depend on an unmeasured queue.
        workers=int(os.environ.get("MLSERVE_WORKERS", "1")),
        access_log=os.environ.get("MLSERVE_ACCESS_LOG", "0") == "1",
    )


if __name__ == "__main__":
    main()
