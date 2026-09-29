"""Minimal webhook receiver for Alertmanager.

Run as a container: ``python -m mlserve.alert_sink``

Lives in the package rather than under ``monitoring/`` for one concrete reason:
``monitoring/`` is bind-mounted as configuration and excluded from the Docker build
context, so a script there would not exist inside the image the service runs from.

Why it exists at all: the monitoring stack has to run at zero cost, so there is no
Slack, PagerDuty or email account behind it. This process is the receiver - it
accepts Alertmanager webhooks and writes them to stdout, so
``docker compose logs alert-sink`` shows the alerts that actually fired.

That matters more than it sounds. Without a receiver, verifying the alerting path
means reading a config file and assuming it works. With one, an alert can be made
to fire and watched arriving - the difference between having alerting and having a
YAML file that describes alerting.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("ALERT_SINK_PORT", "8080"))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - http.server's naming
        # Read the body BEFORE deciding anything, including before the 404 path.
        #
        # Not draining the body first is a real defect, not a style preference: the
        # client has already sent Content-Length bytes, and a server that responds and
        # closes without reading them makes the OS reset the connection. On Windows
        # that surfaced as `ConnectionAbortedError: [WinError 10053]` in the test
        # suite - a flaky test caused by a bug in the handler.
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"

        if self.path not in ("/alerts", "/"):
            self.send_error(404, "not found")
            return

        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            print(f"[{_stamp()}] received a non-JSON body ({len(raw)} bytes)", flush=True)
            self.send_response(400)
            self.end_headers()
            return

        for line in format_alerts(payload):
            print(line, flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_GET(self) -> None:  # noqa: N802
        """Health endpoint so compose can gate the stack on it."""
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"alert sink is running; POST alerts to /alerts\n")

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Silence the default per-request access log; we print alerts ourselves."""
        return


def _stamp() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


def format_alerts(payload: dict) -> list[str]:
    """Render an Alertmanager webhook body as lines to log.

    Split out from the HTTP handler so it can be tested without a socket, which is
    the only part of this module with logic worth testing.
    """
    status = payload.get("status", "unknown")
    alerts = payload.get("alerts") or []
    if not alerts:
        return [f"[{_stamp()}] {status} with no alerts"]

    lines: list[str] = []
    for alert in alerts:
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        lines.append(
            f"[{_stamp()}] {status.upper():8} "
            f"{labels.get('severity', '?'):8} "
            f"{labels.get('alertname', '?')} "
            f"component={labels.get('component', '-')} "
            f"runtime={labels.get('runtime', '-')}"
        )
        lines.append(f"           summary: {annotations.get('summary', '(none)')}")
        if annotations.get("description"):
            lines.append(f"           detail:  {' '.join(annotations['description'].split())}")
    return lines


def main() -> int:
    server = HTTPServer(("0.0.0.0", PORT), Handler)  # noqa: S104 - container contract
    print(f"alert sink listening on :{PORT} (POST /alerts)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
