"""Alert-sink tests.

Small, but this is the component that makes the alerting claim checkable. Alert
routing is otherwise unverifiable without paying for a notification service: you
read a config file and assume it works. This sink means an alert can be made to
fire and watched arriving.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from mlserve.alert_sink import Handler, format_alerts

#: A real Alertmanager webhook body, as Alertmanager actually sends it.
FIRING = {
    "version": "4",
    "status": "firing",
    "receiver": "local-webhook",
    "groupLabels": {"alertname": "MLServeP99LatencyBreach", "component": "api"},
    "alerts": [
        {
            "status": "firing",
            "labels": {
                "alertname": "MLServeP99LatencyBreach",
                "severity": "warning",
                "component": "api",
                "runtime": "onnx-fp32",
            },
            "annotations": {
                "summary": "p99 latency above 50 ms for 5 minutes (onnx-fp32)",
                "description": "Check whether the batch size mix changed before\nassuming a code regression.",
            },
            "startsAt": "2026-09-29T01:00:00Z",
        }
    ],
}

RESOLVED = {
    "version": "4",
    "status": "resolved",
    "alerts": [
        {
            "status": "resolved",
            "labels": {
                "alertname": "MLServeModelNotLoaded",
                "severity": "critical",
                "component": "api",
            },
            "annotations": {"summary": "model loaded again"},
        }
    ],
}


# ------------------------------------------------------------------ rendering


def test_firing_alert_renders_the_labels_that_matter() -> None:
    lines = format_alerts(FIRING)
    joined = "\n".join(lines)
    assert "FIRING" in joined
    assert "MLServeP99LatencyBreach" in joined
    assert "runtime=onnx-fp32" in joined
    assert "p99 latency above 50 ms" in joined


def test_description_is_rendered_without_newlines() -> None:
    """Alertmanager descriptions are multi-line YAML block scalars. Printed raw they
    break the log formatting; collapsed they stay readable."""
    lines = format_alerts(FIRING)
    detail = next(line for line in lines if "detail:" in line)
    assert "\n" not in detail
    assert "Check whether the batch size mix changed before assuming" in detail


def test_resolved_alert_is_labelled_resolved() -> None:
    joined = "\n".join(format_alerts(RESOLVED))
    assert "RESOLVED" in joined
    assert "MLServeModelNotLoaded" in joined


def test_empty_alert_list_does_not_crash() -> None:
    """Alertmanager sends a body with an empty alerts list on some group changes."""
    lines = format_alerts({"status": "firing", "alerts": []})
    assert len(lines) == 1
    assert "no alerts" in lines[0]


def test_missing_annotations_do_not_crash() -> None:
    """A rule with no annotations is misconfigured, but the sink must not 500 —
    returning an error to Alertmanager makes it retry and eventually give up."""
    lines = format_alerts(
        {"status": "firing", "alerts": [{"labels": {"alertname": "X"}, "annotations": {}}]}
    )
    assert any("(none)" in line for line in lines)


def test_missing_labels_do_not_crash() -> None:
    lines = format_alerts({"status": "firing", "alerts": [{}]})
    assert any("?" in line for line in lines)


# ------------------------------------------------------------------ over http


@pytest.fixture
def sink():
    """A real server on an ephemeral port, so the handler is exercised for real."""
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def post(url: str, payload) -> tuple[int, str]:
    body = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_alertmanager_webhook_is_accepted(sink) -> None:
    status, body = post(f"{sink}/alerts", FIRING)
    assert status == 200
    assert json.loads(body) == {"ok": True}


def test_health_endpoint_answers(sink) -> None:
    """Compose gates the stack on this, so it must not do any real work."""
    with urllib.request.urlopen(f"{sink}/healthz", timeout=5) as response:  # noqa: S310
        assert response.status == 200
        assert response.read() == b"ok"


def test_non_json_body_returns_400_not_500(sink) -> None:
    """A 5xx makes Alertmanager retry with backoff and eventually drop the alert.
    A malformed body is the sender's problem and should be reported as such."""
    status, _ = post(f"{sink}/alerts", b"this is not json")
    assert status == 400


def test_unknown_path_returns_404(sink) -> None:
    status, _ = post(f"{sink}/webhook", FIRING)
    assert status == 404


def test_error_response_drains_the_request_body(sink) -> None:
    """A 404 must not abort the connection.

    The handler first responded 404 without reading the body, so the client had
    unread bytes in flight when the server closed. Windows turns that into
    `ConnectionAbortedError: [WinError 10053]`, which appeared as a flaky test - the
    flakiness was the symptom, an undrained body was the cause.

    Sends a large body so the unread bytes certainly exceed the socket buffer, which
    is the difference between "usually works" and "always works".
    """
    big = {"status": "firing", "padding": "x" * 200_000, "alerts": []}
    for path in ("/webhook", "/alerts"):
        status, _ = post(f"{sink}{path}", big)
        assert status in {200, 404}, f"{path} returned {status}"


def test_empty_body_is_tolerated(sink) -> None:
    status, _ = post(f"{sink}/alerts", b"")
    assert status == 200
