"""Stack smoke test - run against a *running* service, not a TestClient.

    python tasks.py up        # or docker compose up -d --build
    python tasks.py smoke

Exits non-zero on the first failed assertion so it is usable as a CI gate or a
post-deploy check. This is deliberately outside ``tests/test_*.py`` so pytest
does not try to run it without a live stack.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("MLSERVE_BASE_URL", "http://localhost:8000").rstrip("/")
PROM = os.environ.get("MLSERVE_PROM_URL", "http://localhost:9090").rstrip("/")

SAMPLE = [0.5] * 64

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}{f' - {detail}' if detail else ''}")
    if not condition:
        failures.append(label)


def get(path: str, base: str = BASE, timeout: float = 10.0):
    with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as response:  # noqa: S310
        return response.status, response.read()


def post(path: str, payload: dict, timeout: float = 30.0):
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def main() -> int:
    print(f"smoke test against {BASE}\n")

    print("ops endpoints")
    try:
        status, body = get("/healthz")
        check("/healthz 200", status == 200, f"status={status}")
    except Exception as exc:  # noqa: BLE001
        check("/healthz reachable", False, str(exc))
        print("\nservice is not reachable - is the stack up? (python tasks.py up)")
        return 1

    status, body = get("/readyz")
    check("/readyz 200", status == 200, f"status={status}")
    check(
        "/readyz reports no missing runtimes",
        b'"runtimes_missing":[]' in body or b'"runtimes_missing": []' in body,
        body[:120].decode(errors="replace"),
    )

    status, body = get("/models")
    models = json.loads(body)
    check("/models lists 3 runtimes", len(models.get("runtimes", [])) == 3)

    status, body = get("/metrics")
    check("/metrics 200", status == 200)
    for metric in (b"mlserve_requests_total", b"mlserve_request_duration_seconds"):
        check(f"/metrics exposes {metric.decode()}", metric in body)

    print("\ninference")
    status, body = post("/predict", {"features": SAMPLE})
    check("/predict 200", status == 200, f"status={status}")
    if status == 200:
        check("/predict returns a class", 0 <= body["predictions"][0]["predicted_class"] <= 9)
        check("/predict reports latency", body["latency_ms"] > 0)

    status, body = post("/predict/batch", {"features": [SAMPLE] * 8})
    check("/predict/batch 200", status == 200)
    check("/predict/batch returns 8 rows", status == 200 and len(body["predictions"]) == 8)

    for runtime in ("torchscript", "onnx-fp32", "onnx-int8"):
        status, _ = post(f"/predict?runtime={runtime}", {"features": SAMPLE})
        check(f"/predict runtime={runtime}", status == 200, f"status={status}")

    print("\nrejection paths")
    status, _ = post("/predict", {"features": [0.5] * 10})
    check("/predict rejects 10 features (422)", status == 422, f"status={status}")
    status, _ = post("/predict", {"features": [99.0] * 64})
    check("/predict rejects out-of-range (422)", status == 422, f"status={status}")
    status, _ = post("/predict/batch", {"features": [SAMPLE] * 500})
    check("/predict/batch rejects 500 rows (413)", status == 413, f"status={status}")

    print("\npersistence")
    status, body = get("/stats")
    check("/stats 200", status == 200)
    check("/stats shows persisted rows", json.loads(body).get("requests_persisted", 0) > 0)

    print("\nprometheus")
    try:
        status, body = get("/api/v1/targets", base=PROM)
        up = body.count(b'"health":"up"')
        check("prometheus reachable", status == 200)
        check("prometheus has an UP target", up >= 1, f"up targets={up}")
    except Exception as exc:  # noqa: BLE001
        check("prometheus reachable", False, str(exc))

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("all smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
