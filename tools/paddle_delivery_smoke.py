"""Verify worker code/HTTP using deterministic synthetic models, never Paddle.

Default self-test, inside a disposable worker container with --network none:
    python /tmp/paddle_delivery_smoke.py --isolated

For isolated two-container HTTP testing on an internal Docker network:
    python /tmp/paddle_delivery_smoke.py --isolated --serve --host 0.0.0.0

Use the actual /worker/app.py shipped in the image. Only model inference is
substituted; FastAPI, JPEG parsing, the tracker, async attributes, and HTTP are
the production implementations. No GPU execution or real-model accuracy is
tested. This entry point is a test fixture and must never serve production.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import cv2
import numpy as np
import uvicorn

NOTICE = (
    "SYNTHETIC PROTOCOL TEST ONLY: real worker code/HTTP; "
    "no Paddle import, model download, GPU execution, or real-model inference. "
    "GPU readiness is simulated only for this isolated test fixture."
)
SESSION_HEADER = "X-Goodvision-Stream-Session-Id"
FRAME_HEADER = "X-Goodvision-Frame-Sequence"
TIME_HEADER = "X-Goodvision-Timestamp-Seconds"
TIMEOUT = 5.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def passed(message: str) -> None:
    print(f"PASS  {message}", flush=True)


def require_no_paddle() -> None:
    require(
        not any(
            name == "paddle" or name.startswith("paddle.")
            or name == "paddlex" or name.startswith("paddlex.")
            for name in sys.modules
        ),
        "unexpected Paddle/PaddleX import in the synthetic protocol check",
    )


def load_worker(path: str) -> ModuleType:
    source = Path(path)
    require(source.is_file(), "the image's actual worker app.py is required")
    spec = importlib.util.spec_from_file_location("smoke_worker_app", source)
    require(spec is not None and spec.loader is not None, "cannot load worker app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require_no_paddle()
    return module


class SyntheticModels:
    """Fixed detections and a complete attribute vector, not model predictions."""

    def __init__(self, worker: ModuleType) -> None:
        self.worker = worker
        self.ready = False
        # The --isolated fixture simulates the health contract so the harness's
        # production GPU gate is exercised, never disabled or bypassed.
        self.gpu_ready = False
        self.tracking_gpu_ready = False
        self.attribute_started = threading.Event()
        self.attribute_release = threading.Event()
        self.attribute_release.set()
        self.detect_calls = 0
        self.attribute_calls = 0

    def ensure_detector_loaded(self) -> None:
        self.ready = True
        self.gpu_ready = True
        self.tracking_gpu_ready = True

    def detect(self, image: Any) -> tuple[Any, ...]:
        self.detect_calls += 1
        height, width = image.shape[:2]
        return (
            self.worker.Detection(
                "person", 0.94,
                (0.1 * width, 0.1 * height, 0.45 * width, 0.9 * height),
            ),
            self.worker.Detection(
                "bottle", 0.82,
                (0.65 * width, 0.3 * height, 0.8 * width, 0.85 * height),
            ),
        )

    def attributes(self, crops: Any) -> tuple[dict[str, Any], ...]:
        self.attribute_calls += 1
        self.attribute_started.set()
        require(self.attribute_release.wait(10.0), "synthetic attributes not released")
        scores = [0.1] * 26
        scores[0] = 0.8
        scores[2] = 0.9
        scores[11] = 0.9
        scores[17] = 0.8
        scores[20] = 0.95
        scores[22] = 0.9
        scores[23] = 0.9
        payload = {
            "class_ids": list(range(26)),
            "label_names": list(self.worker._PEDESTRIAN_ATTRIBUTE_LABELS),
            "scores": scores,
        }
        decoded = self.worker._pedestrian_attributes(payload)
        require(bool(decoded), "production attribute decoder rejected fixture")
        decoded["synthetic_protocol_test"] = True
        return tuple(dict(decoded) for _ in crops)


def metadata(sequence: int, session_id: str = "isolated-protocol-session") -> dict:
    return {
        "Content-Type": "image/jpeg",
        SESSION_HEADER: session_id,
        FRAME_HEADER: str(sequence),
        TIME_HEADER: str(sequence / 10),
    }


def make_jpeg() -> bytes:
    # These are colored geometric shapes, not photographs of people.
    pixels = np.zeros((160, 240, 3), dtype=np.uint8)
    cv2.rectangle(pixels, (24, 16), (108, 144), (60, 170, 220), -1)
    cv2.circle(pixels, (176, 85), 18, (170, 70, 40), -1)
    success, encoded = cv2.imencode(".jpg", pixels)
    require(success, "cannot encode synthetic JPEG")
    return encoded.tobytes()


def http(
    base_url: str, path: str, *, body: bytes | None = None,
    headers: dict | None = None, expected: int = 200,
) -> tuple[dict, Any]:
    # Disable proxy environment lookup, including when tested outside Docker.
    opener = build_opener(ProxyHandler({}))
    req = Request(base_url + path, data=body, headers=headers or {})
    try:
        response = opener.open(req, timeout=TIMEOUT)
    except HTTPError as error:
        response = error
    with response:
        content = response.read()
        require(response.status == expected,
                f"{path}: expected HTTP {expected}, got {response.status}")
        require(response.headers.get_content_type() == "application/json",
                f"{path}: unexpected response content type")
        return json.loads(content), response.headers


def check_snapshot(payload: dict, sequence: int, session_id: str) -> dict:
    require(payload.get("stream_session_id") == session_id, "snapshot session")
    require(payload.get("frame_sequence") == sequence, "snapshot frame sequence")
    require(payload.get("timestamp_seconds") == sequence / 10, "snapshot timestamp")
    objects = payload.get("objects", [])
    require(len(objects) == 2, "synthetic person and bottle not returned")
    by_label = {item["label"]: item for item in objects}
    require(set(by_label) == {"person", "bottle"}, "synthetic labels changed")
    require(by_label["person"]["track_id"].startswith("P"), "person track ID")
    require(by_label["bottle"]["track_id"].startswith("O"), "object track ID")
    for item in objects:
        require(0 <= item["confidence"] <= 1, "snapshot confidence")
        bbox = item["bbox"]
        require(0 <= bbox["x1"] < bbox["x2"] <= 1, "normalized horizontal bbox")
        require(0 <= bbox["y1"] < bbox["y2"] <= 1, "normalized vertical bbox")
    return by_label


def self_test(
    worker: ModuleType, models: SyntheticModels, engine: Any,
    server: uvicorn.Server, port: int,
) -> None:
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.run, name="worker-http-smoke", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15.0
        while not server.started:
            require(thread.is_alive(), "worker HTTP server exited before startup")
            require(time.monotonic() < deadline, "worker HTTP startup timed out")
            threading.Event().wait(0.05)
        require(models.ready, "production lifespan did not warm up the detector")
        require(http(base_url, "/health/live")[0] == {"live": True}, "live health")
        require(
            http(base_url, "/health/ready")[0] == {
                "ready": True, "gpu_ready": True,
            },
            "synthetic ready/GPU capability health",
        )
        passed("actual FastAPI lifespan and health routes")
        require(
            http(base_url, "/health/tracking")[0] == {
                "ready": True, "gpu_ready": True,
            },
            "synthetic tracking readiness",
        )

        jpeg = make_jpeg()
        models.attribute_release.clear()
        first, _ = http(base_url, "/v1/labels", body=jpeg, headers=metadata(1))
        first_objects = check_snapshot(first, 1, "isolated-protocol-session")
        require(
            first_objects["person"]["attributes"] == {}, "first attributes not async"
        )
        require(models.attribute_started.wait(TIMEOUT), "attribute task did not start")
        started = time.monotonic()
        second, _ = http(base_url, "/v1/labels", body=jpeg, headers=metadata(2))
        require(
            time.monotonic() - started < 2.0, "attribute task blocked detector HTTP"
        )
        second_objects = check_snapshot(second, 2, "isolated-protocol-session")
        for label in first_objects:
            require(
                first_objects[label]["track_id"] == second_objects[label]["track_id"],
                "tracking ID changed on an identical synthetic detection",
            )
        require(models.attribute_calls == 1, "attribute work queued while busy")
        models.attribute_release.set()
        require(engine._attribute_idle.wait(TIMEOUT), "attribute task did not finish")
        third, _ = http(base_url, "/v1/labels", body=jpeg, headers=metadata(3))
        third_objects = check_snapshot(third, 3, "isolated-protocol-session")
        attrs = third_objects["person"]["attributes"]
        require(attrs.get("estimated_age_range") == "18-60", "age group decoding")
        require(attrs.get("estimated_gender") == "female-presenting", "gender decoding")
        require(attrs.get("synthetic_protocol_test") is True, "synthetic marker")
        passed("JPEG HTTP, stable tracking, async attributes and real 26-score decoder")

        http(base_url, "/v1/labels", body=jpeg,
             headers={"Content-Type": "image/jpeg"}, expected=400)
        wrong_type = metadata(4)
        wrong_type["Content-Type"] = "application/octet-stream"
        http(base_url, "/v1/labels", body=jpeg, headers=wrong_type, expected=415)
        invalid_metadata = metadata(4)
        invalid_metadata[FRAME_HEADER] = "-1"
        http(base_url, "/v1/labels", body=jpeg, headers=invalid_metadata, expected=400)
        invalid_metadata = metadata(4)
        invalid_metadata[TIME_HEADER] = "nan"
        http(base_url, "/v1/labels", body=jpeg, headers=invalid_metadata, expected=400)
        http(
            base_url, "/v1/labels", body=b"not a jpeg",
            headers=metadata(4), expected=400,
        )
        http(base_url, "/v1/labels", body=b"", headers=metadata(4), expected=413)
        http(base_url, "/v1/labels", body=b"x" * (worker.MAX_JPEG_BYTES + 1),
             headers=metadata(4), expected=413)
        require(engine._lock.acquire(blocking=False), "unexpected busy inference lock")
        try:
            _, response_headers = http(
                base_url, "/v1/labels", body=jpeg, headers=metadata(4), expected=429
            )
            require(response_headers.get("Retry-After") == "1", "busy retry header")
        finally:
            engine._lock.release()
        passed("invalid metadata/JPEG 400, media 415, body size 413, busy 429")

        changed, _ = http(base_url, "/v1/labels", body=jpeg,
                          headers=metadata(1, "second-isolated-session"))
        changed_objects = check_snapshot(changed, 1, "second-isolated-session")
        require(changed_objects["person"]["attributes"] == {}, "old attributes leaked")
        require(engine._attribute_idle.wait(TIMEOUT), "new session attributes stalled")
        require_no_paddle()
        passed("new-session state isolation; no Paddle/PaddleX module imported")
    finally:
        models.attribute_release.set()
        server.should_exit = True
        thread.join(timeout=10.0)
        require(not thread.is_alive(), "test HTTP server did not shut down")
    require(engine._closed, "production lifespan did not close the engine")
    passed("production lifespan clean shutdown")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated", action="store_true", required=True,
                        help="Confirm this is a disposable, network-isolated test.")
    parser.add_argument("--worker-app", default="/worker/app.py")
    parser.add_argument("--serve", action="store_true",
                        help="Serve the fake-model fixture for isolated integration.")
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    print(NOTICE, flush=True)
    started = time.monotonic()
    try:
        require(1 <= args.port <= 65535, "port must be between 1 and 65535")
        require(args.serve or args.host == "127.0.0.1", "self-test must use loopback")
        require_no_paddle()
        worker = load_worker(args.worker_app)
        models = SyntheticModels(worker)
        engine = worker.RichLabelEngine(models)
        application = worker.create_app(engine)
        server = uvicorn.Server(uvicorn.Config(
            application, host=args.host, port=args.port,
            log_level="warning", access_log=False,
        ))
        if args.serve:
            print("Serving synthetic test fixture; never use for production.",
                  flush=True)
            server.run()
            require_no_paddle()
        else:
            self_test(worker, models, engine, server, args.port)
            elapsed = time.monotonic() - started
            passed(f"synthetic worker protocol smoke completed in {elapsed:.1f}s")
    except Exception as exc:
        print(f"FAIL  {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(NOTICE, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
