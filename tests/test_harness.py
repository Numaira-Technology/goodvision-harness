from __future__ import annotations

import hashlib
import hmac
import json
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import cv2
import httpx
import jsonschema
import numpy as np
import pytest
from fastapi.testclient import TestClient

from goodvision_harness.config import load_config
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.delivery import WebhookDelivery
from goodvision_harness.grounding import (
    DryRunGroundingClient,
    parse_grounded_classification,
)
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.result_bus import FrameBus, FramePacket, ResultBus
from goodvision_harness.service import (
    HarnessRuntime,
    _mjpeg_part,
    _render_analysis_jpeg,
    create_app,
)
from goodvision_harness.sliding_window import FrameWindow, SampledFrame

ROOT = Path(__file__).resolve().parents[1]


def _analysis_result(*, detected: bool = True) -> AnalysisResult:
    objects = (
        (
            GroundedObject(
                object_id="object-1",
                label="person",
                confidence=0.91,
                bbox=NormalizedBBox(0.1, 0.2, 0.9, 0.8),
            ),
        )
        if detected
        else ()
    )
    return AnalysisResult(
        message_id="message-1",
        source_id="external-camera",
        stream_session_id="session-1",
        sequence=1,
        emitted_at="2026-08-25T10:00:01.000Z",
        frame=FrameReference(
            source_frame_index=42,
            timestamp_seconds=1.25,
            observed_at="2026-08-25T10:00:00.000Z",
            width=1920,
            height=1080,
        ),
        judgments=(
            EventJudgment(
                event_label="smoking",
                detected=detected,
                confidence=0.94,
                description="Visible event." if detected else "Event is not visible.",
                object_ids=("object-1",) if detected else (),
            ),
        ),
        objects=objects,
        model_name="test-model",
        model_latency_seconds=0.25,
    )


def _jpeg(width: int = 200, height: int = 100) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_qwen_bbox_is_normalized_and_validated() -> None:
    classification = parse_grounded_classification(
        json.dumps(
            {
                "detected": True,
                "confidence": 0.93,
                "description": "A visible event.",
                "objects": [
                    {
                        "label": "person",
                        "confidence": 0.9,
                        "bbox_2d": [100, 200, 900, 800],
                    }
                ],
            }
        )
    )
    assert classification.objects[0].bbox == NormalizedBBox(0.1, 0.2, 0.9, 0.8)

    negative = parse_grounded_classification(
        '{"detected":false,"confidence":0.8,"description":"clear"}'
    )
    assert negative.objects == ()

    with pytest.raises(ValueError, match="at least one grounded object"):
        parse_grounded_classification(
            '{"detected":true,"confidence":0.8,"description":"positive"}'
        )
    with pytest.raises(ValueError, match=r"0\.\.1000"):
        parse_grounded_classification(
            '{"detected":true,"confidence":0.8,"description":"x",'
            '"objects":[{"label":"person","bbox_2d":[0,0,1001,500]}]}'
        )
    with pytest.raises(ValueError, match="x2>x1"):
        NormalizedBBox(0.8, 0.1, 0.2, 0.9)


def test_analysis_result_matches_versioned_json_schema() -> None:
    payload = _analysis_result().as_dict()
    schema = json.loads(
        (ROOT / "schemas" / "analysis-result-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)
    invalid = json.loads(json.dumps(payload))
    invalid["judgments"][0]["object_ids"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)
    assert payload["objects"][0]["bbox"]["coordinate_system"] == (
        "source_normalized_xyxy"
    )
    assert payload["frame"]["width"] == 1920


def test_pipeline_emits_grounded_result_with_source_frame_mapping() -> None:
    config = load_config(ROOT / "config.yaml")
    client = DryRunGroundingClient(config.inference)
    emitted: list[AnalysisResult] = []
    emitted_frames: list[tuple[AnalysisResult, bytes]] = []
    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(
            on_analysis_frame=lambda result, jpeg: emitted_frames.append(
                (result, jpeg)
            ),
            on_analysis_result=emitted.append,
        ),
        source_id="camera-a",
    )
    presentation_jpeg = _jpeg(width=320, height=180)
    frame = SampledFrame(
        sample_index=0,
        source_frame_index=77,
        timestamp_seconds=2.5,
        jpeg_bytes=_jpeg(),
        width=200,
        height=100,
        source_width=1920,
        source_height=1080,
        observed_at="2026-08-25T10:00:00.000Z",
        presentation_jpeg_bytes=presentation_jpeg,
    )
    window = FrameWindow(index=1, frames=(frame,))

    future: Future[Any] = Future()
    future.set_result(client.classify_window(window))
    session._on_window_done(future, window)

    assert len(emitted) == 1
    payload = emitted[0].as_dict()
    assert payload["source_id"] == "camera-a"
    assert payload["frame"]["source_frame_index"] == 77
    assert payload["frame"]["width"] == 1920
    assert payload["judgments"][0]["object_ids"] == ["object-1"]
    assert emitted_frames == [(emitted[0], presentation_jpeg)]


def test_pending_future_remains_until_delivery_callbacks_finish() -> None:
    config = load_config(ROOT / "config.yaml")
    client = DryRunGroundingClient(config.inference)
    callback_entered = threading.Event()
    release_callback = threading.Event()

    def block_result(_result: AnalysisResult) -> None:
        callback_entered.set()
        assert release_callback.wait(2.0)

    session = StreamingSession.for_camera(
        config,
        client,
        SessionCallbacks(on_analysis_result=block_result),
    )
    frame = SampledFrame(
        sample_index=0,
        source_frame_index=1,
        timestamp_seconds=0.5,
        jpeg_bytes=_jpeg(),
        width=200,
        height=100,
    )
    window = FrameWindow(index=1, frames=(frame,))
    future: Future[Any] = Future()
    future.set_result(client.classify_window(window))
    with session._pending_lock:
        session._pending.add(future)

    worker = threading.Thread(target=session._on_window_done, args=(future, window))
    worker.start()
    try:
        assert callback_entered.wait(1.0)
        assert session._has_pending()
    finally:
        release_callback.set()
        worker.join(2.0)
    assert not worker.is_alive()
    assert not session._has_pending()


def test_overlay_burns_normalized_box_without_mutating_contract() -> None:
    result = _analysis_result()
    before = result.as_dict()
    rendered = _render_analysis_jpeg(_jpeg(), result, quality=85)
    decoded = cv2.imdecode(np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    # The 0.1 x-coordinate maps to x=20 on a 200-pixel frame. Sample the
    # vertical box edge well below the text overlay and compare its red channel
    # with an untouched interior region.
    edge = decoded[60:75, 18:23].mean(axis=(0, 1))
    interior = decoded[60:75, 90:110].mean(axis=(0, 1))
    assert edge[2] > edge[1] + 40
    assert edge.mean() > interior.mean() + 20
    assert result.as_dict() == before


def test_result_and_frame_buses_are_bounded_and_resumable() -> None:
    results = ResultBus(history_size=2)
    results.publish({"message_id": "one"})
    results.publish({"message_id": "two"})
    results.publish({"message_id": "three"})
    replay = results.records_after(0)
    assert [item.payload["message_id"] for item in replay] == ["two", "three"]
    assert results.records_after(replay[-1].cursor, timeout_seconds=0.01) == ()

    frames = FrameBus()
    packet = FramePacket(
        jpeg=_jpeg(),
        media_timestamp=1.5,
        frame_sequence=9,
        observed_at="2026-08-25T10:00:00.000Z",
        analysis_message_id="message-9",
    )
    version = frames.publish(packet)
    assert frames.wait_after(0) == (version, packet)
    part = _mjpeg_part(packet)
    assert b"Content-Type: image/jpeg" in part
    assert b"X-Analysis-Message-Id: message-9" in part
    assert packet.jpeg in part


class _StubRuntime:
    def __init__(self) -> None:
        self.results = ResultBus()
        self.frames = FrameBus()
        self.preview_frames = FrameBus()
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True
        self.results.publish(_analysis_result().as_dict())
        self.frames.publish(
            FramePacket(
                jpeg=_jpeg(),
                media_timestamp=1.0,
                frame_sequence=3,
                observed_at="2026-08-25T10:00:00.000Z",
                analysis_message_id="message-1",
            )
        )
        self.preview_frames.publish(
            FramePacket(
                jpeg=_jpeg(),
                media_timestamp=1.1,
                frame_sequence=4,
                observed_at="2026-08-25T10:00:00.100Z",
            )
        )

    def stop(self) -> None:
        self.stopped = True

    def readiness(self) -> tuple[bool, str]:
        return True, "ready"

    def status(self) -> dict[str, Any]:
        return {"pipeline_state": "running", "ready": True}


def test_http_delivery_surface_and_degraded_liveness() -> None:
    stub = _StubRuntime()
    app = create_app(lambda: stub)  # type: ignore[arg-type]
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").status_code == 200
        latest = client.get("/v1/results/latest")
        assert latest.status_code == 200
        assert latest.json()["message_id"] == "message-1"
        assert latest.headers["X-Result-Stream"] == stub.results.epoch
        frame = client.get("/v1/video/latest.jpg")
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/jpeg"
        assert frame.headers["X-Analysis-Message-Id"] == "message-1"
        live_frame = client.get("/v1/video/live/latest.jpg")
        assert live_frame.status_code == 200
        assert "X-Analysis-Message-Id" not in live_frame.headers
        paths = {route.path for route in app.routes}
        assert "/v1/results/events" in paths
        assert "/v1/video/stream.mjpeg" in paths
        assert "/v1/video/live/stream.mjpeg" in paths
    assert stub.started and stub.stopped

    degraded = HarnessRuntime(None, config_error="bad config")
    degraded_app = create_app(lambda: degraded)
    with TestClient(degraded_app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.get("/v1/status").json()["pipeline_state"] == (
            "configuration_error"
        )


def test_readiness_requires_video_and_successful_analysis() -> None:
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))

    class _RunningSession:
        running = True

    runtime._session = _RunningSession()  # type: ignore[assignment]
    runtime._update_state(pipeline_state="running")
    runtime._on_display_frame(_jpeg(), 1.0, 1)
    assert runtime.readiness() == (False, "waiting_for_first_analysis")

    result = _analysis_result()
    runtime._on_analysis_frame(result, _jpeg())
    runtime._on_analysis_result(result)
    assert runtime.readiness() == (True, "ready")
    _, packet = runtime.frames.latest() or (0, None)
    assert packet is not None
    assert packet.analysis_message_id == result.message_id
    assert packet.frame_sequence == result.frame.source_frame_index


def test_webhook_uses_stable_idempotency_and_hmac_signature() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    delivery = WebhookDelivery(
        "http://receiver.test/events",
        secret="shared-secret",
        max_retries=0,
        transport=httpx.MockTransport(handler),
    )
    delivery.start()
    payload = _analysis_result().as_dict()
    assert delivery.submit(payload)
    assert delivery.flush(timeout_seconds=2.0)
    delivery.close()

    assert len(requests) == 1
    request = requests[0]
    assert request.headers["Idempotency-Key"] == payload["message_id"]
    expected = hmac.new(
        b"shared-secret",
        request.content,
        hashlib.sha256,
    ).hexdigest()
    assert request.headers["X-Harness-Signature-256"] == f"sha256={expected}"
    assert delivery.stats()["delivered"] == 1


def test_webhook_close_discards_backlog_and_worker_can_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery = WebhookDelivery(
        "http://receiver.test/events",
        queue_size=1,
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked_delivery(_client: httpx.Client, _item: object) -> None:
        entered.set()
        assert release.wait(2.0)

    monkeypatch.setattr(delivery, "_deliver", blocked_delivery)
    delivery.start()
    payload = _analysis_result().as_dict()
    assert delivery.submit(payload)
    assert entered.wait(1.0)
    payload["message_id"] = "message-2"
    assert delivery.submit(payload)
    try:
        delivery.close(timeout_seconds=0.01)
        assert delivery.stats()["dropped"] == 1
    finally:
        release.set()
        if delivery._thread is not None:
            delivery._thread.join(2.0)
    assert delivery._thread is not None
    assert not delivery._thread.is_alive()


def test_invalid_dry_run_env_degrades_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_DRY_RUN", "not-a-boolean")
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))
    runtime.start()
    status = runtime.status()
    assert status["pipeline_state"] == "configuration_error"
    assert "HARNESS_DRY_RUN" in status["last_error"]
