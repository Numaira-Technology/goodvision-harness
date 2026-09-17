"""Session diagnostics separate missed events from system failures.

No real model, camera or network is used; transports raise the bounded
errors the real client produces.
"""

from __future__ import annotations

import gc
import json
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from goodvision_harness.config import load_config
from goodvision_harness.diagnostics import (
    SessionDiagnostics,
    bind_capture,
    capture_call,
    failure_category,
    note_failure,
)
from goodvision_harness.grounding import (
    AIClubMultiGroundingClient,
    SelectedFeaturesGroundingClient,
)
from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.privacy import contains_private_implementation_detail
from goodvision_harness.service import HarnessRuntime, create_app
from goodvision_harness.sliding_window import FrameWindow, SampledFrame

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("failure", "category"),
    [
        ({"stage": "request", "code": "http_error", "http_status": 401}, "auth"),
        ({"stage": "request", "code": "http_error", "http_status": 403}, "auth"),
        ({"stage": "request", "code": "http_error", "http_status": 429},
         "rate_limited"),
        ({"stage": "request", "code": "http_error", "http_status": 503},
         "server_error"),
        ({"stage": "request", "code": "http_error", "http_status": 404},
         "http_error"),
        ({"stage": "request", "code": "timeout"}, "timeout"),
        ({"stage": "request", "code": "connection"}, "connection"),
        ({"stage": "request", "code": "empty_response"}, "bad_response"),
        ({"stage": "classification", "code": "invalid_response"}, "parse"),
        ({"stage": "input", "code": "invalid_response"}, "input"),
        ({"stage": "merge", "code": "invalid_response"}, "internal"),
        # A branch reports its own error, so its cause survives the branch stage.
        ({"stage": "branch", "code": "timeout"}, "timeout"),
        ({"stage": "branch", "code": "http_error", "http_status": 401}, "auth"),
        ({"stage": "branch", "code": "request_failed"}, "internal"),
        (None, "unknown"),
    ],
)
def test_failure_categories(failure, category):
    assert failure_category(failure) == category


def test_failures_are_only_captured_inside_a_call():
    note_failure(branch="safety", stage="request", code="timeout")
    with capture_call() as capture:
        note_failure(branch="safety", stage="request", code="timeout")
    assert len(capture.failures) == 1


def test_a_branch_on_its_own_thread_records_into_the_same_call():
    def branch() -> None:
        note_failure(
            branch="stranger", stage="request", code="http_error", http_status=401,
        )

    with capture_call() as capture:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(bind_capture(branch)).result()
            # Without the binding the branch has no capture of its own.
            executor.submit(branch).result()
    assert [failure_category(f) for f in capture.failures] == ["auth"]


def _frame(index: float, ts: float, captured: float | None = None) -> SampledFrame:
    return SampledFrame(
        int(index), int(index), ts, b"jpeg", 8, 8, captured_monotonic=captured,
    )


def _single(index: int, ts: float, captured: float | None = None) -> FrameWindow:
    return FrameWindow(index, (_frame(index, ts, captured),))


def test_coverage_gap_rate_and_latency_from_a_fake_clock():
    now = [100.0]
    diag = SessionDiagnostics(clock=lambda: now[0], rate_window_seconds=10.0)
    outcomes = [True, False, False, False, True, True]  # frames 0..5 at 0.4 s
    for index, ok in enumerate(outcomes):
        window = _single(index, index * 0.4, captured=now[0])
        diag.note_submitted("safety", window)
        with capture_call() as capture:
            if not ok:
                note_failure(
                    branch="safety", stage="request", code="http_error",
                    http_status=401,
                )
        diag.note_call("safety", window, capture)
        now[0] += 0.5
        diag.note_completed(
            "safety", window, ok=ok, model_latency_seconds=0.3,
        )
    diag.note_dropped("safety", _single(6, 2.4))
    snap = diag.snapshot()["branches"]["safety"]
    assert snap["succeeded"] == 3
    assert snap["failed"] == 3
    assert snap["dropped"] == 1
    assert snap["failures"]["auth"] == 3
    assert snap["last_failure"]["http_status"] == 401
    # Analyzed at 0.0, 1.6, 2.0: the 1.6 s hole is the longest.
    assert snap["longest_unanalyzed_gap_seconds"] == pytest.approx(1.6)
    # Newest sampled frame (dropped) is 2.4 s; newest analyzed is 2.0 s.
    assert snap["current_unanalyzed_gap_seconds"] == pytest.approx(0.4)
    # Three successes during the first three seconds of the session.
    assert snap["analyzed_per_second"] == pytest.approx(1.0)
    assert snap["latency_seconds"]["capture_to_result"]["p50"] == pytest.approx(0.5)
    assert snap["latency_seconds"]["model"]["count"] == 3
    now[0] += 30.0
    assert diag.snapshot()["branches"]["safety"]["analyzed_per_second"] == 0.0
    inputs = diag.snapshot()["model_inputs"]
    assert [entry["outcome"] for entry in inputs][:2] == ["ok", "ok"]
    assert "jpeg" not in json.dumps(inputs)
    assert diag.model_input_jpeg(inputs[0]["id"]) == b"jpeg"


def _analyze_all(diag, now, count, *, period=0.4, latency=0.8, skip=()):
    for index in range(count):
        timestamp = index * period
        window = _single(index, timestamp, captured=timestamp)
        now[0] = timestamp
        if index in skip:
            diag.note_dropped("safety", window)
            continue
        diag.note_submitted("safety", window)
        with capture_call() as capture:
            pass
        diag.note_call("safety", window, capture)
        now[0] = timestamp + latency
        diag.note_completed("safety", window, ok=True, model_latency_seconds=latency)


@pytest.mark.parametrize(
    ("captured", "category"),
    [(True, "auth"), (False, "internal")],
)
def test_a_raised_call_keeps_the_cause_it_captured(captured, category):
    diag = SessionDiagnostics()
    window = _single(0, 0.0)
    diag.note_submitted("safety", window)
    with capture_call() as capture:
        if captured:
            note_failure(
                branch="safety", stage="request", code="http_error", http_status=401,
            )
    diag.note_call("safety", window, capture)
    diag.note_completed("safety", window, ok=False, internal_error=True)
    failures = diag.snapshot()["branches"]["safety"]["failures"]
    assert failures[category] == 1
    assert sum(failures.values()) == 1


def test_longest_gap_stays_real_after_history_is_evicted():
    now = [0.0]
    diag = SessionDiagnostics(clock=lambda: now[0], history=16)
    _analyze_all(diag, now, 10)
    short = diag.snapshot()["branches"]["safety"]
    assert short["gap_scope"] == "session"
    assert short["longest_unanalyzed_gap_seconds"] == pytest.approx(0.4)
    # Ten minutes at 2.5 FPS with no missing frame: previously the session
    # start was paired with the oldest retained frame (a 395 s false gap).
    now = [0.0]
    diag = SessionDiagnostics(clock=lambda: now[0], history=512)
    _analyze_all(diag, now, 1500)
    long = diag.snapshot()["branches"]["safety"]
    assert long["gap_scope"] == "recent"
    assert long["gap_history_size"] == 512
    assert long["longest_unanalyzed_gap_seconds"] == pytest.approx(0.4)
    # A real hole inside the retained history is still reported.
    now = [0.0]
    diag = SessionDiagnostics(clock=lambda: now[0], history=16)
    _analyze_all(diag, now, 40, skip={33, 34, 35})
    holed = diag.snapshot()["branches"]["safety"]
    assert holed["gap_scope"] == "recent"
    assert holed["longest_unanalyzed_gap_seconds"] == pytest.approx(1.6)
    assert holed["dropped"] == 3
    assert holed["in_flight"] == 0


def _jpeg(width: int, height: int, value: int = 40) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((height, width, 3), value, np.uint8))
    assert ok
    return encoded.tobytes()


class _FailingTransport:
    def __init__(self, error: Exception | None = None, text: str | None = None):
        self.error = error
        self.text = text
        self.calls = 0

    def chat_completion(self, **_kwargs: Any) -> dict:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {}

    def assistant_text(self, _response: dict) -> str:
        return self.text or ""


def _run_camera_session(transport, frames: int = 4):
    config = load_config(ROOT / "config.yaml")
    safety = AIClubMultiGroundingClient(
        config.inference, transport, event_labels=("smoking",),
    )
    client = SelectedFeaturesGroundingClient(
        safety, None, safety_event_labels=("smoking",), include_stranger=False,
        safety_max_image_side=320, safety_jpeg_quality=75,
    )
    ended = threading.Event()
    config = replace(
        config, sliding_window=replace(config.sliding_window, size_frames=1),
    )
    session = StreamingSession.for_camera(
        config, client, SessionCallbacks(on_end=lambda *_: ended.set()),
    )
    session.start()
    for index in range(frames):
        assert session.push_camera_frame(
            _jpeg(640, 360, index * 20), width=640, height=360,
            timestamp_seconds=index * 0.4, source_frame_index=index,
        )
        time.sleep(0.02)
    assert session.camera_source is not None
    session.camera_source.close()
    assert ended.wait(5)
    session.join(5)
    return session


def test_auth_failures_are_counted_and_the_sent_image_is_kept():
    transport = _FailingTransport(
        ModelAPIError("HTTP 401", code="http_error", status_code=401)
    )
    session = _run_camera_session(transport)
    snap = session.diagnostics_snapshot()
    branch = snap["branches"]["analysis"]
    assert transport.calls == 4
    assert branch["failures"]["auth"] == 4
    assert branch["succeeded"] == 0
    assert branch["longest_unanalyzed_gap_seconds"] is None
    assert branch["current_unanalyzed_gap_seconds"] == pytest.approx(1.2)
    entry = snap["model_inputs"][0]
    assert entry["role"] == "safety"
    assert entry["outcome"] == "auth"
    # Diagnostics keep the actual padded model input, not the source frame.
    assert (entry["frames"][0]["width"], entry["frames"][0]["height"]) == (360, 202)
    assert (entry["source_width"], entry["source_height"]) == (640, 360)
    jpeg = session.diagnostics.model_input_jpeg(entry["id"])
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (202, 360)


@pytest.mark.parametrize(
    ("transport", "category"),
    [
        (_FailingTransport(ModelAPIError("slow", code="timeout")), "timeout"),
        (_FailingTransport(text="not json at all"), "parse"),
    ],
)
def test_timeout_and_parse_failures_are_separated(transport, category):
    session = _run_camera_session(transport, frames=2)
    branch = session.diagnostics_snapshot()["branches"]["analysis"]
    assert branch["failures"][category] == 2
    assert sum(branch["failures"].values()) == 2


def test_successful_results_report_capture_to_result_latency():
    text = json.dumps({"no_events": 0.9})
    session = _run_camera_session(_FailingTransport(text=text), frames=3)
    branch = session.diagnostics_snapshot()["branches"]["analysis"]
    assert branch["succeeded"] == 3
    latency = branch["latency_seconds"]["capture_to_result"]
    assert latency["count"] == 3
    assert 0 <= latency["p95"] < 5
    assert branch["longest_unanalyzed_gap_seconds"] == pytest.approx(0.4)


def test_a_finished_session_is_released_but_stays_diagnosable(monkeypatch):
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))
    try:
        runtime.start_camera(
            analysis_mode="multi", features=["smoking"], dry_run=True, owner="release",
        )
        for _ in range(2):
            # A safety-only session caps camera frames at video.max_image_side.
            runtime.push_camera_frame(
                _jpeg(320, 180), width=320, height=180, owner="release",
            )
            time.sleep(0.45)
        session = weakref.ref(runtime._session)
        runtime.stop_session()
        gc.collect()
        assert session() is None, "the finished pipeline is still referenced"
        body = runtime.diagnostics()
        assert body["state"] == "finished"
        assert body["sampling"]["base_fps"] == 2.5
        assert body["branches"]["analysis"]["submitted"] >= 1
        entry = body["model_inputs"][0]
        assert runtime.diagnostics_model_input_jpeg(entry["id"]) is not None
    finally:
        runtime.stop()


def test_diagnostics_http_surface(monkeypatch):
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    runtime = HarnessRuntime(load_config(ROOT / "config.yaml"))
    client = TestClient(create_app(lambda: runtime))
    with client:
        empty = client.get("/v1/diagnostics")
        assert empty.status_code == 200
        assert empty.json()["state"] == "no_session"
        assert client.get(
            "/v1/diagnostics/model-inputs/1/frames/-1.jpg"
        ).status_code == 404
        runtime.start_camera(
            analysis_mode="multi", features=["smoking", "stranger"],
            dry_run=True, owner="diag",
        )
        for _ in range(3):
            runtime.push_camera_frame(
                _jpeg(640, 360), width=640, height=360, owner="diag",
            )
            time.sleep(0.45)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            body = client.get("/v1/diagnostics").json()
            inputs = body.get("model_inputs") or []
            if any(entry["role"] == "safety" for entry in inputs):
                break
            time.sleep(0.05)
        assert body["state"] == "active"
        assert body["session"]["features"] == ["smoking", "stranger"]
        assert set(body["branches"]) <= {"safety", "stranger"}
        assert body["sampling"]["safety"]["fps"] == 2.5
        safety_input = next(e for e in inputs if e["role"] == "safety")
        image = client.get(
            f"/v1/diagnostics/model-inputs/{safety_input['id']}/frames/-1.jpg"
        )
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert image.headers["cache-control"] == "no-store"
        runtime.stop_session()
        finished = client.get("/v1/diagnostics").json()
        assert finished["state"] == "finished"
        assert not contains_private_implementation_detail(json.dumps(finished))
        spec = client.get("/openapi.json").text
        assert "/v1/diagnostics" in spec
        assert not contains_private_implementation_detail(spec)
