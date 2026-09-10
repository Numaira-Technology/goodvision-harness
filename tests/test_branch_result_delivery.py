"""Independent branch delivery preserves exact frames and source-ordered snapshots."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jsonschema
from fastapi.testclient import TestClient

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.result_bus import FrameBus, ResultBus
from goodvision_harness.service import (
    HarnessRuntime,
    _result_events_response,
    create_app,
)


def observation(label, timestamp, *, detected=True, message=None, frame=None):
    return AnalysisResult(
        message_id=message or f"{label}-{timestamp}",
        source_id="camera",
        stream_session_id="session-one",
        sequence=1,
        emitted_at="2026-09-10T00:00:00Z",
        frame=FrameReference(
            int(timestamp * 10) if frame is None else frame,
            timestamp,
            "2026-09-10T00:00:00Z",
            160,
            90,
        ),
        judgments=(EventJudgment(
            label, detected, 0.9, "Current branch observation.",
            ("object-1",) if detected else (),
        ),),
        objects=(GroundedObject(
            "object-1", "unknown_person" if label == "stranger" else "cigarette",
            NormalizedBBox(0.2, 0.2, 0.5, 0.8), 0.9,
            {"subject_id": "subject-one"},
        ),) if detected else (),
        missing_event_labels=("smoking",) if label == "stranger" else ("stranger",),
        model_name="standard",
        model_latency_seconds=0.01,
        analysis_kind="selected_features",
    )


def runtime_without_workers():
    # Exercise the actual service callbacks and routes without starting any
    # camera, model, archive, background thread, or network delivery worker.
    runtime = object.__new__(HarnessRuntime)
    runtime.results = ResultBus(history_size=2)
    runtime._state_lock = threading.RLock()
    runtime._state = {"results_published": 0}
    runtime._generation = 1
    webhook = []
    runtime._delivery = SimpleNamespace(submit=webhook.append)
    runtime.start = lambda: None
    runtime.stop = lambda: None
    return runtime, webhook


def sse_records(runtime, *, feature=None, after=0):
    async def collect():
        checks = 0

        async def disconnected():
            nonlocal checks
            checks += 1
            return checks > 1

        request = SimpleNamespace(headers={}, is_disconnected=disconnected)
        response = _result_events_response(
            request, runtime, after=after, feature=feature,
        )
        result = []
        async for chunk in response.body_iterator:
            text = chunk.decode()
            if text.startswith(":"):
                continue
            lines = text.splitlines()
            result.append((
                next(line[4:] for line in lines if line.startswith("id: ")),
                json.loads(next(
                    line[6:] for line in lines if line.startswith("data: ")
                )),
            ))
        return result

    return asyncio.run(collect())


def test_latest_global_and_feature_use_source_frame_order_and_original_payload():
    bus = ResultBus(history_size=2)
    safety = observation("smoking", 2.0, detected=False).as_dict()
    face = observation("stranger", 1.8).as_dict()
    safety_record = bus.publish(safety)
    face_record = bus.publish(face)

    assert face_record.cursor == safety_record.cursor + 1
    assert bus.latest().payload == safety
    assert bus.latest(feature="stranger").payload == face
    assert bus.latest(feature="smoking").payload == safety
    assert [r.payload for r in bus.records_after(0)] == [safety, face]

    # An older same-feature record cannot move any snapshot backward, even if
    # a caller bypasses the streaming organizer's same-branch rejection.
    bus.publish(observation("smoking", 1.0).as_dict())
    assert bus.latest().payload == safety
    assert bus.latest(feature="smoking").payload == safety
    # Feature cache survives eviction from the bounded arrival history.
    bus.publish(observation("smoking", 3.0, detected=False).as_dict())
    assert bus.latest(feature="stranger").payload == face
    assert len(bus.records_after(0)) == 2

    # Same source time uses frame index, then later arrival for an exact tie.
    same_time = observation("stranger", 3.0, frame=31).as_dict()
    bus.publish(same_time)
    bus.publish(observation("stranger", 3.0, frame=30).as_dict())
    assert bus.latest().payload == same_time
    tied = {**same_time, "message_id": "same-frame-later-branch"}
    bus.publish(tied)
    assert bus.latest().payload == tied
    assert bus.latest(feature="stranger").payload == tied
    snapshot = bus.latest(feature="stranger")
    snapshot.payload["objects"].clear()
    assert bus.latest(feature="stranger").payload == tied


def test_legacy_unknown_person_feature_alias_and_reset_clear_all_watermarks():
    bus = ResultBus(history_size=1)
    old = observation("stranger", 100).as_dict()
    old["judgments"][0]["event_label"] = "unknown_person"
    first = bus.publish(old)
    assert bus.latest(feature="stranger").payload == old
    assert bus.latest(feature="unknown_person").payload == old
    epoch = bus.epoch
    bus.reset()
    assert bus.latest() is None
    assert bus.latest(feature="stranger") is None
    assert bus.records_after(0) == ()
    assert bus.epoch != epoch
    new = replace(
        observation("smoking", 0.0, detected=False), stream_session_id="session-two",
    ).as_dict()
    second = bus.publish(new)
    assert second.cursor == first.cursor + 1
    assert bus.latest().payload == new
    assert bus.latest(feature="stranger") is None


def test_service_sse_webhook_and_feature_latest_keep_independent_exact_frames():
    runtime, webhook = runtime_without_workers()
    safety = observation("smoking", 2.0, detected=False)
    face = observation("stranger", 1.8)
    for result in (safety, face):
        runtime._guarded_callback(1, runtime._on_analysis_result, result)
    original = [safety.as_dict(), face.as_dict()]
    assert webhook == original
    assert runtime._state["results_published"] == 2
    schema = json.loads((
        Path(__file__).parents[1] / "schemas/analysis-result-v2.schema.json"
    ).read_text(encoding="utf-8"))
    for payload in webhook:
        jsonschema.validate(payload, schema)
        assert payload["analysis_complete"] is False

    records = sse_records(runtime)
    assert [p for _, p in records] == original
    assert [cursor for cursor, _ in records] == [
        f"{runtime.results.epoch}:1", f"{runtime.results.epoch}:2",
    ]
    assert [p for _, p in sse_records(runtime, after=1)] == [face.as_dict()]
    assert [p for _, p in sse_records(runtime, feature="stranger")] == [face.as_dict()]

    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        latest = client.get("/v1/results/latest")
        assert latest.json() == safety.as_dict()
        assert latest.headers["X-Result-Cursor"] == "1"
        face_latest = client.get("/v1/results/stranger/latest")
        assert face_latest.status_code == 200
        assert face_latest.json() == face.as_dict()
        assert face_latest.headers["X-Result-Cursor"] == "2"
        assert face_latest.headers["X-Result-Feature"] == "stranger"
        assert client.get("/v1/results/smoking/latest").json() == safety.as_dict()
        assert client.get("/v1/results/overall/latest").json()["frame"] == (
            safety.frame.as_dict()
        )
        assert client.get("/v1/results/fight/latest").status_code == 404


def test_service_generation_guard_cannot_repopulate_reset_feature_cache():
    runtime, webhook = runtime_without_workers()
    runtime._guarded_callback(
        1, runtime._on_analysis_result, observation("stranger", 100),
    )
    runtime._generation = 2
    runtime.results.reset()
    runtime._guarded_callback(
        1, runtime._on_analysis_result, observation("stranger", 101),
    )
    new = replace(
        observation("smoking", 0.0, detected=False), stream_session_id="session-two",
    )
    runtime._guarded_callback(2, runtime._on_analysis_result, new)
    assert len(webhook) == 2
    assert runtime.results.latest().payload == new.as_dict()
    assert runtime.results.latest(feature="stranger") is None
    assert [r.payload for r in runtime.results.records_after(0)] == [new.as_dict()]


def test_exact_frame_callback_keeps_memory_but_does_not_rewind_preview(monkeypatch):
    runtime, _ = runtime_without_workers()
    runtime.frames = FrameBus()
    runtime._rich_labels = None
    runtime.settings = SimpleNamespace(overlay_jpeg_quality=85)
    runtime._state.update(
        analysis_frames_published=0,
        memory_records_accepted=0,
        memory_records_dropped=0,
    )
    memory = []

    def submit(record):
        memory.append(record)
        return True

    runtime._memory_sink = SimpleNamespace(submit=submit)
    monkeypatch.setattr(
        "goodvision_harness.service._render_analysis_jpeg",
        lambda jpeg, result, **kwargs: jpeg,
    )
    safety = observation("smoking", 2.0, detected=False)
    face = observation("stranger", 1.8)
    runtime._guarded_callback(1, runtime._on_analysis_frame, safety, b"safety-frame")
    first_version, first_packet = runtime.frames.latest()
    runtime._guarded_callback(1, runtime._on_analysis_frame, face, b"face-frame")
    assert runtime.frames.latest() == (first_version, first_packet)
    assert first_packet.jpeg == b"safety-frame"
    assert first_packet.analysis_message_id == safety.message_id
    assert [r.observation for r in memory] == [safety.as_dict(), face.as_dict()]
    assert [r.evidence_jpeg for r in memory] == [b"safety-frame", b"face-frame"]

    same_frame = observation("stranger", 2.0)
    runtime._guarded_callback(
        1, runtime._on_analysis_frame, same_frame, b"same-frame-face",
    )
    assert runtime.frames.latest()[1].jpeg == b"same-frame-face"
    assert runtime._state["analysis_frames_published"] == 2
    assert runtime._state["memory_records_accepted"] == 3

    runtime._generation = 2
    runtime.frames.reset()
    runtime._guarded_callback(1, runtime._on_analysis_frame, face, b"old-session")
    assert runtime.frames.latest() is None
    new = replace(
        observation("stranger", 0.0), stream_session_id="session-two",
    )
    runtime._guarded_callback(2, runtime._on_analysis_frame, new, b"new-session")
    assert runtime.frames.latest()[1].jpeg == b"new-session"
