"""Independent branch delivery preserves exact frames and source-ordered snapshots."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jsonschema
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
from goodvision_harness.grounding import (
    GroundedClassification,
    GroundingResponse,
    ModelObject,
)
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.result_bus import FrameBus, ResultBus
from goodvision_harness.service import (
    HarnessRuntime,
    _result_events_response,
    create_app,
)
from goodvision_harness.sliding_window import FrameWindow, SampledFrame
from goodvision_harness.vision_result import PreliminaryVisionResult


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


def sse_records(runtime, *, feature=None, overall_only=False, after=0):
    async def collect():
        checks = 0

        async def disconnected():
            nonlocal checks
            checks += 1
            return checks > 1

        request = SimpleNamespace(headers={}, is_disconnected=disconnected)
        response = _result_events_response(
            request, runtime, after=after, feature=feature,
            overall_only=overall_only,
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


def test_preliminary_result_stays_in_generic_stream_but_not_overall_sse():
    runtime, _ = runtime_without_workers()
    preliminary = PreliminaryVisionResult(
        event_label="smoking",
        detected=True,
        confidence=0.9,
        window_index=1,
        source_frame_index=2,
        window_start_seconds=0.0,
        window_end_seconds=1.0,
        latency_seconds=0.1,
    ).as_dict()
    initial = runtime.results.publish(preliminary, update_snapshots=False)
    final = observation("smoking", 1.0).as_dict()
    completed = runtime.results.publish(final)

    assert [payload for _, payload in sse_records(runtime)] == [preliminary, final]
    overall = sse_records(runtime, overall_only=True)
    assert len(overall) == 1
    assert overall[0][0] == f"{runtime.results.epoch}:{completed.cursor}"
    assert overall[0][1]["type"] == "overall_description"
    assert overall[0][1]["observation_id"] == final["observation_id"]
    assert sse_records(runtime, overall_only=True, after=initial.cursor) == overall
    assert [
        payload["type"] for _, payload in sse_records(runtime, feature="smoking")
    ] == ["analysis_result"]


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

    # A late positive reaches the arrival log without moving either snapshot
    # backward to a superseded observation of the same feature.
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


@pytest.mark.parametrize("older_outcome", ("positive", "negative", "failed"))
def test_single_event_late_positive_reaches_sse_and_webhook_without_frame_rewind(
    older_outcome, monkeypatch,
):
    runtime, webhook = runtime_without_workers()
    config = load_config()
    config = replace(config, inference=replace(config.inference, event_label="smoking"))
    exact, timeline, archived, visions = [], [], [], []
    session = StreamingSession.for_camera(
        config,
        SimpleNamespace(settings=config.inference),
        SessionCallbacks(
            on_analysis_result=runtime._on_analysis_result,
            on_analysis_frame=lambda result, jpeg: exact.append((result, jpeg)),
            on_timeline_result=timeline.append,
            on_vision_result=visions.append,
        ),
        archive=SimpleNamespace(
            submit_result=lambda result, **kwargs: archived.append(result) or True,
            note_window=lambda *args, **kwargs: None,
        ),
    )

    def complete(index, outcome):
        window = FrameWindow(index, (SampledFrame(
            index, index * 10, index * 0.4, f"jpeg-{index}".encode(), 320, 180,
        ),))
        future = Future()
        if outcome == "failed":
            future.set_exception(ValueError("invalid model result"))
        else:
            detected = outcome == "positive"
            objects = (ModelObject(
                "cigarette", NormalizedBBox(0.2, 0.2, 0.4, 0.4),
            ),) if detected else ()
            future.set_result(GroundingResponse(
                window,
                GroundedClassification(detected, 0.98, "Observed", objects),
                1.3 if detected else 0.49,
            ))
        session._apply_window_future(future, window)

    complete(1, "negative")
    original_latest = runtime.results.latest()
    if older_outcome == "positive":
        # Association corrections use the private callback only. They must not
        # send a second notification for the same model observation.
        monkeypatch.setattr(
            session.person_registry, "drain_reassociated_results",
            lambda: (timeline[-1],),
        )
    complete(0, older_outcome)

    assert runtime.results.latest() == original_latest
    assert runtime.results.latest(feature="smoking") == original_latest
    assert [result.frame.source_frame_index for result, _ in exact] == [10]
    assert exact[0][1] == b"jpeg-1"
    assert len(visions) == 1
    assert visions[0].detected is False
    assert session.organizer.snapshot()["decision"]["detected"] is False

    if older_outcome != "failed":
        assert [payload["sequence"] for payload in webhook] == [1, 2]
        assert [payload["judgments"][0]["detected"] for payload in webhook] == [
            False, older_outcome == "positive",
        ]
        assert webhook[-1]["frame"]["source_frame_index"] == 0
        assert webhook[-1]["model"]["latency_seconds"] == (
            1.3 if older_outcome == "positive" else 0.49
        )
        assert timeline[-1].message_id == webhook[-1]["message_id"]
        assert [payload for _, payload in sse_records(runtime)] == webhook
        assert [payload for _, payload in sse_records(runtime, after=1)] == [
            webhook[-1],
        ]
        assert len({payload["message_id"] for payload in webhook}) == 2
        assert archived[-1].message_id == webhook[-1]["message_id"]
    else:
        assert len(webhook) == 1
        assert runtime.results.records_after(1) == ()
        assert len(archived) == 1


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
