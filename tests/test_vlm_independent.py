"""Real spawned-process isolation; no paid model calls and no SAM dependency."""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from threading import Event, Thread

import pytest
from PIL import Image

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.result_bus import ResultBus
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_runtime import SceneRuntime, TemporalInspection
from goodvision_harness.vlm_fusion import VLMFusion
from goodvision_harness.vlm_independent import (
    IndependentEventState,
    IndependentVLMBranch,
    SourceSample,
    VLMProcessConfig,
    utc_now,
)
from goodvision_harness.vlm_independent_model import (
    IndependentVerdict,
    IndependentVLMInspector,
)

ROOT = Path(__file__).resolve().parents[1]
BOX = NormalizedBBox(0.1, 0.1, 0.5, 0.9)


def jpeg():
    with Image.new("RGB", (64, 48), "gray") as picture, io.BytesIO() as data:
        picture.save(data, "JPEG")
        return data.getvalue()


def config(session="vlm-run", generation=1):
    base = load_config(ROOT / "config.yaml")
    return VLMProcessConfig(
        session,
        "camera01",
        generation,
        ("smoking", "fall"),
        base.model_api,
        "test",
        observation_fps=1.25,
        max_in_flight=2,
    )


def job(cfg, at, sequence):
    frames = tuple(
        EvidenceFrame(cfg.stream_session_id, seq, t, jpeg())
        for seq, t in ((sequence - 1, at - 0.8), (sequence, at))
    )
    request = InspectionRequest(
        f"request-{sequence}", (), "independent_source", cfg.labels, at - 0.8, at
    )
    return TemporalInspection(cfg.stream_session_id, cfg.source_id, request, frames)


def verdict(detected=True, sid="p1", box=BOX):
    return IndependentVerdict("smoking", (sid,), detected, 0.9, ((sid, box),))


def report(state, at, sequence, verdicts):
    inspection = job(state.config, at, sequence)
    sample = SourceSample(inspection.frames[-1], utc_now(), time.monotonic())
    return state.report(inspection, sample, verdicts, {"model_seconds": 0.1})


def await_condition(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "independent process did not make progress"
        time.sleep(0.02)


class StubInspector:
    def inspect(self, inspection):
        time.sleep(0.2 if inspection.frames[-1].timestamp_seconds < 1.5 else 0.015)
        return (verdict(),)

    def close(self):
        pass


def stub_factory(_config):
    return StubInspector()


def test_source_confirmation_rules_and_omission_are_unchanged():
    state = IndependentEventState(config())
    first = report(state, 1.0, 10, (verdict(),))
    assert first["events"][0]["phase"] == "suspected"
    duplicate = report(state, 1.0, 10, (verdict(),))
    assert duplicate["events"][0]["positive_observations"] == 1
    second = report(state, 1.8, 18, (verdict(),))
    assert second["events"][0]["phase"] == "confirmed"
    unknown = report(state, 2.6, 26, (verdict(None),))
    assert unknown["events"][0]["phase"] == "confirmed"
    omitted = report(state, 3.4, 34, ())
    assert omitted["judgments"] == []
    assert omitted["missing_event_labels"] == ["smoking", "fall"]
    negative = report(state, 4.2, 42, (verdict(False),))
    assert negative["events"][0]["phase"] == "confirmed"
    cleared = report(state, 5.0, 50, (verdict(False),))
    assert cleared["events"][0]["phase"] == "ended"
    assert cleared["frame"]["timestamp_seconds"] == 5.0
    assert cleared["model"]["latency_seconds"] == 0.1


def test_unrelated_people_do_not_combine_votes():
    state = IndependentEventState(config())
    report(state, 1.0, 10, (verdict(),))
    result = report(state, 1.8, 18, (verdict(box=NormalizedBBox(0.6, 0.1, 0.95, 0.9)),))
    assert all(event["phase"] == "suspected" for event in result["events"])
    assert len({sid for event in result["events"] for sid in event["subject_ids"]}) == 2


def test_independent_model_prompt_and_strict_grounding():
    cfg = config()
    inspector = IndependentVLMInspector(cfg.model_settings, model="test")
    inspection = job(cfg, 1.0, 10)
    messages = inspector.build_messages(inspection)
    assert "allowed_subject_ids" not in messages[1]["content"][0]["text"]
    assert "no external detections" in messages[0]["content"]
    assert "EVERY requested event label" in messages[0]["content"]
    assert all(
        "subject_boxes" not in part.get("text", "") for part in messages[1]["content"]
    )

    raw = {
        "subjects": [{"subject_id": "p1", "bbox": [0.1, 0.1, 0.5, 0.9]}],
        "verdicts": [
            {
                "event_label": "smoking",
                "subject_ids": ["p1"],
                "detected": True,
                "confidence": 0.8,
            }
        ],
    }
    response = {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]
    }
    parsed = inspector._parse(inspection, response)
    assert parsed[0].subject_boxes == (("p1", BOX),)
    raw["verdicts"][0]["subject_ids"] = ["p2"]
    response["choices"][0]["message"]["content"] = json.dumps(raw)
    with pytest.raises(ValueError):
        inspector._parse(inspection, response)
    inspector.close()


def test_actual_process_publishes_while_scene_and_background_are_blocked():
    cfg, bus = config(), ResultBus()
    scene = SceneRuntime(
        cfg.stream_session_id,
        cfg.source_id,
        tracking_mode="batch",
        independent_temporal_inference=True,
    )
    fusion = VLMFusion(scene)
    branch = IndependentVLMBranch(
        cfg, results=bus, fusion=fusion, inspector_factory=stub_factory
    )
    release, held = Event(), Event()

    def hold_scene():
        with scene._lock:
            held.set()
            release.wait(15.0)

    blocker = Thread(target=hold_scene)
    blocker.start()
    assert held.wait(2.0)
    bus.reset()
    branch.start()
    try:
        assert branch._process.pid != os.getpid()
        await_condition(lambda: branch.stats()["state"] == "running")
        started = time.monotonic()
        for sequence in range(0, 97, 6):
            branch.offer(jpeg(), sequence / 15.0, sequence)
            time.sleep(0.09)
        await_condition(lambda: branch.stats().get("published_reports", 0) >= 4)
        records = branch.results.records_after(0)
        assert len(records) >= 4
        assert any(
            event["phase"] == "confirmed" for event in records[-1].payload["events"]
        )
        assert all(
            item.payload["timing"]["source_to_publish_seconds"] < 2.0
            for item in records
        )
        assert time.monotonic() - started < 5.0
        assert bus.latest(feature="smoking").payload["analysis_kind"] == "vlm_event"
        assert fusion.stats()["pending_reports"] > 0
        sequences = [r.payload["frame"]["source_frame_index"] for r in records]
        assert sequences == sorted(set(sequences))
        old_epoch = bus.epoch
        bus.reset()
        assert bus.publish(records[-1].payload, expected_epoch=old_epoch) is None
        assert bus.latest() is None
    finally:
        release.set()
        blocker.join(2.0)
        branch.close(timeout_seconds=6.0)
    assert not branch._process.is_alive()
    assert branch.stats()["state"] == "stopped"


@pytest.mark.parametrize("tracking_mode", ["batch", "immediate"])
def test_fusion_uses_original_evidence_and_does_not_modify_public_report(tracking_mode):
    cfg = config()
    independent = IndependentEventState(cfg)
    scene = SceneRuntime(
        cfg.stream_session_id,
        cfg.source_id,
        tracking_mode=tracking_mode,
        native_memory_scope="session",
        independent_temporal_inference=True,
    )
    fusion = VLMFusion(scene)
    for at, sequence in ((1.0, 10), (1.8, 18)):
        result = report(independent, at, sequence, (verdict(),))
        before = json.dumps(result, sort_keys=True)
        done = set()
        assert not fusion._associate(result, job(cfg, at, sequence).frames, done)
        source_at, source_seq = at + 0.1, sequence + 1
        scene.observe_frame(jpeg(), source_at, source_seq)
        snapshot = RichLabelSnapshot(
            cfg.stream_session_id,
            source_seq,
            source_at,
            (RichLabelObject("sam-person", "person", BOX, 0.99),),
        )
        if tracking_mode == "batch":
            assert scene.observe_tracking_batch(
                (snapshot,),
                completed_timestamp_seconds=source_at,
                completed_frame_sequence=source_seq,
            )
        else:
            scene.observe_tracks(snapshot)
        assert fusion._associate(result, job(cfg, at, sequence).frames, done)
        assert json.dumps(result, sort_keys=True) == before
        assert len(done) == 1
    event = scene.snapshot()["scene"]["events"][0]
    assert event["phase"] == "confirmed"
    assert event["subject_ids"] == ["sam-person"]
    assert event["last_evidence_timestamp_seconds"] == 1.8
    assert len(fusion.links()) == 2
    assert all(
        link["vlm_event_ids"] and link["scene_event_ids"] for link in fusion.links()
    )
    assert all(
        abs(link["subjects"][0]["source_delta_seconds"] - 0.1) < 1e-6
        for link in fusion.links()
    )
    assert scene.take_inspections() == ()
    assert scene.latest_inspection().request.request_id == "request-18"


def test_input_and_background_capacity_remain_bounded():
    cfg = config()
    branch = IndependentVLMBranch(cfg, inspector_factory=stub_factory)
    for sequence in range(100):
        branch.offer(jpeg(), float(sequence), sequence)
    assert branch.stats()["admitted_samples"] == 8
    assert branch.stats()["input_backpressure_dropped"] == 92
    scene = SceneRuntime(cfg.stream_session_id, cfg.source_id)
    fusion = VLMFusion(scene)
    for sequence in range(1, 90):
        payload = report(
            IndependentEventState(cfg), float(sequence), sequence, (verdict(),)
        )
        fusion.offer(payload, ())
    assert fusion.stats()["pending_reports"] == 64
    assert fusion.stats()["capacity_dropped_reports"] == 25
    branch.close(timeout_seconds=2.0)


def test_child_crash_is_visible_and_a_new_session_runs():
    branch = IndependentVLMBranch(config(), inspector_factory=stub_factory)
    branch.start()
    try:
        await_condition(lambda: branch.stats()["state"] == "running")
        branch._process.terminate()
        await_condition(lambda: branch.stats()["state"] == "failed")
        assert branch.stats()["error_code"] == "process_exited"
    finally:
        branch.close(timeout_seconds=6.0)
    new = IndependentVLMBranch(config("next-run", 2), inspector_factory=stub_factory)
    new.start()
    try:
        await_condition(lambda: new.stats()["state"] == "running")
        for seq in range(0, 37, 6):
            new.offer(jpeg(), seq / 15.0, seq)
            time.sleep(0.05)
        await_condition(lambda: new.results.latest() is not None)
        assert new.results.latest().payload["stream_session_id"] == "next-run"
        assert new.results.latest().payload["generation"] == 2
    finally:
        new.close(timeout_seconds=6.0)


def test_network_source_cursor_is_independent_and_drains_without_tracking():
    from threading import Lock
    from types import SimpleNamespace

    lock, frames = Lock(), []
    cursors = []

    def display_frames(cursor, **_kwargs):
        with lock:
            cursors.append(cursor)
            return tuple(frame for frame in frames if frame[0] > cursor)

    branch = IndependentVLMBranch(
        config(),
        inspector_factory=stub_factory,
        connection=SimpleNamespace(display_frames=display_frames),
    )
    branch.start()
    try:
        await_condition(lambda: branch.stats()["state"] == "running")
        for sequence in range(0, 61, 6):
            with lock:
                frames.append((sequence, sequence / 15.0, 64, 48, jpeg()))
            time.sleep(0.06)
        await_condition(lambda: branch.stats().get("published_reports", 0) >= 3)
        await_condition(lambda: branch.wait_until_idle(0.02))
        assert branch.stats()["state"] == "drained"
        assert not branch.offer(jpeg(), 20.0, 300)
        assert cursors == sorted(cursors)
        assert branch.results.latest().payload["stream_session_id"] == "vlm-run"
    finally:
        branch.close(timeout_seconds=6.0)


def test_camera_stop_finishes_admitted_callbacks_before_tracking_end():
    from goodvision_harness.vlm_independent import CameraSourceDispatch

    active, release, received = Event(), Event(), []

    def callback(_jpeg, _at, seq):
        active.set()
        assert release.wait(5.0)
        received.append(seq)

    dispatch = CameraSourceDispatch(callback)
    dispatch.start()
    try:
        dispatch.offer(jpeg(), 1.0, 1)
        assert active.wait(2.0)
        dispatch.offer(jpeg(), 2.0, 2)
        assert not dispatch.finish()
        assert not dispatch.offer(jpeg(), 3.0, 3)
        release.set()
        await_condition(dispatch.finish)
        assert received == [1, 2]
    finally:
        release.set()
        dispatch.close()


def test_unlocalized_negatives_do_not_clear_events_or_discard_valid_positives():
    cfg = config()
    inspector = IndependentVLMInspector(cfg.model_settings, model="test")
    raw = {
        "subjects": [{"subject_id": "p1", "bbox": [0.1, 0.1, 0.5, 0.9]}],
        "verdicts": [
            {
                "event_label": "smoking",
                "subject_ids": ["p1"],
                "detected": True,
                "confidence": 0.9,
            },
            {
                "event_label": "fall",
                "subject_ids": [],
                "detected": False,
                "confidence": None,
            },
        ],
    }

    def parsed():
        return inspector._parse(
            job(cfg, 1.0, 10),
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(raw)}}
                ]
            },
        )

    state = IndependentEventState(cfg)
    report(state, 1.0, 10, parsed())
    result = report(state, 1.8, 18, parsed())
    assert result["events"][0]["phase"] == "confirmed"
    assert not result["analysis_complete"]
    assert result["missing_event_labels"] == ["fall"]
    assert not result["observations"][1]["accepted"]
    unknown_negative = IndependentVerdict("smoking", (), False, 0.9, ())
    for at, seq in ((2.6, 26), (3.4, 34)):
        result = report(state, at, seq, (unknown_negative,))
        assert result["events"][0]["phase"] == "confirmed"
        assert result["judgments"] == []
    raw["verdicts"][1]["detected"] = True
    with pytest.raises(ValueError):
        parsed()
    inspector.close()


def test_pose_change_clears_same_episode_without_merging_ambiguous_people():
    from goodvision_harness.vlm_independent import unambiguous_matches

    # Recorded model boxes during the fall/get-up transition, not stable IDs.
    boxes = [
        NormalizedBBox(*box)
        for box in (
            (0.37, 0.24, 0.73, 0.68),
            (0.36, 0.34, 0.72, 0.68),
            (0.41, 0.37, 0.72, 0.68),
            (0.43, 0.27, 0.75, 0.83),
        )
    ]
    state = IndependentEventState(config())
    for n, box in enumerate(boxes):
        result = report(
            state,
            12.8 + n * 0.8,
            192 + n * 12,
            (IndependentVerdict("fall", ("p1",), n < 2, 0.9, (("p1", box),)),),
        )
    assert len(result["events"]) == 1
    assert result["events"][0]["phase"] == "ended"
    # Two plausible overlapping actors must not share confirmation/clear votes.
    assert (
        unambiguous_matches(
            {"p1": boxes[2]}, {"a": boxes[2], "b": boxes[2]}, pose_tolerance=True
        )
        == {}
    )
    assert (
        unambiguous_matches(
            {"p1": boxes[2], "p2": boxes[2]}, {"a": boxes[2]}, pose_tolerance=True
        )
        == {}
    )
    # Looser temporal matching must not leak into delayed SAM fusion.
    assert unambiguous_matches({"p1": boxes[3]}, {"a": boxes[2]}) == {}
