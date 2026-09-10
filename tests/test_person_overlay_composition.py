"""Late behavior verdicts must follow identities on the exact latest face JPEG."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.delayed_overlay import DelayedOverlayComposer, _BufferedFrame
from goodvision_harness.result_bus import FramePacket
from goodvision_harness.service import HarnessRuntime, _render_analysis_jpeg


def _face(at: float, *, moved: bool = False, empty: bool = False) -> AnalysisResult:
    faces = () if empty else tuple(
        GroundedObject(
            identity, "known_person", NormalizedBBox(x, 0.3, x + 0.15, 0.7),
            0.95, {
                "subject_id": identity, "subject_ids": [identity],
                "person_id": identity, "status": "known",
                "face_observation_state": "observed",
            },
        )
        for identity, x in (("alice", 0.4 if moved else 0.1), ("bob", 0.75))
    )
    return AnalysisResult(
        message_id=f"face-{at}", source_id="camera", stream_session_id="session",
        sequence=int(at * 10) + 1, emitted_at="2026-09-10T10:00:00Z",
        frame=FrameReference(int(at * 10), at, "2026-09-10T10:00:00Z", 320, 240),
        judgments=(EventJudgment("stranger", False, 0.95, "Known faces", ()),),
        objects=faces, model_name="face", model_latency_seconds=0.1,
        analysis_kind="selected_features", missing_event_labels=("smoking",),
    )


def _event(at: float, *, positive: bool = True) -> AnalysisResult:
    source = _face(at)
    return replace(
        source, message_id=f"event-{at}",
        judgments=(EventJudgment(
            "smoking", positive, 0.95, "Smoking" if positive else "Clear",
            ("cigarette",) if positive else (),
        ),),
        objects=(GroundedObject(
            "cigarette", "cigarette", NormalizedBBox(0.12, 0.7, 0.17, 0.85),
            0.95, {"subject_id": "alice", "subject_ids": ["alice"]},
        ),) if positive else (),
        missing_event_labels=("stranger",), model_name="event",
    )


def _jpeg(gray: int) -> bytes:
    ok, output = cv2.imencode(".jpg", np.full((240, 320, 3), gray, np.uint8))
    assert ok
    return output.tobytes()


def _packet(at: float, gray: int = 180) -> FramePacket:
    return FramePacket(_jpeg(gray), at, int(at * 10), "2026-09-10T10:00:00Z")


def _composer(publish=lambda _packet: None, **kwargs) -> DelayedOverlayComposer:
    return DelayedOverlayComposer(
        delay_seconds=0, max_gap_seconds=0.8, output_fps=15,
        render=lambda jpeg, analysis: _render_analysis_jpeg(jpeg, analysis, quality=95),
        publish=publish, **kwargs,
    )


@pytest.mark.parametrize("fused", [False, True])
def test_late_event_uses_latest_face_pixels_and_preserves_other_green_face(fused):
    published = []
    ready = threading.Event()

    def publish(packet):
        published.append(packet)
        ready.set()

    runtime = object.__new__(HarnessRuntime)
    runtime._rich_labels = None
    runtime.settings = type("Settings", (), {"overlay_jpeg_quality": 95})()
    composer = _composer(
        publish,
        compose=runtime._render_fused_label_jpeg if fused else None,
    )
    try:
        original = _face(1)
        latest = _face(2, moved=True)
        for result, gray in ((original, 20), (latest, 60)):
            composer.submit_result(result)
            composer.submit_analysis_frame(result, _jpeg(gray))
        event = _event(1)
        unchanged = event.as_dict()
        composer.submit_result(event)
        composer.submit_frame(_packet(2.1))
        assert ready.wait(3)
        packet = published[-1]
        assert packet.annotation_mode == "face_aligned"
        assert packet.media_timestamp == 2
        assert packet.frame_sequence == 20
        assert packet.analysis_message_id == latest.message_id
        image = cv2.imdecode(np.frombuffer(packet.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert abs(int(image[220, 10, 0]) - 60) < 5  # exact latest face JPEG
        assert int(image[130, 128, 2]) > int(image[130, 128, 1]) + 70
        assert int(image[130, 240, 1]) > int(image[130, 240, 2]) + 70
        assert max(image[195, 38]) - min(image[195, 38]) < 8  # no old cigarette box
        assert event.as_dict() == unchanged
    finally:
        composer.close()


def test_empty_expired_and_reset_faces_never_reuse_old_event_coordinates():
    composer = _composer()
    try:
        for result in (_face(1), _event(1), _face(2, empty=True)):
            composer.submit_result(result)
            composer.submit_analysis_frame(result, _jpeg(50))
        with composer._condition:
            selected = composer._select_person_frame_locked(
                _BufferedFrame(_packet(2.1), 0, composer._generation),
            )
        assert selected is not None
        assert selected[1].analysis.objects == ()
        with composer._condition:
            expired = composer._select_person_frame_locked(
                _BufferedFrame(_packet(6), 0, composer._generation),
            )
        assert expired is not None and expired[1].analysis is None
        assert expired[0].packet.media_timestamp == 6
        composer.reset()
        assert not composer._face_frames
        assert not composer._person_events.has_face_results
    finally:
        composer.close()


def test_late_positive_and_clear_repaint_a_paused_face_frame():
    published = []
    ready = threading.Event()

    def publish(packet):
        published.append(packet)
        ready.set()

    composer = _composer(publish)
    try:
        face = _face(1)
        composer.submit_result(face)
        composer.submit_analysis_frame(face, _jpeg(60))
        composer.submit_frame(_packet(1.2))
        assert ready.wait(3)
        cases = [(_event(1), True), *(
            (_event(1 + index / 100, positive=False), index < 10)
            for index in range(1, 11)
        )]
        for result, positive in cases:
            ready.clear()
            composer.submit_result(result)
            assert ready.wait(3), "late event did not repaint the paused face frame"
            packet = published[-1]
            assert packet.media_timestamp == 1
            image = cv2.imdecode(np.frombuffer(packet.jpeg, np.uint8), cv2.IMREAD_COLOR)
            _blue, green, red = map(int, image[130, 32])
            assert (red > green + 70) if positive else (green > red + 70)
    finally:
        composer.close()


@pytest.mark.parametrize("expect_faces", [False, True])
def test_mixed_session_waits_for_first_face_before_drawing_event(expect_faces):
    composer = _composer()
    try:
        result = _event(1)
        if not expect_faces:
            result = replace(result, missing_event_labels=())
        composer.submit_result(result)
        with composer._condition:
            selected = composer._select_person_frame_locked(
                _BufferedFrame(_packet(2), 0, composer._generation),
            )
        if expect_faces:
            assert selected is not None and selected[1].analysis is None
        else:
            assert selected is None  # pure event sessions keep their renderer
    finally:
        composer.close()


@pytest.mark.parametrize("with_archive", [False, True])
def test_tracking_corrections_reach_display_with_or_without_archive(with_archive):
    runtime = object.__new__(HarnessRuntime)
    runtime._state_lock = threading.RLock()
    snapshots, timeline, archived = [], [], []
    corrected = _event(1)
    runtime._session = SimpleNamespace(
        stream_session_id="session",
        person_registry=SimpleNamespace(
            observe_snapshot=snapshots.append,
            drain_reassociated_results=lambda: (corrected,),
        ),
        archive=(
            SimpleNamespace(submit_result=archived.append) if with_archive else None
        ),
    )
    runtime._on_timeline_result = timeline.append
    snapshot = SimpleNamespace(stream_session_id="session")
    runtime._on_tracking_snapshot(snapshot)
    assert snapshots == [snapshot]
    assert timeline == [corrected]
    assert archived == ([corrected] if with_archive else [])
    runtime._on_tracking_snapshot(SimpleNamespace(stream_session_id="old-session"))
    assert timeline == [corrected]


def test_source_jpeg_is_cached_before_global_evidence_rewind_guard():
    from goodvision_harness.config import load_config

    runtime = HarnessRuntime(config=load_config())
    try:
        runtime.frames.publish(_packet(3))
        face = _face(2)
        runtime._on_timeline_result(face)
        runtime._on_analysis_frame(face, _jpeg(40))
        assert runtime.frames.latest()[1].media_timestamp == 3
        cached = runtime._delayed_overlay._face_frames[face.message_id]
        assert cached.media_timestamp == 2
    finally:
        runtime.stop()
