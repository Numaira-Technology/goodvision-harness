"""Faces and events draw independently while original display pixels keep moving."""

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
from goodvision_harness.delayed_overlay import DelayedOverlayComposer, select_overlay
from goodvision_harness.overlay_fusion import OverlayFusionMatcher
from goodvision_harness.result_bus import FramePacket
from goodvision_harness.service import (
    HarnessRuntime,
    _analysis_overlay_arguments,
    _draw_analysis_overlay,
    _render_analysis_jpeg,
)


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
        delay_seconds=0, max_gap_seconds=0.8, output_fps=30,
        render=lambda jpeg, analysis: _render_analysis_jpeg(jpeg, analysis, quality=95),
        publish=publish, **kwargs,
    )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("face_kind", ["selected_features", "face_recognition"])
def test_late_event_draws_independent_box_and_preserves_both_green_faces(
    fused, face_kind,
):
    published = []
    ready = threading.Event()

    def publish(packet):
        published.append(packet)
        ready.set()

    runtime = object.__new__(HarnessRuntime)
    runtime._rich_labels = None
    runtime._overlay_fusion = OverlayFusionMatcher()
    runtime.settings = type("Settings", (), {"overlay_jpeg_quality": 95})()
    composer = _composer(
        publish,
        compose=runtime._render_fused_label_jpeg if fused else None,
    )
    try:
        original = _face(1)
        latest = replace(_face(2, moved=True), analysis_kind=face_kind)
        for result in (original, latest):
            composer.submit_result(result)
        event = _event(1)
        unchanged = event.as_dict()
        composer.submit_result(event)
        composer.submit_frame(_packet(2.1))
        assert ready.wait(3)
        packet = published[-1]
        assert packet.annotation_mode == "held"
        assert packet.media_timestamp == 2.1
        assert packet.frame_sequence == 21
        assert packet.analysis_message_id == latest.message_id
        image = cv2.imdecode(np.frombuffer(packet.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert abs(int(image[220, 10, 0]) - 180) < 5  # current original frame
        assert int(image[130, 128, 1]) > int(image[130, 128, 2]) + 70
        assert int(image[130, 240, 1]) > int(image[130, 240, 2]) + 70
        assert int(image[202, 38, 2]) > int(image[202, 38, 1]) + 70
        assert event.as_dict() == unchanged
    finally:
        composer.close()


@pytest.mark.parametrize("empty", [False, True])
def test_empty_expired_and_reset_faces_do_not_suppress_event_boxes(empty):
    composer = _composer()
    try:
        for result in (_face(1), _event(1), _face(2, empty=empty)):
            composer.submit_result(result)
        for at in (2.1, 6):
            selected = select_overlay(
                composer._results, at, max_gap_seconds=0.8, interpolate=False,
            )
            objects = selected.analysis.objects
            assert any(item.label == "cigarette" for item in objects)
            faces = [item for item in objects if item.attributes.get("status")]
            assert len(faces) == (2 if not empty and at < 5 else 0)
        composer.reset()
        selected = select_overlay(composer._results, 6, max_gap_seconds=0.8)
        assert selected.analysis is None
    finally:
        composer.close()


@pytest.mark.parametrize("face_kind", ["face_recognition", "selected_features"])
@pytest.mark.parametrize("safety", ["absent", "clear", "positive", "pending_clear"])
def test_expired_face_is_incomplete_without_false_clear_or_losing_event_boxes(
    face_kind, safety,
):
    face = replace(_face(1), analysis_kind=face_kind, missing_event_labels=())
    records = []
    if safety in {"positive", "pending_clear"}:
        records.append(_event(0.8))
    if safety in {"clear", "pending_clear"}:
        records.append(_event(0.9, positive=False))
    records.append(face)
    original = face.as_dict()
    selected = select_overlay(records, 4.1, max_gap_seconds=0.8)
    analysis = selected.analysis
    assert analysis.expired_event_labels == ("stranger",)
    arguments = _analysis_overlay_arguments(analysis)
    assert arguments["incomplete"]
    assert not arguments["face_objects"]
    positive = safety in {"positive", "pending_clear"}
    assert arguments["detected"] is positive
    assert len(arguments["objects"]) == int(positive)
    image = np.full((240, 320, 3), 180, np.uint8)
    rendered = _draw_analysis_overlay(image, analysis)
    if positive:
        assert int(rendered[195, 38, 2]) > int(rendered[195, 38, 1]) + 70
    else:
        # No face rectangles and no synthesized CLEAR verdict/HUD.
        np.testing.assert_array_equal(rendered, image)
    assert analysis.as_dict() == original


@pytest.mark.parametrize("fused", [False, True])
def test_late_positive_and_clear_repaint_a_paused_face_frame(fused):
    published = []
    ready = threading.Event()

    def publish(packet):
        published.append(packet)
        ready.set()

    runtime = object.__new__(HarnessRuntime)
    runtime._rich_labels = None
    runtime._overlay_fusion = OverlayFusionMatcher()
    runtime.settings = SimpleNamespace(overlay_jpeg_quality=95)
    composer = _composer(
        publish, compose=runtime._render_fused_label_jpeg if fused else None,
    )
    try:
        face = _face(1)
        composer.submit_result(face)
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
            assert packet.media_timestamp == 1.2
            image = cv2.imdecode(np.frombuffer(packet.jpeg, np.uint8), cv2.IMREAD_COLOR)
            _blue, green, red = map(int, image[202, 38])
            assert (red > green + 70) if positive else (abs(red - green) < 8)
            assert int(image[130, 32, 1]) > int(image[130, 32, 2]) + 70
    finally:
        composer.close()


@pytest.mark.parametrize("expect_faces", [False, True])
def test_event_box_does_not_wait_for_first_face_or_subject_association(expect_faces):
    published = []
    ready = threading.Event()

    def publish(packet):
        published.append(packet)
        ready.set()

    composer = _composer(publish)
    try:
        result = _event(1)
        result = replace(
            result, objects=(replace(result.objects[0], attributes={}),),
            missing_event_labels=("stranger",) if expect_faces else (),
        )
        composer.submit_result(result)
        composer.submit_frame(_packet(2))
        assert ready.wait(3)
        image = cv2.imdecode(
            np.frombuffer(published[-1].jpeg, np.uint8), cv2.IMREAD_COLOR,
        )
        assert int(image[202, 38, 2]) > int(image[202, 38, 1]) + 70
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


def test_face_annotation_available_even_when_evidence_preview_cannot_rewind():
    from goodvision_harness.config import load_config

    runtime = HarnessRuntime(config=load_config())
    try:
        runtime.frames.publish(_packet(3))
        face = _face(2)
        runtime._on_timeline_result(face)
        runtime._on_analysis_frame(face, _jpeg(40))
        assert runtime.frames.latest()[1].media_timestamp == 3
        composer = runtime._delayed_overlay
        with composer._condition:
            overlay = select_overlay(
                composer._results, 2.1, max_gap_seconds=0.8, interpolate=False,
            )
        assert overlay.analysis is face
    finally:
        runtime.stop()


def test_thirty_original_frames_inherit_sparse_face_annotations():
    published = []
    ready = threading.Event()

    def publish(packet):
        published.append(packet)
        ready.set()

    composer = _composer(publish)
    try:
        first, second = _face(0), _face(0.4, moved=True)
        composer.submit_result(first)
        composer.submit_result(second)
        for index in range(30):
            ready.clear()
            original = FramePacket(
                _jpeg(40 + index * 5), index / 30, index, "2026-09-10T10:00:00Z",
            )
            composer.submit_frame(original)
            assert ready.wait(3)
            output = published[-1]
            assert output.media_timestamp == original.media_timestamp
            assert output.frame_sequence == index
            assert output.analysis_message_id == (
                first.message_id if index < 12 else second.message_id
            )
            pixels = cv2.imdecode(
                np.frombuffer(output.jpeg, np.uint8), cv2.IMREAD_COLOR,
            )
            assert abs(int(pixels[220, 10, 0]) - (40 + index * 5)) < 5
        assert len(published) == 30
        assert composer.stats()["frames_dropped"] == 0
    finally:
        composer.close()
