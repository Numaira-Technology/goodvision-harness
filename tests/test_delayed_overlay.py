from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.delayed_overlay import (
    DelayedOverlayComposer,
    OverlaySelection,
    select_overlay,
)
from goodvision_harness.result_bus import FramePacket
from goodvision_harness.service import _analysis_overlay_arguments


def _result(
    *,
    timestamp: float,
    frame_index: int,
    message_id: str,
    bbox: tuple[float, float, float, float] | None,
    detected: bool = True,
    object_id: str = "object-1",
) -> AnalysisResult:
    objects = (
        (
            GroundedObject(
                object_id=object_id,
                label="cigarette",
                bbox=NormalizedBBox(*bbox),
                confidence=0.8,
            ),
        )
        if bbox is not None
        else ()
    )
    judgment = EventJudgment(
        event_label="smoking",
        detected=detected,
        confidence=0.9,
        description="Smoking evidence." if detected else "No smoking evidence.",
        object_ids=(object_id,) if detected else (),
    )
    return AnalysisResult(
        message_id=message_id,
        source_id="camera",
        stream_session_id="session-1",
        sequence=frame_index + 1,
        emitted_at="2026-08-27T00:00:02.000Z",
        frame=FrameReference(
            source_frame_index=frame_index,
            timestamp_seconds=timestamp,
            observed_at="2026-08-27T00:00:00.000Z",
            width=1280,
            height=720,
        ),
        judgments=(judgment,),
        objects=objects,
        model_name="model",
        model_latency_seconds=2.0,
    )


def _packet(sequence: int, timestamp: float) -> FramePacket:
    return FramePacket(
        jpeg=b"jpeg",
        media_timestamp=timestamp,
        frame_sequence=sequence,
        observed_at="2026-08-27T00:00:00.000Z",
    )


def _selected_features_result(
    *,
    timestamp: float,
    frame_index: int,
    message_id: str,
    stranger: bool | None,
    smoking: bool = False,
    session_id: str = "session-1",
    known_person_id: str | None = None,
) -> AnalysisResult:
    result = _result(
        timestamp=timestamp,
        frame_index=frame_index,
        message_id=message_id,
        bbox=(0.6, 0.5, 0.7, 0.6) if smoking else None,
        detected=smoking,
    )
    judgments = list(result.judgments)
    objects = list(result.objects)
    if stranger is not None:
        object_id = "object-2" if smoking else "object-1"
        judgments.append(EventJudgment(
            event_label="stranger",
            detected=stranger,
            confidence=0.9 if stranger else 0.0,
            description="Confirmed stranger." if stranger else "No stranger.",
            object_ids=(object_id,) if stranger else (),
        ))
        if stranger:
            objects.append(GroundedObject(
                object_id=object_id,
                label="unknown_person",
                bbox=NormalizedBBox(0.2, 0.1, 0.3, 0.25),
                confidence=0.9,
                attributes={"status": "unknown", "person_track_id": "P0001"},
            ))
    if known_person_id is not None:
        objects.append(GroundedObject(
            object_id="known-face",
            label="known_person",
            bbox=NormalizedBBox(0.4, 0.1, 0.5, 0.25),
            confidence=0.95,
            attributes={
                "status": "known", "person_id": known_person_id,
                "name": "Same name", "person_track_id": "P0002",
            },
        ))
    return replace(
        result,
        stream_session_id=session_id,
        analysis_kind="selected_features",
        judgments=tuple(judgments),
        objects=tuple(objects),
        missing_event_labels=("stranger",) if stranger is None else (),
    )


@pytest.mark.parametrize("stranger", (True, False))
@pytest.mark.parametrize("timestamp", (10.4, 10.6, 11.0, 13.1))
def test_carried_overlay_preserves_v2_incompleteness_and_original_evidence(
    stranger: bool, timestamp: float,
) -> None:
    previous = _selected_features_result(
        timestamp=10.0, frame_index=100, message_id="face", stranger=stranger,
    )
    skipped = _selected_features_result(
        timestamp=10.4, frame_index=104, message_id="skipped", stranger=None,
    )
    later = _selected_features_result(
        timestamp=10.8, frame_index=108, message_id="later", stranger=None,
    )
    original = skipped.as_dict()

    selection = select_overlay(
        (previous, skipped, later), timestamp, max_gap_seconds=0.8,
    )

    assert selection.analysis is not None
    arguments = _analysis_overlay_arguments(selection.analysis)
    assert arguments["incomplete"] is True
    assert "Analysis incomplete" in arguments["description"]
    assert arguments["detected"] is (stranger and timestamp <= 13.0)
    assert bool(arguments["objects"]) is (stranger and timestamp <= 13.0)
    assert selection.analysis.as_dict() == (
        original if timestamp < 10.8 else later.as_dict()
    )
    assert skipped.as_dict() == original
    assert original["missing_event_labels"] == ["stranger"]
    assert [item["event_label"] for item in original["judgments"]] == ["smoking"]


def test_select_overlay_keeps_stranger_between_alternating_face_samples() -> None:
    # Recorded live pattern: a busy face branch is omitted from safety results.
    # Arrival may be out of order; the selector receives the sorted timeline.
    records = [
        _selected_features_result(
            timestamp=timestamp,
            frame_index=frame_index,
            message_id=message_id,
            stranger=stranger,
        )
        for timestamp, frame_index, message_id, stranger in (
            (8.813, 127, "R15", None),
            (7.625, 110, "R12", True),
            (8.485, 122, "R14", True),
            (8.032, 116, "R13", None),
        )
    ]
    records.sort(key=lambda item: (
        item.frame.timestamp_seconds, item.frame.source_frame_index,
    ))

    for record, expected_source in zip(
        records, ("R12", "R12", "R14", "R14"), strict=True,
    ):
        selection = select_overlay(
            records,
            record.frame.timestamp_seconds,
            frame_sequence=record.frame.source_frame_index,
            max_gap_seconds=0.8,
        )
        assert selection.analysis is not None
        stranger = next(
            item for item in selection.analysis.judgments
            if item.event_label == "stranger"
        )
        assert stranger.detected is True
        objects = {item.object_id: item for item in selection.analysis.objects}
        assert len(stranger.object_ids) == 1
        face = objects[stranger.object_ids[0]]
        assert face.label == "unknown_person"
        assert face.attributes["person_track_id"] == "P0001"
        if record.message_id != expected_source:
            assert face.object_id == f"held-stranger-{expected_source}-object-1"
            assert face.attributes["annotation_source_message_id"] == expected_source


@pytest.mark.parametrize(
    ("media_timestamp", "expected"), [(13.0, True), (13.001, False)],
)
def test_select_overlay_expires_missing_stranger_from_original_observation(
    media_timestamp: float, expected: bool,
) -> None:
    positive = _selected_features_result(
        timestamp=10.0, frame_index=100, message_id="face", stranger=True,
    )
    skipped = _selected_features_result(
        timestamp=12.8, frame_index=142, message_id="safety", stranger=None,
    )

    selection = select_overlay(
        (positive, skipped), media_timestamp, max_gap_seconds=0.8,
    )

    assert selection.analysis is not None
    active_strangers = [
        item for item in selection.analysis.judgments
        if item.event_label == "stranger" and item.detected
    ]
    assert bool(active_strangers) is expected
    assert any(
        item.label == "unknown_person" for item in selection.analysis.objects
    ) is expected


def test_select_overlay_negative_clears_stranger_and_prevents_revival() -> None:
    positive = _selected_features_result(
        timestamp=10.0, frame_index=100, message_id="positive", stranger=True,
    )
    negative = _selected_features_result(
        timestamp=10.4, frame_index=106, message_id="negative", stranger=False,
    )
    skipped = _selected_features_result(
        timestamp=10.8, frame_index=112, message_id="skipped", stranger=None,
        smoking=True,
    )

    for timestamp in (10.4, 10.8, 11.0):
        selection = select_overlay(
            (positive, negative, skipped), timestamp, max_gap_seconds=0.8,
        )
        assert selection.analysis is not None
        assert not any(
            item.event_label == "stranger" and item.detected
            for item in selection.analysis.judgments
        )
        assert not any(
            item.label == "unknown_person" for item in selection.analysis.objects
        )


def test_select_overlay_carried_stranger_preserves_safety_object_ids() -> None:
    positive = _selected_features_result(
        timestamp=10.0, frame_index=100, message_id="face-source", stranger=True,
    )
    skipped = _selected_features_result(
        timestamp=10.4, frame_index=106, message_id="safety", stranger=None,
        smoking=True,
    )

    for timestamp in (10.4, 10.6):
        selection = select_overlay(
            (positive, skipped), timestamp, max_gap_seconds=0.8,
        )
        assert selection.analysis is not None
        judgments = {
            item.event_label: item for item in selection.analysis.judgments
        }
        objects = {item.object_id: item for item in selection.analysis.objects}
        assert len(objects) == len(selection.analysis.objects) == 2
        assert judgments["smoking"].object_ids == ("object-1",)
        assert objects["object-1"].label == "cigarette"
        expected_id = "held-stranger-face-source-object-1"
        assert judgments["stranger"].object_ids == (expected_id,)
        assert objects[expected_id].label == "unknown_person"
        assert objects[expected_id].attributes["annotation_source_message_id"] == (
            "face-source"
        )
    assert len(skipped.objects) == 1
    assert len(skipped.judgments) == 1
    assert "annotation_source_message_id" not in positive.objects[0].attributes


@pytest.mark.parametrize("stranger", [False, True])
def test_select_overlay_carries_known_face_without_stranger_reference(
    stranger: bool,
) -> None:
    face_sample = _selected_features_result(
        timestamp=10.0, frame_index=100, message_id="face-source",
        stranger=stranger, known_person_id="employee-7",
    )
    skipped = _selected_features_result(
        timestamp=10.4, frame_index=106, message_id="safety", stranger=None,
        smoking=True,
    )

    selection = select_overlay((face_sample, skipped), 10.6, max_gap_seconds=0.8)

    assert selection.analysis is not None
    objects = {item.object_id: item for item in selection.analysis.objects}
    known = objects["held-stranger-face-source-known-face"]
    assert known.attributes["status"] == "known"
    assert known.attributes["person_id"] == "employee-7"
    assert known.attributes["person_track_id"] == "P0002"
    assert known.attributes["annotation_source_message_id"] == "face-source"
    judgments = {item.event_label: item for item in selection.analysis.judgments}
    assert judgments["smoking"].object_ids == ("object-1",)
    assert judgments["stranger"].detected is stranger
    assert judgments["stranger"].object_ids == (
        ("held-stranger-face-source-object-1",) if stranger else ()
    )
    assert len(objects) == 2 + int(stranger)
    assert skipped.objects[0].label == "cigarette"
    assert "annotation_source_message_id" not in face_sample.objects[-1].attributes


@pytest.mark.parametrize(
    ("media_timestamp", "expected"), [(13.0, True), (13.001, False)],
)
def test_select_overlay_expires_missing_known_face_from_original_observation(
    media_timestamp: float, expected: bool,
) -> None:
    face_sample = _selected_features_result(
        timestamp=10.0, frame_index=100, message_id="face", stranger=False,
        known_person_id="employee-7",
    )
    skipped = _selected_features_result(
        timestamp=12.8, frame_index=142, message_id="safety", stranger=None,
    )

    selection = select_overlay(
        (face_sample, skipped), media_timestamp, max_gap_seconds=0.8,
    )

    assert selection.analysis is not None
    assert any(
        item.attributes.get("person_id") == "employee-7"
        for item in selection.analysis.objects
    ) is expected


@pytest.mark.parametrize("barrier", ["explicit-clear", "new-session"])
def test_select_overlay_does_not_revive_known_face_after_clear_or_new_session(
    barrier: str,
) -> None:
    face_sample = _selected_features_result(
        timestamp=10.0, frame_index=100, message_id="face", stranger=False,
        known_person_id="employee-7",
    )
    clear = _selected_features_result(
        timestamp=10.4, frame_index=106, message_id="clear",
        stranger=False if barrier == "explicit-clear" else None,
        session_id="session-1" if barrier == "explicit-clear" else "session-2",
    )
    skipped = _selected_features_result(
        timestamp=10.8, frame_index=112, message_id="safety", stranger=None,
        session_id=clear.stream_session_id,
    )

    selection = select_overlay(
        (face_sample, clear, skipped), 11.0, max_gap_seconds=0.8,
    )

    assert selection.analysis is not None
    assert selection.analysis.objects == ()


@pytest.mark.parametrize(
    "analysis_kind", ["face_recognition", "selected_features", "fixed_five_events"],
)
@pytest.mark.parametrize(
    ("right_person_id", "right_name", "expected_x1"),
    [("employee-7", "Renamed", 0.45), ("employee-8", "Same name", 0.4)],
)
def test_select_overlay_uses_person_id_to_interpolate_known_faces(
    analysis_kind: str, right_person_id: str, right_name: str, expected_x1: float,
) -> None:
    left = replace(_selected_features_result(
        timestamp=10.0, frame_index=100, message_id="left", stranger=False,
        known_person_id="employee-7",
    ), analysis_kind=analysis_kind)
    right = replace(
        left,
        message_id="right",
        frame=replace(left.frame, timestamp_seconds=10.4, source_frame_index=106),
        objects=(replace(
            left.objects[0],
            bbox=NormalizedBBox(0.5, 0.1, 0.6, 0.25),
            attributes={
                **left.objects[0].attributes,
                "person_id": right_person_id, "name": right_name,
            },
        ),),
    )

    selection = select_overlay((left, right), 10.2, max_gap_seconds=0.8)

    assert selection.analysis is not None
    assert selection.analysis.objects[0].bbox.x1 == pytest.approx(expected_x1)
    assert selection.analysis.objects[0].attributes["person_id"] == "employee-7"


@pytest.mark.parametrize("future_timestamp", [10.0, 10.4])
def test_select_overlay_does_not_inherit_future_stranger(
    future_timestamp: float,
) -> None:
    current = _selected_features_result(
        timestamp=10.0, frame_index=100, message_id="current", stranger=None,
    )
    future = _selected_features_result(
        timestamp=future_timestamp, frame_index=102, message_id="future",
        stranger=True,
    )

    selection = select_overlay(
        (current, future), 10.0, frame_sequence=101, max_gap_seconds=0.8,
    )

    assert selection.analysis is not None
    assert not any(
        item.event_label == "stranger" for item in selection.analysis.judgments
    )
    assert selection.analysis.objects == ()


def test_select_overlay_does_not_inherit_stranger_from_another_session() -> None:
    previous = _selected_features_result(
        timestamp=10.0, frame_index=100, message_id="previous", stranger=True,
        session_id="previous-session",
    )
    current = _selected_features_result(
        timestamp=10.4, frame_index=106, message_id="current", stranger=None,
        session_id="current-session",
    )

    selection = select_overlay((previous, current), 10.6, max_gap_seconds=0.8)

    assert selection.analysis is not None
    assert selection.analysis.stream_session_id == "current-session"
    assert not any(
        item.event_label == "stranger" for item in selection.analysis.judgments
    )
    assert selection.analysis.objects == ()


@pytest.mark.parametrize("next_person_id", ["alice", "bob"])
def test_joint_face_interpolation_keeps_person_identity(next_person_id: str) -> None:
    def known_result(timestamp: float, person_id: str, x: float) -> AnalysisResult:
        base = _result(
            timestamp=timestamp, frame_index=int(timestamp * 10),
            message_id=person_id, bbox=None, detected=False,
        )
        return replace(
            base, analysis_kind="fixed_five_events",
            objects=(GroundedObject(
                object_id="face-1", label="known_person",
                bbox=NormalizedBBox(x, 0.2, x + 0.2, 0.5),
                attributes={"status": "known", "person_id": person_id, "name": "Same"},
            ),),
        )

    left = known_result(0.0, "alice", 0.1)
    right = known_result(0.4, next_person_id, 0.2)
    selection = select_overlay((left, right), 0.2, max_gap_seconds=0.8)
    assert selection.analysis is not None
    face = selection.analysis.objects[0]
    assert face.attributes["person_id"] == "alice"
    assert face.bbox.x1 == pytest.approx(0.15 if next_person_id == "alice" else 0.1)


def test_select_overlay_interpolates_compatible_sparse_boxes() -> None:
    left = _result(
        timestamp=0.0,
        frame_index=0,
        message_id="left",
        bbox=(0.1, 0.1, 0.3, 0.3),
        object_id="object-1",
    )
    right = _result(
        timestamp=0.4,
        frame_index=6,
        message_id="right",
        bbox=(0.3, 0.2, 0.5, 0.4),
        object_id="object-99",
    )

    selection = select_overlay(
        (left, right),
        0.2,
        max_gap_seconds=0.8,
    )

    assert selection.mode == "interpolated"
    assert selection.left_message_id == "left"
    assert selection.right_message_id == "right"
    assert selection.analysis is not None
    bbox = selection.analysis.objects[0].bbox
    assert (bbox.x1, bbox.y1, bbox.x2, bbox.y2) == pytest.approx(
        (0.2, 0.15, 0.4, 0.35)
    )
    assert selection.analysis.objects[0].object_id == "object-1"


def test_select_overlay_changes_event_state_only_at_right_keyframe() -> None:
    positive = _result(
        timestamp=0.0,
        frame_index=0,
        message_id="positive",
        bbox=(0.1, 0.1, 0.3, 0.3),
    )
    clear = _result(
        timestamp=0.4,
        frame_index=6,
        message_id="clear",
        bbox=None,
        detected=False,
    )

    between = select_overlay((positive, clear), 0.2, max_gap_seconds=0.8)
    at_clear = select_overlay((positive, clear), 0.4, max_gap_seconds=0.8)

    assert between.analysis is not None
    assert between.analysis.judgments[0].detected is True
    assert len(between.analysis.objects) == 1
    assert at_clear.mode == "exact"
    assert at_clear.analysis is clear
    assert at_clear.analysis.judgments[0].detected is False


def test_select_overlay_latches_bbox_across_long_or_missing_result_gap() -> None:
    left = _result(
        timestamp=0.0,
        frame_index=0,
        message_id="left",
        bbox=(0.1, 0.1, 0.3, 0.3),
    )
    distant_right = _result(
        timestamp=2.0,
        frame_index=30,
        message_id="right",
        bbox=(0.5, 0.1, 0.7, 0.3),
    )

    within_long_gap = select_overlay(
        (left, distant_right),
        1.5,
        max_gap_seconds=0.8,
    )
    after_latest = select_overlay(
        (left,),
        10.0,
        max_gap_seconds=0.8,
    )

    assert within_long_gap.mode == "held"
    assert within_long_gap.analysis is left
    assert after_latest.mode == "held"
    assert after_latest.analysis is left


def test_delayed_composer_preserves_frame_order_and_reorders_results() -> None:
    published: list[FramePacket] = []
    rendered_boxes: list[tuple[float, float, float, float]] = []
    published_event = threading.Event()

    def render(jpeg: bytes, analysis: AnalysisResult) -> bytes:
        bbox = analysis.objects[0].bbox
        rendered_boxes.append((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        return jpeg + b"-rendered"

    def publish(packet: FramePacket) -> None:
        published.append(packet)
        if len(published) == 3:
            published_event.set()

    composer = DelayedOverlayComposer(
        delay_seconds=0.04,
        max_gap_seconds=0.2,
        output_fps=20.0,
        render=render,
        publish=publish,
    )
    right = _result(
        timestamp=0.1,
        frame_index=2,
        message_id="right",
        bbox=(0.3, 0.1, 0.5, 0.3),
        object_id="right-id",
    )
    left = _result(
        timestamp=0.0,
        frame_index=0,
        message_id="left",
        bbox=(0.1, 0.1, 0.3, 0.3),
        object_id="left-id",
    )
    try:
        assert composer.submit_result(right)
        assert composer.submit_result(left)
        composer.submit_frame(_packet(0, 0.0))
        composer.submit_frame(_packet(1, 0.05))
        composer.submit_frame(_packet(2, 0.1))

        assert published_event.wait(1.0)
        assert [item.frame_sequence for item in published] == [0, 1, 2]
        assert [item.annotation_mode for item in published] == [
            "exact",
            "interpolated",
            "exact",
        ]
        assert rendered_boxes[1] == pytest.approx((0.2, 0.1, 0.4, 0.3))
        assert all(
            item.stream_delay_seconds == pytest.approx(0.04)
            for item in published
        )
    finally:
        composer.close()


def test_delayed_composer_accepts_late_anchors_for_future_frames() -> None:
    published = threading.Event()
    composer = DelayedOverlayComposer(
        delay_seconds=0.01,
        max_gap_seconds=0.2,
        output_fps=15.0,
        render=lambda jpeg, analysis: jpeg,
        publish=lambda packet: published.set(),
    )
    try:
        composer.submit_frame(_packet(3, 0.3))
        assert published.wait(1.0)
        assert composer.submit_result(
            _result(
                timestamp=0.15,
                frame_index=1,
                message_id="useful-late",
                bbox=(0.1, 0.1, 0.3, 0.3),
            )
        )
        assert composer.submit_result(
            _result(
                timestamp=0.0,
                frame_index=0,
                message_id="stale",
                bbox=(0.1, 0.1, 0.3, 0.3),
            )
        )
        assert composer.stats()["late_results"] == 2
    finally:
        composer.close()


def test_delayed_composer_renders_latched_bbox_on_every_display_frame() -> None:
    published: list[FramePacket] = []
    published_event = threading.Event()

    def render(jpeg: bytes, analysis: AnalysisResult) -> bytes:
        assert analysis.objects
        return jpeg + b"-bbox"

    def publish(packet: FramePacket) -> None:
        published.append(packet)
        if len(published) == 8:
            published_event.set()

    composer = DelayedOverlayComposer(
        delay_seconds=0.01,
        max_gap_seconds=0.1,
        output_fps=15.0,
        render=render,
        publish=publish,
    )
    positive = _result(
        timestamp=0.0,
        frame_index=0,
        message_id="positive",
        bbox=(0.1, 0.1, 0.3, 0.3),
    )
    try:
        assert composer.submit_result(positive)
        for sequence in range(8):
            composer.submit_frame(_packet(sequence, sequence / 15.0))

        assert published_event.wait(1.0)
        assert len(published) == 8
        assert all(packet.jpeg.endswith(b"-bbox") for packet in published)
        assert [packet.annotation_mode for packet in published] == [
            "exact",
            "held",
            "held",
            "held",
            "held",
            "held",
            "held",
            "held",
        ]
    finally:
        composer.close()


def test_delayed_composer_enriches_frames_without_analysis() -> None:
    published: list[FramePacket] = []
    published_event = threading.Event()
    calls: list[tuple[bytes, float, int]] = []

    def enrich(jpeg: bytes, media_timestamp: float, frame_sequence: int) -> bytes:
        calls.append((jpeg, media_timestamp, frame_sequence))
        return jpeg + b"-enriched"

    def publish(packet: FramePacket) -> None:
        published.append(packet)
        if len(published) == 2:
            published_event.set()

    composer = DelayedOverlayComposer(
        delay_seconds=0.01,
        max_gap_seconds=0.2,
        output_fps=15.0,
        render=lambda jpeg, analysis: jpeg + b"-event",
        publish=publish,
        enrich=enrich,
    )
    try:
        composer.submit_frame(_packet(3, 0.2))
        composer.submit_frame(_packet(4, 0.3))

        assert published_event.wait(1.0)
        assert calls == [(b"jpeg", 0.2, 3), (b"jpeg", 0.3, 4)]
        assert [packet.jpeg for packet in published] == [
            b"jpeg-enriched",
            b"jpeg-enriched",
        ]
        assert all(packet.annotation_mode == "none" for packet in published)
    finally:
        composer.close()


def test_delayed_composer_renders_event_after_enrichment() -> None:
    published: list[FramePacket] = []
    published_event = threading.Event()
    call_order: list[str] = []

    def enrich(jpeg: bytes, media_timestamp: float, frame_sequence: int) -> bytes:
        call_order.append("enrich")
        return jpeg + b"-enriched"

    def render(jpeg: bytes, analysis: AnalysisResult) -> bytes:
        call_order.append("event")
        assert jpeg == b"jpeg-enriched"
        return jpeg + b"-event"

    def publish(packet: FramePacket) -> None:
        published.append(packet)
        published_event.set()

    composer = DelayedOverlayComposer(
        delay_seconds=0.01,
        max_gap_seconds=0.2,
        output_fps=15.0,
        render=render,
        publish=publish,
        enrich=enrich,
    )
    try:
        assert composer.submit_result(
            _result(
                timestamp=0.0,
                frame_index=0,
                message_id="event",
                bbox=(0.1, 0.1, 0.3, 0.3),
            )
        )
        composer.submit_frame(_packet(0, 0.0))

        assert published_event.wait(1.0)
        assert call_order == ["enrich", "event"]
        assert published[0].jpeg == b"jpeg-enriched-event"
        assert published[0].annotation_mode == "exact"
    finally:
        composer.close()


def test_delayed_composer_uses_unified_compose_with_display_time() -> None:
    published: list[FramePacket] = []
    published_event = threading.Event()
    calls: list[tuple[bytes, str, float, int, str | None, str | None]] = []

    def compose(
        jpeg: bytes,
        selection: OverlaySelection,
        media_timestamp: float,
        frame_sequence: int,
    ) -> bytes:
        calls.append(
            (
                jpeg,
                selection.mode,
                media_timestamp,
                frame_sequence,
                selection.left_message_id,
                selection.right_message_id,
            )
        )
        return jpeg + b"-composed"

    def publish(packet: FramePacket) -> None:
        published.append(packet)
        published_event.set()

    composer = DelayedOverlayComposer(
        delay_seconds=0.01,
        max_gap_seconds=0.2,
        output_fps=15.0,
        render=lambda jpeg, analysis: pytest.fail("legacy event renderer called"),
        publish=publish,
        enrich=lambda jpeg, timestamp, sequence: pytest.fail(
            "legacy enrichment called"
        ),
        compose=compose,
    )
    try:
        assert composer.submit_result(
            _result(
                timestamp=0.0,
                frame_index=0,
                message_id="left",
                bbox=(0.1, 0.1, 0.3, 0.3),
            )
        )
        assert composer.submit_result(
            _result(
                timestamp=0.1,
                frame_index=2,
                message_id="right",
                bbox=(0.2, 0.1, 0.4, 0.3),
            )
        )
        composer.submit_frame(_packet(1, 0.05))

        assert published_event.wait(1.0)
        assert calls == [(b"jpeg", "interpolated", 0.05, 1, "left", "right")]
        assert published[0].jpeg == b"jpeg-composed"
        assert published[0].annotation_mode == "interpolated"
    finally:
        composer.close()


def test_delayed_composer_composes_rich_only_frames_without_analysis() -> None:
    published: list[FramePacket] = []
    published_event = threading.Event()
    selections: list[OverlaySelection] = []

    def compose(
        jpeg: bytes,
        selection: OverlaySelection,
        media_timestamp: float,
        frame_sequence: int,
    ) -> bytes:
        assert media_timestamp == 0.2
        assert frame_sequence == 3
        selections.append(selection)
        return jpeg + b"-rich"

    composer = DelayedOverlayComposer(
        delay_seconds=0.01,
        max_gap_seconds=0.2,
        output_fps=15.0,
        render=lambda jpeg, analysis: jpeg,
        publish=lambda packet: (
            published.append(packet),
            published_event.set(),
        ),
        compose=compose,
    )
    try:
        composer.submit_frame(_packet(3, 0.2))

        assert published_event.wait(1.0)
        assert len(selections) == 1
        assert selections[0].analysis is None
        assert published[0].jpeg == b"jpeg-rich"
        assert published[0].annotation_mode == "none"
    finally:
        composer.close()


@pytest.mark.parametrize("with_analysis", (True, False))
def test_delayed_composer_falls_back_to_original_frame_when_compose_fails(
    with_analysis: bool,
) -> None:
    published: list[FramePacket] = []
    published_event = threading.Event()
    rendered_sources: list[bytes] = []

    def render_event(jpeg: bytes, analysis: AnalysisResult) -> bytes:
        assert analysis.message_id == "event"
        rendered_sources.append(jpeg)
        return jpeg + b"-event"

    def broken_compose(
        jpeg: bytes,
        selection: OverlaySelection,
        media_timestamp: float,
        frame_sequence: int,
    ) -> bytes:
        del jpeg, selection, media_timestamp, frame_sequence
        raise RuntimeError("fusion failed")

    composer = DelayedOverlayComposer(
        delay_seconds=0.01,
        max_gap_seconds=0.2,
        output_fps=15.0,
        render=render_event,
        publish=lambda packet: (
            published.append(packet),
            published_event.set(),
        ),
        enrich=lambda jpeg, timestamp, sequence: pytest.fail(
            "failed fusion must not enrich before drawing event boxes"
        ),
        compose=broken_compose,
    )
    try:
        if with_analysis:
            assert composer.submit_result(
                _result(
                    timestamp=0.0,
                    frame_index=0,
                    message_id="event",
                    bbox=(0.1, 0.1, 0.3, 0.3),
                )
            )
        composer.submit_frame(_packet(0, 0.0))

        assert published_event.wait(1.0)
        assert len(published) == 1
        assert rendered_sources == ([b"jpeg"] if with_analysis else [])
        assert published[0].jpeg == (b"jpeg-event" if with_analysis else b"jpeg")
        assert published[0].annotation_mode == ("exact" if with_analysis else "none")
        assert published[0].frame_sequence == 0
        assert published[0].media_timestamp == 0.0
        assert composer.stats()["frames_published"] == 1
        assert composer.stats()["composition_render_errors"] == 1

        composer.reset()
        assert composer.stats()["composition_render_errors"] == 0
    finally:
        composer.close()


def test_delayed_composer_isolates_enrichment_failures_and_resets_stats() -> None:
    published: list[FramePacket] = []
    published_event = threading.Event()

    def enrich(jpeg: bytes, media_timestamp: float, frame_sequence: int) -> bytes:
        if frame_sequence == 0:
            raise RuntimeError("broken enrichment")
        return jpeg + b"-enriched"

    def publish(packet: FramePacket) -> None:
        published.append(packet)
        if len(published) == 2:
            published_event.set()

    composer = DelayedOverlayComposer(
        delay_seconds=0.01,
        max_gap_seconds=0.2,
        output_fps=15.0,
        render=lambda jpeg, analysis: jpeg,
        publish=publish,
        enrich=enrich,
    )
    try:
        composer.submit_frame(_packet(0, 0.0))
        composer.submit_frame(_packet(1, 0.1))

        assert published_event.wait(1.0)
        assert [packet.jpeg for packet in published] == [b"jpeg", b"jpeg-enriched"]
        assert composer.stats()["enrichment_render_errors"] == 1

        composer.reset()
        assert composer.stats()["enrichment_render_errors"] == 0
    finally:
        composer.close()


def test_delayed_composer_remains_backward_compatible_without_enrichment() -> None:
    published: list[FramePacket] = []
    published_event = threading.Event()
    composer = DelayedOverlayComposer(
        delay_seconds=0.01,
        max_gap_seconds=0.2,
        output_fps=15.0,
        render=lambda jpeg, analysis: jpeg + b"-event",
        publish=lambda packet: (published.append(packet), published_event.set()),
    )
    try:
        composer.submit_frame(_packet(0, 0.0))

        assert published_event.wait(1.0)
        assert published[0].jpeg == b"jpeg"
        assert published[0].annotation_mode == "none"
        assert composer.stats()["enrichment_render_errors"] == 0
    finally:
        composer.close()


def test_delayed_composer_reset_does_not_publish_old_session_frames() -> None:
    published: list[FramePacket] = []
    composer = DelayedOverlayComposer(
        delay_seconds=0.05,
        max_gap_seconds=0.2,
        output_fps=15.0,
        render=lambda jpeg, analysis: jpeg,
        publish=published.append,
    )
    try:
        composer.submit_frame(_packet(0, 0.0))
        composer.reset()
        time.sleep(0.08)
        assert published == []
    finally:
        composer.close()
