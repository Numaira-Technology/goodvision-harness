from __future__ import annotations

import threading
import time

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


def test_delayed_composer_falls_back_to_event_when_compose_fails() -> None:
    published: list[FramePacket] = []
    published_event = threading.Event()

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
        render=lambda jpeg, analysis: jpeg + b"-event",
        publish=lambda packet: (
            published.append(packet),
            published_event.set(),
        ),
        enrich=lambda jpeg, timestamp, sequence: jpeg + b"-rich",
        compose=broken_compose,
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
        assert published[0].jpeg == b"jpeg-rich-event"
        assert published[0].annotation_mode == "exact"
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
