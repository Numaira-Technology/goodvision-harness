from __future__ import annotations

from dataclasses import replace

import pytest

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.overlay_fusion import OverlayFusionMatcher
from goodvision_harness.rich_labeling import RichLabelObject


def _rich(
    track_id: str,
    bbox: tuple[float, float, float, float],
    *,
    label: str = "person",
) -> RichLabelObject:
    return RichLabelObject(
        track_id=track_id,
        label=label,
        bbox=NormalizedBBox(*bbox),
        confidence=0.9,
    )


def _analysis(
    *,
    bbox: tuple[float, float, float, float] = (0.2, 0.1, 0.5, 0.8),
    object_label: str = "person",
    alerts: tuple[tuple[str, float], ...] = (("fight", 0.91),),
    detected: bool = True,
    message_id: str = "event-keyframe-1",
    session_id: str = "session-1",
    object_id: str = "event-object-1",
) -> AnalysisResult:
    objects = (
        GroundedObject(
            object_id=object_id,
            label=object_label,
            bbox=NormalizedBBox(*bbox),
            confidence=0.88,
        ),
    )
    judgments = tuple(
        EventJudgment(
            event_label=event_label,
            detected=detected,
            confidence=confidence,
            description=(
                f"Detected {event_label}."
                if detected
                else f"No {event_label} detected."
            ),
            object_ids=(object_id,) if detected else (),
        )
        for event_label, confidence in alerts
    )
    return AnalysisResult(
        message_id=message_id,
        source_id="camera",
        stream_session_id=session_id,
        sequence=1,
        emitted_at="2026-08-30T00:00:01.000Z",
        frame=FrameReference(
            source_frame_index=1,
            timestamp_seconds=1.0,
            observed_at="2026-08-30T00:00:00.000Z",
            width=1280,
            height=720,
        ),
        judgments=judgments,
        objects=objects,
        model_name="model",
        model_latency_seconds=0.1,
    )


def _semantic_analysis() -> AnalysisResult:
    objects = (
        GroundedObject(
            object_id="fight-person",
            label="person",
            bbox=NormalizedBBox(0.381, 0.344, 0.906, 1.0),
            confidence=0.98,
        ),
        GroundedObject(
            object_id="smoking-person",
            label="person",
            bbox=NormalizedBBox(0.381, 0.344, 0.906, 1.0),
            confidence=0.98,
        ),
        GroundedObject(
            object_id="cigarette",
            label="cigarette",
            bbox=NormalizedBBox(0.781, 0.644, 0.862, 0.738),
            confidence=0.92,
        ),
        GroundedObject(
            object_id="fall-person",
            label="person",
            bbox=NormalizedBBox(0.39, 0.35, 0.9, 1.0),
            confidence=0.93,
        ),
        GroundedObject(
            object_id="unknown-face",
            label="unknown_person",
            bbox=NormalizedBBox(0.615527, 0.434649, 0.775671, 0.709401),
            confidence=0.875,
        ),
    )
    judgments = (
        EventJudgment(
            event_label="fight",
            detected=True,
            confidence=0.91,
            description="Fight detected.",
            object_ids=("fight-person",),
        ),
        EventJudgment(
            event_label="smoking",
            detected=True,
            confidence=0.95,
            description="Smoking detected.",
            object_ids=("smoking-person", "cigarette"),
        ),
        EventJudgment(
            event_label="fall",
            detected=True,
            confidence=0.86,
            description="Fall detected.",
            object_ids=("fall-person",),
        ),
        EventJudgment(
            event_label="stranger",
            detected=True,
            confidence=0.875,
            description="Unknown face detected.",
            object_ids=("unknown-face",),
        ),
    )
    return AnalysisResult(
        message_id="event-keyframe-semantic",
        source_id="camera",
        stream_session_id="session-1",
        sequence=1,
        emitted_at="2026-08-30T00:00:01.000Z",
        frame=FrameReference(
            source_frame_index=1,
            timestamp_seconds=1.0,
            observed_at="2026-08-30T00:00:00.000Z",
            width=640,
            height=480,
        ),
        judgments=judgments,
        objects=objects,
        model_name="model",
        model_latency_seconds=0.1,
        analysis_kind="selected_features",
    )


def test_clear_event_produces_no_track_alerts_or_fallback_boxes() -> None:
    plan = OverlayFusionMatcher().plan(
        _analysis(detected=False),
        (_rich("P0001", (0.2, 0.1, 0.5, 0.8)),),
    )

    assert plan.track_alerts == ()
    assert plan.matched_event_object_ids == ()
    assert plan.unmatched_event_object_ids == ()


def test_high_overlap_person_alias_matches_rich_track() -> None:
    plan = OverlayFusionMatcher().plan(
        _analysis(object_label="fighter"),
        (_rich("P0001", (0.21, 0.11, 0.49, 0.79)),),
    )

    assert plan.matched_event_object_ids == ("event-object-1",)
    assert plan.unmatched_event_object_ids == ()
    assert tuple(plan.alerts_by_track_id()) == ("P0001",)
    assert tuple(
        (alert.event_label, alert.confidence)
        for alert in plan.alerts_by_track_id()["P0001"]
    ) == (("fight", 0.91),)


@pytest.mark.parametrize(
    ("analysis", "rich_object"),
    (
        (
            _analysis(
                object_label="cigarette",
                alerts=(("smoking", 0.91),),
            ),
            _rich("P0001", (0.2, 0.1, 0.5, 0.8)),
        ),
        (
            _analysis(object_label="person"),
            _rich("P0001", (0.65, 0.1, 0.9, 0.8)),
        ),
    ),
    ids=("incompatible-label", "incompatible-position"),
)
def test_incompatible_event_object_stays_as_fallback(
    analysis: AnalysisResult,
    rich_object: RichLabelObject,
) -> None:
    plan = OverlayFusionMatcher().plan(analysis, (rich_object,))

    assert plan.track_alerts == ()
    assert plan.matched_event_object_ids == ()
    assert plan.unmatched_event_object_ids == ("event-object-1",)


def test_two_equally_close_people_are_ambiguous_and_stay_as_fallback() -> None:
    plan = OverlayFusionMatcher().plan(
        _analysis(bbox=(0.25, 0.1, 0.55, 0.8)),
        (
            _rich("P0001", (0.2, 0.1, 0.5, 0.8)),
            _rich("P0002", (0.3, 0.1, 0.6, 0.8)),
        ),
    )

    assert plan.track_alerts == ()
    assert plan.matched_event_object_ids == ()
    assert plan.unmatched_event_object_ids == ("event-object-1",)


def test_multiple_alerts_for_one_event_object_share_one_rich_track() -> None:
    plan = OverlayFusionMatcher().plan(
        _analysis(alerts=(("fight", 0.91), ("fall", 0.84))),
        (_rich("P0001", (0.2, 0.1, 0.5, 0.8)),),
    )

    assert plan.matched_event_object_ids == ("event-object-1",)
    assert plan.unmatched_event_object_ids == ()
    assert tuple(
        (alert.event_label, alert.confidence)
        for alert in plan.alerts_by_track_id()["P0001"]
    ) == (("fight", 0.91), ("fall", 0.84))


def test_cached_association_follows_p0001_after_it_moves() -> None:
    matcher = OverlayFusionMatcher()
    analysis = _analysis()
    first = matcher.plan(
        analysis,
        (_rich("P0001", (0.2, 0.1, 0.5, 0.8)),),
    )
    moved = matcher.plan(
        analysis,
        (
            _rich("P0001", (0.6, 0.1, 0.9, 0.8)),
            _rich("P0002", (0.2, 0.1, 0.5, 0.8)),
        ),
    )

    assert tuple(first.alerts_by_track_id()) == ("P0001",)
    assert tuple(moved.alerts_by_track_id()) == ("P0001",)
    assert moved.matched_event_object_ids == ("event-object-1",)
    assert moved.unmatched_event_object_ids == ()


def test_reset_discards_cached_track_association() -> None:
    matcher = OverlayFusionMatcher()
    analysis = _analysis()
    matcher.plan(
        analysis,
        (_rich("P0001", (0.2, 0.1, 0.5, 0.8)),),
    )

    matcher.reset()
    plan = matcher.plan(
        analysis,
        (
            _rich("P0001", (0.6, 0.1, 0.9, 0.8)),
            _rich("P0002", (0.2, 0.1, 0.5, 0.8)),
        ),
    )

    assert tuple(plan.alerts_by_track_id()) == ("P0002",)
    assert plan.matched_event_object_ids == ("event-object-1",)
    assert plan.unmatched_event_object_ids == ()


def test_semantic_routing_aggregates_person_events_and_targets_smoking_item() -> None:
    plan = OverlayFusionMatcher().plan(
        _semantic_analysis(),
        (
            _rich("P0001", (0.37, 0.33, 0.92, 1.0)),
            _rich(
                "O0002",
                (0.75, 0.60, 0.89, 0.78),
                label="cell phone",
            ),
        ),
    )

    assert plan.matched_event_object_ids == (
        "fight-person",
        "cigarette",
        "fall-person",
        "unknown-face",
    )
    assert plan.unmatched_event_object_ids == ()
    assert "smoking-person" not in plan.matched_event_object_ids
    assert tuple(
        (alert.event_label, alert.confidence)
        for alert in plan.alerts_by_track_id()["P0001"]
    ) == (("fight", 0.91), ("fall", 0.86), ("stranger", 0.875))
    assert tuple(
        (alert.event_label, alert.confidence)
        for alert in plan.alerts_by_track_id()["O0002"]
    ) == (("smoking", 0.95),)


def test_smoking_person_evidence_is_not_used_as_a_person_alert_or_fallback() -> None:
    plan = OverlayFusionMatcher().plan(
        _analysis(
            object_label="person",
            alerts=(("smoking", 0.91),),
        ),
        (_rich("P0001", (0.2, 0.1, 0.5, 0.8)),),
    )

    assert plan.track_alerts == ()
    assert plan.matched_event_object_ids == ()
    assert plan.unmatched_event_object_ids == ()
    assert plan.alerts_by_unmatched_object_id() == {}


def test_unmatched_smoking_item_retains_alert_for_rich_style_fallback() -> None:
    plan = OverlayFusionMatcher().plan(
        _analysis(
            bbox=(0.70, 0.55, 0.79, 0.65),
            object_label="cigarette",
            alerts=(("smoking", 0.95),),
            object_id="cigarette",
        ),
        (_rich("P0001", (0.2, 0.1, 0.5, 0.8)),),
    )

    assert plan.matched_event_object_ids == ()
    assert plan.unmatched_event_object_ids == ("cigarette",)
    assert tuple(
        (alert.event_label, alert.confidence)
        for alert in plan.alerts_by_unmatched_object_id()["cigarette"]
    ) == (("smoking", 0.95),)


def test_same_event_keeps_one_object_per_track_and_preserves_other_fallback() -> None:
    base = _analysis()
    primary = replace(base.objects[0], object_id="fighter-1")
    secondary = replace(
        base.objects[0],
        object_id="fighter-2",
        bbox=NormalizedBBox(0.38, 0.1, 0.68, 0.8),
    )
    analysis = replace(
        base,
        objects=(primary, secondary),
        judgments=(
            replace(
                base.judgments[0],
                object_ids=("fighter-1", "fighter-2"),
            ),
        ),
    )

    plan = OverlayFusionMatcher().plan(
        analysis,
        (_rich("P0001", (0.2, 0.1, 0.5, 0.8)),),
    )

    assert plan.matched_event_object_ids == ("fighter-1",)
    assert plan.unmatched_event_object_ids == ("fighter-2",)
    assert tuple(plan.alerts_by_track_id()) == ("P0001",)
    assert tuple(plan.alerts_by_unmatched_object_id()) == ("fighter-2",)


def test_group_person_box_is_kept_as_fallback_instead_of_claiming_one_track() -> None:
    plan = OverlayFusionMatcher().plan(
        _analysis(object_label="people"),
        (_rich("P0001", (0.2, 0.1, 0.5, 0.8)),),
    )

    assert plan.track_alerts == ()
    assert plan.matched_event_object_ids == ()
    assert plan.unmatched_event_object_ids == ("event-object-1",)
