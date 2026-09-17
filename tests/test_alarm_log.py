from __future__ import annotations

from goodvision_harness.alarm_log import AlarmLogger
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)


def _result(
    timestamp: float,
    *,
    detected: bool = True,
    event_label: str = "smoking",
    subject_id: str = "subject-cam-1",
    person_id: str | None = "alice",
    message_id: str | None = None,
) -> AnalysisResult:
    attributes = {"subject_id": subject_id, "subject_ids": [subject_id]}
    if person_id is not None:
        attributes["person_id"] = person_id
        attributes["name"] = person_id
    objects = (
        (GroundedObject("o1", "cigarette", NormalizedBBox(0.2, 0.2, 0.3, 0.3), 0.9, attributes),)
        if detected
        else ()
    )
    return AnalysisResult(
        message_id or f"msg-{timestamp}",
        "camera",
        "session",
        max(1, int(timestamp * 10)),
        "2026-09-16T14:14:03.210Z",
        FrameReference(max(0, int(timestamp * 10)), timestamp, "2026-09-16T14:14:03.210Z", 640, 480),
        (EventJudgment(event_label, detected, 0.91, "visible evidence", ("o1",) if detected else ()),),
        objects,
        "test",
        0.4,
    )


def test_rising_edge_emits_person_event_and_alarm_time_once() -> None:
    logger = AlarmLogger()
    first = logger.observe(_result(1.2))
    again = logger.observe(_result(1.6, message_id="msg-later"))

    assert len(first) == 1
    alarm = first[0]
    assert alarm.person_index == "subject-cam-1"
    assert alarm.person_id == "alice"
    assert alarm.event_label == "smoking"
    assert alarm.timestamp_seconds == 1.2
    assert alarm.observed_at == "2026-09-16T14:14:03.210Z"
    assert "subject-cam-1" in alarm.message
    assert "smoking" in alarm.message
    assert again == ()


def test_negative_clears_the_edge_so_the_next_positive_logs_again() -> None:
    logger = AlarmLogger()
    logger.observe(_result(1.0))
    logger.observe(_result(1.4, detected=False, message_id="neg"))
    second = logger.observe(_result(2.0, message_id="pos-2"))
    assert len(second) == 1
    assert second[0].event_label == "smoking"


def test_public_person_index_is_preferred_in_the_alarm_record() -> None:
    logger = AlarmLogger()
    result = _result(1.5, subject_id="subject-cam-1")
    object.__setattr__(
        result.objects[0],
        "attributes",
        {
            "subject_id": "subject-cam-1",
            "subject_ids": ["subject-cam-1"],
            "person_index": "P0007",
            "person_id": "alice",
        },
    )
    alarm = logger.observe(result)[0]
    assert alarm.person_index == "P0007"
    assert alarm.person_id == "alice"
    assert "P0007" in alarm.message


def test_timeline_result_logs_person_index_and_alarm_time(caplog) -> None:
    import logging

    from goodvision_harness.delayed_overlay import DelayedOverlayComposer
    from goodvision_harness.overlay_fusion import OverlayFusionMatcher
    from goodvision_harness.service import HarnessRuntime

    runtime = object.__new__(HarnessRuntime)
    runtime._rich_labels = None
    runtime._rich_requested = False
    runtime._overlay_fusion = OverlayFusionMatcher()
    runtime._alarm_logger = AlarmLogger()
    runtime._delayed_overlay = DelayedOverlayComposer(
        delay_seconds=0, max_gap_seconds=0.8, output_fps=30,
        render=lambda jpeg, analysis: jpeg, publish=lambda packet: None,
    )
    result = _result(2.5, subject_id="subject-cam-1")
    object.__setattr__(
        result.objects[0],
        "attributes",
        {
            "subject_id": "subject-cam-1",
            "subject_ids": ["subject-cam-1"],
            "person_index": "P0003",
            "person_id": "alice",
        },
    )
    with caplog.at_level(logging.INFO, logger="goodvision_harness.service"):
        runtime._on_timeline_result(result)
    assert runtime._alarm_logger.events()[0].person_index == "P0003"
    assert "P0003" in caplog.text
    assert "smoking" in caplog.text
    assert "2.500" in caplog.text


def test_unassigned_positive_still_logs_an_alarm() -> None:
    logger = AlarmLogger()
    result = _result(3.0, person_id=None, subject_id="subject-cam-9")
    object.__setattr__(
        result.objects[0],
        "attributes",
        {"subject_id": "subject-cam-9", "subject_ids": ["subject-cam-9"]},
    )
    alarms = logger.observe(result)
    assert alarms[0].person_index == "subject-cam-9"


def test_compact_alert_without_boxes_still_logs_an_alarm() -> None:
    logger = AlarmLogger()
    result = AnalysisResult(
        "msg-alert",
        "camera",
        "session",
        1,
        "2026-09-16T14:14:03.210Z",
        FrameReference(1, 1.2, "2026-09-16T14:14:03.210Z", 640, 480),
        (EventJudgment("smoking", True, 0.82, "吸烟。 / Smoking.", ()),),
        (),
        "test",
        0.4,
    )
    alarms = logger.observe(result)
    assert len(alarms) == 1
    assert alarms[0].person_index == "unassigned"
    assert alarms[0].event_label == "smoking"


def test_ungrounded_alert_upgrades_to_the_person_without_a_duplicate_edge() -> None:
    logger = AlarmLogger()
    compact = AnalysisResult(
        "msg-alert",
        "camera",
        "session",
        1,
        "2026-09-16T14:14:03.210Z",
        FrameReference(1, 1.2, "2026-09-16T14:14:03.210Z", 640, 480),
        (EventJudgment("smoking", True, 0.82, "吸烟。 / Smoking.", ()),),
        (),
        "test",
        0.4,
    )
    first = logger.observe(compact)
    upgraded = logger.observe(_result(1.3))
    assert first[0].person_index == "unassigned"
    assert len(upgraded) == 1
    assert upgraded[0].person_index == "subject-cam-1"
    assert logger.observe(_result(1.5, message_id="msg-hold")) == ()
