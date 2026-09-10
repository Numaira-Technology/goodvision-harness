from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from goodvision_harness.config import load_config
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.delayed_overlay import DelayedOverlayComposer, select_overlay
from goodvision_harness.service import (
    HarnessRuntime,
    _analysis_overlay_arguments,
    _render_analysis_jpeg,
)


def _result(
    timestamp: float,
    *,
    smoking: bool | None = None,
    fall: bool | None = None,
    face: bool = True,
    stranger: bool = False,
    session: str = "session-1",
    message: str | None = None,
) -> AnalysisResult:
    objects = []
    judgments = []
    missing = []
    for label, detected in (("smoking", smoking), ("fall", fall)):
        if detected is None:
            missing.append(label)
            continue
        object_id = f"{label}-1"
        if detected:
            objects.append(GroundedObject(
                object_id, "cigarette" if label == "smoking" else "person",
                NormalizedBBox(0.6, 0.4, 0.8, 0.8), 0.9,
            ))
        judgments.append(EventJudgment(
            label, detected, 0.9, f"{label}: {detected}",
            (object_id,) if detected else (),
        ))
    if face:
        # Deliberately reuse an ID from the safety branch: IDs are frame-local.
        objects.append(GroundedObject(
            "smoking-1", "known_person", NormalizedBBox(0.1, 0.2, 0.3, 0.7),
            0.95, {"status": "known", "person_id": "employee-7"},
        ))
        if smoking:
            objects[-1] = replace(objects[-1], object_id="known-1")
        if stranger:
            objects.append(GroundedObject(
                "unknown-1", "unknown_person", NormalizedBBox(0.35, 0.2, 0.5, 0.7),
                0.9, {"status": "unknown"},
            ))
        judgments.append(EventJudgment(
            "stranger", stranger, 0.9, "Face observation",
            ("unknown-1",) if stranger else (),
        ))
    else:
        missing.append("stranger")
    return AnalysisResult(
        message_id=message or f"frame-{timestamp}", source_id="camera",
        stream_session_id=session, sequence=int(timestamp * 10) + 1,
        emitted_at="2026-09-10T09:00:00Z",
        frame=FrameReference(
            int(timestamp * 10), timestamp, "2026-09-10T09:00:00Z", 320, 240,
        ),
        judgments=tuple(judgments), objects=tuple(objects),
        model_name="model", model_latency_seconds=1.0,
        analysis_kind="selected_features", missing_event_labels=tuple(missing),
    )


def _active(analysis) -> set[str]:
    return {item.event_label for item in analysis.judgments if item.detected}


@pytest.mark.parametrize("timestamp", [10.4, 10.6, 14.0, 30.0])
@pytest.mark.parametrize("stranger", [False, True])
def test_face_updates_keep_event_red_until_its_explicit_clear(timestamp, stranger):
    positive = _result(10.0, smoking=True)
    face = _result(10.4, stranger=stranger)
    later_face = _result(14.0, stranger=stranger)
    clear = _result(30.4, smoking=False, stranger=stranger)
    original = face.as_dict()
    records = (positive, face, later_face, clear)

    selection = select_overlay(
        records, timestamp, max_gap_seconds=0.8, clear_after_negatives=1,
    )
    assert selection.analysis is not None
    arguments = _analysis_overlay_arguments(selection.analysis)
    assert "smoking" in _active(selection.analysis)
    assert not arguments["known_faces"]
    assert len(arguments["objects"]) == 1 + int(stranger)
    event = next(item for item in arguments["objects"] if "smoking" in item.label)
    assert event.bbox == positive.objects[0].bbox
    assert event.attributes["annotation_source_message_id"] == positive.message_id
    assert len({item.object_id for item in selection.analysis.objects}) == len(
        selection.analysis.objects
    )
    assert face.as_dict() == original
    assert selection.analysis.as_dict() == (
        face.as_dict() if timestamp < 14.0 else later_face.as_dict()
    )

    for at in (30.4, 31.0):
        after = select_overlay(
            records, at, max_gap_seconds=0.8, clear_after_negatives=1,
        ).analysis
        assert "smoking" not in _active(after)
        assert len(_analysis_overlay_arguments(after)["known_faces"]) == 1


def test_each_event_clears_independently_without_reviving_older_positives():
    records = (
        _result(10, smoking=True, fall=True),
        _result(11, smoking=False),
        _result(12),
        _result(13, fall=False),
        _result(14),
    )
    held = select_overlay(
        records, 12, max_gap_seconds=0.8, clear_after_negatives=1,
    ).analysis
    assert _active(held) == {"fall"}
    clear = select_overlay(
        records, 14, max_gap_seconds=0.8, clear_after_negatives=1,
    ).analysis
    assert _active(clear) == set()
    assert _analysis_overlay_arguments(clear)["known_faces"]


@pytest.mark.parametrize("barrier", ["future", "session"])
def test_event_carry_does_not_cross_session_or_use_future_results(barrier):
    positive = _result(10 if barrier == "session" else 12, smoking=True)
    face = _result(11, session="session-2" if barrier == "session" else "session-1")
    records = sorted((positive, face), key=lambda item: item.frame.timestamp_seconds)
    assert not _active(select_overlay(
        records, 11, max_gap_seconds=0.8, clear_after_negatives=1,
    ).analysis)


def _composer() -> DelayedOverlayComposer:
    return DelayedOverlayComposer(
        delay_seconds=0, max_gap_seconds=0.8, clear_after_negatives=1, output_fps=15,
        render=lambda jpeg, _analysis: jpeg, publish=lambda _packet: None,
    )


@pytest.mark.parametrize("face_first", [False, True])
def test_same_frame_independent_branches_preserve_event_in_either_arrival_order(
    face_first,
):
    positive = _result(10, smoking=True, face=False, message="safety")
    face = _result(10, message="face")
    records = (face, positive) if face_first else (positive, face)
    composer = _composer()
    try:
        for record in records:
            composer.submit_result(record)
        selection = select_overlay(
            composer._results, 10, max_gap_seconds=0.8, clear_after_negatives=1,
        )
        assert _active(selection.analysis) == {"smoking"}
        assert not _analysis_overlay_arguments(selection.analysis)["known_faces"]
        # A correction of this exact observation replaces its prior verdict.
        composer.submit_result(_result(10, smoking=False, face=False, message="safety"))
        selection = select_overlay(
            composer._results, 10, max_gap_seconds=0.8, clear_after_negatives=1,
        )
        assert _active(selection.analysis) == set()
        assert _analysis_overlay_arguments(selection.analysis)["known_faces"]
        assert len(composer._results) == 2
    finally:
        composer.close()


def test_pruning_keeps_event_until_clear_and_clear_blocks_late_old_positive():
    composer = _composer()
    try:
        composer.submit_result(_result(0, smoking=True))
        for timestamp in range(1, 31):
            composer.submit_result(_result(timestamp))
            composer._prune_results_locked(timestamp)
            selection = select_overlay(
                composer._results, timestamp, max_gap_seconds=0.8,
                clear_after_negatives=1,
            )
            assert _active(selection.analysis) == {"smoking"}
            assert len(composer._results) <= 7
        composer.submit_result(_result(31, smoking=False))
        for timestamp in range(32, 41):
            composer.submit_result(_result(timestamp))
            composer._prune_results_locked(timestamp)
        composer.submit_result(_result(20, smoking=True, message="late-positive"))
        after = select_overlay(
            composer._results, 40, max_gap_seconds=0.8, clear_after_negatives=1,
        ).analysis
        assert _active(after) == set()
        composer.reset()
        assert not composer._results
    finally:
        composer.close()


@pytest.mark.parametrize("fused", [False, True])
def test_video_stays_red_on_face_updates_and_returns_green_on_clear(tmp_path, fused):
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    runtime = HarnessRuntime(replace(
        config, face=replace(config.face, gallery_path=tmp_path / "faces.json"),
    ))
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    records = (
        _result(10, smoking=True), _result(11), _result(15),
        _result(16, smoking=False),
    )
    try:
        for timestamp in (10, 11, 15, 16):
            selection = select_overlay(
                records, timestamp, max_gap_seconds=0.8, clear_after_negatives=1,
            )
            jpeg = (
                runtime._render_fused_label_jpeg(
                    encoded.tobytes(), selection, timestamp, int(timestamp * 10),
                ) if fused else _render_analysis_jpeg(
                    encoded.tobytes(), selection.analysis, quality=95,
                )
            )
            pixels = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            blue, green, red = map(int, pixels[189, 192])
            assert (red > 150 and green < 110 and blue < 110) is (timestamp < 16)
            blue, green, red = map(int, pixels[150, 32])
            assert (green > 130 and red < 110 and blue < 110) is (timestamp == 16)
    finally:
        runtime.stop()


def test_negative_frames_hold_status_for_ten_observations_without_old_boxes():
    positive = _result(0, smoking=True, face=False)
    records = [positive]
    for at in range(1, 11):
        negative = _result(at, smoking=False, face=False)
        original = negative.as_dict()
        records.append(negative)
        analysis = select_overlay(records, at, max_gap_seconds=0.8).analysis
        arguments = _analysis_overlay_arguments(analysis)
        assert arguments["detected"] is (at < 10)
        assert arguments["objects"] == ()
        assert not _active(analysis)  # per-frame verdict remains negative
        assert analysis.as_dict() == original
        if at < 10:
            assert f"{at}/10" in arguments["description"]


def test_no_face_positive_resets_streak_and_other_missing_events_do_not_count():
    records = [_result(0, smoking=True, fall=True, face=False)]
    for at in range(1, 10):
        records.append(_result(at, smoking=False, face=False))
    records.extend((
        _result(10, fall=False, face=False),
        _result(11, smoking=True, face=False),
    ))
    for at in range(12, 22):
        records.append(_result(at, smoking=False, face=False))
        analysis = select_overlay(records, at, max_gap_seconds=0.8).analysis
        arguments = _analysis_overlay_arguments(analysis)
        assert ("smoking" in arguments["event_label"]) is (at < 21)
        assert "fall" in arguments["event_label"]


def test_no_face_replays_count_once_and_ten_newer_negatives_beat_late_positive():
    records = []
    for at in range(1, 11):
        negative = _result(at, smoking=False, face=False)
        records.extend((negative, replace(negative, message_id=f"replay-{at}")))
    records.insert(0, _result(0, smoking=True, face=False, message="late-positive"))
    at_nine = select_overlay(records, 9, max_gap_seconds=0.8).analysis
    assert _analysis_overlay_arguments(at_nine)["detected"]
    at_ten = select_overlay(records, 10, max_gap_seconds=0.8).analysis
    assert not _analysis_overlay_arguments(at_ten)["detected"]


def test_compositor_pruning_retains_full_negative_streak_and_blocks_late_positive():
    composer = DelayedOverlayComposer(
        delay_seconds=0, max_gap_seconds=0.8, output_fps=15,
        render=lambda jpeg, _analysis: jpeg, publish=lambda _packet: None,
    )
    try:
        composer.submit_result(_result(0, smoking=True, face=False))
        for at in range(1, 41):
            composer.submit_result(_result(at, smoking=False, face=False))
            composer._prune_results_locked(at)
            analysis = select_overlay(
                composer._results, at, max_gap_seconds=0.8,
            ).analysis
            assert _analysis_overlay_arguments(analysis)["detected"] is (at < 10)
        assert len(composer._results) <= 15
        composer.submit_result(_result(20, smoking=True, face=False, message="late"))
        analysis = select_overlay(composer._results, 40, max_gap_seconds=0.8).analysis
        assert not _analysis_overlay_arguments(analysis)["detected"]
    finally:
        composer.close()
