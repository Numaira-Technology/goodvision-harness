from __future__ import annotations

import builtins
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from workers.paddle_rich_label import gpu_smoke
from workers.paddle_rich_label.app import (
    _PEDESTRIAN_ATTRIBUTE_LABELS,
    Detection,
    IoUTracker,
    PaddleRichModels,
    RichLabelBusyError,
    RichLabelEngine,
    _model_kwargs,
    _pedestrian_attributes,
    create_app,
)

ROOT = Path(__file__).resolve().parents[1]


class _Models:
    ready = True

    def detect(self, image: np.ndarray[Any, Any]) -> tuple[Detection, ...]:
        return (
            Detection("person", 0.94, (10.0, 10.0, 70.0, 95.0)),
            Detection("bottle", 0.82, (80.0, 25.0, 105.0, 85.0)),
        )

    def attributes(
        self, crops: list[np.ndarray[Any, Any]]
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "estimated_age_range": "18-60",
                "estimated_gender": "female-presenting",
            }
            for _ in crops
        )


def test_cpu_models_keep_paddlex_optimized_defaults() -> None:
    assert _model_kwargs("PicoDet-S", "", "cpu") == {
        "model_name": "PicoDet-S",
        "device": "cpu",
    }


def test_non_cpu_models_keep_paddlex_device_defaults() -> None:
    assert _model_kwargs("PP-YOLOE_plus-S", "/models/detector", "gpu:0") == {
        "model_name": "PP-YOLOE_plus-S",
        "device": "gpu:0",
        "model_dir": "/models/detector",
    }


def _jpeg(width: int = 120, height: int = 100) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _attribute_payload(
    scores: list[float] | None = None,
) -> dict[str, list[Any]]:
    return {
        "class_ids": list(range(26)),
        "label_names": list(_PEDESTRIAN_ATTRIBUTE_LABELS),
        "scores": list(scores if scores is not None else [0.0] * 26),
    }


def test_iou_tracker_keeps_ids_and_resets_cleanly() -> None:
    tracker = IoUTracker(match_iou=0.2, max_idle_frames=4)
    first = tracker.update(
        (Detection("person", 0.9, (10.0, 10.0, 50.0, 80.0)),),
        frame_sequence=1,
    )
    second = tracker.update(
        (Detection("person", 0.92, (13.0, 10.0, 53.0, 80.0)),),
        frame_sequence=2,
    )

    assert first[0].track_id == second[0].track_id == "P0001"
    assert tracker._update_tick == 2

    tracker.reset()
    reset = tracker.update(
        (Detection("bottle", 0.8, (1.0, 1.0, 10.0, 20.0)),),
        frame_sequence=3,
    )
    assert reset[0].track_id == "O0001"
    assert tracker._update_tick == 1


def test_iou_tracker_keeps_id_during_fast_non_overlapping_motion() -> None:
    tracker = IoUTracker(match_iou=0.3, max_idle_frames=5)
    first = tracker.update(
        (Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0)),),
        frame_sequence=0,
    )
    second = tracker.update(
        (Detection("person", 0.91, (18.0, 0.0, 38.0, 40.0)),),
        frame_sequence=1,
    )
    third = tracker.update(
        (Detection("person", 0.92, (37.0, 0.0, 57.0, 40.0)),),
        frame_sequence=2,
    )

    assert first[0].track_id == second[0].track_id == third[0].track_id
    assert tracker._next_id == 2


def test_iou_tracker_recovers_same_id_after_short_missed_detection() -> None:
    tracker = IoUTracker(match_iou=0.3, max_idle_frames=5)
    first = tracker.update(
        (Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0)),),
        frame_sequence=0,
    )
    moving = tracker.update(
        (Detection("person", 0.91, (12.0, 0.0, 32.0, 40.0)),),
        frame_sequence=1,
    )
    missed = tracker.update((), frame_sequence=2)
    cleared = tracker.update((), frame_sequence=3)
    recovered = tracker.update(
        (Detection("person", 0.92, (48.0, 0.0, 68.0, 40.0)),),
        frame_sequence=4,
    )

    assert missed[0].track_id == moving[0].track_id
    assert missed[0].bbox[0] > moving[0].bbox[0]
    assert cleared == ()
    assert first[0].track_id == moving[0].track_id == recovered[0].track_id
    assert tracker._next_id == 2


def test_iou_tracker_emits_motion_predicted_bbox_while_occluded() -> None:
    tracker = IoUTracker(match_iou=0.3, max_idle_frames=5)
    tracker.update(
        (Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0)),),
        frame_sequence=0,
    )
    moving = tracker.update(
        (Detection("person", 0.91, (12.0, 0.0, 32.0, 40.0)),),
        frame_sequence=1,
    )
    occluded = tracker.update((), frame_sequence=2)
    assert occluded[0].track_id == moving[0].track_id
    assert occluded[0].bbox[0] == pytest.approx(24.0)
    assert occluded[0].bbox[2] == pytest.approx(44.0)


def test_iou_tracker_holds_only_the_track_missing_from_partial_detection() -> None:
    tracker = IoUTracker(match_iou=0.3, max_idle_frames=5)
    initial = tracker.update(
        (
            Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0)),
            Detection("bottle", 0.8, (60.0, 0.0, 80.0, 40.0)),
        ),
        frame_sequence=0,
    )
    person = next(track for track in initial if track.label == "person")
    bottle = next(track for track in initial if track.label == "bottle")

    partial = tracker.update(
        (Detection("person", 0.92, (4.0, 0.0, 24.0, 40.0)),),
        frame_sequence=1,
    )

    observed_person = next(track for track in partial if track.label == "person")
    held_bottle = next(track for track in partial if track.label == "bottle")
    assert observed_person.track_id == person.track_id
    assert observed_person.bbox == (4.0, 0.0, 24.0, 40.0)
    assert held_bottle.track_id == bottle.track_id
    assert held_bottle.bbox == bottle.bbox
    assert held_bottle.last_update_tick < tracker._update_tick


def test_iou_tracker_idle_ttl_counts_detector_ticks_not_source_frames() -> None:
    tracker = IoUTracker(match_iou=0.3, max_idle_frames=2)
    first = tracker.update(
        (Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0)),),
        frame_sequence=0,
    )
    second = tracker.update(
        (Detection("person", 0.91, (3.0, 0.0, 23.0, 40.0)),),
        frame_sequence=30,
    )
    third = tracker.update(
        (Detection("person", 0.92, (6.0, 0.0, 26.0, 40.0)),),
        frame_sequence=60,
    )

    assert first[0].track_id == second[0].track_id == third[0].track_id
    assert tracker._next_id == 2


def test_iou_tracker_allocates_new_id_after_detector_tick_ttl() -> None:
    tracker = IoUTracker(
        match_iou=0.3,
        max_idle_frames=2,
        visible_idle_frames=0,
    )
    first = tracker.update(
        (Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0)),),
        frame_sequence=0,
    )
    assert tracker.update((), frame_sequence=30) == ()
    assert tracker.update((), frame_sequence=60) == ()
    recovered = tracker.update(
        (Detection("person", 0.92, (0.0, 0.0, 20.0, 40.0)),),
        frame_sequence=90,
    )

    assert first[0].track_id == "P0001"
    assert recovered[0].track_id == "P0002"
    assert tracker._next_id == 3


def test_iou_tracker_uses_motion_to_preserve_ids_through_crossing() -> None:
    tracker = IoUTracker(match_iou=0.3, max_idle_frames=5)

    initial = tracker.update(
        (
            Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0)),
            Detection("person", 0.9, (80.0, 0.0, 100.0, 40.0)),
        ),
        frame_sequence=0,
    )
    left_to_right_id = _track_at_x(initial, 10.0).track_id
    right_to_left_id = _track_at_x(initial, 90.0).track_id

    tracker.update(
        (
            # Reverse detector order on every step so input order cannot carry IDs.
            Detection("person", 0.91, (55.0, 0.0, 75.0, 40.0)),
            Detection("person", 0.92, (25.0, 0.0, 45.0, 40.0)),
        ),
        frame_sequence=1,
    )
    crossed = tracker.update(
        (
            Detection("person", 0.93, (25.0, 0.0, 45.0, 40.0)),
            Detection("person", 0.94, (55.0, 0.0, 75.0, 40.0)),
        ),
        frame_sequence=2,
    )
    separated = tracker.update(
        (
            Detection("person", 0.95, (80.0, 0.0, 100.0, 40.0)),
            Detection("person", 0.95, (0.0, 0.0, 20.0, 40.0)),
        ),
        frame_sequence=3,
    )

    assert _track_at_x(crossed, 65.0).track_id == left_to_right_id
    assert _track_at_x(crossed, 35.0).track_id == right_to_left_id
    assert _track_at_x(separated, 90.0).track_id == left_to_right_id
    assert _track_at_x(separated, 10.0).track_id == right_to_left_id
    assert tracker._next_id == 3


def _track_at_x(tracks: tuple[Any, ...], center_x: float) -> Any:
    return min(
        tracks,
        key=lambda track: abs(((track.bbox[0] + track.bbox[2]) / 2) - center_x),
    )


def test_engine_returns_normalized_boxes_and_estimated_attributes() -> None:
    engine = RichLabelEngine(_Models())

    first = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-a",
        frame_sequence=7,
        timestamp_seconds=1.5,
    )

    assert first["stream_session_id"] == "session-a"
    assert first["frame_sequence"] == 7
    assert [item["track_id"] for item in first["objects"]] == ["O0002", "P0001"]
    person = first["objects"][1]
    assert person["bbox"] == {
        "x1": 10 / 120,
        "y1": 0.1,
        "x2": 70 / 120,
        "y2": 0.95,
    }
    assert person["attributes"] == {}
    assert engine._attribute_idle.wait(1.0)

    second = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-a",
        frame_sequence=8,
        timestamp_seconds=1.6,
    )
    person = second["objects"][1]
    assert person["attributes"]["estimated_age_range"] == "18-60"


def test_engine_does_not_schedule_attributes_for_held_tracks() -> None:
    calls = 0

    class _OneMissModels(_Models):
        show_person = True

        def detect(self, image: np.ndarray[Any, Any]) -> tuple[Detection, ...]:
            del image
            if not self.show_person:
                return ()
            return (Detection("person", 0.94, (10.0, 10.0, 70.0, 95.0)),)

        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            nonlocal calls
            calls += 1
            return super().attributes(crops)

    models = _OneMissModels()
    engine = RichLabelEngine(models)
    engine.attribute_interval_frames = 1
    first = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="held-attribute-session",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    assert engine._attribute_idle.wait(1.0)

    models.show_person = False
    held = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="held-attribute-session",
        frame_sequence=2,
        timestamp_seconds=0.2,
    )

    assert held["objects"][0]["track_id"] == first["objects"][0]["track_id"]
    assert engine._attribute_idle.is_set()
    assert calls == 1


def test_attribute_inference_never_blocks_detector_or_queues_tasks() -> None:
    attribute_started = threading.Event()
    release_attribute = threading.Event()
    calls = 0

    class _BlockingModels(_Models):
        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            nonlocal calls
            calls += 1
            attribute_started.set()
            assert release_attribute.wait(2.0)
            return super().attributes(crops)

    engine = RichLabelEngine(_BlockingModels())
    started = time.perf_counter()
    first = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="non-blocking-session",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    first_latency = time.perf_counter() - started

    try:
        assert first_latency < 0.5
        assert first["objects"][1]["attributes"] == {}
        assert attribute_started.wait(1.0)

        for frame_sequence in range(2, 7):
            started = time.perf_counter()
            result = engine.infer(
                np.zeros((100, 120, 3), dtype=np.uint8),
                stream_session_id="non-blocking-session",
                frame_sequence=frame_sequence,
                timestamp_seconds=frame_sequence / 10,
            )
            assert time.perf_counter() - started < 0.5
            assert len(result["objects"]) == 2
        assert calls == 1
    finally:
        release_attribute.set()

    assert engine._attribute_idle.wait(1.0)
    cached = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="non-blocking-session",
        frame_sequence=7,
        timestamp_seconds=0.7,
    )
    assert cached["objects"][1]["attributes"]["estimated_age_range"] == "18-60"
    assert calls == 1


def test_old_attribute_task_cannot_pollute_reset_session_or_reused_track_id() -> None:
    old_started = threading.Event()
    release_old = threading.Event()
    new_finished = threading.Event()
    calls = 0

    class _SessionModels(_Models):
        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            nonlocal calls
            calls += 1
            if calls == 1:
                old_started.set()
                assert release_old.wait(2.0)
                marker = "old-session"
            else:
                marker = "new-session"
                new_finished.set()
            return tuple({"marker": marker} for _ in crops)

    engine = RichLabelEngine(_SessionModels())
    engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-old",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    assert old_started.wait(1.0)

    reset = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-new",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    new_person = reset["objects"][1]
    assert new_person["track_id"] == "P0001"
    assert new_person["attributes"] == {}

    release_old.set()
    assert engine._attribute_idle.wait(1.0)
    after_old = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-new",
        frame_sequence=2,
        timestamp_seconds=0.2,
    )
    assert after_old["objects"][1]["attributes"] == {}
    assert new_finished.wait(1.0)
    assert engine._attribute_idle.wait(1.0)

    cached = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-new",
        frame_sequence=3,
        timestamp_seconds=0.3,
    )
    assert cached["objects"][1]["track_id"] == "P0001"
    assert cached["objects"][1]["attributes"] == {"marker": "new-session"}
    assert calls == 2


def test_old_attribute_task_cannot_update_an_expired_track() -> None:
    attribute_started = threading.Event()
    release_attribute = threading.Event()

    class _ExpiringModels(_Models):
        show_person = True

        def detect(self, image: np.ndarray[Any, Any]) -> tuple[Detection, ...]:
            del image
            if not self.show_person:
                return ()
            return (Detection("person", 0.94, (10.0, 10.0, 70.0, 95.0)),)

        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            attribute_started.set()
            assert release_attribute.wait(2.0)
            return tuple({"marker": "expired"} for _ in crops)

    models = _ExpiringModels()
    engine = RichLabelEngine(models)
    first = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="same-session",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    assert first["objects"][0]["track_id"] == "P0001"
    assert attribute_started.wait(1.0)

    models.show_person = False
    empty = None
    expiry_frame = int(engine.tracker.max_idle_seconds * 10) + 3
    for frame_sequence in range(2, expiry_frame):
        empty = engine.infer(
            np.zeros((100, 120, 3), dtype=np.uint8),
            stream_session_id="same-session",
            frame_sequence=frame_sequence,
            timestamp_seconds=frame_sequence / 10,
        )
    assert empty is not None
    assert empty["objects"] == []

    release_attribute.set()
    assert engine._attribute_idle.wait(1.0)
    assert "P0001" not in engine.tracker._tracks

    models.show_person = True
    replacement = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="same-session",
        frame_sequence=expiry_frame,
        timestamp_seconds=expiry_frame / 10,
    )
    assert replacement["objects"][0]["track_id"] == "P0002"
    assert replacement["objects"][0]["attributes"] == {}
    assert engine._attribute_idle.wait(1.0)


def test_shutdown_invalidates_in_flight_attributes_and_stops_new_tasks() -> None:
    attribute_started = threading.Event()
    release_attribute = threading.Event()
    calls = 0

    class _ShutdownModels(_Models):
        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            nonlocal calls
            calls += 1
            attribute_started.set()
            assert release_attribute.wait(2.0)
            return tuple({"marker": "too-late"} for _ in crops)

    engine = RichLabelEngine(_ShutdownModels())
    engine.attribute_shutdown_timeout_seconds = 0.01
    engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="shutdown-session",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    assert attribute_started.wait(1.0)

    with TestClient(create_app(engine)) as client:
        assert client.get("/health/live").status_code == 200
    assert engine._closed

    release_attribute.set()
    assert engine._attribute_idle.wait(1.0)
    after_shutdown = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="shutdown-session",
        frame_sequence=2,
        timestamp_seconds=0.2,
    )
    assert after_shutdown["objects"][1]["attributes"] == {}
    assert calls == 1


def test_malformed_attribute_results_back_off_without_losing_boxes() -> None:
    calls = 0

    class _MalformedModels(_Models):
        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            nonlocal calls
            del crops
            calls += 1
            return ()

    engine = RichLabelEngine(_MalformedModels())
    first = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="malformed-session",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    assert len(first["objects"]) == 2
    assert engine._attribute_idle.wait(1.0)

    second = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="malformed-session",
        frame_sequence=2,
        timestamp_seconds=0.2,
    )
    assert len(second["objects"]) == 2
    assert calls == 1


def test_attribute_model_requests_the_complete_public_paddlex_result() -> None:
    seen: dict[str, Any] = {}

    class _AttributeModel:
        def predict(self, images: Any, **kwargs: Any) -> list[dict[str, Any]]:
            seen["images"] = images
            seen.update(kwargs)
            return [{"res": _attribute_payload()}]

    models = PaddleRichModels()
    models._attribute = _AttributeModel()

    result = models.attributes((np.zeros((20, 10, 3), dtype=np.uint8),))

    assert len(result) == 1
    assert seen["batch_size"] == 1
    # PaddleX filters with score > threshold, so -1 also preserves exact zero.
    assert seen["threshold"] == -1.0
    assert len(seen["images"]) == 1


def test_attribute_result_reproduces_pphuman_grouped_postprocessing() -> None:
    scores = [0.0] * 26
    scores[0] = 0.71  # Hat: normal 0.5 threshold.
    scores[1] = 0.31  # Glasses: special 0.3 threshold.
    scores[2:4] = [0.2, 0.8]  # Sleeve argmax.
    scores[4:8] = [0.1, 0.72, 0.2, 0.3]  # Upper argmax if > 0.5.
    scores[8:14] = [0.61, 0.1, 0.73, 0.2, 0.3, 0.4]  # Multi-label lower.
    scores[14] = 0.9
    scores[15:18] = [0.2, 0.3, 0.81]  # Bag argmax if > 0.5.
    scores[18] = 0.61  # Hold: special 0.6 threshold.
    scores[19:22] = [0.2, 0.7, 0.91]  # Age argmax -> >60.
    scores[22] = 0.2  # Female <= 0.5 means Male.
    scores[23:26] = [0.1, 0.82, 0.2]  # Direction argmax -> Side.

    result = _pedestrian_attributes(_attribute_payload(scores))

    assert result["estimated_gender"] == "male-presenting"
    assert result["estimated_gender_confidence"] == 0.8
    assert result["estimated_age_range"] == ">60"
    assert result["estimated_age_confidence"] == 0.91
    assert result["estimated_direction"] == "Side"
    assert result["estimated_direction_confidence"] == 0.82
    assert result["wearing_glasses"] is True
    assert result["wearing_glasses_confidence"] == 0.31
    assert result["wearing_hat"] is True
    assert result["holding_object"] is True
    assert result["bag_type"] == "Backpack"
    assert result["bag_confidence"] == 0.81
    assert result["sleeve_type"] == "LongSleeve"
    assert result["upper_style"] == "UpperLogo"
    assert result["lower_styles"] == ["LowerStripe", "LongCoat"]
    assert result["lower_style_confidences"] == {
        "LowerStripe": 0.61,
        "LongCoat": 0.73,
    }
    assert result["wearing_boots"] is True
    assert result["appearance_attributes"] == [
        "Side",
        "Glasses",
        "Hat",
        "HoldObjectsInFront",
        "Backpack",
        "LongSleeve",
        "UpperLogo",
        "LowerStripe",
        "LongCoat",
        "Boots",
    ]


def test_attribute_argmax_chooses_one_age_and_female_branch() -> None:
    scores = [0.1] * 26
    scores[19:22] = [0.72, 0.93, 0.81]
    scores[22] = 0.87
    scores[23:26] = [0.94, 0.2, 0.3]

    result = _pedestrian_attributes(_attribute_payload(scores))

    assert result["estimated_age_range"] == "18-60"
    assert result["estimated_age_confidence"] == 0.93
    assert result["estimated_gender"] == "female-presenting"
    assert result["estimated_gender_confidence"] == 0.87
    assert result["estimated_direction"] == "Front"


def test_attribute_result_uses_class_ids_when_paddlex_reorders_scores() -> None:
    scores = [0.1] * 26
    scores[17] = 0.84
    scores[19:22] = [0.12, 0.94, 0.31]
    scores[22] = 0.08
    scores[23:26] = [0.11, 0.82, 0.24]
    payload = _attribute_payload(scores)
    order = list(range(25, -1, -1))
    for key in ("class_ids", "label_names", "scores"):
        values = payload[key]
        payload[key] = [values[index] for index in order]

    result = _pedestrian_attributes(payload)

    assert result["estimated_age_range"] == "18-60"
    assert result["estimated_gender"] == "male-presenting"
    assert result["estimated_direction"] == "Side"
    assert result["bag_type"] == "Backpack"


def test_attribute_argmax_ties_keep_the_official_first_class() -> None:
    scores = [0.1] * 26
    scores[2:4] = [0.7, 0.7]
    scores[4:8] = [0.7] * 4
    scores[8:14] = [0.4] * 6
    scores[15:18] = [0.7] * 3
    scores[19:22] = [0.8] * 3
    scores[23:26] = [0.9] * 3

    result = _pedestrian_attributes(_attribute_payload(scores))

    assert result["estimated_age_range"] == "<18"
    assert result["estimated_direction"] == "Front"
    assert result["bag_type"] == "HandBag"
    assert result["sleeve_type"] == "ShortSleeve"
    assert result["upper_style"] == "UpperStride"
    assert result["lower_styles"] == ["LowerStripe"]


def test_attribute_thresholds_are_strict_and_lower_group_has_fallback() -> None:
    scores = [0.0] * 26
    scores[0] = 0.5
    scores[1] = 0.3
    scores[2:4] = [0.6, 0.4]
    scores[4:8] = [0.5, 0.2, 0.1, 0.0]
    scores[8:14] = [0.1, 0.2, 0.49, 0.3, 0.4, 0.05]
    scores[14] = 0.5
    scores[15:18] = [0.5, 0.2, 0.1]
    scores[18] = 0.6
    scores[19:22] = [0.8, 0.1, 0.2]
    scores[22] = 0.5
    scores[23:26] = [0.2, 0.7, 0.1]

    result = _pedestrian_attributes(_attribute_payload(scores))

    assert result["estimated_gender"] == "male-presenting"
    assert result["estimated_gender_confidence"] == 0.5
    assert result["wearing_glasses"] is False
    assert result["wearing_glasses_confidence"] == 0.7
    assert result["wearing_hat"] is False
    assert result["holding_object"] is False
    assert result["holding_object_confidence"] == 0.4
    assert result["bag_type"] == "No bag"
    assert result["bag_confidence"] == 0.5
    assert result["sleeve_type"] == "ShortSleeve"
    assert "upper_style" not in result
    assert result["lower_styles"] == ["LongCoat"]
    assert result["lower_style_confidences"] == {"LongCoat": 0.49}
    assert result["wearing_boots"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["class_ids"].pop(),
        lambda payload: payload["class_ids"].__setitem__(25, 24),
        lambda payload: payload["label_names"].__setitem__(22, "Male"),
        lambda payload: payload["scores"].__setitem__(7, float("nan")),
        lambda payload: payload["scores"].__setitem__(7, 1.01),
    ],
)
def test_malformed_or_partial_attribute_vectors_fail_open(
    mutate: Any,
) -> None:
    payload = _attribute_payload([0.25] * 26)
    mutate(payload)

    assert _pedestrian_attributes(payload) == {}


def test_thresholded_legacy_attribute_payload_is_not_misdecoded() -> None:
    assert _pedestrian_attributes(
        {
            "class_ids": [20, 22, 17],
            "label_names": ["Age18-60", "Female", "Backpack"],
            "scores": [0.91, 0.1, 0.8],
        }
    ) == {}


def test_attribute_model_failure_keeps_detector_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    attribute_attempted = threading.Event()

    class _Detector:
        def predict(self, image: Any, **kwargs: Any) -> list[dict[str, Any]]:
            del image, kwargs
            return [
                {
                    "res": {
                        "boxes": [
                            {
                                "label": "person",
                                "score": 0.91,
                                "coordinate": [10, 10, 70, 95],
                            }
                        ]
                    }
                }
            ]

    def create_model(**kwargs: Any) -> Any:
        model_name = str(kwargs["model_name"])
        calls.append(model_name)
        if "pedestrian_attribute" in model_name:
            attribute_attempted.set()
            raise RuntimeError("attribute model unavailable")
        return _Detector()

    monkeypatch.setitem(
        sys.modules,
        "paddlex",
        SimpleNamespace(create_model=create_model),
    )
    models = PaddleRichModels()
    engine = RichLabelEngine(models)

    result = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-degraded",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )

    assert len(result["objects"]) == 1
    assert result["objects"][0]["label"] == "person"
    assert result["objects"][0]["attributes"] == {}
    assert attribute_attempted.wait(1.0)
    assert engine._attribute_idle.wait(1.0)

    second = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-degraded",
        frame_sequence=2,
        timestamp_seconds=0.2,
    )

    assert len(second["objects"]) == 1
    assert calls.count("PicoDet-S") == 1
    assert calls.count("PP-LCNet_x1_0_pedestrian_attribute") == 1


def test_worker_http_contract_accepts_raw_jpeg() -> None:
    client = TestClient(create_app(RichLabelEngine(_Models())))

    response = client.post(
        "/v1/labels",
        content=_jpeg(),
        headers={
            "Content-Type": "image/jpeg",
            "X-Goodvision-Stream-Session-Id": "session-http",
            "X-Goodvision-Frame-Sequence": "3",
            "X-Goodvision-Timestamp-Seconds": "0.2",
        },
    )

    assert response.status_code == 200
    assert response.json()["stream_session_id"] == "session-http"
    assert len(response.json()["objects"]) == 2


def test_worker_preloads_detector_before_advertising_readiness() -> None:
    class _WarmupModels(_Models):
        ready = False

        def ensure_detector_loaded(self) -> None:
            self.ready = True

    models = _WarmupModels()
    engine = RichLabelEngine(models)

    assert engine.ready is False
    with TestClient(create_app(engine)) as client:
        assert engine.ready is True
        assert client.get("/health/ready").status_code == 200


def test_worker_rejects_missing_metadata_without_loading_models() -> None:
    client = TestClient(create_app(RichLabelEngine(_Models())))

    response = client.post(
        "/v1/labels",
        content=_jpeg(),
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 400


def test_worker_rejects_concurrent_inference_without_queueing() -> None:
    engine = RichLabelEngine(_Models())
    assert engine._lock.acquire(blocking=False)
    try:
        with pytest.raises(RichLabelBusyError):
            engine.infer(
                np.zeros((100, 120, 3), dtype=np.uint8),
                stream_session_id="busy-session",
                frame_sequence=1,
                timestamp_seconds=0.1,
            )

        client = TestClient(create_app(engine))
        response = client.post(
            "/v1/labels",
            content=_jpeg(),
            headers={
                "Content-Type": "image/jpeg",
                "X-Goodvision-Stream-Session-Id": "busy-session",
                "X-Goodvision-Frame-Sequence": "2",
                "X-Goodvision-Timestamp-Seconds": "0.2",
            },
        )
    finally:
        engine._lock.release()

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"


def test_worker_image_uses_the_writable_paddlex_cache_volume() -> None:
    dockerfile = (
        ROOT / "workers" / "paddle_rich_label" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "PADDLE_PDX_CACHE_HOME=/models" in dockerfile
    assert "libgl1" in dockerfile
    assert "import cv2, paddlex" in dockerfile
    assert "USER 10002:10002" in dockerfile
    assert "/health/ready" in dockerfile

    requirements = (
        ROOT / "workers" / "paddle_rich_label" / "requirements.txt"
    ).read_text(encoding="utf-8")
    assert "opencv-python-headless" not in requirements


def test_gpu_worker_is_isolated_and_targets_blackwell_supported_paddle() -> None:
    gpu_dockerfile = (
        ROOT / "workers" / "paddle_rich_label" / "Dockerfile.gpu"
    ).read_text(encoding="utf-8")
    cpu_dockerfile = (
        ROOT / "workers" / "paddle_rich_label" / "Dockerfile"
    ).read_text(encoding="utf-8")
    gpu_compose = (ROOT / "compose.gpu.yaml").read_text(encoding="utf-8")
    delivery_compose = (
        ROOT / "compose.rich-gpu.delivery.yaml"
    ).read_text(encoding="utf-8")

    assert "paddlepaddle-gpu==3.3.0" in gpu_dockerfile
    assert "FROM python:3.11-slim AS runtime" in gpu_dockerfile
    assert "FROM runtime AS customer" in gpu_dockerfile
    assert gpu_dockerfile.rstrip().endswith("FROM runtime AS development")
    assert "COPY workers/paddle_rich_label/app.py ./app.py" in gpu_dockerfile
    assert "python gpu_smoke.py --runtime" in gpu_dockerfile
    assert "/opt/goodvision/internal/perception_check.py --runtime" in gpu_dockerfile
    assert "COPY tools/customer_check.py ./tools/check_runtime.py" in gpu_dockerfile
    assert "COPY tools/customer_check.py ./tools/check_perception.py" in gpu_dockerfile
    assert "runtime_check.py" not in gpu_dockerfile
    assert "packages/stable/cu129" in gpu_dockerfile
    assert "paddle/version/__init__.py" in gpu_dockerfile
    assert "assert '120' in metadata" in gpu_dockerfile
    assert "RICH_DEVICE=gpu:0" in gpu_dockerfile
    assert "USER 10002:10002" in gpu_dockerfile
    assert "/health/ready" in gpu_dockerfile
    assert "paddlepaddle==3.2.2" in cpu_dockerfile
    assert "paddlepaddle-gpu" not in cpu_dockerfile

    for compose_text in (gpu_compose, delivery_compose):
        assert "PP-YOLOE_plus-L" in compose_text
        assert "driver: nvidia" in compose_text
        assert "count: 1" in compose_text
        assert "capabilities: [gpu]" in compose_text

    gpu_smoke = (
        ROOT / "workers" / "paddle_rich_label" / "gpu_smoke.py"
    ).read_text(encoding="utf-8")
    assert "paddle.device.is_compiled_with_cuda()" in gpu_smoke
    assert "paddle.version.cuda_archs()" in gpu_smoke
    assert "paddle.utils.run_check()" in gpu_smoke
    assert "models.detect(image)" in gpu_smoke
    assert "models.attributes((image,))" in gpu_smoke


# Public worker configuration, output boundaries, and validation commands.
@pytest.fixture(autouse=True)
def _clean_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RICH_DEVICE",
        "RICH_DETECTOR_PROFILE",
        "RICH_ATTRIBUTE_PROFILE",
        "RICH_DETECTOR_MODEL",
        "RICH_ATTRIBUTE_MODEL",
        "RICH_DETECTOR_MODEL_DIR",
        "RICH_ATTRIBUTE_MODEL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_profiles_preserve_the_existing_cpu_models() -> None:
    models = PaddleRichModels()

    assert models.device == "cpu"
    assert models.detector_profile == "balanced"
    assert models.detector_name == "PicoDet-S"
    assert models.attribute_profile == "standard"
    assert models.attribute_name == "PP-LCNet_x1_0_pedestrian_attribute"


@pytest.mark.parametrize(
    ("profile", "expected"),
    [("accurate", "PP-YOLOE_plus-L"), ("balanced", "PicoDet-S")],
)
def test_public_profiles_take_priority_over_legacy_model_names(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    expected: str,
) -> None:
    monkeypatch.setenv("RICH_DETECTOR_PROFILE", f" {profile.upper()} ")
    monkeypatch.setenv("RICH_ATTRIBUTE_PROFILE", "standard")
    monkeypatch.setenv("RICH_DETECTOR_MODEL", "legacy-private-detector")
    monkeypatch.setenv("RICH_ATTRIBUTE_MODEL", "legacy-private-attributes")

    models = PaddleRichModels()

    assert models.detector_name == expected
    assert models.detector_profile == profile
    assert models.attribute_name == "PP-LCNet_x1_0_pedestrian_attribute"
    assert models.attribute_profile == "standard"


def test_legacy_gpu_image_default_still_selects_the_accurate_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RICH_DEVICE", "gpu:0")
    monkeypatch.setenv("RICH_DETECTOR_MODEL", "PP-YOLOE_plus-L")
    monkeypatch.setenv("RICH_DETECTOR_PROFILE", " ")

    models = PaddleRichModels()

    assert models.detector_name == "PP-YOLOE_plus-L"
    assert models.detector_profile == "accurate"
    assert models.device == "gpu:0"


@pytest.mark.parametrize("use_profiles", [False, True])
def test_model_directories_remain_compatible_with_both_configuration_forms(
    monkeypatch: pytest.MonkeyPatch,
    use_profiles: bool,
) -> None:
    calls: list[dict[str, Any]] = []

    def create_model(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules, "paddlex", SimpleNamespace(create_model=create_model)
    )
    monkeypatch.setenv("RICH_DETECTOR_MODEL", "private-detector")
    monkeypatch.setenv("RICH_ATTRIBUTE_MODEL", "private-attributes")
    monkeypatch.setenv("RICH_DETECTOR_MODEL_DIR", "/models/local-detector")
    monkeypatch.setenv("RICH_ATTRIBUTE_MODEL_DIR", "/models/local-attributes")
    if use_profiles:
        monkeypatch.setenv("RICH_DETECTOR_PROFILE", "accurate")
        monkeypatch.setenv("RICH_ATTRIBUTE_PROFILE", "standard")

    models = PaddleRichModels()
    models.ensure_detector_loaded()
    models.ensure_attribute_loaded()

    assert calls == [
        {
            "model_name": "PP-YOLOE_plus-L" if use_profiles else "private-detector",
            "model_dir": "/models/local-detector",
            "device": "cpu",
        },
        {
            "model_name": (
                "PP-LCNet_x1_0_pedestrian_attribute"
                if use_profiles else "private-attributes"
            ),
            "model_dir": "/models/local-attributes",
            "device": "cpu",
        },
    ]
    if not use_profiles:
        assert models.detector_profile == models.attribute_profile == "custom"


@pytest.mark.parametrize(
    ("variable", "message"),
    [
        ("RICH_DETECTOR_PROFILE", "unsupported object detection profile"),
        ("RICH_ATTRIBUTE_PROFILE", "unsupported pedestrian attribute profile"),
    ],
)
def test_unknown_profile_fails_neutrally_without_falling_back(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    message: str,
) -> None:
    monkeypatch.setenv(variable, "Paddle-private-unknown-profile")
    monkeypatch.setenv("RICH_DETECTOR_MODEL", "PicoDet-S")
    monkeypatch.setenv("RICH_ATTRIBUTE_MODEL", "PP-LCNet_x1_0_pedestrian_attribute")

    with pytest.raises(ValueError) as error:
        PaddleRichModels()

    assert str(error.value) == message


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("ensure_detector_loaded", "object detection engine is unavailable"),
        ("ensure_attribute_loaded", "pedestrian attribute engine is unavailable"),
    ],
)
def test_missing_backend_reports_neutral_initialization_error(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    message: str,
) -> None:
    monkeypatch.setitem(sys.modules, "paddlex", None)
    models = PaddleRichModels()

    with pytest.raises(RuntimeError) as error:
        getattr(models, method)()

    assert str(error.value) == message
    assert error.value.__suppress_context__


class _BoundaryModels:
    ready = True

    def detect(self, _image: Any) -> tuple[Any, ...]:
        return ()

    def attributes(self, _crops: Any) -> tuple[dict[str, str], ...]:
        return ({"estimated_age_range": "18-60"},)


def test_worker_public_metadata_and_health_are_neutral() -> None:
    with TestClient(create_app(RichLabelEngine(_BoundaryModels()))) as client:
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert schema.json()["info"]["title"] == (
            "Goodvision Object and Attribute Engine"
        )
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/health/live").json() == {"live": True}
        assert client.get("/health/ready").json() == {
            "ready": True, "gpu_ready": False,
        }
        assert "paddle" not in schema.text.lower()


def test_worker_inference_error_does_not_log_backend_name_or_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingModels(_BoundaryModels):
        def detect(self, _image: Any) -> tuple[Any, ...]:
            raise RuntimeError("PaddleX private-model backend failure")

    _, encoded = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    with TestClient(create_app(RichLabelEngine(_FailingModels()))) as client:
        response = client.post(
            "/v1/labels",
            content=encoded.tobytes(),
            headers={
                "Content-Type": "image/jpeg",
                "X-Goodvision-Stream-Session-Id": "test-session",
                "X-Goodvision-Frame-Sequence": "1",
                "X-Goodvision-Timestamp-Seconds": "0.1",
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "inference unavailable"}
    assert "paddle" not in caplog.text.lower()
    assert "private-model" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_worker_warmup_failure_uses_a_neutral_error_boundary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingModels(_BoundaryModels):
        def ensure_detector_loaded(self) -> None:
            raise RuntimeError("PaddleX private-model initialization failure")

    with pytest.raises(RuntimeError) as error:
        with TestClient(create_app(RichLabelEngine(_FailingModels()))):
            pass

    assert str(error.value) == "object and attribute engine initialization failed"
    assert error.value.__suppress_context__
    assert "paddle" not in caplog.text.lower()
    assert all(record.exc_info is None for record in caplog.records)


def _install_smoke_backends(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: str | None = None,
) -> list[str]:
    calls: list[str] = []

    def record(name: str, result: Any = None) -> Any:
        calls.append(name)
        if failure == name:
            raise RuntimeError("Paddle NVIDIA private-backend failure")
        return result

    class _Tensor:
        def __add__(self, _value: float) -> _Tensor:
            return self

        def numpy(self) -> np.ndarray[Any, Any]:
            return np.array([0.0 if failure == "tensor_value" else 2.0])

    class _SmokeModels:
        device = "cpu"
        gpu_ready = False

        def verify_gpu_execution(self) -> None:
            record("readiness_probe")
            self.gpu_ready = failure != "gpu_not_ready"

        def detect(self, _image: Any) -> tuple[Any, ...]:
            assert self.device == "gpu:0"
            return record("detect", ())

        def attributes(self, _crops: Any) -> tuple[dict[str, str], ...]:
            assert self.device == "gpu:0"
            result = ({},) if failure == "empty_attributes" else ({"age": "18-60"},)
            return record("attributes", result)

    backend = SimpleNamespace(
        device=SimpleNamespace(
            is_compiled_with_cuda=lambda: record(
                "compiled", failure != "compiled_false"
            ),
            cuda=SimpleNamespace(
                device_count=lambda: record(
                    "devices", 0 if failure == "no_device" else 1
                ),
                get_device_properties=lambda _device: record("properties"),
            ),
            set_device=lambda _device: record("set_device"),
        ),
        version=SimpleNamespace(
            cuda_archs=lambda: record("architecture", "private-arch")
        ),
        ones=lambda *_args, **_kwargs: record("tensor", _Tensor()),
        utils=SimpleNamespace(run_check=lambda: record("runtime_check")),
    )
    monkeypatch.setitem(sys.modules, "paddle", backend)
    monkeypatch.setitem(
        sys.modules, "app", SimpleNamespace(PaddleRichModels=_SmokeModels)
    )
    return calls


def test_gpu_smoke_checks_real_execution_paths_but_reports_neutral_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_smoke_backends(monkeypatch)

    assert gpu_smoke.main([]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "gpu_ready": True,
        "acceleration": "ready",
        "object_detection": "ready",
        "pedestrian_attributes": "ready",
        "blank_detections": 0,
        "attribute_outputs": 1,
    }
    assert calls == [
        "compiled", "devices", "set_device", "properties", "architecture",
        "tensor", "runtime_check", "detect", "attributes",
        "readiness_probe",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "compiled_false", "no_device", "set_device", "tensor_value",
        "runtime_check", "detect", "attributes", "empty_attributes",
        "readiness_probe", "gpu_not_ready",
    ],
)
def test_gpu_smoke_never_reports_success_when_a_required_check_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    _install_smoke_backends(monkeypatch, failure=failure)

    assert gpu_smoke.main([]) == 1

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "ok": False,
        "error": "object and attribute engine validation failed",
    }
    assert captured.err == ""


def test_gpu_smoke_rejects_a_cpu_override_instead_of_claiming_acceleration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_smoke_backends(monkeypatch)
    monkeypatch.setenv("RICH_DEVICE", "cpu")

    assert gpu_smoke.main([]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert calls == []


def test_gpu_smoke_reports_missing_backend_without_an_import_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "paddle", None)

    assert gpu_smoke.main([]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] is False
    assert captured.err == ""


@pytest.mark.parametrize(
    "failure", [None, "version", "conflict", "image_version", "pip", "stderr"]
)
def test_perception_runtime_mode_never_imports_models_or_gpu_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    failure: str | None,
) -> None:
    versions = {"paddlepaddle-gpu": "3.3.0", "paddlex": "3.7.2"}
    monkeypatch.setattr(
        gpu_smoke.metadata, "version",
        lambda name: "unexpected" if failure == "version" else versions[name],
    )

    def distribution(_name: str) -> object:
        if failure == "conflict":
            return object()
        raise gpu_smoke.metadata.PackageNotFoundError

    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", 1)[0] in {"paddle", "paddlex", "app"}:
            raise AssertionError("runtime audit must not load inference backends")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(gpu_smoke.metadata, "distribution", distribution)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        __version__="bad" if failure == "image_version" else "4.10.0",
    ))
    monkeypatch.setattr(gpu_smoke.subprocess, "run", lambda *_args, **_kwargs: (
        SimpleNamespace(
            returncode=1 if failure == "pip" else 0,
            stdout="No broken requirements found.",
            stderr="unexpected diagnostic" if failure == "stderr" else "",
        )
    ))

    assert gpu_smoke.main(["--runtime"]) == (0 if failure is None else 1)
    record = json.loads(capsys.readouterr().out)
    assert record["ok"] is (failure is None)
    if failure is None:
        assert record == {"ok": True, "runtime": "ready"}


def test_perception_check_rejects_unknown_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert gpu_smoke.main(["--unexpected"]) == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False
