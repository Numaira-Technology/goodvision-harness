"""Synthetic association cases; no claims about detector/GPU accuracy."""

from __future__ import annotations

import pytest
from PIL import Image

from workers.sam_scene.hybrid_tracker import (
    PersonDetection,
    PersonTracker,
    TrackerCapacityError,
    describe_person,
)


def person(x=0.1, score=0.9, color=()):
    return PersonDetection((x, 0.1, x + 0.2, 0.9), score, color)


def update(tracker, detections, sequence):
    return tracker.update(detections, timestamp=sequence * 0.1, frame_sequence=sequence)


def test_motion_keeps_id_without_emitting_prediction():
    tracker = PersonTracker()
    first = update(tracker, [person()], 0)
    assert first[0].object_id == 0 and not first[0].confirmed
    second = update(tracker, [person(0.12)], 1)
    assert second[0].object_id == 0 and second[0].confirmed
    assert update(tracker, [], 2) == ()
    resumed = update(tracker, [person(0.16)], 3)
    assert resumed[0].object_id == 0
    assert resumed[0].detection.bbox == person(0.16).bbox


def test_low_score_cannot_create_person_but_can_continue_repeated_track():
    tracker = PersonTracker()
    assert update(tracker, [person(score=0.3)], 0) == ()
    high = update(tracker, [person()], 1)
    assert update(tracker, [person(score=0.3)], 2) == ()
    update(tracker, [person()], 3)
    low = update(tracker, [person(score=0.3)], 4)
    assert low[0].object_id == high[0].object_id
    assert low[0].detection.confidence == 0.3


def test_same_source_sam_confirmation_enables_weak_track_continuation():
    tracker = PersonTracker()
    initial = update(tracker, [person()], 0)
    original_geometry = tracker.tracks[0].detection.bbox
    counts = tracker.calibrate(initial, [person(0.11)], frame_sequence=0, timestamp=0.0)
    assert counts == {"confirmed": 1, "uncertain": 0, "discovered": 0}
    assert tracker.tracks[0].detection.bbox == original_geometry
    result = update(tracker, [person(0.12, score=0.25)], 1)
    assert len(result) == 1 and result[0].confirmed
    assert result[0].calibration["frame_sequence"] == 0
    assert result[0].detection.bbox == person(0.12).bbox


def test_sam_discovery_only_becomes_observation_after_actual_detector_match():
    tracker = PersonTracker()
    assert update(tracker, [], 0) == ()
    assert tracker.calibrate((), [person()], frame_sequence=0, timestamp=0.0) == {
        "confirmed": 0,
        "uncertain": 0,
        "discovered": 1,
    }
    assert tracker.tracks == {}
    result = update(tracker, [person(0.11, score=0.25)], 1)
    assert len(result) == 1 and result[0].confirmed
    assert result[0].detection.confidence == 0.25
    assert tracker.seed_matches == 1


def test_sam_discovery_cannot_choose_between_two_similar_current_detections():
    tracker = PersonTracker()
    update(tracker, [], 0)
    tracker.calibrate((), [person()], frame_sequence=0, timestamp=0.0)
    assert (
        update(tracker, [person(0.10, score=0.25), person(0.11, score=0.25)], 1) == ()
    )
    assert tracker.seed_matches == 0


def test_low_confidence_sam_cannot_confirm_or_create_candidates():
    tracker = PersonTracker()
    initial = update(tracker, [person()], 0)
    counts = tracker.calibrate(
        initial,
        [person(score=0.1), person(0.6, score=0.2)],
        frame_sequence=0,
        timestamp=0.0,
    )
    assert counts == {"confirmed": 0, "uncertain": 0, "discovered": 0}
    assert tracker.tracks[0].calibration is None and tracker.seeds == []


def test_stale_calibration_does_not_resurrect_old_track_or_seed():
    tracker = PersonTracker(ttl_seconds=0.5)
    first = update(tracker, [person()], 0)
    assert update(tracker, [], 10) == ()
    counts = tracker.calibrate(first, [person()], frame_sequence=0, timestamp=0.0)
    assert counts == {"confirmed": 0, "uncertain": 0, "discovered": 0}
    assert tracker.tracks == {} and tracker.seeds == []


def test_old_calibration_cannot_overwrite_new_current_geometry():
    tracker = PersonTracker()
    first = update(tracker, [person()], 0)
    current = update(tracker, [person(0.15)], 1)
    tracker.calibrate(first, [person(0.11)], frame_sequence=0, timestamp=0.0)
    assert tracker.tracks[0].detection == current[0].detection
    assert tracker.tracks[0].timestamp == 0.1


def test_ambiguous_crossing_allocates_new_ids_instead_of_merging_people():
    tracker = PersonTracker()
    initial = update(tracker, [person(0.1), person(0.3)], 0)
    ambiguous = update(tracker, [person(0.19), person(0.21)], 1)
    assert {o.object_id for o in initial}.isdisjoint(o.object_id for o in ambiguous)
    assert tracker.ambiguous_count > 0
    assert all(not o.confirmed for o in ambiguous)


def test_appearance_conflict_prevents_identity_transfer():
    tracker = PersonTracker()
    red = update(tracker, [person(color=(1.0, 0.0))], 0)
    blue = update(tracker, [person(color=(0.0, 1.0))], 1)
    assert red[0].object_id != blue[0].object_id


def test_ambiguous_sam_overlap_marks_track_uncertain_and_preserves_geometry():
    tracker = PersonTracker()
    first = update(tracker, [person(0.1), person(0.12)], 0)
    tracker.calibrate(first, [person(0.11)], frame_sequence=0, timestamp=0.0)
    assert all(
        track.calibration["status"] == "uncertain" for track in tracker.tracks.values()
    )
    assert all(
        not tracker._observed(track).confirmed for track in tracker.tracks.values()
    )


def test_bounds_and_no_id_reuse_after_expiry():
    tracker = PersonTracker(max_objects=1, max_ids=2, ttl_seconds=0.5)
    assert update(tracker, [person()], 0)[0].object_id == 0
    assert update(tracker, [person()], 10)[0].object_id == 1
    with pytest.raises(TrackerCapacityError):
        update(tracker, [person()], 20)


@pytest.mark.parametrize("sequence,timestamp", [(0, 0.1), (1, 0.0), (2, float("nan"))])
def test_source_order_cannot_regress(sequence, timestamp):
    tracker = PersonTracker()
    update(tracker, [person()], 0)
    with pytest.raises(ValueError):
        tracker.update([person()], timestamp=timestamp, frame_sequence=sequence)


def test_descriptor_is_bounded_and_captures_different_colors():
    with Image.new("RGB", (40, 60), "red") as red:
        red_features = describe_person(red, (0.1, 0.1, 0.9, 0.9))
    with Image.new("RGB", (40, 60), "blue") as blue:
        blue_features = describe_person(blue, (0.1, 0.1, 0.9, 0.9))
    assert len(red_features) == len(blue_features) == 48
    assert red_features != blue_features
    assert all(0 <= v <= 1 for v in red_features)


@pytest.mark.parametrize(
    "bbox,score",
    [
        ((0.0, 0.0, 0.0, 1.0), 0.9),
        ((-1.0, 0.0, 1.0, 1.0), 0.9),
        ((0.0, 0.0, 1.0, 1.0), float("nan")),
    ],
)
def test_malformed_detection_rejected(bbox, score):
    with pytest.raises(ValueError):
        PersonDetection(bbox, score)
