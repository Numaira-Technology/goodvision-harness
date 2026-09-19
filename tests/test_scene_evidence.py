from __future__ import annotations

import math

import pytest

from goodvision_harness.scene_evidence import InspectionPlanner, SceneEvidenceBuffer


def _add(buffer, sequence, timestamp=None, *, size=3, session="session"):
    return buffer.add(
        stream_session_id=session,
        frame_sequence=sequence,
        timestamp_seconds=float(sequence if timestamp is None else timestamp),
        jpeg_bytes=b"x" * size,
    )


def _select(buffer, start=0, end=100, limit=4, session="session"):
    return buffer.select(
        stream_session_id=session,
        start_seconds=start,
        end_seconds=end,
        limit=limit,
    )


def _scene(timestamp, people=("a",), *, shift=0, events=(), session="session"):
    return {
        "stream_session_id": session,
        "entities": [
            {
                "entity_id": person,
                "label": "person",
                "observation_state": "observed",
                "last_observed_timestamp_seconds": timestamp,
                "bbox": {"x1": 0.1 + shift, "y1": 0.1, "x2": 0.3 + shift, "y2": 0.8},
            }
            for person in people
        ],
        "events": list(events),
    }


def _plan(planner, timestamp, **kwargs):
    return planner.plan(_scene(timestamp, **kwargs), timestamp_seconds=timestamp)


def _finish(planner, requests, timestamp, success=True):
    for request in requests:
        planner.complete(
            request.request_id, success=success, timestamp_seconds=timestamp
        )


def test_evidence_sampling_includes_boundaries_and_excludes_future():
    buffer = SceneEvidenceBuffer(history_seconds=20)
    for sequence in range(11):
        assert _add(buffer, sequence)
    frames = _select(buffer, start=2, end=9, limit=4)
    assert [f.timestamp_seconds for f in frames] == [2, 4, 7, 9]
    assert _select(buffer, start=2, end=9, limit=1)[0].frame_sequence == 9
    assert not _select(buffer, start=11, end=12)


def test_out_of_order_frames_do_not_regress_age_eviction():
    buffer = SceneEvidenceBuffer(history_seconds=3)
    assert _add(buffer, 10)
    assert _add(buffer, 8)
    assert not _add(buffer, 6)
    assert _add(buffer, 13)
    assert [f.frame_sequence for f in _select(buffer)] == [10, 13]


def test_evidence_dedup_and_session_reset():
    buffer = SceneEvidenceBuffer()
    assert _add(buffer, 1)
    assert not _add(buffer, 1, timestamp=2)
    assert not _add(buffer, 2, timestamp=1)
    assert not _add(buffer, 2, session="old")
    assert not _select(buffer, session="old")
    buffer.reset()
    assert _add(buffer, 0, session="new")
    assert not _select(buffer)


def test_encoded_memory_and_count_are_bounded():
    buffer = SceneEvidenceBuffer(max_frames=3, max_bytes=7, history_seconds=100)
    for sequence in range(20):
        _add(buffer, sequence)
        assert len(buffer) <= 3
        assert buffer.byte_count <= 7
    assert len(buffer) == 2
    assert buffer.byte_count == 6
    assert not _add(buffer, 21, size=8)
    assert not _add(buffer, 21, size=0)
    assert [f.frame_sequence for f in _select(buffer)] == [18, 19]


@pytest.mark.parametrize("timestamp", [math.nan, math.inf, -1])
def test_invalid_source_time_rejected(timestamp):
    with pytest.raises(ValueError):
        _add(SceneEvidenceBuffer(), 0, timestamp)
    with pytest.raises(ValueError):
        _plan(InspectionPlanner(), timestamp)


def test_new_person_once_then_motion_cooldown():
    planner = InspectionPlanner()
    first = _plan(planner, 0)
    assert {r.reason for r in first} == {"new_person", "full_scene_scan"}
    assert all(r.start_seconds == r.end_seconds == 0 for r in first)
    _finish(planner, first, 0)
    assert not _plan(planner, 1)
    assert not _plan(planner, 2, shift=0.15)
    moved = _plan(planner, 3, shift=0.15)
    assert len(moved) == 1
    assert moved[0].reason == "motion_change"
    assert moved[0].subject_ids == ("a",)
    assert moved[0].as_dict()["request_id"] == moved[0].request_id


def test_motion_accumulates_against_last_significant_position():
    planner = InspectionPlanner(cooldown_seconds=1)
    first = _plan(planner, 0)
    _finish(planner, first, 0)
    assert not _plan(planner, 1, shift=0.03)
    assert not _plan(planner, 2, shift=0.06)
    assert _plan(planner, 3, shift=0.09)[0].reason == "motion_change"


def test_fallback_occurs_without_detected_people():
    planner = InspectionPlanner(full_scene_interval_seconds=10)
    first = _plan(planner, 0, people=())
    assert first[0].reason == "full_scene_scan"
    assert first[0].subject_ids == ()
    _finish(planner, first, 0)
    assert not _plan(planner, 9, people=())
    assert _plan(planner, 10, people=())[0].reason == "full_scene_scan"


def test_pending_is_bounded_and_cannot_dispatch_same_person_twice():
    planner = InspectionPlanner(max_pending_requests=2)
    assert len(_plan(planner, 0, people=("a", "b", "c"))) == 2
    assert not _plan(planner, 4, people=("a", "b", "c"), shift=0.2)
    assert len(planner._pending) == 2


def test_expired_requests_back_off_and_late_completion_cannot_clear_new_work():
    planner = InspectionPlanner(pending_timeout_seconds=2, retry_backoff_seconds=2)
    first = _plan(planner, 0)
    assert not _plan(planner, 2)
    assert not _plan(planner, 3)
    retried = _plan(planner, 4)
    assert {r.subject_ids for r in retried} == {(), ("a",)}
    _finish(planner, first, 4)
    assert len(planner._pending) == 2
    assert not _plan(planner, 5)


def test_failed_inspection_retries_are_bounded_per_cue():
    planner = InspectionPlanner(
        cooldown_seconds=1,
        retry_backoff_seconds=1,
        max_retries=1,
        full_scene_interval_seconds=100,
    )
    first = _plan(planner, 0)
    _finish(planner, first, 0, success=False)
    retried = _plan(planner, 1)
    assert len(retried) == 2
    _finish(planner, retried, 1, success=False)
    for timestamp in range(2, 30):
        assert not _plan(planner, timestamp)


def test_fairness_in_crowd_with_continuous_motion():
    planner = InspectionPlanner(
        max_requests_per_plan=1,
        cooldown_seconds=1,
        full_scene_interval_seconds=100,
    )
    seen = set()
    for timestamp in range(7):
        requests = _plan(
            planner,
            timestamp,
            people=("a", "b", "c", "d"),
            shift=0.15 * (timestamp % 2),
        )
        seen.update(subject for r in requests for subject in r.subject_ids)
        _finish(planner, requests, timestamp)
    assert seen == {"a", "b", "c", "d"}


def test_events_schedule_groups_and_refresh_without_duplicate_members():
    planner = InspectionPlanner()
    event = {"subject_ids": ["a", "b"], "event_label": "fighting", "phase": "confirmed"}
    first = _plan(planner, 0, people=("a", "b"), events=(event,))
    _finish(planner, first, 0)
    requests = _plan(planner, 3, people=("a", "b"), events=(event,))
    assert len(requests) == 1
    assert requests[0].subject_ids == ("a", "b")
    assert requests[0].event_labels == ("fighting",)
    assert requests[0].reason == "event_refresh"
    _finish(planner, requests, 3)
    assert not _plan(planner, 10, people=("a", "b"), events=(event,))
    assert _plan(planner, 11, people=("a", "b"), events=(event,))[0].subject_ids == (
        "a",
        "b",
    )


def test_occlusion_and_held_observations_do_not_trigger_inspections():
    planner = InspectionPlanner()
    first = _plan(planner, 0)
    _finish(planner, first, 0)
    scene = _scene(1, shift=0.2)
    scene["entities"][0]["observation_state"] = "occluded"
    assert not planner.plan(scene, timestamp_seconds=1)
    assert not _plan(planner, 3)
    stale = _scene(3, shift=0.3)
    assert not planner.plan(stale, timestamp_seconds=7)
    assert not _plan(planner, 8)


def test_source_clock_and_session_cannot_regress():
    planner = InspectionPlanner()
    first = _plan(planner, 5)
    _finish(planner, first, 5)
    assert not _plan(planner, 5, people=("b",))
    assert not _plan(planner, 4, people=("b",))
    assert not _plan(planner, 6, people=("b",), session="old")
    planner.reset()
    second = _plan(planner, 0, session="new")
    assert second
    assert {r.request_id for r in first}.isdisjoint(r.request_id for r in second)


def test_bookkeeping_does_not_grow_across_departures():
    planner = InspectionPlanner(max_subjects=4, cooldown_seconds=1)
    for timestamp in range(100):
        requests = _plan(planner, timestamp, people=(f"person-{timestamp}",))
        _finish(planner, requests, timestamp)
        assert len(planner._subjects) <= 4
        assert len(planner._tasks) <= 9


def test_fallback_cannot_be_starved_by_continuous_new_people():
    planner = InspectionPlanner(
        max_requests_per_plan=1,
        cooldown_seconds=1,
        full_scene_interval_seconds=5,
    )
    scans = []
    for timestamp in range(16):
        requests = _plan(planner, timestamp, people=(f"person-{timestamp}",))
        scans.extend(r.end_seconds for r in requests if r.reason == "full_scene_scan")
        _finish(planner, requests, timestamp)
    assert scans == [0, 5, 10, 15]


def test_pending_departed_subjects_stay_bounded_until_expiry():
    planner = InspectionPlanner(
        max_subjects=1,
        max_pending_requests=4,
        pending_timeout_seconds=4,
    )
    for timestamp in range(100):
        _plan(planner, timestamp, people=(f"person-{timestamp}",))
        assert len(planner._subjects) <= 1
        assert len(planner._pending) <= 4
        assert len(planner._tasks) <= 6
