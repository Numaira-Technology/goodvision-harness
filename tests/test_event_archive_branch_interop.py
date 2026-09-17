"""Independent safety/face request ledgers and partial-result interoperability."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

from goodvision_harness.event_archive import EventArchive, EventArchiveSettings

LABELS = {"safety": ("fight", "smoking", "fall", "weapon"), "stranger": ("stranger",)}
TARGET = {"safety": "smoking", "stranger": "stranger"}


def _other(branch):
    return "stranger" if branch == "safety" else "safety"


def _admit(archive, branch, timestamp):
    assert archive.note_window(
        "session",
        f"{branch}:{timestamp:g}",
        timestamp,
        LABELS[branch],
        source_frame_index=round(timestamp * 5),
    )


def _result(branch, timestamp, positive=True, *, missing_other=True):
    label = TARGET[branch]
    return {
        "schema_version": "2",
        "message_id": f"{branch}-message-{timestamp:g}",
        "source_id": "camera",
        "stream_session_id": "session",
        "frame": {
            "timestamp_seconds": timestamp,
            "source_frame_index": round(timestamp * 5),
        },
        "judgments": [
            {
                "event_label": value,
                "detected": positive and value == label,
                "object_ids": ["target"] if positive and value == label else [],
                "confidence": 0.9,
                "description": "Visible evidence",
            }
            for value in LABELS[branch]
        ],
        "objects": [
            {
                "object_id": "target",
                "label": "person",
                "attributes": {"subject_id": "person-1", "subject_type": "person"},
                "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.8},
            }
        ]
        if positive
        else [],
        "missing_event_labels": list(LABELS[_other(branch)]) if missing_other else [],
    }


def _complete(archive, branch, timestamp, positive=True, *, window_id="registered"):
    record = _result(branch, timestamp, positive)
    submitted_id = f"{branch}:{timestamp:g}" if window_id == "registered" else window_id
    assert archive.submit_result(record, window_id=submitted_id)
    assert archive.note_window(
        "session",
        f"{branch}:{timestamp:g}",
        timestamp,
        LABELS[branch],
        "completed",
    )
    return record["message_id"]


def _lookup(archive, message):
    assert archive.flush()
    events = archive.list_events(message_id=message)["events"]
    assert len(events) == 1
    return events[0]


@pytest.mark.parametrize("slow_branch", LABELS)
def test_late_branch_keeps_own_source_frame_and_exact_message_association(
    tmp_path, slow_branch
):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        fast_branch = _other(slow_branch)
        _admit(archive, slow_branch, 1)
        _admit(archive, fast_branch, 4)
        fast_message = _complete(archive, fast_branch, 4)
        fast_event = _lookup(archive, fast_message)
        assert len(archive.list_events()["events"]) == 1
        assert fast_event["event_label"] == TARGET[fast_branch]
        slow_message = _complete(archive, slow_branch, 1)
        slow_event = _lookup(archive, slow_message)
        assert slow_event["event_label"] == TARGET[slow_branch]
        assert slow_event["start_seconds"] == slow_event["last_detected_seconds"] == 1
        assert fast_event["start_seconds"] == fast_event["last_detected_seconds"] == 4
        assert fast_event["event_id"] != slow_event["event_id"]
        assert _lookup(archive, fast_message)["event_id"] == fast_event["event_id"]
        assert len(archive.list_events()["events"]) == 2
    finally:
        archive.close()


@pytest.mark.parametrize("branch", LABELS)
def test_other_branch_completion_cannot_finalize_pending_negative_boundary(
    tmp_path, branch
):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        for timestamp in (0, 1, 2, 3, 4):
            for source in LABELS:
                _admit(archive, source, timestamp)
        first_message = _complete(archive, branch, 0)
        for timestamp in (0, 1, 2, 3, 4):
            _complete(archive, _other(branch), timestamp, False)
        _complete(archive, branch, 1, False)
        _complete(archive, branch, 3, False)
        last_message = _complete(archive, branch, 4)
        first = _lookup(archive, first_message)
        assert not first["finalized"]
        assert first["end_reason"] == "negative"
        assert _lookup(archive, last_message)["event_id"] != first["event_id"]
        # Stranger evidence exposes one representative per person while both
        # source episodes remain independently revisable until the bridge lands.
        expected_rows = 1 if branch == "stranger" else 2
        assert len(archive.list_events()["events"]) == expected_rows
        bridge_message = _complete(archive, branch, 2)
        merged = _lookup(archive, bridge_message)
        assert merged["event_id"] == first["event_id"]
        assert merged["state"] == "active"
        assert merged["last_detected_seconds"] == 4
        assert merged["observation_count"] == 3
        assert _lookup(archive, last_message)["event_id"] == merged["event_id"]
        assert len(archive.list_events()["events"]) == 1
    finally:
        archive.close()


@pytest.mark.parametrize("branch", LABELS)
def test_partial_omission_cannot_erase_other_branch_explicit_negatives(
    tmp_path, branch
):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        for timestamp in (0, 1, 3):
            for source in LABELS:
                _admit(archive, source, timestamp)
        for timestamp in (0, 1, 3):
            _complete(archive, _other(branch), timestamp, False)
            message = _complete(archive, branch, timestamp, timestamp == 0)
            if timestamp == 0:
                first_message = message
        event = _lookup(archive, first_message)
        assert event["end_reason"] == "negative"
        assert event["end_seconds"] == 1
        assert event["finalized"]
    finally:
        archive.close()


@pytest.mark.parametrize("window_id", [None, "not-registered"])
def test_missing_labels_without_explicit_registered_scope_remain_unknown(
    tmp_path, window_id
):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        for timestamp in (0, 1, 3):
            for source in LABELS:
                _admit(archive, source, timestamp)
        first_message = _complete(archive, "safety", 0)
        for timestamp in (1, 3):
            _complete(archive, "stranger", timestamp, False, window_id=window_id)
            _complete(archive, "safety", timestamp, False)
        event = _lookup(archive, first_message)
        assert event["state"] == "active"
        assert event["end_reason"] is None
    finally:
        archive.close()


@pytest.mark.parametrize("outcome", ["failed", "dropped", "cancelled"])
def test_unavailable_face_window_is_unknown_without_blocking_safety_closure(
    tmp_path, outcome
):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        for timestamp in (0, 1, 2, 3):
            for branch in LABELS:
                _admit(archive, branch, timestamp)
        first = {branch: _complete(archive, branch, 0) for branch in LABELS}
        for branch in LABELS:
            _complete(archive, branch, 1, False)
        assert archive.note_window("session", "stranger:2", 2, ("stranger",), outcome)
        _complete(archive, "safety", 2, False)
        for branch in LABELS:
            _complete(archive, branch, 3, False)
        assert _lookup(archive, first["safety"])["end_reason"] == "negative"
        face = _lookup(archive, first["stranger"])
        assert face["state"] == "active"
        assert face["end_reason"] is None
    finally:
        archive.close()


def test_registered_scope_never_discards_real_judgments(tmp_path):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        _admit(archive, "safety", 1)
        # Scope limits omissions only. An explicitly supplied real judgment is
        # retained even if a caller supplied a narrower ledger by mistake.
        record = _result("stranger", 1)
        assert archive.submit_result(record, window_id="safety:1")
        assert _lookup(archive, record["message_id"])["event_label"] == "stranger"
    finally:
        archive.close()


def _jpeg():
    ok, data = cv2.imencode(".jpg", np.full((36, 64, 3), 100, np.uint8))
    assert ok
    return data.tobytes()


def test_both_pending_branches_pin_same_real_clip_beyond_ring_horizon(tmp_path):
    def encoder(manifest, destination, **_):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["frames"][0]["timestamp_seconds"] == 0
        assert data["coverage_boundary"]["timestamp_seconds"] == 3
        destination.write_bytes(b"test-encoding")
        return {"start_seconds": 0, "end_seconds": 3, "duration_seconds": 3, "fps": 5}

    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path),
        encoder=encoder,
        clock=lambda: 0.0,
    )
    try:
        jpeg = _jpeg()
        for index in range(6):
            assert archive.submit_frame(jpeg, index / 5, index, "session")
        for branch in LABELS:
            _admit(archive, branch, 1)
        for index in range(6, 476):
            assert archive.submit_frame(jpeg, index / 5, index, "session")
            if index % 50 == 0:
                assert archive.flush()
        for branch in LABELS:
            _complete(archive, branch, 1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            assert archive.flush()
            events = archive.list_events()["events"]
            if len(events) == 2 and all(
                e["evidence"]["status"] == "ready" for e in events
            ):
                break
            time.sleep(0.01)
        assert len(events) == 2
        for event in events:
            assert event["evidence"]["status"] == "ready", event["evidence"]
            assert event["evidence"]["available"]["duration_seconds"] == 3
    finally:
        archive.close()


def test_duplicate_branch_pins_share_one_bounded_resident_jpeg_budget(tmp_path):
    budget = 8192
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path, buffer_max_bytes=budget),
        clock=lambda: 0.0,
    )
    try:
        jpeg = _jpeg()
        for index in range(50):
            timestamp = index / 5
            assert archive.submit_frame(jpeg, timestamp, index, "session")
            for branch in LABELS:
                _admit(archive, branch, timestamp)
            assert archive.flush()
            with archive._lock:
                session = archive._sessions["session"]
                frames = {id(f): f for f in session.ring.frames}
                frames.update({id(f): f for pin in session.pins.values() for f in pin})
                resident_bytes = sum(len(f.jpeg) for f in frames.values())
                with archive._ingest_lock:
                    assert resident_bytes + archive._queued_frame_bytes <= budget
        assert archive.status()["queue_depth"] <= archive.settings.queue_size
    finally:
        archive.close()
