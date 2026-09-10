from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import numpy as np
import pytest

from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.video_evidence import EvidenceFrame, stage_clip


def result(
    timestamp: float,
    message: str,
    detected: bool = True,
    *,
    subject: str | None = "person-1",
    label: str = "smoking",
) -> dict[str, Any]:
    attributes = {"subject_id": subject, "subject_type": "person"} if subject else {}
    return {
        "message_id": message,
        "source_id": "camera",
        "stream_session_id": "session",
        "frame": {
            "timestamp_seconds": timestamp,
            "source_frame_index": int(timestamp * 5),
            "observed_at": "2026-09-09T00:00:00+00:00",
        },
        "judgments": [
            {
                "event_label": label,
                "detected": detected,
                "object_ids": ["o"] if detected else [],
                "confidence": 0.9,
                "description": "Visible event",
            }
        ],
        "objects": [
            {
                "object_id": "o",
                "label": "person",
                "attributes": attributes,
                "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.8},
            }
        ]
        if detected
        else [],
    }


def jpeg() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))
    assert ok
    return encoded.tobytes()


@pytest.mark.parametrize("timestamp", [0.0092, 0.0098])
@pytest.mark.parametrize("explicit_window", [False, True])
def test_first_result_uses_same_millisecond_timeline_as_window(
    tmp_path: Path, timestamp: float, explicit_window: bool,
) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        archive.note_window("session", "safety:0", timestamp, ("smoking",))
        archive.submit_result(
            result(round(timestamp, 3), "first"),
            window_id="safety:0" if explicit_window else None,
        )
        archive.note_window(
            "session", "safety:0", timestamp, ("smoking",), "completed",
        )
        assert archive.flush()
        events = archive.list_events(message_id="first")["events"]
        assert len(events) == 1
        assert events[0]["start_seconds"] == round(timestamp, 3)
        assert archive.status()["late_results"] == 0
        # A genuinely earlier result must still be rejected by the watermark.
        archive.submit_result(result(round(timestamp, 3) - 0.005, "before"))
        assert archive.flush()
        assert archive.list_events(message_id="before")["events"] == []
        assert archive.status()["late_results"] == 1
    finally:
        archive.close()


def fake_encoder(manifest: Path, destination: Path, **_: Any) -> dict[str, Any]:
    data = json.loads(manifest.read_text())
    destination.write_bytes(f"video-{data['revision']}".encode())
    return {"duration_seconds": 3.0, "fps": 5.0, "partial": False}


def wait_ready(archive: EventArchive, *, timeout: float = 3.0) -> dict[str, Any]:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        archive.flush()
        events = archive.list_events()["events"]
        if events and events[0]["evidence"]["status"] == "ready":
            return events[0]
        time.sleep(0.01)
    raise AssertionError(archive.status())


def test_one_second_pause_is_one_event_and_two_second_negatives_end_it(
    tmp_path: Path,
) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        for timestamp in (0.0, 1.0, 2.0, 3.0):
            archive.submit_result(result(timestamp, f"p{timestamp}"))
        archive.submit_result(result(3.4, "negative", False))
        archive.submit_result(result(4.4, "continued"))
        assert archive.flush()
        events = archive.list_events()["events"]
        assert len(events) == 1
        assert events[0]["state"] == "active"
        archive.submit_result(result(5.0, "n1", False))
        archive.submit_result(result(7.0, "n2", False))
        archive.flush()
        event = archive.list_events()["events"][0]
        assert event["end_reason"] == "negative"
        assert event["end_seconds"] == 5.0
    finally:
        archive.close()


def test_late_positive_merges_then_late_negative_splits_original_anchor_ids(
    tmp_path: Path,
) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        archive.submit_result(result(0, "first"))
        archive.submit_result(result(6, "second"))
        archive.flush()
        first_id = archive.list_events(message_id="first")["events"][0]["event_id"]
        second_id = archive.list_events(message_id="second")["events"][0]["event_id"]
        assert first_id != second_id
        archive.submit_result(result(3, "bridge"))
        archive.flush()
        assert len(archive.list_events()["events"]) == 1
        assert archive.get_event(second_id)["canonical_event_id"] == first_id
        assert (
            archive.list_events(message_id="second")["events"][0]["event_id"]
            == first_id
        )
        archive.submit_result(result(3.5, "late-negative-1", False))
        archive.submit_result(result(5.5, "late-negative-2", False))
        archive.flush()
        assert len(archive.list_events()["events"]) == 2
        assert (
            archive.list_events(message_id="first")["events"][0]["event_id"] == first_id
        )
        assert (
            archive.list_events(message_id="second")["events"][0]["event_id"]
            == second_id
        )
        assert "canonical_event_id" not in archive.get_event(second_id)
    finally:
        archive.close()


def test_different_person_positive_resets_negative_run_for_unmentioned_person(
    tmp_path: Path,
) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        archive.submit_result(result(0, "first"))
        archive.submit_result(result(1, "n1", False))
        archive.submit_result(result(2, "someone-else", subject="person-2"))
        archive.submit_result(result(3, "n2", False))
        archive.flush()
        assert (
            archive.list_events(subject_id="person-1")["events"][0]["state"] == "active"
        )
    finally:
        archive.close()


def test_target_geometry_tracks_without_person_id(tmp_path: Path) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        archive.submit_result(result(0, "first", subject=None, label="fire"))
        archive.submit_result(result(1, "second", subject=None, label="fire"))
        archive.flush()
        events = archive.list_events()["events"]
        assert len(events) == 1
        assert events[0]["subject_type"] == "target"
    finally:
        archive.close()


def test_sse_does_not_publish_each_positive_frame(tmp_path: Path) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        archive.submit_result(result(0, "first"))
        archive.flush()
        cursor = archive.list_events()["cursor"]
        for index in range(1, 5):
            archive.submit_result(result(index / 5, f"p{index}"))
        archive.flush()
        assert archive.changes_after(cursor)["changes"] == []
        assert archive.list_events()["events"][0]["observation_count"] == 5
        assert archive.changes_after("other:0")["reset_reason"] == "epoch_changed"
    finally:
        archive.close()


def test_evidence_revision_keeps_old_available_until_new_clip_ready(
    tmp_path: Path,
) -> None:
    gate = threading.Event()
    entered = threading.Event()

    def encode(manifest: Path, destination: Path, **kwargs: Any) -> dict[str, Any]:
        metadata = json.loads(manifest.read_text())
        if metadata["revision"] == 2:
            entered.set()
            assert gate.wait(3)
        return fake_encoder(manifest, destination, **kwargs)

    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=encode, clock=lambda: 0.0
    )
    try:
        for index in range(31):
            archive.submit_frame(jpeg(), index / 5, index, "session")
        archive.submit_result(result(3, "first"))
        ready = wait_ready(archive)
        event_id = ready["event_id"]
        old = archive.video_path(event_id, 1)
        assert old and old.read_bytes() == b"video-1"
        archive.submit_result(result(2, "earlier"))
        assert entered.wait(3)
        archive.flush()
        revised = archive.get_event(event_id)
        assert revised["evidence"]["revision"] == 2
        assert revised["evidence"]["available"]["revision"] == 1
        gate.set()
        ready = wait_ready(archive)
        assert ready["evidence"]["available"]["revision"] == 2
        assert archive.video_path(event_id, 1).read_bytes() == b"video-1"
        assert archive.video_path(event_id, 2).read_bytes() == b"video-2"
    finally:
        gate.set()
        archive.close()


def test_storage_work_does_not_block_ingress(tmp_path: Path) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path))
    try:
        assert archive._initialized.wait(3)
        done = threading.Event()

        def submit() -> None:
            archive.submit_result(result(0, "first"))
            archive.submit_frame(jpeg(), 0, 0, "session")
            done.set()

        with archive._lock:
            thread = threading.Thread(target=submit)
            thread.start()
            assert done.wait(0.5), "live submission waited for the storage lock"
        thread.join()
    finally:
        archive.close()


def test_window_deadline_rejects_late_result_and_terminal_watermark_finalizes(
    tmp_path: Path,
) -> None:
    clock = [0.0]
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), clock=lambda: clock[0]
    )
    try:
        archive.note_window("session", "first", 0, ("smoking",))
        archive.submit_result(result(0, "first"), window_id="first")
        archive.note_window("session", "first", 0, ("smoking",), "completed")
        archive.submit_result(result(6, "second"))
        archive.flush()
        first = archive.list_events(message_id="first")["events"][0]
        assert first["finalized"]
        archive.note_window("session", "late", 2, ("smoking",))
        archive.flush()
        clock[0] = 66
        archive.submit_result(result(2, "late"), window_id="late")
        archive.flush()
        assert archive.status()["late_results"] == 1
    finally:
        archive.close()


def test_restart_preserves_ids_videos_and_marks_unfinished_session_interrupted(
    tmp_path: Path,
) -> None:
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path),
        encoder=fake_encoder,
        clock=lambda: 0.0,
    )
    for index in range(21):
        archive.submit_frame(jpeg(), index / 5, index, "session")
    archive.submit_result(result(1, "first"))
    event = wait_ready(archive)
    old_epoch = archive.list_events()["epoch"]
    archive.close()
    reopened = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=fake_encoder
    )
    try:
        assert reopened.flush()
        recovered = reopened.get_event(event["event_id"])
        assert recovered["finalized"]
        assert recovered["end_reason"] == "interrupted"
        assert reopened.video_path(event["event_id"], 1).read_bytes() == b"video-1"
        assert (
            reopened.changes_after(f"{old_epoch}:0")["reset_reason"] == "epoch_changed"
        )
        assert reopened.list_events(message_id="first")["count"] == 1
    finally:
        reopened.close()


def test_public_episode_matches_versioned_schema(tmp_path: Path) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        archive.submit_result(result(1, "schema"))
        archive.flush()
        schema = json.loads(
            (
                Path(__file__).resolve().parents[1] / "schemas/event-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator(schema).validate(
            archive.list_events()["events"][0]
        )
    finally:
        archive.close()


def test_long_episode_compacts_history_without_changing_id_count_or_message_lookup(
    tmp_path: Path,
) -> None:
    clock = [0.0]
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), clock=lambda: clock[0]
    )
    try:
        first_id = None
        for index in range(150):
            clock[0] = float(index)
            archive.note_window("session", index, float(index), ("smoking",))
            archive.submit_result(
                result(float(index), f"long-{index}"), window_id=index
            )
            archive.note_window(
                "session", index, float(index), ("smoking",), "completed"
            )
            archive.flush()
            if first_id is None:
                first_id = archive.list_events()["events"][0]["event_id"]
        events = archive.list_events()["events"]
        assert len(events) == 1
        assert events[0]["event_id"] == first_id
        assert events[0]["observation_count"] == 150
        assert (
            archive.list_events(message_id="long-40")["events"][0]["event_id"]
            == first_id
        )
        assert len(archive._observations) < 145
        assert len(archive._sessions["session"].windows) < 70
        archive.submit_result(result(40, "long-40", subject="wrong-late-person"))
        archive.flush()
        assert archive.status()["late_results"] == 1
        assert len(archive.list_events()["events"]) == 1
    finally:
        archive.close()


def test_late_person_association_reuses_event_and_video_revision(
    tmp_path: Path,
) -> None:
    clock = [0.0]
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), clock=lambda: clock[0]
    )
    try:
        archive.submit_result(result(0, "first", subject=None))
        archive.flush()
        before = archive.list_events()["events"][0]
        clock[0] = 10
        corrected = result(0, "first", subject="person-1")
        corrected["objects"][0]["attributes"]["subject_identity"] = {
            "status": "known",
            "person_id": "alice",
            "name": "Alice",
        }
        archive.submit_result(corrected)
        archive.flush()
        after = archive.list_events()["events"][0]
        assert after["event_id"] == before["event_id"]
        assert after["subject_id"] == "person-1"
        assert after["person_id"] == "alice" and after["person_name"] == "Alice"
        assert after["evidence"]["revision"] == before["evidence"]["revision"]
        clock[0] = 66
        archive.submit_result(result(0, "first", subject="too-late"))
        archive.flush()
        assert archive.status()["late_results"] == 1
        assert archive.get_event(before["event_id"])["subject_id"] == "person-1"
    finally:
        archive.close()


def test_fight_participants_share_one_event_without_single_person_identity(
    tmp_path: Path,
) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        payload = result(1, "fight", label="fight")
        payload["objects"][0]["attributes"].update(
            subject_ids=["person-1", "person-2"], person_id="alice", name="Alice"
        )
        second = dict(payload["objects"][0], object_id="o2")
        payload["objects"].append(second)
        payload["judgments"][0]["object_ids"].append("o2")
        archive.submit_result(payload)
        archive.flush()
        events = archive.list_events()["events"]
        assert len(events) == 1 and events[0]["subject_type"] == "group"
        assert events[0]["subject_ids"] == ["person-1", "person-2"]
        assert events[0]["person_id"] is None and events[0]["person_name"] is None
        assert (
            archive.list_events(subject_id="person-1")["events"][0]["event_id"]
            == events[0]["event_id"]
        )
        assert archive.list_events(subject_id="person-2")["count"] == 1
    finally:
        archive.close()


def test_fast_completed_active_window_remains_reassociable_for_wall_deadline(
    tmp_path: Path,
) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        for index in range(30):
            archive.note_window("session", index, float(index), ("smoking",))
            archive.submit_result(
                result(float(index), f"fast-{index}"), window_id=index
            )
            archive.note_window(
                "session", index, float(index), ("smoking",), "completed"
            )
        archive.flush()
        archive._compact_observations()
        correction = result(4, "fast-4")
        correction["objects"][0]["attributes"]["subject_identity"] = {
            "status": "known",
            "person_id": "alice",
            "name": "Alice",
        }
        archive.submit_result(correction)
        archive.flush()
        assert archive.status()["late_results"] == 0
    finally:
        archive.close()


def test_restart_resumes_staged_revision_and_preserves_published_old_bytes(
    tmp_path: Path,
) -> None:
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path),
        encoder=fake_encoder,
        clock=lambda: 0.0,
    )
    for index in range(21):
        archive.submit_frame(jpeg(), index / 5, index, "session")
    archive.submit_result(result(1, "first"))
    event = wait_ready(archive)
    event_id = event["event_id"]
    archive.close()
    # Crash fixture: the new task is durable, but has not yet committed a video.
    with sqlite3.connect(tmp_path / "events.sqlite3") as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload FROM events WHERE event_id=?", (event_id,)
            ).fetchone()[0]
        )
        payload["evidence"].update(revision=2, status="encoding")
        connection.execute(
            "UPDATE events SET payload=? WHERE event_id=?",
            (json.dumps(payload), event_id),
        )
    frames = [EvidenceFrame(jpeg(), index / 5, index) for index in range(15)]
    for revision in (1, 2):
        stage_clip(
            tmp_path / "staging" / f"{event_id}-r{revision}",
            frames,
            {
                "event_id": event_id,
                "revision": revision,
                "fps": 5,
                "requested_start_seconds": 0,
                "requested_end_seconds": 3,
                "partial_reason": None,
            },
        )
    encoded = []

    def encode(manifest: Path, destination: Path, **kwargs: Any) -> dict[str, Any]:
        encoded.append(json.loads(manifest.read_text())["revision"])
        return fake_encoder(manifest, destination, **kwargs)

    reopened = EventArchive(EventArchiveSettings(directory=tmp_path), encoder=encode)
    try:
        ready = wait_ready(reopened)
        assert encoded == [2]
        assert ready["evidence"]["available"]["revision"] == 2
        assert reopened.video_path(event_id, 1).read_bytes() == b"video-1"
        assert reopened.video_path(event_id, 2).read_bytes() == b"video-2"
    finally:
        reopened.close()


def test_unavailable_encoder_reports_failure_without_stopping_archive(
    tmp_path: Path,
) -> None:
    archive = EventArchive(
        EventArchiveSettings(
            directory=tmp_path, ffmpeg_path=str(tmp_path / "missing-encoder")
        ),
        clock=lambda: 0.0,
    )
    try:
        for index in range(21):
            archive.submit_frame(jpeg(), index / 5, index, "session")
        archive.submit_result(result(1, "first"))
        until = time.monotonic() + 3
        while time.monotonic() < until:
            archive.flush()
            event = archive.list_events()["events"][0]
            if event["evidence"]["status"] == "failed":
                break
            time.sleep(0.01)
        assert event["evidence"]["failure_reason"] == "encoder_unavailable"
        assert archive.status()["state"] == "running"
        assert archive.submit_result(result(5, "still-working"))
    finally:
        archive.close()


def test_video_lease_covers_path_lookup_until_response_finishes(tmp_path: Path) -> None:
    clock = [0.0]
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path, retention_days=1e-7),
        encoder=fake_encoder,
        clock=lambda: clock[0],
    )
    try:
        for index in range(21):
            archive.submit_frame(jpeg(), index / 5, index, "session")
        archive.submit_result(result(1, "first"))
        event = wait_ready(archive)
        leased = archive.acquire_video(event["event_id"], 1)
        assert leased is not None
        path, release = leased
        archive.end_session("session")
        time.sleep(0.02)
        clock[0] = 2
        archive.flush()
        assert path.is_file() and archive.get_event(event["event_id"]) is not None
        release()
        release()
        clock[0] = 4
        archive.flush()
        assert archive.get_event(event["event_id"]) is None
        assert not path.exists()
    finally:
        archive.close()


def test_quota_counts_encoder_temporaries_and_keeps_failed_event_visible(
    tmp_path: Path,
) -> None:
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path, max_storage_bytes=2 * 1024 * 1024),
        encoder=fake_encoder,
        clock=lambda: 0.0,
    )
    try:
        archive.flush()
        (tmp_path / "videos" / ".encoder.tmp.mp4").write_bytes(b"x" * (1792 * 1024))
        for index in range(21):
            archive.submit_frame(jpeg(), index / 5, index, "session")
        archive.submit_result(result(1, "first"))
        archive.flush()
        event = archive.list_events()["events"][0]
        assert event["evidence"]["status"] == "failed"
        assert event["evidence"]["failure_reason"] == "storage_quota_exceeded"
        assert archive.get_event(event["event_id"]) is not None
        assert archive.status()["storage_bytes"] >= 1792 * 1024
    finally:
        archive.close()
