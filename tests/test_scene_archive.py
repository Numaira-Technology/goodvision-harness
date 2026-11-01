"""Persistence, metadata privacy, queue backpressure, and journal retention."""

import json
import sqlite3
import time
from threading import Event
from types import SimpleNamespace

import pytest

from goodvision_harness import scene_archive
from goodvision_harness.scene_archive import SceneArchive


def snapshot(*, event_id="event-1", phase="suspected", session="session-1", sequence=1):
    return {
        "stream_session_id": session,
        "source_id": "camera",
        "events": [
            {
                "event_id": event_id,
                "event_label": "smoking",
                "phase": phase,
                "subject_ids": ["subject-1"],
                "first_evidence_timestamp_seconds": 1,
                "last_evidence_timestamp_seconds": sequence,
                "frame_sequence": sequence,
                "confidence": 0.9,
                "positive_observations": 1,
                "negative_observations": 0,
                "observation_state": "observed",
            }
        ],
        "transitions": [],
    }


def wait_for(predicate, *, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        Event().wait(0.01)
    assert predicate(), "archive worker did not finish before timeout"


def persisted(archive, count=1):
    wait_for(
        lambda: (
            archive.stats()["persisted_batches"] >= count
            or archive.stats()["failures"] > 0
        )
    )
    assert archive.stats()["failures"] == 0, archive.stats()


def test_flush_waits_for_final_scene_write_without_adding_queue_barriers(tmp_path):
    archive = SceneArchive(tmp_path)
    gate, entered = Event(), Event()
    write = archive._write

    def delayed_write(connection, records):
        entered.set()
        assert gate.wait(3)
        write(connection, records)

    archive._write = delayed_write
    try:
        assert archive.submit(snapshot(phase="ended"), timeout_seconds=0.1)
        assert entered.wait(3)
        assert not archive.flush(0.02)
        assert archive.stats()["queued"] == 0
        gate.set()
        assert archive.flush(3)
        assert archive.list_events()[0]["phase"] == "ended"
    finally:
        gate.set()
        archive.close()


def test_event_upsert_retraction_and_restart_keep_session_history(tmp_path):
    archive = SceneArchive(tmp_path)
    try:
        assert archive.submit(snapshot())
        assert archive.submit(snapshot(phase="confirmed", sequence=2))
        assert archive.submit(snapshot(session="session-2", phase="ended", sequence=3))
        persisted(archive, 3)
        assert len(archive.list_events()) == 2
        by_session = {item["stream_session_id"]: item for item in archive.list_events()}
        assert by_session["session-1"]["phase"] == "confirmed"
        assert by_session["session-2"]["phase"] == "ended"
    finally:
        archive.close()
    reopened = SceneArchive(tmp_path)
    try:
        assert len(reopened.list_events()) == 2
        corrected = snapshot()
        corrected["events"] = []
        corrected["transitions"] = [
            {
                "type": "retract",
                "event_id": "event-1",
                "event_label": "smoking",
                "subject_ids": ["subject-1"],
                "timestamp_seconds": 2,
                "frame_sequence": 2,
            }
        ]
        assert reopened.submit(corrected)
        persisted(reopened)
        first = next(
            item
            for item in reopened.list_events()
            if item["stream_session_id"] == "session-1"
        )
        assert first["phase"] == "retracted"
        assert first["first_evidence_timestamp_seconds"] == 1
        assert "_retraction_only" not in first
    finally:
        reopened.close()


def test_current_state_takes_precedence_over_old_retraction_transition(tmp_path):
    archive = SceneArchive(tmp_path)
    try:
        current = snapshot(phase="confirmed", sequence=3)
        current["transitions"] = [
            {
                "type": "retract",
                "event_id": "event-1",
                "event_label": "smoking",
                "subject_ids": ["subject-1"],
                "timestamp_seconds": 1,
                "frame_sequence": 1,
            }
        ]
        assert archive.submit({"scene": current, "irrelevant": "secret"})
        persisted(archive)
        assert archive.list_events()[0]["phase"] == "confirmed"
        assert archive.list_events()[0]["frame_sequence"] == 3
    finally:
        archive.close()


def test_queue_and_database_only_receive_allowlisted_metadata(tmp_path):
    archive = SceneArchive(tmp_path)
    try:
        current = snapshot()
        current["events"][0].update(
            identity={"name": "PRIVATE-NAME", "person_id": "PRIVATE-ID"},
            history=[{"jpeg_bytes": "PRIVATE-JPEG"}],
            api_key="PRIVATE-KEY",
            embedding=[1.0, 2.0],
        )
        current["entities"] = [{"name": "PRIVATE-NAME", "jpeg_bytes": b"PRIVATE-JPEG"}]
        assert archive.submit(current)
        persisted(archive)
        stored = archive.list_events()
        encoded = json.dumps(stored)
        assert "PRIVATE" not in encoded
        assert "identity" not in encoded
        stored[0]["subject_ids"].append("caller-mutated")
        assert archive.list_events()[0]["subject_ids"] == ["subject-1"]
    finally:
        archive.close()
    assert b"PRIVATE" not in (tmp_path / "scene-events.sqlite3").read_bytes()


def test_retention_removes_only_expired_metadata(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(
        scene_archive,
        "time",
        SimpleNamespace(
            time=lambda: clock[0],
            monotonic=time.monotonic,
        ),
    )
    archive = SceneArchive(tmp_path, retention_days=1 / 86400)
    try:
        assert archive.submit(snapshot(event_id="old"))
        persisted(archive)
        clock[0] = 1002.0
        assert archive.submit(snapshot(event_id="current"))
        persisted(archive, 2)
        assert [item["event_id"] for item in archive.list_events()] == ["current"]
        assert archive.stats()["retention_deleted"] == 1
    finally:
        archive.close()


def test_capacity_evicts_old_rows_and_database_stays_within_storage_budget(tmp_path):
    untouched = tmp_path / "user-notes.txt"
    untouched.write_text("preserve this file")
    archive = SceneArchive(tmp_path, max_storage_bytes=64 * 1024)
    try:
        batch = snapshot()
        batch["events"] = [
            snapshot(event_id=f"event-{index}")["events"][0] for index in range(150)
        ]
        assert archive.submit(batch)
        persisted(archive)
        state = archive.stats()
        assert 0 < state["stored_events"] < 150
        assert state["capacity_evicted"] > 0
        assert state["storage_bytes"] <= 64 * 1024
        assert archive.list_events()[0]["event_id"] == "event-149"
    finally:
        archive.close()
    assert untouched.read_text() == "preserve this file"
    assert (tmp_path / "scene-events.sqlite3").stat().st_size <= 64 * 1024


def test_queue_full_and_shutdown_are_bounded(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    original = SceneArchive._write

    def blocked(self, connection, records):
        entered.set()
        assert release.wait(3)
        return original(self, connection, records)

    monkeypatch.setattr(SceneArchive, "_write", blocked)
    archive = SceneArchive(tmp_path, queue_size=1)
    try:
        assert archive.submit(snapshot(event_id="one"))
        assert entered.wait(1)
        assert archive.submit(snapshot(event_id="two"))
        before = time.monotonic()
        assert not archive.submit(snapshot(event_id="three"))
        archive.close(timeout_seconds=0.01)
        assert time.monotonic() - before < 0.5
        assert archive.stats()["queue_dropped"] == 1
        assert archive.stats()["close_timeouts"] == 1
        assert not archive.submit(snapshot(event_id="after-close"))
    finally:
        release.set()
        # The bounded close is asserted above; cleanup must await durable writes
        # even when the Windows runner's filesystem takes longer than a second.
        archive.close(timeout_seconds=10)
    assert archive.stats()["state"] == "closed"
    assert archive.stats()["persisted_batches"] == 2


def test_failed_disk_write_is_counted_and_can_be_retried(tmp_path, monkeypatch):
    original = SceneArchive._write
    attempts = []

    def fail_once(self, connection, records):
        attempts.append(None)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("do not expose secret connection details")
        return original(self, connection, records)

    monkeypatch.setattr(SceneArchive, "_write", fail_once)
    archive = SceneArchive(tmp_path)
    try:
        assert archive.submit(snapshot())
        wait_for(lambda: archive.stats()["failures"] == 1)
        assert archive.stats()["persisted_batches"] == 0
        assert not archive.list_events()
        assert archive.stats()["last_error"] == "OperationalError"
        assert archive.failed_for_run("session-1")
        assert not archive.failed_for_run("next-run")
        assert archive.submit(snapshot())
        wait_for(lambda: archive.stats()["persisted_batches"] == 1)
        assert not archive.failed_for_run("session-1")
        assert len(archive.list_events()) == 1
    finally:
        archive.close()


def test_invalid_or_empty_snapshots_do_not_create_fake_events(tmp_path):
    archive = SceneArchive(tmp_path)
    try:
        empty = snapshot()
        empty["events"] = []
        assert not archive.submit(empty)
        malformed = snapshot()
        malformed["events"][0]["confidence"] = float("nan")
        assert not archive.submit(malformed)
        assert not archive.submit({"scene": None})
        assert not archive.list_events()
        assert archive.stats()["submitted"] == 0
        assert archive.stats()["invalid_snapshots"] == 2
    finally:
        archive.close()


@pytest.mark.parametrize(
    "options",
    [
        {"retention_days": 0},
        {"retention_days": float("inf")},
        {"max_storage_bytes": 100},
        {"queue_size": True},
        {"queue_size": 0},
    ],
)
def test_invalid_archive_limits_are_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        SceneArchive(tmp_path, **options)
