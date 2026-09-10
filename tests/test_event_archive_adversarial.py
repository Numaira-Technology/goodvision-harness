"""Adversarial source-time, storage-pressure and short-recording scenarios."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time

import cv2
import numpy as np
import pytest

from goodvision_harness import event_archive as archive_module
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings


def _result(timestamp, *, positive=True, label="smoking"):
    return {
        "message_id": f"m-{label}-{timestamp}",
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
                "detected": positive,
                "object_ids": ["p"] if positive else [],
                "confidence": 0.9,
                "description": "Visible action",
            }
        ],
        "objects": [
            {
                "object_id": "p",
                "label": "person",
                "attributes": {
                    "subject_id": "person-1",
                    "subject_type": "person",
                },
            }
        ]
        if positive
        else [],
    }


def _admit(archive, timestamps, label="smoking"):
    for timestamp in timestamps:
        assert archive.note_window(
            "session", f"{label}-{timestamp}", timestamp, (label,)
        )


def _complete(archive, timestamp, *, positive=True, label="smoking"):
    window = f"{label}-{timestamp}"
    assert archive.submit_result(
        _result(timestamp, positive=positive, label=label), window_id=window
    )
    assert archive.note_window("session", window, timestamp, (label,), "completed")


def _jpeg(value=0):
    ok, encoded = cv2.imencode(".jpg", np.full((90, 160, 3), value, np.uint8))
    assert ok
    return encoded.tobytes()


def test_pending_window_inside_negative_confirmation_can_rejoin_both_segments(tmp_path):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        _admit(archive, (0.0, 1.0, 2.0, 3.0, 4.0))
        _complete(archive, 0.0)
        _complete(archive, 1.0, positive=False)
        _complete(archive, 3.0, positive=False)
        _complete(archive, 4.0)
        assert archive.flush()
        first = archive.list_events(message_id="m-smoking-0.0")["events"][0]
        assert not first["finalized"], (
            "t=2 may still invalidate the t=1..3 negative run"
        )
        _complete(archive, 2.0)
        assert archive.flush()
        events = archive.list_events()["events"]
        assert len(events) == 1
        assert events[0]["event_id"] == first["event_id"]
        assert events[0]["last_detected_seconds"] == 4.0
        assert events[0]["state"] == "active"
    finally:
        archive.close()


def test_timed_out_intermediate_request_breaks_explicit_negative_run(tmp_path):
    clock = [0.0]
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), clock=lambda: clock[0]
    )
    try:
        _admit(archive, (0.0, 1.0, 2.0, 3.0))
        _complete(archive, 0.0)
        _complete(archive, 1.0, positive=False)
        _complete(archive, 3.0, positive=False)
        assert archive.flush()
        clock[0] = 66.0
        assert archive.flush()
        event = archive.list_events()["events"][0]
        assert event["end_reason"] == "timeout", (
            "an unavailable t=2 is not confirmed negative"
        )
    finally:
        archive.close()


def test_other_label_pending_request_does_not_block_smoking_finalization(tmp_path):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        _admit(archive, (0.0, 1.0, 3.0))
        _admit(archive, (2.0,), label="stranger")
        _complete(archive, 0.0)
        _complete(archive, 1.0, positive=False)
        _complete(archive, 3.0, positive=False)
        assert archive.flush()
        assert archive.list_events()["events"][0]["finalized"]
    finally:
        archive.close()


def test_stalled_durable_staging_never_blocks_live_frame_or_result_submission(
    tmp_path, monkeypatch
):
    entered, release, submitted = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original = archive_module.stage_clip

    def slow_stage(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)

    monkeypatch.setattr(archive_module, "stage_clip", slow_stage)
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        for index in range(17):
            assert archive.submit_frame(_jpeg(), index / 5, index, "session")
        assert archive.submit_result(_result(1.0))
        assert entered.wait(2)

        def submit():
            assert archive.submit_result(_result(2.0))
            assert archive.submit_frame(_jpeg(), 3.4, 17, "session")
            submitted.set()

        thread = threading.Thread(target=submit)
        thread.start()
        assert submitted.wait(0.25), "live submission waited for archive fsync"
        thread.join()
    finally:
        release.set()
        archive.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="video encoder unavailable")
def test_stopping_shortly_after_alert_produces_actual_short_playable_clip(tmp_path):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        for index in range(3):
            assert archive.submit_frame(_jpeg(index * 80), index / 5, index, "session")
        assert archive.submit_result(_result(0.2))
        assert archive.end_session("session")
        deadline = time.monotonic() + 5
        event = None
        while time.monotonic() < deadline:
            assert archive.flush()
            events = archive.list_events()["events"]
            event = events[0] if events else None
            if event and event["evidence"]["status"] == "ready":
                break
            time.sleep(0.01)
        assert event and event["evidence"]["status"] == "ready"
        available = event["evidence"]["available"]
        assert available["partial"]
        assert available["duration_seconds"] == pytest.approx(0.6)
        assert available["frame_count"] == 3
        capture = cv2.VideoCapture(str(archive.video_path(event["event_id"])))
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        assert len(frames) == 3
        assert frames[-1].mean() > 120
    finally:
        archive.close()


def test_age_retention_removes_metadata_even_when_no_video_was_possible(tmp_path):
    clock = [0.0]
    archive = EventArchive(
        EventArchiveSettings(
            directory=tmp_path,
            retention_days=1e-7,
        ),
        clock=lambda: clock[0],
    )
    try:
        assert archive.submit_result(_result(0.0))
        assert archive.end_session("session")
        assert archive.flush()
        first = archive.list_events()
        event_id = first["events"][0]["event_id"]
        assert archive.get_event(event_id)["evidence"]["status"] == "failed"
        time.sleep(0.02)
        clock[0] = 2.0
        assert archive.flush()
        assert archive.get_event(event_id) is None
        assert archive.list_events(message_id="m-smoking-0.0")["events"] == []
        assert archive.changes_after(f"{first['epoch']}:{first['cursor']}")[
            "reset_reason"
        ]
        with sqlite3.connect(tmp_path / "events.sqlite3") as connection:
            for table in ("events", "anchors", "observations", "aliases", "links"):
                assert (
                    connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                    == 0
                ), table
    finally:
        archive.close()


def test_restart_retention_also_deletes_persisted_observations(tmp_path):
    first = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    assert first.submit_result(_result(0.0))
    assert first.end_session("session")
    assert first.flush()
    first.close()
    time.sleep(0.02)
    clock = [0.0]
    reopened = EventArchive(
        EventArchiveSettings(
            directory=tmp_path,
            retention_days=1e-7,
        ),
        clock=lambda: clock[0],
    )
    try:
        assert reopened.flush()
        clock[0] = 2.0
        assert reopened.flush()
        assert reopened.list_events()["events"] == []
        with sqlite3.connect(tmp_path / "events.sqlite3") as connection:
            assert (
                connection.execute("SELECT count(*) FROM observations").fetchone()[0]
                == 0
            )
    finally:
        reopened.close()


def test_long_episode_storage_updates_are_incremental(tmp_path):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        assert archive.flush()
        before = archive._db.total_changes
        samples = 80
        for index in range(samples):
            assert archive.submit_result(_result(index / 5))
        assert archive.flush()
        changes = archive._db.total_changes - before
        assert archive.list_events()["count"] == 1
        # Generous fixed work budget per observation: writing all historical
        # observations and links on every new frame grows quadratically.
        assert changes <= samples * 32, (
            f"{samples} samples rewrote {changes} database rows"
        )
    finally:
        archive.close()


def test_frame_ingress_memory_budget_includes_waiting_jpegs(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = EventArchive._open_store

    def paused_open(archive):
        entered.set()
        assert release.wait(5)
        return original(archive)

    monkeypatch.setattr(EventArchive, "_open_store", paused_open)
    archive = EventArchive(
        EventArchiveSettings(
            directory=tmp_path,
            buffer_max_bytes=8192,
            queue_size=64,
        )
    )
    try:
        assert entered.wait(2)
        jpeg = _jpeg()
        for index in range(32):
            archive.submit_frame(jpeg, index / 5, index, "session")
        with archive._queue.mutex:
            queued_bytes = sum(
                len(payload[1].jpeg)
                for kind, payload in archive._queue.queue
                if kind == "frame"
            )
        assert queued_bytes <= archive.settings.buffer_max_bytes, (
            f"{queued_bytes} queued JPEG bytes exceed "
            f"{archive.settings.buffer_max_bytes} budget"
        )
    finally:
        release.set()
        archive.close()


def test_close_with_saturated_ingress_still_terminates_owner_thread(
    tmp_path, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    original = EventArchive._open_store

    def paused_open(archive):
        entered.set()
        assert release.wait(5)
        return original(archive)

    monkeypatch.setattr(EventArchive, "_open_store", paused_open)
    archive = EventArchive(EventArchiveSettings(directory=tmp_path, queue_size=1))
    try:
        assert entered.wait(2)
        assert archive.submit_frame(_jpeg(), 0.0, 0, "session")
        archive.close(timeout=0.01)
        release.set()
        archive._thread.join(timeout=2)
        assert not archive._thread.is_alive(), (
            "full ingress prevented close signal delivery"
        )
    finally:
        release.set()
        if archive._thread.is_alive():
            # Ensure a failed regression does not leave a daemon writing into
            # the test directory after its test has ended.
            archive._queue.put(("close", None), timeout=2)
            archive._thread.join(timeout=2)
        archive._encoder_thread.join(timeout=2)


def test_fast_file_keeps_pending_window_evidence_beyond_source_ring_horizon(tmp_path):
    def encode(manifest, destination, **kwargs):
        metadata = json.loads(manifest.read_text())
        records = metadata["frames"]
        destination.write_bytes(b"test-video")
        return {
            "fps": metadata["fps"],
            "frame_count": len(records),
            "start_seconds": records[0]["timestamp_seconds"],
            "end_seconds": records[-1]["timestamp_seconds"] + 0.2,
            "partial": bool(metadata.get("partial_reason")),
        }

    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=encode, clock=lambda: 0.0
    )
    try:
        jpeg = _jpeg()
        for index in range(6):
            assert archive.submit_frame(jpeg, index / 5, index, "session")
        assert archive.note_window("session", "slow", 1.0, ("smoking",))
        for index in range(6, 476):
            assert archive.submit_frame(jpeg, index / 5, index, "session")
        assert archive.flush()
        # Only media time advanced: the inference remains inside its 65-second
        # wall deadline, although the ordinary 90-second source ring rolled on.
        assert archive.submit_result(_result(1.0), window_id="slow")
        assert archive.note_window("session", "slow", 1.0, ("smoking",), "completed")
        deadline = time.monotonic() + 3
        event = None
        while time.monotonic() < deadline:
            assert archive.flush()
            events = archive.list_events()["events"]
            event = events[0] if events else None
            if event and event["evidence"]["status"] in {"ready", "failed"}:
                break
            time.sleep(0.01)
        assert event and event["evidence"]["status"] == "ready", event
        assert event["evidence"]["available"]["start_seconds"] == 0.0
        assert event["evidence"]["available"]["end_seconds"] == 3.0
    finally:
        archive.close()
