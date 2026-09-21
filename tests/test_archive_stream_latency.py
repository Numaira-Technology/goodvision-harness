"""Archive JSON/SSE must progress while durable video staging is stalled."""

from __future__ import annotations

import json
import socket
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from test_event_archive import fake_encoder, jpeg, result, wait_ready
from test_scene_event_archive import episode, scene

from goodvision_harness import event_archive as archive_module
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.event_routes import install_event_routes


@contextmanager
def event_server(archive, release=None):
    app = FastAPI()
    app.state.runtime = SimpleNamespace(event_archive=lambda: archive)
    install_event_routes(app)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        # This in-process server must not replace pytest's logging handlers or
        # disable propagation for tests that run after the HTTP regression.
        app, lifespan="off", ws="none", log_config=None, log_level=None,
    ))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True,
    )
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{port}"
    finally:
        if release is not None:
            release.set()
        server.should_exit = True
        thread.join(5)
        sock.close()
        assert not thread.is_alive()


def next_change(lines, label):
    for line in lines:
        if line.startswith("data: "):
            change = json.loads(line[6:])
            if any(event["event_label"] == label for event in change.get("events", [])):
                return change
    raise AssertionError("event stream closed before the expected event")


def test_json_and_open_sse_publish_new_events_before_video_staging_finishes(
    tmp_path, monkeypatch,
):
    entered, release = threading.Event(), threading.Event()
    original = archive_module.stage_clip

    def stalled_stage(*args):
        entered.set()
        assert release.wait(10), "test did not release video staging"
        return original(*args)

    monkeypatch.setattr(archive_module, "stage_clip", stalled_stage)
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=fake_encoder,
        clock=lambda: 0.0,
    )
    try:
        for index in range(21):
            assert archive.submit_frame(jpeg(), index / 5, index, "session")
        assert archive.submit_scene(scene(episode("smoke", "smoking", 1, 2)))
        assert entered.wait(3)
        with event_server(archive, release) as url, httpx.Client(timeout=2) as client:
            with client.stream("GET", url + "/v1/events/stream") as response:
                assert response.status_code == 200
                lines = response.iter_lines()
                first = next_change(lines, "smoking")
                # The first video is still waiting on fsync. A distinct, newly
                # confirmed event must reach the same open SSE connection.
                assert archive.submit_scene(scene(episode("weapon", "weapon", 2, 3)))
                second = next_change(lines, "weapon")
                assert second["cursor"] > first["cursor"]
                page = client.get(url + "/v1/events?event_label=weapon").json()
                assert len(page["events"]) == 1
                event = page["events"][0]
                assert event["status"] == "active"
                assert event["evidence"]["available"] is None
                detail = client.get(url + "/v1/events/" + event["event_id"]).json()
                assert detail == event
                assert not release.is_set()
                assert not archive._sessions["session"].ended
        release.set()
        assert archive.wait_until_idle(5)
        assert all(
            e["evidence"]["status"] == "ready" for e in archive.list_events()["events"]
        )
    finally:
        release.set()
        archive.close()


def test_readers_use_immutable_view_while_archive_owner_is_blocked(tmp_path):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        archive.submit_result(result(1, "message"))
        assert archive.flush()
        before = archive.list_events()
        event_id = before["events"][0]["event_id"]
        with ThreadPoolExecutor() as pool:
            with archive._lock:
                # Model slow writer maintenance on another thread; API calls
                # must still finish, including an indexed historical lookup.
                page = pool.submit(archive.list_events, message_id="message").result(1)
                detail = pool.submit(archive.get_event, event_id).result(1)
                changes = pool.submit(archive.changes_after).result(1)
                assert page["events"][0] == detail
                assert changes["changes"]
        page["events"][0]["evidence"]["status"] = "corrupted-by-reader"
        changes["changes"][0]["events"][0]["subject_ids"].clear()
        assert archive.get_event(event_id) == detail
        assert archive.changes_after()["changes"][0]["events"][0]["subject_ids"]
    finally:
        archive.close()


@pytest.mark.parametrize("already_staged", [False, True])
@pytest.mark.parametrize("event_revised", [False, True])
def test_durable_plan_resumes_after_restart(
    tmp_path, monkeypatch, already_staged, event_revised,
):
    class PausedEvidenceArchive(EventArchive):
        def _encode_loop(self):
            self._closed.wait()

    archive = PausedEvidenceArchive(
        EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0,
    )
    try:
        for index in range(21):
            archive.submit_frame(jpeg(), index / 5, index, "session")
        archive.submit_result(result(1, "first"))
        assert archive.flush()
        event = archive.list_events()["events"][0]
        assert event["evidence"]["status"] == "encoding"
        assert not list((tmp_path / "staging").glob("*/manifest.json"))
        if already_staged:
            with archive._lock:
                metadata = next(iter(archive._jobs.values()))
                frames = archive._queue.source_frames(
                    "session", tuple(metadata["source_sequences"]),
                )
            archive_module.stage_clip(
                tmp_path / "staging" / f"{event['event_id']}-r1", frames, metadata,
            )

            def must_not_stage_again(*args):
                raise AssertionError("durable manifest should resume at encoding")

            monkeypatch.setattr(archive_module, "stage_clip", must_not_stage_again)
    finally:
        archive.close()
    with sqlite3.connect(tmp_path / "events.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_jobs").fetchone()[0] == 1
        if event_revised:
            stored = json.loads(db.execute(
                "SELECT payload FROM events WHERE event_id=?", (event["event_id"],),
            ).fetchone()[0])
            stored["finalized"] = True
            stored["evidence"].update(
                revision=2, status="expired", available=None,
                failure_reason="identity_confirmed",
            )
            db.execute("UPDATE events SET payload=? WHERE event_id=?", (
                json.dumps(stored), event["event_id"],
            ))
    reopened = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=fake_encoder,
    )
    try:
        if event_revised:
            assert reopened.wait_until_idle(5)
            ready = reopened.get_event(event["event_id"])
            assert ready["evidence"]["revision"] == 2
            assert ready["evidence"]["status"] == "expired"
            assert ready["evidence"]["available"] is None
        else:
            ready = wait_ready(reopened)
            assert ready["evidence"]["available"]["revision"] == 1
        assert ready["event_id"] == event["event_id"]
        old_video = reopened.video_path(ready["event_id"], 1)
        assert old_video is not None, "admitted historical jobs must survive restart"
        assert old_video.read_bytes() == b"video-1"
        assert reopened.wait_until_idle(3)
        with sqlite3.connect(tmp_path / "events.sqlite3") as db:
            assert db.execute("SELECT COUNT(*) FROM evidence_jobs").fetchone()[0] == 0
    finally:
        reopened.close()


def test_revised_event_keeps_sources_for_an_older_unstaged_video(tmp_path):
    class PausedEvidenceArchive(EventArchive):
        def _encode_loop(self):
            self._closed.wait()

    archive = PausedEvidenceArchive(
        EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0,
    )
    try:
        for index in range(21):
            archive.submit_frame(jpeg(), index / 5, index, "session")
        archive.submit_result(result(1, "first"))
        assert archive.flush()
        with archive._lock:
            metadata = next(iter(archive._jobs.values()))
            sequences = tuple(metadata["source_sequences"])
            expected = archive._queue.source_frames("session", sequences)
            # A later revision/retraction no longer needs the old window, but
            # its admitted video job has not yet copied these durable sources.
            event = next(iter(archive._events.values()))
            event["finalized"] = True
            event["evidence"].update(
                revision=2, status="expired", failure_reason="identity_confirmed",
                requested_window={"start_seconds": 100, "end_seconds": 103},
            )
            archive._sessions["session"].ended = True
            archive._sessions["session"].windows.clear()
            archive._observations.clear()
            archive._prune_ingress()
            assert archive._queue.source_frames("session", sequences) == expected
            assert archive._queue.source_window("session", 4, 5) == []
    finally:
        archive.close()


def test_failed_commit_does_not_mark_nested_event_changes_as_persisted(tmp_path):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        archive.submit_scene(scene(episode("smoke", "smoking", 1, 2)))
        assert archive.flush()
        event_id = archive.list_events()["events"][0]["event_id"]
        with archive._lock:
            archive._events[event_id]["description"] = "updated"
            archive._events[event_id]["evidence"]["failure_reason"] = "changed"
            archive._scene_events[event_id]["phase"] = "ended"
            archive._db.execute("""
                CREATE TRIGGER fail_scene BEFORE INSERT ON scene_episodes
                BEGIN SELECT RAISE(ABORT, 'injected failure'); END
            """)
            with pytest.raises(sqlite3.IntegrityError):
                archive._persist()
            archive._db.execute("DROP TRIGGER fail_scene")
            archive._persist()
            stored = json.loads(archive._db.execute(
                "SELECT payload FROM events WHERE event_id=?", (event_id,),
            ).fetchone()[0])
            assert stored["description"] == "updated"
            assert stored["evidence"]["failure_reason"] == "changed"
    finally:
        archive.close()


def test_restart_drains_more_persisted_plans_than_encoder_queue_capacity(tmp_path):
    class PausedEvidenceArchive(EventArchive):
        def _encode_loop(self):
            self._closed.wait()

    archive = PausedEvidenceArchive(
        EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0,
    )
    try:
        for index in range(21):
            archive.submit_frame(jpeg(), index / 5, index, "session")
        archive.submit_result(result(1, "first"))
        assert archive.flush()
    finally:
        archive.close()
    # A crash can leave a running job plus a full queue of durable plans.
    with sqlite3.connect(tmp_path / "events.sqlite3") as db:
        event = json.loads(db.execute("SELECT payload FROM events").fetchone()[0])
        plan = json.loads(db.execute("SELECT payload FROM evidence_jobs").fetchone()[0])
        for index in range(33):
            event_id = f"event-{index:032x}"
            db.execute("INSERT INTO events VALUES (?,?)", (
                event_id, json.dumps({**event, "event_id": event_id}),
            ))
            db.execute("INSERT INTO evidence_jobs VALUES (?,?,?)", (
                event_id, 1, json.dumps({**plan, "event_id": event_id}),
            ))
    entered, release = threading.Event(), threading.Event()

    def stalled_encode(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return fake_encoder(*args, **kwargs)

    reopened = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=stalled_encode,
    )
    try:
        assert entered.wait(3)
        assert reopened.flush()
        assert len(reopened._jobs) == 34
        release.set()
        assert reopened.wait_until_idle(10)
        assert len(reopened.list_events()["events"]) == 34
        assert all(
            e["evidence"]["status"] == "ready" for e in reopened.list_events()["events"]
        )
    finally:
        release.set()
        reopened.close()


def test_failure_transition_remains_readable_when_persistence_is_unavailable(
    tmp_path, monkeypatch,
):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0.0)
    try:
        assert archive.flush()
        original = archive._persist

        def fail_storage():
            raise OSError("injected storage failure")

        monkeypatch.setattr(archive, "_persist", fail_storage)
        archive.submit_scene(scene(episode("failed", "smoking", 1, 2)))
        assert archive.flush()
        page = archive.list_events()
        assert page["events"][0]["evidence"]["failure_reason"] == (
            "evidence_storage_unavailable"
        )
        assert archive.flush()
        assert archive.get_event(page["events"][0]["event_id"]) == page["events"][0]
        monkeypatch.setattr(archive, "_persist", original)
    finally:
        archive.close()
