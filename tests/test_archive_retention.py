"""Quota reclamation preserves source evidence and durable replay boundaries."""

import queue

import pytest
from test_archive_ingress import window

from goodvision_harness.archive_ingress import ArchiveIngress
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.video_evidence import EvidenceFrame


def acknowledged(ingress, item):
    ingress.put(item)
    ingress.get()
    ingress.task_done()


@pytest.mark.parametrize("ended", [False, True])
def test_rolling_prune_preserves_clip_successor_pending_source_and_restart(
    tmp_path, ended,
):
    path = tmp_path / "ingress.sqlite3"
    ingress = ArchiveIngress(path, max_bytes=100_000)
    acknowledged(ingress, ("start", ("run", "camera")))
    for sequence, stamp in enumerate((0, 1, 2, 3, 4, 5, 6, 100)):
        acknowledged(ingress, (
            "frame", ("run", EvidenceFrame(bytes([sequence]) * 1000, stamp, sequence)),
        ))
    acknowledged(ingress, ("window", window(at=2)))
    if ended:
        acknowledged(ingress, ("end", ("run", "stopped")))
    ingress.put(("frame", ("run", EvidenceFrame(bytes([0]) * 1000, 0, 0))))
    ingress.put(("frame", ("run", EvidenceFrame(b"future", 200, 8))))
    try:
        before = ingress._bytes
        released = ingress.prune_processed(
            "run", before=90, protected_ranges=[(2, 3.5)], protected_windows={"job"},
        )
        assert released > 3000
        assert ingress._bytes == before - released
        assert [f.timestamp_seconds for f in ingress.source_window("run", 0, 300)] == [
            0, 2, 3, 4, 100, 200,
        ]
        progress = ingress.processed_runs()["run"]
        assert progress["source_id"] == "camera"
        assert progress["latest"] == 100  # never the unprocessed future source
        assert progress["ended"] is ended
        assert progress["windows"] == {"job": window(at=2)}
        remaining = ingress._bytes
    finally:
        ingress.close()
    recovered = ArchiveIngress(path, max_bytes=100_000)
    try:
        assert recovered._bytes == remaining
        assert recovered.processed_runs()["run"] == progress
        for stamp in (0, 200):
            kind, (run, frame) = recovered.get()
            assert (kind, run, frame.timestamp_seconds) == ("frame", "run", stamp)
            recovered.task_done()
        assert recovered.unfinished_tasks == 0
    finally:
        recovered.close()


@pytest.mark.parametrize("state", ["pending", "failed"])
def test_unprocessed_facts_protect_run_sources_under_pressure(tmp_path, state):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    try:
        acknowledged(ingress, ("frame", ("run", EvidenceFrame(b"evidence", 1, 1))))
        ingress.put(("window", window()))
        if state == "failed":
            ingress.get()
            ingress.task_failed()
            ingress.task_done()
        before = ingress._bytes
        assert ingress.prune_processed(
            "run", before=100, protected_ranges=[], protected_windows=set(),
        ) == 0
        assert ingress._bytes == before
        assert ingress.pending_for("run") == 1
        assert ingress.source_window("run", 0, 2)[0].jpeg == b"evidence"
    finally:
        ingress.close()


def test_prune_cursor_advances_past_protected_prefix_and_revisits_it(tmp_path):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    try:
        for sequence in range(70):
            acknowledged(ingress, ("frame", (
                "run", EvidenceFrame(b"source", sequence, sequence),
            )))
        for _ in range(2):
            ingress.prune_processed(
                "run", before=100, protected_ranges=[(0, 63)],
                protected_windows=set(),
            )
        assert [f.source_frame_index for f in ingress.source_window("run", 0, 100)] == (
            list(range(64))
        )
        for _ in range(3):
            ingress.prune_processed(
                "run", before=100, protected_ranges=[], protected_windows=set(),
            )
        assert ingress.source_window("run", 0, 100) == []
        assert ingress.processed_runs()["run"]["latest"] == 69
    finally:
        ingress.close()


def test_latest_processed_window_preserves_progress_beyond_last_frame(tmp_path):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    try:
        acknowledged(ingress, ("frame", ("run", EvidenceFrame(b"source", 1, 1))))
        acknowledged(ingress, ("window", window(at=30)))
        ingress.prune_processed(
            "run", before=float("inf"), protected_ranges=[], protected_windows=set(),
        )
        assert ingress.source_window("run", 0, 100) == []
        assert ingress.processed_runs()["run"]["latest"] == 30
    finally:
        ingress.close()


def test_continuous_capture_reuses_quota_without_waiting_for_run_end(tmp_path):
    clock = [0.0]
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path, max_storage_bytes=4 * 1024**2),
        clock=lambda: clock[0],
    )
    try:
        assert archive.start_session("run", "camera")
        for sequence in range(48):
            clock[0] += 1
            assert archive.submit_frame(b"j" * 100_000, sequence * 10, sequence, "run")
            assert archive.flush()
        assert not archive._sessions["run"].ended
        assert archive.status()["dropped"] == 0
        assert archive._queue._bytes < archive._queue.max_bytes
        assert archive._queue.source_window("run", 0, 10) == []
        assert archive._queue.source_latest("run") == 470
    finally:
        archive.close()


@pytest.mark.parametrize("restart,ended", [(False, True), (True, True), (True, False)])
def test_recently_ended_run_releases_raw_sources_without_age_expiry(
    tmp_path, restart, ended,
):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0)
    try:
        assert archive.start_session("run", "camera")
        assert archive.submit_frame(b"j" * 1000, 1, 1, "run")
        if ended:
            assert archive.end_session("run")
        assert archive.flush()
        if restart:
            archive.close()
            archive = EventArchive(
                EventArchiveSettings(directory=tmp_path), clock=lambda: 0,
            )
            assert archive.flush()
            assert "run" not in archive._sessions
        # No seven-day wait, even though the original source quota is full.
        with archive._lock:
            archive._queue.max_bytes = archive._queue._bytes
            with pytest.raises(queue.Full):
                archive._queue.put(("frame", ("new", EvidenceFrame(b"new", 0, 0))))
            archive._retention()
            archive._queue.put(("frame", ("new", EvidenceFrame(b"new", 0, 0))))
        assert archive.flush()
        assert archive._queue.source_window("run", 0, 2) == []
        assert archive._queue.source_window("new", 0, 1)[0].jpeg == b"new"
    finally:
        archive.close()


@pytest.mark.parametrize("protection", ["scene_event", "late_window"])
def test_pressure_preserves_revisable_clip_while_pruning_unrelated_history(
    tmp_path, protection,
):
    from test_event_archive import fake_encoder, jpeg
    from test_scene_event_archive import episode, frames, scene

    clock = [0.0]
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=fake_encoder,
        clock=lambda: clock[0],
    )
    try:
        frames(archive)
        if protection == "scene_event":
            assert archive.submit_scene(scene(episode("ongoing", "smoking", 3, 4)))
            assert archive.wait_until_idle(3)
        assert archive.submit_frame(jpeg(), 100, 500, "session")
        assert archive.flush()
        clock[0] = 100
        if protection == "late_window":
            assert archive.note_window("session", "late", 3, ("smoking",))
            assert archive.flush()
        before = archive._queue.source_window("session", 2, 5)
        assert before
        with archive._lock:
            archive._queue.max_bytes = archive._queue._bytes
            archive._retention()
        assert archive._queue.source_window("session", 2, 5) == before
        assert archive._queue.source_window("session", 0, 1) == []
        assert archive._queue.source_window("session", 6, 7) == []
        if protection == "scene_event":
            event = archive.list_events()["events"][0]
            assert not event["finalized"]
            assert event["evidence"]["status"] == "ready"
        else:
            assert "late" in archive._queue.processed_runs()["session"]["windows"]
    finally:
        archive.close()
