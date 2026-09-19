import queue
import sqlite3
from pathlib import Path
from threading import Event, Thread

import pytest

from goodvision_harness.archive_ingress import ArchiveIngress
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.video_evidence import EvidenceFrame


def window(run="run", *, at=1):
    return {
        "session_id": run, "window_id": "job", "timestamp": at,
        "labels": [], "outcome": "admitted", "admitted": 0,
        "source_frame_index": 1,
    }


def test_pending_facts_and_exact_sources_survive_restart(tmp_path):
    path = tmp_path / "ingress.sqlite3"
    first = ArchiveIngress(path, max_bytes=100_000)
    first.put(("start", ("a", "camera")))
    first.put(("frame", ("a", EvidenceFrame(b"private-image", 1.5, 14))))
    first.put(("end", ("a", "stopped")))
    assert first.get() == ("start", ["a", "camera"])
    first.task_done()
    first.close()
    recovered = ArchiveIngress(path, max_bytes=100_000)
    try:
        kind, (run, frame) = recovered.get()
        assert (kind, run, frame) == (
            "frame", "a", EvidenceFrame(b"private-image", 1.5, 14),
        )
        recovered.task_done()
        assert recovered.get() == ("end", ["a", "stopped"])
        recovered.task_done()
        assert recovered.unfinished_tasks == 0
        assert recovered.source_window("a", 0, 3) == [frame]
        assert recovered.source_window("other-run", 0, 3) == []
    finally:
        recovered.close()


def test_processing_failure_remains_recoverable_not_success(tmp_path):
    path = tmp_path / "ingress.sqlite3"
    first = ArchiveIngress(path, max_bytes=100_000)
    first.put(("end", ("a", "stopped")))
    first.get()
    first.task_failed()
    first.task_done()
    assert first.pending_for("a") == 1
    first.close()
    recovered = ArchiveIngress(path, max_bytes=100_000)
    try:
        assert recovered.get() == ("end", ["a", "stopped"])
        recovered.task_done()
    finally:
        recovered.close()


def test_waiter_restart_does_not_replay_process_local_barrier(tmp_path):
    path = tmp_path / "ingress.sqlite3"
    first = ArchiveIngress(path, max_bytes=100_000)
    first.put(("barrier", Event()))
    first.put(("close", None))
    first.put(("end", ("a", "stopped")))
    first.close()
    second = ArchiveIngress(path, max_bytes=100_000)
    try:
        assert second.get() == ("end", ["a", "stopped"])
    finally:
        second.close()


def test_encoding_result_preserves_path_without_pickle(tmp_path):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    try:
        payload = [{"event_id": "event-a"}, Path("a.mp4"), {"fps": 5}, None]
        ingress.put(("encoded", payload))
        assert ingress.get() == ("encoded", payload)
    finally:
        ingress.close()


def test_run_completion_does_not_wait_for_new_run_ingress(tmp_path):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    try:
        ingress.put(("end", ("old", "stopped")))
        ingress.put(("start", ("new", "camera")))
        ingress.get()
        ingress.task_done()
        assert ingress.pending_for("old") == 0
        assert ingress.pending_for("new") == 1
    finally:
        ingress.close()


def test_conflicting_source_never_overwrites_accepted_image(tmp_path):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    try:
        ingress.put(("frame", ("a", EvidenceFrame(b"original", 1, 1))))
        with pytest.raises(ValueError, match="conflicting"):
            ingress.put(("frame", ("a", EvidenceFrame(b"replacement", 1, 1))))
        assert ingress.source_window("a", 0, 2)[0].jpeg == b"original"
    finally:
        ingress.close()


def test_archive_has_early_sources_after_memory_ring_expires(tmp_path):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path))
    try:
        for i in range(3):
            assert archive.submit_frame(b"jpeg-" + bytes([i]), i, i, "run")
        assert archive.submit_frame(b"latest", 200, 200, "run")
        assert archive.flush()
        assert archive._sessions["run"].ring.frames[0].timestamp_seconds == 200
        assert [f.source_frame_index for f in archive._queue.source_window(
            "run", 0, 2,
        )] == [0, 1, 2]
    finally:
        archive.close()


def test_pending_run_count_includes_windows_and_failure_blocks_only_its_run(tmp_path):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    try:
        ingress.put(("window", window("old")))
        ingress.put(("end", ("old", "stopped")))
        ingress.put(("start", ("new", "camera")))
        assert ingress.pending_for("old") == 2
        assert ingress.get()[0] == "window"
        ingress.task_failed()
        ingress.task_done()
        assert ingress.get() == ("start", ["new", "camera"])
        ingress.task_done()
        with pytest.raises(queue.Empty):
            ingress.get()
        assert ingress.pending_for("old") == 2
        assert ingress.pending_for("new") == 0
    finally:
        ingress.close()


def test_processed_watermark_excludes_future_admitted_frames(tmp_path):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    try:
        ingress.put(("frame", ("run", EvidenceFrame(b"early", 1, 1))))
        ingress.get()
        ingress.task_done()
        ingress.put(("frame", ("run", EvidenceFrame(b"future", 200, 200))))
        assert ingress.source_latest("run") == 200
        assert ingress.processed_runs()["run"]["latest"] == 1
    finally:
        ingress.close()


def test_restarted_archive_restores_end_and_window_before_processing_tail(tmp_path):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    ingress.put(("start", ("run", "camera")))
    ingress.put(("window", window()))
    ingress.put(("end", ("run", "stopped")))
    for _ in range(3):
        ingress.get()
        ingress.task_done()
    ingress.put(("result", ({
        "stream_session_id": "run", "source_id": "camera", "message_id": "tail",
        "frame": {"timestamp_seconds": 1}, "judgments": [], "objects": [],
    }, {}, "job")))
    ingress.close()
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 1000)
    try:
        assert archive.flush()
        assert archive._sessions["run"].ended
        assert archive._sessions["run"].source_id == "camera"
        assert archive._sessions["run"].latest == 1
        assert archive.status()["late_results"] == 0
        assert archive.status()["storage_failures"] == 0
    finally:
        archive.close()


def test_source_reader_lock_does_not_block_new_fact_admission(tmp_path):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    accepted = Event()
    try:
        with ingress._read_lock:
            writer = Thread(target=lambda: (
                ingress.put(("start", ("run", "camera"))), accepted.set(),
            ))
            writer.start()
            assert accepted.wait(1), "source reader blocked the producer"
        writer.join(1)
    finally:
        ingress.close()


def test_duplicate_source_quota_counts_only_new_fact_and_restarts_exactly(tmp_path):
    path = tmp_path / "ingress.sqlite3"
    ingress = ArchiveIngress(path, max_bytes=100_000)
    frame = EvidenceFrame(b"j" * 1000, 1, 1)
    ingress.put(("frame", ("run", frame)))
    ingress.max_bytes = ingress._bytes + 50
    ingress.put(("frame", ("run", frame)))
    before = ingress._bytes
    ingress.close()
    recovered = ArchiveIngress(path, max_bytes=100_000)
    try:
        assert recovered._bytes == before
        assert recovered.source_window("run", 0, 2) == [frame]
    finally:
        recovered.close()


def test_source_retention_never_deletes_pending_failed_or_protected_runs(tmp_path):
    ingress = ArchiveIngress(tmp_path / "ingress.sqlite3", max_bytes=100_000)
    try:
        for run in ("expired", "protected", "failed"):
            ingress.put(("frame", (run, EvidenceFrame(b"original", 1, 1))))
            ingress.put(("end", (run, "stopped")))
            ingress.get()
            if run == "failed":
                ingress.task_failed()
            ingress.task_done()
            if run != "failed":
                ingress.get()
                ingress.task_done()
        before = ingress._bytes
        assert ingress.prune_completed_runs(
            before=float("inf"), protected={"protected"},
        ) > 0
        assert ingress._bytes < before
        assert ingress.source_window("expired", 0, 2) == []
        assert ingress.source_window("protected", 0, 2)
        assert ingress.source_window("failed", 0, 2)
        assert ingress.pending_for("failed") == 2
    finally:
        ingress.close()


def test_archive_close_releases_both_ingress_connections(tmp_path):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path))
    assert archive.start_session("run", "camera")
    assert archive.flush()
    archive.close()
    archive.close()
    assert archive.status()["state"] == "stopped"
    for connection in (archive._queue._db, archive._queue._reader):
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    original = tmp_path / "ingress.sqlite3"
    original.rename(tmp_path / "closed-ingress.sqlite3")
