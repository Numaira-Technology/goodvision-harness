"""Live throughput excludes EOF catch-up and never counts replayed observations."""

from dataclasses import replace

import httpx
import pytest
from test_perception_client import RouteServer
from test_video_tracking_session import frames, submit, wait_for

from goodvision_harness import sam_tracking_pipeline
from goodvision_harness.config import RichLabelSettings
from goodvision_harness.perception import (
    PerceptionSessionHTTPClient,
    create_tracking_coordinator,
)


@pytest.fixture(autouse=True)
def private_spool(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_SAM_SPOOL_DIRECTORY", str(tmp_path / "sam-spool"))


def test_ordered_pending_frames_and_eof_timing_do_not_inflate_live_fps(monkeypatch):
    records = []
    monkeypatch.setattr(
        sam_tracking_pipeline, "log_timing",
        lambda event, **fields: records.append((event, fields)),
    )
    server = RouteServer("sam_live")
    client = PerceptionSessionHTTPClient(
        "http://worker.test", method="sam_live", transport=httpx.MockTransport(server),
    )
    received = []
    coordinator = create_tracking_coordinator(
        client,
        replace(RichLabelSettings(), backend="sam31_stream",
                processing_method="sam_live", max_fps=4,
                video_queue_policy="durable_ordered"),
        received.append,
    )
    try:
        submit(coordinator, frames(1, 1))
        wait_for(lambda: coordinator.stats()["processed_frames"] == 1)
        server.entered.clear()
        server.release.clear()
        submit(coordinator, frames(2, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(3, 4))
        before = coordinator.stats()
        assert before["pending_frames"] == 4
        assert before["session_dropped_backpressure"] == 0
        coordinator.flush()
        capture_seconds = coordinator.stats()["capture_elapsed_seconds"]
        server.release.set()
        assert coordinator.wait_until_idle(3)
        stats = coordinator.stats()
        assert stats["session_source_frames"] == stats["session_admitted_frames"] == 6
        assert stats["completed_during_capture"] == 1
        assert stats["completed_after_capture"] == 5
        assert stats["completed_analysis_fps"] == pytest.approx(1 / capture_seconds)
        assert stats["capture_elapsed_seconds"] == capture_seconds
        assert [f.frame_sequence for b in received for f in b.frames] == [
            3, 6, 9, 12, 15, 18,
        ]
        wait_for(lambda: sum(
            name == "sam_coordinator_timing" and bool(row["input_frames"])
            for name, row in records
        ) == 3)
        completed = [row for name, row in records
                     if name == "sam_coordinator_timing" and row["input_frames"]]
        assert [row["outcome"] for row in completed] == ["ok"] * 3
        assert [row["request_id"] for row in completed] == [
            batch.request_id for batch in received
        ]
        assert completed[0]["completion_interval_seconds"] is None
        assert all(row["completion_interval_seconds"] >= 0 for row in completed[1:])
        assert all(row["commit_seconds"] >= 0 for row in completed)
        assert all(row["append_seconds"] >= 0 for row in completed)
        assert [item["sequence"] for row in completed
                for item in row["source_to_result"]] == [3, 6, 9, 12, 15, 18]
        assert not any(name == "sam_queue_overflow" for name, _ in records)
        coordinator.reset("next-camera")
        stats = coordinator.stats()
        assert stats["completed_analysis_fps"] is None
        assert stats["completed_during_capture"] == 0
        assert stats["completed_after_capture"] == 0
        assert stats["session_source_frames"] == 0
    finally:
        server.release.set()
        coordinator.close()


def test_durable_queue_preserves_previously_overflowed_source(monkeypatch):
    records = []
    monkeypatch.setattr(
        sam_tracking_pipeline, "log_timing",
        lambda event, **fields: records.append((event, fields)),
    )
    server = RouteServer("sam_live")
    client = PerceptionSessionHTTPClient(
        "http://worker.test", method="sam_live", transport=httpx.MockTransport(server),
    )
    coordinator = create_tracking_coordinator(
        client,
        replace(RichLabelSettings(), backend="sam31_stream",
                processing_method="sam_live", max_fps=4,
                video_queue_policy="durable_ordered"),
        None,
    )
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 5))
        losses = [row for name, row in records if name == "sam_queue_overflow"]
        assert losses == []
        assert coordinator.stats()["session_dropped_backpressure"] == 0
        assert coordinator.stats()["pending_frames"] == 5
        coordinator.flush()
        server.release.set()
        assert coordinator.wait_until_idle(3)
        assert coordinator.stats()["processed_frames"] == 6
    finally:
        server.release.set()
        coordinator.close()
