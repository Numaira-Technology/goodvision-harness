"""Latest-candidate overload, durable execution and lifecycle (no GPU claims)."""

import threading
from concurrent.futures import ThreadPoolExecutor

from test_sam_tracking_pipeline import output_sequences, setup_pipeline
from test_video_tracking_session import frames, submit, wait_for

from goodvision_harness.sam_tracking_spool import TrackingSpool
from goodvision_harness.video_tracking import TrackingCommitBackpressure


def test_slow_http_keeps_candidates_replaceable_and_dispatches_only_the_freshest(
    tmp_path,
):
    received = []
    server, _, coordinator = setup_pipeline(
        tmp_path, received.append, queue_policy="latest_only",
        realtime_max_age_seconds=30,
    )
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 1))
        # The old implementation froze frame 6 here, then sent it after the
        # slow HTTP request despite many newer candidates. No durable source
        # should be selected until that request releases its dispatch slot.
        assert coordinator.stats()["prepared_frames"] == 0
        with coordinator._condition:
            submit(coordinator, frames(3, 39))
        stats = coordinator.stats()
        assert stats["pending_capacity"] == stats["candidate_frames"] == 2
        assert stats["effective_max_append_frames"] == 1
        assert stats["realtime_max_age_seconds"] == 30
        assert stats["preparing_frames"] == 0
        assert stats["prepared_frames"] == 0
        assert stats["disk_pending_frames"] == 0
        assert stats["session_admitted_frames"] == 1
        assert stats["session_candidate_frames"] == 41
        assert stats["skipped_by_reason"] == {"superseded": 38, "expired": 0}
        assert list(coordinator._candidate_frames) == [120, 123]
        assert coordinator._spool.stats()["retained_sources"] == 1
        assert len(server.requests) == 1
        coordinator.flush()
        server.release.set()
        assert coordinator.wait_until_idle(5)
        assert output_sequences(received) == [3, 123]
        assert [f.timestamp_seconds for b in received for f in b.frames] == [
            .25, 10.25,
        ]
        assert server.creates == ["camera"]
        assert {b.predictor_epoch for b in received} == {"epoch-1"}
        assert coordinator.stats()["session_admitted_frames"] == 2
        assert coordinator.stats()["processed_frames"] == 2
        assert not coordinator._source_arrivals
        assert not coordinator._candidate_frames
        assert server.requests[-1]["is_last_batch"]
        assert server.requests[-1]["frames"] == []
        assert all(r["state"] == "committed" for r in coordinator._spool.records())
    finally:
        server.release.set()
        coordinator.close()


def test_slow_admission_does_not_hold_capture_lock_or_write_replaced_candidates(
    tmp_path, monkeypatch,
):
    entered, release = threading.Event(), threading.Event()
    admitted, received = [], []
    original = TrackingSpool.admit

    def blocked_admit(self, generation, session, frame, arrived):
        admitted.append(frame.frame_sequence)
        if frame.frame_sequence == 3:
            entered.set()
            assert release.wait(5)
        return original(self, generation, session, frame, arrived)

    monkeypatch.setattr(TrackingSpool, "admit", blocked_admit)
    server, _, coordinator = setup_pipeline(
        tmp_path, received.append, queue_policy="latest_only",
        # This exercises replacement during blocked disk admission, not source
        # expiry. Keep slow journal initialization outside the default 2s TTL.
        realtime_max_age_seconds=30,
    )
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        submit(coordinator, frames(1, 1))
        assert entered.wait(2)
        # This must finish while the selected frame's disk admission is blocked.
        pool.submit(submit, coordinator, frames(2, 100)).result(timeout=1)
        assert coordinator.stats()["candidate_frames"] == 2
        assert admitted == [3]
        assert not server.requests
        release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(5)
        assert admitted == [3, 303]
        assert output_sequences(received) == admitted
        assert coordinator.stats()["skipped_by_reason"]["expired"] == 0
    finally:
        release.set()
        pool.shutdown(wait=True)
        coordinator.close()


def test_slow_preparation_keeps_two_replaceable_candidates_and_no_prepared_backlog(
    tmp_path, monkeypatch,
):
    entered, release = threading.Event(), threading.Event()
    prepared, received = [], []
    server, client, coordinator = setup_pipeline(
        tmp_path, received.append, queue_policy="latest_only",
    )
    original = client.prepare_append

    def blocked_prepare(inputs, **kwargs):
        prepared.extend(f.frame_sequence for f in inputs)
        if inputs[0].frame_sequence == 3:
            entered.set()
            assert release.wait(5)
        return original(inputs, **kwargs)

    monkeypatch.setattr(client, "prepare_append", blocked_prepare)
    try:
        submit(coordinator, frames(1, 1))
        assert entered.wait(2)
        submit(coordinator, frames(2, 30))
        stats = coordinator.stats()
        assert stats["candidate_frames"] == 2
        assert stats["preparing_frames"] == 1
        assert stats["prepared_frames"] == 0
        assert not server.requests
        release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(5)
        assert prepared == output_sequences(received) == [3, 93]
    finally:
        release.set()
        coordinator.close()


def test_blocked_commit_keeps_admitted_results_and_replaces_only_candidate(tmp_path):
    entered, release = threading.Event(), threading.Event()
    received = []

    def commit(batch):
        if not release.is_set():
            entered.set()
            raise TrackingCommitBackpressure("blocked history consumer")
        received.append(batch)

    server, _, coordinator = setup_pipeline(
        tmp_path, commit, queue_policy="latest_only",
    )
    try:
        submit(coordinator, frames(1, 1))
        assert entered.wait(2)
        submit(coordinator, frames(2, 1))
        wait_for(lambda: coordinator.stats()["received_batches"] == 2)
        submit(coordinator, frames(3, 30))
        stats = coordinator.stats()
        assert stats["pending_commits"] == 2
        assert stats["candidate_frames"] == 2
        assert stats["prepared_frames"] == stats["preparing_frames"] == 0
        assert coordinator._spool.stats()["recoverable_results"] == 2
        assert len(server.requests) == 2
        assert [f.frame_sequence for f in coordinator.retained_sources(
            coordinator._committing.result,
        )] == [3]
        release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(5)
        assert output_sequences(received) == [3, 6, 96]
        ids = [r["request_id"] for r in server.requests]
        assert len(ids) == len(set(ids))
    finally:
        release.set()
        coordinator.close()


def test_reset_does_not_send_old_candidate_or_publish_old_inflight_result(tmp_path):
    received, native = [], []
    server, _, coordinator = setup_pipeline(
        tmp_path, received.append, queue_policy="latest_only",
        on_native_batch=lambda batch, generation: native.append(batch),
    )
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 1))
        assert coordinator.stats()["prepared_frames"] == 0
        submit(coordinator, frames(3, 19))
        coordinator.reset("new")
        assert coordinator.stats()["session_candidate_frames"] == 0
        submit(coordinator, frames(1, 1), session="new")
        coordinator.flush()
        server.release.set()
        assert coordinator.wait_until_idle(5)
        assert [b.stream_session_id for b in received] == ["new"]
        assert all(b.stream_session_id == "new" for b in native)
        assert output_sequences(received) == [3]
        # Sent work retains recovery facts; memory candidates from
        # the old generation must never be selected by the new run.
        assert coordinator._spool.stats()["retained_sources"] == 1
        assert coordinator._spool.load(0, [3])[0].jpeg_bytes == b"jpeg"
    finally:
        server.release.set()
        coordinator.close()
