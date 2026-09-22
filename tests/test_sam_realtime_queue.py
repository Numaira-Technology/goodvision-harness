"""Bounded live ingress through real HTTP clients; model doubles, not GPU claims."""

import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_sam_tracking_pipeline import output_sequences, setup_pipeline
from test_video_tracking_session import frames, submit, wait_for

from goodvision_harness import sam_tracking_pipeline
from goodvision_harness.sam_tracking_spool import TrackingSpool


@pytest.fixture(params=[0.0, 0.25], ids=["normal-spool", "slow-spool"])
def advance_age(monkeypatch, request):
    """Advance source age without racing CI disk I/O against a 100ms TTL.

    Only this module's clock is offset; Event waits and HTTP timeouts remain
    real, and the clock keeps ticking so idle/close timeouts stay bounded.
    """
    offset = 0.0
    monkeypatch.setattr(
        sam_tracking_pipeline, "time",
        SimpleNamespace(monotonic=lambda: time.monotonic() + offset),
    )
    admit = TrackingSpool.admit

    def delayed_admit(*args, **kwargs):
        # Exceed the old 100-150ms TTL before the first worker can even start.
        if request.param:
            time.sleep(request.param)
        return admit(*args, **kwargs)

    monkeypatch.setattr(TrackingSpool, "admit", delayed_admit)

    def advance(coordinator):
        nonlocal offset
        with coordinator._condition:
            offset += 31.0
            coordinator._condition.notify_all()

    return advance


def test_overload_replaces_only_unsent_frames_and_preserves_one_epoch(tmp_path):
    received = []
    server, _, coordinator = setup_pipeline(
        tmp_path,
        received.append,
        queue_policy="realtime_latest",
        realtime_max_age_seconds=10,
    )
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 19))
        stats = coordinator.stats()
        assert stats["pending_frames"] == stats["disk_pending_frames"] == 2
        assert stats["prepared_frames"] == stats["preparing_frames"] == 0
        assert stats["skipped_by_reason"] == {"superseded": 17, "expired": 0}
        assert stats["session_admitted_frames"] == 20
        assert coordinator._spool.stats()["retained_sources"] == 3
        assert coordinator._spool.load(0, [3])[0].jpeg_bytes == b"jpeg"
        server.release.set()
        coordinator.flush()
        # Allow HTTP completion and durable journal writes on shared runners;
        # replacement counts, exact output and the epoch are asserted below.
        assert coordinator.wait_until_idle(10), coordinator.stats()
        assert output_sequences(received) == [3, 57, 60]
        assert server.creates == ["camera"]
        assert {batch.predictor_epoch for batch in received} == {"epoch-1"}
        assert all(len(r["frames"]) <= 1 for r in server.requests)
        assert coordinator.stats()["processed_frames"] == 3
        assert len(coordinator._source_arrivals) == 0
        assert len(coordinator._spool.skipped_records()) == 17
        assert all(r["state"] == "committed" for r in coordinator._spool.records())
    finally:
        server.release.set()
        coordinator.close()


@pytest.mark.parametrize("queue_policy", ["realtime_latest", "latest_only"])
def test_pending_expires_without_another_arrival_while_request_stays_owned(
    tmp_path, monkeypatch, advance_age, queue_policy
):
    received = []
    server, _, coordinator = setup_pipeline(
        tmp_path,
        received.append,
        queue_policy=queue_policy,
        realtime_max_age_seconds=30,
    )
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        expiry_waiting = threading.Event()
        condition_wait = coordinator._condition.wait

        def observed_wait(timeout=None):
            if timeout is not None and 0 < timeout <= 30:
                expiry_waiting.set()
            return condition_wait(timeout)

        monkeypatch.setattr(coordinator._condition, "wait", observed_wait)
        submit(coordinator, frames(2, 1))
        # Verify a real expiry timer was armed before simulating its wakeup;
        # an implementation that waits only for new arrivals must still fail.
        assert expiry_waiting.wait(2)
        advance_age(coordinator)
        wait_for(lambda: coordinator.stats()["skipped_by_reason"]["expired"] == 1)
        assert coordinator.stats()["pending_frames"] == 0
        assert len(server.requests) == 1
        assert coordinator._requesting is not None
        assert coordinator._spool.load(0, [3])[0].jpeg_bytes == b"jpeg"
        server.release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(3)
        assert output_sequences(received) == [3]
        if queue_policy == "realtime_latest":
            assert coordinator._spool.skipped_records()[0]["sequence"] == 6
    finally:
        server.release.set()
        coordinator.close()


@pytest.mark.parametrize("queue_policy", ["realtime_latest", "latest_only"])
def test_slow_preparation_is_retired_before_send_and_empty_session_can_finish(
    tmp_path, monkeypatch, advance_age, queue_policy
):
    received = []
    server, client, coordinator = setup_pipeline(
        tmp_path,
        received.append,
        queue_policy=queue_policy,
        realtime_max_age_seconds=30,
    )
    entered, release = threading.Event(), threading.Event()
    prepare = client.prepare_append

    def delayed_prepare(*args, **kwargs):
        result = prepare(*args, **kwargs)
        entered.set()
        assert release.wait(3)
        return result

    monkeypatch.setattr(client, "prepare_append", delayed_prepare)
    try:
        submit(coordinator, frames(1, 1))
        assert entered.wait(2)
        advance_age(coordinator)
        coordinator.flush()
        release.set()
        assert coordinator.wait_until_idle(3)
        assert coordinator.stats()["state"] == "finished"
        assert received == server.requests == []
        assert len(server.deletes) == 1
        row = coordinator._spool.records()[0]
        assert row["state"] == "abandoned"
        assert row["payload"] is None and row["payload_hash"]
        assert coordinator._spool.skipped_records()[0]["reason"] == "expired"
    finally:
        release.set()
        coordinator.close()


def test_latest_only_expires_prepared_and_memory_candidates_without_cancelling_http(
    tmp_path, advance_age,
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
        wait_for(lambda: coordinator.stats()["prepared_frames"] == 1)
        submit(coordinator, frames(3, 2))
        assert coordinator.stats()["candidate_frames"] == 2
        advance_age(coordinator)
        wait_for(lambda: coordinator.stats()["skipped_by_reason"]["expired"] == 3)
        assert coordinator.stats()["pending_frames"] == 0
        assert not coordinator._candidate_frames
        assert list(coordinator._source_arrivals) == [3]
        assert len(server.requests) == 1
        assert coordinator._requesting is not None
        assert coordinator._spool.stats()["retained_sources"] == 1
        skipped = coordinator._spool.skipped_records()
        assert [row["sequence"] for row in skipped] == [6]
        assert skipped[0]["reason"] == "expired"
        abandoned = coordinator._spool.records()[1]
        assert abandoned["state"] == "abandoned"
        assert abandoned["payload"] is None and abandoned["payload_hash"]
        # The preparation slot is reusable before the original request returns.
        submit(coordinator, frames(5, 1))
        wait_for(lambda: coordinator.stats()["prepared_frames"] == 1)
        assert len(server.requests) == 1
        server.release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(4)
        assert output_sequences(received) == [3, 15]
        assert server.creates == ["camera"]
        assert not coordinator._source_arrivals
    finally:
        server.release.set()
        coordinator.close()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_limits_keep_same_session_and_do_not_wait_to_fill(tmp_path, batch_size):
    received = []
    server, _, coordinator = setup_pipeline(
        tmp_path,
        received.append,
        queue_policy="realtime_latest",
        realtime_append_frames=batch_size,
        realtime_pending_frames=8,
        realtime_max_age_seconds=10,
    )
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)  # first frame never waits for a full batch
        submit(coordinator, frames(2, 8))
        server.release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(3)
        assert output_sequences(received) == list(range(3, 28, 3))
        assert server.creates == ["camera"]
        assert max(len(r["frames"]) for r in server.requests) == batch_size
        assert coordinator.stats()["effective_max_append_frames"] == batch_size
        assert coordinator.stats()["skipped_by_reason"] == {
            "superseded": 0,
            "expired": 0,
        }
    finally:
        server.release.set()
        coordinator.close()


def test_skip_audit_is_atomic_and_cannot_discard_sent_or_partial_batches(tmp_path):
    spool = TrackingSpool(tmp_path / "sources.sqlite3", keep_open=True)
    try:
        spool.begin(0, "camera")
        for frame in frames(1, 2):
            spool.admit(0, "camera", frame, 1.0)
        spool.batch("request", 0, 1, [3, 6])
        spool.prepared("request", b"payload", "hash")
        with pytest.raises(RuntimeError):
            spool.skip(0, [3], "expired", request_id="request")
        assert spool.skipped_records() == []
        assert len(spool.load(0, [3, 6])) == 2
        spool.mark("request", "requesting")
        with pytest.raises(RuntimeError):
            spool.skip(0, [3, 6], "expired", request_id="request")
        with pytest.raises(RuntimeError):
            spool.skip(0, [3, 6], "superseded")
        assert spool.records()[0]["payload"] == b"payload"
        assert spool.skipped_records() == []
    finally:
        spool.close()


def test_skip_audit_failure_preserves_sources_and_reports_failed_session(
    tmp_path, monkeypatch
):
    server, _, coordinator = setup_pipeline(
        tmp_path,
        queue_policy="realtime_latest",
        realtime_max_age_seconds=10,
        realtime_pending_frames=1,
    )
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 1))

        def fail_audit(*args, **kwargs):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(coordinator._spool, "skip", fail_audit)
        with pytest.raises(OSError):
            submit(coordinator, frames(3, 1))
        assert coordinator.stats()["state"] == "failed"
        assert coordinator.stats()["recovery_required"] is True
        assert coordinator._spool.load(0, [3, 6]) == frames(1, 2)
        assert coordinator._spool.skipped_records() == []
    finally:
        server.release.set()
        coordinator.close()


def test_deadline_coverage_counts_skipped_frames_and_keeps_real_result_age(tmp_path):
    server, _, coordinator = setup_pipeline(
        tmp_path,
        queue_policy="realtime_latest",
        realtime_max_age_seconds=10,
        result_deadline_seconds=0.1,
        realtime_pending_frames=1,
    )
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        time.sleep(0.15)
        submit(coordinator, frames(2, 2))
        server.release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(3)
        stats = coordinator.stats()
        assert stats["session_admitted_frames"] == 3
        assert stats["processed_frames"] == 2
        assert stats["results_after_deadline"] >= 1
        assert stats["results_within_deadline"] + stats["results_after_deadline"] == 2
        assert (
            stats["admitted_deadline_coverage"] == stats["results_within_deadline"] / 3
        )
        assert stats["latest_result_age_seconds"] >= stats["source_to_result_seconds"]
        coordinator.reset("new")
        assert coordinator.stats()["latest_result_age_seconds"] is None
        assert coordinator.stats()["admitted_deadline_coverage"] is None
        assert coordinator.stats()["skipped_by_reason"] == {
            "superseded": 0,
            "expired": 0,
        }
    finally:
        server.release.set()
        coordinator.close()


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1, 1), (2, 2, 1), (4, 1)])
def test_production_continuation_keeps_temporal_state_across_append_sizes(chunks):
    """Exercise production adapter and causal postprocessing with fake kernels.

    This checks adapter state/confirmation semantics, not CUDA accuracy or speed.
    """
    from workers.sam_scene.sam_live import CausalPostprocessing, SamLiveContinuation

    class Model(CausalPostprocessing):
        initializations = 0

        def init_state(self, *, resource_path, **kwargs):
            self.initializations += 1
            return {
                "num_frames": len(resource_path),
                "feature_cache": {},
                "tracker_metadata": {},
                "seen": [],
                "generator_state": {
                    "hotstart_buffer": [],
                    "postprocess_yield_list": [],
                    "unconfirmed_obj_ids_per_frame": {},
                },
            }

        def initialize_causal_prompt(self, state):
            assert state["seen"] == []

        def _compile_model(self):
            pass

        def _get_processing_order(self, state, start, count, reverse):
            return range(start, start + count), start + count - 1

        def _run_single_frame_inference(self, state, index, reverse):
            assert index == len(state["seen"])
            state["seen"].append(index)
            return {
                "removed_obj_ids": set(),
                "suppressed_obj_ids": set(),
                "unconfirmed_obj_ids": {7} if len(state["seen"]) < 3 else set(),
                "obj_id_to_mask": {7: "mask"},
            }

        def _cache_frame_outputs(self, *args, **kwargs):
            pass

        def _postprocess_output(self, state, raw, *args):
            return {"out_obj_ids": [7], "confirmed": not raw["unconfirmed_obj_ids"]}

    class Retention:
        def install(self, state, **kwargs):
            pass

        def append(self, model, state, images, **kwargs):
            state["num_frames"] += len(images)

    native = SimpleNamespace(
        _torch=SimpleNamespace(
            inference_mode=nullcontext,
            autocast=lambda **kwargs: nullcontext(),
            bfloat16="test",
            cuda=SimpleNamespace(synchronize=lambda: None),
        )
    )
    backend = SamLiveContinuation.__new__(SamLiveContinuation)
    backend.video = SimpleNamespace(backend=native)
    backend.settings = SimpleNamespace(
        max_session_frames=100, max_cumulative_native_ids=100
    )
    backend.model, backend.retention = Model(), Retention()
    session = SimpleNamespace(
        native_state=None, native_retention_installed=False, total_frames=0
    )
    results, states = [], []
    for size in chunks:
        results.extend(backend.run(session, [object()] * size, final=False))
        session.total_frames += size
        states.append(id(session.native_state))
    assert [index for index, _ in results] == list(range(5))
    assert [out["confirmed"] for _, out in results] == [False, False, True, True, True]
    assert backend.model.initializations == 1
    assert len(set(states)) == 1
    assert session.native_state["seen"] == list(range(5))
    assert list(backend.run(session, [], final=True)) == []


def test_expiry_during_commit_backpressure_keeps_real_results_and_evidence(
    tmp_path, advance_age
):
    from goodvision_harness.video_tracking import TrackingCommitBackpressure

    received = []
    blocked, release = threading.Event(), threading.Event()

    def commit(batch):
        if not release.is_set():
            blocked.set()
            raise TrackingCommitBackpressure("busy inspection")
        received.append(batch)

    server, _, coordinator = setup_pipeline(
        tmp_path,
        commit,
        queue_policy="realtime_latest",
        realtime_max_age_seconds=30,
    )
    try:
        submit(coordinator, frames(1, 1))
        assert blocked.wait(2)
        submit(coordinator, frames(2, 1))
        wait_for(lambda: coordinator.stats()["received_batches"] == 2)
        submit(coordinator, frames(3, 2))
        advance_age(coordinator)
        wait_for(lambda: coordinator.stats()["skipped_by_reason"]["expired"] == 2)
        assert coordinator._spool.stats()["recoverable_results"] == 2
        assert coordinator._spool.load(0, [3, 6]) == frames(1, 2)
        assert [
            frame.frame_sequence
            for frame in coordinator.retained_sources(coordinator._committing.result)
        ] == [3]
        assert len(server.requests) == 2
        release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(3)
        assert output_sequences(received) == [3, 6]
        assert all(row["state"] == "committed" for row in coordinator._spool.records())
    finally:
        release.set()
        coordinator.close()


def test_retiring_empty_session_cannot_finish_replacement_generation(
    tmp_path, monkeypatch, advance_age
):
    prepared, release_prepare = threading.Event(), threading.Event()
    deleting, release_delete = threading.Event(), threading.Event()
    received = []
    _, client, coordinator = setup_pipeline(
        tmp_path,
        received.append,
        queue_policy="realtime_latest",
        realtime_max_age_seconds=30,
    )
    original_prepare, discard = client.prepare_append, client.discard_session

    def slow_first_prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        if kwargs["stream_session_id"] == "camera":
            prepared.set()
            assert release_prepare.wait(3)
        return result

    def blocked_empty_delete(*args, **kwargs):
        if coordinator._flush:
            deleting.set()
            assert release_delete.wait(3)
        return discard(*args, **kwargs)

    monkeypatch.setattr(client, "prepare_append", slow_first_prepare)
    monkeypatch.setattr(client, "discard_session", blocked_empty_delete)
    try:
        submit(coordinator, frames(1, 1))
        assert prepared.wait(2)
        advance_age(coordinator)
        coordinator.flush()
        release_prepare.set()
        assert deleting.wait(2)
        coordinator.reset("replacement")
        release_delete.set()
        wait_for(lambda: not coordinator._in_flight)
        assert coordinator.stats()["state"] == "running"
        assert coordinator.submit(
            b"jpeg",
            stream_session_id="replacement",
            frame_sequence=30,
            timestamp_seconds=10,
        )
        assert coordinator.wait_until_idle(3)
        assert output_sequences(received) == [30]
        assert received[0].stream_session_id == "replacement"
    finally:
        release_prepare.set()
        release_delete.set()
        coordinator.close()


def test_public_factory_wires_opt_in_policy_and_reports_it(tmp_path):
    from dataclasses import replace

    import httpx
    from test_perception_client import RouteServer

    from goodvision_harness.config import RichLabelSettings
    from goodvision_harness.perception import (
        PerceptionSessionHTTPClient,
        create_tracking_coordinator,
        processing_status,
    )

    client = PerceptionSessionHTTPClient(
        "http://worker.test",
        method="sam_live",
        transport=httpx.MockTransport(RouteServer("sam_live")),
    )
    rich = replace(
        RichLabelSettings(),
        backend="sam31_stream",
        processing_method="sam_live",
        video_queue_policy="realtime_latest",
        video_realtime_pending_frames=4,
        video_realtime_append_frames=2,
        video_realtime_max_age_seconds=3,
    )
    coordinator = create_tracking_coordinator(client, rich, None)
    try:
        status = processing_status(coordinator, "sam_live")
        assert status["queue_policy"] == "realtime_latest"
        assert status["effective_max_append_frames"] == 2
        assert status["result_deadline_seconds"] == 4
        assert status["pending_capacity"] == 4
        assert status["realtime_max_age_seconds"] == 3
        assert status["session_admitted_frames"] == 0
        assert coordinator.stats()["pending_capacity"] == 4
        assert coordinator.stats()["realtime_max_age_seconds"] == 3
    finally:
        coordinator.close()
