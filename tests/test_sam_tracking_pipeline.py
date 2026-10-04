"""Real HTTP protocol with blocked stages; no GPU timing claims."""

import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from test_perception_client import RouteServer
from test_video_tracking_session import frames, submit, wait_for

from goodvision_harness.rich_labeling import RichLabelProtocolError
from goodvision_harness.sam_tracking_pipeline import (
    PipelinedVideoSessionTrackingCoordinator,
)
from goodvision_harness.sam_tracking_spool import TrackingSpool, TrackingSpoolFull
from goodvision_harness.video_tracking import TrackingCommitBackpressure


def setup_pipeline(tmp_path, callback=None, **kwargs):
    from goodvision_harness.perception import PerceptionSessionHTTPClient

    server = RouteServer("sam_live")
    client = PerceptionSessionHTTPClient(
        "http://worker.test",
        method="sam_live",
        transport=httpx.MockTransport(server),
        timeout_seconds=4,
    )
    coordinator = PipelinedVideoSessionTrackingCoordinator(
        client,
        max_fps=kwargs.pop("max_fps", 4),
        batch_frames=1,
        batch_seconds=0.1,
        pending_capacity=4,
        on_batch=callback,
        spool_directory=tmp_path,
        **kwargs,
    )
    return server, client, coordinator


def output_sequences(batches):
    return [snapshot.frame_sequence for batch in batches for snapshot in batch.frames]


@pytest.mark.parametrize("queue_policy", ["durable_ordered", "realtime_latest"])
def test_first_frames_use_persistent_data_without_spool_override(
    tmp_path, monkeypatch, queue_policy,
):
    # Reproduce a delivery with only its existing persistent settings path.
    # No helper-supplied spool_directory may mask the default-path regression.
    from goodvision_harness.perception import PerceptionSessionHTTPClient

    monkeypatch.delenv("GOODVISION_SAM_SPOOL_DIRECTORY", raising=False)
    monkeypatch.delenv("GOODVISION_TRACKING_SPOOL_DIRECTORY", raising=False)
    monkeypatch.setenv(
        "HARNESS_DEPLOYMENT_SETTINGS_PATH", str(tmp_path / "data/settings.json"),
    )
    app = tmp_path / "app"
    app.mkdir()
    monkeypatch.chdir(app)
    received = []
    client = PerceptionSessionHTTPClient(
        "http://worker.test", method="sam_live",
        transport=httpx.MockTransport(RouteServer("sam_live")), timeout_seconds=4,
    )
    coordinator = PipelinedVideoSessionTrackingCoordinator(
        client, max_fps=4, batch_frames=1, batch_seconds=0.1,
        on_batch=received.append, queue_policy=queue_policy,
    )
    try:
        for index in range(3):
            submit(coordinator, frames(index, 1))
            assert coordinator.wait_until_idle(3)
        coordinator.flush()
        assert coordinator.wait_until_idle(3)
        assert output_sequences(received) == [0, 3, 6]
        assert coordinator._journal().path.parent == tmp_path / "data/sam-spool"
        assert not (app / ".goodvision").exists()
    finally:
        coordinator.close()


@pytest.mark.parametrize("selection", ["argument", "neutral", "legacy", "local"])
def test_spool_location_preserves_explicit_overrides(tmp_path, monkeypatch, selection):
    from goodvision_harness.perception import PerceptionSessionHTTPClient

    for name in (
        "GOODVISION_SAM_SPOOL_DIRECTORY", "GOODVISION_TRACKING_SPOOL_DIRECTORY",
        "HARNESS_DEPLOYMENT_SETTINGS_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    kwargs = {}
    expected = tmp_path / selection
    if selection != "local":
        monkeypatch.setenv(
            "HARNESS_DEPLOYMENT_SETTINGS_PATH", str(tmp_path / "data/settings.json"),
        )
        monkeypatch.setenv("GOODVISION_SAM_SPOOL_DIRECTORY", str(tmp_path / "legacy"))
    if selection in {"argument", "neutral"}:
        monkeypatch.setenv(
            "GOODVISION_TRACKING_SPOOL_DIRECTORY", str(tmp_path / "neutral"),
        )
    if selection == "argument":
        kwargs["spool_directory"] = expected
    if selection == "local":
        expected = tmp_path / ".goodvision/sam-spool"
    client = PerceptionSessionHTTPClient("http://worker.test", method="sam_live")
    coordinator = PipelinedVideoSessionTrackingCoordinator(client, **kwargs)
    try:
        assert coordinator._journal().path.resolve().parent == expected
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    "queue_policy", ["durable_ordered", "realtime_latest", "latest_only"],
)
@pytest.mark.parametrize("reuse_worker", [False, True])
def test_empty_run_finishes_and_allows_next_run(tmp_path, queue_policy, reuse_worker):
    received = []
    server, _, coordinator = setup_pipeline(
        tmp_path, received.append, queue_policy=queue_policy,
    )
    try:
        if reuse_worker:
            submit(coordinator, frames(1, 1))
            coordinator.flush()
            assert coordinator.wait_until_idle(3)
        coordinator.reset("empty")
        requests = len(server.requests)
        coordinator.flush()
        assert coordinator.wait_until_idle(1), (
            "empty run must not wait for a native session"
        )
        assert coordinator.stats()["state"] == "finished"
        assert len(server.requests) == requests
        coordinator.reset("next")
        submit(coordinator, frames(1, 1), session="next")
        coordinator.flush()
        assert coordinator.wait_until_idle(3)
        assert received[-1].stream_session_id == "next"
        assert output_sequences(received)[-1] == 3
    finally:
        coordinator.close()


@pytest.mark.parametrize("prefix_frames", [1, 2, 3])
def test_learned_prepare_overlaps_prefix_without_new_arrival(
    tmp_path, monkeypatch, prefix_frames,
):
    received = []
    server, client, coordinator = setup_pipeline(tmp_path, received.append)
    append, prepare = client.append, client.prepare_append
    busy, release, next_prepared = (threading.Event() for _ in range(3))
    calls = 0

    def observed_append(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = append(*args, **kwargs)
        if calls == 4:
            busy.set()
            assert release.wait(3)
        elif calls < 4:
            time.sleep(0.35)
        return result

    def observed_prepare(inputs, **kwargs):
        time.sleep(0.025)
        result = prepare(inputs, **kwargs)
        if inputs[0].frame_sequence == 12:
            next_prepared.set()
        return result

    monkeypatch.setattr(client, "append", observed_append)
    monkeypatch.setattr(client, "prepare_append", observed_prepare)
    try:
        # Learn from real completed stages, not a configured network guess.
        for index in range(3):
            submit(coordinator, frames(index, 1))
            assert coordinator.wait_until_idle(3)
        assert len(coordinator._append_durations) == 3
        submit(coordinator, frames(3, 1))
        assert busy.wait(2)
        submit(coordinator, frames(4, prefix_frames))
        # Real journal I/O and the learned preparation lead can consume the
        # predicted delay before submit returns, especially on shared CI hosts.
        # The deterministic forecast test below checks that positive delays
        # defer preparation; this check verifies overlap with no new source.
        assert next_prepared.wait(2)
        assert len(server.requests) == 4  # still exactly one append in flight
        wait_for(lambda: coordinator.stats()["prepared_frames"] == prefix_frames)
        release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(4)
        assert output_sequences(received) == list(range(0, (4 + prefix_frames) * 3, 3))
        assert all(row["state"] == "committed" for row in coordinator._spool.records())
        # The final empty append is not a duration sample for real frame batches.
        assert len(coordinator._append_durations) == 5
    finally:
        release.set()
        coordinator.close()


def test_prepare_forecast_warmup_bounds_small_batches_and_reset(tmp_path):
    _, _, coordinator = setup_pipeline(tmp_path)
    try:
        with coordinator._condition:
            coordinator._native_ready = coordinator._generation
            coordinator._requesting = SimpleNamespace(started=100.0)
            coordinator._pending.extend(SimpleNamespace(size=4) for _ in range(3))
            # Warmup does not infer timings, even after a long-running request.
            assert coordinator._preparation_schedule(200)[0] is None
            coordinator._append_durations.extend([2.0] * 3)
            coordinator._prepare_durations.extend([0.4] * 3)
            wait, facts = coordinator._preparation_schedule(100.0)
            assert wait == pytest.approx(1.55)
            assert facts["prepare_lead_seconds"] == pytest.approx(0.45)
            assert coordinator._preparation_schedule(101.55)[0] == pytest.approx(0)
            # A bad prediction cannot postpone work beyond request completion.
            assert coordinator._preparation_schedule(300)[0] == 0
            coordinator._prepare_durations.extend([5.0] * 3)
            assert coordinator._preparation_schedule(100)[1][
                "prepare_lead_seconds"
            ] == 0.9
            coordinator._pending.pop()
            assert coordinator._preparation_schedule(300)[0] == 0
            assert coordinator._preparation_schedule(100)[1][
                "predicted_prefix_frames"
            ] == 2
            coordinator._pending.pop()
            assert coordinator._preparation_schedule(100)[0] == pytest.approx(1.1)
            assert coordinator._preparation_schedule(100)[1][
                "predicted_prefix_frames"
            ] == 1
            coordinator._requesting.started = 0
            assert coordinator._preparation_schedule(300)[0] is None
            coordinator._requesting = None
            assert coordinator._preparation_schedule(100)[0] == 0
            coordinator.reset("new")
            assert not coordinator._append_durations
            assert not coordinator._prepare_durations
    finally:
        coordinator.close()


@pytest.mark.parametrize("native_fails", [False, True])
def test_result_stage_timing_precedes_commit_and_preserves_durable_native_order(
    tmp_path, monkeypatch, native_fails,
):
    from goodvision_harness import sam_tracking_pipeline

    # Windows monotonic() can quantize 20 ms to 16 ms. Use a high-resolution
    # monotonic test clock; no production timing or stage thresholds change.
    monkeypatch.setattr(sam_tracking_pipeline, "time", SimpleNamespace(
        monotonic=time.perf_counter, sleep=time.sleep,
    ))
    logs, received, native = [], [], []
    monkeypatch.setattr(
        sam_tracking_pipeline, "log_timing",
        lambda event, **fields: logs.append((event, fields)),
    )

    def commit(batch):
        assert any(event == "sam_native_result_timing" for event, _ in logs)
        received.append(batch)

    _, _, coordinator = setup_pipeline(tmp_path, commit)
    spool = coordinator._journal()
    mark, result = spool.mark, spool.result

    def delayed_mark(request_id, state, error_type=None):
        if state == "requesting":
            time.sleep(0.02)
        return mark(request_id, state, error_type)

    def delayed_result(request_id, batch):
        time.sleep(0.03)
        return result(request_id, batch)

    def publish(batch, generation):
        assert spool.records()[0]["state"] == "received"
        assert generation == coordinator._generation
        assert received == []
        native.append(batch)
        time.sleep(0.04)
        if native_fails:
            raise ValueError("private native failure details")

    monkeypatch.setattr(spool, "mark", delayed_mark)
    monkeypatch.setattr(spool, "result", delayed_result)
    coordinator.on_native_batch = publish
    try:
        submit(coordinator, frames(1, 1))
        assert coordinator.wait_until_idle(3)
        assert output_sequences(received) == output_sequences(native) == [3]
        assert spool.records()[0]["state"] == "committed"
        ready = next(fields for event, fields in logs
                     if event == "sam_native_result_timing")
        complete = next(fields for event, fields in logs
                        if event == "sam_coordinator_timing")
        assert ready["request_id"] == received[0].request_id
        assert ready["stream_session_id"] == "camera"
        assert ready["native_publish_outcome"] == (
            "error" if native_fails else "completed"
        )
        for key, minimum in (
            ("journal_request_mark_seconds", 0.02),
            ("journal_result_seconds", 0.03),
            ("native_publish_seconds", 0.04),
        ):
            assert ready[key] >= minimum
            assert complete[key] == ready[key]
        assert complete["append_seconds"] >= ready["journal_request_mark_seconds"]
        assert complete["commit_queue_seconds"] >= (
            ready["journal_result_seconds"] + ready["native_publish_seconds"]
        )
        assert "private native failure details" not in repr(logs)
    finally:
        coordinator.close()


def test_phase_aligned_camera_samples_survive_timestamp_jitter_and_busy_http(tmp_path):
    # Actual source stamps from the failed camera run: each is the next 3fps
    # phase slot, even when the source-frame quantization makes a gap < 1/3s.
    stamps = [
        (0, 0.0), (5, 0.336024), (10, 0.704030), (15, 1.024085),
        (20, 1.360176), (25, 1.696223), (30, 2.000201), (35, 2.336129),
        (40, 2.672581), (45, 3.008155), (49, 3.344327), (54, 3.712244),
    ]
    received = []
    server, _, coordinator = setup_pipeline(tmp_path, received.append, max_fps=3)
    server.release.clear()
    try:
        for sequence, timestamp in stamps:
            assert coordinator.submit(
                b"jpeg", stream_session_id="camera", frame_sequence=sequence,
                timestamp_seconds=timestamp,
            )
        assert server.entered.wait(2)
        assert coordinator.stats()["session_sampled_out"] == 0
        assert coordinator.stats()["session_dropped_backpressure"] == 0
        assert coordinator._spool.stats()["retained_sources"] == len(stamps)
        coordinator.flush()
        server.release.set()
        assert coordinator.wait_until_idle(5)
        assert [(frame.frame_sequence, frame.timestamp_seconds)
                for batch in received for frame in batch.frames] == stamps
        assert all(row["state"] == "committed" for row in coordinator._spool.records())
    finally:
        server.release.set()
        coordinator.close()


@pytest.mark.parametrize(
    "rate,expected", [(3, [0, 5, 10, 15, 20, 25, 30]),
                      (2, [0, 8, 15, 23, 30]), (1.1, [0, 14, 28])],
)
def test_phase_sampling_keeps_configured_rate_and_resets_its_clock(
    tmp_path, rate, expected,
):
    received = []
    _, _, coordinator = setup_pipeline(tmp_path, received.append, max_fps=rate)
    try:
        admitted = []
        for sequence in range(31):
            if coordinator.submit(
                b"jpeg", stream_session_id="camera", frame_sequence=sequence,
                timestamp_seconds=sequence / 15,
            ):
                admitted.append(sequence)
        assert admitted == expected
        assert coordinator.wait_until_idle(5)
        assert output_sequences(received) == admitted
        # A source gap advances to the next slot, with no catch-up burst.
        assert coordinator.submit(
            b"jpeg", stream_session_id="camera", frame_sequence=1000,
            timestamp_seconds=100,
        )
        assert not coordinator.submit(
            b"jpeg", stream_session_id="camera", frame_sequence=1001,
            timestamp_seconds=100.01,
        )
        assert coordinator.wait_until_idle(5)
        coordinator.reset("new")
        assert coordinator.submit(
            b"jpeg", stream_session_id="new", frame_sequence=0, timestamp_seconds=0,
        )
        coordinator.flush()
        assert coordinator.wait_until_idle(5)
        assert received[-1].stream_session_id == "new"
        assert output_sequences(received[-1:]) == [0]
        assert coordinator.stats()["session_sampled_out"] == 0
    finally:
        coordinator.close()


def _park_prepare(coordinator, monkeypatch):
    """Stop the prepare loop from draining _pending, so a backlog would grow."""
    monkeypatch.setattr(
        coordinator, "_preparation_schedule", lambda now: (None, {})
    )


def test_latest_only_drops_oldest_pending_when_behind(tmp_path, monkeypatch):
    """With latest_only, a stalled pipeline keeps only the newest ~capacity frames."""
    _, _, coordinator = setup_pipeline(tmp_path, max_fps=30, latest_only=True)
    try:
        _park_prepare(coordinator, monkeypatch)
        for seq in range(20):
            assert coordinator.submit(
                b"jpeg", stream_session_id="camera", frame_sequence=seq,
                timestamp_seconds=seq * 0.1,
            )
        stats = coordinator.stats()
        assert stats["submitted"] == 20
        # Freshness backpressure caps the backlog at pending_capacity (4)...
        assert stats["disk_pending_frames"] <= 4
        # ...and the dropped remainder is counted.
        assert stats["dropped_backpressure"] == 16
    finally:
        coordinator.close()


def test_without_latest_only_backlog_is_retained(tmp_path, monkeypatch):
    """Default (durable) behaviour keeps every sample even when stalled."""
    _, _, coordinator = setup_pipeline(tmp_path, max_fps=30)  # latest_only off
    try:
        _park_prepare(coordinator, monkeypatch)
        for seq in range(20):
            coordinator.submit(
                b"jpeg", stream_session_id="camera", frame_sequence=seq,
                timestamp_seconds=seq * 0.1,
            )
        stats = coordinator.stats()
        assert stats["dropped_backpressure"] == 0
        assert stats["disk_pending_frames"] == 20  # whole backlog retained
    finally:
        coordinator.close()


def test_latest_only_releases_spool_under_overload(tmp_path, monkeypatch):
    """Dropped samples must free their spooled JPEGs, so sustained overload keeps
    admitting new frames instead of eventually raising TrackingSpoolFull."""
    # Budget fits a few bounded pending frames but not an unbounded backlog.
    _, _, coordinator = setup_pipeline(
        tmp_path, max_fps=30, latest_only=True, spool_max_bytes=16384,
    )
    try:
        _park_prepare(coordinator, monkeypatch)
        for seq in range(40):  # far more than the spool could hold un-dropped
            assert coordinator.submit(
                b"x" * 1000, stream_session_id="camera", frame_sequence=seq,
                timestamp_seconds=seq * 0.1,
            )
        stats = coordinator.stats()
        assert stats["submitted"] == 40
        assert stats["disk_pending_frames"] <= 4      # backlog stayed bounded
        assert stats["dropped_backpressure"] == 36
        assert not stats.get("recovery_required")     # session never failed
        # The journal only retains the small bounded prefix, not all 40 frames.
        assert coordinator._journal().stats()["retained_sources"] <= 4
    finally:
        coordinator.close()


def test_prepare_and_http_progress_while_previous_commit_waits(tmp_path, monkeypatch):
    commit_entered, commit_release = threading.Event(), threading.Event()
    prepared_while_http_busy = threading.Event()
    received = []

    def commit(batch):
        if not received:
            commit_entered.set()
            assert commit_release.wait(3)
        received.append(batch)

    server, client, coordinator = setup_pipeline(tmp_path, commit)
    prepare = client.prepare_append

    def observe_prepare(inputs, **kwargs):
        result = prepare(inputs, **kwargs)
        if inputs[0].frame_sequence == 6:
            prepared_while_http_busy.set()
        return result

    monkeypatch.setattr(client, "prepare_append", observe_prepare)
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 4))
        assert prepared_while_http_busy.wait(2)
        assert len(server.requests) == 1
        server.release.set()
        assert commit_entered.wait(2)
        wait_for(lambda: len(server.requests) == 2)
        assert received == []  # second HTTP finished before first live commit
        commit_release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(3)
        assert output_sequences(received) == [3, 6, 9, 12, 15]
        assert coordinator.stats()["processed_frames"] == 5
        assert coordinator.stats()["state"] == "finished"
        assert server.requests[-1]["is_last_batch"] is True
        assert server.requests[-1]["frames"] == []
    finally:
        server.release.set()
        commit_release.set()
        coordinator.close()


def test_slow_prepare_does_not_block_previous_request_or_commit(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    received = []
    server, client, coordinator = setup_pipeline(tmp_path, received.append)
    prepare = client.prepare_append

    def blocked_prepare(inputs, **kwargs):
        if inputs[0].frame_sequence == 6:
            entered.set()
            assert release.wait(3)
        return prepare(inputs, **kwargs)

    monkeypatch.setattr(client, "prepare_append", blocked_prepare)
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 4))
        assert entered.wait(2)
        server.release.set()
        wait_for(lambda: output_sequences(received) == [3])
        assert coordinator.stats()["preparing_frames"] == 4
        release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(3)
        assert output_sequences(received) == [3, 6, 9, 12, 15]
    finally:
        server.release.set()
        release.set()
        coordinator.close()


def test_ordered_spool_survives_burst_without_replacing_waiting_sources(tmp_path):
    received = []
    server, _, coordinator = setup_pipeline(tmp_path, received.append)
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 40))
        assert coordinator.stats()["session_admitted_frames"] == 41
        assert coordinator.stats()["dropped_backpressure"] == 0
        assert coordinator._spool.stats()["retained_sources"] == 41
        coordinator.flush()
        server.release.set()
        assert coordinator.wait_until_idle(8)
        assert output_sequences(received) == [3 * index for index in range(1, 42)]
        assert coordinator._spool.stats()["retained_sources"] == 0
        assert all(row["state"] == "committed" for row in coordinator._spool.records())
    finally:
        server.release.set()
        coordinator.close()


def test_reset_preserves_old_recovery_facts_and_never_commits_old_epoch(tmp_path):
    received = []
    server, _, coordinator = setup_pipeline(tmp_path, received.append)
    server.release.clear()
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 8))
        wait_for(lambda: coordinator.stats()["prepared_frames"] == 4)
        coordinator.reset("new")
        submit(coordinator, frames(1, 1), session="new")
        coordinator.flush()
        server.release.set()
        assert coordinator.wait_until_idle(5)
        assert [batch.stream_session_id for batch in received] == ["new"]
        assert coordinator.stats()["processed_frames"] == 1
        # The old request completed after reset; only the new frame batch is
        # allowed to train the new run's predictor (not old work or empty EOF).
        assert len(coordinator._append_durations) == 1
        assert len(coordinator._prepare_durations) == 1
        assert coordinator._spool.stats()["retained_sources"] == 9
        assert len({row["generation"] for row in coordinator._spool.records()}) == 2
    finally:
        server.release.set()
        coordinator.close()


def test_commit_backpressure_retries_same_result_without_duplicate_requests(tmp_path):
    blocked = threading.Event()
    release = threading.Event()
    received, attempts = [], []

    def commit(batch):
        attempts.append(batch.request_id)
        if not release.is_set():
            blocked.set()
            raise TrackingCommitBackpressure("inspection capacity")
        received.append(batch)

    server, _, coordinator = setup_pipeline(tmp_path, commit)
    try:
        submit(coordinator, frames(1, 1))
        assert blocked.wait(2)
        submit(coordinator, frames(2, 8))
        wait_for(lambda: coordinator.stats()["received_batches"] >= 2)
        assert coordinator.stats()["failed"] == 0
        assert coordinator._spool.stats()["recoverable_results"] >= 2
        assert len(server.requests) == 2
        assert [
            frame.frame_sequence
            for frame in coordinator.retained_sources(coordinator._committing.result)
        ] == [3]
        release.set()
        coordinator.flush()
        assert coordinator.wait_until_idle(5)
        assert output_sequences(received) == [3 * index for index in range(1, 10)]
        ids = [request["request_id"] for request in server.requests]
        assert len(ids) == len(set(ids))
        assert attempts[0] == received[0].request_id
        assert coordinator.retained_sources(received[0]) == ()
    finally:
        release.set()
        coordinator.close()


@pytest.mark.parametrize("queue_policy", ["durable_ordered", "latest_only"])
def test_commit_failure_retains_original_frames_payload_and_result(
    tmp_path, queue_policy,
):
    def reject(_batch):
        raise ValueError("private callback error")

    server, _, coordinator = setup_pipeline(tmp_path, reject, queue_policy=queue_policy)
    try:
        submit(coordinator, frames(1, 1))
        wait_for(lambda: coordinator.stats()["state"] == "failed")
        assert coordinator.wait_until_idle(3)
        assert coordinator.stats()["recovery_required"] is True
        row = coordinator._spool.records()[0]
        assert row["state"] == "commit_failed"
        assert row["payload"] and row["payload_hash"] and row["result"]
        assert row["error_type"] == "ValueError"
        assert coordinator._spool.load(0, [3])[0].jpeg_bytes == b"jpeg"
        assert not coordinator.submit(
            b"new", stream_session_id="camera", frame_sequence=99, timestamp_seconds=99
        )
        assert len(server.requests) == 1
    finally:
        coordinator.close()


def test_prepared_payload_has_same_wire_contract_and_rejects_other_session(tmp_path):
    _, client, coordinator = setup_pipeline(tmp_path)
    try:
        client.open_session("camera")
        ready = client.prepare_append(frames(1, 1), stream_session_id="camera")
        result = client.append(ready.frames, stream_session_id="camera", prepared=ready)
        assert result.request_id == ready.request_id
        assert result.frames[0].frame_sequence == 3
        other = client.prepare_append(frames(2, 1), stream_session_id="camera")
        client.discard_session()
        client.open_session("camera")
        with pytest.raises(RichLabelProtocolError, match="prepared append"):
            client.append(other.frames, stream_session_id="camera", prepared=other)
    finally:
        coordinator.close()


def test_spool_quota_rejects_new_work_without_erasing_admitted_bytes(tmp_path):
    spool = TrackingSpool(tmp_path / "private.sqlite3", max_bytes=4096)
    frame = frames(1, 1)[0]
    spool.begin(0, "camera")
    spool.admit(0, "camera", frame, 1)
    with pytest.raises(TrackingSpoolFull):
        spool.admit(0, "camera", type(frame)(6, 0.5, b"x" * 2048), 2)
    reopened = TrackingSpool(spool.path, max_bytes=4096)
    assert reopened.load(0, [3]) == (frame,)
    assert reopened.stats()["retained_sources"] == 1


@pytest.mark.parametrize("queue_policy", ["durable_ordered", "latest_only"])
def test_close_waits_for_preparation_to_unwind_and_keeps_uncommitted_sources(
    tmp_path,
    monkeypatch,
    queue_policy,
):
    entered, release = threading.Event(), threading.Event()
    received = []
    _, client, coordinator = setup_pipeline(
        tmp_path, received.append, queue_policy=queue_policy,
    )
    prepare = client.prepare_append

    def blocked_prepare(inputs, **kwargs):
        entered.set()
        assert release.wait(3)
        return prepare(inputs, **kwargs)

    monkeypatch.setattr(client, "prepare_append", blocked_prepare)
    try:
        submit(coordinator, frames(1, 1))
        assert entered.wait(2)
        assert coordinator._spool._connection is not None
        coordinator.close(timeout_seconds=0)
        assert coordinator._spool._connection is None
        assert not coordinator.close_complete()
        release.set()
        wait_for(coordinator.close_complete)
        assert received == []
        assert coordinator._spool.load(0, [3])[0].jpeg_bytes == b"jpeg"
        assert coordinator._spool._connection is None
    finally:
        release.set()
        coordinator.close()


def test_timed_close_still_persists_a_late_validated_sam_result(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    received = []
    _, client, coordinator = setup_pipeline(tmp_path, received.append)
    append = client.append

    def late_append(*args, **kwargs):
        result = append(*args, **kwargs)
        entered.set()
        assert release.wait(3)
        return result

    monkeypatch.setattr(client, "append", late_append)
    try:
        submit(coordinator, frames(1, 1))
        assert entered.wait(2)
        assert coordinator._spool._connection is not None
        coordinator.close(timeout_seconds=0)
        assert coordinator._spool._connection is None
        assert not coordinator.close_complete()
        release.set()
        wait_for(coordinator.close_complete)
        assert received == []  # Retired-run geometry cannot be delivered live.
        rows = coordinator._spool.records()
        assert len(rows) == 1
        assert rows[0]["state"] == "interrupted"
        assert rows[0]["result"] is not None
        assert rows[0]["payload"] is not None
        assert coordinator._spool.load(0, [3])[0].jpeg_bytes == b"jpeg"
        assert coordinator._spool._connection is None
    finally:
        release.set()
        coordinator.close()
