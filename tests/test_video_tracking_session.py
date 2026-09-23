"""Continuous source correspondence and lifecycle via the real HTTP client.

The server here is a deterministic protocol double, never a GPU/model claim.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import importlib.util
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from goodvision_harness.rich_labeling import RichLabelProtocolError, RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.video_tracking import MAX_RESPONSE_BYTES, VideoFrame
from goodvision_harness.video_tracking_session import (
    VideoSessionTrackingCoordinator,
    VideoSessionTrackingHTTPClient,
    continuous_source_history_seconds,
)


def frames(start=0, count=16):
    # Sparse capture sequence numbers must not be interpreted as missing inputs.
    return tuple(
        VideoFrame(index * 3, index / 4, b"jpeg")
        for index in range(start, start + count)
    )


@pytest.mark.parametrize(
    "fps,expected", [(4, 32), (0.5, 88), (1, 56), (2, 40), (15, 32)]
)
def test_continuous_retention_covers_two_requests_within_runtime_cap(fps, expected):
    value = continuous_source_history_seconds(fps)
    assert value == expected
    assert 32 <= value <= 88


@pytest.mark.parametrize("fps", [0, 0.49, -1, True, float("nan"), float("inf"), "4"])
def test_continuous_retention_rejects_unbounded_or_invalid_source_rates(fps):
    with pytest.raises(ValueError):
        continuous_source_history_seconds(fps)


def test_remote_second_response_keeps_original_evidence_without_hiding_true_lag():
    # Reproduce the real failure: the first request emits 0/.25 after capture
    # passes 10s; the next emits .5..7.25 after capture reaches 17.25s.
    results = []
    for history in (16, continuous_source_history_seconds(4)):
        runtime = SceneRuntime(
            "camera",
            "source",
            mode="active",
            tracking_mode="batch",
            native_memory_scope="session",
            tracking_history_seconds=history,
        )
        try:
            for seq in range(44):
                runtime.observe_frame(b"original-jpeg", seq / 4, seq, force=True)
            first = tuple(
                RichLabelSnapshot("camera", seq, seq / 4, ()) for seq in range(2)
            )
            assert runtime.observe_tracking_batch(
                first, completed_timestamp_seconds=0.25, completed_frame_sequence=1
            )
            for seq in range(44, 70):
                runtime.observe_frame(b"original-jpeg", seq / 4, seq, force=True)
            second = tuple(
                RichLabelSnapshot("camera", seq, seq / 4, ()) for seq in range(2, 30)
            )
            accepted = runtime.observe_tracking_batch(
                second, completed_timestamp_seconds=7.25, completed_frame_sequence=29
            )
            state = runtime.snapshot()
            results.append(accepted)
            assert state["capture_timestamp_seconds"] == 17.25
            if accepted:
                assert state["tracking_lag_seconds"] == 10
                assert state["tracking_completed_timestamp_seconds"] == 7.25
                assert state["scene"]["events"] == []
            else:
                assert state["tracking_last_rejection"] == "evidence_expired"
        finally:
            runtime.close()
    assert results == [False, True]


def limits():
    return {
        "max_frames": 16,
        "max_request_bytes": 32 * 1024 * 1024,
        "max_jpeg_bytes": 8 * 1024 * 1024,
        "max_total_jpeg_bytes": 24 * 1024 * 1024,
        "max_image_pixels": 3840 * 2160,
        "max_total_image_pixels": 64 * 1024 * 1024,
        "inference_max_side": 1280,
        "max_objects": 16,
        "request_timeout_seconds": 60,
        "max_inflight_requests": 1,
        "startup_frames": 16,
        "max_append_frames": 16,
        "history_frames": 32,
        "max_session_frames": 65536,
        "max_cumulative_native_ids": 2048,
        "idle_seconds": 120,
        "max_active_sessions": 1,
    }


class SessionServer:
    def __init__(self):
        self.sessions = {}
        self.creates, self.requests, self.deletes = [], [], []
        self.mutate = None
        self.status = 200
        self.entered, self.release, self.cancelled = (
            threading.Event() for _ in range(3)
        )
        self.release.set()
        self.empty_first = False
        self.settings = limits()
        self.raw_body = None

    async def __call__(self, request):
        path = request.url.path
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "enabled": True,
                    "ready": True,
                    "gpu_ready": True,
                    "backend": "sam3.1_video",
                    "memory_scope": "session",
                    "native_sam_video_tracking": True,
                    "sam3_commit": "test-commit",
                    "limits": self.settings,
                    "error_code": None,
                    "active_session": None,
                },
            )
        if request.method == "POST" and path == "/v1/video/sessions":
            source = json.loads(request.content)["stream_session_id"]
            number = len(self.creates) + 1
            identity = {
                "video_session_id": f"video-{number}",
                "stream_session_id": source,
                "predictor_epoch": f"epoch-{number}",
                "backend": "sam3.1_video",
                "memory_scope": "session",
                "state": "active",
            }
            self.sessions[f"video-{number}"] = {
                "identity": identity,
                "pending": [],
                "accepted": None,
                "emitted": None,
                "append_count": 0,
            }
            self.creates.append(source)
            return httpx.Response(200, json=identity | {"limits": self.settings})
        session_id = path.split("/")[4]
        state = self.sessions[session_id]
        if request.method == "DELETE":
            self.deletes.append(session_id)
            return httpx.Response(
                200,
                json=state["identity"]
                | {
                    "state": "closed",
                    "discarded_pending_frames": len(state["pending"]),
                },
            )
        payload = json.loads(request.content)
        self.requests.append(payload)
        self.entered.set()
        try:
            while not self.release.is_set():
                await asyncio.sleep(0.002)
        finally:
            self.cancelled.set()
        if self.status != 200:
            return httpx.Response(
                self.status, json={"detail": "private-source-expired"}
            )
        if self.raw_body is not None:
            return httpx.Response(200, content=self.raw_body)
        state["append_count"] += 1
        state["pending"].extend(payload["frames"])
        if payload["frames"]:
            last = payload["frames"][-1]
            state["accepted"] = last["frame_sequence"], last["timestamp_seconds"]
        count = (
            len(state["pending"])
            if payload["is_last_batch"]
            else max(
                0, len(state["pending"]) - self.settings.get("max_pending_frames", 14)
            )
        )
        if (
            self.empty_first
            and state["append_count"] == 1
            and not payload["is_last_batch"]
        ):
            count = 0
        emitted = state["pending"][:count]
        state["pending"] = state["pending"][count:]
        snapshots = []
        for frame in emitted:
            seq, at = frame["frame_sequence"], frame["timestamp_seconds"]
            epoch = state["identity"]["predictor_epoch"]
            snapshots.append(
                {
                    "stream_session_id": payload["stream_session_id"],
                    "frame_sequence": seq,
                    "timestamp_seconds": at,
                    "objects": [
                        {
                            "track_id": f"sam31-{epoch}-1",
                            "label": "person",
                            "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9},
                            "attributes": {
                                "native_track": {
                                    "backend": "sam3.1_video",
                                    "epoch": epoch,
                                    "object_id": "1",
                                }
                            },
                            "observation_state": "observed",
                            "last_observed_frame_sequence": seq,
                            "last_observed_timestamp_seconds": at,
                        }
                    ]
                    if seq % 6
                    else [],
                }
            )
        if emitted:
            state["emitted"] = (
                emitted[-1]["frame_sequence"],
                emitted[-1]["timestamp_seconds"],
            )
        accepted, emitted_stamp = state["accepted"], state["emitted"] or (None, None)
        result = state["identity"] | {
            "state": "closed" if payload["is_last_batch"] else "active",
            "request_id": payload["request_id"],
            "frames": snapshots,
            "accepted_frame_sequence": accepted[0],
            "accepted_timestamp_seconds": accepted[1],
            "processed_frame_sequence": accepted[0],
            "processed_timestamp_seconds": accepted[1],
            "emitted_frame_sequence": emitted_stamp[0],
            "emitted_timestamp_seconds": emitted_stamp[1],
            "pending_frame_count": len(state["pending"]),
        }
        if self.mutate:
            self.mutate(result)
        return httpx.Response(200, json=result)


@pytest.fixture
def connection():
    server = SessionServer()
    client = VideoSessionTrackingHTTPClient(
        "http://local.test/v1/video/sessions",
        transport=httpx.MockTransport(server),
        timeout_seconds=1,
    )
    try:
        yield server, client
    finally:
        server.release.set()
        client.close()


def submit(coordinator, inputs, session="camera"):
    for frame in inputs:
        assert coordinator.submit(
            frame.jpeg_bytes,
            stream_session_id=session,
            frame_sequence=frame.frame_sequence,
            timestamp_seconds=frame.timestamp_seconds,
        )


def wait_for(predicate, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("bounded worker did not reach expected state")


def test_delayed_prefix_and_empty_tail_flush_cover_every_source_once(connection):
    server, client = connection
    assert client.gpu_ready()
    session = client.open_session("camera")
    first = client.append(frames(), stream_session_id="camera")
    assert [f.frame_sequence for f in first.frames] == [0, 3]
    assert first.accepted_frame_sequence == 45 and first.completed_frame_sequence == 3
    assert first.pending_frame_count == 14
    second = client.append(frames(16), stream_session_id="camera")
    assert [f.frame_sequence for f in second.frames] == list(range(6, 54, 3))
    tail = client.append((), stream_session_id="camera", is_last_batch=True)
    all_frames = first.frames + second.frames + tail.frames
    assert [f.frame_sequence for f in all_frames] == [
        f.frame_sequence for f in frames(count=32)
    ]
    assert len({obj.track_id for f in all_frames for obj in f.objects}) == 1
    assert all(
        result.predictor_epoch == session.predictor_epoch
        for result in (first, second, tail)
    )
    assert tail.pending_frame_count == 0 and tail.state == "closed"
    assert tail.completed_frame_sequence == tail.accepted_frame_sequence == 93
    assert server.requests[-1]["frames"] == []
    assert client.stats()["pending_frame_count"] == 0
    with pytest.raises(RichLabelProtocolError):
        client.append(frames(32), stream_session_id="camera")


def test_short_recording_can_finish_without_invented_startup_frames(connection):
    server, client = connection
    client.open_session("camera")
    result = client.append(
        frames(count=1), stream_session_id="camera", is_last_batch=True
    )
    assert len(result.frames) == 1 and result.pending_frame_count == 0
    assert len(server.requests[0]["frames"]) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(predictor_epoch="wrong"),
        lambda p: p.update(memory_scope="batch"),
        lambda p: p.update(request_id="other"),
        lambda p: p.update(accepted_frame_sequence=True),
        lambda p: p.update(processed_frame_sequence=99),
        lambda p: p.update(emitted_timestamp_seconds=None),
        lambda p: p.update(pending_frame_count=13),
        lambda p: p.update(extra="private"),
        lambda p: p["frames"][0].update(timestamp_seconds=0.01),
        lambda p: p["frames"].reverse(),
        lambda p: p["frames"].pop(0),
        lambda p: p["frames"][1]["objects"][0].update(observation_state="predicted"),
        lambda p: p["frames"][1]["objects"][0].update(
            last_observed_timestamp_seconds=None
        ),
        lambda p: p["frames"][1]["objects"][0]["attributes"]["native_track"].update(
            epoch="old"
        ),
        lambda p: p["frames"][1]["objects"].append(
            copy.deepcopy(p["frames"][1]["objects"][0])
        ),
    ],
)
def test_invalid_response_never_partially_advances_session(connection, mutation):
    server, client = connection
    client.open_session("camera")
    server.mutate = mutation
    with pytest.raises(RichLabelProtocolError):
        client.append(frames(), stream_session_id="camera")
    assert client.stats()["state"] == "invalid"
    assert client.stats()["accepted_frames"] == client.stats()["native_ids"] == 0
    with pytest.raises(RichLabelProtocolError):
        client.append(frames(16), stream_session_id="camera")
    assert len(server.requests) == 1


def test_changed_native_track_name_in_later_append_invalidates_continuity(connection):
    server, client = connection
    client.open_session("camera")
    client.append(frames(), stream_session_id="camera")
    server.mutate = lambda p: p["frames"][1]["objects"][0].update(
        track_id="reused-as-other"
    )
    with pytest.raises(RichLabelProtocolError, match="identity changed"):
        client.append(frames(16), stream_session_id="camera")
    assert client.stats()["accepted_frame_sequence"] == 45


def test_local_bounds_and_regression_fail_before_sending(connection):
    server, client = connection
    client.open_session("camera")
    for inputs, final in (
        ((), False),
        ((), True),
        (frames(count=15), False),
        (frames(count=17), False),
    ):
        with pytest.raises((ValueError, RichLabelProtocolError)):
            client.append(inputs, stream_session_id="camera", is_last_batch=final)
    assert server.requests == []
    client.append(frames(), stream_session_id="camera")
    for inputs in (frames(), (replace(frames(16)[0], timestamp_seconds=0),)):
        with pytest.raises(RichLabelProtocolError):
            client.append(inputs, stream_session_id="camera")
    assert len(server.requests) == 1


def test_worker_byte_and_total_frame_limits_are_enforced_locally(connection):
    server, client = connection
    server.settings["max_request_bytes"] = 100
    server.settings["max_session_frames"] = 16
    client.open_session("camera")
    with pytest.raises(RichLabelProtocolError, match="request limit"):
        client.append(frames(), stream_session_id="camera")
    assert server.requests == []
    client.discard_session()
    server.settings["max_request_bytes"] = 32 * 1024 * 1024
    client.open_session("camera")
    client.append(frames(), stream_session_id="camera")
    with pytest.raises(RichLabelProtocolError, match="frame budget"):
        client.append(frames(16), stream_session_id="camera")
    client.append((), stream_session_id="camera", is_last_batch=True)


@pytest.mark.parametrize("status", [409, 410, 503])
def test_expired_or_failed_append_requires_explicit_new_epoch(connection, status):
    server, client = connection
    session = client.open_session("camera")
    server.status = status
    with pytest.raises(httpx.HTTPStatusError):
        client.append(frames(), stream_session_id="camera")
    assert client.stats()["state"] == "invalid"
    assert "private" not in json.dumps(client.stats())
    client.discard_session()
    server.status = 200
    new = client.open_session("camera")
    assert new.predictor_epoch != session.predictor_epoch
    assert server.deletes == [session.video_session_id]


def test_append_total_deadline_cancels_actual_io_without_retry():
    server = SessionServer()
    client = VideoSessionTrackingHTTPClient(
        "http://local.test/v1/video/sessions",
        transport=httpx.MockTransport(server),
        timeout_seconds=0.1,
    )
    try:
        client.open_session("camera")
        server.release.clear()
        before = time.monotonic()
        with pytest.raises(httpx.TimeoutException):
            client.append(frames(), stream_session_id="camera")
        assert time.monotonic() - before < 0.5
        assert server.cancelled.wait(1)
        assert len(server.requests) == 1
        assert client.stats()["state"] == "invalid"
    finally:
        server.release.set()
        client.close()


@pytest.mark.parametrize(
    "body",
    [b'{"x":1,"x":2}', b" " * (MAX_RESPONSE_BYTES + 1)],
    ids=["duplicate-json", "oversize-body"],
)
def test_duplicate_json_and_oversized_response_fail_closed(connection, body):
    server, client = connection
    client.open_session("camera")
    server.raw_body = body
    with pytest.raises(RichLabelProtocolError):
        client.append(frames(), stream_session_id="camera")
    assert client.stats()["state"] == "invalid"


def test_coordinator_keeps_one_epoch_and_flushes_tail_before_idle(connection):
    server, client = connection
    results = []
    coordinator = VideoSessionTrackingCoordinator(client, on_batch=results.append)
    try:
        coordinator.reset("camera")
        submit(coordinator, frames())
        assert coordinator.wait_until_idle()
        submit(coordinator, frames(16))
        assert coordinator.wait_until_idle()
        coordinator.flush()
        assert coordinator.wait_until_idle()
        assert len(results) == 3
        assert len({r.predictor_epoch for r in results}) == 1
        assert [f.frame_sequence for r in results for f in r.frames] == list(
            range(0, 96, 3)
        )
        assert len(server.creates) == 1
        status = coordinator.stats()
        assert status["state"] == "finished" and status["pending_frame_count"] == 0
        assert (
            status["accepted_frame_sequence"] == status["emitted_frame_sequence"] == 93
        )
        assert coordinator.snapshot_at(0).frame_sequence == 0
    finally:
        coordinator.close()


def test_native_display_is_available_before_session_scene_commit(connection):
    _, client = connection
    order = []

    def commit(result):
        assert order[-1] == ("native", result)
        order.append(("scene", result))

    coordinator = VideoSessionTrackingCoordinator(client, on_batch=commit)
    coordinator.on_native_batch = lambda result, generation: order.append(
        ("native", result),
    )
    try:
        coordinator.reset("camera")
        submit(coordinator, frames())
        assert coordinator.wait_until_idle()
        assert [kind for kind, _ in order] == ["native", "scene"]
    finally:
        coordinator.close()


def test_initial_empty_emission_is_not_a_callback_or_negative_scene(connection):
    server, client = connection
    server.empty_first = True
    results = []
    coordinator = VideoSessionTrackingCoordinator(client, on_batch=results.append)
    try:
        submit(coordinator, frames())
        assert coordinator.wait_until_idle()
        assert results == [] and coordinator.latest() is None
        assert coordinator.stats()["accepted_frame_sequence"] == 45
        assert coordinator.stats()["emitted_frame_sequence"] is None
        coordinator.flush()
        assert coordinator.wait_until_idle()
        assert len(results) == 1 and len(results[0].frames) == 16
    finally:
        coordinator.close()


def test_slow_startup_never_sends_fewer_than_sixteen_without_final(connection):
    server, client = connection
    coordinator = VideoSessionTrackingCoordinator(client, max_fps=1, batch_seconds=0.01)
    try:
        submit(coordinator, (VideoFrame(i, i, b"jpeg") for i in range(15)))
        assert not coordinator.wait_until_idle(0.1)
        assert server.requests == []
        coordinator.flush()
        assert coordinator.wait_until_idle()
        assert len(server.requests[0]["frames"]) == 15
        assert server.requests[0]["is_last_batch"] is True
    finally:
        coordinator.close()


def test_small_append_target_keeps_full_startup_then_emits_without_long_timer(
    connection,
):
    server, client = connection
    server.settings.update(hotstart_delay=5, max_pending_frames=4)
    results = []
    coordinator = VideoSessionTrackingCoordinator(
        client, batch_frames=4, batch_seconds=10, on_batch=results.append
    )
    try:
        submit(coordinator, frames(count=15))
        assert not coordinator.wait_until_idle(0.05)
        assert server.creates == server.requests == []
        submit(coordinator, frames(15, 1))
        assert coordinator.wait_until_idle()
        assert len(results[0].frames) == 12
        submit(coordinator, frames(16, 4))
        assert coordinator.wait_until_idle()
        assert [len(body["frames"]) for body in server.requests] == [16, 4]
        assert results[1].predictor_epoch == results[0].predictor_epoch
        assert [frame.frame_sequence for frame in results[1].frames] == [36, 39, 42, 45]
        status = coordinator.stats()
        assert status["startup_frames"] == status["max_append_frames"] == 16
        assert status["append_target_frames"] == status["last_request_frames"] == 4
        assert (
            status["total_queue_wait_seconds"] >= status["last_queue_wait_seconds"] >= 0
        )
        assert status["transport"]["frames"] == 20
        coordinator.flush()
        assert coordinator.wait_until_idle()
        assert [f.frame_sequence for r in results for f in r.frames] == [
            f.frame_sequence for f in frames(count=20)
        ]
    finally:
        coordinator.close()


def test_short_followup_timer_dispatches_partial_without_startup_shortcut(connection):
    server, client = connection
    coordinator = VideoSessionTrackingCoordinator(
        client, batch_frames=4, batch_seconds=0.02
    )
    try:
        submit(coordinator, frames(count=3))
        assert not coordinator.wait_until_idle(0.05)
        assert not server.requests
        submit(coordinator, frames(3, 13))
        assert coordinator.wait_until_idle()
        submit(coordinator, frames(16, 1))
        assert coordinator.wait_until_idle()
        assert [len(body["frames"]) for body in server.requests] == [16, 1]
        assert coordinator.stats()["last_queue_wait_seconds"] >= 0.015
        coordinator.flush()
        assert coordinator.wait_until_idle()
    finally:
        coordinator.close()


def test_small_target_combines_backlog_and_retains_one_inflight_bounded_queue(
    connection,
):
    server, client = connection
    server.release.clear()
    coordinator = VideoSessionTrackingCoordinator(
        client, batch_frames=4, batch_seconds=0.5
    )
    try:
        submit(coordinator, frames())
        assert server.entered.wait(1)
        submit(coordinator, frames(16, 32))
        assert len(server.requests) == 1
        assert coordinator.stats()["pending_frames"] == 16
        assert coordinator.stats()["dropped_backpressure"] == 16
        server.release.set()
        assert coordinator.wait_until_idle()
        assert [len(body["frames"]) for body in server.requests] == [16, 16]
        assert coordinator.stats()["append_target_frames"] == 4
        assert coordinator.stats()["last_request_frames"] == 16
    finally:
        server.release.set()
        coordinator.close()


def test_startup_uses_validated_health_capability_without_inventing_source_frames(
    connection,
):
    server, client = connection
    # Future protocol capability exercised with a double, not a claim that the
    # current native ring supports fewer than sixteen startup frames.
    server.settings.update(
        startup_frames=4, max_append_frames=4, hotstart_delay=3, max_pending_frames=2
    )
    assert client.gpu_ready()
    coordinator = VideoSessionTrackingCoordinator(
        client, batch_frames=2, batch_seconds=10
    )
    try:
        submit(coordinator, frames(count=3))
        assert not coordinator.wait_until_idle(0.05)
        assert not server.requests
        submit(coordinator, frames(3, 1))
        assert coordinator.wait_until_idle()
        submit(coordinator, frames(4, 2))
        assert coordinator.wait_until_idle()
        coordinator.flush()
        assert coordinator.wait_until_idle()
        assert [len(body["frames"]) for body in server.requests] == [4, 2, 0]
        assert [f.frame_sequence for f in coordinator.history()] == [0, 3, 6, 9, 12, 15]
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    "extra",
    [
        {"hotstart_delay": 5},
        {"max_pending_frames": 4},
        {"hotstart_delay": 7, "max_pending_frames": 6},
        {"hotstart_delay": 5, "max_pending_frames": 14},
        {"hotstart_delay": True, "max_pending_frames": 4},
        {"startup_frames": 0},
        {"startup_frames": 17},
    ],
)
def test_malformed_new_session_capabilities_are_rejected_before_native_use(
    connection, extra
):
    server, client = connection
    server.settings.update(extra)
    assert not client.gpu_ready()
    assert client.startup_frames == 16
    with pytest.raises(RichLabelProtocolError):
        client.open_session("camera")
    assert not server.requests


def test_response_cannot_exceed_advertised_hotstart_pending_count(connection):
    server, client = connection
    server.settings.update(hotstart_delay=5, max_pending_frames=4)
    server.empty_first = True
    client.open_session("camera")
    with pytest.raises(RichLabelProtocolError, match="pending source limit"):
        client.append(frames(), stream_session_id="camera")
    assert client.stats()["accepted_frames"] == 0
    assert client.stats()["state"] == "invalid"


def test_backpressure_drops_only_waiting_inputs_not_accepted_delayed_frames(connection):
    server, client = connection
    server.release.clear()
    results = []
    coordinator = VideoSessionTrackingCoordinator(client, on_batch=results.append)
    try:
        submit(coordinator, frames())
        assert server.entered.wait(1)
        before = time.monotonic()
        submit(coordinator, frames(16, 32))
        assert time.monotonic() - before < 0.2
        status = coordinator.stats()
        assert status["pending_frames"] == 16 and status["dropped_backpressure"] == 16
        server.release.set()
        assert coordinator.wait_until_idle()
        coordinator.flush()
        assert coordinator.wait_until_idle()
        delivered = [f.frame_sequence for r in results for f in r.frames]
        assert delivered == [f.frame_sequence for f in frames() + frames(32)]
        assert len(server.creates) == 1
    finally:
        server.release.set()
        coordinator.close()


def test_reset_discards_inflight_old_source_and_creates_new_epoch(connection):
    server, client = connection
    server.release.clear()
    results = []
    coordinator = VideoSessionTrackingCoordinator(client, on_batch=results.append)
    try:
        submit(coordinator, frames(), "old")
        assert server.entered.wait(1)
        coordinator.reset("new")
        submit(coordinator, frames(), "new")
        server.release.set()
        assert coordinator.wait_until_idle()
        assert results and {r.stream_session_id for r in results} == {"new"}
        assert server.creates == ["old", "new"]
        assert "video-1" in server.deletes
        assert coordinator.stats()["stale_results"] == 1
    finally:
        server.release.set()
        coordinator.close()


def test_stop_returns_bounded_and_never_publishes_late_native_result(connection):
    server, client = connection
    server.release.clear()
    results = []
    coordinator = VideoSessionTrackingCoordinator(client, on_batch=results.append)
    submit(coordinator, frames())
    assert server.entered.wait(1)
    before = time.monotonic()
    coordinator.close(timeout_seconds=0.02)
    assert time.monotonic() - before < 0.2
    server.release.set()
    coordinator._thread.join(2)
    assert not coordinator._thread.is_alive()
    assert results == []
    assert "video-1" in server.deletes


def test_callback_rejection_stops_continuity_until_explicit_reset(connection):
    _server, client = connection
    calls = []

    def reject(result):
        calls.append(result)
        raise ValueError("private missing source")

    coordinator = VideoSessionTrackingCoordinator(client, on_batch=reject)
    try:
        submit(coordinator, frames())
        assert coordinator.wait_until_idle()
        assert coordinator.stats()["state"] == "failed"
        assert coordinator.latest() is None
        assert coordinator.stats()["failed"] == 1
        assert not coordinator.submit(
            b"jpeg",
            stream_session_id="camera",
            frame_sequence=100,
            timestamp_seconds=10,
        )
        assert "private" not in json.dumps(coordinator.stats())
        coordinator.reset("new")
        coordinator._on_batch = None
        submit(coordinator, frames(), "new")
        assert coordinator.wait_until_idle()
        assert coordinator.latest().stream_session_id == "new"
    finally:
        coordinator.close()


@pytest.mark.parametrize("hotstart_delay", [3, 5, 15])
@pytest.mark.parametrize("compress", [False, True])
def test_actual_session_worker_routes_match_client_delayed_source_contract(
    hotstart_delay, compress
):
    from fastapi import FastAPI

    from workers.sam_scene.app import decode_jpeg
    from workers.sam_scene.continuous import register_session_routes

    # Reuse only the worker's explicit Python backend double. Health must remain
    # unavailable: this joint transport test does not attest real CUDA readiness.
    spec = importlib.util.spec_from_file_location(
        "session_worker_double",
        Path(__file__).with_name("test_sam31_continuous_worker.py"),
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    selected = fixture.engine(hotstart_delay=hotstart_delay)
    app = FastAPI()
    register_session_routes(app, selected, threading.Lock(), decode_jpeg)
    client = VideoSessionTrackingHTTPClient(
        "http://asgi.test/v1/video/sessions",
        transport=httpx.ASGITransport(app=app),
        timeout_seconds=2,
        transport_max_side=512 if compress else None,
        transport_jpeg_quality=65 if compress else None,
    )
    try:
        assert not client.gpu_ready()
        session = client.open_session("camera")
        assert session.limits["hotstart_delay"] == hotstart_delay
        assert session.limits["max_pending_frames"] == hotstart_delay - 1
        outputs = []
        inputs = []
        for start, count, final in ((0, 16, False), (16, 3, False), (19, 0, True)):
            records = json.loads(
                fixture.payload(count, start=start, final=final, size=(720, 540))
            )["frames"]
            chunk = tuple(
                VideoFrame(
                    f["frame_sequence"],
                    f["timestamp_seconds"],
                    base64.b64decode(f["jpeg_base64"]),
                )
                for f in records
            )
            inputs.extend(chunk)
            outputs.append(
                client.append(chunk, stream_session_id="camera", is_last_batch=final)
            )
        assert [
            frame.frame_sequence for result in outputs for frame in result.frames
        ] == [frame.frame_sequence for frame in inputs]
        assert len({result.predictor_epoch for result in outputs}) == 1
        assert [len(result.frames) for result in outputs] == [
            17 - hotstart_delay,
            3,
            hotstart_delay - 1,
        ]
        assert outputs[-1].state == "closed"
        assert outputs[-1].pending_frame_count == 0
        assert not selected.has_active_session
        assert selected.backend.calls == 3
        metrics = client.transport_stats()
        assert metrics["frames"] == 19
        if compress:
            assert metrics["transport_jpeg_bytes"] < metrics["source_jpeg_bytes"]
    finally:
        client.close()
        selected.close()


def test_long_session_retains_only_bounded_unemitted_source_metadata(connection):
    _server, client = connection
    client.open_session("camera")
    delivered = 0
    for start in range(0, 512, 16):
        result = client.append(frames(start), stream_session_id="camera")
        delivered += len(result.frames)
        assert client.stats()["pending_frame_count"] == 14
        assert client.stats()["native_ids"] == 1
        assert all(
            isinstance(stamp, tuple) and len(stamp) == 2 for stamp in client._sources
        )
    delivered += len(
        client.append((), stream_session_id="camera", is_last_batch=True).frames
    )
    assert delivered == client.stats()["accepted_frames"] == 512


def test_missing_final_sources_and_lag_exceeding_source_bound_are_rejected(connection):
    server, client = connection
    client.open_session("camera")
    client.append(frames(), stream_session_id="camera")
    server.mutate = lambda payload: payload["frames"].pop()
    with pytest.raises(RichLabelProtocolError):
        client.append((), stream_session_id="camera", is_last_batch=True)
    assert client.stats()["state"] == "invalid"
