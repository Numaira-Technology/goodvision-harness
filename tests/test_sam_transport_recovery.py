"""Lost replies are abandoned only on the live freshness path; no real HTTP."""

import threading

import httpx
import pytest
from test_perception_client import RouteServer
from test_video_tracking_session import frames, submit

from goodvision_harness.perception import PerceptionSessionHTTPClient
from goodvision_harness.sam_tracking_pipeline import (
    PipelinedVideoSessionTrackingCoordinator,
)


class LostReply(RouteServer):
    def __init__(self, *, error_type=httpx.RemoteProtocolError, fail_delete=False):
        super().__init__("sam_live")
        self.error_type = error_type
        self.fail_delete = fail_delete
        self.lost = False

    async def __call__(self, request):
        if self.fail_delete and request.method == "DELETE":
            raise httpx.ConnectError("mock unreachable", request=request)
        response = await super().__call__(request)
        if request.url.path.endswith("/append") and not self.lost:
            self.lost = True
            # The native worker has ALREADY accepted and advanced this frame.
            raise self.error_type("mock lost response", request=request)
        return response


def pipeline(tmp_path, server, *, policy="latest_only", **kwargs):
    client = PerceptionSessionHTTPClient(
        "http://worker.test",
        method="sam_live",
        timeout_seconds=4,
        transport=httpx.MockTransport(server),
    )
    received = []
    coordinator = PipelinedVideoSessionTrackingCoordinator(
        client,
        queue_policy=policy,
        max_fps=4,
        batch_frames=1,
        batch_seconds=0.01,
        spool_directory=tmp_path,
        on_batch=received.append,
        **kwargs,
    )
    return client, coordinator, received


@pytest.mark.parametrize("policy", ["latest_only", "realtime_latest"])
def test_lost_reply_restarts_one_epoch_and_uses_newest_unsent_source(tmp_path, policy):
    server = LostReply()
    server.release.clear()
    rebound = []
    client, coordinator, received = pipeline(
        tmp_path,
        server,
        policy=policy,
        on_native_recovery=lambda session, generation: rebound.append(
            (session, generation)
        ),
    )
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 1))
        submit(coordinator, frames(3, 1))
        server.release.set()
        assert coordinator.wait_until_idle(3)
        assert len(server.requests) == 2
        first, next_request = server.requests
        assert first["frames"][0]["frame_sequence"] == 3
        assert next_request["frames"][0]["frame_sequence"] == 9
        assert first["request_id"] != next_request["request_id"]
        assert server.sessions["video-1"]["append_count"] == 1
        assert server.creates == ["camera", "camera"]
        assert server.deletes == ["video-1"]
        assert rebound == [("camera", 1)]
        assert [batch.predictor_epoch for batch in received] == ["epoch-2"]
        assert coordinator._last_input == (9, 0.75)
        assert coordinator.stats()["transport_recoveries"] == 1
        assert coordinator.stats()["transport_abandoned_frames"] == 1
        assert not coordinator.stats()["recovery_required"]
        assert not coordinator.stats()["recovering"]
        records = coordinator._journal().records()
        assert records[0]["state"] == "transport_abandoned"
        assert records[0]["payload"] is not None
        assert client._native_session.predictor_epoch == "epoch-2"
    finally:
        server.release.set()
        coordinator.close()


def test_no_candidate_allows_later_ingress_and_does_not_reset_other_camera(tmp_path):
    broken = LostReply()
    healthy = RouteServer("sam_live")
    _, coordinator, received = pipeline(tmp_path / "broken", broken)
    _, other, other_received = pipeline(tmp_path / "healthy", healthy)
    try:
        submit(other, frames(1, 1), session="other")
        assert other.wait_until_idle(2)
        submit(coordinator, frames(1, 1))
        assert coordinator.wait_until_idle(2)
        assert coordinator.stats()["recovering"]
        assert broken.creates == ["camera"]
        submit(coordinator, frames(2, 1))
        submit(other, frames(2, 1), session="other")
        assert coordinator.wait_until_idle(2)
        assert other.wait_until_idle(2)
        assert received[0].frames[0].frame_sequence == 6
        assert healthy.creates == ["other"]
        assert healthy.deletes == []
        assert [batch.predictor_epoch for batch in other_received] == [
            "epoch-1",
            "epoch-1",
        ]
    finally:
        coordinator.close()
        other.close()


def test_durable_mode_replays_identical_request_without_new_epoch(tmp_path):
    server = LostReply()
    _, coordinator, _ = pipeline(tmp_path, server, policy="durable_ordered")
    try:
        submit(coordinator, frames(1, 1))
        assert coordinator.wait_until_idle(3)
        # This double does not implement worker idempotence; just prove the
        # existing identical replay policy is retained, without a native reset.
        assert len(server.requests) == 2
        assert server.requests[0] == server.requests[1]
        assert server.creates == ["camera"]
        assert coordinator.stats()["transport_recoveries"] == 0
    finally:
        coordinator.close()


def test_failed_cleanup_reports_failure_and_keeps_remote_handle(tmp_path):
    server = LostReply(fail_delete=True)
    client, coordinator, received = pipeline(tmp_path, server)
    try:
        submit(coordinator, frames(1, 1))
        assert coordinator.wait_until_idle(2)
        assert coordinator.stats()["state"] == "failed"
        assert coordinator.stats()["recovery_required"]
        assert coordinator.stats()["last_error"] == "ConnectError"
        assert client._native_session.video_session_id == "video-1"
        assert server.creates == ["camera"]
        assert received == []
    finally:
        coordinator.close()


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.HTTPStatusError])
def test_non_recoverable_failures_do_not_restart(tmp_path, error_type):
    server = LostReply(error_type=error_type)
    if error_type is httpx.HTTPStatusError:
        # Real response status failure rather than constructing a status exception.
        server.lost = True
        server.status = 503
    _, coordinator, received = pipeline(tmp_path, server)
    try:
        submit(coordinator, frames(1, 1))
        assert coordinator.wait_until_idle(2)
        assert coordinator.stats()["state"] == "failed"
        assert coordinator.stats()["transport_recoveries"] == 0
        assert server.creates == ["camera"]
        assert received == []
    finally:
        coordinator.close()


def test_reset_while_reopening_does_not_rebind_or_publish_stale_epoch(
    tmp_path, monkeypatch
):
    server = LostReply()
    rebound = []
    client, coordinator, received = pipeline(
        tmp_path,
        server,
        on_native_recovery=lambda session, generation: rebound.append(
            (session, generation)
        ),
    )
    entered, release = threading.Event(), threading.Event()
    original = client.open_session

    def blocked_open(session):
        native = original(session)
        if native.predictor_epoch == "epoch-2":
            entered.set()
            assert release.wait(3)
        return native

    monkeypatch.setattr(client, "open_session", blocked_open)
    try:
        submit(coordinator, frames(1, 1))
        assert coordinator.wait_until_idle(2)
        submit(coordinator, frames(2, 1))
        assert entered.wait(2)
        coordinator.reset("replacement")
        submit(coordinator, frames(3, 1), session="replacement")
        release.set()
        assert coordinator.wait_until_idle(3)
        assert rebound == []
        assert len(received) == 1
        assert received[0].stream_session_id == "replacement"
        assert received[0].predictor_epoch == "epoch-3"
        assert server.deletes == ["video-1", "video-2"]
    finally:
        release.set()
        coordinator.close()


def test_flush_after_recovery_with_no_candidate_finishes_without_old_eof(tmp_path):
    server = LostReply()
    _, coordinator, _ = pipeline(tmp_path, server)
    try:
        # Ensure this run had an earlier successful batch before its lost reply.
        server.lost = True
        submit(coordinator, frames(1, 1))
        assert coordinator.wait_until_idle(2)
        server.lost = False
        submit(coordinator, frames(2, 1))
        assert coordinator.wait_until_idle(2)
        coordinator.flush()
        assert coordinator.wait_until_idle(2)
        assert coordinator.stats()["state"] == "finished"
        assert len(server.requests) == 2
    finally:
        coordinator.close()


def test_journal_failure_after_http_return_keeps_request_worker_alive(
    tmp_path, monkeypatch
):
    server = RouteServer("sam_live")
    _, coordinator, _ = pipeline(tmp_path, server)

    def failed_result(*args):
        raise OSError("mock journal write failure")

    monkeypatch.setattr(coordinator._journal(), "result", failed_result)
    try:
        submit(coordinator, frames(1, 1))
        assert coordinator.wait_until_idle(2)
        assert coordinator.stats()["state"] == "failed"
        assert coordinator._thread.is_alive()
        assert coordinator.stats()["transport_recoveries"] == 0
    finally:
        coordinator.close()
