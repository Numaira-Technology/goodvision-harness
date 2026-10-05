"""Deterministic recovery lifecycle edges; no real HTTP or inference."""

from dataclasses import replace
from threading import Event
from time import monotonic

import httpx
import pytest
from test_perception_client import RouteServer
from test_sam_transport_recovery import LostReply, pipeline
from test_video_tracking_session import frames, submit, wait_for


@pytest.mark.parametrize("status", [404, 410])
def test_already_removed_remote_session_allows_fresh_epoch(tmp_path, status):
    class AlreadyGone(LostReply):
        async def __call__(self, request):
            response = await super().__call__(request)
            if request.method == "DELETE":
                return httpx.Response(status, json={"detail": "already removed"})
            return response

    server = AlreadyGone()
    client, coordinator, received = pipeline(tmp_path, server)
    try:
        submit(coordinator, frames(1, 1))
        assert coordinator.wait_until_idle(2)
        assert client._native_session is None
        assert coordinator.stats()["recovering"]
        submit(coordinator, frames(2, 1))
        assert coordinator.wait_until_idle(2)
        assert server.creates == ["camera", "camera"]
        assert [item.predictor_epoch for item in received] == ["epoch-2"]
        assert coordinator.stats()["state"] == "running"
        assert not coordinator.stats()["recovery_required"]
    finally:
        coordinator.close()


def test_repeated_lost_replies_never_replay_failed_sources(tmp_path):
    class RepeatedLostReply(RouteServer):
        def __init__(self):
            super().__init__("sam_live")

        async def __call__(self, request):
            response = await super().__call__(request)
            if request.url.path.endswith("/append"):
                session = request.url.path.split("/")[-2]
                if session in {"video-1", "video-2"}:
                    raise httpx.RemoteProtocolError("lost reply", request=request)
            return response

    server = RepeatedLostReply()
    _, coordinator, received = pipeline(tmp_path, server)
    try:
        for index in (1, 2, 3):
            submit(coordinator, frames(index, 1))
            assert coordinator.wait_until_idle(2)
        assert [request["frames"][0]["frame_sequence"]
                for request in server.requests] == [3, 6, 9]
        assert len({request["request_id"] for request in server.requests}) == 3
        assert server.deletes == ["video-1", "video-2"]
        assert server.creates == ["camera", "camera", "camera"]
        assert [item.predictor_epoch for item in received] == ["epoch-3"]
        assert coordinator.stats()["transport_recoveries"] == 2
        assert coordinator.stats()["transport_abandoned_frames"] == 2
        assert not coordinator.stats()["recovery_required"]
        assert coordinator._last_input == (9, 0.75)
    finally:
        coordinator.close()


def test_candidate_expiring_during_cleanup_is_not_sent_to_fresh_epoch(tmp_path):
    deleting, release_delete = Event(), Event()

    class BlockedCleanup(LostReply):
        async def __call__(self, request):
            if request.method == "DELETE" and not release_delete.is_set():
                deleting.set()
                # Block only this in-memory protocol double, not the test thread.
                import asyncio

                while not release_delete.is_set():
                    await asyncio.sleep(0.002)
            return await super().__call__(request)

    server = BlockedCleanup()
    server.release.clear()
    _, coordinator, received = pipeline(tmp_path, server)
    try:
        submit(coordinator, frames(1, 1))
        assert server.entered.wait(2)
        submit(coordinator, frames(2, 1))
        server.release.set()
        assert deleting.wait(2)
        # Advance this source's age deterministically rather than sleeping until
        # the freshness deadline. New input remains available during cleanup.
        with coordinator._condition:
            coordinator._pending[-1] = replace(
                coordinator._pending[-1], arrived=monotonic() - 31,
            )
            coordinator._condition.notify_all()
        wait_for(lambda: coordinator.stats()["candidate_frames"] == 0)
        release_delete.set()
        assert coordinator.wait_until_idle(2)
        assert server.creates == ["camera"]
        assert coordinator.stats()["recovering"]
        assert len(server.requests) == 1
        submit(coordinator, frames(3, 1))
        assert coordinator.wait_until_idle(2)
        assert [request["frames"][0]["frame_sequence"]
                for request in server.requests] == [3, 9]
        assert received[0].frames[0].frame_sequence == 9
        assert received[0].predictor_epoch == "epoch-2"
    finally:
        release_delete.set()
        server.release.set()
        coordinator.close()


def test_rejected_recovery_binding_cleans_newly_opened_remote_session(tmp_path):
    server = LostReply()
    client, coordinator, received = pipeline(
        tmp_path, server, on_native_recovery=lambda session, generation: False,
    )
    try:
        submit(coordinator, frames(1, 1))
        assert coordinator.wait_until_idle(2)
        submit(coordinator, frames(2, 1))
        assert coordinator.wait_until_idle(2)
        assert coordinator.stats()["state"] == "failed"
        assert coordinator.stats()["recovery_required"]
        assert server.creates == ["camera", "camera"]
        assert server.deletes == ["video-1", "video-2"]
        assert client._native_session is None
        assert len(server.requests) == 1
        assert received == []
    finally:
        coordinator.close()
