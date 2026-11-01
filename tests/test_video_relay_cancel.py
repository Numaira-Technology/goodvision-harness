"""Accepted remote relay slots are released when the requesting scope cancels."""

from __future__ import annotations

import asyncio

import anyio
import httpx
import pytest
from test_local_gpu_runtime import rig as rig

from goodvision_harness.gateway_video import GatewayVideo
from goodvision_harness.service import create_app

TEST_RELAY_KEY = "gateway-test-auth-key-0123456789abcdef"


@pytest.fixture(autouse=True)
def configured_relay_key(monkeypatch):
    monkeypatch.setenv("HARNESS_RELAY_KEY", TEST_RELAY_KEY)


@pytest.mark.parametrize(
    "path",
    [
        "/v1/video/connect",
        "/v1/fleet/cameras/camera01/connect",
    ],
)
@pytest.mark.parametrize("cancellation", ["asyncio", "asyncio-repeat", "anyio"])
def test_cancelled_local_start_rolls_back_actual_remote_slot_and_allows_retry(
    rig,
    monkeypatch,
    tmp_path,
    path,
    cancellation,
):
    runtime = rig.make()
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "local")
    constructed = []

    class Relay:
        def __init__(self, *_args):
            self.entered = asyncio.Event()
            self.cleanup_entered = asyncio.Event()
            self.release_cleanup = asyncio.Event()
            self.closed = 0
            constructed.append(self)

        async def start(self):
            self.entered.set()
            if len(constructed) == 1:
                await asyncio.Event().wait()

        async def close(self):
            # A real receiver/socket close yields. AnyIO level cancellation
            # would interrupt this and the following remote rollback unless
            # cleanup is shielded from the request's cancelled scope.
            self.cleanup_entered.set()
            if cancellation == "asyncio-repeat" and self is constructed[0]:
                await self.release_cleanup.wait()
            await asyncio.sleep(0)
            self.closed += 1

    monkeypatch.setattr("goodvision_harness.video_relay.LocalVideoRelay", Relay)

    async def audit():
        app = create_app(lambda: runtime)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gpu",
            ) as remote:
                video = GatewayVideo(
                    "http://gpu",
                    settings_path=tmp_path / "gateway.json",
                )
                payload = {"stream_url": "rtsp://fixture-camera/live"}
                cancelled = []

                async def connect():
                    try:
                        await video.connect(remote, path, payload)
                    except asyncio.CancelledError:
                        cancelled.append(True)
                        raise

                async def wait_started():
                    with anyio.fail_after(2):
                        while not constructed:
                            await asyncio.sleep(0)
                        await constructed[0].entered.wait()
                    assert runtime._video_connection.running
                    assert runtime._video_connection.status()["connection_state"] == (
                        "connecting"
                    )

                try:
                    if cancellation != "anyio":
                        request = asyncio.create_task(connect())
                        await wait_started()
                        request.cancel()
                        if cancellation == "asyncio-repeat":
                            with anyio.fail_after(2):
                                await constructed[0].cleanup_entered.wait()
                            request.cancel()
                            await asyncio.sleep(0)
                            assert not request.done(), "cleanup was abandoned"
                            constructed[0].release_cleanup.set()
                        with pytest.raises(asyncio.CancelledError):
                            await request
                    else:
                        async with anyio.create_task_group() as requests:
                            requests.start_soon(connect)
                            await wait_started()
                            requests.cancel_scope.cancel()

                    assert cancelled == [True]
                    assert constructed[0].closed == 1
                    assert not video.relays
                    assert not runtime._video_connection.running
                    assert runtime._video_connection.status()["connection_state"] == (
                        "disconnected"
                    )
                    # The real runtime rejects connect with 409 while an orphan
                    # adapter remains running. A 200 retry proves it was freed.
                    retry = await video.connect(remote, path, payload)
                    assert retry.status_code == 200
                    assert list(video.relays) == ["camera01"]
                finally:
                    for relay in constructed:
                        relay.release_cleanup.set()
                    await remote.post(
                        path.removesuffix("connect") + "disconnect",
                        json={},
                    )
                    await video.close()

    asyncio.run(audit())
