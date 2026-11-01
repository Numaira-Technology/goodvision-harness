"""Negotiation owns remote acceptance even while its reply is in transit."""

from __future__ import annotations

import asyncio

import anyio
import httpx
import pytest
from test_local_gpu_runtime import rig as rig
from test_video_connection import wait_for

from goodvision_harness import gateway_video
from goodvision_harness.analysis_gateway import create_app as gateway_app
from goodvision_harness.relay_auth import RELAY_AUTH_HEADER
from goodvision_harness.service import create_app as remote_app

KEY = "negotiation-test-auth-key-0123456789abcdef"
PATHS = ["/v1/video/connect", "/v1/fleet/cameras/camera01/connect"]
PAYLOAD = {"stream_url": "rtsp://fixture-camera/live"}


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("cancellation", ["asyncio", "asyncio-repeat", "anyio"])
@pytest.mark.parametrize("status", [200, 409])
def test_cancelled_reply_joins_negotiation_and_rolls_back_only_accepted_slot(
    rig,
    monkeypatch,
    tmp_path,
    path,
    cancellation,
    status,
):
    runtime = rig.make()
    monkeypatch.setenv("HARNESS_RELAY_KEY", KEY)
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "local")
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "gateway.json"))
    monkeypatch.delenv("CLOUD_GPU_CONTROLLER_URL", raising=False)
    constructed = []

    class Relay:
        def __init__(self, *_args):
            constructed.append(self)

        async def start(self):
            await asyncio.sleep(0)

        async def close(self):
            await asyncio.sleep(0)

    monkeypatch.setattr("goodvision_harness.video_relay.LocalVideoRelay", Relay)

    async def audit():
        reply_ready = asyncio.Event()
        release_reply = asyncio.Event()
        calls, owners, cancelled = [], [], []
        request = None
        app = remote_app(lambda: runtime)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gpu",
            ) as server:
                if status == 409:
                    existing = await server.post(
                        path,
                        json={**PAYLOAD, "relay": True},
                        headers={RELAY_AUTH_HEADER: KEY},
                    )
                    assert existing.status_code == 200

                async def upstream(message):
                    calls.append(message.url.path)
                    if len(calls) == 1:
                        owners.append(asyncio.current_task())
                        assert message.extensions["timeout"] == {
                            phase: 30 for phase in ("connect", "read", "write", "pool")
                        }
                    response = await server.request(
                        message.method,
                        str(message.url),
                        content=await message.aread(),
                        headers=message.headers,
                    )
                    if len(calls) == 1:
                        assert response.status_code == status
                        assert runtime._video_connection.running
                        reply_ready.set()
                        await release_reply.wait()
                    return httpx.Response(
                        response.status_code,
                        content=response.content,
                        headers=response.headers,
                    )

                gateway = gateway_app(
                    "http://gpu", transport=httpx.MockTransport(upstream)
                )
                video = gateway.state.video
                async with gateway.router.lifespan_context(gateway):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=gateway),
                        base_url="http://console",
                    ) as console:

                        async def connect():
                            try:
                                await console.post(path, json=PAYLOAD)
                            except asyncio.CancelledError:
                                cancelled.append(True)
                                raise

                        async def wait_reply():
                            with anyio.fail_after(3):
                                await reply_ready.wait()
                            assert video.lock.locked()
                            assert not video.relays

                        try:
                            if cancellation == "anyio":
                                async with anyio.create_task_group() as group:
                                    group.start_soon(connect)
                                    try:
                                        await wait_reply()
                                        initial = runtime._video_connection
                                        group.cancel_scope.cancel()
                                        with anyio.CancelScope(shield=True):
                                            await asyncio.sleep(0)
                                            assert video.lock.locked()
                                            assert not owners[0].done()
                                            release_reply.set()
                                    finally:
                                        release_reply.set()
                            else:
                                request = asyncio.create_task(connect())
                                await wait_reply()
                                initial = runtime._video_connection
                                request.cancel()
                                await asyncio.sleep(0)
                                if cancellation == "asyncio-repeat":
                                    request.cancel()
                                    await asyncio.sleep(0)
                                assert not request.done(), (
                                    "negotiation owner was abandoned"
                                )
                                assert video.lock.locked()
                                assert not owners[0].done()
                                release_reply.set()
                                with pytest.raises(asyncio.CancelledError):
                                    await asyncio.wait_for(request, 3)
                            assert cancelled == [True]
                            assert owners[0].done() and not owners[0].cancelled()
                            assert owners[0].result().status_code == status
                            assert not video.lock.locked()
                            assert not video.relays and not constructed
                            disconnect = path.removesuffix("connect") + "disconnect"
                            if status == 200:
                                assert calls == [path, disconnect]
                                await asyncio.to_thread(
                                    wait_for, lambda: not initial.running
                                )
                                retried = await console.post(path, json=PAYLOAD)
                                assert retried.status_code == 200, retried.text
                                assert runtime._video_connection is not initial
                                assert len(constructed) == 1
                                assert list(video.relays) == ["camera01"]
                            else:
                                assert calls == [path]
                                assert runtime._video_connection is initial
                                assert initial.running
                        finally:
                            release_reply.set()
                            if request is not None:
                                await asyncio.gather(request, return_exceptions=True)
                            await server.post(
                                path.removesuffix("connect") + "disconnect", json={}
                            )

    asyncio.run(audit())


@pytest.mark.parametrize("failure", ["total-timeout", "network-error"])
def test_unconfirmed_negotiation_has_total_budget_and_never_closes_existing_slot(
    rig,
    monkeypatch,
    tmp_path,
    failure,
):
    runtime = rig.make()
    monkeypatch.setenv("HARNESS_RELAY_KEY", KEY)
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "local")
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "gateway.json"))
    monkeypatch.delenv("CLOUD_GPU_CONTROLLER_URL", raising=False)
    # MockTransport ignores HTTP phase timeouts. A stalled response must still
    # hit the production total timeout, without relying on a real network.
    if failure == "total-timeout":
        monkeypatch.setattr(gateway_video, "_CONNECT_TIMEOUT_SECONDS", 0.03)

    async def audit():
        calls, owners, completed = [], [], []
        app = remote_app(lambda: runtime)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gpu",
            ) as server:
                response = await server.post(
                    PATHS[0],
                    json={**PAYLOAD, "relay": True},
                    headers={RELAY_AUTH_HEADER: KEY},
                )
                assert response.status_code == 200
                existing = runtime._video_connection

                async def upstream(request):
                    calls.append(request.url.path)
                    owners.append(asyncio.current_task())
                    try:
                        if failure == "network-error":
                            raise httpx.ReadError("private camera/network secret")
                        await asyncio.Event().wait()
                    finally:
                        completed.append(True)

                gateway = gateway_app(
                    "http://gpu", transport=httpx.MockTransport(upstream)
                )
                async with gateway.router.lifespan_context(gateway):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=gateway),
                        base_url="http://console",
                    ) as console:
                        try:
                            reply = await asyncio.wait_for(
                                console.post(PATHS[0], json=PAYLOAD), 3
                            )
                            assert reply.status_code == 503
                            assert "secret" not in reply.text
                            assert calls == [PATHS[0]]
                            assert completed == [True] and owners[0].done()
                            assert not gateway.state.video.lock.locked()
                            assert not gateway.state.video.relays
                            assert (
                                runtime._video_connection is existing
                                and existing.running
                            )
                        finally:
                            await server.post("/v1/video/disconnect", json={})

    asyncio.run(audit())
