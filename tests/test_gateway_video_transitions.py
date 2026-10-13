"""Disconnect transitions cannot close a later successful local relay."""

from __future__ import annotations

import asyncio

import anyio
import httpx
import pytest
from test_local_gpu_runtime import rig as rig

from goodvision_harness.analysis_gateway import create_app as gateway_app
from goodvision_harness.service import create_app as remote_app


class ReplyBody(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content
        self.closed = False

    async def __aiter__(self):
        yield self.content

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize(
    "path",
    [
        "/v1/video/connect",
        "/v1/fleet/cameras/camera01/connect",
    ],
)
def test_old_disconnect_finishes_before_new_connection_is_accepted(
    rig,
    monkeypatch,
    tmp_path,
    path,
):
    runtime = rig.make()
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "local")
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "gateway.json"))
    constructed = []

    class Relay:
        def __init__(self, *_args):
            self.closed = 0
            constructed.append(self)

        async def start(self):
            pass

        async def close(self):
            await asyncio.sleep(0)
            self.closed += 1

    monkeypatch.setattr("goodvision_harness.video_relay.LocalVideoRelay", Relay)

    async def audit():
        disconnect_path = path.removesuffix("connect") + "disconnect"
        effective = asyncio.Event()
        release_reply = asyncio.Event()
        connect_entered = asyncio.Event()
        transitions = []
        replies = []
        remote = remote_app(lambda: runtime)
        async with remote.router.lifespan_context(remote):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=remote),
                base_url="http://gpu",
            ) as server:

                async def upstream(request):
                    if request.url.path == path:
                        transitions.append("connect requested")
                    response = await server.request(
                        request.method,
                        request.url.path,
                        content=await request.aread(),
                        headers={"content-type": "application/json"},
                    )
                    if request.url.path == disconnect_path and not effective.is_set():
                        assert response.status_code == 200
                        assert not runtime._video_connection.running
                        transitions.append("disconnect effective")
                        effective.set()
                        await release_reply.wait()
                        transitions.append("disconnect reply returned")
                    body = ReplyBody(response.content)
                    replies.append(body)
                    return httpx.Response(
                        response.status_code,
                        stream=body,
                        headers={**response.headers, "x-remote-result": "preserved"},
                    )

                gateway = gateway_app(
                    "http://gpu",
                    transport=httpx.MockTransport(upstream),
                )
                async with gateway.router.lifespan_context(gateway):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=gateway),
                        base_url="http://console",
                    ) as console:
                        payload = {"stream_url": "rtsp://fixture-camera/live"}
                        assert (
                            await console.post(path, json=payload)
                        ).status_code == 200
                        old = constructed[0]
                        video = gateway.state.video
                        original_connect = video.connect

                        async def observed_connect(*args, **kwargs):
                            connect_entered.set()
                            return await original_connect(*args, **kwargs)

                        monkeypatch.setattr(video, "connect", observed_connect)
                        disconnect = asyncio.create_task(
                            console.post(disconnect_path, json={}),
                        )
                        reconnect = None
                        try:
                            await asyncio.wait_for(effective.wait(), 2)
                            reconnect = asyncio.create_task(
                                console.post(path, json=payload)
                            )
                            await asyncio.wait_for(connect_entered.wait(), 2)
                            await asyncio.sleep(0)
                            # This request has reached connect, while the old
                            # remote disconnect succeeded but has not replied.
                            assert transitions.count("connect requested") == 1
                            assert not reconnect.done()
                            release_reply.set()
                            disconnected = await asyncio.wait_for(disconnect, 2)
                            connected = await asyncio.wait_for(reconnect, 2)
                            assert (
                                disconnected.status_code == connected.status_code == 200
                            )
                            assert (
                                disconnected.headers["x-remote-result"] == "preserved"
                            )
                            assert (
                                disconnected.headers["content-type"]
                                == "application/json"
                            )
                            assert (
                                disconnected.json()["connection_state"]
                                == "disconnected"
                            )
                            assert transitions == [
                                "connect requested",
                                "disconnect effective",
                                "disconnect reply returned",
                                "connect requested",
                            ]
                            assert len(constructed) == 2
                            assert old.closed == 1
                            assert constructed[1].closed == 0
                            assert video.relays == {"camera01": constructed[1]}
                        finally:
                            release_reply.set()
                            await asyncio.gather(
                                *(task for task in (disconnect, reconnect) if task),
                                return_exceptions=True,
                            )
                            await server.post(disconnect_path, json={})
                    assert all(body.closed for body in replies)

    asyncio.run(audit())


@pytest.mark.parametrize(
    "path",
    [
        "/v1/video/connect",
        "/v1/fleet/cameras/camera01/connect",
    ],
)
@pytest.mark.parametrize("failure", ["upstream_rejected", "local_cleanup"])
def test_disconnect_preserves_rejected_response_or_retains_failed_cleanup(
    monkeypatch,
    tmp_path,
    path,
    failure,
):
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "local")
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "gateway.json"))
    replies = []
    constructed = []

    class Relay:
        def __init__(self, *_args):
            self.fail = failure == "local_cleanup"
            self.closed = 0
            constructed.append(self)

        async def start(self):
            pass

        async def close(self):
            await asyncio.sleep(0)
            if self.fail:
                self.fail = False
                raise RuntimeError("local receiver close failed")
            self.closed += 1

    monkeypatch.setattr("goodvision_harness.video_relay.LocalVideoRelay", Relay)

    async def upstream(request):
        if request.url.path == path:
            return httpx.Response(200, json={"video_relay": {"id": "fixture"}})
        assert (
            request.url.raw_path
            == (
                path.removesuffix("connect") + "disconnect?reason=user%2Faction"
            ).encode()
        )
        assert request.headers["x-control"] == "preserved"
        assert await request.aread() == b"opaque-disconnect-body"
        reply = ReplyBody(b"upstream-disconnect-result")
        replies.append(reply)
        return httpx.Response(
            409 if failure == "upstream_rejected" else 200,
            stream=reply,
            headers={"content-type": "text/plain", "x-remote-result": "preserved"},
        )

    async def audit():
        gateway = gateway_app("http://gpu", transport=httpx.MockTransport(upstream))
        async with gateway.router.lifespan_context(gateway):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=gateway),
                base_url="http://console",
            ) as console:
                assert (await console.post(path, json={})).status_code == 200
                relay = constructed[0]
                disconnect_path = (
                    path.removesuffix("connect") + "disconnect?reason=user%2Faction"
                )
                request = console.post(
                    disconnect_path,
                    content=b"opaque-disconnect-body",
                    headers={"x-control": "preserved"},
                )
                if failure == "local_cleanup":
                    with pytest.raises(RuntimeError, match="receiver close failed"):
                        await request
                    assert replies[0].closed
                else:
                    response = await request
                    assert response.status_code == 409
                    assert response.content == b"upstream-disconnect-result"
                    assert response.headers["content-type"] == "text/plain"
                    assert response.headers["x-remote-result"] == "preserved"
                assert relay.closed == 0
                assert gateway.state.video.relays == {"camera01": relay}
                assert all(reply.closed for reply in replies)
                # A failed local close remains owned for another attempt and
                # is cleaned at shutdown, rather than silently discarded.
        assert relay.closed == 1

    asyncio.run(audit())


@pytest.mark.parametrize("cancellation", ["anyio", "asyncio-repeat"])
def test_cancelled_disconnect_closes_upstream_before_releasing_transition(
    monkeypatch,
    tmp_path,
    cancellation,
):
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "gateway.json"))

    async def audit():
        close_entered = asyncio.Event()
        release_cleanup = asyncio.Event()
        relay_entered = asyncio.Event()
        cancelled = []

        class Body(ReplyBody):
            async def aclose(self):
                close_entered.set()
                if cancellation == "asyncio-repeat":
                    await release_cleanup.wait()
                await anyio.sleep(0)
                self.closed = True

        class Relay:
            def __init__(self):
                self.attempts = 0
                self.closed = False

            async def close(self):
                self.attempts += 1
                if self.attempts == 1:
                    relay_entered.set()
                    await asyncio.Event().wait()
                self.closed = True

        body = Body(b"upstream accepted disconnect")
        gateway = gateway_app(
            "http://gpu",
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, stream=body),
            ),
        )
        relay = Relay()
        gateway.state.video.relays["camera01"] = relay
        async with gateway.router.lifespan_context(gateway):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=gateway),
                base_url="http://console",
            ) as console:

                async def disconnect():
                    try:
                        await console.post("/v1/video/disconnect", json={})
                    except asyncio.CancelledError:
                        cancelled.append(True)
                        raise

                try:
                    if cancellation == "anyio":
                        async with anyio.create_task_group() as requests:
                            requests.start_soon(disconnect)
                            with anyio.fail_after(2):
                                await relay_entered.wait()
                            requests.cancel_scope.cancel()
                    else:
                        request = asyncio.create_task(disconnect())
                        await asyncio.wait_for(relay_entered.wait(), 2)
                        request.cancel()
                        await asyncio.wait_for(close_entered.wait(), 2)
                        request.cancel()
                        await asyncio.sleep(0)
                        assert not request.done(), "upstream cleanup was abandoned"
                        assert gateway.state.video.lock.locked()
                        release_cleanup.set()
                        with pytest.raises(asyncio.CancelledError):
                            await request
                    assert cancelled == [True]
                    assert body.closed
                    assert not gateway.state.video.lock.locked()
                    assert gateway.state.video.relays == {"camera01": relay}
                finally:
                    release_cleanup.set()
        assert relay.closed

    asyncio.run(audit())
