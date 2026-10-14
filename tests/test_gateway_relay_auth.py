"""Public gateway callers cannot retrieve the private relay negotiation."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from test_local_gpu_runtime import rig as rig
from test_relay_auth import KEY, PRIVATE_AUTH, PRIVATE_URL, private_runtime

from goodvision_harness.analysis_gateway import create_app as gateway_app
from goodvision_harness.relay_auth import RELAY_AUTH_HEADER
from goodvision_harness.service import create_app as remote_app


class Body(httpx.AsyncByteStream):
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
@pytest.mark.parametrize("mode", ["local", "gpu", "missing-key"])
def test_gateway_privilege_is_internal_and_never_returned_to_browser(
    rig,
    monkeypatch,
    tmp_path,
    path,
    mode,
):
    monkeypatch.setenv(
        "HARNESS_VIDEO_PULL_LOCATION", "gpu" if mode == "gpu" else "local"
    )
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "gateway.json"))
    monkeypatch.delenv("HARNESS_RELAY_KEY_FILE", raising=False)
    if mode == "missing-key":
        monkeypatch.delenv("HARNESS_RELAY_KEY", raising=False)
    else:
        monkeypatch.setenv("HARNESS_RELAY_KEY", KEY)
    runtime = private_runtime(rig)
    captured = []
    received = []
    bodies = []

    class Relay:
        def __init__(self, _origin, _path, options):
            captured.append(options)

        async def start(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr("goodvision_harness.video_relay.LocalVideoRelay", Relay)

    async def audit():
        remote = remote_app(lambda: runtime)
        async with remote.router.lifespan_context(remote):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=remote),
                base_url="http://gpu",
            ) as server:

                async def upstream(request):
                    received.append(request)
                    response = await server.request(
                        request.method,
                        str(request.url),
                        content=await request.aread(),
                        headers=request.headers,
                    )
                    body = Body(response.content)
                    bodies.append(body)
                    return httpx.Response(
                        response.status_code,
                        stream=body,
                        headers=response.headers,
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
                        try:
                            # Even the correct fake internal key from a caller
                            # must be stripped before generic GPU-side pull.
                            response = await console.post(
                                path,
                                json={"relay": True},
                                headers={
                                    RELAY_AUTH_HEADER: (
                                        "spoofed-browser-key"
                                        if mode == "local"
                                        else KEY
                                    ),
                                },
                            )
                            assert "video_relay" not in response.text
                            assert PRIVATE_URL not in response.text
                            assert PRIVATE_AUTH not in response.text
                            assert KEY not in response.text
                            if mode == "local":
                                assert response.status_code == 200
                                assert len(received) == len(captured) == 1
                                auth_headers = [
                                    value
                                    for name, value in received[0].headers.multi_items()
                                    if name.lower() == RELAY_AUTH_HEADER.lower()
                                ]
                                assert auth_headers == [KEY]
                                assert captured[0]["stream_url"] == PRIVATE_URL
                                assert captured[0]["whep_headers"] == {
                                    "Authorization": PRIVATE_AUTH,
                                }
                                assert captured[0]["id"] not in response.text
                                assert runtime._video_connection.running
                                assert list(gateway.state.video.relays) == ["camera01"]
                            else:
                                assert response.status_code == (
                                    403 if mode == "gpu" else 503
                                )
                                assert not captured
                                assert not gateway.state.video.relays
                                assert runtime._video_connection is None
                                if mode == "gpu":
                                    assert len(received) == 1
                                    assert RELAY_AUTH_HEADER not in received[0].headers
                                else:
                                    assert not received
                            assert all(body.closed for body in bodies)
                        finally:
                            if runtime._video_connection is not None:
                                await server.post(
                                    path.removesuffix("connect") + "disconnect",
                                    json={},
                                )

    asyncio.run(audit())
