"""All analysis stays remote, including browser frames and streamed output."""

import asyncio
import gzip
import json
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from goodvision_harness.analysis_gateway import create_app


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def test_gateway_never_constructs_local_analysis_runtime(monkeypatch):
    from goodvision_harness.service import HarnessRuntime
    from goodvision_harness.service import create_app as service_app

    monkeypatch.setenv("HARNESS_ANALYSIS_URL", "http://gpu:8080")
    construct = Mock(side_effect=AssertionError("local analysis must not start"))
    monkeypatch.setattr(HarnessRuntime, "from_environment", construct)
    with TestClient(service_app()) as client:
        assert client.get("/health/live").json()["local_analysis"] is False
        assert not hasattr(client.app.state, "runtime")
    construct.assert_not_called()


@pytest.mark.parametrize(
    "path,body,media",
    [
        ("/v1/faces/gallery/person-a", b"\xff\xd8\xffimage", "image/jpeg"),
        (
            "/v1/fleet/cameras/camera02/connect",
            b'{"source":"camera"}',
            "application/json",
        ),
    ],
)
def test_upload_and_escaped_query_are_forwarded_to_fixed_gpu_host(
    path, body, media, monkeypatch
):
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "gpu")
    seen = []
    content = gzip.compress(b'{"remote":true}')
    stream = Chunks([content[:5], content[5:]])

    async def remote(request):
        seen.append((request, await request.aread()))
        return httpx.Response(
            201,
            stream=stream,
            headers={
                "content-encoding": "gzip",
                "content-type": "application/json",
                "connection": "x-private",
                "x-private": "remove",
                "x-result": "keep",
            },
        )

    with TestClient(
        create_app("http://gpu:8080", transport=httpx.MockTransport(remote))
    ) as client:
        response = client.post(
            path + "?name=a%2Fb&frame=2",
            content=body,
            headers={
                "content-type": media,
                "connection": "x-local",
                "x-local": "remove",
            },
        )
        assert response.status_code == 201
        assert response.json() == {"remote": True}
        assert response.headers["x-result"] == "keep"
        assert "x-private" not in response.headers
    request, forwarded = seen[0]
    assert request.url.host == "gpu"
    assert request.url.raw_path == (path + "?name=a%2Fb&frame=2").encode()
    assert forwarded == body
    assert request.headers["content-type"] == media
    assert "x-local" not in request.headers
    assert stream.closed


def test_sse_and_mjpeg_are_streamed_without_json_decoding():
    payload = [b"event: result\ndata: {}\n\n", b"event: done\ndata: {}\n\n"]
    stream = Chunks(payload)
    with TestClient(
        create_app(
            "http://gpu",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, stream=stream, headers={"content-type": "text/event-stream"}
                ),
            ),
        )
    ) as client:
        result = client.get("/v1/results/events")
        assert result.content == b"".join(payload)
    assert stream.closed


def test_remote_failure_never_falls_back_to_local_and_console_remains_available():
    def remote(request):
        raise httpx.ConnectError("private remote information")

    with TestClient(
        create_app("http://gpu", transport=httpx.MockTransport(remote))
    ) as client:
        response = client.post("/v1/analysis/start", json={})
        assert response.status_code == 503
        assert "private remote" not in response.text
        assert client.get("/").status_code == 200
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503


def test_cloud_cannot_accept_analysis_before_ready_or_during_backup():
    state = "starting"
    calls = []

    def remote(request):
        calls.append(request.url.host)
        if request.url.host == "controller":
            return httpx.Response(200, json={"state": state})
        return httpx.Response(200, stream=Chunks([b'{"remote":true}']))

    with TestClient(
        create_app(
            "http://gpu",
            controller="http://controller",
            transport=httpx.MockTransport(remote),
        )
    ) as client:
        for state in ("starting", "stopping", "cleanup_required"):  # noqa: B007
            assert client.post("/v1/analysis/start", json={}).status_code == 503
        assert "gpu" not in calls
        state = "ready"
        assert client.post("/v1/analysis/start", json={}).json()["remote"] is True


def test_browser_camera_binary_and_controls_round_trip_without_decoding(monkeypatch):
    import websockets.asyncio.client

    class Remote:
        def __init__(self):
            self.received = []

        async def __aenter__(self):
            self.queue = asyncio.Queue()
            await self.queue.put('{"type":"hello"}')
            return self

        async def __aexit__(self, *args):
            self.closed = True

        async def send(self, message):
            self.received.append(message)
            await self.queue.put(message)

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self.queue.get()

    remote = Remote()
    connect = Mock(return_value=remote)
    monkeypatch.setattr(websockets.asyncio.client, "connect", connect)
    with (
        TestClient(create_app("http://gpu:8080")) as client,
        client.websocket_connect("/ws") as socket,
    ):
        assert socket.receive_json() == {"type": "hello"}
        socket.send_json({"type": "start"})
        assert socket.receive_json() == {"type": "start"}
        socket.send_bytes(b"opaque-jpeg-camera-frame")
        assert socket.receive_bytes() == b"opaque-jpeg-camera-frame"
    assert json.loads(remote.received[0]) == {"type": "start"}
    assert remote.received[1] == b"opaque-jpeg-camera-frame"
    assert remote.closed
    assert connect.call_args.args[0] == "ws://gpu:8080/ws"
