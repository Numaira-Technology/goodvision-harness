"""Local RTSP/WHEP ingress with remote analysis and explicit GPU-pull opt-in."""

import json
from unittest.mock import AsyncMock, Mock

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_local_gpu_runtime import rig as rig
from test_video_connection import wait_for

from goodvision_harness.analysis_gateway import create_app as gateway_app
from goodvision_harness.relay_auth import RELAY_AUTH_HEADER
from goodvision_harness.service import create_app
from goodvision_harness.video_relay import HEADER, RelayStreamAdapter

TEST_RELAY_KEY = "gateway-test-auth-key-0123456789abcdef"


@pytest.fixture(autouse=True)
def configured_relay_key(monkeypatch):
    monkeypatch.setenv("HARNESS_RELAY_KEY", TEST_RELAY_KEY)


def packet(sequence=10, stamp=3.25):
    ok, jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), 120, np.uint8))
    assert ok
    return HEADER.pack(sequence, stamp) + jpeg.tobytes()


def test_receiver_preserves_source_identity_and_bounds_pending_frames():
    adapter = RelayStreamAdapter("rtsp://private-camera/live")
    adapter.push(packet(10, 3.25))
    adapter.push(packet(11, 3.3))
    frames = adapter.frames()
    frame = next(frames)
    assert (frame.index, frame.timestamp_seconds, frame.bgr.shape) == (
        11,
        3.3,
        (48, 64, 3),
    )
    with pytest.raises(ValueError, match="increasing"):
        adapter.push(packet(11, 3.4))
    with pytest.raises(ValueError, match="timestamp"):
        adapter.push(packet(12, float("nan")))
    adapter.stop()
    assert next(frames, None) is None
    assert adapter.push(packet(12, 4)) is False


@pytest.mark.parametrize(
    "path,socket_prefix",
    [
        ("/v1/video/connect", "/v1/video/relay/"),
        (
            "/v1/fleet/cameras/camera01/connect",
            "/v1/fleet/cameras/camera01/video/relay/",
        ),
    ],
)
def test_gpu_host_never_opens_camera_when_receiving_local_relay(
    rig, monkeypatch, path, socket_prefix
):
    runtime = rig.make()
    network = Mock(side_effect=AssertionError("GPU host must not pull local camera"))
    monkeypatch.setattr(
        "goodvision_harness.video_connection.NetworkStreamAdapter", network
    )
    with TestClient(create_app(lambda: runtime)) as client:
        response = client.post(
            path,
            json={"stream_url": "rtsp://private-camera/live", "relay": True},
            headers={RELAY_AUTH_HEADER: TEST_RELAY_KEY},
        )
        assert response.status_code == 200, response.text
        token = response.json()["video_relay"]["id"]
        assert runtime._session is None
        with client.websocket_connect(
            socket_prefix + token,
            headers={RELAY_AUTH_HEADER: TEST_RELAY_KEY},
        ) as socket:
            assert socket.receive_json() == {"type": "ready"}
            socket.send_bytes(packet())
            assert socket.receive_json() == {"accepted": True}
            wait_for(lambda: runtime._video_connection.status()["connected"])
            assert runtime._video_connection._latest.index == 10
            assert runtime._video_connection._latest.timestamp_seconds == 3.25
        wait_for(lambda: not runtime._video_connection.running)
        assert runtime._video_connection.status()["connection_state"] == "error"
    network.assert_not_called()


def test_gateway_defaults_to_local_pull_and_hides_private_relay_metadata(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("HARNESS_VIDEO_PULL_LOCATION", raising=False)
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "settings.json"))
    relay = Mock(start=AsyncMock(), close=AsyncMock())
    factory = Mock(return_value=relay)
    monkeypatch.setattr("goodvision_harness.video_relay.LocalVideoRelay", factory)
    seen = []

    async def remote(request):
        body = await request.aread()
        assert int(request.headers["content-length"]) == len(body)
        seen.append((request.url.path, json.loads(body)))
        return httpx.Response(
            200,
            json={
                "connected": False,
                "video_relay": {
                    "id": "opaque",
                    "stream_url": "rtsp://private/live",
                    "fps": 30,
                    "whep_headers": {"Authorization": "camera-secret"},
                },
            },
        )

    with TestClient(
        gateway_app("http://gpu", transport=httpx.MockTransport(remote))
    ) as client:
        assert client.get("/gateway/video-source").json()["pull_location"] == "local"
        result = client.post(
            "/v1/fleet/cameras/camera02/connect",
            json={"stream_url": "rtsp://private/live"},
        )
        assert result.status_code == 200
        assert "video_relay" not in result.json()
        assert "camera-secret" not in result.text
        assert seen[0][1]["relay"] is True
        assert (
            factory.call_args.args[1] == "/v1/fleet/cameras/camera02/video/relay/opaque"
        )
        relay.start.assert_awaited_once()
    relay.close.assert_awaited_once()


def test_gpu_pull_is_optional_persistent_and_does_not_create_local_relay(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "settings.json"))
    factory = Mock(side_effect=AssertionError("GPU mode must not pull locally"))
    monkeypatch.setattr("goodvision_harness.video_relay.LocalVideoRelay", factory)
    bodies = []

    async def remote(request):
        bodies.append(json.loads(await request.aread()))
        return httpx.Response(200, stream=BytesStream(b'{"connected":true}'))

    with TestClient(
        gateway_app("http://gpu", transport=httpx.MockTransport(remote))
    ) as client:
        saved = client.put("/gateway/video-source", json={"pull_location": "gpu"})
        assert saved.json()["applies_to"] == "next_connection"
        assert (
            client.post(
                "/v1/video/connect", json={"stream_url": "rtsp://camera/live"}
            ).status_code
            == 200
        )
        assert "relay" not in bodies[0]
        assert (
            client.put(
                "/gateway/video-source", json={"pull_location": "invalid"}
            ).status_code
            == 400
        )
    assert gateway_app("http://gpu").state.video.location == "gpu"
    factory.assert_not_called()


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, value):
        self.value = value

    async def __aiter__(self):
        yield self.value


@pytest.mark.parametrize(
    "path",
    [
        "/v1/video/connect",
        "/v1/fleet/cameras/camera01/connect",
    ],
)
@pytest.mark.parametrize(
    "fault",
    ["missing", "not-object", "missing-id", "invalid-json"],
)
def test_successful_remote_connect_rolls_back_failed_negotiation_and_can_retry(
    rig,
    monkeypatch,
    tmp_path,
    path,
    fault,
):
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "local")
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "gateway.json"))
    runtime = rig.make()
    relay = Mock(start=AsyncMock(), close=AsyncMock())
    factory = Mock(return_value=relay)
    monkeypatch.setattr("goodvision_harness.video_relay.LocalVideoRelay", factory)
    calls, accepted = [], []
    disconnect = path.removesuffix("connect") + "disconnect"
    with TestClient(create_app(lambda: runtime)) as remote_client:

        async def remote(request):
            calls.append(request.url.path)
            response = remote_client.request(
                request.method,
                request.url.path,
                content=await request.aread(),
                headers={
                    "content-type": "application/json",
                    **{
                        key: value
                        for key, value in request.headers.items()
                        if key.lower() == RELAY_AUTH_HEADER.lower()
                    },
                },
            )
            if request.url.path == path and response.status_code == 200:
                accepted.append(runtime._video_connection)
                assert accepted[-1].running
                if len(accepted) == 1:
                    if fault == "invalid-json":
                        return httpx.Response(200, content=b"not-json")
                    result = response.json()
                    result.pop("video_relay")
                    if fault == "not-object":
                        result["video_relay"] = []
                    elif fault == "missing-id":
                        result["video_relay"] = {}
                    return httpx.Response(200, json=result)
            return httpx.Response(response.status_code, content=response.content)

        with TestClient(
            gateway_app(
                "http://gpu",
                transport=httpx.MockTransport(remote),
            )
        ) as client:
            response = client.post(path, json={"stream_url": "rtsp://private/live"})
            assert response.status_code == 503
            assert calls == [path, disconnect]
            wait_for(lambda: not accepted[0].running)
            assert not accepted[0].running
            assert not client.app.state.video.relays
            response = client.post(path, json={"stream_url": "rtsp://private/live"})
            assert response.status_code == 200, response.text
            assert len(accepted) == 2
            assert accepted[1] is not accepted[0]
            assert accepted[1].running
            factory.assert_called_once()
            relay.start.assert_awaited_once()
        relay.close.assert_awaited_once()


def test_failed_local_close_still_disconnects_remote_and_preserves_start_error(
    rig,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "local")
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "gateway.json"))
    runtime = rig.make()
    failed = Mock(
        start=AsyncMock(side_effect=RuntimeError("private startup details")),
        close=AsyncMock(side_effect=OSError("private local close details")),
    )
    good = Mock(start=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(
        "goodvision_harness.video_relay.LocalVideoRelay",
        Mock(side_effect=[failed, good]),
    )
    calls = []
    with TestClient(create_app(lambda: runtime)) as remote_client:

        async def remote(request):
            calls.append(request.url.path)
            response = remote_client.request(
                request.method,
                request.url.path,
                content=await request.aread(),
                headers={
                    "content-type": "application/json",
                    **{
                        key: value
                        for key, value in request.headers.items()
                        if key.lower() == RELAY_AUTH_HEADER.lower()
                    },
                },
            )
            return httpx.Response(response.status_code, content=response.content)

        with TestClient(
            gateway_app(
                "http://gpu",
                transport=httpx.MockTransport(remote),
            )
        ) as client:
            response = client.post(
                "/v1/video/connect", json={"stream_url": "rtsp://private/live"}
            )
            assert response.status_code == 503
            assert (
                response.json()["detail"]
                == "Local video relay could not connect to GPU host"
            )
            assert "private" not in response.text
            failed.close.assert_awaited_once()
            assert calls == ["/v1/video/connect", "/v1/video/disconnect"]
            wait_for(lambda: not runtime._video_connection.running)
            assert not runtime._video_connection.running
            assert (
                client.post(
                    "/v1/video/connect",
                    json={"stream_url": "rtsp://private/live"},
                ).status_code
                == 200
            )
        good.close.assert_awaited_once()


@pytest.mark.parametrize("disconnect_error", ["http-error", "lost-reply"])
def test_rollback_error_keeps_original_negotiation_error_and_allows_retry_after_close(
    rig,
    monkeypatch,
    tmp_path,
    disconnect_error,
):
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "local")
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "gateway.json"))
    runtime = rig.make()
    relay = Mock(start=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(
        "goodvision_harness.video_relay.LocalVideoRelay", Mock(return_value=relay)
    )
    accepted = 0
    with TestClient(create_app(lambda: runtime)) as remote_client:

        async def remote(request):
            nonlocal accepted
            response = remote_client.request(
                request.method,
                request.url.path,
                content=await request.aread(),
                headers={
                    "content-type": "application/json",
                    **{
                        key: value
                        for key, value in request.headers.items()
                        if key.lower() == RELAY_AUTH_HEADER.lower()
                    },
                },
            )
            if request.url.path.endswith("connect") and response.status_code == 200:
                accepted += 1
                if accepted == 1:
                    result = response.json()
                    result.pop("video_relay")
                    return httpx.Response(200, json=result)
            if request.url.path.endswith("disconnect"):
                wait_for(lambda: not runtime._video_connection.running)
                assert not runtime._video_connection.running
                if disconnect_error == "lost-reply":
                    raise httpx.ReadError("private rollback details", request=request)
                return httpx.Response(500, json={"detail": "private rollback details"})
            return httpx.Response(response.status_code, content=response.content)

        with TestClient(
            gateway_app(
                "http://gpu",
                transport=httpx.MockTransport(remote),
            )
        ) as client:
            response = client.post(
                "/v1/video/connect", json={"stream_url": "rtsp://private/live"}
            )
            assert response.status_code == 503
            assert "does not support local video ingress" in response.json()["detail"]
            assert "private" not in response.text
            wait_for(lambda: not runtime._video_connection.running)
            assert not runtime._video_connection.running
            assert (
                client.post(
                    "/v1/video/connect",
                    json={"stream_url": "rtsp://private/live"},
                ).status_code
                == 200
            )


def test_remote_connect_rejection_never_disconnects_existing_connection(
    rig,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HARNESS_VIDEO_PULL_LOCATION", "local")
    monkeypatch.setenv("HARNESS_GATEWAY_SETTINGS_PATH", str(tmp_path / "gateway.json"))
    runtime = rig.make()
    calls = []
    with TestClient(
        create_app(lambda: runtime), headers={RELAY_AUTH_HEADER: TEST_RELAY_KEY},
    ) as remote_client:
        assert (
            remote_client.post(
                "/v1/video/connect",
                json={
                    "stream_url": "rtsp://private/live",
                    "relay": True,
                },
            ).status_code
            == 200
        )
        original = runtime._video_connection

        async def remote(request):
            calls.append(request.url.path)
            response = remote_client.request(
                request.method,
                request.url.path,
                content=await request.aread(),
                headers={
                    "content-type": "application/json",
                    **{
                        key: value
                        for key, value in request.headers.items()
                        if key.lower() == RELAY_AUTH_HEADER.lower()
                    },
                },
            )
            return httpx.Response(response.status_code, content=response.content)

        with TestClient(
            gateway_app(
                "http://gpu",
                transport=httpx.MockTransport(remote),
            )
        ) as client:
            assert (
                client.post(
                    "/v1/video/connect",
                    json={
                        "stream_url": "rtsp://private/live",
                    },
                ).status_code
                == 409
            )
            assert calls == ["/v1/video/connect"]
            assert runtime._video_connection is original
            assert original.running
