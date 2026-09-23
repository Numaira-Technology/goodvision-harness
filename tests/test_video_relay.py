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
from goodvision_harness.service import create_app
from goodvision_harness.video_relay import HEADER, RelayStreamAdapter


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
            path, json={"stream_url": "rtsp://private-camera/live", "relay": True}
        )
        assert response.status_code == 200, response.text
        token = response.json()["video_relay"]["id"]
        assert runtime._session is None
        with client.websocket_connect(socket_prefix + token) as socket:
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
