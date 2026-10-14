"""Internal relay authentication before private camera lookup or side effects."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import uvicorn
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_local_gpu_runtime import rig as rig
from test_video_connection import wait_for

from goodvision_harness.relay_auth import (
    RELAY_AUTH_HEADER,
    relay_authorized,
    relay_headers,
    relay_key,
)
from goodvision_harness.service import create_app
from goodvision_harness.video_relay import HEADER

KEY = "gateway-test-auth-key-0123456789abcdef"
PRIVATE_URL = "webrtc+http://camera.example/whep?access_token=fixture-source-secret"
PRIVATE_AUTH = "Bearer fixture-camera-private-header"


def private_runtime(rig):
    runtime = rig.make()
    runtime.update_deployment_settings({"stream_url": PRIVATE_URL})
    runtime.config = replace(
        runtime.config,
        stream=replace(
            runtime.config.stream, whep_headers={"Authorization": PRIVATE_AUTH},
        ),
    )
    return runtime


def packet(sequence=31, stamp=9.125):
    ok, jpeg = cv2.imencode(".jpg", np.full((48, 64, 3), 120, np.uint8))
    assert ok
    return HEADER.pack(sequence, stamp) + jpeg.tobytes()


@pytest.mark.parametrize("path", [
    "/v1/video/connect", "/v1/fleet/cameras/camera01/connect",
])
@pytest.mark.parametrize("headers", [
    {}, {"X-Fleet-Coordinator": "1"},
    {RELAY_AUTH_HEADER: "wrong-key", "X-Fleet-Coordinator": "1"},
])
def test_untrusted_relay_cannot_read_saved_source_or_allocate_adapter(
    rig, monkeypatch, path, headers,
):
    monkeypatch.setenv("HARNESS_RELAY_KEY", KEY)
    monkeypatch.delenv("HARNESS_RELAY_KEY_FILE", raising=False)
    runtime = private_runtime(rig)
    read_source = Mock(side_effect=AssertionError("private source must not be read"))
    connect = Mock(side_effect=AssertionError("unauthorized connect side effect"))
    with TestClient(create_app(lambda: runtime)) as client:
        monkeypatch.setattr(runtime, "configured_stream_url", read_source)
        monkeypatch.setattr(runtime, "connect_stream", connect)
        response = client.post(path, json={"relay": True}, headers=headers)
        assert response.status_code == 403
        assert PRIVATE_URL not in response.text and PRIVATE_AUTH not in response.text
        assert "video_relay" not in response.text
    read_source.assert_not_called()
    connect.assert_not_called()
    assert runtime._video_connection is None


def test_relay_fails_closed_when_key_is_missing(rig, monkeypatch):
    monkeypatch.delenv("HARNESS_RELAY_KEY", raising=False)
    monkeypatch.delenv("HARNESS_RELAY_KEY_FILE", raising=False)
    runtime = private_runtime(rig)
    with TestClient(create_app(lambda: runtime)) as client:
        response = client.post(
            "/v1/video/connect", json={"relay": True},
            headers={RELAY_AUTH_HEADER: KEY},
        )
        assert response.status_code == 403
        assert runtime._video_connection is None


@pytest.mark.parametrize("path,socket_prefix", [
    ("/v1/video/connect", "/v1/video/relay/"),
    (
        "/v1/fleet/cameras/camera01/connect",
        "/v1/fleet/cameras/camera01/video/relay/",
    ),
])
def test_authenticated_negotiation_requires_key_and_matching_socket_capability(
    rig, monkeypatch, path, socket_prefix,
):
    monkeypatch.setenv("HARNESS_RELAY_KEY", KEY)
    runtime = private_runtime(rig)
    with TestClient(create_app(lambda: runtime)) as client:
        response = client.post(
            path, json={"relay": True}, headers={RELAY_AUTH_HEADER: KEY},
        )
        assert response.status_code == 200
        options = response.json()["video_relay"]
        assert options["stream_url"] == PRIVATE_URL
        assert options["whep_headers"] == {"Authorization": PRIVATE_AUTH}
        adapter = runtime._video_connection._adapter
        for headers, token in [
            ({}, options["id"]),
            ({RELAY_AUTH_HEADER: "wrong-key"}, options["id"]),
            ({RELAY_AUTH_HEADER: KEY}, "wrong-capability"),
        ]:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(socket_prefix + token, headers=headers):
                    pass
            assert adapter.attached is False
            assert adapter._closed is False
        with client.websocket_connect(
            socket_prefix + options["id"], headers={RELAY_AUTH_HEADER: KEY},
        ) as stream:
            assert stream.receive_json() == {"type": "ready"}
            stream.send_bytes(packet())
            assert stream.receive_json() == {"accepted": True}
            wait_for(lambda: runtime._video_connection._latest is not None)
            frame = runtime._video_connection._latest
            assert (frame.index, frame.timestamp_seconds) == (31, 9.125)


def test_auth_file_can_appear_after_gateway_startup_and_explicit_invalid_key_fails(
    rig, monkeypatch, tmp_path,
):
    monkeypatch.delenv("HARNESS_RELAY_KEY", raising=False)
    filename = tmp_path / "internal-key"
    monkeypatch.setenv("HARNESS_RELAY_KEY_FILE", str(filename))
    runtime = private_runtime(rig)
    with TestClient(create_app(lambda: runtime)) as client:
        headers = {RELAY_AUTH_HEADER: KEY}
        denied = client.post("/v1/video/connect", json={"relay": True}, headers=headers)
        assert denied.status_code == 403 and runtime._video_connection is None
        filename.write_text(KEY + "\n", "utf-8")
        allowed = client.post(
            "/v1/video/connect", json={"relay": True}, headers=headers,
        )
        assert allowed.status_code == 200
        monkeypatch.setenv("HARNESS_RELAY_KEY", "too-short")
        denied = client.post("/v1/video/connect", json={"relay": True}, headers=headers)
        assert denied.status_code == 403


def test_internal_headers_replace_spoofed_case_variants_without_exposing_key(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("HARNESS_RELAY_KEY", KEY)
    headers = relay_headers({
        "x-goodvision-relay-key": "attacker",
        "X-GOODVISION-RELAY-KEY": "another-attacker",
        "Content-Type": "application/json",
    })
    names = [name for name in headers if name.lower() == RELAY_AUTH_HEADER.lower()]
    assert len(names) == 1
    assert relay_authorized(headers)
    monkeypatch.delenv("HARNESS_RELAY_KEY")
    filename = tmp_path / "invalid-file"
    filename.write_text("x" * 1024, "utf-8")
    monkeypatch.setenv("HARNESS_RELAY_KEY_FILE", str(filename))
    assert relay_key() is None
    with pytest.raises(HTTPException) as error:
        relay_headers()
    assert error.value.status_code == 503 and KEY not in str(error.value.detail)


def test_ordinary_nonrelay_connect_stays_compatible_without_internal_key(
    rig, monkeypatch,
):
    monkeypatch.delenv("HARNESS_RELAY_KEY", raising=False)
    monkeypatch.delenv("HARNESS_RELAY_KEY_FILE", raising=False)
    runtime = private_runtime(rig)
    connect = Mock(return_value={"connected": True})
    monkeypatch.setattr(runtime, "connect_stream", connect)
    with TestClient(create_app(lambda: runtime)) as client:
        response = client.post("/v1/video/connect", json={})
        assert response.status_code == 200
        assert "video_relay" not in response.text
    connect.assert_called_once_with(None, camera_layout=None)


def test_authenticated_primary_routes_private_http_and_socket_to_second_lane(
    rig, monkeypatch,
):
    monkeypatch.setenv("HARNESS_RELAY_KEY", KEY)
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "2")
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "2")
    worker_runtime = private_runtime(rig)
    worker_app = create_app(lambda: worker_runtime)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(
        worker_app, log_level="warning", lifespan="on",
    ))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True,
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "1")
        monkeypatch.setenv("HARNESS_FLEET_ORIGINS", '["http://self", "' + origin + '"]')
        primary_runtime = private_runtime(rig)
        monkeypatch.setattr(primary_runtime, "tracking_camera_capacity", lambda: 2)
        primary = create_app(lambda: primary_runtime)
        with TestClient(primary) as client:
            denied = client.post(
                "/v1/fleet/cameras/camera02/connect",
                json={"relay": True}, headers={"X-Fleet-Coordinator": "1"},
            )
            assert denied.status_code == 403
            assert worker_runtime._video_connection is None
            response = client.post(
                "/v1/fleet/cameras/camera02/connect", json={"relay": True},
                headers={RELAY_AUTH_HEADER: KEY},
            )
            assert response.status_code == 200, response.text
            options = response.json()["video_relay"]
            assert options["stream_url"] == PRIVATE_URL
            adapter = worker_runtime._video_connection._adapter
            route = "/v1/fleet/cameras/camera02/video/relay/" + options["id"]
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(route):
                    pass
            assert adapter.attached is False
            with client.websocket_connect(
                route, headers={RELAY_AUTH_HEADER: KEY},
            ) as stream:
                assert stream.receive_json() == {"type": "ready"}
                stream.send_bytes(packet(73, 12.25))
                assert stream.receive_json() == {"accepted": True}
                wait_for(lambda: worker_runtime._video_connection._latest is not None)
                frame = worker_runtime._video_connection._latest
                assert (frame.index, frame.timestamp_seconds) == (73, 12.25)
            fleet = primary.state.fleet
            with pytest.raises(HTTPException) as error:
                asyncio.run(fleet.request_json(
                    2, "POST", "/v1/video/connect", {"relay": True}, coordinated=True,
                ))
            assert error.value.status_code == 403
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive()


@pytest.mark.parametrize("value", [["visible", 3], "visible", 42, None])
def test_generic_fleet_json_keeps_non_object_response_compatibility(
    monkeypatch, value,
):
    import httpx
    from test_fleet import make_app

    app = make_app(
        monkeypatch, capacity=2,
        peer_handler=lambda request: httpx.Response(
            200, content=json.dumps(value).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        ),
    )
    with TestClient(app) as client:
        result = client.portal.call(
            app.state.fleet.request_json, 2, "GET", "/v1/plain-json",
        )
        assert result.status_code == 200
        assert json.loads(result.body) == value
