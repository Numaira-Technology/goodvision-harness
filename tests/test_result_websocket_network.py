"""Real TCP/WebSocket delivery through a primary, peer and analysis gateway."""

import json
import socket
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, WebSocket
from test_branch_result_delivery import observation, runtime_without_workers
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from goodvision_harness.analysis_gateway import create_app as gateway_app
from goodvision_harness.service import create_app


@contextmanager
def running(app):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_config=None,
            log_level=None,
            ws="websockets-sansio",
            ws_per_message_deflate=False,
            timeout_graceful_shutdown=2,
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(5)
        listener.close()
        assert not thread.is_alive(), "subscription prevented server shutdown"


def receive(ws):
    return json.loads(ws.recv(timeout=3))


def test_actual_gateway_and_fleet_connections_resume_and_isolate_cameras(monkeypatch):
    primary, _ = runtime_without_workers()
    peer, _ = runtime_without_workers()
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "1")
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "1")
    monkeypatch.delenv("HARNESS_FLEET_ORIGINS", raising=False)
    with running(create_app(lambda: peer)) as peer_url:
        monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "2")
        monkeypatch.setenv(
            "HARNESS_FLEET_ORIGINS",
            json.dumps(
                [
                    "http://unused-local",
                    peer_url,
                ]
            ),
        )
        with running(create_app(lambda: primary)) as primary_url:
            with running(gateway_app(primary_url)) as gateway_url:
                base = gateway_url.replace("http:", "ws:")
                paths = [
                    ("/v1/results/ws", primary, None),
                    ("/v1/fleet/cameras/camera01/results/ws", primary, "camera01"),
                    ("/v1/fleet/cameras/camera02/results/ws", peer, "camera02"),
                    ("/v1/fleet/cameras/camera02/results/smoking/ws", peer, "camera02"),
                    ("/v1/fleet/cameras/camera02/results/overall/ws", peer, "camera02"),
                ]
                for count, (path, runtime, camera_id) in enumerate(paths, 1):
                    runtime.results.reset()
                    with connect(base + path, proxy=None) as ws:
                        assert receive(ws)["type"] == "hello"
                        result = observation("smoking", count).as_dict()
                        runtime.results.publish(result)
                        delivered = receive(ws)
                        assert delivered["data"]["message_id"] == result["message_id"]
                        assert delivered.get("camera_id") == camera_id
                        assert delivered["data"].get("camera_id") == camera_id
                        token = delivered["id"]
                        # The primary remains responsive with an open subscriber.
                        assert (
                            httpx.get(primary_url + "/health/live").status_code == 200
                        )
                    following = observation("smoking", count + 0.5).as_dict()
                    runtime.results.publish(following)
                    with connect(
                        base + path + "?last_event_id=" + token,
                        proxy=None,
                    ) as ws:
                        assert receive(ws)["type"] == "hello"
                        assert (
                            receive(ws)["data"]["message_id"] == following["message_id"]
                        )


@pytest.mark.parametrize(
    "payload,code",
    [
        ("not json", 1011),
        ("[]", 1011),
        (b"binary", 1011),
        ('{"large":"' + "x" * (2 * 1024 * 1024) + '"}', 1009),
    ],
    ids=["invalid-json", "array", "binary", "too-large"],
)
def test_gateway_rejects_invalid_upstream_and_closes_peer(payload, code):
    upstream = FastAPI()
    disconnected = threading.Event()

    @upstream.websocket("/v1/results/ws")
    async def bad_result(ws: WebSocket):
        await ws.accept()
        if isinstance(payload, bytes):
            await ws.send_bytes(payload)
        else:
            await ws.send_text(payload)
        await ws.receive()
        disconnected.set()

    with running(upstream) as upstream_url:
        with running(gateway_app(upstream_url)) as url:
            with connect(
                url.replace("http:", "ws:") + "/v1/results/ws", proxy=None
            ) as ws:
                with pytest.raises(ConnectionClosed) as error:
                    while True:
                        assert receive(ws)["type"] == "error"
                assert error.value.rcvd.code == code
            assert disconnected.wait(3)


def test_gateway_preserves_resume_header_sanitizes_headers_and_cleans_up():
    upstream = FastAPI()
    seen = []
    disconnected = threading.Event()

    @upstream.websocket("/v1/results/ws")
    async def report(ws: WebSocket):
        await ws.accept()
        seen.append((dict(ws.query_params), dict(ws.headers)))
        await ws.send_json({"type": "hello"})
        await ws.receive()
        disconnected.set()

    with running(upstream) as upstream_url:
        with running(gateway_app(upstream_url)) as url:
            with connect(
                url.replace("http:", "ws:") + "/v1/results/ws?after=3",
                additional_headers={
                    "Last-Event-ID": "epoch:2",
                    "Cookie": "private=1",
                    "X-Fleet-Coordinator": "1",
                },
                proxy=None,
            ) as ws:
                assert receive(ws)["type"] == "hello"
            assert disconnected.wait(3)
    assert seen[0][0] == {"after": "3"}
    assert seen[0][1]["last-event-id"] == "epoch:2"
    assert "cookie" not in seen[0][1]
    assert "x-fleet-coordinator" not in seen[0][1]


def test_gateway_unavailable_upstream_is_explicit_and_has_no_local_fallback():
    # A bound, non-listening port deterministically refuses upstream connections.
    with socket.socket() as closed_peer:
        closed_peer.bind(("127.0.0.1", 0))
        origin = f"http://127.0.0.1:{closed_peer.getsockname()[1]}"
        app = gateway_app(origin)
        with running(app) as url:
            with connect(
                url.replace("http:", "ws:") + "/v1/results/ws", proxy=None
            ) as ws:
                assert receive(ws) == {
                    "type": "error",
                    "message": "result service unavailable",
                }
                with pytest.raises(ConnectionClosed) as error:
                    ws.recv(timeout=3)
                assert error.value.rcvd.code == 1013
        assert not hasattr(app.state, "runtime")


def test_upstream_policy_close_reaches_subscriber():
    upstream = FastAPI()

    @upstream.websocket("/v1/results/ws")
    async def denied(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"type": "error", "message": "invalid cursor"})
        await ws.close(1008)

    with running(upstream) as upstream_url:
        with running(gateway_app(upstream_url)) as url:
            with connect(
                url.replace("http:", "ws:") + "/v1/results/ws", proxy=None
            ) as ws:
                assert receive(ws)["type"] == "error"
                with pytest.raises(ConnectionClosed) as error:
                    ws.recv(timeout=3)
                assert error.value.rcvd.code == 1008
