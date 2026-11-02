"""Real camera publish -> WHEP media, with synthetic frames and no models."""

import asyncio
import socket

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

pytest.importorskip("aiortc")
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame

from goodvision_harness.local_camera import (
    CameraPublication,
    LocalCameraPublications,
    install_local_camera_routes,
)
from goodvision_harness.stream_adapter import parse_stream_source


class PatternTrack(VideoStreamTrack):
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(np.full((120, 160, 3), 145, np.uint8), "bgr24")
        frame.pts, frame.time_base = pts, time_base
        return frame


def test_owned_camera_uses_listener_port_behind_docker_or_https_proxy():
    from starlette.requests import Request

    cameras = LocalCameraPublications()
    publication_id = "a" * 32
    resource = f"/v1/local-camera/{publication_id}"
    cameras.publications[publication_id] = CameraPublication(peer=None)
    request = Request({
        "type": "http", "scheme": "https", "path": "/v1/local-camera",
        "server": ("0.0.0.0", 8080),
        "headers": [(b"host", b"camera.example:18080")],
    })
    cameras.bind_origin(publication_id, request)
    source = "webrtc+https://camera.example:18080" + resource + "/whep"
    assert cameras.stream_options(source) == {
        "whep_url": "http://127.0.0.1:8080" + resource + "/whep",
    }
    # A matching path on another source/installation is still an external URL.
    for other in [
        source.replace("camera.example", "external.example"),
        source.replace(":18080", ":18081"),
        source.replace("webrtc+https", "webrtc+http"),
        source.replace(publication_id, "b" * 32),
        source.replace("camera.example", "user:secret@camera.example"),
        source + "?token=other", source + "#other",
        "rtsp://camera.example:18080" + resource + "/whep",
    ]:
        assert cameras.stream_options(other) == {}
    assert LocalCameraPublications().stream_options(source) == {}
    cameras.publications.clear()
    assert cameras.stream_options(source) == {}


def receive_camera_on_loopback(options, slot, stop, result):
    import aioice.ice

    from goodvision_harness.whep_process import _worker

    aioice.ice.get_host_addresses = lambda **_kwargs: ["127.0.0.1"]
    _worker(options, slot, stop, result)


def test_real_camera_frames_reach_spawned_receiver_with_mapped_http_port(monkeypatch):
    import aioice.ice
    import uvicorn

    from goodvision_harness import whep_process
    from goodvision_harness.stream_adapter import NetworkStreamAdapter

    monkeypatch.setattr(aioice.ice, "get_host_addresses", lambda **_: ["127.0.0.1"])
    receiver = whep_process.WhepProcessReceiver
    monkeypatch.setattr(
        whep_process, "WhepProcessReceiver",
        lambda options: receiver(options, worker=receive_camera_on_loopback),
    )

    async def scenario():
        app = FastAPI()
        cameras = install_local_camera_routes(app)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        # Reserve a different, non-listening browser-facing port. This models
        # the unavailable host port inside a container without any proxy.
        external = socket.socket()
        external.bind(("127.0.0.1", 0))
        public_port = external.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="error"))
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        publisher = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        track = PatternTrack()
        adapter = stream = reading = None
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    await asyncio.sleep(0.01)
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", trust_env=False,
                headers={"Host": f"127.0.0.1:{public_port}"},
            ) as client:
                publisher.addTransceiver(track, direction="sendonly")
                await publisher.setLocalDescription(await publisher.createOffer())
                response = await client.post(
                    "/v1/local-camera", content=publisher.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                )
                assert response.status_code == 201, response.text
                resource = response.headers["Location"]
                await publisher.setRemoteDescription(
                    RTCSessionDescription(response.text, "answer"),
                )
                async with asyncio.timeout(5):
                    while not (await client.get(resource)).json()["frames_received"]:
                        await asyncio.sleep(0.01)
                public = f"webrtc+http://127.0.0.1:{public_port}{resource}/whep"
                adapter = NetworkStreamAdapter(public, **cameras.stream_options(public))
                assert adapter.source.source_url == public
                assert adapter.source.transport_url == (
                    f"http://127.0.0.1:{port}{resource}/whep"
                )
                stream = adapter.frames()
                reading = asyncio.create_task(asyncio.to_thread(next, stream, None))
                received = await asyncio.wait_for(asyncio.shield(reading), 15)
                assert received.bgr.shape == (120, 160, 3)
                assert 130 < received.bgr.mean() < 160
        finally:
            if adapter is not None:
                adapter.stop()
            if reading is not None:
                await asyncio.wait_for(asyncio.shield(reading), 5)
            if stream is not None:
                stream.close()
            track.stop()
            await publisher.close()
            await cameras.aclose()
            server.should_exit = True
            await asyncio.wait_for(serving, 5)
            sock.close()
            external.close()

    asyncio.run(scenario())


def test_real_publish_whep_fanout_and_stop():
    async def scenario():
        app = FastAPI()
        cameras = install_local_camera_routes(app)
        publisher = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        viewers = []
        resources = []
        track = PatternTrack()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app),
                base_url="http://testserver",
            ) as client:
                publisher.addTransceiver(track, direction="sendonly")
                await publisher.setLocalDescription(await publisher.createOffer())
                response = await client.post(
                    "/v1/local-camera",
                    content=publisher.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                )
                assert response.status_code == 201, response.text
                resource = response.headers["Location"]
                await publisher.setRemoteDescription(
                    RTCSessionDescription(response.text, "answer"),
                )
                for _ in range(100):
                    status = (await client.get(resource)).json()
                    if status["frames_received"] > 0:
                        break
                    await asyncio.sleep(0.05)
                assert status["state"] == "connected"
                assert status["frames_received"] > 0
                source = parse_stream_source(
                    "webrtc+http://testserver" + resource + "/whep"
                )
                assert source.protocol == "webrtc"
                for _ in range(2):
                    viewer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
                    viewers.append(viewer)
                    ready = asyncio.get_running_loop().create_future()
                    viewer.on(
                        "track", lambda track, ready=ready: ready.set_result(track)
                    )
                    viewer.addTransceiver("video", direction="recvonly")
                    await viewer.setLocalDescription(await viewer.createOffer())
                    response = await client.post(
                        resource + "/whep",
                        content=viewer.localDescription.sdp,
                        headers={"Content-Type": "application/sdp"},
                    )
                    assert response.status_code == 201, response.text
                    resources.append(response.headers["Location"])
                    assert response.headers["Content-Type"].startswith(
                        "application/sdp"
                    )
                    await viewer.setRemoteDescription(
                        RTCSessionDescription(response.text, "answer")
                    )
                    received = await ready
                    frame = await asyncio.wait_for(received.recv(), 5)
                    assert (frame.width, frame.height) == (160, 120)
                    assert 130 < frame.to_ndarray(format="bgr24").mean() < 160
                    if len(viewers) == 2:
                        assert (await client.get(resource)).json()["viewers"] == 2
                assert (await client.delete(resources[0])).status_code == 204
                assert (await client.get(resource)).json()["viewers"] == 1
                remaining = next(iter(cameras.publications.values())).readers.copy()
                # Stopping the publisher also closes any remaining viewer.
                publication = next(iter(cameras.publications.values()))
                assert (await client.delete(resource)).status_code == 204
                assert all(
                    r.peer.connectionState == "closed" for r in remaining.values()
                )
                assert not publication.readers or all(
                    r.track.readyState == "ended" for r in publication.readers.values()
                )
                assert publication.peer.connectionState == "closed"
                assert publication.track.readyState == "ended"
                assert not cameras.publications
                assert (await client.get(resource)).status_code == 404
                assert (await client.delete(resource)).status_code == 204
        finally:
            track.stop()
            await publisher.close()
            await asyncio.gather(*(viewer.close() for viewer in viewers))
            await cameras.aclose()

    asyncio.run(scenario())


def test_invalid_offers_do_not_leave_publications():
    async def scenario():
        app = FastAPI()
        cameras = install_local_camera_routes(app)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://testserver",
        ) as client:
            for body, content_type, expected in [
                ("v=0", "text/plain", 415),
                ("not sdp", "application/sdp", 400),
                ("v=0\r\n" + "x" * 65536, "application/sdp", 413),
                ("v=0\r\n", "application/sdp", 400),
            ]:
                response = await client.post(
                    "/v1/local-camera",
                    content=body,
                    headers={"Content-Type": content_type},
                )
                assert response.status_code == expected, response.text
                assert not cameras.publications
            response = await client.post(
                "/v1/local-camera/missing/whep",
                content="v=0\r\n",
                headers={"Content-Type": "application/sdp"},
            )
            assert response.status_code == 404
            assert (await client.get("/assets/local_camera.js")).status_code == 200

    asyncio.run(scenario())


def test_abandoned_offer_expires_without_leaving_media_tasks():
    async def scenario():
        app = FastAPI()
        cameras = install_local_camera_routes(app)
        cameras.idle_timeout_seconds = 0.05
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        track = PatternTrack()
        baseline = asyncio.all_tasks()
        try:
            peer.addTransceiver(track, direction="sendonly")
            await peer.setLocalDescription(await peer.createOffer())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app),
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/v1/local-camera",
                    content=peer.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                )
                assert response.status_code == 201
                publication = next(iter(cameras.publications.values()))
                # Simulate a tab closing before applying the SDP answer.
                # Expiry closes the peer asynchronously; its closed state can
                # precede completed cleanup on a busy Windows event loop.
                # Shield the task so a timeout cannot make the leak check pass
                # by cancelling the very work whose completion we are testing.
                await asyncio.wait_for(asyncio.shield(publication.drain), 5)
                assert not cameras.publications
                assert publication.peer.connectionState == "closed"
                assert publication.drain.done()
                assert (
                    await client.get(response.headers["Location"])
                ).status_code == 404
                # ICE/connection callbacks can finish just after the drain.
                # wait() leaves unfinished tasks alive, so the assertion below
                # still detects leaks instead of cleaning them up for the test.
                pending = asyncio.all_tasks() - baseline
                if pending:
                    await asyncio.wait(pending, timeout=5)
                assert not (asyncio.all_tasks() - baseline)
        finally:
            track.stop()
            await peer.close()
            await cameras.aclose()

    asyncio.run(scenario())


def test_gateway_keeps_camera_publishing_local_when_analysis_is_unavailable():
    from fastapi.testclient import TestClient

    from goodvision_harness.analysis_gateway import create_app

    def remote(_request):
        raise AssertionError("Camera publishing must not be proxied to analysis")

    app = create_app("http://gpu:8080", transport=httpx.MockTransport(remote))
    with TestClient(app) as client:
        assert client.get("/assets/local_camera.js").status_code == 200
        assert client.post("/v1/local-camera", content="bad").status_code == 415
        assert client.get("/v1/local-camera/expired").status_code == 404
