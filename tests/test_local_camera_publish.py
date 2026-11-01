"""Real camera publish -> WHEP media, with synthetic frames and no models."""

import asyncio

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

from goodvision_harness.local_camera import install_local_camera_routes
from goodvision_harness.stream_adapter import parse_stream_source


class PatternTrack(VideoStreamTrack):
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(np.full((120, 160, 3), 145, np.uint8), "bgr24")
        frame.pts, frame.time_base = pts, time_base
        return frame


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
                await asyncio.sleep(0.15)
                assert not cameras.publications
                assert publication.peer.connectionState == "closed"
                assert publication.drain.done()
                assert (
                    await client.get(response.headers["Location"])
                ).status_code == 404
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
