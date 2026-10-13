"""Offline Linux container verification for two real gateway WHEP relays.

Run in the analysis-gateway image with --network none, --read-only,
--tmpfs /tmp:size=64m and --shm-size=192m. No models or external services run.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from itertools import pairwise
from pathlib import Path

import aioice.ice
import httpx
import numpy as np
import uvicorn
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame
from fastapi import FastAPI, Request, WebSocket
from starlette.websockets import WebSocketDisconnect

from goodvision_harness import whep_process
from goodvision_harness.analysis_gateway import create_app
from goodvision_harness.video_relay import HEADER, RelayStreamAdapter


def receive_loopback(options, slot, stop, result):
    """Use actual ICE/DTLS/SRTP with no external STUN or non-loopback network."""
    import aiortc

    aioice.ice.get_host_addresses = lambda **kwargs: ["127.0.0.1"]
    peer = aiortc.RTCPeerConnection
    aiortc.RTCPeerConnection = lambda: peer(RTCConfiguration(iceServers=[]))
    whep_process._worker(options, slot, stop, result)


class Pattern(VideoStreamTrack):
    def __init__(self, value):
        super().__init__()
        self.image = np.full((120, 160, 3), value, dtype=np.uint8)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(self.image, format="bgr24")
        frame.pts, frame.time_base = pts, time_base
        return frame


async def wait_for(predicate, timeout=45):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("two-route video verification timed out")
        await asyncio.sleep(0.05)


async def start_server(app):
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await wait_for(lambda: server.started or task.done(), timeout=10)
        assert server.started, "HTTP server failed to start"
    except BaseException:
        server.should_exit = True
        await task
        listener.close()
        raise
    return f"http://127.0.0.1:{port}", server, task, listener


async def verify():
    assert os.name == "posix", "this verification requires a Linux container"
    assert Path("/proc/self/fd").is_dir()
    shm = os.statvfs("/dev/shm")
    temporary = os.statvfs("/tmp")
    assert shm.f_blocks * shm.f_frsize == 192 * 1024 * 1024
    assert temporary.f_blocks * temporary.f_frsize == 64 * 1024 * 1024
    aioice.ice.get_host_addresses = lambda **kwargs: ["127.0.0.1"]
    receiver_class = whep_process.WhepProcessReceiver
    whep_process.WhepProcessReceiver = lambda options: receiver_class(
        options, worker=receive_loopback,
    )
    peers = {}
    source = web.Application()

    async def offer(request):
        camera = request.match_info["camera"]
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        peers[camera] = peer
        peer.addTrack(Pattern(60 if camera == "camera01" else 180))
        for transceiver in peer.getTransceivers():
            transceiver.setCodecPreferences([
                codec for codec in RTCRtpSender.getCapabilities("video").codecs
                if codec.mimeType == "video/H264"
            ])
        await peer.setRemoteDescription(RTCSessionDescription(
            sdp=await request.text(), type="offer",
        ))
        await peer.setLocalDescription(await peer.createAnswer())
        assert peer.getTransceivers()[0]._codecs[0].mimeType == "video/H264"
        return web.Response(
            status=201, text=peer.localDescription.sdp,
            content_type="application/sdp",
            headers={"Location": f"/session/{camera}"},
        )

    async def remove(request):
        peer = peers.pop(request.match_info["camera"], None)
        if peer is not None:
            await peer.close()
        return web.Response(status=204)

    source.router.add_post("/whep/{camera}", offer)
    source.router.add_delete("/session/{camera}", remove)
    source_runner = web.AppRunner(source)
    await source_runner.setup()
    source_site = web.TCPSite(source_runner, "127.0.0.1", 0)
    await source_site.start()
    source_port = source_site._server.sockets[0].getsockname()[1]
    source_origin = f"webrtc+http://127.0.0.1:{source_port}"
    received = {"camera01": [], "camera02": []}
    adapters = {}
    fake_gpu = FastAPI()

    @fake_gpu.post("/v1/fleet/cameras/{camera}/connect")
    async def connect(camera: str, request: Request):
        payload = await request.json()
        assert payload["relay"] is True
        adapter = RelayStreamAdapter(payload["stream_url"])
        adapters[camera] = adapter
        return {"connected": False, "video_relay": {
            "id": adapter.token, "stream_url": payload["stream_url"],
            "fps": 15, "whep_headers": {},
        }}

    @fake_gpu.websocket("/v1/fleet/cameras/{camera}/video/relay/{token}")
    async def receive(socket: WebSocket, camera: str, token: str):
        adapter = adapters[camera]
        assert token == adapter.token
        await socket.accept()
        await socket.send_json({"type": "ready"})
        try:
            while True:
                packet = await socket.receive_bytes()
                sequence, stamp = HEADER.unpack_from(packet)
                accepted = adapter.push(packet)
                if not accepted:
                    await socket.send_json({"accepted": False})
                    break
                frame = adapter._frame
                assert (frame.index, frame.timestamp_seconds) == (sequence, stamp)
                received[camera].append((sequence, stamp, float(frame.bgr.mean())))
                await socket.send_json({"accepted": True})
        except WebSocketDisconnect:
            pass
        finally:
            adapter.stop()

    @fake_gpu.post("/v1/fleet/cameras/{camera}/disconnect")
    async def disconnect(camera: str):
        adapters[camera].stop()
        return {"connected": False}

    servers = []
    try:
        remote = await start_server(fake_gpu)
        servers.append(remote)
        with tempfile.TemporaryDirectory(prefix="gateway-verification-") as temporary:
            os.environ["HARNESS_GATEWAY_SETTINGS_PATH"] = str(
                Path(temporary) / "settings.json"
            )
            gateway = create_app(remote[0])
            local = await start_server(gateway)
            servers.append(local)
            async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
                for camera in received:
                    response = await client.post(
                        local[0] + f"/v1/fleet/cameras/{camera}/connect",
                        json={"stream_url": source_origin + f"/whep/{camera}"},
                    )
                    assert response.status_code == 200, response.text
                    assert "video_relay" not in response.json()
                await wait_for(
                    lambda: all(len(rows) >= 6 for rows in received.values())
                )
                backing = {}
                for camera, relay in gateway.state.video.relays.items():
                    assert not relay.task.done(), f"{camera} relay stopped"
                    receiver = relay.adapter._whep_receiver
                    assert receiver is not None and receiver.pid is not None
                    assert receiver._slot.capacity == 32 * 1024 * 1024
                    arena = receiver._slot.pixels._wrapper._state[0][0]
                    location = os.readlink(f"/proc/self/fd/{arena.fd}")
                    assert location.startswith("/dev/shm/"), location
                    backing[camera] = location
                assert len(backing) == 2 and len(set(backing.values())) == 2
                for camera, rows in received.items():
                    assert all(
                        b[0] > a[0] and b[1] >= a[1]
                        for a, b in pairwise(rows)
                    )
                    expected = 60 if camera == "camera01" else 180
                    assert all(abs(row[2] - expected) < 15 for row in rows), rows
                healthy = await client.get(local[0] + "/health/live")
                assert healthy.status_code == 200
                assert healthy.json()["local_analysis"] is False
                before = len(received["camera02"])
                response = await client.post(
                    local[0] + "/v1/fleet/cameras/camera01/disconnect", json={},
                )
                assert response.status_code == 200, response.text
                await wait_for(lambda: len(received["camera02"]) >= before + 3)
                assert "camera01" not in gateway.state.video.relays
                assert not gateway.state.video.relays["camera02"].task.done()
                print(json.dumps({
                    "verification": "two real H264/WHEP gateway relays",
                    "frames": {
                        camera: len(rows) for camera, rows in received.items()
                    },
                    "default_slot_bytes": 32 * 1024 * 1024,
                    "slot_backing": backing, "healthy": True,
                    "disconnect_isolation": True,
                }))
    finally:
        for _, server, task, listener in reversed(servers):
            server.should_exit = True
            await asyncio.wait_for(task, timeout=15)
            listener.close()
        await asyncio.gather(*(peer.close() for peer in peers.values()))
        await source_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(verify())

