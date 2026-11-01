"""Browser camera publishing and bounded, video-only WHEP fan-out."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response


@dataclass
class CameraReader:
    peer: Any
    track: Any
    expiry: asyncio.Task | None = None


@dataclass
class CameraPublication:
    peer: Any
    track: Any = None
    frames_received: int = 0
    drain: asyncio.Task | None = None
    readers: dict[str, CameraReader] = field(default_factory=dict)


class LocalCameraPublications:
    def __init__(self) -> None:
        self.publications: dict[str, CameraPublication] = {}
        self.idle_timeout_seconds = 30

    def get(self, publication_id: str) -> CameraPublication:
        publication = self.publications.get(publication_id)
        if publication is None:
            raise HTTPException(404, "Camera publication has stopped or expired")
        return publication

    async def publish(self, sdp: str) -> tuple[str, str]:
        try:
            from aiortc import (
                RTCConfiguration,
                RTCPeerConnection,
                RTCSessionDescription,
            )
        except ImportError:
            raise HTTPException(
                503, "Camera publishing support is not installed"
            ) from None
        if len(self.publications) >= 4:
            raise HTTPException(409, "Stop an existing camera publication first")
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        publication = CameraPublication(peer=peer)
        publication_id = uuid4().hex
        self.publications[publication_id] = publication

        @peer.on("track")
        def track_received(track):
            if track.kind == "video" and publication.track is None:
                publication.track = track

        @peer.on("connectionstatechange")
        async def connection_changed():
            if peer.connectionState in {"failed", "closed"}:
                await self.close(publication_id)

        try:
            with anyio.fail_after(15):
                await peer.setRemoteDescription(RTCSessionDescription(sdp, "offer"))
                if publication.track is None or peer.sctp is not None:
                    raise ValueError("missing video track")
                # Audio and data channels are outside this camera utility.
                for transceiver in peer.getTransceivers():
                    transceiver.direction = (
                        "recvonly"
                        if transceiver.receiver.track is publication.track
                        else "inactive"
                    )
                await peer.setLocalDescription(await peer.createAnswer())
            publication.drain = asyncio.create_task(
                self._drain(publication_id, publication),
            )
            return publication_id, peer.localDescription.sdp
        except (Exception, asyncio.CancelledError) as exc:
            await self.close(publication_id)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise HTTPException(400, "Could not negotiate the camera video") from None

    async def _drain(self, publication_id, publication) -> None:
        # One owned receive task, including before the first viewer. Each
        # viewer retains at most one frame and never blocks camera reception.
        try:
            while True:
                frame = await asyncio.wait_for(
                    publication.track.recv(),
                    timeout=self.idle_timeout_seconds,
                )
                publication.frames_received += 1
                for reader in publication.readers.values():
                    reader.track.push(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            await self.close(publication_id)

    async def subscribe(self, publication_id: str, sdp: str) -> tuple[str, str]:
        publication = self.get(publication_id)
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

        if not publication.frames_received:
            raise HTTPException(409, "Camera video is not ready yet")
        if len(publication.readers) >= 8:
            raise HTTPException(409, "Camera viewer limit reached")
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        track = _viewer_track()
        reader_id = uuid4().hex
        reader = CameraReader(peer, track)
        publication.readers[reader_id] = reader

        @peer.on("connectionstatechange")
        async def connection_changed():
            if peer.connectionState in {"failed", "closed"}:
                await self.close_reader(publication_id, reader_id)

        async def expire_unconnected():
            await asyncio.sleep(30)
            if peer.connectionState != "connected":
                await self.close_reader(publication_id, reader_id)

        try:
            with anyio.fail_after(15):
                await peer.setRemoteDescription(RTCSessionDescription(sdp, "offer"))
                video = [t for t in peer.getTransceivers() if t.kind == "video"]
                if not video or peer.sctp is not None:
                    raise ValueError("missing video receiver")
                peer.addTrack(track)
                for transceiver in peer.getTransceivers():
                    transceiver.direction = (
                        "sendonly" if transceiver.sender.track is track else "inactive"
                    )
                await peer.setLocalDescription(await peer.createAnswer())
                if not any(
                    t.currentDirection in {"sendonly", "sendrecv"} for t in video
                ):
                    raise ValueError("missing video receiver")
            # The timeout task belongs to this reader and is cancelled on close.
            reader.expiry = asyncio.create_task(expire_unconnected())
            return reader_id, peer.localDescription.sdp
        except (Exception, asyncio.CancelledError) as exc:
            await self.close_reader(publication_id, reader_id)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise HTTPException(400, "Could not negotiate the camera viewer") from None

    async def close_reader(self, publication_id: str, reader_id: str) -> None:
        publication = self.publications.get(publication_id)
        reader = publication.readers.pop(reader_id, None) if publication else None
        if reader is not None:
            expiry = reader.expiry
            if expiry is not None and expiry is not asyncio.current_task():
                expiry.cancel()
                await asyncio.gather(expiry, return_exceptions=True)
            reader.track.stop()
            await reader.peer.close()

    async def close(self, publication_id: str) -> None:
        publication = self.publications.pop(publication_id, None)
        if publication is None:
            return
        expiry_tasks = []
        for reader in publication.readers.values():
            if reader.expiry is not None:
                reader.expiry.cancel()
                expiry_tasks.append(reader.expiry)
            reader.track.stop()
        await asyncio.gather(
            *expiry_tasks,
            *(reader.peer.close() for reader in publication.readers.values()),
            return_exceptions=True,
        )
        if publication.track is not None:
            publication.track.stop()
        if (
            publication.drain is not None
            and publication.drain is not asyncio.current_task()
        ):
            publication.drain.cancel()
            await asyncio.gather(publication.drain, return_exceptions=True)
        await publication.peer.close()

    async def aclose(self) -> None:
        await asyncio.gather(*(self.close(key) for key in list(self.publications)))


def _viewer_track():
    # Keep WebRTC optional for installations that only use RTSP/RTMP.
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError

    class LatestVideoTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self):
            super().__init__()
            self.frames = asyncio.Queue(maxsize=1)

        def push(self, frame):
            if self.frames.full():
                self.frames.get_nowait()
            self.frames.put_nowait(frame)

        async def recv(self):
            if self.readyState != "live":
                raise MediaStreamError
            frame = await self.frames.get()
            if frame is None:
                raise MediaStreamError
            return frame

        def stop(self):
            super().stop()
            self.push(None)

    return LatestVideoTrack()


async def _offer(request: Request) -> str:
    if request.headers.get("content-type", "").split(";")[0] != "application/sdp":
        raise HTTPException(415, "Expected an SDP video offer")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 65536:
            raise HTTPException(413, "Camera offer is too large")
    try:
        sdp = body.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "Invalid camera offer") from None
    if not sdp.startswith("v=0"):
        raise HTTPException(400, "Invalid camera offer")
    return sdp


def install_local_camera_routes(app: FastAPI) -> LocalCameraPublications:
    cameras = LocalCameraPublications()
    app.state.local_cameras = cameras

    @app.get("/assets/local_camera.js", include_in_schema=False)
    async def camera_script():
        return Response(
            (Path(__file__).parent / "assets" / "local_camera.js").read_bytes(),
            media_type="application/javascript",
        )

    @app.post("/v1/local-camera", status_code=201)
    async def publish(request: Request):
        publication_id, sdp = await cameras.publish(await _offer(request))
        path = f"/v1/local-camera/{publication_id}"
        return Response(
            sdp,
            status_code=201,
            media_type="application/sdp",
            headers={
                "Location": path,
                "Cache-Control": "no-store",
            },
        )

    @app.get("/v1/local-camera/{publication_id}")
    async def status(publication_id: str):
        publication = cameras.get(publication_id)
        return {
            "state": publication.peer.connectionState,
            "frames_received": publication.frames_received,
            "viewers": len(publication.readers),
        }

    @app.delete("/v1/local-camera/{publication_id}", status_code=204)
    async def stop(publication_id: str):
        await cameras.close(publication_id)
        return Response(status_code=204)

    @app.post("/v1/local-camera/{publication_id}/whep", status_code=201)
    async def whep(request: Request, publication_id: str):
        reader_id, sdp = await cameras.subscribe(publication_id, await _offer(request))
        return Response(
            sdp,
            status_code=201,
            media_type="application/sdp",
            headers={
                "Location": f"/v1/local-camera/{publication_id}/viewers/{reader_id}",
                "Cache-Control": "no-store",
            },
        )

    @app.delete(
        "/v1/local-camera/{publication_id}/viewers/{reader_id}", status_code=204
    )
    async def stop_viewer(publication_id: str, reader_id: str):
        await cameras.close_reader(publication_id, reader_id)
        return Response(status_code=204)

    return cameras
