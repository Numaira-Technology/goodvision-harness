"""Bounded source relay: local camera access, GPU-host analysis and state."""

from __future__ import annotations

import asyncio
import json
import math
import struct
import threading
from io import BytesIO
from uuid import uuid4

from .stream_adapter import StreamAdapterError, parse_stream_source
from .video_to_stream import LiveFrame

HEADER = struct.Struct("!Qd")
MAX_PACKET = 16 * 1024 * 1024


class RelayStreamAdapter:
    """A latest-frame input with source timestamps preserved across hosts."""

    realtime = True

    def __init__(self, source_url):
        self.source = parse_stream_source(source_url)
        self.token = uuid4().hex
        self._condition = threading.Condition()
        self._frame = None
        self._last = (-1, -1.0)
        self._closed = False
        self.attached = False

    def push(self, packet):
        import cv2
        import numpy as np
        from PIL import Image

        if not HEADER.size < len(packet) <= MAX_PACKET:
            raise ValueError("invalid relay packet size")
        sequence, stamp = HEADER.unpack_from(packet)
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError("invalid relay timestamp")
        jpeg = packet[HEADER.size :]
        with Image.open(BytesIO(jpeg)) as source:
            if source.format != "JPEG" or source.width * source.height > 32_000_000:
                raise ValueError("invalid relay image")
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("invalid relay image")
        with self._condition:
            if self._closed:
                return False
            if sequence <= self._last[0] or stamp < self._last[1]:
                raise ValueError("relay sources must be increasing")
            self._last = (sequence, stamp)
            self._frame = LiveFrame(sequence, stamp, bgr)
            self._condition.notify_all()
        return True

    def frames(self):
        with self._condition:
            self._condition.wait_for(lambda: self._frame is not None or self._closed)
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._frame is not None or self._closed
                )
                if self._closed:
                    return
                frame, self._frame = self._frame, None
            yield frame

    def stop(self):
        with self._condition:
            self._closed = True
            self._frame = None
            self._condition.notify_all()


async def receive_relay(socket, runtime, token):
    """Only the opaque capability returned by connect can feed this connection."""
    from .relay_auth import relay_authorized

    if not relay_authorized(socket.headers):
        await socket.close(code=1008)
        return
    connection = runtime._video_connection
    adapter = getattr(connection, "_adapter", None)
    if (
        not isinstance(adapter, RelayStreamAdapter)
        or adapter.token != token
        or adapter.attached
    ):
        await socket.close(code=1008)
        return
    adapter.attached = True
    await socket.accept()
    try:
        await socket.send_json({"type": "ready"})
        while True:
            packet = await socket.receive_bytes()
            accepted = await asyncio.to_thread(adapter.push, packet)
            await socket.send_json({"accepted": accepted})
            if not accepted:
                break
    except Exception:
        # Decoder/socket details can contain private camera information.
        pass
    finally:
        adapter.stop()
        try:
            await socket.close()
        except RuntimeError:
            pass


async def proxy_relay(socket, origin, path):
    """Route additional camera sockets to their existing isolated runtime."""
    from websockets.asyncio.client import connect

    from .relay_auth import relay_authorized, relay_headers

    if not relay_authorized(socket.headers):
        await socket.close(code=1008)
        return
    url = ("wss" if origin.startswith("https:") else "ws") + origin[origin.index(":") :]
    try:
        async with connect(
            url + path, compression=None, max_size=MAX_PACKET, proxy=None,
            additional_headers=relay_headers(),
        ) as peer:
            await socket.accept()

            async def upload():
                while True:
                    message = await socket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    await peer.send(
                        message.get("bytes")
                        if message.get("bytes") is not None
                        else message["text"]
                    )

            async def download():
                async for value in peer:
                    if isinstance(value, bytes):
                        await socket.send_bytes(value)
                    else:
                        await socket.send_text(value)

            tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        pass
    finally:
        try:
            await socket.close()
        except RuntimeError:
            pass


class LocalVideoRelay:
    def __init__(self, origin, path, options):
        from .stream_adapter import NetworkStreamAdapter

        self.url = (
            ("wss" if origin.startswith("https:") else "ws")
            + origin[origin.index(":") :]
            + path
        )
        self.adapter = NetworkStreamAdapter(
            options["stream_url"],
            output_fps=options["fps"],
            whep_headers=options["whep_headers"],
        )
        self.task = None
        self.socket = None
        self.first_frame = asyncio.Event()

    async def start(self):
        from websockets.asyncio.client import connect

        from .relay_auth import relay_headers

        self.socket = await connect(
            self.url, compression=None, max_size=MAX_PACKET, proxy=None, open_timeout=5,
            additional_headers=relay_headers(),
        )
        if (
            json.loads(await asyncio.wait_for(self.socket.recv(), timeout=5)).get(
                "type"
            )
            != "ready"
        ):
            await self.socket.close()
            raise RuntimeError("video relay was not accepted")
        self.task = asyncio.create_task(self.run())

    async def run(self):
        from .video_to_stream import encode_jpeg

        frames = self.adapter.frames()
        try:
            while (frame := await asyncio.to_thread(next, frames, None)) is not None:
                jpeg, _, _ = await asyncio.to_thread(
                    encode_jpeg,
                    frame.bgr,
                    max_image_side=None,
                    quality=95,
                )
                packet = HEADER.pack(frame.index, frame.timestamp_seconds) + jpeg
                if len(packet) > MAX_PACKET:
                    raise StreamAdapterError("relay frame exceeds capacity")
                await self.socket.send(packet)
                ack = json.loads(await asyncio.wait_for(self.socket.recv(), timeout=15))
                if ack.get("accepted") is not True:
                    break
                self.first_frame.set()
        except Exception:
            pass
        finally:
            await asyncio.to_thread(self.adapter.stop)
            await self.socket.close()

    async def close(self):
        await asyncio.to_thread(self.adapter.stop)
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.socket is not None:
            await self.socket.close()
