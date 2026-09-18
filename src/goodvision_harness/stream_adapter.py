"""Network video adapters for RTSP, RTMP, and WebRTC/WHEP sources.

RTSP and RTMP use the OpenCV video backend. WebRTC uses a WHEP endpoint,
written as webrtc+https:// (or webrtc+http:// for local development). The
adapter emits the same LiveFrame objects as VideoFileStream, keeping the rest
of the sampling, windowing, inference, and display pipeline unchanged.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

import cv2

from .privacy import PUBLIC_STREAM_ERROR, PUBLIC_STREAM_SUPPORT_ERROR
from .stream_diagnostics import StreamDiagnostics
from .video_to_stream import LiveFrame

StreamProtocol = Literal["rtsp", "rtmp", "webrtc"]
_END = object()


class StreamAdapterError(RuntimeError):
    """A network stream could not be opened, decoded, or negotiated."""

    def __init__(self, message: str, *, code: str = "connection_failed") -> None:
        super().__init__(message)
        self.code = (
            "missing_support" if code == "missing_support" else "connection_failed"
        )

    @property
    def public_message(self) -> str:
        return (
            PUBLIC_STREAM_SUPPORT_ERROR
            if self.code == "missing_support" else PUBLIC_STREAM_ERROR
        )


@dataclass(frozen=True, slots=True)
class StreamSource:
    protocol: StreamProtocol
    source_url: str
    transport_url: str
    host: str


def parse_stream_source(value: str) -> StreamSource:
    """Validate a supported source URL without performing any I/O."""

    source_url = str(value).strip()
    if not source_url:
        raise ValueError("stream URL must not be empty")
    parsed = urlsplit(source_url)
    scheme = parsed.scheme.lower()
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("stream URL must include a host")

    if scheme in {"rtsp", "rtsps"}:
        protocol: StreamProtocol = "rtsp"
        transport_url = source_url
    elif scheme in {"rtmp", "rtmps"}:
        protocol = "rtmp"
        transport_url = source_url
    elif scheme in {"webrtc+http", "webrtc+https"}:
        protocol = "webrtc"
        transport_url = urlunsplit(
            parsed._replace(scheme=scheme.removeprefix("webrtc+"))
        )
    else:
        raise ValueError(
            "unsupported stream URL; use rtsp://, rtmp://, or "
            "webrtc+https:// pointing to a WHEP endpoint"
        )

    return StreamSource(
        protocol=protocol,
        source_url=source_url,
        transport_url=transport_url,
        host=parsed.hostname,
    )


@dataclass(frozen=True, slots=True)
class _ProducerError:
    error: BaseException


class NetworkStreamAdapter:
    """Adapt a supported network stream to the VideoFileStream interface."""

    realtime = True

    def __init__(
        self,
        source_url: str,
        *,
        output_fps: float = 15.0,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 1.0,
        whep_headers: Mapping[str, str] | None = None,
        connection_id: str | None = None,
    ) -> None:
        if output_fps <= 0:
            raise ValueError("output_fps must be positive")
        if reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be non-negative")
        if reconnect_delay_seconds < 0:
            raise ValueError("reconnect_delay_seconds must be non-negative")
        self.source = parse_stream_source(source_url)
        self.output_fps = float(output_fps)
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.whep_headers = dict(whep_headers or {})
        self.dropped_frames = 0
        self._diagnostics = StreamDiagnostics(connection_id or uuid.uuid4().hex,
                                              self.source.protocol, self.output_fps)
        self._stop = threading.Event()
        self._capture_lock = threading.Lock()
        self._capture: Any | None = None
        self._producer: threading.Thread | None = None
        self._queue: queue.Queue[LiveFrame | _ProducerError | object] = queue.Queue(
            maxsize=2
        )

    @property
    def protocol(self) -> StreamProtocol:
        return self.source.protocol

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def frames(self) -> Iterator[LiveFrame]:
        try:
            if self.protocol in {"rtsp", "rtmp"}:
                yield from self._opencv_frames()
            else:
                yield from self._whep_frames()
        finally:
            self._diagnostics.emit(final=True)

    def _opencv_frames(self) -> Iterator[LiveFrame]:
        failures = 0
        emitted = 0
        origin: float | None = None
        last_timestamp = 0.0
        period = 1.0 / self.output_fps
        next_emit = 0.0

        while not self.stopped:
            capture = cv2.VideoCapture(self.source.transport_url)
            with self._capture_lock:
                self._capture = capture
            try:
                if not capture.isOpened():
                    failures += 1
                else:
                    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    while not self.stopped:
                        read_started = time.monotonic()
                        read_ok, image = capture.read()
                        if self.stopped:
                            break
                        if not read_ok or image is None:
                            self._diagnostics.count('read_failures')
                            break
                        failures = 0
                        now = time.monotonic()
                        self._diagnostics.received(image, now, now-read_started,
                                                   emitted if now+1e-9 >= next_emit else None)
                        if now + 1e-9 < next_emit:
                            self.dropped_frames += 1
                            self._diagnostics.count('rate_limit_skipped_frames')
                            self._diagnostics.emit()
                            continue
                        next_emit = now + period
                        if origin is None:
                            origin = now
                        timestamp = max(last_timestamp, now - origin)
                        last_timestamp = timestamp
                        self._diagnostics.enqueued(emitted)
                        self._diagnostics.dequeued(emitted)
                        self._diagnostics.emit()
                        yield LiveFrame(
                            index=emitted,
                            timestamp_seconds=timestamp,
                            bgr=image,
                        )
                        emitted += 1
                    if not self.stopped:
                        failures += 1

                if failures > self.reconnect_attempts:
                    action = "open" if emitted == 0 else "continue"
                    raise StreamAdapterError(
                        f"could not {action} {self.protocol.upper()} stream "
                        f"from host {self.source.host}"
                    )
            finally:
                # Keep all native VideoCapture operations on this owner thread;
                # cross-thread read()/release() races can crash OpenCV/FFmpeg.
                capture.release()
                with self._capture_lock:
                    if self._capture is capture:
                        self._capture = None

            if self._stop.wait(self.reconnect_delay_seconds):
                return

    def _whep_frames(self) -> Iterator[LiveFrame]:
        if self._producer is not None:
            raise RuntimeError("WebRTC stream has already been consumed")
        self._producer = threading.Thread(
            target=self._whep_thread_main,
            name="whep-video-receiver",
            daemon=True,
        )
        self._producer.start()
        try:
            while True:
                item = self._queue.get()
                if item is _END:
                    return
                if isinstance(item, _ProducerError):
                    if self.stopped:
                        return
                    error = item.error
                    if isinstance(error, StreamAdapterError):
                        raise error
                    raise StreamAdapterError(
                        f"WebRTC stream from host {self.source.host} failed"
                    ) from error
                assert isinstance(item, LiveFrame)
                self._diagnostics.dequeued(item.index)
                yield item
        finally:
            self.stop()
            self._producer.join(timeout=2.0)

    def _whep_thread_main(self) -> None:
        try:
            asyncio.run(self._receive_whep())
        except BaseException as exc:
            self._put_control(_ProducerError(exc))
        finally:
            self._put_control(_END)

    async def _receive_whep(self) -> None:
        try:
            import aiohttp
            from aiortc import RTCPeerConnection, RTCSessionDescription
            from aiortc.mediastreams import MediaStreamError
        except ImportError as exc:
            raise StreamAdapterError(
                "WebRTC support requires the streaming-harness optional dependencies",
                code="missing_support",
            ) from exc

        pc = RTCPeerConnection()
        track_ready: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        session_url: str | None = None
        stats_task: asyncio.Task | None = None

        @pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "video" and not track_ready.done():
                track_ready.set_result(track)

        pc.addTransceiver("video", direction="recvonly")
        try:
            timeout = aiohttp.ClientTimeout(total=20.0)
            async with aiohttp.ClientSession(
                headers=self.whep_headers, timeout=timeout
            ) as http:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                local = pc.localDescription
                if local is None:
                    raise StreamAdapterError("WebRTC offer did not produce local SDP")

                async with http.post(
                    self.source.transport_url,
                    data=local.sdp,
                    headers={
                        "Accept": "application/sdp",
                        "Content-Type": "application/sdp",
                    },
                    allow_redirects=True,
                ) as response:
                    answer_sdp = await response.text()
                    if response.status == 406:
                        raise StreamAdapterError(
                            "WHEP server counter-offers are not supported"
                        )
                    if response.status != 201:
                        raise StreamAdapterError(
                            f"WHEP endpoint returned HTTP {response.status}"
                        )
                    if (
                        not response.headers.get("Content-Type", "")
                        .lower()
                        .startswith("application/sdp")
                    ):
                        raise StreamAdapterError(
                            "WHEP endpoint did not return an SDP answer"
                        )
                    location = response.headers.get("Location")
                    if not location:
                        raise StreamAdapterError(
                            "WHEP endpoint did not return a session Location"
                        )
                    session_url = urljoin(str(response.url), location)

                try:
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=answer_sdp, type="answer")
                    )
                    self._diagnostics.configure_sdp(answer_sdp)
                    stats_task = asyncio.create_task(self._poll_receive_diagnostics(pc))
                    try:
                        track = await asyncio.wait_for(track_ready, timeout=15.0)
                    except asyncio.TimeoutError as exc:
                        raise StreamAdapterError(
                            "WHEP session did not deliver a video track"
                        ) from exc

                    await self._pump_track(track, pc, MediaStreamError)
                finally:
                    if session_url is not None:
                        try:
                            async with http.delete(session_url):
                                pass
                        except aiohttp.ClientError:
                            pass
        finally:
            if stats_task is not None:
                stats_task.cancel()
                await asyncio.gather(stats_task, return_exceptions=True)
                await self._sample_receive_diagnostics(pc, 0.0)
            await pc.close()

    async def _sample_receive_diagnostics(self, pc: Any, loop_lag: float) -> None:
        try:
            report = await asyncio.wait_for(pc.getStats(), timeout=1.0)
            self._diagnostics.webrtc_stats(report, pc, loop_lag)
        except Exception:
            # Missing statistics cannot interrupt receiving video.
            self._diagnostics.count('stats_errors')
        self._diagnostics.emit()

    async def _poll_receive_diagnostics(self, pc: Any) -> None:
        while not self.stopped:
            deadline = time.monotonic() + 5.0
            await asyncio.sleep(5.0)
            await self._sample_receive_diagnostics(pc, time.monotonic()-deadline)

    async def _pump_track(
        self, track: Any, pc: Any, media_stream_error: type[BaseException]
    ) -> None:
        emitted = 0
        origin: float | None = None
        last_timestamp = 0.0
        period = 1.0 / self.output_fps
        next_emit = 0.0
        receive_started = time.monotonic()
        receive_task: asyncio.Task[Any] | None = asyncio.create_task(track.recv())
        try:
            while not self.stopped:
                done, _ = await asyncio.wait({receive_task}, timeout=0.5)
                if not done:
                    self._diagnostics.count('receive_wait_over_500ms')
                    if pc.connectionState in {"closed", "failed"}:
                        raise StreamAdapterError("WebRTC connection closed")
                    continue
                try:
                    frame = receive_task.result()
                except media_stream_error:
                    self._diagnostics.count('track_ended')
                    return
                now = time.monotonic()
                self._diagnostics.received(frame, now, now-receive_started,
                                           emitted if now+1e-9 >= next_emit else None)
                receive_started = now
                receive_task = asyncio.create_task(track.recv())
                if now + 1e-9 < next_emit:
                    self.dropped_frames += 1
                    self._diagnostics.count('rate_limit_skipped_frames')
                    continue
                next_emit = now + period
                if origin is None:
                    origin = now
                timestamp = max(last_timestamp, now - origin)
                last_timestamp = timestamp
                conversion_started = time.monotonic()
                try:
                    bgr = frame.to_ndarray(format="bgr24")
                except Exception:
                    self._diagnostics.count('color_conversion_errors')
                    raise
                self._diagnostics.timing('color_conversion', time.monotonic()-conversion_started)
                self._put_latest(
                    LiveFrame(
                        index=emitted,
                        timestamp_seconds=timestamp,
                        bgr=bgr,
                    )
                )
                emitted += 1
        finally:
            if receive_task is not None and not receive_task.done():
                receive_task.cancel()
                await asyncio.gather(receive_task, return_exceptions=True)

    def _put_latest(self, frame: LiveFrame) -> None:
        self._diagnostics.enqueued(frame.index)
        while True:
            try:
                self._queue.put_nowait(frame)
                return
            except queue.Full:
                try:
                    dropped = self._queue.get_nowait()
                    self.dropped_frames += 1
                    if isinstance(dropped, LiveFrame):
                        self._diagnostics.dequeued(dropped.index, dropped=True)
                except queue.Empty:
                    continue

    def _put_control(self, item: _ProducerError | object) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    dropped = self._queue.get_nowait()
                    if isinstance(dropped, LiveFrame):
                        self._diagnostics.dequeued(dropped.index, dropped=True)
                except queue.Empty:
                    continue
