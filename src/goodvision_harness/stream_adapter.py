"""Network video adapters for RTSP, RTMP, and WebRTC/WHEP sources.

RTSP/RTMP use OpenCV or an explicit FFmpeg color pipeline. WebRTC uses WHEP,
written as webrtc+https:// (or webrtc+http:// for local development). The
adapter emits the same LiveFrame objects as VideoFileStream, keeping the rest
of the sampling, windowing, inference, and display pipeline unchanged.
"""

from __future__ import annotations

import asyncio
import math
import os
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
from .stream_decode import DecodeSupportError, FFmpegReceiver
from .stream_diagnostics import StreamDiagnostics
from .stream_timing import SourceCadence
from .video_to_stream import LiveFrame

StreamProtocol = Literal["rtsp", "rtmp", "webrtc"]


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


class NetworkStreamAdapter:
    """Adapt a supported network stream to the VideoFileStream interface."""

    realtime = True

    def __init__(
        self,
        source_url: str,
        *,
        output_fps: float = 30.0,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 1.0,
        whep_headers: Mapping[str, str] | None = None,
        connection_id: str | None = None,
        decoder_backend: str | None = None,
    ) -> None:
        if not math.isfinite(output_fps) or output_fps <= 0:
            raise ValueError("output_fps must be positive and finite")
        if reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be non-negative")
        if reconnect_delay_seconds < 0:
            raise ValueError("reconnect_delay_seconds must be non-negative")
        self.source = parse_stream_source(source_url)
        self.output_fps = float(output_fps)
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.whep_headers = dict(whep_headers or {})
        self.decoder_backend = decoder_backend or os.getenv(
            "HARNESS_STREAM_DECODER", "opencv",
        )
        if self.decoder_backend not in {"opencv", "ffmpeg"}:
            raise ValueError("stream decoder must be opencv or ffmpeg")
        self._receiver: FFmpegReceiver | None = None
        self._whep_receiver: Any | None = None
        self._whep_consumed = False
        self.dropped_frames = 0
        self._diagnostics = StreamDiagnostics(connection_id or uuid.uuid4().hex,
                                              self.source.protocol, self.output_fps)
        self._stop = threading.Event()
        self._capture_lock = threading.Lock()
        self._capture: Any | None = None
        self._queue: queue.Queue[LiveFrame] = queue.Queue(maxsize=2)

    @property
    def protocol(self) -> StreamProtocol:
        return self.source.protocol

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()
        with self._capture_lock:
            receiver = self._receiver
            whep_receiver = self._whep_receiver
        if receiver is not None:
            receiver.stop()
        if whep_receiver is not None:
            whep_receiver.stop()

    def frames(self) -> Iterator[LiveFrame]:
        try:
            if self.protocol in {"rtsp", "rtmp"}:
                if self.decoder_backend == "ffmpeg":
                    yield from self._ffmpeg_frames()
                else:
                    yield from self._opencv_frames()
            else:
                yield from self._whep_frames()
        finally:
            # WHEP's real receive counters live and are emitted in the child.
            # A zeroed parent summary would overwrite/misrepresent that result.
            if self.protocol != "webrtc":
                self._diagnostics.emit(final=True)

    def _ffmpeg_frames(self) -> Iterator[LiveFrame]:
        cadence = SourceCadence(self.output_fps)
        emitted = failures = 0
        while not self.stopped:
            receiver = FFmpegReceiver(
                self.source.transport_url, self.protocol, self.output_fps,
            )
            with self._capture_lock:
                self._receiver = receiver
            cadence.reconnect()
            try:
                wait_started = time.monotonic()
                for frame in receiver.frames():
                    if self.stopped:
                        return
                    failures = 0
                    now = time.monotonic()
                    pts = (
                        float(frame.pts * frame.time_base)
                        if frame.pts is not None and frame.time_base is not None
                        else None
                    )
                    timestamp, selected = cadence.select(now, pts)
                    self._diagnostics.received(frame, now, now-wait_started,
                                               emitted if selected else None)
                    self._diagnostics.configure_decoder(
                        "ffmpeg", receiver.metadata, receiver.decode_errors,
                        cadence.discontinuities,
                    )
                    wait_started = now
                    if not selected:
                        self.dropped_frames += 1
                        self._diagnostics.count("rate_limit_skipped_frames")
                        continue
                    started = time.monotonic()
                    image = frame.to_ndarray(format="bgr24")
                    self._diagnostics.timing(
                        "color_conversion", time.monotonic()-started,
                    )
                    self._diagnostics.enqueued(emitted)
                    self._diagnostics.dequeued(emitted)
                    self._diagnostics.emit()
                    yield LiveFrame(
                        index=emitted, timestamp_seconds=timestamp, bgr=image,
                    )
                    emitted += 1
                failures += 1
            except DecodeSupportError as exc:
                raise StreamAdapterError("video decoder support unavailable",
                                         code="missing_support") from exc
            except Exception as exc:
                if self.stopped:
                    return
                failures += 1
                self._diagnostics.count("decoder_restarts")
                if failures > self.reconnect_attempts:
                    raise StreamAdapterError("video receiver failed") from exc
            finally:
                receiver.stop()
                with self._capture_lock:
                    self._receiver = None
            if failures > self.reconnect_attempts:
                raise StreamAdapterError("video receiver ended")
            if self._stop.wait(self.reconnect_delay_seconds):
                return

    def _opencv_frames(self) -> Iterator[LiveFrame]:
        failures = 0
        emitted = 0
        cadence = SourceCadence(self.output_fps)

        while not self.stopped:
            cadence.reconnect()
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
                        raw_pts = capture.get(cv2.CAP_PROP_POS_MSEC)
                        pts = (
                            raw_pts / 1000
                            if math.isfinite(raw_pts) and raw_pts > 0 else None
                        )
                        timestamp, selected = cadence.select(now, pts)
                        self._diagnostics.received(
                            image, now, now-read_started,
                            emitted if selected else None, source_pts_seconds=pts,
                        )
                        if not selected:
                            self.dropped_frames += 1
                            self._diagnostics.count('rate_limit_skipped_frames')
                            self._diagnostics.emit()
                            continue
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
        from .whep_process import WhepProcessReceiver

        if self._whep_consumed:
            raise RuntimeError("WebRTC stream has already been consumed")
        self._whep_consumed = True
        if self.stopped:
            return
        receiver = WhepProcessReceiver({
            "source_url": self.source.source_url,
            "output_fps": self.output_fps,
            "whep_headers": self.whep_headers,
            "connection_id": self._diagnostics.connection_id,
        })
        with self._capture_lock:
            self._whep_receiver = receiver
        try:
            if not self.stopped:
                yield from receiver.frames()
        finally:
            receiver.close()
            self.dropped_frames += receiver.dropped_frames
            with self._capture_lock:
                self._whep_receiver = None

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

        transceiver = pc.addTransceiver("video", direction="recvonly")
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
                    from .webrtc_jitter import protect_receiver

                    protect_receiver(transceiver.receiver, answer_sdp, self._diagnostics)
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
        cadence = SourceCadence(self.output_fps)
        receive_started = time.monotonic()
        receive_task: asyncio.Task[Any] | None = asyncio.create_task(track.recv())
        try:
            while not self.stopped:
                done, _ = await asyncio.wait({receive_task}, timeout=0.5)
                if not done:
                    self._diagnostics.count('receive_wait_over_500ms')
                    if pc.connectionState in {"closed", "failed"}:
                        raise StreamAdapterError("WebRTC connection closed")
                    if time.monotonic() - receive_started >= 15.0:
                        raise StreamAdapterError("WebRTC video receive timed out")
                    continue
                try:
                    frame = receive_task.result()
                except media_stream_error:
                    self._diagnostics.count('track_ended')
                    return
                now = time.monotonic()
                pts = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None and frame.time_base is not None else None
                )
                timestamp, selected = cadence.select(now, pts)
                self._diagnostics.received(frame, now, now-receive_started,
                                           emitted if selected else None)
                receive_started = now
                receive_task = asyncio.create_task(track.recv())
                if not selected:
                    self.dropped_frames += 1
                    self._diagnostics.count('rate_limit_skipped_frames')
                    continue
                conversion_started = time.monotonic()
                try:
                    # This loop also receives ICE/RTP datagrams. Conversion may
                    # stall under analysis load, so keep it off the receive loop.
                    # Await each conversion to preserve order and bound work.
                    bgr = await asyncio.to_thread(frame.to_ndarray, format="bgr24")
                except Exception:
                    self._diagnostics.count('color_conversion_errors')
                    raise
                self._diagnostics.timing(
                    'color_conversion', time.monotonic()-conversion_started,
                )
                if self.stopped:
                    return
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
