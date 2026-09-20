"""A persistent video connection shared by preview and analysis sessions.

The capture owner keeps one raw frame and one preview JPEG. Analysis subscribes
to new frames from that owner; detaching an analysis session never disconnects
the camera or opens another decoder.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator, Mapping
from typing import Any

from .frame_crop import FrameCropSelector
from .latency_trace import log_timing, trace_id
from .privacy import PUBLIC_STREAM_ERROR
from .stream_adapter import NetworkStreamAdapter, StreamAdapterError
from .video_to_stream import LiveFrame, encode_jpeg


class VideoConnection:
    """Own a single network decoder without invoking any analysis components."""

    def __init__(
        self,
        stream_url: str,
        *,
        output_fps: float,
        whep_headers: Mapping[str, str] | None = None,
        frame_crop: str = "none",
        preview_max_side: int | None = None,
        jpeg_quality: int = 80,
    ) -> None:
        if not math.isfinite(output_fps) or output_fps <= 0:
            raise ValueError("output_fps must be positive and finite")
        if preview_max_side is not None and (isinstance(preview_max_side, bool) or preview_max_side <= 0):
            raise ValueError("preview_max_side must be positive")
        if isinstance(jpeg_quality, bool) or not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self._crop = FrameCropSelector(frame_crop)
        self.connection_id = uuid.uuid4().hex
        self._adapter = NetworkStreamAdapter(
            stream_url, output_fps=output_fps, whep_headers=whep_headers,
            connection_id=self.connection_id,
        )
        self.source = self._adapter.source
        self.stream_url = self.source.source_url
        self._preview_max_side = preview_max_side
        self._jpeg_quality = jpeg_quality
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._preview_thread: threading.Thread | None = None
        self._display_frames = deque()
        self._display_bytes = 0
        self._display_skipped = 0
        self._display_encoded = 0
        self._display_encode_errors = 0
        self._display_encoded_at: dict[int, float] = {}
        self._latest_received_at: float | None = None
        self._delivery_started = time.monotonic()
        self._delivery_sessions: set[str] = set()
        self._delivery = dict(requests=0, empty_requests=0, frames=0, bytes=0,
                              prepare_seconds=0.0, prepare_max_seconds=0.0,
                              encoded_age_max_seconds=0.0)
        self._state = "disconnected"
        self._error: StreamAdapterError | None = None
        self._sequence = 0
        self._latest: LiveFrame | None = None
        self._jpeg: bytes | None = None

    @property
    def running(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                if self._thread.is_alive() and not self._stop.is_set():
                    return
                raise RuntimeError("create a new video connection to reconnect")
            if self._stop.is_set():
                raise RuntimeError("create a new video connection to reconnect")
            self._state = "connecting"
            self._thread = threading.Thread(
                target=self._capture,
                name=f"video-connection-{self.connection_id[:8]}",
                daemon=True,
            )
            self._preview_thread = threading.Thread(
                target=self._encode_preview, name="video-display-encoder", daemon=True,
            )
            self._preview_thread.start()
            self._thread.start()

    def stop(self, timeout: float = 0.2) -> None:
        """Request shutdown; native decoder cleanup stays on its owner thread."""
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be non-negative and finite")
        with self._condition:
            self._stop.set()
            thread = self._thread
            self._state = (
                "disconnecting" if thread is not None and thread.is_alive()
                else "disconnected"
            )
            self._latest = None
            self._jpeg = None
            self._display_frames.clear()
            self._display_encoded_at.clear()
            self._display_bytes = 0
            self._condition.notify_all()
        self._adapter.stop()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=timeout)

    def status(self) -> dict[str, Any]:
        with self._condition:
            state = self._state
            if self._stop.is_set():
                state = (
                    "disconnecting"
                    if self._thread is not None and self._thread.is_alive()
                    else "disconnected"
                )
            result: dict[str, Any] = {
                "connection_state": state,
                "connected": state == "connected",
                "connection_id": self.connection_id,
                "display_capture_skipped_frames": self._display_skipped,
                "display_encoded_frames": self._display_encoded,
                "display_encode_errors": self._display_encode_errors,
            }
            if state == "error" and self._error is not None:
                result["connection_error"] = self._error.public_message
            return result

    def latest_jpeg(self) -> bytes | None:
        with self._condition:
            return self._jpeg

    def display_frames(self, after: int = -1, *, max_age_seconds: float = 0.5,
                       limit: int = 8) -> tuple:
        """Bounded presentation history; never used to queue analysis work.

        The caller's playback budget determines which frames are still useful.
        History remains bounded by the encoder's byte/frame limits. Skipped
        source sequences remain visible to the contour validator.
        """
        with self._condition:
            if not self._display_frames:
                return ()
            latest = self._display_frames[-1]
            if after < 0:
                return (latest,)
            return tuple(item for item in self._display_frames
                         if item[0] > after
                         and item[1] >= latest[1] - max_age_seconds)[:limit]

    def subscribe(self) -> VideoSubscription:
        with self._condition:
            if self._stop.is_set() or self._state == "disconnected":
                raise RuntimeError("video is not connected")
            if self._error is not None:
                raise StreamAdapterError(
                    self._error.public_message, code=self._error.code,
                ) from None
            return VideoSubscription(self, self._sequence)

    def record_display_delivery(self, *, frames: list[dict], byte_count: int,
                                prepare_seconds: float, stream_session_id: str) -> None:
        """Response preparation/encoded-frame age, not socket delivery latency."""
        now = time.monotonic()
        with self._condition:
            stats = self._delivery
            sid = trace_id(stream_session_id)
            if sid is not None and len(self._delivery_sessions) < 8:
                self._delivery_sessions.add(sid)
            stats['requests'] += 1
            stats['empty_requests'] += not frames
            stats['frames'] += len(frames)
            stats['bytes'] += byte_count
            stats['prepare_seconds'] += prepare_seconds
            stats['prepare_max_seconds'] = max(stats['prepare_max_seconds'], prepare_seconds)
            ages = [now-self._display_encoded_at[f['source_sequence']] for f in frames
                    if f['source_sequence'] in self._display_encoded_at]
            if ages:
                stats['encoded_age_max_seconds'] = max(stats['encoded_age_max_seconds'], max(ages))
            elapsed = now-self._delivery_started
            if elapsed < 5:
                return
            payload = dict(stats, window_seconds=elapsed,
                           stream_session_ids=sorted(self._delivery_sessions),
                           prepare_mean_seconds=stats['prepare_seconds']/stats['requests'],
                           response_first_source_sequence=frames[0]['source_sequence'] if frames else None,
                           response_last_source_sequence=frames[-1]['source_sequence'] if frames else None)
            self._delivery = {key: 0 for key in stats}
            self._delivery_sessions.clear()
            self._delivery_started = now
        log_timing('network_display_delivery_timing', connection_id=self.connection_id,
                   **payload)

    def _capture(self) -> None:
        try:
            for frame in self._adapter.frames():
                with self._condition:
                    if self._stop.is_set():
                        break
                    self._latest = frame
                    self._latest_received_at = time.monotonic()
                    self._sequence += 1
                    self._state = "connected"
                    self._condition.notify_all()
            if not self._stop.is_set():
                raise StreamAdapterError(PUBLIC_STREAM_ERROR)
        except Exception as exc:
            with self._condition:
                if not self._stop.is_set():
                    code = (
                        exc.code if isinstance(exc, StreamAdapterError)
                        else "connection_failed"
                    )
                    self._error = StreamAdapterError(PUBLIC_STREAM_ERROR, code=code)
                    self._state = "error"
        finally:
            self._adapter.stop()
            with self._condition:
                if self._stop.is_set():
                    self._state = "disconnected"
                    self._jpeg = None
                self._latest = None
                self._condition.notify_all()

    def _encode_preview(self) -> None:
        """Encoding cannot hold up the decoder or an analysis subscriber."""
        sequence = 0
        window_start = time.monotonic()
        window_frames = 0
        window_encode = 0.0
        window_max_encode = 0.0
        window_wait = window_max_wait = 0.0
        window_bytes = 0
        last_error_report = 0.0
        while not self._stop.is_set():
            with self._condition:
                self._condition.wait_for(lambda: self._stop.is_set()
                                         or self._error is not None
                                         or self._sequence > sequence)
                if self._stop.is_set() or self._error is not None:
                    return
                frame = self._latest
                received_at = self._latest_received_at
                self._display_skipped += max(0, self._sequence - sequence - 1)
                sequence = self._sequence
            if frame is None:
                continue
            try:
                encode_started = time.monotonic()
                wait = 0.0 if received_at is None else encode_started-received_at
                preview = next(self._crop.frames(iter((frame,)), stopped=self._stop.is_set), None)
                if preview is None:
                    continue
                jpeg, width, height = encode_jpeg(
                    preview.bgr, max_image_side=self._preview_max_side,
                    quality=self._jpeg_quality,
                )
                elapsed = time.monotonic() - encode_started
                window_encode += elapsed
                window_max_encode = max(window_max_encode, elapsed)
                window_frames += 1
                window_wait += wait
                window_max_wait = max(window_max_wait, wait)
                window_bytes += len(jpeg)
            except Exception as exc:
                # Display is optional; an encoder failure must not stop analysis.
                with self._condition:
                    self._display_skipped += 1
                    self._display_encode_errors += 1
                now = time.monotonic()
                if now-last_error_report >= 5:
                    log_timing('network_display_encode_error', connection_id=self.connection_id,
                               source_sequence=frame.index, error_type=type(exc).__name__,
                               total_errors=self._display_encode_errors)
                    last_error_report = now
                continue
            with self._condition:
                if self._stop.is_set():
                    return
                self._jpeg = jpeg
                self._display_encoded += 1
                self._display_frames.append((frame.index, frame.timestamp_seconds,
                                             width, height, jpeg))
                self._display_encoded_at[frame.index] = time.monotonic()
                self._display_bytes += len(jpeg)
                while (len(self._display_frames) > 120 or self._display_bytes > 32 * 1024**2):
                    removed = self._display_frames.popleft()
                    self._display_bytes -= len(removed[4])
                    self._display_encoded_at.pop(removed[0], None)
            now = time.monotonic()
            if now - window_start >= 5:
                log_timing("network_capture_display_timing", connection_id=self.connection_id,
                           source_sequence=frame.index, source_timestamp_seconds=frame.timestamp_seconds,
                           captured_frames=sequence, encoded_frames=self._display_encoded,
                           display_capture_skipped_frames=self._display_skipped,
                           window_seconds=now-window_start, window_encoded_frames=window_frames,
                           window_encoded_fps=window_frames/(now-window_start),
                           window_encode_mean_seconds=window_encode/max(1, window_frames),
                           window_encode_max_seconds=window_max_encode,
                           window_encode_wait_mean_seconds=window_wait/max(1, window_frames),
                           window_encode_wait_max_seconds=window_max_wait,
                           window_jpeg_bytes=window_bytes, jpeg_quality=self._jpeg_quality,
                           display_encode_errors=self._display_encode_errors,
                           width=width, height=height)
                window_start, window_frames, window_encode, window_max_encode = now, 0, 0.0, 0.0
                window_wait, window_max_wait, window_bytes = 0.0, 0.0, 0


class VideoSubscription:
    """A session's bounded, independently stoppable view of a connection."""

    realtime = True

    def __init__(self, connection: VideoConnection, sequence: int) -> None:
        self._connection = connection
        self._sequence = sequence
        self._stop = threading.Event()
        self._consumed = False
        self.source = connection.source
        self.dropped_frames = 0

    @property
    def stopped(self) -> bool:
        return self._stop.is_set() or self._connection._stop.is_set()

    @property
    def running(self) -> bool:
        return not self.stopped and self._connection.running

    def stop(self) -> None:
        with self._connection._condition:
            self._stop.set()
            self._connection._condition.notify_all()

    def frames(self) -> Iterator[LiveFrame]:
        connection = self._connection
        with connection._condition:
            if self._consumed:
                raise RuntimeError("video subscription has already been consumed")
            self._consumed = True
        try:
            while True:
                with connection._condition:
                    connection._condition.wait_for(
                        lambda: self.stopped
                        or connection._error is not None
                        or connection._sequence > self._sequence
                        or connection._state == "disconnected"
                    )
                    if self.stopped or connection._state == "disconnected":
                        return
                    if connection._error is not None:
                        raise StreamAdapterError(
                            connection._error.public_message,
                            code=connection._error.code,
                        ) from None
                    frame = connection._latest
                    self.dropped_frames += max(
                        0, connection._sequence - self._sequence - 1,
                    )
                    self._sequence = connection._sequence
                if frame is not None:
                    yield frame
        finally:
            self._stop.set()
