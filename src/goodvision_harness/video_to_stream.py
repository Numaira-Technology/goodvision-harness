"""Live frame sources: a video file paced on the wall clock, plus a push source.

`VideoFileStream` is the "video → live camera" adapter: it decodes a file
sequentially and releases each frame only once the wall clock reaches the
frame's media timestamp, so downstream consumers experience the file exactly
like a live feed. `PushSampledSource` is the mirror image for real cameras:
frames captured elsewhere (for example in the browser) are pushed in and
stamped with session-relative arrival time.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .sliding_window import SampledFrame


@dataclass(frozen=True, slots=True)
class LiveFrame:
    """One decoded BGR frame surfaced by a live source."""

    index: int
    timestamp_seconds: float
    bgr: Any


@dataclass(frozen=True, slots=True)
class VideoInfo:
    duration_seconds: float
    source_fps: float
    frame_count: int
    width: int
    height: int


def probe_video(path: str | Path) -> VideoInfo:
    """Read container metadata, raising when the file is not a usable video."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError("the video does not report a valid duration")
    return VideoInfo(
        duration_seconds=frame_count / fps,
        source_fps=fps,
        frame_count=int(frame_count),
        width=width,
        height=height,
    )


def encode_jpeg(
    bgr: Any,
    *,
    max_image_side: int | None,
    quality: int,
) -> tuple[bytes, int, int]:
    """Encode with an optional side cap; None preserves the input dimensions."""

    height, width = bgr.shape[:2]
    longest_side = max(height, width)
    if max_image_side is not None and longest_side > max_image_side:
        scale = max_image_side / longest_side
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    encode_ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encode_ok:
        raise ValueError("failed to JPEG-encode frame")
    return encoded.tobytes(), width, height


def crop_frame(bgr: Any, *, mode: str) -> Any:
    """Select one horizontal view before any resizing or JPEG encoding."""

    if mode == "none":
        return bgr
    if mode not in {"left_half", "right_half"}:
        raise ValueError("frame crop mode must be none, left_half, or right_half")
    if getattr(bgr, "ndim", 0) < 2:
        raise ValueError("frame must have at least two dimensions")
    width = int(bgr.shape[1])
    if width < 2:
        raise ValueError("frame must be at least two pixels wide to crop in half")
    midpoint = width // 2
    if mode == "left_half":
        return bgr[:, :midpoint]
    return bgr[:, midpoint:]


class VideoFileStream:
    """Emit decoded frames from a file at wall-clock pace, like a camera.

    Frames arrive at most at ``display_fps`` and, in realtime mode, never
    earlier than their media timestamp. Frames that fall more than
    ``MAX_LAG_SECONDS`` behind schedule are dropped instead of queued, because
    a live camera cannot pause reality; ``dropped_frames`` counts them.
    """

    MAX_LAG_SECONDS = 0.25

    def __init__(
        self,
        path: str | Path,
        *,
        display_fps: float = 15.0,
        playback_speed: float = 1.0,
        start_seconds: float = 0.0,
        realtime: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None:
        if display_fps <= 0:
            raise ValueError("display_fps must be positive")
        if playback_speed <= 0:
            raise ValueError("playback_speed must be positive")
        if isinstance(start_seconds, bool) or start_seconds < 0:
            raise ValueError("start_seconds must be a non-negative number")
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"video does not exist: {self.path}")
        self.info = probe_video(self.path)
        self.display_fps = float(display_fps)
        self.playback_speed = float(playback_speed)
        self.start_seconds = float(start_seconds)
        self.realtime = realtime
        self.dropped_frames = 0
        self._stop = stop_event or threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def frames(self) -> Iterator[LiveFrame]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"OpenCV could not open video: {self.path}")

        source_fps = self.info.source_fps
        source_index = max(0, math.ceil(self.start_seconds * source_fps - 1e-9))
        if source_index and not capture.set(cv2.CAP_PROP_POS_FRAMES, source_index):
            capture.release()
            raise ValueError(
                f"OpenCV could not seek video to {self.start_seconds:.3f}s"
            )

        emit_period = 1.0 / self.display_fps
        next_emit_time = source_index / source_fps
        started_wall = time.monotonic()
        emitted = 0
        try:
            while not self._stop.is_set():
                read_ok, image = capture.read()
                if not read_ok:
                    break
                timestamp = source_index / source_fps
                source_index += 1

                if timestamp + 1e-9 < next_emit_time:
                    continue
                while next_emit_time <= timestamp + 1e-9:
                    next_emit_time += emit_period

                if self.realtime:
                    target = started_wall + (
                        (timestamp - self.start_seconds) / self.playback_speed
                    )
                    if not _sleep_until(target, self._stop):
                        break
                    if time.monotonic() - target > self.MAX_LAG_SECONDS:
                        self.dropped_frames += 1
                        continue

                yield LiveFrame(
                    index=emitted,
                    timestamp_seconds=timestamp,
                    bgr=image,
                )
                emitted += 1
        finally:
            capture.release()


def _sleep_until(target: float, stop: threading.Event) -> bool:
    """Sleep until the monotonic target; False when stop was set first."""

    while True:
        remaining = target - time.monotonic()
        if remaining <= 0:
            return not stop.is_set()
        if stop.wait(min(remaining, 0.05)):
            return False


class PushSampledSource:
    """Model-ready frames pushed from outside (for example a browser camera).

    Each pushed JPEG becomes a ``SampledFrame`` stamped with session-relative
    arrival time, so the sliding-window and inference layers treat camera
    frames exactly like frames sampled from a file. When the queue is full the
    newest frame is dropped — live sources must never build a backlog.
    """

    def __init__(self, *, max_queue: int = 32) -> None:
        if max_queue < 2:
            raise ValueError("max_queue must be at least 2")
        self._queue: queue.Queue[SampledFrame | None] = queue.Queue(maxsize=max_queue)
        self._started_wall = time.monotonic()
        self._last_timestamp = 0.0
        self._sample_index = 0
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self.dropped_frames = 0

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def push_jpeg(
        self,
        jpeg_bytes: bytes,
        *,
        width: int,
        height: int,
        timestamp_seconds: float | None = None,
        source_frame_index: int | None = None,
    ) -> bool:
        """Enqueue one camera frame; False when closed or the queue is full."""

        if self._closed.is_set() or not jpeg_bytes:
            return False
        with self._lock:
            # Clamp to monotonic non-decreasing session time.
            supplied_timestamp = (
                time.monotonic() - self._started_wall
                if timestamp_seconds is None
                else float(timestamp_seconds)
            )
            timestamp = max(self._last_timestamp, supplied_timestamp)
            source_index = (
                self._sample_index
                if source_frame_index is None
                else int(source_frame_index)
            )
            frame = SampledFrame(
                sample_index=self._sample_index,
                source_frame_index=source_index,
                timestamp_seconds=timestamp,
                jpeg_bytes=jpeg_bytes,
                width=width,
                height=height,
            )
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                self.dropped_frames += 1
                return False
            # Only committed frames advance the contiguous sample index.
            self._sample_index += 1
            self._last_timestamp = timestamp
        return True

    def close(self) -> None:
        """Stop accepting frames and unblock the consumer."""

        if self._closed.is_set():
            return
        self._closed.set()
        while True:
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    continue

    def frames(self) -> Iterator[SampledFrame]:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            yield frame
