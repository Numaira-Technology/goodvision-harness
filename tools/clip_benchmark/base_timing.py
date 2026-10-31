"""Original paced file feeder and frame metadata, without unused bus adapters."""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import cv2


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


class TimedVideoFileStream:
    """Pace each decoded original once and record input before candidate sampling.

    This is a private file feeder, not a replacement production source or branch.
    The nominal frame grid and decoder positions are retained separately.
    """

    MAX_LAG_SECONDS = 0.25

    def __init__(self, path, *, start_seconds, stop, input_observer, repeat=False):
        self.path = path
        self.info = probe_video(path)
        self.start_seconds = start_seconds
        self.shared_stop, self.local_stop = stop, Event()
        self.input_observer = input_observer
        self.repeat = repeat
        self.eof_seen = False
        self.last_raw_index = None
        self.input_timing_statistics = {
            "decoded_inputs": 0, "selected": 0, "late_dropped": 0,
            "observer_wall_seconds": 0.0, "observer_max_wall_seconds": 0.0,
        }

    @property
    def stopped(self):
        return self.local_stop.is_set() or self.shared_stop.is_set()

    def stop(self):
        self.local_stop.set()

    def _wait_until(self, target):
        while not self.stopped:
            remaining = target - time.monotonic()
            if remaining <= 0:
                return True
            self.local_stop.wait(min(remaining, 0.05))
        return False

    def frames(self, *, origin=None):
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            capture.release()
            raise ValueError("OpenCV could not open the input video")
        source_fps = self.info.source_fps
        source_index = max(0, math.ceil(self.start_seconds * source_fps - 1e-9))
        loop_index, loop_start, loop_frames = 0, 0, 0
        playback_index = source_index
        if source_index and not capture.set(cv2.CAP_PROP_POS_FRAMES, source_index):
            capture.release()
            raise ValueError("OpenCV could not seek the requested input range")
        origin, emitted = time.monotonic() if origin is None else origin, 0
        try:
            while not self.stopped:
                read_ok, image = capture.read()
                if not read_ok:
                    if not self.repeat:
                        self.eof_seen = True
                        break
                    if source_index < self.info.frame_count:
                        raise ValueError("decoder ended before the original frame count")
                    if loop_frames == 0 or not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        raise ValueError("OpenCV could not repeat the input video")
                    loop_index += 1
                    loop_start, loop_frames, source_index = playback_index, 0, 0
                    continue
                timestamp = playback_index / source_fps
                decoder_next = capture.get(cv2.CAP_PROP_POS_FRAMES)
                decoder_pts = capture.get(cv2.CAP_PROP_POS_MSEC)
                target = origin + timestamp - self.start_seconds
                if not self._wait_until(target):
                    break
                ingress = time.monotonic()
                late = ingress - target > self.MAX_LAG_SECONDS
                self.last_raw_index = playback_index
                source_frame_index = source_index
                source_index += 1
                playback_index += 1
                loop_frames += 1
                disposition = "late_dropped" if late else "selected"
                stats = self.input_timing_statistics
                stats["decoded_inputs"] += 1
                stats[disposition] += 1
                observer_started = time.monotonic()
                try:
                    self.input_observer({
                        "raw_frame_index": self.last_raw_index,
                        **({
                            "loop_index": loop_index,
                            "source_frame_index": source_frame_index,
                            "source_timestamp_seconds": source_frame_index / source_fps,
                            "loop_start_seconds": loop_start / source_fps,
                        } if self.repeat else {}),
                        "decoder_frame_index": (
                            decoder_next - 1 if math.isfinite(decoder_next) and decoder_next >= 1 else None
                        ),
                        "decoder_timestamp_msec": (
                            decoder_pts if math.isfinite(decoder_pts) and decoder_pts >= 0 else None
                        ),
                        "timestamp_seconds": timestamp,
                        "ingress_monotonic": ingress,
                        "scheduled_monotonic": target,
                        "lag_seconds": ingress - target,
                        "disposition": disposition,
                        "frame_sequence": None if late else emitted,
                    })
                finally:
                    elapsed = time.monotonic() - observer_started
                    stats["observer_wall_seconds"] += elapsed
                    stats["observer_max_wall_seconds"] = max(stats["observer_max_wall_seconds"], elapsed)
                if self.stopped:
                    break
                if late:
                    continue
                yield LiveFrame(index=emitted, timestamp_seconds=timestamp, bgr=image)
                emitted += 1
        finally:
            capture.release()

