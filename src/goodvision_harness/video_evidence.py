"""Bounded compressed-frame history and independently decodable event clips."""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class EvidenceFrame:
    jpeg: bytes
    timestamp_seconds: float
    source_frame_index: int


class FrameRing:
    """Owner-thread-only JPEG ring bounded by both source time and bytes."""

    def __init__(self, seconds: float, max_bytes: int) -> None:
        self.seconds = seconds
        self.max_bytes = max_bytes
        self.frames: deque[EvidenceFrame] = deque()
        self.bytes_used = 0
        self.evicted = 0

    def append(self, frame: EvidenceFrame) -> None:
        if self.frames and frame.timestamp_seconds <= self.frames[-1].timestamp_seconds:
            return
        if len(frame.jpeg) > self.max_bytes:
            self.evicted += 1
            return
        self.frames.append(frame)
        self.bytes_used += len(frame.jpeg)
        cutoff = frame.timestamp_seconds - self.seconds
        while self.frames and (
            self.frames[0].timestamp_seconds < cutoff
            or self.bytes_used > self.max_bytes
        ):
            self.bytes_used -= len(self.frames.popleft().jpeg)
            self.evicted += 1

    def select(
        self, start: float, end: float, anchor: float, fps: float
    ) -> tuple[list[EvidenceFrame], str | None]:
        """Select one contiguous real segment; never bridge a source outage."""
        candidates = [
            frame
            for frame in self.frames
            if start - 1e-6 <= frame.timestamp_seconds < end - 1e-6
        ]
        if not candidates:
            return [], "source_frames_unavailable"
        groups: list[list[EvidenceFrame]] = [[]]
        max_gap = max(1.0, 2.5 / fps)
        for frame in candidates:
            if groups[-1] and (
                frame.timestamp_seconds - groups[-1][-1].timestamp_seconds > max_gap
            ):
                groups.append([])
            groups[-1].append(frame)
        selected = min(
            groups,
            key=lambda group: min(abs(f.timestamp_seconds - anchor) for f in group),
        )
        if min(abs(f.timestamp_seconds - anchor) for f in selected) > max_gap:
            return [], "anchor_frame_unavailable"
        reason = None
        if len(groups) > 1:
            reason = "source_gap"
        elif selected[0].timestamp_seconds > start + 1.5 / fps:
            reason = "pre_roll_unavailable"
        elif selected[-1].timestamp_seconds + 1.5 / fps < end:
            reason = "post_roll_unavailable"
        return selected, reason


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stage_clip(
    directory: Path, frames: list[EvidenceFrame], metadata: dict[str, Any]
) -> Path:
    """Publish a recovery manifest only after all selected JPEGs are durable."""
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, frame in enumerate(frames):
        filename = f"{index:04d}.jpg"
        with (directory / filename).open("wb") as handle:
            handle.write(frame.jpeg)
            handle.flush()
            os.fsync(handle.fileno())
        records.append(
            {
                "filename": filename,
                "timestamp_seconds": frame.timestamp_seconds,
                "source_frame_index": frame.source_frame_index,
            }
        )
    manifest = directory / "manifest.json"
    atomic_json(manifest, {**metadata, "frames": records})
    return manifest


def encode_staged_clip(
    manifest_path: Path,
    destination: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    timeout_seconds: float = 10.0,
    max_image_side: int = 720,
) -> dict[str, Any]:
    """Encode with software H.264, validate, then atomically publish the MP4."""
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = metadata["frames"]
    if not records:
        raise ValueError("source_frames_unavailable")
    fps = float(metadata["fps"])
    first = float(records[0]["timestamp_seconds"])
    last = float(records[-1]["timestamp_seconds"])
    end = min(float(metadata["requested_end_seconds"]), last + 1.0 / fps)
    count = max(1, math.ceil((end - first) * fps - 1e-6))
    selected = []
    cursor = 0
    for index in range(count):
        timestamp = first + index / fps
        while cursor + 1 < len(records) and (
            float(records[cursor + 1]["timestamp_seconds"]) <= timestamp + 1e-6
        ):
            cursor += 1
        selected.append(records[cursor])
    encoded_frames = []
    size: tuple[int, int] | None = None
    for record in selected:
        filename = str(record["filename"])
        if Path(filename).name != filename or not filename.endswith(".jpg"):
            raise ValueError("invalid_staged_frame")
        image = cv2.imdecode(
            np.frombuffer((manifest_path.parent / filename).read_bytes(), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError("invalid_source_frame")
        if size is None:
            height, width = image.shape[:2]
            scale = min(1.0, max_image_side / max(width, height))
            size = (
                max(2, int(width * scale) // 2 * 2),
                max(2, int(height * scale) // 2 * 2),
            )
        if (image.shape[1], image.shape[0]) != size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("invalid_source_frame")
        encoded_frames.append(jpeg.tobytes())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp.mp4")
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-frames:v",
        str(count),
    ]
    if metadata.get("max_output_bytes"):
        command.extend(["-fs", str(int(metadata["max_output_bytes"]))])
    command.append(str(temporary))
    try:
        completed = subprocess.run(
            command,
            input=b"".join(encoded_frames),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        if completed.returncode or not temporary.is_file():
            raise RuntimeError("video_encoding_failed")
        capture = cv2.VideoCapture(str(temporary))
        try:
            ok, _ = capture.read()
            actual_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if not ok or actual_count != count or abs(actual_fps - fps) > 0.01:
            raise RuntimeError("video_validation_failed")
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    assert size is not None
    return {
        "start_seconds": first,
        "end_seconds": first + count / fps,
        "duration_seconds": count / fps,
        "fps": fps,
        "frame_count": count,
        "width": size[0],
        "height": size[1],
        "byte_length": destination.stat().st_size,
        "mime_type": "video/mp4",
        "partial": bool(metadata.get("partial_reason")),
        "partial_reason": metadata.get("partial_reason"),
    }
