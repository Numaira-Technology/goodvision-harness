"""Bounded local frame extraction shared by the standalone demo helpers."""

import math
import subprocess

# A .MOV filename alone does not prevent FFmpeg from detecting a playlist and
# following nested files/URLs. This demo accepts MOV/MP4 containers only, with
# external tracks disabled; probing and extraction must use the same boundary.
_INPUT_OPTIONS = ("-protocol_whitelist", "file", "-f", "mov", "-enable_drefs", "0")


def validate_times(times):
    times = list(times)
    if not times or any(
        type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times
    ):
        raise ValueError("sample times must be finite, nonnegative seconds")
    return times


def duration(video_path):
    value = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            *_INPUT_OPTIONS,
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout.strip()
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("clip has no finite positive duration")
    return seconds


def frame_jpeg(video_path, t, *, width=768):
    validate_times([t])
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{t:.6f}",
            *_INPUT_OPTIONS,
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-vf",
            f"scale={width}:{width}:force_original_aspect_ratio=decrease:"
            "force_divisible_by=2",
            "-f",
            "mjpeg",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=15,
    )
    # ffmpeg can exit successfully without producing a frame (for example at EOF).
    if not proc.stdout.startswith(b"\xff\xd8") or not proc.stdout.endswith(b"\xff\xd9"):
        raise ValueError("no decodable frame at the requested time")
    return proc.stdout
