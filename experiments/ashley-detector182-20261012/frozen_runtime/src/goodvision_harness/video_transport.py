"""Optional bounded independent H264 upload copies; source JPEGs stay intact."""

from __future__ import annotations

import io
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import httpx

from .video_tracking import VideoFrame, _validate_frames

MAX_H264_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class EncodedH264:
    data: bytes
    width: int
    height: int
    frame_count: int
    prepare_seconds: float
    encode_seconds: float
    total_seconds: float


def ensure_h264_encoder_available() -> None:
    try:
        import av

        if not av.Codec("libx264", "w").is_encoder:
            raise RuntimeError("H264 encoder is unavailable")
    except ImportError:
        raise RuntimeError(
            "H264 transport requires the optional video dependency"
        ) from None


def encode_h264_batch(
    frames: Sequence[VideoFrame],
    *,
    max_side: int = 512,
    crf: int = 23,
    deadline: float | None = None,
) -> EncodedH264:
    """One IDR/SPS/PPS per request, I/P only; ordered metadata owns source time.

    Full images are resized directly from the source JPEG into even dimensions
    for 4:2:0 chroma, with no crop and no intervening lossy JPEG. Codec state is
    local to this call; neither native object state nor video history lives here.
    """
    _validate_frames(frames)
    frames = tuple(frames)
    _validate_frames(frames)
    if type(max_side) is not int or not 128 <= max_side <= 1280:
        raise ValueError("H264 max_side must be an integer in [128, 1280]")
    if type(crf) is not int or not 18 <= crf <= 35:
        raise ValueError("H264 CRF must be an integer in [18, 35]")
    if deadline is None:
        deadline = time.monotonic() + 30
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("H264 deadline must be finite")

    def check():
        if time.monotonic() >= deadline:
            raise httpx.TimeoutException("H264 preparation exceeded total deadline")

    started = time.perf_counter()
    check()
    import av
    import cv2
    import numpy as np
    from PIL import Image

    codec = None
    source_size = output_size = None
    pixels = total_bytes = 0
    prepare_seconds = encode_seconds = 0.0
    chunks = []

    def append_packets(packets):
        nonlocal total_bytes
        for packet in packets:
            chunk = bytes(packet)
            total_bytes += len(chunk)
            if total_bytes > MAX_H264_BYTES:
                raise ValueError("H264 upload exceeds byte limit")
            chunks.append(chunk)

    for index, source in enumerate(frames):
        check()
        prepare_started = time.perf_counter()
        with Image.open(io.BytesIO(source.jpeg_bytes)) as header:
            width, height = header.size
            pixels += width * height
            if (
                header.format != "JPEG"
                or min(width, height) < 2
                or width * height > 3840 * 2160
                or pixels > 64 * 1024 * 1024
            ):
                raise ValueError("H264 source exceeds decoded pixel limits")
            if source_size is not None and source_size != header.size:
                raise ValueError("H264 source resolution changed within batch")
            source_size = header.size
        image = cv2.imdecode(
            np.frombuffer(source.jpeg_bytes, np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if image is None or image.shape[:2] != (height, width):
            raise ValueError("H264 source JPEG is invalid")
        scale = min(1.0, max_side / max(width, height))
        output_size = tuple(
            min(max_side // 2 * 2, max(2, 2 * round(value * scale / 2)))
            for value in (width, height)
        )
        if output_size != (width, height):
            image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = index
        frame.time_base = Fraction(1, 4)
        prepare_seconds += time.perf_counter() - prepare_started
        check()
        encode_started = time.perf_counter()
        if codec is None:
            codec = av.CodecContext.create("libx264", "w")
            codec.width, codec.height = output_size
            codec.pix_fmt = "yuv420p"
            codec.time_base = Fraction(1, 4)
            codec.framerate = Fraction(4, 1)
            codec.thread_count = 2
            codec.options = {
                "preset": "veryfast",
                "tune": "zerolatency",
                "crf": str(crf),
                "profile": "baseline",
                "bf": "0",
                "g": "16",
                "x264-params": "repeat-headers=1:annexb=1:scenecut=0:aud=1:ref=1",
            }
        append_packets(codec.encode(frame))
        encode_seconds += time.perf_counter() - encode_started
        check()
    encode_started = time.perf_counter()
    append_packets(codec.encode(None))
    encoded = b"".join(chunks)
    encode_seconds += time.perf_counter() - encode_started
    check()
    return EncodedH264(
        encoded,
        *output_size,
        len(frames),
        prepare_seconds,
        encode_seconds,
        time.perf_counter() - started,
    )
