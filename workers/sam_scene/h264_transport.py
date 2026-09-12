"""Bounded independent Annex-B batches; inspect allocation metadata before AV."""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from dataclasses import dataclass, replace

MAX_H264_BYTES = 8 * 1024 * 1024


def ensure_h264_decoder_available() -> None:
    """Fail startup explicitly when the optional CPU decoder is unavailable."""
    import av

    if not av.Codec("h264", "r").is_decoder:
        raise RuntimeError("H264 decoder is unavailable")


class H264TransportError(ValueError):
    """A client-controlled malformed or oversized elementary stream."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _require(condition, message, status_code=400):
    if not condition:
        raise H264TransportError(message, status_code)


class _Bits:
    def __init__(self, nal):
        data = bytes(nal)
        self.data = data.replace(b"\x00\x00\x03", b"\x00\x00")
        self.position = 0

    def read(self, count=1):
        _require(
            0 <= count <= 32 and self.position + count <= len(self.data) * 8,
            "truncated H264 header",
        )
        value = 0
        for _ in range(count):
            byte, offset = divmod(self.position, 8)
            value = (value << 1) | ((self.data[byte] >> (7 - offset)) & 1)
            self.position += 1
        return value

    def ue(self, maximum=65535):
        zeros = 0
        while self.read() == 0:
            zeros += 1
            _require(zeros <= 24, "unbounded H264 header integer")
        value = (1 << zeros) - 1 + self.read(zeros)
        _require(value <= maximum, "H264 header integer exceeds limit")
        return value


def _hrd(bits):
    count = bits.ue(31) + 1
    bits.read(8)
    for _ in range(count):
        bits.ue()
        bits.ue()
        bits.read()
    bits.read(20)


def _vui(bits):
    if bits.read():  # aspect_ratio_info_present
        if bits.read(8) == 255:
            _require(
                bits.read(16) > 0 and bits.read(16) > 0, "invalid H264 aspect ratio"
            )
    if bits.read():
        bits.read()
    if bits.read():  # video signal type
        bits.read(4)
        if bits.read():
            bits.read(24)
    if bits.read():
        bits.ue(5)
        bits.ue(5)
    if bits.read():
        units, scale = bits.read(32), bits.read(32)
        _require(units > 0 and 0 < scale <= units * 480, "invalid H264 timing")
        bits.read()
    nal_hrd = bits.read()
    if nal_hrd:
        _hrd(bits)
    vcl_hrd = bits.read()
    if vcl_hrd:
        _hrd(bits)
    if nal_hrd or vcl_hrd:
        bits.read()
    _require(bits.read() == 0, "H264 picture structure extension is unsupported")
    _require(bits.read() == 1, "H264 bitstream buffer restriction is required")
    bits.read()
    for _ in range(4):
        bits.ue(16)
    _require(bits.ue(16) == 0, "reordered H264 frames are unsupported")
    bits.ue(4)  # max_dec_frame_buffering, before any decoder allocation


@dataclass(frozen=True, slots=True)
class H264Header:
    width: int
    height: int
    coded_width: int
    coded_height: int
    frame_count: int
    reference_frames: int
    profile: int


def _sps(nal, max_side, max_image_pixels):
    _require(len(nal) <= 4096, "H264 parameter set exceeds limit", 413)
    bits = _Bits(nal[1:])
    profile = bits.read(8)
    _require(profile in (66, 77), "H264 requires 8-bit 420 baseline or main")
    bits.read(8)
    _require(0 < bits.read(8) <= 51, "unsupported H264 level")
    sps_id = bits.ue(31)
    frame_num_bits = bits.ue(12) + 4
    order = bits.ue(2)
    poc_bits, delta_always_zero = 0, False
    if order == 0:
        poc_bits = bits.ue(12) + 4
    elif order == 1:
        delta_always_zero = bool(bits.read())
        bits.ue()
        bits.ue()
        for _ in range(bits.ue(16)):
            bits.ue()
    references = bits.ue(4)
    bits.read()
    coded_width = (bits.ue(79) + 1) * 16
    coded_height = (bits.ue(79) + 1) * 16
    _require(bits.read() == 1, "interlaced H264 is unsupported")
    bits.read()
    left = right = top = bottom = 0
    if bits.read():
        left, right, top, bottom = (bits.ue(640) for _ in range(4))
    width = coded_width - 2 * (left + right)
    height = coded_height - 2 * (top + bottom)
    _require(
        2 <= width <= max_side
        and 2 <= height <= max_side
        and coded_width <= ((max_side + 15) // 16) * 16
        and coded_height <= ((max_side + 15) // 16) * 16
        and coded_width * coded_height <= max_image_pixels,
        "H264 coded dimensions exceed limit",
        413,
    )
    _require(bits.read() == 1, "H264 VUI buffer limits are required")
    _vui(bits)
    return (
        H264Header(width, height, coded_width, coded_height, 0, references, profile),
        sps_id,
        (frame_num_bits, order, poc_bits, delta_always_zero),
    )


def _pps(nal, sps_id, profile):
    _require(len(nal) <= 4096, "H264 parameter set exceeds limit", 413)
    bits = _Bits(nal[1:])
    pps_id = bits.ue(255)
    _require(bits.ue(31) == sps_id, "unknown H264 parameter set reference")
    entropy = bits.read()
    _require(profile != 66 or entropy == 0, "invalid baseline H264 entropy coding")
    _require(bits.read() == 0 and bits.ue(7) == 0, "H264 slice groups are unsupported")
    bits.ue(3)
    bits.ue(3)
    return pps_id


def validate_h264(
    data: bytes,
    *,
    expected_frames: int,
    max_image_pixels: int,
    max_total_image_pixels: int,
    max_side: int = 1280,
    max_encoded_bytes: int = MAX_H264_BYTES,
) -> H264Header:
    """Read SPS/PPS/slice headers and limits without importing or invoking AV."""
    _require(
        type(expected_frames) is int and 1 <= expected_frames <= 16,
        "H264 requires one through sixteen frames",
    )
    _require(
        type(max_side) is int and 2 <= max_side <= 1280, "invalid H264 dimension limit"
    )
    _require(
        type(max_image_pixels) is int
        and max_image_pixels > 0
        and type(max_total_image_pixels) is int
        and max_total_image_pixels > 0,
        "invalid H264 pixel limits",
    )
    _require(
        isinstance(data, bytes)
        and 0 < len(data) <= min(max_encoded_bytes, MAX_H264_BYTES),
        "H264 encoded size exceeds limit",
        413,
    )
    starts = []
    for match in re.finditer(b"\x00\x00(?:\x00)?\x01", data):
        starts.append((match.start(), match.end()))
        _require(len(starts) <= 256, "H264 NAL count exceeds limit", 413)
    _require(starts and starts[0][0] == 0, "H264 Annex-B start code missing")
    header = sps_id = pps_id = None
    syntax = picture_key = None
    last_mb = -1
    picture_count = slice_count = 0
    first_picture = True
    view = memoryview(data)
    for index, (_, start) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(data)
        nal = view[start:end]
        _require(len(nal) >= 2 and not nal[0] & 128, "invalid H264 NAL header")
        kind = nal[0] & 31
        if kind == 9:
            _require(
                len(nal) <= 16 and (picture_count == 0 or slice_count > 0),
                "empty H264 access unit",
            )
            _require(
                _Bits(nal[1:]).read(3) in (0, 1), "H264 B pictures are unsupported"
            )
            picture_count += 1
            _require(picture_count <= expected_frames, "extra H264 pictures")
            slice_count = 0
            picture_key, last_mb = None, -1
            first_picture = picture_count == 1
        elif kind == 7:
            _require(
                header is None and picture_count == 1 and slice_count == 0,
                "H264 parameter sets changed within batch",
            )
            header, sps_id, syntax = _sps(nal, max_side, max_image_pixels)
            _require(
                header.coded_width * header.coded_height * expected_frames
                <= max_total_image_pixels,
                "H264 total decoded pixel limit exceeded",
                413,
            )
        elif kind == 8:
            _require(
                header is not None
                and pps_id is None
                and picture_count == 1
                and slice_count == 0,
                "invalid H264 picture parameter set",
            )
            pps_id = _pps(nal, sps_id, header.profile)
        elif kind in (1, 5):
            _require(
                header is not None and pps_id is not None and picture_count > 0,
                "H264 picture lacks parameter sets",
            )
            _require(not first_picture or kind == 5, "H264 batch must begin with IDR")
            bits = _Bits(nal[1:129])
            first_mb = bits.ue(header.coded_width * header.coded_height // 256 - 1)
            _require(
                (slice_count == 0 and first_mb == 0)
                or (slice_count > 0 and first_mb > last_mb),
                "H264 access unit hides extra or reordered pictures",
            )
            _require(bits.ue(9) in (0, 2, 5, 7), "H264 requires I/P pictures")
            _require(bits.ue(255) == pps_id, "unknown H264 picture parameter set")
            frame_num_bits, order, poc_bits, delta_always_zero = syntax
            key = (bits.read(frame_num_bits), kind, bool(nal[0] & 96))
            if kind == 5:
                key += (bits.ue(65535),)
            if order == 0:
                key += (bits.read(poc_bits),)
            elif order == 1 and not delta_always_zero:
                key += (bits.ue(),)
            _require(
                picture_key is None or picture_key == key,
                "H264 slices in one access unit identify different pictures",
            )
            picture_key, last_mb = key, first_mb
            slice_count += 1
        elif kind == 6:
            _require(
                picture_count == 1 and slice_count == 0 and len(nal) <= 4096,
                "unsupported H264 supplemental metadata",
            )
        else:
            raise H264TransportError("unsupported H264 NAL type")
    _require(
        header is not None
        and pps_id is not None
        and slice_count > 0
        and picture_count == expected_frames,
        "incomplete H264 picture sequence",
    )
    return replace(header, frame_count=picture_count)


def decode_h264_batch(
    data: bytes,
    *,
    expected_frames: int,
    max_image_pixels: int,
    max_total_image_pixels: int,
    max_side: int = 1280,
    max_encoded_bytes: int = MAX_H264_BYTES,
    check: Callable[[], None] | None = None,
) -> list:
    header = validate_h264(
        data,
        expected_frames=expected_frames,
        max_image_pixels=max_image_pixels,
        max_total_image_pixels=max_total_image_pixels,
        max_side=max_side,
        max_encoded_bytes=max_encoded_bytes,
    )
    if check:
        check()
    import av

    images = []
    try:
        with av.open(
            io.BytesIO(data), format="h264", options={"err_detect": "explode"}
        ) as container:
            _require(
                len(container.streams) == 1 and len(container.streams.video) == 1,
                "invalid H264 elementary stream",
            )
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            stream.thread_count = 2
            stream.codec_context.options = {"err_detect": "explode"}
            for frame in container.decode(video=0):
                if check:
                    check()
                _require(
                    len(images) < expected_frames
                    and not frame.is_corrupt
                    and (frame.width, frame.height) == (header.width, header.height)
                    and frame.pict_type != av.video.frame.PictureType.B,
                    "H264 decoded source contract mismatch",
                )
                images.append(frame.to_image())
        _require(len(images) == expected_frames, "incomplete H264 EOF flush")
        if check:
            check()
        return images
    except BaseException as exc:
        for image in images:
            image.close()
        if isinstance(exc, av.FFmpegError):
            raise H264TransportError("invalid H264 encoded pictures") from None
        raise
