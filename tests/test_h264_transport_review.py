"""Adversarial preflight headers must be rejected before importing the decoder.

These tiny fixtures exercise the wire parser only; their slice payloads are not
real encoded images and must never reach AV or native inference.
"""

import builtins

import pytest

from workers.sam_scene.h264_transport import (
    H264TransportError,
    decode_h264_batch,
    validate_h264,
)


class Bits:
    def __init__(self):
        self.value = ""

    def put(self, value, count=1):
        self.value += f"{value:0{count}b}"
        return self

    def ue(self, value):
        binary = f"{value + 1:b}"
        self.value += "0" * (len(binary) - 1) + binary
        return self

    def nal(self, header):
        bits = self.value + "1"
        bits += "0" * (-len(bits) % 8)
        raw = int(bits, 2).to_bytes(len(bits) // 8, "big")
        escaped = bytearray()
        zeros = 0
        for byte in raw:
            if zeros >= 2 and byte <= 3:
                escaped.append(3)
                zeros = 0
            escaped.append(byte)
            zeros = zeros + 1 if byte == 0 else 0
        return b"\x00\x00\x00\x01" + bytes([header]) + escaped


def sps(*, vui=True, restriction=True, reorder=0, buffering=1):
    b = Bits().put(66, 8).put(192, 8).put(30, 8)
    b.ue(0).ue(0).ue(2).ue(1).put(0).ue(31).ue(23).put(1).put(1).put(0)
    b.put(int(vui))
    if vui:
        # Aspect, overscan, video signal, chroma location, timing, HRDs, pic_struct.
        for _ in range(8):
            b.put(0)
        b.put(int(restriction))
        if restriction:
            b.put(1).ue(0).ue(0).ue(8).ue(8).ue(reorder).ue(buffering)
    return b.nal(0x67)


def pps():
    return (
        Bits()
        .ue(0)
        .ue(0)
        .put(0)
        .put(0)
        .ue(0)
        .ue(0)
        .ue(0)
        .put(0)
        .put(0, 2)
        .ue(0)
        .ue(0)
        .ue(0)
        .put(1)
        .put(0)
        .put(0)
        .nal(0x68)
    )


def aud():
    return Bits().put(1, 3).nal(0x09)


def picture(frame_number, *, first_mb=0, idr=False):
    b = Bits().ue(first_mb).ue(2 if idr else 0).ue(0).put(frame_number, 4)
    if idr:
        b.ue(0)
    return b.nal(0x65 if idr else 0x41)


def stream(*, header=None, second=None):
    return (
        aud()
        + (header if header is not None else sps())
        + pps()
        + picture(0, idr=True)
        + aud()
        + (second if second is not None else picture(1))
    )


LIMITS = dict(
    expected_frames=2,
    max_image_pixels=512 * 384,
    max_total_image_pixels=2 * 512 * 384,
    max_side=512,
)


def test_control_header_advertises_two_bounded_ip_pictures():
    header = validate_h264(stream(), **LIMITS)
    assert (header.width, header.height, header.frame_count) == (512, 384, 2)


def assert_rejected_before_av(payload, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "av" or name.startswith("av."):
            raise AssertionError("unvalidated picture count or DPB reached AV")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(H264TransportError):
        decode_h264_batch(payload, **LIMITS)


@pytest.mark.parametrize(
    "hidden",
    [
        picture(1) + picture(2),
        picture(1) + picture(2, first_mb=384),
        picture(1) + picture(1),
        picture(1) + picture(1, first_mb=384) + picture(1, first_mb=128),
        picture(1, first_mb=384),
    ],
)
def test_aud_cannot_hide_extra_pictures_or_invalid_slice_order(hidden, monkeypatch):
    assert_rejected_before_av(stream(second=hidden), monkeypatch)


@pytest.mark.parametrize(
    "header",
    [
        sps(vui=False),
        sps(restriction=False),
        sps(reorder=1),
        sps(buffering=5),
    ],
)
def test_missing_or_excessive_decoder_buffer_bound_fails_before_av(header, monkeypatch):
    assert_rejected_before_av(stream(header=header), monkeypatch)


def test_two_slices_of_one_picture_keep_valid_source_count():
    header = validate_h264(
        stream(second=picture(1) + picture(1, first_mb=384)), **LIMITS
    )
    assert header.frame_count == 2
