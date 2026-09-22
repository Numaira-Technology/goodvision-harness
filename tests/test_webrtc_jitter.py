from types import SimpleNamespace

import pytest

pytest.importorskip("aiortc")
from aiortc.jitterbuffer import JitterBuffer
from aiortc.rtcrtpreceiver import TimestampMapper
from aiortc.rtp import RtpPacket

from goodvision_harness.webrtc_jitter import VideoJitterGuard, protect_receiver


def packet(seq, timestamp=None, *, ssrc=7, idr=False):
    p = RtpPacket(
        payload_type=101,
        sequence_number=seq,
        timestamp=seq * 3000 if timestamp is None else timestamp,
        ssrc=ssrc,
    )
    p._data = b"\x00\x00\x01" + (b"\x65" if idr else b"\x61") + b"test"
    return p


def guard(clock=lambda: 0.0, reset=lambda: None):
    return VideoJitterGuard(
        JitterBuffer(capacity=128, is_video=True),
        h264_payload_types={101},
        clock=clock,
        on_reset=reset,
    )


def test_late_missing_packets_cannot_rewind_or_jump_pts_by_thirteen_hours():
    g, mapper = guard(), TimestampMapper()
    for seq in range(110, 250):
        _, frame = g.add(packet(seq))
        if frame:
            mapper.map(frame.timestamp)
    before = g._buffer._origin
    for seq in (100, 101, 249):
        assert g.add(packet(seq))[1] is None
    assert g._buffer._origin == before
    _, frame = g.add(packet(250))
    assert mapper.map(frame.timestamp) < 5 * 90000


def test_sequence_wrap_and_legal_reordering_keep_complete_frames():
    g = guard()
    output = []
    for seq, stamp in (
        (65534, 0),
        (0, 6000),
        (65535, 3000),
        (1, 9000),
        (2, 12000),
        (3, 15000),
    ):
        pli, frame = g.add(packet(seq, stamp))
        assert not pli
        if frame:
            output.append(frame.timestamp)
    assert output == [0, 3000, 6000, 9000]


def test_overflow_waits_for_idr_before_resuming_decode():
    g = guard()
    g.add(packet(1))
    pli, _ = g.add(packet(200))
    assert pli
    assert g.add(packet(201))[1] is None
    assert g.add(packet(202, idr=True))[1] is None
    _, frame = g.add(packet(203))
    assert frame.timestamp == 202 * 3000
    assert g.add(packet(204))[1] is not None


def test_new_source_resets_and_retired_source_cannot_take_over_again():
    resets = []
    g = guard(reset=lambda: resets.append(True))
    g.add(packet(900))
    assert g.add(packet(100, ssrc=8, idr=True))[0]
    _, frame = g.add(packet(101, ssrc=8))
    assert frame is not None
    assert g.add(packet(901, ssrc=7)) == (False, None)
    assert len(resets) == 1


def test_same_ssrc_restart_requires_outage_and_progress_then_idr():
    now, resets = [0.0], []
    g = guard(clock=lambda: now[0], reset=lambda: resets.append(True))
    g.add(packet(900))
    # Consecutive retransmissions in active traffic are still old, not a restart.
    g.add(packet(100))
    g.add(packet(101))
    assert not resets
    now[0] = 2.0
    g.add(packet(10, 0))
    assert g.add(packet(11, 3000))[0]
    assert len(resets) == 1
    assert g.add(packet(12, 6000, idr=True))[1] is None
    assert g.add(packet(13, 9000))[1].timestamp == 6000


def test_installed_guard_preserves_large_keyframe_beyond_default_packet_capacity():
    receiver = SimpleNamespace(
        _RTCRtpReceiver__jitter_buffer=JitterBuffer(capacity=128, is_video=True),
    )
    protect_receiver(
        receiver,
        "a=rtpmap:101 H264/90000\r\n",
        SimpleNamespace(count=lambda name: None),
    )
    guard = receiver._RTCRtpReceiver__jitter_buffer
    assert guard._buffer.capacity == 2048
    expected = []
    for sequence in range(342):
        part = packet(sequence, 0, idr=sequence == 0)
        expected.append(part._data)
        assert guard.add(part) == (False, None)
    pli, frame = guard.add(packet(342, 3000))
    assert not pli
    assert frame.timestamp == 0 and frame.data == b"".join(expected)


def test_missing_fragment_has_bounded_reassembly_delay_and_waits_for_keyframe():
    now = [0.0]
    guard = VideoJitterGuard(
        JitterBuffer(capacity=2048, is_video=True),
        h264_payload_types={101},
        clock=lambda: now[0],
    )
    guard.add(packet(10, 0))
    guard.add(packet(12, 3000))  # Missing fragment 11 blocks the incomplete frame.
    now[0] = 0.4
    assert guard.add(packet(13, 6000)) == (False, None)
    now[0] = 0.51
    assert guard.add(packet(14, 9000)) == (True, None)
    assert guard.add(packet(15, 12000, idr=True))[1] is None
    assert guard.add(packet(16, 15000))[1].timestamp == 12000
    assert guard.add(packet(11, 0)) == (False, None)
