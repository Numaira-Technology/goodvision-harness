"""Per-receiver protection against late RTP rewinding aiortc's video buffer.

Keep aiortc's depayloading, NACK and jitter implementation. Discard packets older
than its consumed origin; only a new SSRC or confirmed sequence progress after
an outage may reset it. H264 recovery waits for an IDR after a buffer reset.
"""

from __future__ import annotations

import re
import time
from collections import deque


def _has_idr(data: bytes) -> bool:
    # depayload(H264) emits Annex B; both three/four byte start codes work.
    return any(part and part[0] & 31 == 5 for part in data.split(b"\x00\x00\x01")[1:])


class VideoJitterGuard:
    def __init__(
        self,
        buffer,
        *,
        h264_payload_types=(),
        on_reset=lambda: None,
        count=lambda name: None,
        clock=time.monotonic,
    ):
        self._buffer = buffer
        self._h264 = set(h264_payload_types)
        self._on_reset, self._count, self._clock = on_reset, count, clock
        self._ssrc = None
        self._retired_ssrcs = deque(maxlen=8)
        self._last_progress = clock()
        self._restart_candidate = None
        self._waiting_for_idr = False
        self._buffer_origin = None
        self._origin_started = clock()

    def _reset(self):
        self._buffer = type(self._buffer)(capacity=self._buffer.capacity, is_video=True)
        self._on_reset()
        self._restart_candidate = None
        self._waiting_for_idr = True
        self._buffer_origin = None
        self._origin_started = self._clock()
        self._count("rtp_source_resets")

    def add(self, packet):
        now = self._clock()
        reset = False
        if self._ssrc is not None and packet.ssrc != self._ssrc:
            if packet.ssrc in self._retired_ssrcs:
                self._count("rtp_retired_source_packets_discarded")
                return False, None
            self._retired_ssrcs.append(self._ssrc)
            self._reset()
            reset = True
        self._ssrc = packet.ssrc
        origin = self._buffer._origin
        if origin is not None and (packet.sequence_number - origin) & 65535 > 32767:
            candidate = self._restart_candidate
            # Active forward traffic wins over retransmissions, even if several
            # old packets are consecutive. A true same-SSRC restart can recover
            # after 1s without forward progress plus two progressing packets.
            if (
                now - self._last_progress >= 1.0
                and candidate is not None
                and now - candidate[2] <= 1.0
                and (packet.sequence_number - candidate[0]) & 65535 == 1
                and 0 <= (packet.timestamp - candidate[1]) & 0xFFFFFFFF < 0x80000000
            ):
                self._reset()
                reset = True
            else:
                self._count("rtp_late_packets_discarded")
                self._restart_candidate = (
                    packet.sequence_number,
                    packet.timestamp,
                    now,
                )
                return False, None
        self._last_progress = now
        self._restart_candidate = None
        # Extra capacity accommodates a large single keyframe; it must not let
        # one missing fragment queue seconds of following video. Give NACKs a
        # bounded 500ms reorder window, then request a fresh decoding reference.
        origin = self._buffer._origin
        if origin is not None and now - self._origin_started >= 0.5:
            first = self._buffer._packets[origin % self._buffer.capacity]
            if first is None or first.timestamp != packet.timestamp:
                self._buffer = type(self._buffer)(
                    capacity=self._buffer.capacity,
                    is_video=True,
                )
                self._buffer_origin = None
                self._count("rtp_reassembly_timeouts")
                reset = True
        pli, frame = self._buffer.add(packet)
        if self._buffer._origin != self._buffer_origin:
            self._buffer_origin = self._buffer._origin
            self._origin_started = now
        if pli or reset:
            self._waiting_for_idr = True
            self._count("rtp_keyframe_requests")
        if self._waiting_for_idr and packet.payload_type in self._h264 and frame:
            if not _has_idr(frame.data):
                self._count("rtp_frames_waiting_for_keyframe")
                frame = None
            else:
                self._waiting_for_idr = False
                self._count("rtp_keyframe_recoveries")
        return pli or reset, frame


def protect_receiver(receiver, answer_sdp, diagnostics):
    """Instance-scoped aiortc adapter; fail explicitly on incompatible internals."""
    from aiortc.rtcrtpreceiver import TimestampMapper

    attribute = "_RTCRtpReceiver__jitter_buffer"
    original = getattr(receiver, attribute, None)
    if original is None or not hasattr(original, "_origin"):
        raise RuntimeError("unsupported WebRTC jitter buffer implementation")
    # Install before receiving. A high-resolution IDR can exceed aiortc's 128
    # packet default (our 2560x720 fixture reaches 342); truncating every IDR
    # would prevent recovery forever. Keep reassembly bounded, independently
    # from the UDP socket buffer. This does not change prefetch latency.
    original = type(original)(capacity=max(2048, original.capacity), is_video=True)
    payload_types = {
        int(match[1])
        for match in re.finditer(
            r"(?im)^a=rtpmap:(\d+) H264/90000\s*$",
            answer_sdp,
        )
    }

    def reset_timestamp():
        receiver._RTCRtpReceiver__timestamp_mapper = TimestampMapper()

    setattr(
        receiver,
        attribute,
        VideoJitterGuard(
            original,
            h264_payload_types=payload_types,
            on_reset=reset_timestamp,
            count=diagnostics.count,
        ),
    )
