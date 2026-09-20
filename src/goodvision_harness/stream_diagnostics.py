"""Bounded, image-free receive diagnostics; no endpoint or SDP is logged."""
from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter, deque
from typing import Any

from .latency_trace import log_timing, trace_id


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


class StreamDiagnostics:
    """One connection, bounded aggregates, shared by producer and consumer."""

    def __init__(self, connection_id: str, protocol: str, target_fps: float):
        self.connection_id = trace_id(connection_id)
        self.protocol = protocol
        self.target_fps = target_fps
        self._lock = threading.Lock()
        self._started = self._window_started = time.monotonic()
        self._last_received = self._last_pts = self._pts_origin = self._clock_origin = None
        self._totals = Counter()
        self._window = Counter()
        self._timings: dict[str, list] = {}
        self._recent = deque(maxlen=8)
        self._queued: dict[int, float] = {}
        self._rtp_previous: dict[int, tuple[int, int]] = {}
        self._rtp: list[dict] = []
        self._transport: list[dict] = []
        self._codecs: list[dict] = []
        self._clock_rate = None
        self._state = {}

    def _count(self, name: str, amount: int = 1):
        self._totals[name] += amount
        self._window[name] += amount

    def _timing(self, name: str, value: float):
        if not math.isfinite(value) or value < 0:
            return
        entry = self._timings.setdefault(name, [0, 0.0, 0.0, deque(maxlen=128)])
        entry[0] += 1
        entry[1] += value
        entry[2] = max(entry[2], value)
        entry[3].append(value)

    def count(self, name: str, amount: int = 1):
        with self._lock:
            self._count(name, amount)

    def timing(self, name: str, value: float):
        with self._lock:
            self._timing(name, value)

    def configure_decoder(self, backend, metadata, errors, discontinuities):
        with self._lock:
            self._state.update(
                decoder_backend=backend,
                source_video_format={key: metadata.get(key) for key in (
                    "codec_name", "pix_fmt", "color_space", "color_transfer",
                    "color_primaries", "color_range", "avg_frame_rate",
                )},
                normalized_to_sdr=metadata.get("color_transfer") in {
                    "smpte2084", "arib-std-b67",
                },
                decoder_error_chunks=errors,
                source_clock_discontinuities=discontinuities,
            )

    def configure_sdp(self, sdp: str):
        video = False
        codecs = []
        for line in sdp.splitlines():
            if line.startswith('m='):
                video = line.startswith('m=video ')
            match = re.fullmatch(r'a=rtpmap:(\d+) ([A-Za-z0-9_-]+)/(\d+)(?:/\d+)?', line.strip())
            if video and match:
                codecs.append(dict(payload_type=int(match[1]), codec=match[2], clock_rate=int(match[3])))
        with self._lock:
            self._codecs = codecs[:16]
            rates = {c['clock_rate'] for c in codecs if c['clock_rate'] > 0}
            self._clock_rate = next(iter(rates)) if len(rates) == 1 else None

    def received(self, frame: Any, now: float, wait: float, source_sequence: int | None):
        pts = getattr(frame, 'pts', None)
        base = getattr(frame, 'time_base', None)
        pts_seconds = _number(pts * base) if pts is not None and base is not None else None
        corrupt = getattr(frame, 'is_corrupt', None)
        key_frame = getattr(frame, 'key_frame', None)
        shape = getattr(frame, 'shape', ())
        width = getattr(frame, 'width', shape[1] if len(shape) >= 2 else None)
        height = getattr(frame, 'height', shape[0] if len(shape) >= 2 else None)
        with self._lock:
            self._count('decoded_frames')
            self._timing('receive_wait', wait)
            gap = None if self._last_received is None else now - self._last_received
            if gap is not None:
                self._timing('decoded_frame_gap', gap)
                if gap > .2:
                    self._count('decoded_gaps_over_200ms')
            self._last_received = now
            if corrupt is not None:
                self._count('corruption_flag_available_frames')
                if corrupt:
                    self._count('corrupt_frames')
            if key_frame:
                self._count('key_frames')
            if pts_seconds is not None:
                if self._last_pts is not None:
                    delta = pts_seconds - self._last_pts
                    if delta <= 0:
                        self._count('nonincreasing_pts_frames')
                        self._pts_origin = self._clock_origin = None
                    else:
                        self._timing('decoded_pts_gap', delta)
                if self._pts_origin is None:
                    self._pts_origin, self._clock_origin = pts_seconds, now
                self._state['receiver_pts_delay_growth_seconds'] = (
                    now - self._clock_origin - (pts_seconds - self._pts_origin))
                self._last_pts = pts_seconds
            self._recent.append(dict(source_sequence=source_sequence, pts=pts,
                                     pts_seconds=pts_seconds, corrupt=corrupt,
                                     key_frame=key_frame, receive_gap_seconds=gap))
            self._state.update(width=width, height=height, last_decoded_pts_seconds=pts_seconds)

    def enqueued(self, sequence: int):
        with self._lock:
            self._queued[sequence] = time.monotonic()
            while len(self._queued) > 8:
                self._queued.pop(next(iter(self._queued)))
            self._count('emitted_frames')
            self._state['last_source_sequence'] = sequence

    def dequeued(self, sequence: int, *, dropped: bool = False):
        with self._lock:
            at = self._queued.pop(sequence, None)
            self._count('handoff_queue_dropped_frames' if dropped else 'consumed_frames')
            if at is not None and not dropped:
                self._timing('handoff_queue_wait', time.monotonic() - at)

    def webrtc_stats(self, report: Any, pc: Any, loop_lag: float):
        with self._lock:
            self._timing('stats_loop_lag', max(0.0, loop_lag))
            self._state.update(connection_state=pc.connectionState,
                               ice_connection_state=pc.iceConnectionState)
            streams, transports = [], []
            for stat in report.values():
                if stat.type == 'inbound-rtp' and stat.kind == 'video':
                    received, lost, jitter = stat.packetsReceived, stat.packetsLost, stat.jitter
                    previous = self._rtp_previous.get(stat.ssrc)
                    reset = previous is not None and received < previous[0]
                    dr = received - previous[0] if previous and not reset else None
                    dl = lost - previous[1] if previous and not reset else None
                    streams.append(dict(ssrc=stat.ssrc, packets_received=received, packets_lost=lost,
                                        packets_received_delta=dr, packets_lost_delta=dl,
                                        counter_reset=reset, jitter_rtp_ticks=jitter,
                                        clock_rate=self._clock_rate,
                                        jitter_seconds=jitter / self._clock_rate if self._clock_rate else None))
                    self._rtp_previous[stat.ssrc] = (received, lost)
                elif stat.type == 'transport':
                    transports.append({key: getattr(stat, attr, None) for key, attr in (
                        ('packets_received', 'packetsReceived'), ('bytes_received', 'bytesReceived'),
                        ('packets_sent', 'packetsSent'), ('bytes_sent', 'bytesSent'), ('dtls_state', 'dtlsState'))})
            self._rtp, self._transport = streams[:8], transports[:8]
            while len(self._rtp_previous) > 8:
                self._rtp_previous.pop(next(iter(self._rtp_previous)))
            self._state['stats_observed_elapsed_seconds'] = time.monotonic() - self._started

    def emit(self, *, final: bool = False):
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._window_started
            if not final and elapsed < 5:
                return
            fields = dict(connection_id=self.connection_id, protocol=self.protocol,
                          final=final, target_fps=self.target_fps, window_seconds=elapsed,
                          elapsed_seconds=now-self._started, totals=dict(self._totals),
                          window=dict(self._window), recent_frames=list(self._recent),
                          window_decoded_fps=self._window['decoded_frames']/max(elapsed, 1e-6),
                          window_emitted_fps=self._window['emitted_frames']/max(elapsed, 1e-6),
                          last_decoded_age_seconds=None if self._last_received is None else now-self._last_received,
                          negotiated_video_codecs=self._codecs, inbound_rtp=self._rtp,
                          transport=self._transport, nack_pli_counters_available=False, **self._state)
            timings = {}
            for name, (count, total, maximum, samples) in self._timings.items():
                ordered = sorted(samples)
                timings[name] = dict(count=count, mean_seconds=total/count, max_seconds=maximum,
                                     sampled_p95_seconds=ordered[math.ceil(.95*len(ordered))-1])
            fields['timings'] = timings
            self._timings.clear()
            self._window.clear()
            self._recent.clear()
            self._window_started = now
        log_timing('network_receive_timing', **fields)
