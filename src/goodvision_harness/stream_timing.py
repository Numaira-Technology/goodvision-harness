"""Source-clock sampling without cumulative drift from network arrival jitter."""

from __future__ import annotations

import math


class SourceCadence:
    """Keep media time across reconnects; use arrival time only without valid PTS."""

    def __init__(self, fps: float):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        self.period = 1.0 / fps
        self.origin: float | None = None
        self.pts_origin: float | None = None
        self.pts_offset = 0.0
        self.last_pts: float | None = None
        self.last_timestamp = -self.period
        self.next_sample = 0.0
        self.discontinuities = 0

    def reconnect(self) -> None:
        self.pts_origin = self.last_pts = None

    def select(self, now: float, pts: float | None) -> tuple[float, bool]:
        if self.origin is None:
            self.origin = now
        arrival = now - self.origin
        if pts is not None and math.isfinite(pts):
            if self.last_pts is not None and pts < self.last_pts - 1e-6:
                self.pts_origin = None
                self.discontinuities += 1
            if self.pts_origin is None:
                self.pts_origin = pts
                self.pts_offset = max(arrival, self.last_timestamp + self.period)
            timestamp = self.pts_offset + pts - self.pts_origin
            self.last_pts = pts
        else:
            timestamp = max(arrival, self.last_timestamp)
        if timestamp <= self.last_timestamp + 1e-9:
            return timestamp, False
        self.last_timestamp = timestamp
        # Small rounding allowance for millisecond PTS; never reset the next
        # deadline to this frame's arrival time (which halves jittery 15 FPS).
        tolerance = min(0.001, self.period * 0.02)
        selected = timestamp + tolerance >= self.next_sample
        if selected:
            slots = (
                math.floor((timestamp + tolerance - self.next_sample) / self.period) + 1
            )
            self.next_sample += slots * self.period
        return timestamp, selected
