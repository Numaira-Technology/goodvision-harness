"""Adapt the delayed display buffer from recent model latency."""

from __future__ import annotations

import math
from collections import deque


class LatencyTracker:
    """Bounded p95 latency window used to size the annotated-frame delay."""

    def __init__(
        self,
        *,
        window_size: int = 32,
        min_samples: int = 8,
        min_delay: float = 0.4,
        max_delay: float = 8.0,
        headroom: float = 0.2,
        min_change: float = 0.1,
    ) -> None:
        if window_size < 1 or min_samples < 1 or min_samples > window_size:
            raise ValueError("latency window bounds are invalid")
        self.window_size = window_size
        self.min_samples = min_samples
        self.min_delay = float(min_delay)
        self.max_delay = float(max_delay)
        self.headroom = float(headroom)
        self.min_change = float(min_change)
        self._samples: deque[float] = deque(maxlen=window_size)
        self._last_delay: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._last_delay = None

    def observe(self, latency: float) -> None:
        if not math.isfinite(latency) or latency < 0:
            return
        self._samples.append(float(latency))

    def recommended_delay(self) -> float | None:
        if len(self._samples) < self.min_samples:
            return None
        ordered = sorted(self._samples)
        rank = max(1, math.ceil(0.95 * len(ordered)))
        delay = ordered[rank - 1] + self.headroom
        delay = min(self.max_delay, max(self.min_delay, delay))
        delay = round(delay, 1)
        if (
            self._last_delay is not None
            and abs(delay - self._last_delay) < self.min_change
        ):
            return None
        self._last_delay = delay
        return delay
