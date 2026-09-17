"""Per-session analysis diagnostics: what was analyzed, when, and why it failed.

The goal is to tell a missed alarm apart from a system problem: whether the
action was ever sampled, whether a request failed (authentication, timeout,
parse, ...), and how long a frame took from capture to a published result.
Everything is bounded and in memory; no provider text is stored. Model-input
JPEGs are kept only for the last few requests so an operator can see exactly
what the analysis engine received.
"""

from __future__ import annotations

import functools
import itertools
import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .sliding_window import FrameWindow

FAILURE_CATEGORIES = (
    "auth",
    "rate_limited",
    "timeout",
    "connection",
    "server_error",
    "http_error",
    "bad_response",
    "parse",
    "input",
    "internal",
    "unknown",
)
_MAX_CAPTURED_FAILURES = 16
_MAX_CAPTURED_INPUTS = 4


# ------------------------------------------------------------ call capture

@dataclass(slots=True)
class CallCapture:
    """Facts noted by a client while one pipeline call runs on this thread."""

    failures: list[dict[str, Any]] = field(default_factory=list)
    model_inputs: list[tuple[str, FrameWindow]] = field(default_factory=list)
    # Branches of one call may run on their own threads; see bind_capture.
    lock: threading.Lock = field(default_factory=threading.Lock)


_local = threading.local()


@contextmanager
def capture_call() -> Iterator[CallCapture]:
    """Collect failures and model inputs noted by code running in this call."""

    previous = getattr(_local, "capture", None)
    capture = CallCapture()
    _local.capture = capture
    try:
        yield capture
    finally:
        _local.capture = previous


def bind_capture(function: Callable[..., Any]) -> Callable[..., Any]:
    """Let ``function`` record into this call's capture from another thread.

    A branch handed to a nested executor would otherwise record nothing,
    because the capture is thread-local, and its failure would be reported
    only as a generic branch error.
    """

    capture = getattr(_local, "capture", None)
    if capture is None:
        return function

    @functools.wraps(function)
    def run(*args: Any, **kwargs: Any) -> Any:
        previous = getattr(_local, "capture", None)
        _local.capture = capture
        try:
            return function(*args, **kwargs)
        finally:
            _local.capture = previous

    return run


def note_failure(
    *, branch: str, stage: str, code: str, http_status: int | None = None,
) -> None:
    capture = getattr(_local, "capture", None)
    if capture is None:
        return
    with capture.lock:
        if len(capture.failures) >= _MAX_CAPTURED_FAILURES:
            return
        capture.failures.append({
            "branch": str(branch),
            "stage": str(stage),
            "code": str(code),
            "http_status": http_status if isinstance(http_status, int) else None,
        })


def note_model_input(role: str, window: FrameWindow) -> None:
    """Record the exact window (after any re-encoding) sent for analysis."""

    capture = getattr(_local, "capture", None)
    if capture is None:
        return
    with capture.lock:
        if len(capture.model_inputs) >= _MAX_CAPTURED_INPUTS:
            return
        capture.model_inputs.append((str(role), window))


def failure_category(failure: Mapping[str, Any] | None) -> str:
    """Map a bounded failure record to one operator-facing category."""

    if not failure:
        return "unknown"
    stage = failure.get("stage")
    code = failure.get("code")
    status = failure.get("http_status")
    if stage == "classification":
        return "parse"
    if stage == "input":
        return "input"
    if stage == "merge":
        return "internal"
    if code == "timeout":
        return "timeout"
    if code == "connection":
        return "connection"
    if code == "http_error":
        if status in {401, 403}:
            return "auth"
        if status == 429:
            return "rate_limited"
        if isinstance(status, int) and status >= 500:
            return "server_error"
        return "http_error"
    if code in {"invalid_response", "empty_response"}:
        return "bad_response"
    # A branch reports its own error, so its code is classified above and only
    # a code that says nothing is left as an internal problem.
    if stage == "branch":
        return "internal"
    return "unknown"


# ---------------------------------------------------------------- summary

def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[rank], 4)


def _latency_summary(values: deque[float]) -> dict[str, Any]:
    data = list(values)
    return {
        "count": len(data),
        "last": round(data[-1], 4) if data else None,
        "p50": _percentile(data, 0.5),
        "p95": _percentile(data, 0.95),
        "max": round(max(data), 4) if data else None,
    }


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@dataclass(slots=True)
class _Pending:
    submitted_at: float
    capture: CallCapture | None = None


class _BranchStats:
    def __init__(self, *, history: int) -> None:
        self.submitted = 0
        self.dropped = 0
        self.succeeded = 0
        self.failed = 0
        self.failures: Counter[str] = Counter()
        self.parse_failures: Counter[str] = Counter()
        self.last_failure: dict[str, Any] | None = None
        self.success_times: deque[float] = deque(maxlen=history)
        self.analyzed_timestamps: deque[float] = deque(maxlen=history)
        # Once old analyzed timestamps are evicted, the session start no longer
        # bounds the retained history and must not be used as a gap endpoint.
        self.analyzed_history_truncated = False
        self.first_source_timestamp: float | None = None
        self.newest_source_timestamp: float | None = None
        self.end_to_end: deque[float] = deque(maxlen=history)
        self.queue_wait: deque[float] = deque(maxlen=history)
        self.model_latency: deque[float] = deque(maxlen=history)
        self.pending: dict[int, _Pending] = {}

    def see_source(self, timestamp: float) -> None:
        if self.first_source_timestamp is None:
            self.first_source_timestamp = timestamp
        newest = self.newest_source_timestamp
        if newest is None or timestamp > newest:
            self.newest_source_timestamp = timestamp


class SessionDiagnostics:
    """Thread-safe, bounded diagnostics for one streaming session."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        rate_window_seconds: float = 10.0,
        history: int = 512,
        model_input_history: int = 8,
    ) -> None:
        if rate_window_seconds <= 0:
            raise ValueError("rate_window_seconds must be positive")
        self._clock = clock
        self._rate_window = float(rate_window_seconds)
        self._history = int(history)
        self._lock = threading.Lock()
        self._started = clock()
        self._branches: dict[str, _BranchStats] = {}
        self._inputs: deque[dict[str, Any]] = deque(maxlen=model_input_history)
        self._input_ids = itertools.count(1)

    def _branch(self, key: str) -> _BranchStats:
        stats = self._branches.get(key)
        if stats is None:
            stats = self._branches[key] = _BranchStats(history=self._history)
        return stats

    # ----------------------------------------------------------- recording

    def note_submitted(self, key: str, window: FrameWindow) -> None:
        now = self._clock()
        with self._lock:
            stats = self._branch(key)
            stats.submitted += 1
            stats.see_source(window.end_timestamp)
            stats.pending[window.index] = _Pending(submitted_at=now)
            # Bound pending state if completions were never reported.
            while len(stats.pending) > self._history:
                stats.pending.pop(next(iter(stats.pending)))

    def note_dropped(self, key: str, window: FrameWindow) -> None:
        with self._lock:
            stats = self._branch(key)
            stats.dropped += 1
            stats.see_source(window.end_timestamp)

    def note_call(
        self, key: str, window: FrameWindow, capture: CallCapture,
    ) -> None:
        """Store what the worker observed; called before the result is applied."""

        inputs = capture.model_inputs or [(key, window)]
        with self._lock:
            pending = self._branch(key).pending.get(window.index)
            if pending is not None:
                pending.capture = capture
            for role, sent in inputs:
                frames = [
                    {
                        "sample_index": frame.sample_index,
                        "source_frame_index": frame.source_frame_index,
                        "timestamp_seconds": round(frame.timestamp_seconds, 4),
                        "width": frame.width,
                        "height": frame.height,
                        "bytes": len(frame.jpeg_bytes),
                        "jpeg": frame.jpeg_bytes,
                    }
                    for frame in sent.frames
                ]
                newest = window.newest_frame
                source_width, source_height = newest.original_dimensions
                self._inputs.append({
                    "id": next(self._input_ids),
                    "branch": key,
                    "role": role,
                    "window_index": window.index,
                    "sent_at": _utc_now(),
                    "source_width": source_width,
                    "source_height": source_height,
                    "frames": frames,
                    "outcome": "pending",
                })

    def note_completed(
        self,
        key: str,
        window: FrameWindow,
        *,
        ok: bool,
        model_latency_seconds: float | None = None,
        classification_category: str | None = None,
        detected_labels: tuple[str, ...] = (),
        internal_error: bool = False,
    ) -> None:
        now = self._clock()
        newest = window.newest_frame
        captured = getattr(newest, "captured_monotonic", None)
        with self._lock:
            stats = self._branch(key)
            pending = stats.pending.pop(window.index, None)
            capture = pending.capture if pending is not None else None
            failures = list(capture.failures) if capture is not None else []
            category: str | None = None
            if ok:
                stats.succeeded += 1
                stats.success_times.append(now)
                history = stats.analyzed_timestamps
                if len(history) == history.maxlen:
                    stats.analyzed_history_truncated = True
                history.append(window.end_timestamp)
                if captured is not None:
                    stats.end_to_end.append(max(0.0, now - captured))
                    if pending is not None:
                        stats.queue_wait.append(
                            max(0.0, pending.submitted_at - captured)
                        )
                if model_latency_seconds is not None and model_latency_seconds > 0:
                    stats.model_latency.append(float(model_latency_seconds))
            else:
                stats.failed += 1
                if failures:
                    # A captured cause (401, timeout, ...) explains the raised
                    # exception better than the fact that it was raised.
                    category = failure_category(failures[0])
                elif internal_error:
                    category = "internal"
                else:
                    category = "parse" if classification_category else "unknown"
                stats.failures[category] += 1
                if category == "parse" and classification_category:
                    stats.parse_failures[classification_category] += 1
                first = failures[0] if failures else {}
                stats.last_failure = {
                    "category": category,
                    "stage": first.get("stage"),
                    "code": first.get("code"),
                    "http_status": first.get("http_status"),
                    "source_timestamp_seconds": round(window.end_timestamp, 4),
                    "at": _utc_now(),
                }
            for entry in self._inputs:
                if entry["branch"] == key and entry["window_index"] == window.index:
                    entry["outcome"] = "ok" if ok else category
                    entry["detected_labels"] = list(detected_labels)
                    if model_latency_seconds is not None:
                        entry["model_latency_seconds"] = round(
                            float(model_latency_seconds), 4
                        )

    # ------------------------------------------------------------- reading

    def _branch_snapshot(self, stats: _BranchStats, now: float) -> dict[str, Any]:
        recent = [t for t in stats.success_times if now - t <= self._rate_window]
        elapsed = min(self._rate_window, max(now - self._started, 1e-6))
        analyzed = sorted(stats.analyzed_timestamps)
        longest_gap = None
        if stats.analyzed_history_truncated:
            points = analyzed
        elif stats.first_source_timestamp is not None:
            points = [stats.first_source_timestamp, *analyzed]
        else:
            points = []
        gaps = [b - a for a, b in itertools.pairwise(points)]
        if gaps:
            longest_gap = round(max(gaps), 4)
        current_gap = None
        if stats.newest_source_timestamp is not None:
            reference = analyzed[-1] if analyzed else stats.first_source_timestamp
            if reference is not None:
                current_gap = round(
                    max(0.0, stats.newest_source_timestamp - reference), 4
                )
        return {
            "submitted": stats.submitted,
            "dropped": stats.dropped,
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "in_flight": len(stats.pending),
            "analyzed_per_second": round(len(recent) / elapsed, 3),
            "rate_window_seconds": self._rate_window,
            "longest_unanalyzed_gap_seconds": longest_gap,
            # "session" spans the whole session; "recent" spans only the last
            # gap_history_size analyzed frames after older ones were evicted.
            "gap_scope": "recent" if stats.analyzed_history_truncated else "session",
            "gap_history_size": len(analyzed),
            "current_unanalyzed_gap_seconds": current_gap,
            "newest_analyzed_source_seconds": (
                round(analyzed[-1], 4) if analyzed else None
            ),
            "newest_sampled_source_seconds": (
                round(stats.newest_source_timestamp, 4)
                if stats.newest_source_timestamp is not None else None
            ),
            "failures": {
                name: stats.failures.get(name, 0) for name in FAILURE_CATEGORIES
            },
            "parse_failures": dict(stats.parse_failures),
            "last_failure": dict(stats.last_failure) if stats.last_failure else None,
            "latency_seconds": {
                "capture_to_result": _latency_summary(stats.end_to_end),
                "capture_to_submit": _latency_summary(stats.queue_wait),
                "model": _latency_summary(stats.model_latency),
            },
        }

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            return {
                "uptime_seconds": round(now - self._started, 3),
                "branches": {
                    key: self._branch_snapshot(stats, now)
                    for key, stats in self._branches.items()
                },
                "model_inputs": [
                    {
                        **{k: v for k, v in entry.items() if k != "frames"},
                        "frames": [
                            {k: v for k, v in frame.items() if k != "jpeg"}
                            for frame in entry["frames"]
                        ],
                    }
                    for entry in reversed(self._inputs)
                ],
            }

    def model_input_jpeg(self, input_id: int, frame: int = -1) -> bytes | None:
        with self._lock:
            for entry in self._inputs:
                if entry["id"] == input_id:
                    frames = entry["frames"]
                    try:
                        return frames[frame]["jpeg"]
                    except IndexError:
                        return None
        return None
