"""Fast, fail-open rich object labels for the live video path.

The detector service is deliberately kept behind a very small HTTP contract.
Frames are sent as raw JPEG bytes while their stream identity travels in
headers.  Responses are parsed fail-closed so a delayed response can never be
mistaken for the current stream.  :class:`RichLabelCoordinator` provides the
opposite failure policy to callers: detector failures are recorded, but never
escape its background worker or interrupt the video path.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import httpx

from .contracts import (
    BBOX_COORDINATE_SYSTEM,
    NormalizedBBox,
    json_mapping_as_dict,
    readonly_json_mapping,
)

LOGGER = logging.getLogger(__name__)

STREAM_SESSION_HEADER = "X-GoodVision-Stream-Session-Id"
FRAME_SEQUENCE_HEADER = "X-GoodVision-Frame-Sequence"
TIMESTAMP_HEADER = "X-GoodVision-Timestamp-Seconds"


class RichLabelProtocolError(RuntimeError):
    """The detector returned a response outside the rich-label contract."""


@dataclass(frozen=True, slots=True)
class RichLabelObject:
    """One tracked object in source-normalized coordinates."""

    track_id: str
    label: str
    bbox: NormalizedBBox
    confidence: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _required_text(self.track_id, "track_id")
        _required_text(self.label, "label")
        if not isinstance(self.bbox, NormalizedBBox):
            raise TypeError("bbox must be a NormalizedBBox")
        if self.confidence is not None:
            _unit_interval(self.confidence, "confidence")
        object.__setattr__(
            self,
            "attributes",
            readonly_json_mapping(self.attributes, name="rich label attributes"),
        )

    def as_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "track_id": self.track_id,
            "label": self.label,
            "bbox": self.bbox.as_dict(),
            "attributes": json_mapping_as_dict(self.attributes),
        }
        if self.confidence is not None:
            record["confidence"] = round(float(self.confidence), 6)
        return record

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RichLabelObject:
        payload = _strict_mapping(value, "object")
        _exact_keys(
            payload,
            required={"track_id", "label", "bbox"},
            optional={"confidence", "attributes"},
            name="object",
        )
        bbox_payload = _strict_mapping(payload["bbox"], "object.bbox")
        _exact_keys(
            bbox_payload,
            required={"x1", "y1", "x2", "y2"},
            optional={"coordinate_system"},
            name="object.bbox",
        )
        coordinate_system = bbox_payload.get(
            "coordinate_system",
            BBOX_COORDINATE_SYSTEM,
        )
        if coordinate_system != BBOX_COORDINATE_SYSTEM:
            raise RichLabelProtocolError(
                "object.bbox has an unsupported coordinate system"
            )
        try:
            bbox = NormalizedBBox(
                _strict_number(bbox_payload["x1"], "object.bbox.x1"),
                _strict_number(bbox_payload["y1"], "object.bbox.y1"),
                _strict_number(bbox_payload["x2"], "object.bbox.x2"),
                _strict_number(bbox_payload["y2"], "object.bbox.y2"),
            )
            return cls(
                track_id=_strict_text(payload["track_id"], "object.track_id"),
                label=_strict_text(payload["label"], "object.label"),
                bbox=bbox,
                confidence=(
                    None
                    if payload.get("confidence") is None
                    else _strict_number(payload["confidence"], "object.confidence")
                ),
                attributes=_strict_mapping(
                    payload.get("attributes", {}),
                    "object.attributes",
                ),
            )
        except (TypeError, ValueError) as exc:
            raise RichLabelProtocolError(f"invalid rich-label object: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RichLabelSnapshot:
    """Detector output anchored to one source frame."""

    stream_session_id: str
    frame_sequence: int
    timestamp_seconds: float
    objects: tuple[RichLabelObject, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.stream_session_id, "stream_session_id")
        _frame_sequence(self.frame_sequence)
        _timestamp(self.timestamp_seconds)
        if not isinstance(self.objects, tuple):
            raise TypeError("objects must be a tuple")
        if any(not isinstance(item, RichLabelObject) for item in self.objects):
            raise TypeError("objects must contain RichLabelObject values")
        track_ids = [item.track_id for item in self.objects]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track_id values must be unique within a snapshot")

    def as_dict(self) -> dict[str, object]:
        return {
            "stream_session_id": self.stream_session_id,
            "frame_sequence": self.frame_sequence,
            "timestamp_seconds": float(self.timestamp_seconds),
            "objects": [item.as_dict() for item in self.objects],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RichLabelSnapshot:
        payload = _strict_mapping(value, "response")
        _exact_keys(
            payload,
            required={
                "stream_session_id",
                "frame_sequence",
                "timestamp_seconds",
                "objects",
            },
            optional=set(),
            name="response",
        )
        raw_objects = payload["objects"]
        if not isinstance(raw_objects, list):
            raise RichLabelProtocolError("response.objects must be an array")
        try:
            return cls(
                stream_session_id=_strict_text(
                    payload["stream_session_id"],
                    "response.stream_session_id",
                ),
                frame_sequence=_strict_integer(
                    payload["frame_sequence"],
                    "response.frame_sequence",
                ),
                timestamp_seconds=_strict_number(
                    payload["timestamp_seconds"],
                    "response.timestamp_seconds",
                ),
                objects=tuple(RichLabelObject.from_dict(item) for item in raw_objects),
            )
        except RichLabelProtocolError:
            raise
        except (TypeError, ValueError) as exc:
            raise RichLabelProtocolError(f"invalid rich-label response: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RichLabelSelection:
    """Rich labels selected for a display frame."""

    snapshot: RichLabelSnapshot | None
    mode: str
    left_frame_sequence: int | None = None
    right_frame_sequence: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"none", "exact", "held", "interpolated"}:
            raise ValueError("unsupported rich-label selection mode")
        if self.snapshot is None and self.mode != "none":
            raise ValueError("a non-empty selection requires a snapshot")

    @property
    def objects(self) -> tuple[RichLabelObject, ...]:
        """The selected objects, convenient for renderers."""

        return self.snapshot.objects if self.snapshot is not None else ()


class RichLabelClient(Protocol):
    """Minimal detector interface consumed by the coordinator."""

    def infer(
        self,
        jpeg_bytes: bytes,
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
    ) -> RichLabelSnapshot: ...


class DeferredRichLabelHistory:
    """Read history after GPU warmup without resetting a face session's tracks."""

    def __init__(
        self, coordinator: Callable[[], RichLabelCoordinator | None],
    ) -> None:
        self._coordinator = coordinator

    def history(self) -> tuple[RichLabelSnapshot, ...]:
        coordinator = self._coordinator()
        return coordinator.history() if coordinator is not None else ()


class RichLabelHTTPClient:
    """Synchronous raw-JPEG client for a PaddleDetection adapter service."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        timeout_seconds: float = 2.0,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint_url = _required_text(endpoint_url, "endpoint_url")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self._headers = _http_headers(headers)
        self._client = httpx.Client(
            timeout=float(timeout_seconds),
            transport=transport,
        )
        self._lock = threading.Lock()
        self._closed = False

    def gpu_ready(self) -> bool:
        """Require the remote worker to confirm usable GPU inference.

        The worker may live on another machine. Host graphics adapters or an
        enabled configuration flag cannot establish its inference capability.
        Old, unreachable, CPU-only and not-yet-ready workers remain disabled.
        """

        health_url = httpx.URL(self.endpoint_url).copy_with(
            path="/health/ready", query=None, fragment=None,
        )
        try:
            with self._lock:
                if self._closed:
                    return False
                response = self._client.get(
                    health_url, headers=self._headers, timeout=2.0,
                )
            response.raise_for_status()
            payload = response.json()
            return (
                isinstance(payload, dict)
                and payload.get("ready") is True
                and payload.get("gpu_ready") is True
            )
        except (httpx.HTTPError, ValueError):
            return False

    def infer(
        self,
        jpeg_bytes: bytes,
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
    ) -> RichLabelSnapshot:
        if not isinstance(jpeg_bytes, bytes) or not jpeg_bytes:
            raise ValueError("jpeg_bytes must be non-empty bytes")
        session_id = _required_text(stream_session_id, "stream_session_id")
        sequence = _frame_sequence(frame_sequence)
        timestamp = _timestamp(timestamp_seconds)
        request_headers = dict(self._headers)
        request_headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "image/jpeg",
                STREAM_SESSION_HEADER: session_id,
                FRAME_SEQUENCE_HEADER: str(sequence),
                TIMESTAMP_HEADER: _metadata_float(timestamp),
            }
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("rich-label HTTP client is closed")
            response = self._client.post(
                self.endpoint_url,
                content=jpeg_bytes,
                headers=request_headers,
            )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            raise RichLabelProtocolError("rich-label response is not application/json")
        snapshot = RichLabelSnapshot.from_dict(_strict_json(response.content))
        if snapshot.stream_session_id != session_id:
            raise RichLabelProtocolError(
                "rich-label response belongs to another session"
            )
        if snapshot.frame_sequence != sequence:
            raise RichLabelProtocolError("rich-label response belongs to another frame")
        if not math.isclose(
            snapshot.timestamp_seconds,
            timestamp,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RichLabelProtocolError(
                "rich-label response has a mismatched timestamp"
            )
        return snapshot

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._client.close()

    def __enter__(self) -> RichLabelHTTPClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class _PendingFrame:
    jpeg_bytes: bytes
    stream_session_id: str
    frame_sequence: int
    timestamp_seconds: float
    generation: int
    enqueued_at: float


class RichLabelCoordinator:
    """Single-worker, latest-only controller for optional rich detection.

    ``submit`` is non-blocking.  If inference is slower than the input stream,
    only the most recent waiting frame is retained.  All client errors are
    contained by the worker; callers continue receiving either the last valid
    state or an empty selection.
    """

    def __init__(
        self,
        client: RichLabelClient,
        *,
        max_fps: float = 5.0,
        history_size: int = 16,
        max_interpolation_gap_seconds: float = 1.0,
        max_hold_seconds: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(max_fps) or max_fps <= 0:
            raise ValueError("max_fps must be finite and positive")
        if isinstance(history_size, bool) or not isinstance(history_size, int):
            raise TypeError("history_size must be an integer")
        if history_size < 2:
            raise ValueError("history_size must be at least 2")
        if (
            not math.isfinite(max_interpolation_gap_seconds)
            or max_interpolation_gap_seconds <= 0
        ):
            raise ValueError("max_interpolation_gap_seconds must be positive")
        if not math.isfinite(max_hold_seconds) or max_hold_seconds <= 0:
            raise ValueError("max_hold_seconds must be positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.max_fps = float(max_fps)
        self.history_size = history_size
        self.max_interpolation_gap_seconds = float(
            max_interpolation_gap_seconds
        )
        self.max_hold_seconds = float(max_hold_seconds)
        self._client = client
        self._clock = clock
        self._condition = threading.Condition()
        self._generation = 0
        self._stream_session_id: str | None = None
        self._pending: _PendingFrame | None = None
        self._history: list[RichLabelSnapshot] = []
        self._successful_timings: deque[float] = deque(maxlen=10)
        self._last_success_completed_at: float | None = None
        self._last_started_at: float | None = None
        self._thread: threading.Thread | None = None
        self._in_flight = False
        self._closed = False
        self._client_closed = False
        self._client_close_lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "submitted": 0,
            "started": 0,
            "succeeded": 0,
            "failed": 0,
            "replaced": 0,
            "rejected_session": 0,
            "stale_results": 0,
            "resets": 0,
            "last_error": None,
            "last_latency_seconds": None,
            "last_queue_seconds": None,
            "last_total_latency_seconds": None,
            "last_attempt_frame_sequence": None,
            "last_attempt_succeeded": None,
            "last_frame_sequence": None,
        }

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("rich-label coordinator is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name="rich-label-worker",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        jpeg_bytes: bytes,
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
    ) -> bool:
        """Queue a frame, replacing any older frame that has not started."""

        if not isinstance(jpeg_bytes, bytes) or not jpeg_bytes:
            raise ValueError("jpeg_bytes must be non-empty bytes")
        session_id = _required_text(stream_session_id, "stream_session_id")
        sequence = _frame_sequence(frame_sequence)
        timestamp = _timestamp(timestamp_seconds)
        with self._condition:
            if self._closed:
                return False
            if self._stream_session_id is None:
                self._stream_session_id = session_id
            elif session_id != self._stream_session_id:
                self._stats["rejected_session"] += 1
                return False
            if self._pending is not None:
                self._stats["replaced"] += 1
            self._pending = _PendingFrame(
                jpeg_bytes=jpeg_bytes,
                stream_session_id=session_id,
                frame_sequence=sequence,
                timestamp_seconds=timestamp,
                generation=self._generation,
                enqueued_at=self._clock(),
            )
            self._stats["submitted"] += 1
            self._ensure_started_locked()
            self._condition.notify_all()
            return True

    submit_frame = submit

    def reset(self, stream_session_id: str | None = None) -> None:
        """Clear labels and prevent an in-flight old-session result from leaking."""

        session_id = (
            None
            if stream_session_id is None
            else _required_text(stream_session_id, "stream_session_id")
        )
        with self._condition:
            if self._closed:
                return
            self._generation += 1
            if self._pending is not None:
                self._stats["replaced"] += 1
            self._pending = None
            self._history.clear()
            self._successful_timings.clear()
            self._last_success_completed_at = None
            self._stream_session_id = session_id
            self._last_started_at = None
            self._stats["resets"] += 1
            self._stats["last_error"] = None
            self._stats["last_latency_seconds"] = None
            self._stats["last_queue_seconds"] = None
            self._stats["last_total_latency_seconds"] = None
            self._stats["last_attempt_frame_sequence"] = None
            self._stats["last_attempt_succeeded"] = None
            self._stats["last_frame_sequence"] = None
            self._condition.notify_all()

    def select(
        self,
        stream_session_id: str,
        timestamp_seconds: float,
        *,
        frame_sequence: int | None = None,
    ) -> RichLabelSelection:
        session_id = _required_text(stream_session_id, "stream_session_id")
        timestamp = _timestamp(timestamp_seconds)
        if frame_sequence is not None:
            _frame_sequence(frame_sequence)
        with self._condition:
            if session_id != self._stream_session_id:
                return RichLabelSelection(None, "none")
            history = tuple(self._history)
            display_hold_seconds = self._display_hold_seconds_locked()
        return select_rich_labels(
            history,
            timestamp,
            max_gap_seconds=self.max_interpolation_gap_seconds,
            max_hold_seconds=display_hold_seconds,
            frame_sequence=frame_sequence,
        )

    def latest(
        self,
        stream_session_id: str | None = None,
    ) -> RichLabelSnapshot | None:
        """Return the newest completed snapshot for the active session."""

        with self._condition:
            if (
                stream_session_id is not None
                and stream_session_id != self._stream_session_id
            ):
                return None
            return self._history[-1] if self._history else None

    def history(self) -> tuple[RichLabelSnapshot, ...]:
        with self._condition:
            return tuple(self._history)

    def stats(self) -> dict[str, Any]:
        with self._condition:
            payload = dict(self._stats)
            payload.update(
                {
                    "state": self._state_locked(),
                    "max_fps": self.max_fps,
                    "history_size": len(self._history),
                    "history_capacity": self.history_size,
                    "display_hold_seconds": round(
                        self._display_hold_seconds_locked(), 6
                    ),
                    "stream_session_id": self._stream_session_id,
                    "pending": self._pending is not None,
                    "in_flight": self._in_flight,
                }
            )
            return payload

    def wait_until_idle(self, timeout_seconds: float = 2.0) -> bool:
        """Wait for tests or orderly shutdown; normal video code need not block."""

        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and non-negative")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._pending is not None or self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and non-negative")
        with self._condition:
            if not self._closed:
                self._closed = True
                self._generation += 1
                self._pending = None
                self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout_seconds)
        if thread is None or not thread.is_alive():
            self._close_client()

    def _ensure_started_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="rich-label-worker",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    item = self._next_item_locked()
                    if item is None:
                        return
                    self._in_flight = True
                    self._last_started_at = self._clock()
                    self._stats["started"] += 1
                started_at = self._clock()
                snapshot: RichLabelSnapshot | None = None
                error: Exception | None = None
                try:
                    snapshot = self._client.infer(
                        item.jpeg_bytes,
                        stream_session_id=item.stream_session_id,
                        frame_sequence=item.frame_sequence,
                        timestamp_seconds=item.timestamp_seconds,
                    )
                    _matching_snapshot(snapshot, item)
                except Exception as exc:  # fail open at the video boundary
                    error = exc
                completed_at = self._clock()
                latency = max(0.0, completed_at - started_at)
                with self._condition:
                    self._in_flight = False
                    if (
                        item.generation != self._generation
                        or item.stream_session_id != self._stream_session_id
                        or self._closed
                    ):
                        self._stats["stale_results"] += 1
                    else:
                        # Local queue and completion timing excludes display
                        # buffering; it cannot isolate provider-side queue time.
                        self._stats["last_latency_seconds"] = round(latency, 6)
                        self._stats["last_queue_seconds"] = round(
                            max(0.0, started_at - item.enqueued_at), 6,
                        )
                        self._stats["last_total_latency_seconds"] = round(
                            max(0.0, completed_at - item.enqueued_at), 6,
                        )
                        self._stats["last_attempt_frame_sequence"] = item.frame_sequence
                        self._stats["last_attempt_succeeded"] = error is None
                        if error is not None:
                            LOGGER.warning("rich-label inference failed")
                            self._stats["failed"] += 1
                            self._stats["last_error"] = "rich-label inference failed"
                        else:
                            assert snapshot is not None
                            self._successful_timings.append(
                                max(0.0, completed_at - item.enqueued_at)
                            )
                            if self._last_success_completed_at is not None:
                                self._successful_timings.append(
                                    max(
                                        0.0,
                                        completed_at - self._last_success_completed_at,
                                    )
                                )
                            self._last_success_completed_at = completed_at
                            self._insert_snapshot_locked(snapshot)
                            self._stats["succeeded"] += 1
                            self._stats["last_error"] = None
                            self._stats["last_frame_sequence"] = snapshot.frame_sequence
                    self._condition.notify_all()
        finally:
            self._close_client()
            with self._condition:
                self._in_flight = False
                self._condition.notify_all()

    def _next_item_locked(self) -> _PendingFrame | None:
        interval = 1.0 / self.max_fps
        while True:
            while not self._closed and self._pending is None:
                self._condition.wait()
            if self._closed:
                return None
            if self._last_started_at is not None:
                remaining = interval - (self._clock() - self._last_started_at)
                if remaining > 0:
                    self._condition.wait(remaining)
                    continue
            item = self._pending
            self._pending = None
            assert item is not None
            return item

    def _insert_snapshot_locked(self, snapshot: RichLabelSnapshot) -> None:
        key = _snapshot_key(snapshot)
        keys = [_snapshot_key(item) for item in self._history]
        index = bisect_right(keys, key)
        if index > 0 and keys[index - 1] == key:
            self._history[index - 1] = snapshot
        else:
            self._history.insert(index, snapshot)
        if len(self._history) > self.history_size:
            del self._history[: len(self._history) - self.history_size]

    def _display_hold_seconds_locked(self) -> float:
        # CPU inference can take longer than the configured decorative hold.
        # Cover both the last result's age on arrival and the next update's
        # cadence. Only successful completions establish this display budget:
        # repeated errors cannot keep stale boxes alive indefinitely.
        # Raw history used for identity association is never extended here.
        observed = max(self._successful_timings, default=0.0)
        return min(
            max(15.0, self.max_hold_seconds),
            self.max_hold_seconds + 2.0 * observed,
        )

    def _state_locked(self) -> str:
        if self._closed:
            return "closed"
        if self._in_flight:
            return "processing"
        if self._stats["last_error"] is not None:
            return "degraded"
        if self._history:
            return "ready"
        return "idle"

    def _close_client(self) -> None:
        close = getattr(self._client, "close", None)
        if not callable(close):
            return
        with self._client_close_lock:
            if self._client_closed:
                return
            self._client_closed = True
            try:
                close()
            except Exception:
                LOGGER.warning("could not close rich-label client")


def select_rich_labels(
    snapshots: Sequence[RichLabelSnapshot],
    timestamp_seconds: float,
    *,
    max_gap_seconds: float,
    max_hold_seconds: float | None = None,
    frame_sequence: int | None = None,
) -> RichLabelSelection:
    """Select or interpolate labels using detector-provided stable track IDs.

    ``max_gap_seconds`` limits motion interpolation only. When the next
    keyframe is already known, a long gap holds the left state until that
    keyframe explicitly updates or clears it, matching the delayed event
    compositor and avoiding unannotated flashes. At the unknown history tail,
    ``max_hold_seconds`` remains bounded so a worker outage cannot leave stale
    decorative boxes on screen forever.
    """

    timestamp = _timestamp(timestamp_seconds)
    if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be finite and positive")
    hold_limit = (
        max_gap_seconds if max_hold_seconds is None else max_hold_seconds
    )
    if not math.isfinite(hold_limit) or hold_limit <= 0:
        raise ValueError("max_hold_seconds must be finite and positive")
    if frame_sequence is not None:
        _frame_sequence(frame_sequence)
    if not snapshots:
        return RichLabelSelection(None, "none")
    if any(not isinstance(item, RichLabelSnapshot) for item in snapshots):
        raise TypeError("snapshots must contain RichLabelSnapshot values")
    sessions = {item.stream_session_id for item in snapshots}
    if len(sessions) != 1:
        raise ValueError("snapshots must belong to one stream session")
    ordered = sorted(snapshots, key=_snapshot_key)
    keys = [_snapshot_key(item) for item in ordered]
    query_sequence = math.inf if frame_sequence is None else frame_sequence
    position = bisect_right(keys, (timestamp, query_sequence)) - 1
    if position < 0:
        return RichLabelSelection(None, "none")
    left = ordered[position]
    exact_sequence = frame_sequence is None or left.frame_sequence == frame_sequence
    if math.isclose(
        timestamp,
        left.timestamp_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ) and exact_sequence:
        return RichLabelSelection(
            left,
            "exact",
            left.frame_sequence,
            None,
        )
    right = ordered[position + 1] if position + 1 < len(ordered) else None
    hold_age = max(0.0, timestamp - left.timestamp_seconds)
    if right is None:
        if hold_age > hold_limit + 1e-9:
            return RichLabelSelection(None, "none")
        return RichLabelSelection(left, "held", left.frame_sequence, None)
    gap = right.timestamp_seconds - left.timestamp_seconds
    if gap <= 0 or gap > max_gap_seconds + 1e-9:
        # Long/invalid gaps are unsafe for geometry interpolation, but the
        # known right keyframe is authoritative at its own timestamp. Holding
        # until then prevents sparse inference from blinking the bbox off.
        return RichLabelSelection(left, "held", left.frame_sequence, None)
    alpha = max(0.0, min(1.0, (timestamp - left.timestamp_seconds) / gap))
    interpolated = _interpolate_snapshot(
        left,
        right,
        alpha,
        timestamp_seconds=timestamp,
        frame_sequence=frame_sequence,
    )
    return RichLabelSelection(
        interpolated,
        "interpolated",
        left.frame_sequence,
        right.frame_sequence,
    )


def _interpolate_snapshot(
    left: RichLabelSnapshot,
    right: RichLabelSnapshot,
    alpha: float,
    *,
    timestamp_seconds: float,
    frame_sequence: int | None,
) -> RichLabelSnapshot:
    right_by_track = {item.track_id: item for item in right.objects}
    objects: list[RichLabelObject] = []
    for item in left.objects:
        candidate = right_by_track.get(item.track_id)
        if candidate is None:
            objects.append(item)
            continue
        confidence = item.confidence
        if confidence is not None and candidate.confidence is not None:
            confidence = _lerp(confidence, candidate.confidence, alpha)
        objects.append(
            replace(
                item,
                bbox=NormalizedBBox(
                    _lerp(item.bbox.x1, candidate.bbox.x1, alpha),
                    _lerp(item.bbox.y1, candidate.bbox.y1, alpha),
                    _lerp(item.bbox.x2, candidate.bbox.x2, alpha),
                    _lerp(item.bbox.y2, candidate.bbox.y2, alpha),
                ),
                confidence=confidence,
            )
        )
    return RichLabelSnapshot(
        stream_session_id=left.stream_session_id,
        frame_sequence=(
            left.frame_sequence if frame_sequence is None else frame_sequence
        ),
        timestamp_seconds=timestamp_seconds,
        objects=tuple(objects),
    )


def _matching_snapshot(snapshot: object, item: _PendingFrame) -> None:
    if not isinstance(snapshot, RichLabelSnapshot):
        raise RichLabelProtocolError("rich-label client returned an invalid snapshot")
    if snapshot.stream_session_id != item.stream_session_id:
        raise RichLabelProtocolError("rich-label result belongs to another session")
    if snapshot.frame_sequence != item.frame_sequence:
        raise RichLabelProtocolError("rich-label result belongs to another frame")
    if not math.isclose(
        snapshot.timestamp_seconds,
        item.timestamp_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RichLabelProtocolError("rich-label result has a mismatched timestamp")


def _snapshot_key(snapshot: RichLabelSnapshot) -> tuple[float, int]:
    return snapshot.timestamp_seconds, snapshot.frame_sequence


def _lerp(left: float, right: float, alpha: float) -> float:
    return float(left) + ((float(right) - float(left)) * float(alpha))


def _strict_json(content: bytes) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid numeric constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key}")
            result[key] = value
        return result

    try:
        decoded = content.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RichLabelProtocolError("rich-label response is not strict JSON") from exc
    return _strict_mapping(value, "response")


def _strict_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RichLabelProtocolError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise RichLabelProtocolError(f"{name} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    name: str,
) -> None:
    keys = set(value)
    missing = required - keys
    unexpected = keys - required - optional
    if missing:
        rendered = ", ".join(sorted(missing))
        raise RichLabelProtocolError(f"{name} is missing fields: {rendered}")
    if unexpected:
        rendered = ", ".join(sorted(unexpected))
        raise RichLabelProtocolError(f"{name} has unexpected fields: {rendered}")


def _strict_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RichLabelProtocolError(f"{name} must be non-empty text")
    if value != value.strip():
        raise RichLabelProtocolError(f"{name} must not have surrounding whitespace")
    return value


def _strict_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RichLabelProtocolError(f"{name} must be an integer")
    return value


def _strict_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RichLabelProtocolError(f"{name} must be a number")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise RichLabelProtocolError(f"{name} must be finite")
    return rendered


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _frame_sequence(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("frame_sequence must be a non-negative integer")
    return value


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timestamp_seconds must be a finite non-negative number")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < 0:
        raise ValueError("timestamp_seconds must be a finite non-negative number")
    return rendered


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be between 0 and 1")
    rendered = float(value)
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return rendered


def _metadata_float(value: float) -> str:
    return format(float(value), ".17g")


def _http_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        raise TypeError("headers must be a mapping")
    rendered: dict[str, str] = {}
    reserved = {
        "accept",
        "content-type",
        STREAM_SESSION_HEADER.lower(),
        FRAME_SEQUENCE_HEADER.lower(),
        TIMESTAMP_HEADER.lower(),
    }
    for key, value in headers.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("header names must be non-empty text")
        if not isinstance(value, str):
            raise TypeError("header values must be text")
        if key.lower() in reserved:
            raise ValueError(f"header {key} is managed by the rich-label client")
        rendered[key] = value
    return rendered
