"""Optional continuous SAM video sessions with bounded source provenance.

The transport reuses the batch client's absolute deadline and cancellation loop.
Only frames acknowledged by this session can become delayed observations. An
uncertain append result invalidates the session; recovery requires an explicit
reset, never an automatic retry or continuation of an uncertain native epoch.
"""

from __future__ import annotations

import base64
import json
import math
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx

from .rich_labeling import RichLabelProtocolError, RichLabelSnapshot
from .video_tracking import (
    BACKEND,
    MAX_BATCH_JPEG_BYTES,
    VideoFrame,
    VideoTrackingCoordinator,
    VideoTrackingHTTPClient,
    _text,
    _validate_frames,
)

MAX_PENDING_SOURCES = 32
LIMIT_FIELDS = {
    "max_frames",
    "max_request_bytes",
    "max_jpeg_bytes",
    "max_total_jpeg_bytes",
    "max_image_pixels",
    "max_total_image_pixels",
    "inference_max_side",
    "max_objects",
    "request_timeout_seconds",
    "max_inflight_requests",
    "startup_frames",
    "max_append_frames",
    "history_frames",
    "max_session_frames",
    "max_cumulative_native_ids",
    "idle_seconds",
    "max_active_sessions",
}
IDENTITY_FIELDS = {
    "video_session_id",
    "stream_session_id",
    "predictor_epoch",
    "backend",
    "memory_scope",
    "state",
}
WATERMARK_FIELDS = {
    f"{name}_{suffix}"
    for name in ("accepted", "processed", "emitted")
    for suffix in ("frame_sequence", "timestamp_seconds")
}


def _require(value: bool, message: str) -> None:
    if not value:
        raise RichLabelProtocolError(message)


def _finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def continuous_source_history_seconds(fps: float) -> float:
    """Bound source retention for native lookahead and two remote requests.

    JPEG byte/frame caps still apply independently; this allowance never changes
    capture freshness, request deadlines, or the source times used as evidence.
    """
    if not _finite_number(fps) or fps < 0.5:
        raise ValueError("continuous source retention requires finite fps >= 0.5")
    return max(32.0, 32 / fps + 24.0)


def _limits(value: Any) -> dict[str, Any]:
    _require(
        isinstance(value, dict) and set(value) == LIMIT_FIELDS, "invalid session limits"
    )
    for key, number in value.items():
        _require(
            type(number)
            in (
                (int, float)
                if key in {"idle_seconds", "request_timeout_seconds"}
                else (int,)
            )
            and _finite_number(number)
            and number > 0,
            "invalid session limit value",
        )
    for key, expected in {
        "startup_frames": 16,
        "max_append_frames": 16,
        "history_frames": 32,
        "max_inflight_requests": 1,
        "max_active_sessions": 1,
    }.items():
        _require(value[key] == expected, "unsupported session limits")
    _require(
        value["max_session_frames"] <= 65536
        and value["max_cumulative_native_ids"] <= 2048,
        "unbounded native session limits",
    )
    return dict(value)


def _stamp(payload: Mapping[str, Any], prefix: str, *, nullable: bool = False):
    sequence = payload[f"{prefix}_frame_sequence"]
    timestamp = payload[f"{prefix}_timestamp_seconds"]
    if nullable and sequence is None and timestamp is None:
        return None
    _require(type(sequence) is int and sequence >= 0, "invalid source watermark")
    _require(
        _finite_number(timestamp) and timestamp >= 0,
        "invalid source watermark",
    )
    return sequence, timestamp


def _same_stamp(left, right) -> bool:
    return (
        left == right
        if left is None or right is None
        else (
            type(left[0]) is int
            and type(right[0]) is int
            and type(left[1]) in (int, float)
            and type(right[1]) in (int, float)
            and left[0] == right[0]
            and math.isclose(left[1], right[1], rel_tol=0, abs_tol=1e-6)
        )
    )


@dataclass(frozen=True, slots=True)
class VideoSession:
    video_session_id: str
    stream_session_id: str
    predictor_epoch: str
    limits: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VideoSessionTrackingResult:
    video_session_id: str
    stream_session_id: str
    request_id: str
    predictor_epoch: str
    state: str
    frames: tuple[RichLabelSnapshot, ...]
    accepted_frame_sequence: int
    accepted_timestamp_seconds: float
    processed_frame_sequence: int
    processed_timestamp_seconds: float
    emitted_frame_sequence: int | None
    emitted_timestamp_seconds: float | None
    pending_frame_count: int
    memory_scope: str = "session"
    backend: str = BACKEND

    @property
    def completed_frame_sequence(self) -> int | None:
        return self.emitted_frame_sequence

    @property
    def completed_timestamp_seconds(self) -> float | None:
        return self.emitted_timestamp_seconds

    @property
    def batch_id(self) -> str:
        return self.request_id


class VideoSessionTrackingHTTPClient(VideoTrackingHTTPClient):
    """One native session, one HTTP operation, at most 32 delayed source stamps."""

    def __init__(self, endpoint: str, *, timeout_seconds: float = 75.0, transport=None):
        super().__init__(endpoint, timeout_seconds=timeout_seconds, transport=transport)
        base = httpx.URL(self.endpoint)
        self.endpoint = str(base.copy_with(path="/v1/video/sessions"))
        self._health = str(base.copy_with(path="/health/video/session"))
        self._session_operation = threading.Lock()
        self._session_state = threading.RLock()
        self._native_session: VideoSession | None = None
        self._state = "new"
        self._sources: tuple[tuple[int, float], ...] = ()
        self._accepted = self._emitted = None
        self._accepted_count = 0
        self._native_ids: dict[str, str] = {}

    def gpu_ready(self) -> bool:
        try:
            value = self._request(
                "GET",
                self._health,
                deadline=time.monotonic() + min(self._timeout_seconds, 5),
            )
            _require(
                set(value)
                == {
                    "enabled",
                    "ready",
                    "gpu_ready",
                    "backend",
                    "memory_scope",
                    "native_sam_video_tracking",
                    "sam3_commit",
                    "limits",
                    "error_code",
                    "active_session",
                },
                "invalid session health fields",
            )
            _limits(value["limits"])
            return (
                value["enabled"] is True
                and value["ready"] is True
                and value["gpu_ready"] is True
                and value["backend"] == BACKEND
                and value["memory_scope"] == "session"
                and value["native_sam_video_tracking"] is True
            )
        except (httpx.HTTPError, ValueError, RichLabelProtocolError):
            return False

    def _identity(self, payload, session):
        _require(
            all(
                payload[key] == expected
                for key, expected in {
                    "video_session_id": session.video_session_id,
                    "stream_session_id": session.stream_session_id,
                    "predictor_epoch": session.predictor_epoch,
                    "backend": BACKEND,
                    "memory_scope": "session",
                }.items()
            ),
            "native session identity mismatch",
        )

    def open_session(self, stream_session_id: str) -> VideoSession:
        _text(stream_session_id, "stream_session_id")
        if not self._session_operation.acquire(blocking=False):
            raise httpx.PoolTimeout("session operation already active")
        try:
            with self._session_state:
                _require(
                    self._native_session is None,
                    "discard previous session before opening",
                )

            def parse(payload):
                _require(
                    set(payload) == IDENTITY_FIELDS | {"limits"},
                    "invalid session creation fields",
                )
                _require(
                    payload["stream_session_id"] == stream_session_id
                    and payload["backend"] == BACKEND
                    and payload["memory_scope"] == "session"
                    and payload["state"] == "active",
                    "invalid session creation identity",
                )
                return VideoSession(
                    _text(payload["video_session_id"], "video_session_id"),
                    stream_session_id,
                    _text(payload["predictor_epoch"], "predictor_epoch"),
                    _limits(payload["limits"]),
                )

            result = self._request(
                "POST",
                self.endpoint,
                deadline=time.monotonic() + self._timeout_seconds,
                payload_factory=lambda: {"stream_session_id": stream_session_id},
                result_parser=parse,
            )
            with self._session_state:
                self._native_session = result
                self._state = "active"
                self._sources = ()
                self._accepted = self._emitted = None
                self._accepted_count = 0
                self._native_ids.clear()
            return result
        finally:
            self._session_operation.release()

    def _parse_append(self, payload, session, request_id, candidates, accepted, final):
        _require(
            set(payload)
            == IDENTITY_FIELDS
            | WATERMARK_FIELDS
            | {"request_id", "frames", "pending_frame_count"},
            "invalid append response fields",
        )
        self._identity(payload, session)
        _require(
            payload["request_id"] == request_id, "append request identity mismatch"
        )
        _require(
            payload["state"] == ("closed" if final else "active"),
            "invalid append state",
        )
        received = _stamp(payload, "accepted")
        processed = _stamp(payload, "processed")
        emitted = _stamp(payload, "emitted", nullable=True)
        _require(
            _same_stamp(received, accepted) and _same_stamp(processed, accepted),
            "append accepted/processed source mismatch",
        )
        records = payload["frames"]
        _require(
            isinstance(records, list) and len(records) <= len(candidates),
            "invalid delayed source count",
        )
        snapshots = tuple(RichLabelSnapshot.from_dict(row) for row in records)
        native_ids = dict(self._native_ids)
        for stamp, snapshot in zip(candidates, snapshots, strict=False):
            _require(
                snapshot.stream_session_id == session.stream_session_id
                and _same_stamp(
                    (snapshot.frame_sequence, snapshot.timestamp_seconds), stamp
                ),
                "delayed result is not the next accepted source frame",
            )
            _require(
                len(snapshot.objects) <= min(64, session.limits["max_objects"]),
                "too many native objects",
            )
            seen = set()
            for obj in snapshot.objects:
                native = obj.attributes.get("native_track")
                _require(
                    isinstance(native, Mapping)
                    and set(native) == {"backend", "epoch", "object_id"},
                    "invalid native provenance",
                )
                native_id = _text(native["object_id"], "native object ID")
                _require(
                    native["backend"] == BACKEND
                    and native["epoch"] == session.predictor_epoch
                    and obj.label == "person"
                    and obj.observation_state == "observed"
                    and _same_stamp(
                        (
                            obj.last_observed_frame_sequence,
                            obj.last_observed_timestamp_seconds,
                        ),
                        stamp,
                    ),
                    "invalid native observation source",
                )
                _require(native_id not in seen, "duplicate native object ID")
                seen.add(native_id)
                _require(
                    native_id not in native_ids
                    or native_ids[native_id] == obj.track_id,
                    "native track identity changed within session",
                )
                native_ids[native_id] = obj.track_id
        _require(
            len(native_ids) <= session.limits["max_cumulative_native_ids"],
            "native ID budget exceeded",
        )
        _require(
            len(set(native_ids.values())) == len(native_ids), "native track IDs collide"
        )
        expected_emitted = (
            candidates[len(snapshots) - 1] if snapshots else self._emitted
        )
        _require(_same_stamp(emitted, expected_emitted), "emitted watermark mismatch")
        remaining = candidates[len(snapshots) :]
        _require(
            type(payload["pending_frame_count"]) is int
            and payload["pending_frame_count"] == len(remaining),
            "pending source count mismatch",
        )
        _require(
            not final or (not remaining and _same_stamp(emitted, accepted)),
            "incomplete final source flush",
        )
        return (
            VideoSessionTrackingResult(
                session.video_session_id,
                session.stream_session_id,
                request_id,
                session.predictor_epoch,
                payload["state"],
                snapshots,
                *received,
                *processed,
                *(emitted if emitted is not None else (None, None)),
                len(remaining),
            ),
            remaining,
            native_ids,
        )

    def append(
        self,
        frames: Sequence[VideoFrame],
        *,
        stream_session_id: str,
        is_last_batch: bool = False,
        request_id: str | None = None,
    ) -> VideoSessionTrackingResult:
        if not self._session_operation.acquire(blocking=False):
            raise httpx.PoolTimeout("session operation already active")
        sent = False
        try:
            _require(
                isinstance(frames, Sequence) and len(frames) <= 16,
                "append requires at most sixteen source frames",
            )
            frames = tuple(frames)
            if frames:
                _validate_frames(frames)
            _require(type(is_last_batch) is bool, "final flag must be boolean")
            request_id = _text(request_id or uuid4().hex, "request_id")
            with self._session_state:
                session = self._native_session
                _require(
                    session is not None and self._state == "active",
                    "native session is not active",
                )
                _require(
                    stream_session_id == session.stream_session_id,
                    "source session mismatch",
                )
                _require(
                    bool(frames) or (is_last_batch and self._accepted is not None),
                    "empty append is not a final flush",
                )
                _require(
                    self._accepted is not None or is_last_batch or len(frames) == 16,
                    "initial nonfinal append requires sixteen frames",
                )
                if frames and self._accepted is not None:
                    _require(
                        frames[0].frame_sequence > self._accepted[0]
                        and frames[0].timestamp_seconds > self._accepted[1],
                        "append source order regressed",
                    )
                candidates = self._sources + tuple(
                    (f.frame_sequence, f.timestamp_seconds) for f in frames
                )
                _require(
                    len(candidates) <= MAX_PENDING_SOURCES,
                    "delayed source budget exceeded",
                )
                _require(
                    self._accepted_count + len(frames)
                    <= session.limits["max_session_frames"],
                    "session frame budget exceeded",
                )
                _require(
                    len(frames) <= session.limits["max_frames"]
                    and all(
                        len(f.jpeg_bytes) <= session.limits["max_jpeg_bytes"]
                        for f in frames
                    )
                    and sum(len(f.jpeg_bytes) for f in frames)
                    <= session.limits["max_total_jpeg_bytes"],
                    "append exceeds worker byte limits",
                )
                accepted = candidates[-1] if frames else self._accepted
            empty_payload = {
                "stream_session_id": stream_session_id,
                "request_id": request_id,
                "frames": [
                    {
                        "frame_sequence": f.frame_sequence,
                        "timestamp_seconds": f.timestamp_seconds,
                        "jpeg_base64": "",
                    }
                    for f in frames
                ],
                "is_last_batch": is_last_batch,
            }
            body_size = len(
                json.dumps(
                    empty_payload, ensure_ascii=False, separators=(",", ":")
                ).encode()
            )
            body_size += sum(4 * ((len(f.jpeg_bytes) + 2) // 3) for f in frames)
            _require(
                body_size <= session.limits["max_request_bytes"],
                "append exceeds worker request limit",
            )
            sent = True
            result, remaining, native_ids = self._request(
                "POST",
                f"{self.endpoint}/{quote(session.video_session_id, safe='')}/append",
                deadline=time.monotonic() + self._timeout_seconds,
                payload_factory=lambda: {
                    "stream_session_id": stream_session_id,
                    "request_id": request_id,
                    "frames": [
                        {
                            "frame_sequence": f.frame_sequence,
                            "timestamp_seconds": f.timestamp_seconds,
                            "jpeg_base64": base64.b64encode(f.jpeg_bytes).decode(
                                "ascii"
                            ),
                        }
                        for f in frames
                    ],
                    "is_last_batch": is_last_batch,
                },
                result_parser=lambda payload: self._parse_append(
                    payload, session, request_id, candidates, accepted, is_last_batch
                ),
            )
            with self._session_state:
                self._sources, self._native_ids = remaining, native_ids
                self._accepted = accepted
                self._emitted = (
                    (result.emitted_frame_sequence, result.emitted_timestamp_seconds)
                    if result.emitted_frame_sequence is not None
                    else None
                )
                self._accepted_count += len(frames)
                self._state = result.state
            return result
        except Exception:
            if sent:
                with self._session_state:
                    self._state = "invalid"
            raise
        finally:
            self._session_operation.release()

    def discard_session(self, timeout_seconds: float = 2.0) -> None:
        if not self._session_operation.acquire(blocking=False):
            raise httpx.PoolTimeout("session operation already active")
        try:
            with self._session_state:
                session = self._native_session
                self._state = "closed"
            if session is None:
                return

            def parse(payload):
                _require(
                    set(payload) == IDENTITY_FIELDS | {"discarded_pending_frames"},
                    "invalid session deletion fields",
                )
                self._identity(payload, session)
                _require(
                    payload["state"] == "closed"
                    and type(payload["discarded_pending_frames"]) is int
                    and 0 <= payload["discarded_pending_frames"] <= MAX_PENDING_SOURCES,
                    "invalid discarded source count",
                )

            self._request(
                "DELETE",
                f"{self.endpoint}/{quote(session.video_session_id, safe='')}",
                deadline=time.monotonic()
                + min(self._timeout_seconds, max(0.001, timeout_seconds)),
                result_parser=parse,
            )
        finally:
            with self._session_state:
                self._native_session = None
                self._sources = ()
                self._native_ids.clear()
            self._session_operation.release()

    def stats(self) -> dict[str, Any]:
        with self._session_state:
            return {
                "state": self._state,
                "memory_scope": "session",
                "backend": BACKEND,
                "continuous_identity": self._state == "active",
                "pending_frame_count": len(self._sources),
                "accepted_frames": self._accepted_count,
                "native_ids": len(self._native_ids),
                "accepted_frame_sequence": self._accepted[0]
                if self._accepted
                else None,
                "accepted_timestamp_seconds": self._accepted[1]
                if self._accepted
                else None,
                "processed_frame_sequence": self._accepted[0]
                if self._accepted
                else None,
                "processed_timestamp_seconds": self._accepted[1]
                if self._accepted
                else None,
                "emitted_frame_sequence": self._emitted[0] if self._emitted else None,
                "emitted_timestamp_seconds": self._emitted[1]
                if self._emitted
                else None,
            }

    def close(self) -> None:
        try:
            self.discard_session()
        except Exception:
            pass
        with self._session_state:
            self._state = "closed"
        super().close()


class VideoSessionTrackingCoordinator(VideoTrackingCoordinator):
    """One append in flight and at most sixteen not-yet-accepted source JPEGs."""

    def __init__(
        self,
        client: VideoSessionTrackingHTTPClient,
        *,
        max_fps=4,
        batch_frames=16,
        batch_seconds=4,
        history_size=128,
        on_batch=None,
    ):
        if batch_frames != 16:
            raise ValueError("session tracking requires batch_frames=16 for startup")
        super().__init__(
            client,
            max_fps=max_fps,
            batch_frames=batch_frames,
            batch_seconds=batch_seconds,
            history_size=history_size,
            on_batch=on_batch,
        )
        self._server_generation = None
        self._failed_session = False
        self._finished_session = False
        self._cleanup_requested = False
        self._stats.update(
            accepted_frame_sequence=None,
            accepted_timestamp_seconds=None,
            processed_frame_sequence=None,
            processed_timestamp_seconds=None,
            emitted_frame_sequence=None,
            emitted_timestamp_seconds=None,
            pending_frame_count=0,
        )

    def reset(self, stream_session_id=None) -> None:
        with self._condition:
            super().reset(stream_session_id)
            self._failed_session = self._finished_session = False
            self._cleanup_requested = self._server_generation is not None
            self._stats.update(
                accepted_frame_sequence=None,
                accepted_timestamp_seconds=None,
                processed_frame_sequence=None,
                processed_timestamp_seconds=None,
                emitted_frame_sequence=None,
                emitted_timestamp_seconds=None,
                pending_frame_count=0,
            )
            self._condition.notify_all()

    def submit(
        self, jpeg_bytes, *, stream_session_id, frame_sequence, timestamp_seconds
    ):
        frame = VideoFrame(frame_sequence, timestamp_seconds, jpeg_bytes)
        _text(stream_session_id, "stream_session_id")
        with self._condition:
            if (
                self._closed
                or self._failed_session
                or self._finished_session
                or self._flush
            ):
                return False
            if self._session is None:
                self._session = stream_session_id
            if stream_session_id != self._session:
                self._stats["rejected_session"] += 1
                return False
            if (
                frame_sequence <= self._last_input[0]
                or timestamp_seconds <= self._last_input[1]
            ):
                self._stats["rejected_order"] += 1
                return False
            self._last_input = frame_sequence, timestamp_seconds
            if timestamp_seconds - self._last_sample < 1 / self.max_fps - 1e-8:
                self._stats["sampled_out"] += 1
                return False
            self._last_sample = timestamp_seconds
            while self._pending and (
                len(self._pending) >= 16
                or self._pending_bytes + len(jpeg_bytes) > MAX_BATCH_JPEG_BYTES
            ):
                dropped, _ = self._pending.popleft()
                self._pending_bytes -= len(dropped.jpeg_bytes)
                self._stats["dropped_backpressure"] += 1
            self._pending.append((frame, time.monotonic()))
            self._pending_bytes += len(jpeg_bytes)
            self._stats["submitted"] += 1
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="native-video-session", daemon=True
                )
                self._thread.start()
            self._condition.notify_all()
            return True

    submit_frame = submit

    def flush(self) -> None:
        """Finish this source, including the native delayed tail; reject new input."""
        with self._condition:
            if self._closed or self._failed_session or self._finished_session:
                return
            if (
                not self._pending
                and self._server_generation is None
                and not self._in_flight
            ):
                self._finished_session = True
                return
            self._flush = True
            self._condition.notify_all()

    def _discard(self):
        try:
            self._client.discard_session(timeout_seconds=2)
        except Exception:
            pass
        self._server_generation = None

    def _run(self):
        try:
            while True:
                with self._condition:
                    while not self._closed:
                        age = (
                            time.monotonic() - self._pending[0][1]
                            if self._pending
                            else 0
                        )
                        if self._cleanup_requested:
                            break
                        if (
                            not self._failed_session
                            and not self._finished_session
                            and (
                                self._flush
                                or len(self._pending) == 16
                                or (
                                    self._server_generation == self._generation
                                    and self._pending
                                    and age >= self._batch_seconds
                                )
                            )
                        ):
                            break
                        self._condition.wait(0.1 if self._pending else None)
                    if self._closed:
                        return
                    cleanup = self._cleanup_requested
                    self._cleanup_requested = False
                    self._in_flight = True
                    if not cleanup:
                        frames = tuple(frame for frame, _ in self._pending)
                        self._pending.clear()
                        self._pending_bytes = 0
                        generation, session, final = (
                            self._generation,
                            self._session,
                            self._flush,
                        )
                        self._in_flight = True
                        self._stats["started"] += 1
                if cleanup:
                    self._discard()
                    with self._condition:
                        self._in_flight = False
                        self._condition.notify_all()
                    continue
                started = time.monotonic()
                try:
                    if self._server_generation != generation:
                        self._discard()
                        self._client.open_session(session)
                        self._server_generation = generation
                    with self._condition:
                        stale = self._closed or generation != self._generation
                    if stale:
                        self._discard()
                        continue
                    result = self._client.append(
                        frames, stream_session_id=session, is_last_batch=final
                    )
                    with self._condition:
                        stale = self._closed or generation != self._generation
                        if stale:
                            self._stats["stale_results"] += 1
                    if stale:
                        self._discard()
                        continue
                    if result.frames and self._on_batch is not None:
                        self._on_batch(result)
                    with self._condition:
                        if not self._closed and generation == self._generation:
                            self._history.extend(result.frames)
                            self._stats.update(
                                succeeded=self._stats["succeeded"] + 1,
                                last_error=None,
                                last_frame_sequence=result.emitted_frame_sequence,
                                completed_timestamp_seconds=result.emitted_timestamp_seconds,
                                **{
                                    key: getattr(result, key)
                                    for key in WATERMARK_FIELDS
                                    | {"pending_frame_count"}
                                },
                            )
                            if final:
                                self._finished_session = True
                                self._flush = False
                except Exception as exc:
                    with self._condition:
                        if generation == self._generation and not self._closed:
                            self._stats["failed"] += 1
                            self._stats["last_error"] = type(exc).__name__
                            self._failed_session = True
                            self._pending.clear()
                            self._pending_bytes = 0
                            self._flush = False
                            self._generation += 1
                    self._discard()
                finally:
                    with self._condition:
                        self._stats["last_latency_seconds"] = time.monotonic() - started
                        self._in_flight = False
                        self._condition.notify_all()
        finally:
            self._discard()
            self._client.close()

    def stats(self):
        with self._condition:
            result = super().stats()
            result.update(
                memory_scope="session",
                state="closed"
                if self._closed
                else "failed"
                if self._failed_session
                else "finished"
                if self._finished_session
                else "running",
                continuous_identity=self._server_generation == self._generation
                and not self._failed_session
                and not self._closed
                and not self._finished_session,
            )
            return result

    def wait_until_idle(self, timeout_seconds=2.0):
        end = time.monotonic() + timeout_seconds
        with self._condition:
            while (
                self._pending
                or self._in_flight
                or self._cleanup_requested
                or (self._flush and not self._finished_session)
            ):
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def snapshot_at(self, timestamp_seconds, frame_sequence=None):
        with self._condition:
            return next(
                (
                    snapshot
                    for snapshot in reversed(self._history)
                    if snapshot.timestamp_seconds <= timestamp_seconds
                    and (
                        frame_sequence is None
                        or snapshot.frame_sequence == frame_sequence
                    )
                ),
                None,
            )
