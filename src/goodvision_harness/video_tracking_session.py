"""Optional continuous SAM video sessions with bounded source provenance.

The transport reuses the batch client's absolute deadline and cancellation loop.
Only frames acknowledged by this session can become delayed observations. An
uncertain append result invalidates the session. A route with an explicit
idempotent append contract may retry the identical request within its original
deadline; it never advances to a later source while an outcome is uncertain.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
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

from .latency_trace import log_timing, trace_id
from .rich_labeling import RichLabelProtocolError, RichLabelSnapshot
from .video_tracking import (
    BACKEND,
    MAX_BATCH_JPEG_BYTES,
    SerializedVideoPayload,
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
HOTSTART_LIMIT_FIELDS = {"hotstart_delay", "max_pending_frames"}
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


def _limits(value: Any, *, causal: bool = False) -> dict[str, Any]:
    _require(
        isinstance(value, dict)
        and set(value) - {"h264_transport"}
        in (LIMIT_FIELDS, LIMIT_FIELDS | HOTSTART_LIMIT_FIELDS),
        "invalid session limits",
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
            and (
                number in (0, 1) if key == "h264_transport"
                else number >= 0 if causal and key == "max_pending_frames"
                else number > 0
            ),
            "invalid session limit value",
        )
    for key, expected in {
        "history_frames": 32,
        "max_inflight_requests": 1,
        "max_active_sessions": 1,
    }.items():
        _require(value[key] == expected, "unsupported session limits")
    _require(
        1 <= value["startup_frames"] <= value["max_append_frames"] <= 16,
        "unsupported session startup/append limits",
    )
    if HOTSTART_LIMIT_FIELDS <= set(value):
        _require(
            value["hotstart_delay"] in ((1,) if causal else (3, 5, 15))
            and value["max_pending_frames"] == value["hotstart_delay"] - 1,
            "unsupported session hotstart limits",
        )
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
class PreparedSessionAppend:
    """Encoding has no authority to advance the native session watermarks."""

    video_session_id: str
    stream_session_id: str
    request_id: str
    frames: tuple[VideoFrame, ...]
    payload: SerializedVideoPayload
    payload_hash: str
    deadline: float


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

    backend = BACKEND
    session_path = "/v1/video/sessions"
    health_path = "/health/video/session"
    native_sam_video_tracking = True
    append_metadata_fields = frozenset()
    health_metadata_fields = frozenset()
    append_transport_retries = 0
    validate_limits = staticmethod(_limits)

    def prepare_snapshots(self, payload, snapshots):
        """Extension point after source and native provenance validation."""
        return snapshots

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 75.0,
        transport=None,
        transport_max_side: int | None = None,
        transport_jpeg_quality: int | None = None,
        transport_codec: str = "jpeg",
        transport_crf: int = 23,
    ):
        super().__init__(
            endpoint,
            timeout_seconds=timeout_seconds,
            transport=transport,
            transport_max_side=transport_max_side,
            transport_jpeg_quality=transport_jpeg_quality,
        )
        if transport_codec not in ("jpeg", "h264"):
            raise ValueError("session transport codec must be jpeg or h264")
        if type(transport_crf) is not int or not 18 <= transport_crf <= 35:
            raise ValueError("session H264 CRF must be an integer in [18, 35]")
        if transport_codec == "h264":
            from .video_transport import ensure_h264_encoder_available

            ensure_h264_encoder_available()
        self._transport_codec = transport_codec
        self._transport_crf = transport_crf
        base = httpx.URL(self.endpoint)
        self.endpoint = str(base.copy_with(path=self.session_path))
        self._health = str(base.copy_with(path=self.health_path))
        self._session_operation = threading.Lock()
        self._session_state = threading.RLock()
        self._native_session: VideoSession | None = None
        self._state = "new"
        self._sources: tuple[tuple[int, float], ...] = ()
        self._accepted = self._emitted = None
        self._accepted_count = 0
        self._native_ids: dict[str, str] = {}
        self._capabilities: dict[str, Any] = {
            "startup_frames": 16,
            "max_append_frames": 16,
        }
        self._health_active_session = None

    @property
    def startup_frames(self) -> int:
        with self._session_state:
            return self._capabilities["startup_frames"]

    @property
    def max_append_frames(self) -> int:
        with self._session_state:
            return self._capabilities["max_append_frames"]

    def gpu_ready(self) -> bool:
        try:
            value = self._request(
                "GET",
                self._health,
                deadline=time.monotonic() + min(self._timeout_seconds, 5),
            )
            _require(
                set(value) - self.health_metadata_fields
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
            limits = self.validate_limits(value["limits"])
            ready = (
                value["enabled"] is True
                and value["ready"] is True
                and value["gpu_ready"] is True
                and value["backend"] == self.backend
                and value["memory_scope"] == "session"
                and value["native_sam_video_tracking"] is self.native_sam_video_tracking
                and (
                    self._transport_codec != "h264" or limits.get("h264_transport") == 1
                )
            )
            if ready:
                with self._session_state:
                    self._capabilities = limits
                    self._health_active_session = value["active_session"]
            return ready
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
                    "backend": self.backend,
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
                    and payload["backend"] == self.backend
                    and payload["memory_scope"] == "session"
                    and payload["state"] == "active",
                    "invalid session creation identity",
                )
                return VideoSession(
                    _text(payload["video_session_id"], "video_session_id"),
                    stream_session_id,
                    _text(payload["predictor_epoch"], "predictor_epoch"),
                    self.validate_limits(payload["limits"]),
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
                self._capabilities = dict(result.limits)
                self._state = "active"
                self._sources = ()
                self._accepted = self._emitted = None
                self._accepted_count = 0
                self._native_ids.clear()
                if (
                    self._transport_codec == "h264"
                    and result.limits.get("h264_transport") != 1
                ):
                    self._state = "invalid"
                    raise RichLabelProtocolError(
                        "worker does not advertise H264 transport"
                    )
            return result
        finally:
            self._session_operation.release()

    def _parse_append(self, payload, session, request_id, candidates, accepted, final):
        _require(
            set(payload)
            == IDENTITY_FIELDS
            | WATERMARK_FIELDS
            | {"request_id", "frames", "pending_frame_count"}
            | self.append_metadata_fields,
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
                    native["backend"] == self.backend
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
            len(remaining)
            <= session.limits.get("max_pending_frames", MAX_PENDING_SOURCES),
            "native response exceeds pending source limit",
        )
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
                self.prepare_snapshots(payload, snapshots),
                *received,
                *processed,
                *(emitted if emitted is not None else (None, None)),
                len(remaining),
                backend=self.backend,
            ),
            remaining,
            native_ids,
        )

    def prepare_append(
        self, frames: Sequence[VideoFrame], *, stream_session_id: str,
        request_id: str | None = None,
    ) -> PreparedSessionAppend:
        """Prepare one nonfinal batch independently of the previous append.

        Only immutable source frames and negotiated limits are read here. The
        ordered append path still validates provenance and updates all native
        state. EOF is sent separately after every prepared batch has finished.
        """
        frames = tuple(frames)
        _validate_frames(frames)
        request_id = _text(request_id or uuid4().hex, "request_id")
        with self._session_state:
            session = self._native_session
            _require(session is not None, "native session is not active")
            _require(stream_session_id == session.stream_session_id,
                     "source session mismatch")
            _require(len(frames) <= session.limits["max_append_frames"],
                     "append exceeds worker frame limits")
        started = time.monotonic()
        deadline = started + self._timeout_seconds
        metadata = {
            "stream_session_id": stream_session_id,
            "request_id": request_id,
            "frames": [
                {"frame_sequence": f.frame_sequence,
                 "timestamp_seconds": f.timestamp_seconds} for f in frames
            ],
            "is_last_batch": False,
        }
        metrics = {}
        if self._transport_codec == "h264":
            from .video_transport import encode_h264_batch

            _require(session.limits.get("h264_transport") == 1,
                     "worker does not advertise H264 transport")
            encoded = encode_h264_batch(
                frames, max_side=self._transport_max_side or 512,
                crf=self._transport_crf, deadline=deadline,
            )
            _require(len(encoded.data) <= session.limits["max_total_jpeg_bytes"],
                     "H264 exceeds worker byte limits")
            payload = metadata | {"video": {
                "format": "h264_annexb",
                "data_base64": base64.b64encode(encoded.data).decode("ascii"),
            }}
            self._record_transport(
                frames, len(encoded.data), encoded.total_seconds,
                codec="h264", encode_seconds=encoded.encode_seconds,
            )
            metrics = {
                "compression_seconds": encoded.total_seconds,
                "video_encode_seconds": encoded.encode_seconds,
                "transport_bytes": len(encoded.data), "transport_codec": "h264",
            }
        else:
            transport_started = time.monotonic()
            wire_frames = self._transport_frames(frames, deadline=deadline)
            _require(all(len(f.jpeg_bytes) <= session.limits["max_jpeg_bytes"]
                         for f in wire_frames)
                     and sum(len(f.jpeg_bytes) for f in wire_frames)
                     <= session.limits["max_total_jpeg_bytes"],
                     "JPEG append exceeds worker byte limits")
            payload = metadata | {"frames": [
                {**stamp, "jpeg_base64": base64.b64encode(f.jpeg_bytes).decode("ascii")}
                for stamp, f in zip(metadata["frames"], wire_frames, strict=True)
            ]}
            metrics = {
                "compression_seconds": time.monotonic() - transport_started,
                "video_encode_seconds": 0.0,
                "transport_bytes": sum(len(f.jpeg_bytes) for f in wire_frames),
                "transport_codec": "jpeg",
            }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        _require(len(body) <= session.limits["max_request_bytes"],
                 "append exceeds worker request limit")
        if time.monotonic() >= deadline:
            raise httpx.TimeoutException("video preparation exceeded total deadline")
        return PreparedSessionAppend(
            session.video_session_id, stream_session_id, request_id, frames,
            SerializedVideoPayload(body, metadata, time.monotonic() - started, metrics),
            hashlib.sha256(body).hexdigest(), deadline,
        )

    def append(
        self,
        frames: Sequence[VideoFrame],
        *,
        stream_session_id: str,
        is_last_batch: bool = False,
        request_id: str | None = None,
        prepared: PreparedSessionAppend | None = None,
    ) -> VideoSessionTrackingResult:
        if not self._session_operation.acquire(blocking=False):
            raise httpx.PoolTimeout("session operation already active")
        sent = False
        deadline = time.monotonic() + self._timeout_seconds
        try:
            _require(
                isinstance(frames, Sequence) and len(frames) <= 16,
                "append requires at most sixteen source frames",
            )
            frames = tuple(frames)
            if frames:
                _validate_frames(frames)
            _require(type(is_last_batch) is bool, "final flag must be boolean")
            request_id = _text(
                request_id or (prepared.request_id if prepared else uuid4().hex),
                "request_id",
            )
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
                    self._accepted is not None
                    or is_last_batch
                    or len(frames) >= session.limits["startup_frames"],
                    "initial nonfinal append requires the advertised startup frames",
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
                    len(frames)
                    <= min(
                        session.limits["max_frames"],
                        session.limits["max_append_frames"],
                    ),
                    "append exceeds worker frame limits",
                )
                accepted = candidates[-1] if frames else self._accepted
                if prepared is not None:
                    _require(
                        not is_last_batch and prepared.frames == frames
                        and prepared.stream_session_id == stream_session_id
                        and prepared.video_session_id == session.video_session_id
                        and prepared.request_id == request_id
                        and hashlib.sha256(prepared.payload.body).hexdigest()
                        == prepared.payload_hash,
                        "prepared append source/session mismatch",
                    )
                    deadline = min(deadline, prepared.deadline)

            wire_payload = None

            def payload_factory():
                nonlocal sent, wire_payload
                if wire_payload is not None:
                    return wire_payload
                if prepared is not None:
                    sent = True
                    wire_payload = prepared.payload
                    return wire_payload
                if self._transport_codec == "h264" and frames:
                    from .video_transport import encode_h264_batch

                    _require(
                        session.limits.get("h264_transport") == 1,
                        "worker does not advertise H264 transport",
                    )
                    encoded = encode_h264_batch(
                        frames,
                        max_side=self._transport_max_side or 512,
                        crf=self._transport_crf,
                        deadline=deadline,
                    )
                    _require(
                        len(encoded.data) <= session.limits["max_total_jpeg_bytes"],
                        "H264 exceeds worker byte limits",
                    )
                    payload = {
                        "stream_session_id": stream_session_id,
                        "request_id": request_id,
                        "frames": [
                            {
                                "frame_sequence": f.frame_sequence,
                                "timestamp_seconds": f.timestamp_seconds,
                            }
                            for f in frames
                        ],
                        "is_last_batch": is_last_batch,
                        "video": {
                            "format": "h264_annexb",
                            "data_base64": base64.b64encode(encoded.data).decode(
                                "ascii"
                            ),
                        },
                    }
                    self._record_transport(
                        frames,
                        len(encoded.data),
                        encoded.total_seconds,
                        codec="h264",
                        encode_seconds=encoded.encode_seconds,
                    )
                else:
                    payload = jpeg_payload()
                _require(
                    len(
                        json.dumps(
                            payload, ensure_ascii=False, separators=(",", ":")
                        ).encode()
                    )
                    <= session.limits["max_request_bytes"],
                    "append exceeds worker request limit",
                )
                sent = True
                wire_payload = payload
                return wire_payload

            def jpeg_payload():
                wire_frames = self._transport_frames(frames, deadline=deadline)
                _require(
                    all(
                        len(f.jpeg_bytes) <= session.limits["max_jpeg_bytes"]
                        for f in wire_frames
                    )
                    and sum(len(f.jpeg_bytes) for f in wire_frames)
                    <= session.limits["max_total_jpeg_bytes"],
                    "append exceeds worker byte limits",
                )
                return {
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
                        for f in wire_frames
                    ],
                    "is_last_batch": is_last_batch,
                }

            for attempt in range(1, self.append_transport_retries + 2):
                try:
                    result, remaining, native_ids = self._request(
                        "POST",
                        f"{self.endpoint}/"
                        f"{quote(session.video_session_id, safe='')}/append",
                        deadline=deadline,
                        payload_factory=payload_factory,
                        result_parser=lambda payload: self._parse_append(
                            payload, session, request_id, candidates,
                            accepted, is_last_batch,
                        ),
                    )
                    if attempt > 1:
                        log_timing("sam_client_append_recovered",
                                   stream_session_id=trace_id(stream_session_id),
                                   request_id=trace_id(request_id), attempt=attempt)
                    break
                except (httpx.RemoteProtocolError, httpx.ConnectError,
                        httpx.ReadError, httpx.WriteError) as exc:
                    # The causal worker caches the last request by ID and payload
                    # digest. Keep the session-operation lock and exact payload:
                    # a lost response can be replayed without advancing SAM twice.
                    # HTTP status, validation and deadline failures still fail closed.
                    if (attempt > self.append_transport_retries
                            or time.monotonic() + 0.2 >= deadline):
                        raise
                    log_timing("sam_client_append_retry",
                               stream_session_id=trace_id(stream_session_id),
                               request_id=trace_id(request_id), attempt=attempt,
                               next_attempt=attempt + 1, error_type=type(exc).__name__,
                               remaining_seconds=max(0.0, deadline - time.monotonic()))
                    time.sleep(0.2)
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
        discarded = False
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
            discarded = True
        except httpx.HTTPStatusError as exc:
            discarded = exc.response.status_code in (404, 410)
            raise
        finally:
            with self._session_state:
                # A timed-out preparation can still occupy the HTTP slot. Keep
                # its sole remote handle until DELETE can actually be admitted.
                if discarded:
                    self._native_session = None
                self._sources = ()
                self._native_ids.clear()
            self._session_operation.release()

    def stats(self) -> dict[str, Any]:
        with self._session_state:
            return {
                "state": self._state,
                "memory_scope": "session",
                "backend": self.backend,
                "continuous_identity": self._state == "active",
                "pending_frame_count": len(self._sources),
                "accepted_frames": self._accepted_count,
                "native_ids": len(self._native_ids),
                "cleanup_pending": self._native_session is not None
                and self._state in ("closed", "invalid"),
                "startup_frames": self.startup_frames,
                "max_append_frames": self.max_append_frames,
                "hotstart_delay": self._capabilities.get("hotstart_delay", 15),
                "max_pending_frames": self._capabilities.get(
                    "max_pending_frames", MAX_PENDING_SOURCES
                ),
                "transport": self.transport_stats(),
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

    async def _shutdown(self) -> None:
        # The base invokes shutdown only after the active task has unwound.
        # Retry only DELETE for this known token, never an uncertain append.
        with self._session_state:
            session = self._native_session
        if session is not None:
            deadline = time.monotonic() + min(self._timeout_seconds, 1.0)
            try:
                await asyncio.wait_for(
                    self._exchange(
                        "DELETE",
                        f"{self.endpoint}/{quote(session.video_session_id, safe='')}",
                        None,
                        None,
                        deadline=deadline,
                    ),
                    timeout=max(0.001, deadline - time.monotonic()),
                )
            except (Exception, asyncio.CancelledError):
                pass
            finally:
                with self._session_state:
                    if self._native_session is session:
                        self._native_session = None
        await super()._shutdown()

    def close(self) -> None:
        try:
            self.discard_session()
        except Exception:
            pass
        with self._session_state:
            self._state = "closed"
        super().close()

    def close_complete(self) -> bool:
        """Local HTTP work has unwound; remote ownership needs fresh health."""
        with self._lock:
            return self._closed and (
                self._thread is None or not self._thread.is_alive()
            )


class VideoSessionTrackingCoordinator(VideoTrackingCoordinator):
    """One append in flight and at most sixteen not-yet-accepted source JPEGs.

    batch_frames is a dispatch target after the worker's advertised startup.
    An accumulated backlog is combined up to max_append_frames to avoid more
    round trips. Neither a short target nor its timer fabricates startup frames.
    """

    def __init__(
        self,
        client: VideoSessionTrackingHTTPClient,
        *,
        max_fps=4,
        batch_frames=16,
        batch_seconds=4,
        history_size=128,
        on_batch=None,
        pending_capacity=16,
    ):
        if type(pending_capacity) is not int or not 1 <= pending_capacity <= 16:
            raise ValueError("pending capacity must be an integer within 1..16")
        self._pending_capacity = pending_capacity
        self._source_arrivals: dict[int, float] = {}
        self._capture_started_at: float | None = None
        self._capture_finished_at: float | None = None
        self._last_completion_at: float | None = None
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
            processed_frames=0,
            source_to_result_seconds=None,
            session_source_frames=0,
            session_admitted_frames=0,
            session_sampled_out=0,
            session_dropped_backpressure=0,
            completed_during_capture=0,
            completed_after_capture=0,
        )

    def reset(self, stream_session_id=None) -> None:
        with self._condition:
            super().reset(stream_session_id)
            self._failed_session = self._finished_session = False
            self._source_arrivals.clear()
            self._capture_started_at = self._capture_finished_at = None
            self._last_completion_at = None
            self._cleanup_requested = self._server_generation is not None
            self._stats.update(
                accepted_frame_sequence=None,
                accepted_timestamp_seconds=None,
                processed_frame_sequence=None,
                processed_timestamp_seconds=None,
                emitted_frame_sequence=None,
                emitted_timestamp_seconds=None,
                pending_frame_count=0,
                processed_frames=0,
                source_to_result_seconds=None,
                session_source_frames=0,
                session_admitted_frames=0,
                session_sampled_out=0,
                session_dropped_backpressure=0,
                completed_during_capture=0,
                completed_after_capture=0,
            )
            self._condition.notify_all()

    def submit(
        self, jpeg_bytes, *, stream_session_id, frame_sequence, timestamp_seconds
    ):
        frame = VideoFrame(frame_sequence, timestamp_seconds, jpeg_bytes)
        _text(stream_session_id, "stream_session_id")
        dropped_sequences = []
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
            if self._capture_started_at is None:
                self._capture_started_at = time.monotonic()
            self._stats["session_source_frames"] += 1
            if timestamp_seconds - self._last_sample < 1 / self.max_fps - 1e-8:
                self._stats["sampled_out"] += 1
                self._stats["session_sampled_out"] += 1
                return False
            self._last_sample = timestamp_seconds
            while self._pending and (
                len(self._pending) >= self._pending_capacity
                or self._pending_bytes + len(jpeg_bytes) > MAX_BATCH_JPEG_BYTES
            ):
                dropped, _ = self._pending.popleft()
                dropped_sequences.append(dropped.frame_sequence)
                self._pending_bytes -= len(dropped.jpeg_bytes)
                self._stats["dropped_backpressure"] += 1
                self._stats["session_dropped_backpressure"] += 1
                self._source_arrivals.pop(dropped.frame_sequence, None)
            self._source_arrivals[frame_sequence] = time.monotonic()
            # Includes pending requests and native delayed outputs, independent
            # of the history capacity and stream duration.
            while len(self._source_arrivals) > 64:
                self._source_arrivals.pop(next(iter(self._source_arrivals)))
            self._pending.append((frame, time.monotonic()))
            self._pending_bytes += len(jpeg_bytes)
            self._stats["submitted"] += 1
            self._stats["session_admitted_frames"] += 1
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="native-video-session", daemon=True
                )
                self._thread.start()
            self._condition.notify_all()
        if dropped_sequences:
            log_timing(
                "sam_queue_overflow",
                stream_session_id=trace_id(stream_session_id),
                dropped_sequences=dropped_sequences,
                admitted_sequence=frame_sequence,
                pending_capacity=self._pending_capacity,
                reason="existing_pending_capacity_or_byte_limit",
            )
        return True

    submit_frame = submit

    def mark_capture_finished(self, *, stream_session_id, generation) -> bool:
        """Freeze the live-rate denominator without flushing or closing ingress.

        Camera uploads may have finished while accepted downstream work still
        drains. Stale run signals cannot alter a replacement run's clock.
        """
        with self._condition:
            if (
                self._closed or stream_session_id != self._session
                or generation != self._generation
            ):
                return False
            if self._capture_finished_at is None:
                self._capture_finished_at = time.monotonic()
            return True

    def flush(self) -> None:
        """Finish this source, including the native delayed tail; reject new input."""
        with self._condition:
            if self._closed or self._failed_session or self._finished_session:
                return
            if self._capture_finished_at is None:
                self._capture_finished_at = time.monotonic()
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
                        started_session = self._server_generation == self._generation
                        target = (
                            min(self._batch_frames, self._client.max_append_frames)
                            if started_session
                            else self._client.startup_frames
                        )
                        if (
                            not self._failed_session
                            and not self._finished_session
                            and (
                                self._flush
                                or len(self._pending) >= target
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
                        self._stats["last_queue_wait_seconds"] = age
                        self._stats["total_queue_wait_seconds"] += age
                        frames = tuple(
                            self._pending.popleft()[0]
                            for _ in range(
                                min(len(self._pending), self._client.max_append_frames)
                            )
                        )
                        self._stats["last_request_frames"] = len(frames)
                        self._pending_bytes -= sum(
                            len(frame.jpeg_bytes) for frame in frames
                        )
                        generation, session, final = (
                            self._generation,
                            self._session,
                            self._flush and not self._pending,
                        )
                        self._in_flight = True
                        self._stats["started"] += 1
                        dispatch_counts = {
                            key: self._stats[key] for key in (
                                "session_source_frames", "session_admitted_frames",
                                "session_sampled_out", "session_dropped_backpressure",
                            )
                        }
                        dispatch_counts["pending_frames"] = len(self._pending)
                if cleanup:
                    self._discard()
                    with self._condition:
                        self._in_flight = False
                        self._condition.notify_all()
                    continue
                started = time.monotonic()
                timing = {
                    "stream_session_id": trace_id(session),
                    "generation": generation,
                    "input_sequences": [frame.frame_sequence for frame in frames],
                    "input_frames": len(frames),
                    "final": final,
                    "queue_oldest_seconds": age,
                    "outcome": "error",
                    **dispatch_counts,
                }
                log_timing(
                    "sam_coordinator_dispatch", **(timing | {"outcome": "started"}),
                )
                try:
                    setup_started = time.monotonic()
                    if self._server_generation != generation:
                        self._discard()
                        self._client.open_session(session)
                        self._server_generation = generation
                    timing["session_setup_seconds"] = time.monotonic() - setup_started
                    with self._condition:
                        stale = self._closed or generation != self._generation
                    if stale:
                        timing["outcome"] = "stale"
                        self._discard()
                        continue
                    append_started = time.monotonic()
                    result = self._client.append(
                        frames, stream_session_id=session, is_last_batch=final
                    )
                    timing["append_seconds"] = time.monotonic() - append_started
                    timing["request_id"] = trace_id(result.request_id)
                    timing["output_sequences"] = [
                        frame.frame_sequence for frame in result.frames
                    ]
                    timing["output_frames"] = len(result.frames)
                    with self._condition:
                        stale = self._closed or generation != self._generation
                        if stale:
                            self._stats["stale_results"] += 1
                    if stale:
                        timing["outcome"] = "stale"
                        self._discard()
                        continue
                    commit_started = time.monotonic()
                    if result.frames and self._on_batch is not None:
                        self._on_batch(result)
                    timing["commit_seconds"] = time.monotonic() - commit_started
                    with self._condition:
                        if not self._closed and generation == self._generation:
                            completed_at = time.monotonic()
                            timing["completion_interval_seconds"] = (
                                completed_at - self._last_completion_at
                                if self._last_completion_at is not None else None
                            )
                            self._last_completion_at = completed_at
                            timing["source_to_result"] = []
                            self._history.extend(result.frames)
                            for snapshot in result.frames:
                                arrival = self._source_arrivals.pop(
                                    snapshot.frame_sequence, None,
                                )
                                if arrival is not None:
                                    self._stats["source_to_result_seconds"] = (
                                        completed_at - arrival
                                    )
                                    timing["source_to_result"].append({
                                        "sequence": snapshot.frame_sequence,
                                        "seconds": completed_at - arrival,
                                    })
                            self._stats["processed_frames"] += len(result.frames)
                            self._stats[
                                "completed_during_capture"
                                if self._capture_finished_at is None
                                else "completed_after_capture"
                            ] += len(result.frames)
                            timing["during_capture"] = self._capture_finished_at is None
                            timing["outcome"] = "ok"
                            timing["pending_frames_at_completion"] = len(self._pending)
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
                    timing["error_type"] = type(exc).__name__
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
                    timing["total_seconds"] = time.monotonic() - started
                    with self._condition:
                        self._stats["last_latency_seconds"] = time.monotonic() - started
                        self._in_flight = False
                        self._condition.notify_all()
                    log_timing("sam_coordinator_timing", **timing)
        finally:
            self._discard()
            self._client.close()

    def stats(self):
        with self._condition:
            result = super().stats()
            capture_seconds = (
                max(0.0, (
                    self._capture_finished_at
                    if self._capture_finished_at is not None else time.monotonic()
                ) - self._capture_started_at)
                if self._capture_started_at is not None else 0.0
            )
            result.update(
                capture_elapsed_seconds=capture_seconds,
                # EOF catch-up never improves the live analysis rate.
                completed_analysis_fps=(
                    self._stats["completed_during_capture"] / capture_seconds
                    if capture_seconds > 0 else None
                ),
                backend=getattr(self._client, "backend", BACKEND),
                memory_scope="session",
                pending_capacity=self._pending_capacity,
                startup_frames=self._client.startup_frames,
                append_target_frames=min(
                    self._batch_frames, self._client.max_append_frames
                ),
                max_append_frames=self._client.max_append_frames,
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

    def close(self, timeout_seconds: float = 2.0) -> None:
        with self._condition:
            if self._capture_finished_at is None:
                self._capture_finished_at = time.monotonic()
        super().close(timeout_seconds)

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

    def close_complete(self) -> bool:
        """A close timeout is not completion of the inference/cleanup thread."""
        with self._condition:
            stopped = self._closed and (
                self._thread is None or not self._thread.is_alive()
            )
        client_closed = getattr(self._client, "close_complete", None)
        return stopped and (client_closed is None or client_closed())

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
