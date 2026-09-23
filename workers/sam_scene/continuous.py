"""Optional bounded native SAM3.1 sessions sharing the worker's one GPU slot."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

if __package__:
    from .sam31 import (
        BACKEND,
        SAM3_COMMIT,
        SourceFrame,
        VideoBatch,
        _invalid_constant,
        _unique_fields,
        native_snapshot,
        parse_batch,
    )
    from .streaming_state import BoundedRetention, finalize_pending, make_prod_model
else:
    from sam31 import (
        BACKEND,
        SAM3_COMMIT,
        SourceFrame,
        VideoBatch,
        _invalid_constant,
        _unique_fields,
        native_snapshot,
        parse_batch,
    )
    from streaming_state import BoundedRetention, finalize_pending, make_prod_model

LOGGER = logging.getLogger("goodvision.sam31.session")
MAX_H264_BYTES = 8 * 1024 * 1024


def _timing_log(message, fields, *, level=logging.INFO):
    """Selected scalar diagnostics must never change the operation's result."""
    try:
        values = " ".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in fields.items()
        )
        LOGGER.log(level, "%s %s", message, values, extra={"sam_timing": fields})
    except Exception:
        pass


class _AppendTiming:
    """Disjoint host wall-clock spans; these are not CUDA kernel timings.

    Advancing the native iterator includes its existing waits/synchronization.
    Snapshot conversion can also wait for device-to-host transfers. No extra
    CUDA synchronization is introduced for measurement.
    """

    phases = (
        "validation", "decode", "native_advance", "snapshot", "retention",
        "response_build", "response_serialize", "cleanup",
    )

    def __init__(self, route, diagnostics):
        self.started = self.changed = time.perf_counter()
        self.phase = "validation"
        self.seconds = dict.fromkeys(self.phases, 0.0)
        self.fields = diagnostics if diagnostics is not None else {}
        self.fields.update(
            route=route,
            request_id=None,
            input_frames=0,
            output_frames=0,
            pending_frames=0,
            input_first_frame_sequence=None,
            input_last_frame_sequence=None,
            output_first_frame_sequence=None,
            output_last_frame_sequence=None,
            accepted_frame_sequence=None,
            emitted_frame_sequence=None,
            total_session_frames=0,
            is_last_batch=None,
            replayed=False,
            response_bytes=0,
        )

    def mark(self, phase):
        now = time.perf_counter()
        self.seconds[self.phase] += now - self.changed
        self.changed, self.phase = now, phase

    def observe_session(self, session):
        self.fields.update(
            pending_frames=len(session.sources),
            accepted_frame_sequence=(
                session.accepted.frame_sequence if session.accepted else None
            ),
            emitted_frame_sequence=(
                session.emitted.frame_sequence if session.emitted else None
            ),
            total_session_frames=session.total_frames,
        )

    def finish(self, outcome, status_code):
        self.mark(self.phase)
        fields = self.fields | {
            "outcome": outcome,
            "status_code": status_code,
            "phase": self.phase,
            "timing_basis": "host_wall_clock",
            **{f"{key}_seconds": value for key, value in self.seconds.items()},
            "total_seconds": self.changed - self.started,
        }
        # Preserve coarse names consumed by existing diagnostic tooling.
        fields.update(
            validation_decode_seconds=(
                self.seconds["validation"] + self.seconds["decode"]
            ),
            native_output_seconds=(
                self.seconds["native_advance"] + self.seconds["snapshot"]
            ),
            retention_response_seconds=sum(
                self.seconds[key]
                for key in ("retention", "response_build", "response_serialize")
            ),
        )
        _timing_log(
            "Native session append timing", fields,
            level=logging.INFO if outcome == "completed" else logging.WARNING,
        )


@dataclass
class _PreparedAppend:
    """CPU-only input; no native state may be read or mutated by decoding."""

    batch: VideoBatch
    final: bool
    digest: str
    images: list = field(default_factory=list)
    source_sizes: list = field(default_factory=list)

    def close(self):
        for image in self.images:
            image.close()
        self.images.clear()


class _RequestResources:
    """Keep lane/buffer ownership until an abandoned worker actually exits.

    Cancelling asyncio.to_thread cannot stop an active codec/native call. A
    pending thread may also start after HTTP cleanup; run() rejects that case.
    """

    def __init__(self, lane):
        self.lane = lane
        self.preparation = None
        self.prepared = None
        self._mutex = threading.Lock()
        self._running = False
        self._closing = False
        self._closed = False

    def run(self, fn):
        with self._mutex:
            if self._closing:
                raise TimeoutError("session operation abandoned")
            self._running = True
        try:
            return fn()
        finally:
            with self._mutex:
                self._running = False
                if self._closing:
                    self._release()

    def close(self):
        with self._mutex:
            self._closing = True
            if not self._running:
                self._release()

    def _release(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.prepared is not None:
                self.prepared.close()
        finally:
            try:
                if self.preparation is not None:
                    self.preparation.release()
            finally:
                if self.lane is not None:
                    self.lane.release()


@dataclass(frozen=True, slots=True)
class SessionSettings:
    enabled: bool = False
    h264_enabled: bool = False
    hotstart_delay: int = 15
    postprocess_batch_size: int = 1
    history_frames: int = 32
    max_session_frames: int = 65536
    max_cumulative_native_ids: int = 2048
    idle_seconds: float = 120.0

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.h264_enabled) is not bool:
            raise ValueError("invalid session enabled flag")
        if type(self.hotstart_delay) is not int or self.hotstart_delay not in {
            3,
            5,
            15,
        }:
            raise ValueError("session hotstart_delay must be 3, 5, or 15")
        if type(
            self.postprocess_batch_size
        ) is not int or self.postprocess_batch_size not in {1, 4}:
            raise ValueError("session postprocess_batch_size must be 1 or 4")
        for name, low, high in (
            ("history_frames", 32, 64),
            ("max_session_frames", 16, 65536),
            ("max_cumulative_native_ids", 16, 2048),
        ):
            if (
                type(getattr(self, name)) is not int
                or not low <= getattr(self, name) <= high
            ):
                raise ValueError(f"invalid session {name}")
        if (
            type(self.idle_seconds) not in (int, float)
            or not math.isfinite(self.idle_seconds)
            or not 5 <= self.idle_seconds <= 600
        ):
            raise ValueError("session idle_seconds must be within 5..600")

    @classmethod
    def from_env(cls):
        values = {}
        defaults = cls()
        for name in cls.__dataclass_fields__:
            default = getattr(defaults, name)
            value = os.environ.get("SAM31_SESSION_" + name.upper())
            if value is None:
                continue
            if type(default) is bool:
                if value.lower() not in {"true", "false"}:
                    raise ValueError(
                        f"SAM31_SESSION_{name.upper()} must be true or false"
                    )
                values[name] = value.lower() == "true"
            else:
                values[name] = type(default)(value)
        return cls(**values)


@dataclass
class Session:
    video_session_id: str
    stream_session_id: str
    predictor_epoch: str
    touched: float
    status: str = "active"
    native_state: Any = None
    native_retention_installed: bool = False
    total_frames: int = 0
    emitted_index: int = -1
    accepted: SourceFrame | None = None
    emitted: SourceFrame | None = None
    source_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None
    sources: dict[int, SourceFrame] = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    last_request_id: str | None = None
    last_payload_hash: str | None = None
    last_response: bytes | None = None


@dataclass(frozen=True, slots=True)
class SessionProfile:
    """Immutable wire and latency contract supplied by one perception plugin."""

    route: str
    backend_name: str
    startup_frames: int = 1
    max_append_frames: int = 4
    output_delay_frames: int = 0
    native_sam_video_tracking: bool = True

    def __post_init__(self):
        if not _identifier(self.route) or not _identifier(self.backend_name):
            raise ValueError("invalid perception profile identifier")
        for name, low, high in (
            ("startup_frames", 1, 16),
            ("max_append_frames", 1, 16),
            ("output_delay_frames", 0, 14),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("invalid perception profile limit")
        if self.startup_frames > self.max_append_frames:
            raise ValueError("startup exceeds append capacity")


class SessionRegistry:
    """One ownership domain across legacy and pluggable session engines.

    HTTP callers hold the existing GPU admission slot during create/append/delete.
    This registry also serializes direct creation and keeps cancelled native state
    owned until backend teardown has actually finished.
    """

    def __init__(self):
        self.engines = []
        self.mutex = threading.RLock()
        self.on_state_discarded = None

    def note_state_discarded(self):
        if self.on_state_discarded is not None:
            self.on_state_discarded()

    def add(self, engine):
        with self.mutex:
            if engine not in self.engines:
                self.engines.append(engine)
                engine.registry = self

    @property
    def has_active_session(self):
        return any(engine.has_active_session for engine in tuple(self.engines))

    def expire(self):
        for engine in tuple(self.engines):
            engine.expire()

    def close(self):
        for engine in tuple(self.engines):
            engine.close()


def _json(payload):
    try:
        raw = json.loads(
            payload, object_pairs_hook=_unique_fields, parse_constant=_invalid_constant
        )
        if type(raw) is not dict:
            raise ValueError()
        return raw
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise HTTPException(400, "invalid session request") from None


def _identifier(value):
    return (
        type(value) is str
        and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value) is not None
    )


@dataclass(frozen=True, slots=True)
class SessionVideoBatch(VideoBatch):
    video_bytes: bytes


def _parse_h264_batch(raw, settings):
    frames = raw["frames"]
    if type(frames) is not list:
        raise HTTPException(400, "invalid session frame metadata")
    if not 1 <= len(frames) <= settings.max_frames:
        raise HTTPException(413, "video batch frame count outside allowed bounds")
    sources = []
    previous_sequence, previous_timestamp = -1, -1.0
    for frame in frames:
        if type(frame) is not dict or set(frame) != {
            "frame_sequence",
            "timestamp_seconds",
        }:
            raise HTTPException(400, "invalid session frame metadata")
        sequence, timestamp = frame["frame_sequence"], frame["timestamp_seconds"]
        try:
            valid = (
                type(sequence) is int
                and 0 <= sequence < 2**63
                and type(timestamp) in (int, float)
                and math.isfinite(timestamp)
                and timestamp >= 0
            )
        except OverflowError:
            valid = False
        if not valid:
            raise HTTPException(400, "invalid session frame metadata")
        if sequence <= previous_sequence or timestamp <= previous_timestamp:
            raise HTTPException(409, "video source frames must strictly increase")
        sources.append(SourceFrame(sequence, float(timestamp), b""))
        previous_sequence, previous_timestamp = sequence, timestamp
    video = raw["video"]
    if (
        type(video) is not dict
        or set(video) != {"format", "data_base64"}
        or video["format"] != "h264_annexb"
        or type(video["data_base64"]) is not str
    ):
        raise HTTPException(400, "invalid session video transport")
    encoded = video["data_base64"]
    max_bytes = min(MAX_H264_BYTES, settings.max_total_jpeg_bytes)
    if not encoded or len(encoded) > 4 * ((max_bytes + 2) // 3):
        raise HTTPException(413, "video bitstream exceeds byte bound")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(400, "invalid session video encoding") from None
    if not data or len(data) > max_bytes:
        raise HTTPException(413, "video bitstream exceeds byte bound")
    return SessionVideoBatch(
        raw["stream_session_id"], raw["request_id"], tuple(sources), data
    )


def _h264_transport():
    # Import the codec adapter only when explicitly configured or requested.
    if __package__:
        from . import h264_transport
    else:
        import h264_transport
    return h264_transport


def parse_append(payload, settings, *, h264_enabled=False, diagnostics=None):
    if not payload or len(payload) > settings.max_request_bytes:
        raise HTTPException(413, "session request exceeds byte bound")
    raw = _json(payload)
    expected = {"stream_session_id", "request_id", "frames", "is_last_batch"}
    is_video = "video" in raw
    if (
        set(raw) != (expected | {"video"} if is_video else expected)
        or type(raw.get("is_last_batch")) is not bool
    ):
        raise HTTPException(400, "invalid session append fields")
    if not _identifier(raw["stream_session_id"]) or not _identifier(raw["request_id"]):
        raise HTTPException(400, "invalid session identifiers")
    if diagnostics is not None:
        diagnostics.update(
            request_id=raw["request_id"], is_last_batch=raw["is_last_batch"]
        )
    if is_video:
        if not h264_enabled:
            raise HTTPException(400, "session video transport is disabled")
        batch = _parse_h264_batch(raw, settings)
    elif raw["frames"] == [] and raw["is_last_batch"]:
        batch = VideoBatch(raw["stream_session_id"], raw["request_id"], ())
    else:
        batch = parse_batch(
            json.dumps(
                {
                    "stream_session_id": raw["stream_session_id"],
                    "batch_id": raw["request_id"],
                    "frames": raw["frames"],
                },
                separators=(",", ":"),
            ).encode(),
            settings,
        )
    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return batch, raw["is_last_batch"], digest


class NativeContinuation:
    def __init__(self, video_engine, settings):
        self.video = video_engine
        self.settings = settings
        self.model = None
        self.retention = None
        self.gpu_ready = False

    def warmup(self):
        if not self.video.gpu_ready:
            raise RuntimeError("native CUDA video model is unavailable")
        self.model = make_prod_model(
            self.video.backend,
            hotstart_delay=self.settings.hotstart_delay,
            postprocess_batch_size=self.settings.postprocess_batch_size,
        )
        self.retention = BoundedRetention(self.model, self.settings.history_frames)
        from PIL import Image

        images = [Image.new("RGB", (320, 192)) for _ in range(16)]
        session = Session("warmup", "warmup", "warmup", time.monotonic())
        try:
            seen = [index for index, _ in self.run(session, images, final=False)]
            session.total_frames = 16
            session.emitted_index = len(seen) - 1
            self.prune(session)
            seen.extend(index for index, _ in self.run(session, [], final=True))
            if seen != list(range(16)):
                raise RuntimeError("native continuation warmup omitted source frames")
            self.video.backend._torch.cuda.synchronize()
            self.gpu_ready = True
            LOGGER.info(
                "Native continuation CUDA warmup completed hotstart_delay=%d "
                "postprocess_batch_size=%d",
                self.settings.hotstart_delay,
                self.settings.postprocess_batch_size,
            )
        finally:
            self.discard(session)
            for image in images:
                image.close()

    def run(self, session, images, *, final):
        native = self.video.backend
        with (
            native._torch.inference_mode(),
            native._torch.autocast(device_type="cuda", dtype=native._torch.bfloat16),
        ):
            state = session.native_state
            if state is None:
                state = self.model.init_state(
                    resource_path=list(images),
                    offload_video_to_cpu=True,
                    async_loading_frames=False,
                )
                session.native_state = state
                self.model.add_prompt(
                    inference_state=state, frame_idx=0, text_str="person"
                )
                # A first-and-final short recording needs no ring and must use
                # this session's configured profile, not the batch model's 15.
                if not final:
                    self.retention.install(state)
                    session.native_retention_installed = True
            elif images:
                self.retention.append(
                    self.model,
                    state,
                    images,
                    frame_limit=self.settings.max_session_frames,
                )
            if images:
                yield from self.model.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=session.total_frames,
                    max_frame_num_to_track=len(images),
                    reverse=False,
                    is_last_batch=final,
                )
            else:
                yield from finalize_pending(self.model, state)
            if (
                state["tracker_metadata"].get("max_obj_id", -1)
                >= self.settings.max_cumulative_native_ids
            ):
                raise HTTPException(409, "reset_required")
            native._torch.cuda.synchronize()

    def prune(self, session):
        if session.native_state is not None and session.native_retention_installed:
            self.retention.prune(
                session.native_state, emitted_through=session.emitted_index
            )

    def memory_stats(self):
        return self.video.backend.memory_stats()

    def discard(self, session):
        if session.native_state is not None:
            # Upstream reset loops over expired input frame indices; releasing this
            # exclusively owned graph is the supported forward-only teardown.
            session.native_state.clear()
            session.native_state = None
            session.native_retention_installed = False

    def close(self):
        self.gpu_ready = False
        self.retention = None
        self.model = None


class SessionEngine:
    def __init__(
        self,
        video_engine,
        settings=None,
        backend=None,
        *,
        profile=None,
        production=None,
        registry=None,
    ):
        self.video = video_engine
        self.settings = settings or SessionSettings.from_env()
        self._production = backend is None if production is None else production
        self.backend = backend or NativeContinuation(video_engine, self.settings)
        self.profile = profile
        self.registry = registry
        self.session: Session | None = None
        self._mutex = threading.RLock()
        self.ready = False
        self.error_code = None
        if registry is not None:
            registry.add(self)

    @property
    def backend_name(self):
        return self.profile.backend_name if self.profile is not None else BACKEND

    @property
    def max_pending_frames(self):
        return (
            self.profile.output_delay_frames
            if self.profile is not None
            else self.settings.hotstart_delay - 1
        )

    @property
    def gpu_ready(self):
        return (
            self.settings.enabled
            and self.ready
            and self._production
            and self.video.gpu_ready
            and getattr(self.backend, "gpu_ready", False) is True
        )

    @property
    def has_active_session(self):
        with self._mutex:
            return self.session is not None and (
                self.session.status == "active" or self.session.native_state is not None
            )

    def limits(self):
        return self.video.settings.limits() | {
            "startup_frames": self.profile.startup_frames if self.profile else 16,
            "max_append_frames": self.profile.max_append_frames if self.profile else 16,
            "hotstart_delay": self.max_pending_frames + 1,
            "max_pending_frames": self.max_pending_frames,
            "history_frames": self.settings.history_frames,
            "max_session_frames": self.settings.max_session_frames,
            "max_cumulative_native_ids": self.settings.max_cumulative_native_ids,
            "idle_seconds": self.settings.idle_seconds,
            "max_active_sessions": 1,
            "h264_transport": int(self.settings.h264_enabled),
        }

    def warmup(self):
        if not self.settings.enabled:
            return
        try:
            if self.video.settings.max_frames != 16:
                raise RuntimeError("session mode requires a 16-frame input capacity")
            if self.settings.h264_enabled:
                _h264_transport().ensure_h264_decoder_available()
            self.backend.warmup()
            self.ready = True
        except Exception:
            self.error_code = "session_initialization_failed"
            self.backend.close()
            raise

    def _base(self, session):
        return {
            "video_session_id": session.video_session_id,
            "stream_session_id": session.stream_session_id,
            "predictor_epoch": session.predictor_epoch,
            "backend": self.backend_name,
            "memory_scope": "session",
            "state": session.status,
        }

    def create(self, payload):
        raw = _json(payload)
        if set(raw) != {"stream_session_id"} or not _identifier(
            raw["stream_session_id"]
        ):
            raise HTTPException(400, "invalid session creation request")
        ownership = self.registry.mutex if self.registry is not None else self._mutex
        with ownership, self._mutex:
            if self.registry is not None:
                self.registry.expire()
            else:
                self.expire()
            if (
                self.registry.has_active_session
                if self.registry is not None
                else self.has_active_session
            ):
                raise HTTPException(409, "active_session_exists")
            self.session = Session(
                uuid.uuid4().hex,
                raw["stream_session_id"],
                uuid.uuid4().hex,
                time.monotonic(),
            )
            return self._base(self.session) | {"limits": self.limits()}

    def _get(self, token):
        if self.session is None or token != self.session.video_session_id:
            raise HTTPException(404, "unknown video session")
        return self.session

    def expire(self):
        # Caller holds the shared admission slot; never runs concurrently with CUDA.
        with self._mutex:
            session = self.session
            if session is not None and (
                session.cancel.is_set()
                or time.monotonic() - session.touched > self.settings.idle_seconds
            ):
                self._discard(session)
                self.session = None

    def invalidate(self, token):
        # Safe from the event-loop thread while abandoned CUDA still executes.
        with self._mutex:
            if self.session is not None and self.session.video_session_id == token:
                self.session.cancel.set()
                self.session.status = "invalidated"
                self.session.last_response = None

    def _discard(self, session):
        had_state = session.status == "active" or session.native_state is not None
        self.backend.discard(session)
        session.sources.clear()
        session.status = "closed"
        if had_state and self.registry is not None:
            self.registry.note_state_discarded()

    def delete(self, token):
        with self._mutex:
            session = self._get(token)
            discarded = len(session.sources)
            session.cancel.set()
            self._discard(session)
            session.last_response = None
            return self._base(session) | {"discarded_pending_frames": discarded}

    def close(self):
        self.ready = False
        if self.session is not None:
            self._discard(self.session)
            self.session = None
        self.backend.close()

    def _check_append(self, session, batch, final, digest):
        """Read-only checks, repeated under GPU admission before committing."""
        if batch.stream_session_id != session.stream_session_id:
            raise HTTPException(409, "source_session_mismatch")
        if session.last_request_id == batch.batch_id:
            if digest != session.last_payload_hash:
                raise HTTPException(409, "request_id_conflict")
            if session.last_response is not None:
                return session.last_response
        if session.status != "active" or session.cancel.is_set():
            raise HTTPException(409, "reset_required")
        limits = self.limits()
        if len(batch.frames) > limits["max_append_frames"]:
            raise HTTPException(413, "append exceeds route frame capacity")
        if not session.total_frames and (
            not batch.frames
            or (not final and len(batch.frames) < limits["startup_frames"])
        ):
            raise HTTPException(
                409, f"startup_requires_{limits['startup_frames']}_frames"
            )
        if (
            batch.frames
            and session.accepted is not None
            and (
                batch.frames[0].frame_sequence <= session.accepted.frame_sequence
                or batch.frames[0].timestamp_seconds
                <= session.accepted.timestamp_seconds
            )
        ):
            raise HTTPException(409, "source_frames_must_increase")
        return None

    def _decode_append(self, prepared, *, decode_jpeg, check):
        """Bounded CPU decoding/resizing only; caller owns image cleanup."""
        batch = prepared.batch
        settings = self.video.settings
        check()
        if isinstance(batch, SessionVideoBatch):
            transport = _h264_transport()
            try:
                prepared.images = transport.decode_h264_batch(
                    batch.video_bytes,
                    expected_frames=len(batch.frames),
                    max_image_pixels=settings.max_image_pixels,
                    max_total_image_pixels=settings.max_total_image_pixels,
                    max_side=1280,
                    max_encoded_bytes=min(
                        MAX_H264_BYTES, settings.max_total_jpeg_bytes
                    ),
                    check=check,
                )
            except transport.H264TransportError as exc:
                raise HTTPException(
                    exc.status_code, "invalid session video bitstream"
                ) from None
            if len(prepared.images) != len(batch.frames):
                raise RuntimeError("video decoder omitted source frames")
        total_pixels = 0
        for offset, frame in enumerate(batch.frames):
            check()
            if isinstance(batch, SessionVideoBatch):
                image = prepared.images[offset]
            else:
                image = decode_jpeg(frame.jpeg_bytes, settings)
                prepared.images.append(image)
            check()
            total_pixels += image.width * image.height
            if total_pixels > settings.max_total_image_pixels:
                raise HTTPException(413, "video source pixel budget exceeded")
            prepared.source_sizes.append(image.size)
            image.thumbnail((settings.inference_max_side,) * 2)
        check()

    def prepare_append(self, token, payload, *, decode_jpeg, check, diagnostics):
        check()
        batch, final, digest = parse_append(
            payload, self.video.settings, h264_enabled=self.settings.h264_enabled,
            diagnostics=diagnostics,
        )
        prepared = _PreparedAppend(batch, final, digest)
        # Do not expire/discard state here: those operations can access CUDA.
        with self._mutex:
            session = self._get(token)
            replay = self._check_append(session, batch, final, digest)
            over_quota = (
                session.total_frames + len(batch.frames)
                > self.settings.max_session_frames
            )
        # Cached output and quota cleanup are handled under GPU admission. Neither
        # needs new images. The lane gate prevents another append overtaking us.
        if replay is not None or over_quota:
            return prepared
        started = time.perf_counter()
        try:
            self._decode_append(prepared, decode_jpeg=decode_jpeg, check=check)
            return prepared
        except BaseException:
            prepared.close()
            raise
        finally:
            diagnostics["preparation_decode_seconds"] = time.perf_counter() - started

    def append(
        self, token, payload, *, deadline, decode_jpeg, diagnostics=None, prepared=None
    ):
        timing = _AppendTiming(
            self.profile.route if self.profile else "sam31_session", diagnostics
        )
        outcome, status_code = "completed", 200
        try:
            return self._append(
                token, payload, deadline=deadline, decode_jpeg=decode_jpeg,
                timing=timing, prepared=prepared,
            )
        except BaseException as exc:
            outcome = "failed"
            status_code = (
                exc.status_code if isinstance(exc, HTTPException)
                else 504 if isinstance(exc, TimeoutError)
                else 503
            )
            raise
        finally:
            # Observability must not change request success or cleanup behavior.
            try:
                stats = getattr(self.backend, "memory_stats", None)
                if callable(stats):
                    timing.fields.update(stats())
            except Exception:
                pass
            timing.finish(outcome, status_code)

    def _append(self, token, payload, *, deadline, decode_jpeg, timing, prepared=None):
        if prepared is None:
            batch, final, digest = parse_append(
                payload, self.video.settings, h264_enabled=self.settings.h264_enabled,
                diagnostics=timing.fields,
            )
        else:
            batch, final, digest = prepared.batch, prepared.final, prepared.digest
            timing.fields.update(request_id=batch.batch_id, is_last_batch=final)
        timing.fields.update(
            input_frames=len(batch.frames),
            input_first_frame_sequence=(
                batch.frames[0].frame_sequence if batch.frames else None
            ),
            input_last_frame_sequence=(
                batch.frames[-1].frame_sequence if batch.frames else None
            ),
        )
        with self._mutex:
            self.expire()
            session = self._get(token)
            timing.observe_session(session)
            replay = self._check_append(session, batch, final, digest)
            if replay is not None:
                session.touched = time.monotonic()
                timing.mark("response_build")
                timing.fields.update(replayed=True, response_bytes=len(replay))
                return json.loads(replay)
            if (
                session.total_frames + len(batch.frames)
                > self.settings.max_session_frames
            ):
                self.invalidate(token)
                self._discard(session)
                raise HTTPException(409, "reset_required")
        owns_prepared = prepared is None
        if owns_prepared:
            prepared = _PreparedAppend(batch, final, digest)
        began_native = False
        committed_inputs = False
        try:

            def check():
                if session.cancel.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("session request deadline exceeded")

            source_size, output_size = session.source_size, session.output_size
            check()
            if owns_prepared:
                timing.mark("decode")
                self._decode_append(prepared, decode_jpeg=decode_jpeg, check=check)
            for image, size in zip(prepared.images, prepared.source_sizes, strict=True):
                check()
                if source_size is not None and source_size != size:
                    self.invalidate(token)
                    raise HTTPException(409, "reset_required")
                source_size = size
                if output_size is not None and image.size != output_size:
                    raise RuntimeError("session output resolution changed")
                output_size = image.size
            timing.mark("validation")
            start = session.total_frames
            sources = dict(session.sources)
            for offset, frame in enumerate(batch.frames):
                # Persist only exact source metadata, not uploaded JPEG bodies.
                sources[start + offset] = SourceFrame(
                    frame.frame_sequence, frame.timestamp_seconds, b""
                )
            if len(sources) > 32:
                raise RuntimeError("pending source metadata exceeded bound")
            with self._mutex:
                check()
                session.source_size, session.output_size = source_size, output_size
                session.sources = sources
                committed_inputs = True
            snapshots = []
            check()
            began_native = True
            timing.mark("native_advance")
            outputs = self.backend.run(session, prepared.images, final=final)
            try:
                output_iterator = iter(outputs)
                while True:
                    timing.mark("native_advance")
                    try:
                        index, output = next(output_iterator)
                    except StopIteration:
                        break
                    timing.mark("validation")
                    check()
                    if (
                        type(index) is not int
                        or index != session.emitted_index + 1
                        or index not in session.sources
                    ):
                        raise RuntimeError("native output violates source prefix")
                    source = session.sources[index]
                    serialize = getattr(
                        self.backend, "serialize_snapshot", native_snapshot
                    )
                    timing.mark("snapshot")
                    snapshots.append(
                        serialize(
                            output,
                            batch=VideoBatch(
                                session.stream_session_id, batch.batch_id, (source,)
                            ),
                            index=0,
                            epoch=session.predictor_epoch,
                            image_size=session.output_size,
                            max_objects=self.video.settings.max_objects,
                        )
                    )
                    timing.mark("validation")
                    if any(
                        int(item["attributes"]["native_track"]["object_id"])
                        >= self.settings.max_cumulative_native_ids
                        for item in snapshots[-1]["objects"]
                    ):
                        raise HTTPException(409, "reset_required")
                    session.emitted_index = index
                    session.emitted = source
                    del session.sources[index]
                    timing.fields["output_frames"] += 1
                    if timing.fields["output_first_frame_sequence"] is None:
                        timing.fields["output_first_frame_sequence"] = (
                            source.frame_sequence
                        )
                    timing.fields["output_last_frame_sequence"] = source.frame_sequence
            except BaseException:
                timing.fields["failure_phase"] = timing.phase
                raise
            finally:
                timing.mark("native_advance")
                if hasattr(outputs, "close"):
                    outputs.close()
            timing.mark("validation")
            session.total_frames += len(batch.frames)
            if batch.frames:
                last = batch.frames[-1]
                session.accepted = SourceFrame(
                    last.frame_sequence, last.timestamp_seconds, b""
                )
            if final and session.sources:
                raise RuntimeError("native EOF omitted arrived source frames")
            if (
                len(session.sources) != session.total_frames - session.emitted_index - 1
                or len(session.sources) > self.max_pending_frames
            ):
                raise RuntimeError("native pending count violated bound")
            timing.mark("retention")
            self.backend.prune(session)
            check()
            timing.mark("response_build")
            route_metadata = None
            if self.profile is not None:
                metadata = getattr(self.backend, "route_metadata", None)
                route_metadata = (
                    metadata(session, snapshots) if metadata is not None else {}
                ) | {"route": self.profile.route}
            if final:
                timing.mark("retention")
                self._discard(session)
                timing.mark("response_build")
            accepted, emitted = session.accepted, session.emitted
            response = self._base(session) | {
                "request_id": batch.batch_id,
                "frames": snapshots,
                "accepted_frame_sequence": accepted.frame_sequence,
                "accepted_timestamp_seconds": accepted.timestamp_seconds,
                "processed_frame_sequence": accepted.frame_sequence,
                "processed_timestamp_seconds": accepted.timestamp_seconds,
                "emitted_frame_sequence": emitted.frame_sequence if emitted else None,
                "emitted_timestamp_seconds": emitted.timestamp_seconds
                if emitted
                else None,
                "pending_frame_count": len(session.sources),
            }
            if self.profile is not None:
                response["route_metadata"] = route_metadata
            timing.mark("response_serialize")
            encoded = json.dumps(
                response, separators=(",", ":"), allow_nan=False
            ).encode()
            timing.fields["response_bytes"] = len(encoded)
            timing.mark("validation")
            if len(encoded) > 2 * 1024 * 1024:
                raise RuntimeError("session response exceeded byte bound")
            check()
            with self._mutex:
                # An HTTP cancellation can arrive between serialization and
                # acquiring this lock. It must never be resurrected by replay.
                check()
                session.last_request_id = batch.batch_id
                session.last_payload_hash = digest
                session.last_response = encoded
                session.touched = time.monotonic()
            return response
        except BaseException as exc:
            timing.fields.setdefault("failure_phase", timing.phase)
            timing.mark("cleanup")
            if began_native or committed_inputs or session.cancel.is_set():
                self.invalidate(token)
                self._discard(session)
                session.status = "invalidated"
                if began_native and not isinstance(exc, (HTTPException, TimeoutError)):
                    self.ready = False
                    self.error_code = "native_session_failed"
                    # Both APIs share the same CUDA modules. A native failure
                    # cannot leave the sibling batch API attesting readiness.
                    self.video.ready = False
                    self.video.error_code = "native_session_failed"
            raise
        finally:
            timing.mark("cleanup")
            if owns_prepared:
                prepared.close()
            timing.observe_session(session)


def register_session_routes(
    app,
    engine,
    admission,
    decode_jpeg,
    *,
    api_prefix="/v1/video",
    health_path="/health/video/session",
    legacy=True,
    queued_admission=False,
    preparation_admission=None,
):
    if preparation_admission is not None and not queued_admission:
        raise ValueError("CPU preparation requires queued camera admission")
    if legacy:
        app.state.session_engine = engine
    # One request per camera, including abandoned work until its thread exits.
    # Upload/preparation must not hold the GPU or let the same lane overtake.
    upload_admission = threading.Lock() if queued_admission else None

    @app.get(health_path)
    async def health():
        return JSONResponse(
            {
                "enabled": engine.settings.enabled,
                "ready": engine.gpu_ready,
                "gpu_ready": engine.gpu_ready,
                "backend": engine.backend_name,
                "memory_scope": "session",
                "native_sam_video_tracking": (
                    engine.profile.native_sam_video_tracking if engine.profile else True
                ),
                "sam3_commit": SAM3_COMMIT,
                "limits": engine.limits(),
                "error_code": engine.error_code,
                "active_session": (
                    engine.registry.has_active_session
                    if not legacy and engine.registry is not None
                    else engine.has_active_session
                ),
            },
            status_code=200 if engine.gpu_ready else 503,
        )

    async def handle(request, operation, token=None):
        started = time.perf_counter()
        received_bytes = 0
        admission_started = admission_finished = None
        body_started = body_finished = dispatch_started = None
        preparation_wait_started = preparation_started = preparation_finished = None
        worker_started = processing_finished = None
        response_encode_seconds = 0.0
        diagnostics = {
            "route": engine.profile.route if engine.profile else "sam31_session",
            "api_prefix": api_prefix,
            "request_id": None,
        }
        phase = "admission"

        def log_request(message, outcome, status_code):
            finished = time.perf_counter()
            _timing_log(
                message,
                diagnostics | {
                    "operation": operation,
                    "outcome": outcome,
                    "status_code": status_code,
                    "phase": phase,
                    "received_bytes": received_bytes,
                    "queue_wait_seconds": (
                        (admission_finished or finished) - admission_started
                        if admission_started is not None else 0.0
                    ),
                    "body_read_seconds": (
                        (body_finished or finished) - body_started
                        if body_started is not None else 0.0
                    ),
                    "preparation_wait_seconds": (
                        (preparation_started or finished) - preparation_wait_started
                        if preparation_wait_started is not None else 0.0
                    ),
                    "preparation_seconds": (
                        (preparation_finished or finished) - preparation_started
                        if preparation_started is not None else 0.0
                    ),
                    "worker_dispatch_seconds": (
                        worker_started - dispatch_started
                        if worker_started is not None and dispatch_started is not None
                        else None
                    ),
                    "processing_seconds": (
                        (processing_finished or finished) - dispatch_started
                        if dispatch_started is not None else 0.0
                    ),
                    "http_response_encode_seconds": response_encode_seconds,
                    "total_seconds": finished - started,
                    "timing_basis": "host_wall_clock",
                },
                level=logging.INFO if outcome == "completed" else logging.WARNING,
            )

        # Cleanup must remain possible after a background/model failure removes
        # readiness. DELETE still uses the same admission, token and deadline
        # checks, and never runs inference or opens a replacement session.
        if operation != "delete" and (
            not engine.settings.enabled
            or not engine.ready
            or (engine._production and not engine.gpu_ready)
        ):
            log_request("Native session request rejected", "rejected", 503)
            raise HTTPException(503, "native video sessions unavailable")
        deadline = time.monotonic() + engine.video.settings.request_timeout_seconds
        if upload_admission is not None and not upload_admission.acquire(False):
            log_request("Native session request rejected", "rejected", 429)
            raise HTTPException(429, "camera busy", headers={"Retry-After": "1"})
        acquired = False
        transferred = False
        abandoned = threading.Event()
        task = None
        resources = _RequestResources(upload_admission)

        async def acquire_gpu():
            nonlocal acquired, admission_started, admission_finished, phase
            phase = "admission"
            admission_started = time.perf_counter()
            if queued_admission:
                acquired = await admission.acquire_async(deadline)
            else:
                acquired = admission.acquire(blocking=False)
            admission_finished = time.perf_counter()
            if not acquired:
                raise HTTPException(
                    429, "inference busy", headers={"Retry-After": "1"}
                )

        def retire_abandoned_create(done):
            if (
                operation != "create"
                or not abandoned.is_set()
                or not done.done()
                or done.cancelled()
            ):
                return
            if done.exception() is not None:
                return
            created = done.result()
            # The token comparison inside invalidate cannot affect a newer session.
            engine.invalidate(created["video_session_id"])
            if admission.acquire(blocking=False):
                try:
                    engine.expire()
                finally:
                    admission.release()

        def consume_preparation(done):
            if not done.cancelled():
                done.exception()

        try:
            if not queued_admission:
                await acquire_gpu()
            phase = "upload"
            body_started = time.perf_counter()

            async def receive():
                nonlocal received_bytes
                body = bytearray()
                cap = (
                    engine.video.settings.max_request_bytes
                    if operation == "append"
                    else 4096
                )
                async for chunk in request.stream():
                    received_bytes += len(chunk)
                    if len(body) + len(chunk) > cap:
                        raise HTTPException(413, "session request exceeds byte bound")
                    body.extend(chunk)
                return bytes(body)

            payload = await asyncio.wait_for(
                receive(), max(0, deadline - time.monotonic())
            )
            body_finished = time.perf_counter()
            if operation == "delete" and payload:
                raise HTTPException(400, "session delete body must be empty")
            if operation == "append" and preparation_admission is not None:
                phase = "preparation_wait"
                preparation_wait_started = time.perf_counter()
                if not await preparation_admission.acquire_async(deadline):
                    raise HTTPException(
                        429, "preparation busy", headers={"Retry-After": "1"}
                    )
                resources.preparation = preparation_admission
                preparation_started = time.perf_counter()
                phase = "preparation"

                def check_preparation():
                    if abandoned.is_set() or time.monotonic() >= deadline:
                        raise TimeoutError("session preparation abandoned")

                def prepare():
                    resources.prepared = engine.prepare_append(
                        token, payload, decode_jpeg=decode_jpeg,
                        check=check_preparation, diagnostics=diagnostics,
                    )

                preparation_task = asyncio.create_task(
                    asyncio.to_thread(resources.run, prepare)
                )
                preparation_task.add_done_callback(consume_preparation)
                await asyncio.wait_for(
                    asyncio.shield(preparation_task),
                    max(0, deadline - time.monotonic()),
                )
                preparation_finished = time.perf_counter()
            if queued_admission:
                await acquire_gpu()
            phase = "processing"
            dispatch_started = time.perf_counter()

            def process():
                nonlocal worker_started
                worker_started = time.perf_counter()
                if abandoned.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("session operation abandoned")
                if operation == "create":
                    return engine.create(payload)
                if operation == "delete":
                    return engine.delete(token)
                return engine.append(
                    token, payload, deadline=deadline, decode_jpeg=decode_jpeg,
                    diagnostics=diagnostics, prepared=resources.prepared,
                )

            def invoke():
                try:
                    return resources.run(process)
                finally:
                    admission.release()

            task = asyncio.create_task(asyncio.to_thread(invoke))
            transferred = True

            def consume(done):
                if not done.cancelled():
                    done.exception()
                    retire_abandoned_create(done)

            task.add_done_callback(consume)
            result = await asyncio.wait_for(
                asyncio.shield(task), max(0, deadline - time.monotonic())
            )
            processing_finished = time.perf_counter()
            phase = "response_encode"
            response = JSONResponse(result)
            response_encode_seconds = time.perf_counter() - processing_finished
            log_request("Native session request completed", "completed", 200)
            return response
        except TimeoutError:
            abandoned.set()
            if task is not None:
                retire_abandoned_create(task)
            if token is not None and (task is not None or not queued_admission):
                engine.invalidate(token)
            log_request("native session deadline exceeded", "timeout", 504)
            raise HTTPException(504, "native session deadline exceeded") from None
        except asyncio.CancelledError:
            abandoned.set()
            if task is not None:
                retire_abandoned_create(task)
            if token is not None and (task is not None or not queued_admission):
                engine.invalidate(token)
            log_request("Native session request cancelled", "cancelled", 499)
            raise
        except HTTPException as exc:
            log_request("Native session request failed", "failed", exc.status_code)
            raise
        except Exception:
            LOGGER.exception("Native session operation failed: operation=%s", operation)
            if token is not None and (task is not None or not queued_admission):
                engine.invalidate(token)
            log_request("native session operation failed", "failed", 503)
            raise HTTPException(503, "native session unavailable") from None
        finally:
            try:
                if acquired and not transferred:
                    try:
                        engine.expire()
                    finally:
                        admission.release()
            finally:
                resources.close()

    @app.post(api_prefix + "/sessions")
    async def create(request: Request):
        return await handle(request, "create")

    @app.post(api_prefix + "/sessions/{video_session_id}/append")
    async def append(video_session_id: str, request: Request):
        return await handle(request, "append", video_session_id)

    @app.delete(api_prefix + "/sessions/{video_session_id}")
    async def delete(video_session_id: str, request: Request):
        return await handle(request, "delete", video_session_id)
