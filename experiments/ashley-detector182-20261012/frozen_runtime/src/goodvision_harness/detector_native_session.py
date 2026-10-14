"""Ordered detector-frame adapter for the Ashley native scene path.

The worker owns IoU IDs. This local session only binds those IDs to a source
epoch; it does not claim SAM video memory or invent detections on empty frames.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field, replace
from uuid import uuid4

from .rich_labeling import (
    RichLabelHTTPClient,
    RichLabelProtocolError,
    RichLabelSnapshot,
)
from .video_tracking import SerializedVideoPayload, VideoFrame, _text, _validate_frames
from .video_tracking_session import PreparedSessionAppend


BACKEND = "detector_stream"
MODELS = frozenset({"yolo26-s", "rfdetr-l"})


@dataclass(frozen=True, slots=True)
class DetectorSession:
    video_session_id: str
    stream_session_id: str
    predictor_epoch: str
    limits: dict[str, int] = field(default_factory=lambda: {
        "startup_frames": 1, "max_append_frames": 4,
    })


@dataclass(frozen=True, slots=True)
class DetectorTrackingResult:
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
    emitted_frame_sequence: int
    emitted_timestamp_seconds: float
    pending_frame_count: int = 0
    memory_scope: str = "session"
    backend: str = BACKEND
    detector_model: str = ""
    detector_provenance: tuple[dict[str, str], ...] = ()
    # The original worker predicted tracks are preserved for offline audit.
    # Native scene observations below contain only source-observed boxes.
    predicted_frames: tuple[dict, ...] = ()

    @property
    def completed_frame_sequence(self) -> int:
        return self.emitted_frame_sequence

    @property
    def completed_timestamp_seconds(self) -> float:
        return self.emitted_timestamp_seconds

    @property
    def batch_id(self) -> str:
        return self.request_id


class DetectorNativeSessionAdapter:
    """Present ordered `/v1/tracks` frames to the durable native coordinator."""

    backend = BACKEND
    startup_frames = 1
    max_append_frames = 4

    def __init__(self, endpoint: str, *, model: str, timeout_seconds: float = 75.0):
        if model not in MODELS:
            raise ValueError("unsupported detector model")
        self.model = model
        self._timeout_seconds = timeout_seconds
        self._client = RichLabelHTTPClient(
            endpoint, timeout_seconds=timeout_seconds, mode="tracks",
            include_attributes=False, expected_detector_model=model,
        )
        self._operation = threading.Lock()
        self._session: DetectorSession | None = None
        self._last_stamp: tuple[int, float] | None = None
        self._provenance: dict[str, str] | None = None
        self._closed = False

    def gpu_ready(self) -> bool:
        raise RuntimeError("detector readiness is established by the first real frame")

    def open_session(self, stream_session_id: str) -> DetectorSession:
        _text(stream_session_id, "stream_session_id")
        with self._operation:
            if self._closed or self._session is not None:
                raise RichLabelProtocolError("detector session is not reusable")
            session = DetectorSession(uuid4().hex, stream_session_id, uuid4().hex)
            self._session = session
            self._last_stamp = None
            self._provenance = None
            return session

    def prepare_append(
        self, frames, *, stream_session_id: str, request_id: str | None = None,
    ) -> PreparedSessionAppend:
        frames = tuple(frames)
        _validate_frames(frames)
        if len(frames) > self.max_append_frames:
            raise ValueError("detector append exceeds frame cap")
        session = self._session
        if session is None or session.stream_session_id != stream_session_id:
            raise RichLabelProtocolError("detector source session mismatch")
        request_id = _text(request_id or uuid4().hex, "request_id")
        body = json.dumps({
            "backend": BACKEND, "detector_model": self.model,
            "session": session.video_session_id, "request_id": request_id,
            "frames": [
                [f.frame_sequence, f.timestamp_seconds,
                 hashlib.sha256(f.jpeg_bytes).hexdigest()]
                for f in frames
            ],
        }, separators=(",", ":"), allow_nan=False).encode()
        payload = SerializedVideoPayload(body, {}, 0.0, {})
        return PreparedSessionAppend(
            session.video_session_id, stream_session_id, request_id, frames,
            payload, hashlib.sha256(body).hexdigest(),
            time.monotonic() + self._timeout_seconds,
        )

    def append(
        self, frames, *, stream_session_id: str, is_last_batch: bool = False,
        request_id: str | None = None, prepared: PreparedSessionAppend | None = None,
    ) -> DetectorTrackingResult:
        frames = tuple(frames)
        if frames:
            _validate_frames(frames)
        if not frames and not is_last_batch:
            raise ValueError("empty detector append requires EOF")
        if len(frames) > self.max_append_frames:
            raise ValueError("detector append exceeds frame cap")
        with self._operation:
            session = self._session
            if self._closed or session is None or session.stream_session_id != stream_session_id:
                raise RichLabelProtocolError("detector session is not active")
            if prepared is not None:
                if (
                    prepared.video_session_id != session.video_session_id
                    or prepared.stream_session_id != stream_session_id
                    or prepared.request_id != request_id
                    or prepared.frames != frames
                    or hashlib.sha256(prepared.payload.body).hexdigest()
                    != prepared.payload_hash
                ):
                    raise RichLabelProtocolError("prepared detector append changed")
            if frames and self._last_stamp is not None and (
                frames[0].frame_sequence <= self._last_stamp[0]
                or frames[0].timestamp_seconds <= self._last_stamp[1]
            ):
                raise RichLabelProtocolError("detector source order regressed")
            if not frames and self._last_stamp is None:
                raise RichLabelProtocolError("detector EOF has no source frame")
            observations = []
            predicted_frames = []
            provenance_frames = []
            for frame in frames:
                raw = self._client.infer(
                    frame.jpeg_bytes, stream_session_id=stream_session_id,
                    frame_sequence=frame.frame_sequence,
                    timestamp_seconds=frame.timestamp_seconds,
                )
                provenance = self._client.last_detector_provenance
                if provenance is None or (
                    self._provenance is not None and provenance != self._provenance
                ):
                    raise RichLabelProtocolError("detector provenance changed within session")
                self._provenance = dict(provenance)
                provenance_frames.append(dict(provenance))
                observed, predicted = [], []
                for item in raw.objects:
                    if item.label != "person":
                        continue
                    if item.observation_state == "predicted":
                        predicted.append(item.as_dict())
                        continue
                    if (
                        item.observation_state != "observed"
                        or item.last_observed_frame_sequence != frame.frame_sequence
                        or not math.isclose(
                            item.last_observed_timestamp_seconds,
                            frame.timestamp_seconds, rel_tol=0, abs_tol=1e-6,
                        )
                    ):
                        raise RichLabelProtocolError("detector observation source mismatch")
                    observed.append(replace(item, attributes={
                        **item.attributes,
                        "native_track": {
                            "backend": BACKEND,
                            "epoch": session.predictor_epoch,
                            "object_id": item.track_id,
                        },
                    }))
                if len({item.track_id for item in observed}) != len(observed):
                    raise RichLabelProtocolError("duplicate detector IoU ID")
                observations.append(RichLabelSnapshot(
                    stream_session_id, frame.frame_sequence,
                    frame.timestamp_seconds, tuple(observed),
                ))
                predicted_frames.append({
                    "frame_sequence": frame.frame_sequence,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "objects": predicted,
                })
                self._last_stamp = frame.frame_sequence, frame.timestamp_seconds
            stamp = self._last_stamp
            if is_last_batch:
                self._session = None
            return DetectorTrackingResult(
                session.video_session_id, stream_session_id,
                _text(request_id or uuid4().hex, "request_id"),
                session.predictor_epoch, "closed" if is_last_batch else "active",
                tuple(observations), stamp[0], stamp[1], stamp[0], stamp[1],
                stamp[0], stamp[1], detector_model=self.model,
                detector_provenance=tuple(provenance_frames),
                predicted_frames=tuple(predicted_frames),
            )

    def discard_session(self, *, timeout_seconds: float = 2.0) -> None:
        del timeout_seconds
        with self._operation:
            self._session = None
            self._last_stamp = None
            self._provenance = None

    def close(self) -> None:
        with self._operation:
            self._closed = True
            self._session = None
            self._client.close()
