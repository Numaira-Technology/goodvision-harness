"""YOLO / RF-DETR rich-label worker — same /v1/labels contract as the Paddle one.

A single-frame detector (detectors.build_detector) feeds the detector-agnostic
IoUTracker, so each object carries a stable track_id exactly as the harness
RichLabelHTTPClient expects. No masks (optional in the contract); boxes only.

Request:  POST /v1/labels
  headers: x-goodvision-stream-session-id, x-goodvision-frame-sequence,
           x-goodvision-timestamp-seconds; Content-Type: image/jpeg; body = JPEG
Response: {stream_session_id, frame_sequence, timestamp_seconds, objects:[...]}
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np
from detectors import MODEL_REGISTRY, Detector, build_detector
from fastapi import FastAPI, HTTPException, Request
from tracker import IoUTracker

LOGGER = logging.getLogger("uvicorn.error")
MAX_JPEG_BYTES = 8 * 1024 * 1024
MAX_TRACKING_SESSIONS = 128
# Per-request override of the active model, set by the harness from the console
# selection (see rich_label_models.DETECTOR_MODEL_HEADER). Absent => default.
DETECTOR_MODEL_HEADER = "x-goodvision-detector-model"


class RichLabelBusyError(RuntimeError):
    """Another inference is already running on this single-GPU worker."""


@dataclass
class _SessionState:
    active_model: str
    tracker: IoUTracker = field(default_factory=IoUTracker)
    last_source_key: tuple[float, int] | None = None


class DetectorEngine:
    """Serves one or more detector models, switchable per request.

    Models load only for a real input frame; no startup or dummy inference. Each session keeps its
    own tracker and source ordering. Switching a session's model resets only its
    tracker, since boxes from a different detector cannot share track continuity.
    The least recently used session is evicted when the session limit is reached.
    """

    def __init__(self) -> None:
        self.default_model = os.environ.get("DETECTOR_MODEL", "yolo26-s").strip()
        self._detectors: dict[str, Detector] = {}
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()

    def _get_detector(self, model_key: str) -> Detector:
        detector = self._detectors.get(model_key)
        if detector is None:
            detector = build_detector(model_key)  # validates against MODEL_REGISTRY
            self._detectors[model_key] = detector
        return detector

    def resolve_model(self, requested: str | None) -> str:
        key = (requested or self.default_model).strip()
        if key not in MODEL_REGISTRY:
            raise ValueError(
                f"unknown detector model {key!r}; "
                f"choose one of {sorted(MODEL_REGISTRY)}"
            )
        return key

    def infer(
        self,
        image: np.ndarray,
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
        model_key: str | None = None,
    ) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RichLabelBusyError("rich-label inference is already running")
        try:
            active_model = self.resolve_model(model_key)
            state = self._sessions.get(stream_session_id)
            if (
                state is not None
                and state.last_source_key is not None
                and (
                    timestamp_seconds < state.last_source_key[0]
                    or frame_sequence <= state.last_source_key[1]
                )
            ):
                raise ValueError("source frame metadata must increase")
            detector = self._get_detector(active_model)
            detections = detector.detect(image)
            if state is None:
                state = _SessionState(active_model)
                self._sessions[stream_session_id] = state
                if len(self._sessions) > MAX_TRACKING_SESSIONS:
                    self._sessions.popitem(last=False)
            elif state.active_model != active_model:
                state.tracker.reset()
                state.active_model = active_model
            self._sessions.move_to_end(stream_session_id)
            tracker = state.tracker
            tracks = tracker.update(
                detections,
                frame_sequence=frame_sequence,
                timestamp_seconds=timestamp_seconds,
            )
            state.last_source_key = (timestamp_seconds, frame_sequence)
            height, width = image.shape[:2]
            return {
                "detector_model": active_model,
                "detector_provenance": {
                    "detector_model": active_model,
                    "weights_sha256": detector.provenance["weights"]["sha256"],
                    "actual_parameter_device": detector.provenance["actual_parameter_device"],
                },
                "stream_session_id": stream_session_id,
                "frame_sequence": frame_sequence,
                "timestamp_seconds": timestamp_seconds,
                "objects": [
                    {
                        "track_id": track.track_id,
                        "label": track.label,
                        "confidence": round(track.confidence, 6),
                        "bbox": {
                            "x1": track.bbox[0] / width,
                            "y1": track.bbox[1] / height,
                            "x2": track.bbox[2] / width,
                            "y2": track.bbox[3] / height,
                        },
                        "attributes": {},
                        "observation_state": (
                            "observed"
                            if track.last_update_tick == tracker._update_tick
                            else "predicted"
                        ),
                        "last_observed_timestamp_seconds": (
                            track.last_observed_timestamp_seconds
                        ),
                        "last_observed_frame_sequence": track.last_frame_sequence,
                    }
                    for track in tracks
                ],
            }
        finally:
            self._lock.release()


def _required_header(request: Request, name: str) -> str:
    value = request.headers.get(name, "").strip()
    if not value or len(value) > 160:
        raise HTTPException(status_code=400, detail=f"missing {name}")
    return value


def create_app() -> FastAPI:
    application = FastAPI(title="Detector Rich Label", docs_url=None, redoc_url=None)
    engine = DetectorEngine()

    async def _infer_request(request: Request) -> dict:
        import asyncio

        session_id = _required_header(request, "x-goodvision-stream-session-id")
        try:
            frame_sequence = int(
                _required_header(request, "x-goodvision-frame-sequence")
            )
            timestamp_seconds = float(
                _required_header(request, "x-goodvision-timestamp-seconds")
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="invalid frame metadata"
            ) from exc
        if (
            frame_sequence < 0
            or not math.isfinite(timestamp_seconds)
            or timestamp_seconds < 0
        ):
            raise HTTPException(status_code=400, detail="invalid frame metadata")
        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type.lower() != "image/jpeg":
            raise HTTPException(status_code=415, detail="expected image/jpeg")
        payload = await request.body()
        if not payload or len(payload) > MAX_JPEG_BYTES:
            raise HTTPException(status_code=413, detail="invalid JPEG size")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="invalid JPEG")
        requested_model = request.headers.get(DETECTOR_MODEL_HEADER, "").strip() or None
        try:
            return await asyncio.to_thread(
                engine.infer,
                image,
                stream_session_id=session_id,
                frame_sequence=frame_sequence,
                timestamp_seconds=timestamp_seconds,
                model_key=requested_model,
            )
        except RichLabelBusyError as exc:
            raise HTTPException(
                status_code=429, detail="inference busy", headers={"Retry-After": "1"}
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            LOGGER.exception("detector inference failed")
            raise HTTPException(
                status_code=503, detail="inference unavailable"
            ) from None

    # The harness RichLabelHTTPClient posts to /v1/tracks in tracking mode
    # (mode="tracks") and /v1/labels otherwise. This worker always emits the
    # observation metadata the tracking contract requires, so both routes share
    # one handler; the client validates that metadata on the /v1/tracks path.
    application.post("/v1/labels")(_infer_request)
    application.post("/v1/tracks")(_infer_request)

    return application


app = create_app()

