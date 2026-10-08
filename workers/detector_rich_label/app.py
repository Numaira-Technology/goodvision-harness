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

import cv2
import numpy as np
from detectors import MODEL_REGISTRY, Detector, build_detector
from fastapi import FastAPI, HTTPException, Request
from tracker import IoUTracker

LOGGER = logging.getLogger("uvicorn.error")
MAX_JPEG_BYTES = 8 * 1024 * 1024
# Per-request override of the active model, set by the harness from the console
# selection (see rich_label_models.DETECTOR_MODEL_HEADER). Absent => default.
DETECTOR_MODEL_HEADER = "x-goodvision-detector-model"


class RichLabelBusyError(RuntimeError):
    """Another inference is already running on this single-GPU worker."""


class DetectorEngine:
    """Serves one or more detector models, switchable per request.

    The default model (DETECTOR_MODEL) is warmed at startup; any other model the
    console selects is lazy-built and cached on first use. Switching the active
    model resets the tracker, since boxes from a different detector cannot share
    track_id continuity.
    """

    def __init__(self) -> None:
        self.default_model = os.environ.get("DETECTOR_MODEL", "yolo26-s").strip()
        self._detectors: dict[str, Detector] = {}
        self._warmed_models: set[str] = set()
        self.tracker = IoUTracker()
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._session_id: str | None = None
        self._active_model: str | None = None
        self._last_source_key: tuple[float, int] | None = None

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

    def warmup(self) -> None:
        self._get_detector(self.default_model).warmup()
        self._warmed_models.add(self.default_model)

    @property
    def gpu_ready(self) -> bool:
        detector = self._detectors.get(self.default_model)
        return (
            self.default_model in self._warmed_models
            and detector is not None
            and detector.gpu_ready
        )

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
            with self._state_lock:
                if (
                    stream_session_id != self._session_id
                    or active_model != self._active_model
                ):
                    # New session or a mid-stream model switch: track_ids restart.
                    self.tracker.reset()
                    self._session_id = stream_session_id
                    self._active_model = active_model
                    self._last_source_key = None
                if self._last_source_key is not None and (
                    timestamp_seconds < self._last_source_key[0]
                    or frame_sequence <= self._last_source_key[1]
                ):
                    raise ValueError("source frame metadata must increase")
            detector = self._get_detector(active_model)
            if active_model not in self._warmed_models:
                detector.warmup()
                self._warmed_models.add(active_model)
            detections = detector.detect(image)
            with self._state_lock:
                tracks = self.tracker.update(
                    detections,
                    frame_sequence=frame_sequence,
                    timestamp_seconds=timestamp_seconds,
                )
                self._last_source_key = (timestamp_seconds, frame_sequence)
                height, width = image.shape[:2]
                return {
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
                                if track.last_update_tick == self.tracker._update_tick
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

    @application.on_event("startup")
    def _startup() -> None:
        try:
            engine.warmup()
            LOGGER.info("detector rich-label ready model=%s gpu=%s available=%s",
                        engine.default_model, engine.gpu_ready,
                        sorted(MODEL_REGISTRY))
        except Exception:
            LOGGER.exception("detector warmup failed")

    @application.get("/health/live")
    def _live() -> dict:
        return {"alive": True, "model": engine.default_model}

    @application.get("/health/ready")
    def _ready() -> dict:
        return {"ready": True, "gpu_ready": engine.gpu_ready,
                "model": engine.default_model, "available": sorted(MODEL_REGISTRY)}

    @application.get("/health/tracking")
    def _tracking() -> dict:
        return {"ready": True, "gpu_ready": engine.gpu_ready,
                "model": engine.default_model, "available": sorted(MODEL_REGISTRY)}

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
            raise HTTPException(status_code=429, detail="inference busy",
                                headers={"Retry-After": "1"}) from exc
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
