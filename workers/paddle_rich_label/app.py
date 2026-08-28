"""Isolated PaddleX object, tracking, and pedestrian-attribute worker.

The Harness sends display-resolution JPEGs over a narrow HTTP contract. Paddle
dependencies stay in this process so their NumPy/OpenCV constraints cannot
affect the VLM, InsightFace, WebRTC, or compositor runtimes.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response

LOGGER = logging.getLogger(__name__)
MAX_JPEG_BYTES = 8 * 1024 * 1024
PERSON_LABELS = frozenset({"person", "pedestrian"})


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(slots=True)
class Track:
    track_id: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    last_frame_sequence: int
    attributes: dict[str, Any] = field(default_factory=dict)
    last_attribute_frame: int | None = None


class IoUTracker:
    """Small deterministic single-camera tracker around Paddle detections."""

    def __init__(self, *, match_iou: float = 0.3, max_idle_frames: int = 20) -> None:
        if not 0.0 < match_iou <= 1.0:
            raise ValueError("match_iou must be between 0 and 1")
        if max_idle_frames < 1:
            raise ValueError("max_idle_frames must be positive")
        self.match_iou = float(match_iou)
        self.max_idle_frames = int(max_idle_frames)
        self._tracks: dict[str, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_sequence: int,
    ) -> tuple[Track, ...]:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_sequence - track.last_frame_sequence <= self.max_idle_frames
        }
        available = set(self._tracks)
        current: list[Track] = []
        for detection in sorted(
            detections,
            key=lambda item: item.confidence,
            reverse=True,
        ):
            candidates = [
                (
                    self._tracks[track_id],
                    _iou(self._tracks[track_id].bbox, detection.bbox),
                )
                for track_id in available
                if self._tracks[track_id].label == detection.label
            ]
            candidates.sort(key=lambda item: item[1], reverse=True)
            if candidates and candidates[0][1] >= self.match_iou:
                track = candidates[0][0]
                available.remove(track.track_id)
                track.confidence = detection.confidence
                track.bbox = detection.bbox
                track.last_frame_sequence = frame_sequence
            else:
                prefix = "P" if detection.label in PERSON_LABELS else "O"
                track_id = f"{prefix}{self._next_id:04d}"
                self._next_id += 1
                track = Track(
                    track_id=track_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    last_frame_sequence=frame_sequence,
                )
                self._tracks[track_id] = track
            current.append(track)
        return tuple(sorted(current, key=lambda item: item.track_id))


class PaddleRichModels:
    """Lazy PaddleX model facade; models are downloaded only inside the sidecar."""

    def __init__(self) -> None:
        self.device = os.environ.get("RICH_DEVICE", "cpu").strip() or "cpu"
        self.detector_name = (
            os.environ.get("RICH_DETECTOR_MODEL", "PP-YOLOE_plus-S").strip()
            or "PP-YOLOE_plus-S"
        )
        self.attribute_name = (
            os.environ.get(
                "RICH_ATTRIBUTE_MODEL", "PP-LCNet_x1_0_pedestrian_attribute"
            ).strip()
            or "PP-LCNet_x1_0_pedestrian_attribute"
        )
        self.detector_dir = os.environ.get("RICH_DETECTOR_MODEL_DIR", "").strip()
        self.attribute_dir = os.environ.get("RICH_ATTRIBUTE_MODEL_DIR", "").strip()
        self.detection_threshold = _env_float(
            "RICH_DETECTION_THRESHOLD", 0.45, minimum=0.0, maximum=1.0
        )
        self._detector: Any | None = None
        self._attribute: Any | None = None
        self._lock = threading.Lock()
        self._attribute_state_lock = threading.Lock()
        self._attribute_loading = False
        self._attribute_retry_at = 0.0
        self._attribute_retry_seconds = _env_float(
            "RICH_ATTRIBUTE_RETRY_SECONDS",
            60.0,
            minimum=1.0,
            maximum=3600.0,
        )

    @property
    def ready(self) -> bool:
        return self._detector is not None

    @property
    def attributes_ready(self) -> bool:
        return self._attribute is not None

    def ensure_detector_loaded(self) -> None:
        if self._detector is not None:
            return
        with self._lock:
            if self._detector is not None:
                return
            try:
                from paddlex import create_model
            except ImportError as exc:
                raise RuntimeError("PaddleX is not installed in the worker") from exc
            detector_kwargs = _model_kwargs(
                self.detector_name, self.detector_dir, self.device
            )
            self._detector = create_model(**detector_kwargs)

    def ensure_attribute_loaded(self) -> None:
        if self._attribute is not None:
            return
        with self._lock:
            if self._attribute is not None:
                return
            try:
                from paddlex import create_model
            except ImportError as exc:
                raise RuntimeError("PaddleX is not installed in the worker") from exc
            attribute_kwargs = _model_kwargs(
                self.attribute_name, self.attribute_dir, self.device
            )
            self._attribute = create_model(**attribute_kwargs)

    def prepare_attributes(self) -> bool:
        """Warm the optional attribute model without delaying detector boxes."""

        if self.attributes_ready:
            return True
        now = time.monotonic()
        with self._attribute_state_lock:
            if self.attributes_ready:
                return True
            if self._attribute_loading or now < self._attribute_retry_at:
                return False
            self._attribute_loading = True
            threading.Thread(
                target=self._warm_attribute_model,
                name="paddle-attribute-warmup",
                daemon=True,
            ).start()
            return False

    def _warm_attribute_model(self) -> None:
        error: Exception | None = None
        try:
            self.ensure_attribute_loaded()
        except Exception as exc:
            error = exc
        with self._attribute_state_lock:
            self._attribute_loading = False
            self._attribute_retry_at = (
                time.monotonic() + self._attribute_retry_seconds
                if error is not None
                else 0.0
            )
        if error is not None:
            LOGGER.warning(
                "pedestrian attribute model unavailable: %s",
                type(error).__name__,
            )

    def detect(self, image: np.ndarray[Any, Any]) -> tuple[Detection, ...]:
        self.ensure_detector_loaded()
        assert self._detector is not None
        results = self._detector.predict(
            image,
            batch_size=1,
            threshold=self.detection_threshold,
        )
        first = next(iter(results), None)
        payload = _model_result(first)
        detections: list[Detection] = []
        for raw in payload.get("boxes", []):
            if not isinstance(raw, Mapping):
                continue
            coordinate = raw.get("coordinate")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 4:
                continue
            try:
                bbox = tuple(float(value) for value in coordinate)
                confidence = float(raw.get("score"))
            except (TypeError, ValueError):
                continue
            label = str(raw.get("label", "object")).strip().lower()
            if (
                not label
                or not math.isfinite(confidence)
                or confidence < self.detection_threshold
                or not _valid_pixel_bbox(bbox, image.shape[1], image.shape[0])
            ):
                continue
            detections.append(Detection(label, confidence, bbox))
        return tuple(detections)

    def attributes(
        self,
        crops: Sequence[np.ndarray[Any, Any]],
    ) -> tuple[dict[str, Any], ...]:
        if not crops:
            return ()
        if self._attribute is None:
            raise RuntimeError("pedestrian attribute model is not ready")
        results = self._attribute.predict(list(crops), batch_size=len(crops))
        return tuple(_pedestrian_attributes(_model_result(item)) for item in results)


class RichLabelBusyError(RuntimeError):
    """A newer request must retry instead of queueing behind stale work."""


class RichLabelEngine:
    def __init__(self, models: PaddleRichModels | None = None) -> None:
        self.models = models or PaddleRichModels()
        self.tracker = IoUTracker(
            match_iou=_env_float("RICH_TRACK_IOU", 0.3, minimum=0.01, maximum=1.0),
            max_idle_frames=_env_int("RICH_TRACK_MAX_IDLE_FRAMES", 20),
        )
        self.attribute_interval_frames = _env_int(
            "RICH_ATTRIBUTE_INTERVAL_FRAMES", 10
        )
        self.max_people = _env_int("RICH_MAX_PEOPLE", 8)
        self._session_id: str | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.models.ready

    def infer(
        self,
        image: np.ndarray[Any, Any],
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
    ) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise RichLabelBusyError("rich-label inference is already running")
        try:
            if stream_session_id != self._session_id:
                self.tracker.reset()
                self._session_id = stream_session_id
            detections = self.models.detect(image)
            tracks = self.tracker.update(
                detections,
                frame_sequence=frame_sequence,
            )
            self._refresh_person_attributes(image, tracks, frame_sequence)
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
                        "attributes": dict(track.attributes),
                    }
                    for track in tracks
                ],
            }
        finally:
            self._lock.release()

    def _refresh_person_attributes(
        self,
        image: np.ndarray[Any, Any],
        tracks: Sequence[Track],
        frame_sequence: int,
    ) -> None:
        candidates = [
            track
            for track in tracks
            if track.label in PERSON_LABELS
            and (
                track.last_attribute_frame is None
                or frame_sequence - track.last_attribute_frame
                >= self.attribute_interval_frames
            )
        ]
        candidates.sort(
            key=lambda track: _bbox_area(track.bbox),
            reverse=True,
        )
        candidates = candidates[: self.max_people]
        crops: list[np.ndarray[Any, Any]] = []
        selected: list[Track] = []
        for track in candidates:
            crop = _crop(image, track.bbox)
            if crop is None:
                continue
            crops.append(crop)
            selected.append(track)
        if not crops:
            return
        prepare_attributes = getattr(self.models, "prepare_attributes", None)
        if callable(prepare_attributes) and not prepare_attributes():
            return
        try:
            attributes = self.models.attributes(crops)
        except Exception:
            LOGGER.warning("pedestrian attribute inference failed")
            return
        for track, values in zip(selected, attributes, strict=False):
            track.attributes.update(values)
            track.last_attribute_frame = frame_sequence


def create_app(engine: RichLabelEngine | None = None) -> FastAPI:
    application = FastAPI(title="Goodvision Paddle rich-label worker")
    selected_engine = engine or RichLabelEngine()
    application.state.engine = selected_engine

    @application.get("/health/live")
    async def live() -> dict[str, bool]:
        return {"live": True}

    @application.get("/health/ready")
    async def ready() -> Response:
        status = 200 if selected_engine.ready else 503
        return Response(
            content=(b'{"ready":true}' if status == 200 else b'{"ready":false}'),
            media_type="application/json",
            status_code=status,
        )

    @application.post("/v1/labels")
    async def labels(request: Request) -> dict[str, Any]:
        session_id = _required_header(
            request,
            "x-goodvision-stream-session-id",
        )
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
        try:
            return await asyncio.to_thread(
                selected_engine.infer,
                image,
                stream_session_id=session_id,
                frame_sequence=frame_sequence,
                timestamp_seconds=timestamp_seconds,
            )
        except RichLabelBusyError as exc:
            raise HTTPException(
                status_code=429,
                detail="inference busy",
                headers={"Retry-After": "1"},
            ) from exc
        except Exception as exc:
            LOGGER.exception("rich-label inference failed")
            raise HTTPException(
                status_code=503, detail="inference unavailable"
            ) from exc

    return application


def _required_header(request: Request, name: str) -> str:
    value = request.headers.get(name, "").strip()
    if not value or len(value) > 160:
        raise HTTPException(status_code=400, detail=f"missing {name}")
    return value


def _model_kwargs(model_name: str, model_dir: str, device: str) -> dict[str, Any]:
    values: dict[str, Any] = {"model_name": model_name, "device": device}
    if model_dir:
        values["model_dir"] = model_dir
    return values


def _model_result(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    payload = getattr(value, "json", value)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, Mapping):
        return {}
    nested = payload.get("res", payload)
    return nested if isinstance(nested, Mapping) else {}


def _pedestrian_attributes(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_labels = payload.get("label_names", [])
    raw_scores = payload.get("scores", [])
    labels = [str(value).split("(", 1)[0].strip() for value in raw_labels]
    scores: list[float | None] = []
    for value in raw_scores:
        try:
            rendered = float(value)
        except (TypeError, ValueError):
            rendered = math.nan
        scores.append(rendered if math.isfinite(rendered) else None)
    result: dict[str, Any] = {}
    age_map = {
        "AgeLess18": "<18",
        "Age18-60": "18-60",
        "AgeOver60": ">60",
    }
    gender_map = {
        "Female": "female-presenting",
        "Male": "male-presenting",
    }
    for label, score in zip(labels, scores, strict=False):
        if label in age_map:
            result["estimated_age_range"] = age_map[label]
            if score is not None:
                result["estimated_age_confidence"] = round(score, 6)
        elif label in gender_map:
            result["estimated_gender"] = gender_map[label]
            if score is not None:
                result["estimated_gender_confidence"] = round(score, 6)
    visible = [label for label in labels if label and label not in age_map | gender_map]
    if visible:
        result["appearance_attributes"] = visible[:12]
    return result


def _crop(
    image: np.ndarray[Any, Any],
    bbox: tuple[float, float, float, float],
) -> np.ndarray[Any, Any] | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, math.floor(x1)))
    top = max(0, min(height - 1, math.floor(y1)))
    right = max(left + 1, min(width, math.ceil(x2)))
    bottom = max(top + 1, min(height, math.ceil(y2)))
    crop = image[top:bottom, left:right]
    return crop if crop.size else None


def _valid_pixel_bbox(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> bool:
    x1, y1, x2, y2 = bbox
    return (
        all(math.isfinite(value) for value in bbox)
        and 0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    )


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise RuntimeError(f"{name} is out of range")
    return value


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be positive")
    return value


app = create_app()
