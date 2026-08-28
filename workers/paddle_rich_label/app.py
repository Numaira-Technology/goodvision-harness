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
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
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
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    motion_updates: int = 0


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
        candidates: list[tuple[float, float, float, str, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                if track.label != detection.label:
                    continue
                association = _association_score(
                    track,
                    detection,
                    frame_sequence=frame_sequence,
                    match_iou=self.match_iou,
                )
                if association is None:
                    continue
                score, predicted_overlap, normalized_distance = association
                candidates.append(
                    (
                        score,
                        predicted_overlap,
                        normalized_distance,
                        track_id,
                        detection_index,
                    )
                )
        candidates.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2],
                item[3],
                item[4],
            )
        )
        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, Track] = {}
        for _score, _overlap, _distance, track_id, detection_index in candidates:
            if (
                track_id in matched_tracks
                or detection_index in matched_detections
            ):
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = self._tracks[track_id]

        current: list[Track] = []
        for detection_index, detection in enumerate(detections):
            track = assignments.get(detection_index)
            if track is not None:
                _update_track_motion(
                    track,
                    detection,
                    frame_sequence=frame_sequence,
                )
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
            os.environ.get("RICH_DETECTOR_MODEL", "PicoDet-S").strip()
            or "PicoDet-S"
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
        self.ensure_attribute_loaded()
        assert self._attribute is not None
        results = self._attribute.predict(list(crops), batch_size=len(crops))
        return tuple(_pedestrian_attributes(_model_result(item)) for item in results)


class RichLabelBusyError(RuntimeError):
    """A newer request must retry instead of queueing behind stale work."""


@dataclass(frozen=True, slots=True)
class _AttributeTarget:
    track_id: str
    track: Track


@dataclass(frozen=True, slots=True)
class _AttributeJob:
    stream_session_id: str
    session_generation: int
    frame_sequence: int
    targets: tuple[_AttributeTarget, ...]
    crops: tuple[np.ndarray[Any, Any], ...]


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
        self._session_generation = 0
        self._state_lock = threading.Lock()
        self._lock = threading.Lock()
        self._attribute_in_flight = False
        self._attribute_retry_at = 0.0
        self._attribute_retry_seconds = _env_float(
            "RICH_ATTRIBUTE_RETRY_SECONDS",
            60.0,
            minimum=1.0,
            maximum=3600.0,
        )
        self._attribute_idle = threading.Event()
        self._attribute_idle.set()
        self._attribute_thread: threading.Thread | None = None
        self.attribute_shutdown_timeout_seconds = _env_float(
            "RICH_ATTRIBUTE_SHUTDOWN_TIMEOUT_SECONDS",
            5.0,
            minimum=0.0,
            maximum=60.0,
        )
        self._closed = False

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
            detections = self.models.detect(image)
            with self._state_lock:
                if stream_session_id != self._session_id:
                    self.tracker.reset()
                    self._session_id = stream_session_id
                    self._session_generation += 1
                tracks = self.tracker.update(
                    detections,
                    frame_sequence=frame_sequence,
                )
                self._schedule_person_attributes_locked(
                    image,
                    tracks,
                    frame_sequence,
                )
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

    def close(self) -> None:
        """Invalidate optional background work and drain it for a bounded time."""

        with self._state_lock:
            self._closed = True
            self._session_generation += 1
            attribute_thread = self._attribute_thread
        if attribute_thread is None or attribute_thread is threading.current_thread():
            return
        attribute_thread.join(timeout=self.attribute_shutdown_timeout_seconds)
        if attribute_thread.is_alive():
            LOGGER.warning("pedestrian attribute task exceeded shutdown timeout")

    def _schedule_person_attributes_locked(
        self,
        image: np.ndarray[Any, Any],
        tracks: Sequence[Track],
        frame_sequence: int,
    ) -> None:
        if (
            self._closed
            or self._attribute_in_flight
            or time.monotonic() < self._attribute_retry_at
        ):
            return
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
        targets: list[_AttributeTarget] = []
        for track in candidates:
            crop = _crop(image, track.bbox)
            if crop is None:
                continue
            crops.append(crop)
            targets.append(_AttributeTarget(track.track_id, track))
        if not crops:
            return
        assert self._session_id is not None
        job = _AttributeJob(
            stream_session_id=self._session_id,
            session_generation=self._session_generation,
            frame_sequence=frame_sequence,
            targets=tuple(targets),
            crops=tuple(crops),
        )
        self._attribute_in_flight = True
        self._attribute_idle.clear()
        attribute_thread = threading.Thread(
            target=self._run_attribute_job,
            args=(job,),
            name="paddle-attribute-inference",
            daemon=True,
        )
        self._attribute_thread = attribute_thread
        try:
            attribute_thread.start()
        except Exception as exc:
            self._attribute_in_flight = False
            self._attribute_thread = None
            self._attribute_retry_at = (
                time.monotonic() + self._attribute_retry_seconds
            )
            self._attribute_idle.set()
            LOGGER.warning(
                "could not start pedestrian attribute inference: %s",
                type(exc).__name__,
            )

    def _run_attribute_job(self, job: _AttributeJob) -> None:
        attributes: tuple[dict[str, Any], ...] = ()
        error: Exception | None = None
        try:
            attributes = tuple(self.models.attributes(job.crops))
            if len(attributes) != len(job.targets) or any(
                not isinstance(values, Mapping) for values in attributes
            ):
                raise RuntimeError("pedestrian attribute result count/type mismatch")
        except Exception as exc:
            error = exc

        with self._state_lock:
            try:
                if error is not None:
                    self._attribute_retry_at = (
                        time.monotonic() + self._attribute_retry_seconds
                    )
                elif (
                    not self._closed
                    and job.stream_session_id == self._session_id
                    and job.session_generation == self._session_generation
                ):
                    for target, values in zip(
                        job.targets,
                        attributes,
                        strict=False,
                    ):
                        current = self.tracker._tracks.get(target.track_id)
                        if current is not target.track:
                            continue
                        if not isinstance(values, Mapping):
                            continue
                        current.attributes.update(values)
                        current.last_attribute_frame = job.frame_sequence
            finally:
                self._attribute_in_flight = False
                if self._attribute_thread is threading.current_thread():
                    self._attribute_thread = None
                self._attribute_idle.set()

        if error is not None:
            LOGGER.warning(
                "pedestrian attribute inference failed: %s",
                type(error).__name__,
            )


def create_app(engine: RichLabelEngine | None = None) -> FastAPI:
    selected_engine = engine or RichLabelEngine()

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        yield
        selected_engine.close()

    application = FastAPI(
        title="Goodvision Paddle rich-label worker",
        lifespan=lifespan,
    )
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


def _association_score(
    track: Track,
    detection: Detection,
    *,
    frame_sequence: int,
    match_iou: float,
) -> tuple[float, float, float] | None:
    gap = max(1, frame_sequence - track.last_frame_sequence)
    predicted_bbox = _predicted_bbox(track, frame_sequence)
    predicted_overlap = _iou(predicted_bbox, detection.bbox)
    previous_overlap = _iou(track.bbox, detection.bbox)
    predicted_center = _bbox_center(predicted_bbox)
    detection_center = _bbox_center(detection.bbox)
    center_distance = math.hypot(
        predicted_center[0] - detection_center[0],
        predicted_center[1] - detection_center[1],
    )
    distance_scale = max(
        _bbox_diagonal(predicted_bbox),
        _bbox_diagonal(detection.bbox),
        1.0,
    )
    normalized_distance = center_distance / distance_scale
    predicted_area = _bbox_area(predicted_bbox)
    detection_area = _bbox_area(detection.bbox)
    size_similarity = (
        min(predicted_area, detection_area) / max(predicted_area, detection_area)
        if predicted_area > 0 and detection_area > 0
        else 0.0
    )
    center_limit = min(2.5, 1.25 + (0.12 * max(0, gap - 1)))
    if not (
        predicted_overlap >= max(0.05, match_iou * 0.5)
        or previous_overlap >= match_iou
        or (normalized_distance <= center_limit and size_similarity >= 0.25)
    ):
        return None

    proximity = max(0.0, 1.0 - (normalized_distance / center_limit))
    direction = _direction_alignment(track, detection.bbox)
    score = (
        (3.5 * predicted_overlap)
        + (0.5 * previous_overlap)
        + (1.5 * proximity)
        + (0.35 * size_similarity)
        + ((0.8 * direction) if track.motion_updates else 0.0)
    )
    return score, predicted_overlap, normalized_distance


def _predicted_bbox(
    track: Track,
    frame_sequence: int,
) -> tuple[float, float, float, float]:
    gap = max(0, frame_sequence - track.last_frame_sequence)
    offset_x = track.velocity_x * gap
    offset_y = track.velocity_y * gap
    return (
        track.bbox[0] + offset_x,
        track.bbox[1] + offset_y,
        track.bbox[2] + offset_x,
        track.bbox[3] + offset_y,
    )


def _update_track_motion(
    track: Track,
    detection: Detection,
    *,
    frame_sequence: int,
) -> None:
    gap = frame_sequence - track.last_frame_sequence
    if gap > 0:
        old_center = _bbox_center(track.bbox)
        new_center = _bbox_center(detection.bbox)
        observed_x = (new_center[0] - old_center[0]) / gap
        observed_y = (new_center[1] - old_center[1]) / gap
        if track.motion_updates:
            track.velocity_x = (0.35 * track.velocity_x) + (0.65 * observed_x)
            track.velocity_y = (0.35 * track.velocity_y) + (0.65 * observed_y)
        else:
            track.velocity_x = observed_x
            track.velocity_y = observed_y
        track.motion_updates += 1
    track.confidence = detection.confidence
    track.bbox = detection.bbox
    track.last_frame_sequence = frame_sequence


def _direction_alignment(
    track: Track,
    detection_bbox: tuple[float, float, float, float],
) -> float:
    if not track.motion_updates:
        return 0.0
    current_center = _bbox_center(track.bbox)
    detection_center = _bbox_center(detection_bbox)
    observed_x = detection_center[0] - current_center[0]
    observed_y = detection_center[1] - current_center[1]
    expected_norm = math.hypot(track.velocity_x, track.velocity_y)
    observed_norm = math.hypot(observed_x, observed_y)
    if expected_norm <= 1e-9 or observed_norm <= 1e-9:
        return 0.5
    cosine = (
        (track.velocity_x * observed_x) + (track.velocity_y * observed_y)
    ) / (expected_norm * observed_norm)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _bbox_center(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_diagonal(bbox: tuple[float, float, float, float]) -> float:
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


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
