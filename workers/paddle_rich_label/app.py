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
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("goodvision.object_attributes")
MAX_JPEG_BYTES = 8 * 1024 * 1024
PERSON_LABELS = frozenset({"person", "pedestrian"})
_DETECTOR_PROFILES = {
    "balanced": "PicoDet-S",
    "accurate": "PP-YOLOE_plus-L",
}
_ATTRIBUTE_PROFILES = {
    "standard": "PP-LCNet_x1_0_pedestrian_attribute",
}


def _model_selection(
    *,
    profile_variable: str,
    legacy_variable: str,
    profiles: Mapping[str, str],
    default_profile: str,
    capability: str,
) -> tuple[str, str]:
    """Resolve a public profile while preserving existing private settings."""

    profile = os.environ.get(profile_variable, "").strip().lower()
    if profile:
        if profile not in profiles:
            # Never echo an unrecognized value: it may contain a private name.
            raise ValueError(f"unsupported {capability} profile")
        return profile, profiles[profile]
    model_name = os.environ.get(legacy_variable, "").strip()
    if not model_name:
        return default_profile, profiles[default_profile]
    selected_profile = next(
        (name for name, model in profiles.items() if model == model_name),
        "custom",
    )
    return selected_profile, model_name

# PP-Human's pedestrian attribute model is trained against this fixed PA100K
# layout. PaddleX normally filters the public classification result by a
# threshold, so callers must request all classes and reconstruct this vector by
# class id before applying PaddleDetection's grouped post-processing rules.
_PEDESTRIAN_ATTRIBUTE_LABELS = (
    "Hat",
    "Glasses",
    "ShortSleeve",
    "LongSleeve",
    "UpperStride",
    "UpperLogo",
    "UpperPlaid",
    "UpperSplice",
    "LowerStripe",
    "LowerPattern",
    "LongCoat",
    "Trousers",
    "Shorts",
    "Skirt&Dress",
    "Boots",
    "HandBag",
    "ShoulderBag",
    "Backpack",
    "HoldObjectsInFront",
    "AgeLess18",
    "Age18-60",
    "AgeOver60",
    "Female",
    "Front",
    "Side",
    "Back",
)
_ATTRIBUTE_THRESHOLD = 0.5
_GLASSES_THRESHOLD = 0.3
_HOLD_OBJECT_THRESHOLD = 0.6


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
    last_update_tick: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    last_attribute_frame: int | None = None
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    motion_updates: int = 0
    last_observed_timestamp_seconds: float | None = None


class IoUTracker:
    """Small deterministic single-camera tracker around Paddle detections."""

    def __init__(
        self,
        *,
        match_iou: float = 0.3,
        max_idle_frames: int = 20,
        visible_idle_frames: int = 1,
        max_idle_seconds: float = 3.0,
    ) -> None:
        if not 0.0 < match_iou <= 1.0:
            raise ValueError("match_iou must be between 0 and 1")
        if max_idle_frames < 1:
            raise ValueError("max_idle_frames must be positive")
        if visible_idle_frames < 0:
            raise ValueError("visible_idle_frames cannot be negative")
        if not math.isfinite(max_idle_seconds) or max_idle_seconds <= 0:
            raise ValueError("max_idle_seconds must be finite and positive")
        self.match_iou = float(match_iou)
        self.max_idle_frames = int(max_idle_frames)
        self.visible_idle_frames = int(visible_idle_frames)
        self.max_idle_seconds = float(max_idle_seconds)
        self._tracks: dict[str, Track] = {}
        self._next_id = 1
        self._update_tick = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._update_tick = 0

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_sequence: int,
        timestamp_seconds: float | None = None,
    ) -> tuple[Track, ...]:
        if timestamp_seconds is not None and (
            not math.isfinite(timestamp_seconds) or timestamp_seconds < 0
        ):
            raise ValueError("timestamp_seconds must be finite and non-negative")
        self._update_tick += 1
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if (
                0 <= timestamp_seconds - track.last_observed_timestamp_seconds
                <= self.max_idle_seconds
                if timestamp_seconds is not None
                and track.last_observed_timestamp_seconds is not None
                else self._update_tick - track.last_update_tick <= self.max_idle_frames
            )
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
                track.last_update_tick = self._update_tick
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
                    last_update_tick=self._update_tick,
                )
                self._tracks[track_id] = track
            track.last_observed_timestamp_seconds = timestamp_seconds
            current.append(track)
        visible = {track.track_id: track for track in current}
        for track_id, track in self._tracks.items():
            if (
                track_id not in visible
                and self._update_tick - track.last_update_tick
                <= self.visible_idle_frames
            ):
                visible[track_id] = track
        return tuple(sorted(visible.values(), key=lambda item: item.track_id))


class PaddleRichModels:
    """Lazy PaddleX model facade; models are downloaded only inside the sidecar."""

    def __init__(self) -> None:
        self.device = os.environ.get("RICH_DEVICE", "cpu").strip() or "cpu"
        self.detector_profile, self.detector_name = _model_selection(
            profile_variable="RICH_DETECTOR_PROFILE",
            legacy_variable="RICH_DETECTOR_MODEL",
            profiles=_DETECTOR_PROFILES,
            default_profile="balanced",
            capability="object detection",
        )
        self.attribute_profile, self.attribute_name = _model_selection(
            profile_variable="RICH_ATTRIBUTE_PROFILE",
            legacy_variable="RICH_ATTRIBUTE_MODEL",
            profiles=_ATTRIBUTE_PROFILES,
            default_profile="standard",
            capability="pedestrian attribute",
        )
        self.detector_dir = os.environ.get("RICH_DETECTOR_MODEL_DIR", "").strip()
        self.attribute_dir = os.environ.get("RICH_ATTRIBUTE_MODEL_DIR", "").strip()
        self.detection_threshold = _env_float(
            "RICH_DETECTION_THRESHOLD", 0.45, minimum=0.0, maximum=1.0
        )
        self._detector: Any | None = None
        self._attribute: Any | None = None
        self._lock = threading.Lock()
        self._gpu_ready = False
        self._tracking_gpu_ready = False

    @property
    def ready(self) -> bool:
        return self._detector is not None

    @property
    def attributes_ready(self) -> bool:
        return self._attribute is not None

    @property
    def gpu_ready(self) -> bool:
        """True only after the selected GPU and both model paths have executed."""

        return self._gpu_ready

    @property
    def tracking_gpu_ready(self) -> bool:
        """Detector-only GPU capability, independent of optional attributes."""

        return self._tracking_gpu_ready

    def verify_gpu_execution(self) -> None:
        """Probe in the worker process; a remote worker may use a different host."""

        self.verify_tracking_execution()
        self.verify_attribute_execution()

    def verify_tracking_execution(self) -> None:
        """Verify the kernel and detector before enabling the tracking path."""

        self._gpu_ready = False
        self._tracking_gpu_ready = False
        if self.device != "gpu" and not (
            self.device.startswith("gpu:")
            and self.device.removeprefix("gpu:").isdigit()
        ):
            return
        device_index = 0 if self.device == "gpu" else int(self.device.split(":")[1])
        try:
            import paddle

            if not paddle.device.is_compiled_with_cuda():
                return
            if paddle.device.cuda.device_count() <= device_index:
                return
            probe = paddle.to_tensor([1.0], place=paddle.CUDAPlace(device_index))
            result = probe + 1.0
            if not result.place.is_gpu_place() or float(result.numpy()[0]) != 2.0:
                return
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            self.detect(image)
        except Exception:
            return
        self._tracking_gpu_ready = True

    def verify_attribute_execution(self) -> None:
        self._gpu_ready = False
        if not self._tracking_gpu_ready:
            return
        try:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
            attributes = self.attributes((image,))
            if len(attributes) != 1 or not attributes[0]:
                return
        except Exception:
            # Neither a requested GPU nor installed drivers prove that the
            # runtime can execute. Keep CPU fallback out of the capability.
            return
        self._gpu_ready = True

    def ensure_detector_loaded(self) -> None:
        if self._detector is not None:
            return
        with self._lock:
            if self._detector is not None:
                return
            try:
                from paddlex import create_model

                detector_kwargs = _model_kwargs(
                    self.detector_name, self.detector_dir, self.device
                )
                self._detector = create_model(**detector_kwargs)
            except Exception:
                raise RuntimeError("object detection engine is unavailable") from None

    def ensure_attribute_loaded(self) -> None:
        if self._attribute is not None:
            return
        with self._lock:
            if self._attribute is not None:
                return
            try:
                from paddlex import create_model

                attribute_kwargs = _model_kwargs(
                    self.attribute_name, self.attribute_dir, self.device
                )
                self._attribute = create_model(**attribute_kwargs)
            except Exception:
                raise RuntimeError(
                    "pedestrian attribute engine is unavailable"
                ) from None

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
        results = self._attribute.predict(
            list(crops),
            batch_size=len(crops),
            # The default PaddleX classifier threshold removes the low-scoring
            # alternatives which PP-Human needs for argmax groups and for the
            # implicit Male branch of the single Female output.
            # PaddleX uses score > threshold, so zero would still discard an
            # exact-zero class. Its public API accepts -1 and then returns all
            # 26 PA100K outputs without changing the model's scores.
            threshold=-1.0,
        )
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
            visible_idle_frames=_env_int(
                "RICH_TRACK_VISIBLE_IDLE_FRAMES", 1, minimum=0
            ),
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
        self._last_source_key: tuple[float, int] | None = None

    @property
    def ready(self) -> bool:
        return self.models.ready

    @property
    def gpu_ready(self) -> bool:
        return self.ready and getattr(self.models, "gpu_ready", False) is True

    @property
    def tracking_gpu_ready(self) -> bool:
        return self.ready and getattr(self.models, "tracking_gpu_ready", False) is True

    def warmup_tracking(self) -> None:
        verify = getattr(self.models, "verify_tracking_execution", None)
        if callable(verify):
            verify()

    def warmup_attributes(self) -> None:
        verify = getattr(self.models, "verify_attribute_execution", None)
        if callable(verify):
            verify()

    def start_attribute_warmup(self) -> None:
        """Keep optional model startup off the detector readiness path."""

        def run() -> None:
            try:
                self.warmup_attributes()
            except Exception:
                LOGGER.warning("optional attribute capability probe failed")
            finally:
                with self._state_lock:
                    self._attribute_in_flight = False
                    self._attribute_thread = None
                    self._attribute_idle.set()

        with self._state_lock:
            if self._closed or self._attribute_in_flight:
                return
            self._attribute_in_flight = True
            self._attribute_idle.clear()
            self._attribute_thread = threading.Thread(
                target=run, name="attribute-capability-warmup", daemon=True,
            )
            self._attribute_thread.start()

    def warmup(self, *, require_gpu: bool = False) -> None:
        """Load the detector and verify any selected GPU before readiness."""

        if require_gpu:
            verify_gpu = getattr(self.models, "verify_gpu_execution", None)
            if callable(verify_gpu):
                verify_gpu()
            return
        ensure_loaded = getattr(self.models, "ensure_detector_loaded", None)
        if callable(ensure_loaded):
            ensure_loaded()

    def infer(
        self,
        image: np.ndarray[Any, Any],
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
        include_attributes: bool = True,
        include_tracking_metadata: bool = False,
    ) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise RichLabelBusyError("rich-label inference is already running")
        try:
            with self._state_lock:
                if stream_session_id != self._session_id:
                    self.tracker.reset()
                    self._session_id = stream_session_id
                    self._session_generation += 1
                    self._last_source_key = None
                if self._last_source_key is not None and (
                    timestamp_seconds < self._last_source_key[0]
                    or frame_sequence <= self._last_source_key[1]
                ):
                    raise ValueError("source frame metadata must increase")
            detections = self.models.detect(image)
            with self._state_lock:
                tracks = self.tracker.update(
                    detections,
                    frame_sequence=frame_sequence,
                    timestamp_seconds=timestamp_seconds,
                )
                self._last_source_key = (timestamp_seconds, frame_sequence)
                if include_attributes:
                    self._schedule_person_attributes_locked(
                        image, tracks, frame_sequence,
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
                            "attributes": (
                                dict(track.attributes) if include_attributes else {}
                            ),
                            **({
                                "observation_state": (
                                    "observed"
                                    if (
                                        track.last_update_tick
                                        == self.tracker._update_tick
                                    )
                                    else "predicted"
                                ),
                                "last_observed_timestamp_seconds": (
                                    track.last_observed_timestamp_seconds
                                ),
                                "last_observed_frame_sequence": (
                                    track.last_frame_sequence
                                ),
                            } if include_tracking_metadata else {}),
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
            and track.last_update_tick == self.tracker._update_tick
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
            name="object-attribute-inference",
            daemon=True,
        )
        self._attribute_thread = attribute_thread
        try:
            attribute_thread.start()
        except Exception:
            self._attribute_in_flight = False
            self._attribute_thread = None
            self._attribute_retry_at = (
                time.monotonic() + self._attribute_retry_seconds
            )
            self._attribute_idle.set()
            LOGGER.warning("could not start pedestrian attribute inference")

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
            LOGGER.warning("pedestrian attribute inference failed")


def create_app(engine: RichLabelEngine | None = None) -> FastAPI:
    require_gpu = engine is None
    selected_engine = engine or RichLabelEngine()

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        try:
            if require_gpu:
                await asyncio.to_thread(selected_engine.warmup_tracking)
                if selected_engine.tracking_gpu_ready:
                    selected_engine.start_attribute_warmup()
            else:
                await asyncio.to_thread(selected_engine.warmup)
        except Exception:
            LOGGER.error("object and attribute engine initialization failed")
            raise RuntimeError(
                "object and attribute engine initialization failed"
            ) from None
        try:
            yield
        finally:
            selected_engine.close()

    application = FastAPI(
        title="Goodvision Object and Attribute Engine",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    application.state.engine = selected_engine

    @application.get("/health/live")
    async def live() -> dict[str, bool]:
        return {"live": True}

    @application.get("/health/ready")
    async def ready() -> Response:
        is_ready = selected_engine.ready and (
            not require_gpu or selected_engine.gpu_ready
        )
        status = 200 if is_ready else 503
        return JSONResponse(
            content={"ready": status == 200, "gpu_ready": selected_engine.gpu_ready},
            status_code=status,
        )

    @application.get("/health/tracking")
    async def tracking_ready() -> Response:
        is_ready = selected_engine.ready and (
            not require_gpu or selected_engine.tracking_gpu_ready
        )
        return JSONResponse(
            content={
                "ready": is_ready, "gpu_ready": selected_engine.tracking_gpu_ready,
            },
            status_code=200 if is_ready else 503,
        )

    @application.post("/v1/labels")
    async def labels(request: Request) -> dict[str, Any]:
        return await infer_request(request, tracking=False, include_attributes=True)

    @application.post("/v1/tracks")
    async def tracks(
        request: Request, include_attributes: bool = False,
    ) -> dict[str, Any]:
        return await infer_request(
            request, tracking=True, include_attributes=include_attributes,
        )

    async def infer_request(
        request: Request, *, tracking: bool, include_attributes: bool,
    ) -> dict[str, Any]:
        gpu_available = (
            selected_engine.tracking_gpu_ready
            if tracking else selected_engine.gpu_ready
        )
        if require_gpu and not gpu_available:
            raise HTTPException(status_code=503, detail="inference unavailable")
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
                include_attributes=include_attributes,
                include_tracking_metadata=tracking,
            )
        except RichLabelBusyError as exc:
            raise HTTPException(
                status_code=429,
                detail="inference busy",
                headers={"Retry-After": "1"},
            ) from exc
        except Exception:
            LOGGER.error("object and attribute inference failed")
            raise HTTPException(
                status_code=503, detail="inference unavailable"
            ) from None

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
    scores = _pedestrian_attribute_scores(payload)
    result: dict[str, Any] = {}
    appearance: list[str] = []

    # PP-Human has only one gender output: Female. Male is the official
    # complementary branch, not a separate class returned by PaddleX.
    female_score = scores.get(22)
    if female_score is not None:
        is_female = female_score > _ATTRIBUTE_THRESHOLD
        result["estimated_gender"] = (
            "female-presenting" if is_female else "male-presenting"
        )
        result["estimated_gender_confidence"] = _binary_confidence(
            female_score, is_female
        )

    age = _argmax_complete_group(
        scores,
        ((19, "<18"), (20, "18-60"), (21, ">60")),
    )
    if age is not None:
        result["estimated_age_range"] = age[0]
        result["estimated_age_confidence"] = _round_confidence(age[1])

    direction = _argmax_complete_group(
        scores,
        ((23, "Front"), (24, "Side"), (25, "Back")),
    )
    if direction is not None:
        result["estimated_direction"] = direction[0]
        result["estimated_direction_confidence"] = _round_confidence(direction[1])
        appearance.append(direction[0])

    _decode_boolean_attribute(
        result,
        appearance,
        scores,
        class_id=1,
        value_key="wearing_glasses",
        confidence_key="wearing_glasses_confidence",
        positive_label="Glasses",
        threshold=_GLASSES_THRESHOLD,
    )
    _decode_boolean_attribute(
        result,
        appearance,
        scores,
        class_id=0,
        value_key="wearing_hat",
        confidence_key="wearing_hat_confidence",
        positive_label="Hat",
        threshold=_ATTRIBUTE_THRESHOLD,
    )
    _decode_boolean_attribute(
        result,
        appearance,
        scores,
        class_id=18,
        value_key="holding_object",
        confidence_key="holding_object_confidence",
        positive_label="HoldObjectsInFront",
        threshold=_HOLD_OBJECT_THRESHOLD,
    )

    bag = _argmax_complete_group(
        scores,
        ((15, "HandBag"), (16, "ShoulderBag"), (17, "Backpack")),
    )
    if bag is not None:
        bag_label, bag_score = bag
        has_bag = bag_score > _ATTRIBUTE_THRESHOLD
        result["bag_type"] = bag_label if has_bag else "No bag"
        result["bag_confidence"] = (
            _round_confidence(bag_score)
            if has_bag
            else _round_confidence(1.0 - bag_score)
        )
        if has_bag:
            appearance.append(bag_label)

    sleeves = _argmax_complete_group(
        scores,
        ((2, "ShortSleeve"), (3, "LongSleeve")),
    )
    if sleeves is not None:
        result["sleeve_type"] = sleeves[0]
        result["sleeve_confidence"] = _round_confidence(sleeves[1])
        appearance.append(sleeves[0])

    upper = _argmax_complete_group(
        scores,
        (
            (4, "UpperStride"),
            (5, "UpperLogo"),
            (6, "UpperPlaid"),
            (7, "UpperSplice"),
        ),
    )
    if upper is not None and upper[1] > _ATTRIBUTE_THRESHOLD:
        result["upper_style"] = upper[0]
        result["upper_style_confidence"] = _round_confidence(upper[1])
        appearance.append(upper[0])

    lower_group = (
        (8, "LowerStripe"),
        (9, "LowerPattern"),
        (10, "LongCoat"),
        (11, "Trousers"),
        (12, "Shorts"),
        (13, "Skirt&Dress"),
    )
    lower_scores = _complete_group(scores, lower_group)
    if lower_scores is not None:
        selected_lower = [
            (label, score)
            for (_class_id, label), score in zip(
                lower_group, lower_scores, strict=True
            )
            if score > _ATTRIBUTE_THRESHOLD
        ]
        # This fallback is intentional: PaddleDetection always shows the
        # highest-scoring lower attribute when none clear the threshold.
        if not selected_lower:
            winning_index = max(
                range(len(lower_scores)), key=lower_scores.__getitem__
            )
            selected_lower = [
                (lower_group[winning_index][1], lower_scores[winning_index])
            ]
        result["lower_styles"] = [label for label, _score in selected_lower]
        result["lower_style_confidences"] = {
            label: _round_confidence(score) for label, score in selected_lower
        }
        appearance.extend(label for label, _score in selected_lower)

    _decode_boolean_attribute(
        result,
        appearance,
        scores,
        class_id=14,
        value_key="wearing_boots",
        confidence_key="wearing_boots_confidence",
        positive_label="Boots",
        threshold=_ATTRIBUTE_THRESHOLD,
    )

    if appearance:
        result["appearance_attributes"] = appearance[:12]
    return result


def _pedestrian_attribute_scores(payload: Mapping[str, Any]) -> dict[int, float]:
    """Return a validated, complete PA100K vector keyed by fixed class id.

    Grouped post-processing is unsafe when PaddleX has filtered or reordered a
    partial result. Fail open with no attributes unless ids, names, and scores
    prove that all 26 classes are present exactly once.
    """

    raw_scores = _attribute_sequence(payload.get("scores"))
    raw_ids = _attribute_sequence(payload.get("class_ids"))
    raw_labels = _attribute_sequence(payload.get("label_names"))
    attribute_count = len(_PEDESTRIAN_ATTRIBUTE_LABELS)
    if not (
        len(raw_ids) == attribute_count
        and len(raw_scores) == attribute_count
        and len(raw_labels) == attribute_count
    ):
        return {}

    decoded: dict[int, float] = {}
    for raw_id, raw_label, raw_score in zip(
        raw_ids, raw_labels, raw_scores, strict=True
    ):
        class_id = _attribute_class_id(raw_id)
        score = _attribute_score(raw_score)
        if class_id is None or score is None or class_id in decoded:
            return {}
        label = str(raw_label).split("(", 1)[0].strip().casefold()
        if label != _PEDESTRIAN_ATTRIBUTE_LABELS[class_id].casefold():
            return {}
        decoded[class_id] = score
    return decoded if len(decoded) == attribute_count else {}


def _attribute_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, np.ndarray):
        return tuple(value.reshape(-1).tolist())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return tuple(value)
    return ()


def _attribute_class_id(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        rendered = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rendered) or not rendered.is_integer():
        return None
    class_id = int(rendered)
    return class_id if 0 <= class_id < len(_PEDESTRIAN_ATTRIBUTE_LABELS) else None


def _attribute_score(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def _complete_group(
    scores: Mapping[int, float],
    group: Sequence[tuple[int, str]],
) -> tuple[float, ...] | None:
    if any(class_id not in scores for class_id, _label in group):
        return None
    return tuple(scores[class_id] for class_id, _label in group)


def _argmax_complete_group(
    scores: Mapping[int, float],
    group: Sequence[tuple[int, str]],
) -> tuple[str, float] | None:
    group_scores = _complete_group(scores, group)
    if group_scores is None:
        return None
    winning_index = max(range(len(group_scores)), key=group_scores.__getitem__)
    return group[winning_index][1], group_scores[winning_index]


def _decode_boolean_attribute(
    result: dict[str, Any],
    appearance: list[str],
    scores: Mapping[int, float],
    *,
    class_id: int,
    value_key: str,
    confidence_key: str,
    positive_label: str,
    threshold: float,
) -> None:
    score = scores.get(class_id)
    if score is None:
        return
    selected = score > threshold
    result[value_key] = selected
    result[confidence_key] = _binary_confidence(score, selected)
    if selected:
        appearance.append(positive_label)


def _binary_confidence(score: float, selected: bool) -> float:
    return _round_confidence(score if selected else 1.0 - score)


def _round_confidence(score: float) -> float:
    return round(score, 6)


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


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


app = create_app()
