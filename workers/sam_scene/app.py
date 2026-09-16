"""SAM image keyframes and optional native SAM3.1 finite video batches."""

from __future__ import annotations

import asyncio
import copy
import io
import logging
import math
import os
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("goodvision.sam_scene")
SAM3_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
SESSION_HEADER = "X-GoodVision-Stream-Session-Id"
SEQUENCE_HEADER = "X-GoodVision-Frame-Sequence"
TIMESTAMP_HEADER = "X-GoodVision-Timestamp-Seconds"
BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Settings:
    checkpoint_path: str = "/models/sam3.pt"
    labels: tuple[str, ...] = ("person",)
    confidence: float = 0.5
    keyframe_seconds: float = 0.4
    keyframe_requests: int = 5
    track_ttl_seconds: float = 1.2
    source_gap_seconds: float = 2.0
    match_iou: float = 0.25
    ambiguity_margin: float = 0.15
    crossing_iou: float = 0.5
    max_tracks: int = 64
    max_sessions: int = 8
    session_idle_seconds: float = 60.0
    max_jpeg_bytes: int = 8 * 1024 * 1024
    max_image_pixels: int = 3840 * 2160
    inference_max_side: int = 1280
    request_timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        if not self.checkpoint_path.strip():
            raise ValueError("checkpoint_path is required")
        if (
            not self.labels
            or len(self.labels) > 16
            or len(set(self.labels)) != len(self.labels)
            or any(not s.strip() or len(s) > 80 for s in self.labels)
        ):
            raise ValueError("labels must contain 1 to 16 unique short text prompts")
        for name in ("confidence", "match_iou", "ambiguity_margin", "crossing_iou"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in (
            "keyframe_seconds",
            "track_ttl_seconds",
            "source_gap_seconds",
            "session_idle_seconds",
            "request_timeout_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "keyframe_requests",
            "max_tracks",
            "max_sessions",
            "max_jpeg_bytes",
            "max_image_pixels",
            "inference_max_side",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_env(cls) -> Settings:
        values: dict[str, Any] = {}
        defaults = cls()
        for name in cls.__dataclass_fields__:
            value = os.environ.get(f"SAM_SCENE_{name.upper()}")
            if value is None:
                continue
            default = getattr(defaults, name)
            if name == "labels":
                values[name] = tuple(s.strip() for s in value.split(","))
            elif isinstance(default, int):
                values[name] = int(value)
            elif isinstance(default, float):
                values[name] = float(value)
            else:
                values[name] = value
        return cls(**values)


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    bbox: BBox
    confidence: float
    contours: tuple[tuple[tuple[float, float], ...], ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.label
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("invalid detection")
        if len(self.bbox) != 4 or any(
            not math.isfinite(x) or not 0 <= x <= 1 for x in self.bbox
        ):
            raise ValueError("bbox must be normalized xyxy")
        if self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]:
            raise ValueError("bbox must have positive area")


class Detector(Protocol):
    def warmup(self) -> None: ...
    def detect(self, image: Any) -> Sequence[Detection]: ...
    def close(self) -> None: ...


def _validated_image_detector_state(
    checkpoint_state: Mapping[str, Any], expected_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep image weights, permitting only the pinned model's optional SAM2 neck.

    Sam3DualViTDetNeck constructs sam2_convs as deepcopy(convs). The official
    video checkpoint may contain this unused branch, while our image builder
    disables it. Its exact names and shapes must mirror the image neck; an
    arbitrary extra detector key must never silently bypass strict loading.
    """
    detector_state = {
        key.removeprefix("detector."): value
        for key, value in checkpoint_state.items()
        if key.startswith("detector.")
    }
    if not detector_state:
        raise RuntimeError("Expected official SAM 3 checkpoint with detector weights")
    missing = expected_state.keys() - detector_state.keys()
    if missing:
        raise RuntimeError("SAM 3 checkpoint is missing required image detector keys")
    image_prefix = "backbone.visual.convs."
    optional_prefix = "backbone.visual.sam2_convs."
    optional_shapes = {
        optional_prefix + key.removeprefix(image_prefix): value.shape
        for key, value in expected_state.items()
        if key.startswith(image_prefix)
    }
    validated = {}
    for key, value in detector_state.items():
        if key in expected_state:
            expected_shape = expected_state[key].shape
        elif key in optional_shapes:
            expected_shape = optional_shapes[key]
        else:
            raise RuntimeError("SAM 3 checkpoint has unexpected image detector keys")
        if getattr(value, "shape", None) != expected_shape:
            raise RuntimeError(
                "SAM 3 checkpoint has incompatible detector tensor shapes"
            )
        if key in expected_state:
            validated[key] = value
    return validated


class SamImageDetector:
    """Real CUDA inference; a local SAM 3 checkpoint is mandatory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.gpu_ready = False
        self.processor: Any = None
        self._torch: Any = None

    def warmup(self) -> None:
        checkpoint = Path(self.settings.checkpoint_path)
        if not checkpoint.is_file():
            raise RuntimeError(
                "SAM 3 checkpoint missing; mount approved sam3.pt at configured path"
            )
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable")
        self._torch = torch
        # Official builder defaults to non-strict checkpoint loading. Load the
        # detector ourselves so wrong/random/incompatible weights fail readiness.
        model = build_sam3_image_model(
            device="cpu",
            eval_mode=True,
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        checkpoint_data = torch.load(
            str(checkpoint), map_location="cpu", weights_only=True
        )
        state = checkpoint_data.get("model", checkpoint_data)
        detector_state = _validated_image_detector_state(state, model.state_dict())
        model.load_state_dict(detector_state, strict=True)
        del detector_state, state, checkpoint_data
        model = model.to(device="cuda").eval()
        self.processor = Sam3Processor(
            model,
            device="cuda",
            confidence_threshold=self.settings.confidence,
        )
        self.detect(Image.new("RGB", (640, 360)))
        torch.cuda.synchronize()
        self.gpu_ready = True

    def detect(self, image: Any) -> tuple[Detection, ...]:
        if self.processor is None:
            raise RuntimeError("SAM model not initialized")
        import numpy as np

        if __package__:
            from .sam31 import native_mask_geometry
        else:
            from sam31 import native_mask_geometry

        torch = self._torch
        # SAM's processor upsamples masks to input dimensions. Bounding the
        # supplied image bounds mask memory; normalized boxes still map to source.
        image = image.copy()
        image.thumbnail(
            (self.settings.inference_max_side, self.settings.inference_max_side)
        )
        width, height = image.size
        found: list[Detection] = []
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            state = self.processor.set_image(image)
            for label in self.settings.labels:
                self.processor.reset_all_prompts(state)
                result = self.processor.set_text_prompt(state=state, prompt=label)
                boxes = result["boxes"].detach().float().cpu().tolist()
                scores = result["scores"].detach().float().cpu().tolist()
                masks = result["masks"].detach().cpu().numpy()
                if masks.shape == (len(boxes), 1, height, width):
                    masks = masks[:, 0]
                if masks.shape != (len(boxes), height, width):
                    raise RuntimeError("SAM image masks do not match source dimensions")
                if masks.dtype != np.bool_:
                    raise RuntimeError("SAM image masks must be binary")
                for box, score, mask in zip(boxes, scores, masks, strict=True):
                    normalized = tuple(
                        max(0.0, min(1.0, value / scale))
                        for value, scale in zip(
                            box, (width, height, width, height), strict=True
                        )
                    )
                    try:
                        _, polygons, _, _ = native_mask_geometry(mask)
                        found.append(Detection(
                            label, normalized, float(score),
                            tuple(tuple(tuple(point) for point in polygon)
                                  for polygon in polygons),
                        ))
                    except ValueError:
                        continue
            del state
        return tuple(found)

    def close(self) -> None:
        self.gpu_ready = False
        self.processor = None
        if self._torch is not None:
            self._torch.cuda.empty_cache()


@dataclass(slots=True)
class Track:
    track_id: str
    label: str
    bbox: BBox
    confidence: float
    observed_sequence: int
    observed_timestamp: float
    velocity: tuple[float, float] = (0.0, 0.0)
    contours: tuple[tuple[tuple[float, float], ...], ...] = ()

    def predicted_bbox(self, timestamp: float) -> BBox:
        dt = max(0.0, timestamp - self.observed_timestamp)
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        # Clamp translation rather than independent corners to retain shape.
        dx = max(-self.bbox[0], min(1 - self.bbox[2], dx))
        dy = max(-self.bbox[1], min(1 - self.bbox[3], dy))
        return (
            self.bbox[0] + dx,
            self.bbox[1] + dy,
            self.bbox[2] + dx,
            self.bbox[3] + dy,
        )


def _iou(a: BBox, b: BBox) -> float:
    intersect = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return intersect / (area - intersect) if area > intersect else 0.0


@dataclass(slots=True)
class Session:
    epoch: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    tracks: dict[str, Track] = field(default_factory=dict)
    next_id: int = 1
    sequence: int = -1
    timestamp: float = -1.0
    last_keyframe_timestamp: float = -math.inf
    requests_since_keyframe: int = 0
    image_size: tuple[int, int] | None = None
    touched: float = 0.0


class BusyError(RuntimeError):
    pass


class FrameOrderError(RuntimeError):
    pass


class SceneEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        detector: Detector | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.detector = detector or SamImageDetector(self.settings)
        self._production = detector is None
        self._clock = clock
        self._lock = threading.Lock()
        self.sessions: dict[str, Session] = {}
        self.ready = False
        self.error_code: str | None = None

    @property
    def gpu_ready(self) -> bool:
        # Injection never constitutes evidence of real GPU execution.
        return self._production and self.ready and self.detector.gpu_ready is True

    def warmup(self) -> None:
        self.detector.warmup()
        self.ready = True

    def close(self) -> None:
        self.ready = False
        self.sessions.clear()
        self.detector.close()

    def infer(
        self,
        image: Any,
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise BusyError("inference busy")
        try:
            return self._infer(
                image, stream_session_id, frame_sequence, timestamp_seconds, deadline
            )
        finally:
            self._lock.release()

    def _infer(
        self,
        image: Any,
        session_id: str,
        sequence: int,
        timestamp: float,
        deadline: float | None,
    ) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("model not ready")
        now = self._clock()
        self.sessions = {
            sid: s
            for sid, s in self.sessions.items()
            if now - s.touched < self.settings.session_idle_seconds
        }
        existing = self.sessions.get(session_id)
        if existing and (
            sequence <= existing.sequence or timestamp < existing.timestamp
        ):
            raise FrameOrderError(
                "source frame moved backwards; use a new stream session after restart"
            )
        if existing is None and len(self.sessions) >= self.settings.max_sessions:
            raise BusyError("active session limit reached")
        reset = (
            existing is None
            or image.size != existing.image_size
            or timestamp - existing.timestamp > self.settings.source_gap_seconds
        )
        # Results which time out must never mutate a stream's accepted state.
        session = Session() if reset else copy.deepcopy(existing)
        session.image_size = image.size
        session.tracks = {
            tid: t
            for tid, t in session.tracks.items()
            if timestamp - t.observed_timestamp <= self.settings.track_ttl_seconds
        }
        session.requests_since_keyframe += 1
        keyframe = (
            reset
            or not session.tracks
            or timestamp - session.last_keyframe_timestamp
            >= self.settings.keyframe_seconds
            or session.requests_since_keyframe >= self.settings.keyframe_requests
        )
        if keyframe:
            detections = self._filter_detections(self.detector.detect(image))
            self._associate(session, detections, sequence, timestamp)
            session.last_keyframe_timestamp = timestamp
            session.requests_since_keyframe = 0
        else:
            # When predictions collide, stop drawing both and force re-detection.
            colliding = self._colliding_tracks(session, timestamp)
            for track_id in colliding:
                del session.tracks[track_id]
            if colliding:
                session.last_keyframe_timestamp = -math.inf
        session.sequence, session.timestamp, session.touched = sequence, timestamp, now
        objects = []
        for track in session.tracks.values():
            observed = track.observed_sequence == sequence
            bbox = track.bbox if observed else track.predicted_bbox(timestamp)
            objects.append(
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "bbox": dict(zip(("x1", "y1", "x2", "y2"), bbox, strict=True)),
                    "confidence": track.confidence,
                    "attributes": ({"mask_contours": {
                        "coordinate_system": "normalized",
                        "source_frame_sequence": sequence,
                        "source_timestamp_seconds": timestamp,
                        "polygons": track.contours,
                    }} if observed and track.contours else {}),
                    "observation_state": "observed" if observed else "predicted",
                    "last_observed_timestamp_seconds": track.observed_timestamp,
                    "last_observed_frame_sequence": track.observed_sequence,
                }
            )
        if deadline is not None and self._clock() >= deadline:
            raise TimeoutError("inference deadline exceeded")
        self.sessions[session_id] = session
        return {
            "stream_session_id": session_id,
            "frame_sequence": sequence,
            "timestamp_seconds": timestamp,
            "objects": objects,
        }

    def _filter_detections(self, detections: Sequence[Detection]) -> list[Detection]:
        selected: list[Detection] = []
        for detection in sorted(detections, key=lambda d: -d.confidence):
            if detection.confidence < self.settings.confidence:
                continue
            if any(
                d.label == detection.label and _iou(d.bbox, detection.bbox) > 0.85
                for d in selected
            ):
                continue
            selected.append(detection)
            if len(selected) >= self.settings.max_tracks:
                break
        return selected

    def _colliding_tracks(self, session: Session, timestamp: float) -> set[str]:
        tracks = list(session.tracks.values())
        collided: set[str] = set()
        for index, left in enumerate(tracks):
            for right in tracks[index + 1 :]:
                if (
                    left.label == right.label
                    and _iou(
                        left.predicted_bbox(timestamp), right.predicted_bbox(timestamp)
                    )
                    >= self.settings.crossing_iou
                ):
                    collided.update((left.track_id, right.track_id))
        return collided

    def _associate(
        self,
        session: Session,
        detections: Sequence[Detection],
        sequence: int,
        timestamp: float,
    ) -> None:
        retired = self._colliding_tracks(session, timestamp)
        candidates: dict[str, list[tuple[float, int]]] = {}
        for tid, track in session.tracks.items():
            if tid in retired:
                continue
            matches = [
                (_iou(track.predicted_bbox(timestamp), d.bbox), i)
                for i, d in enumerate(detections)
                if track.label == d.label
            ]
            candidates[tid] = sorted(
                (m for m in matches if m[0] >= self.settings.match_iou), reverse=True
            )
        # Both sides must have a clear winner. A crowded ambiguous component
        # loses its old IDs, including unmatched candidates, rather than waiting
        # and accidentally transferring old identity/event evidence afterwards.
        claims: dict[int, list[tuple[float, str]]] = {}
        for tid, matches in candidates.items():
            if (
                len(matches) > 1
                and matches[0][0] - matches[1][0] < self.settings.ambiguity_margin
            ):
                retired.add(tid)
                continue
            if matches:
                score, index = matches[0]
                claims.setdefault(index, []).append((score, tid))
        for claim in claims.values():
            if len(claim) > 1:
                retired.update(tid for _, tid in claim)
        # Expand retirement through every possible association. Otherwise an
        # ambiguous track could vanish while its candidate is assigned to a
        # different old identity with an apparently unique best match.
        while True:
            ambiguous_indices = {
                index for tid in retired for _, index in candidates.get(tid, [])
            }
            expanded = {
                tid
                for tid, matches in candidates.items()
                if any(index in ambiguous_indices for _, index in matches)
            }
            if expanded <= retired:
                break
            retired.update(expanded)
        assignments: dict[int, str] = {}
        for index, claim in claims.items():
            claim.sort(reverse=True)
            if len(claim) == 1 and claim[0][1] not in retired:
                assignments[index] = claim[0][1]
        for tid in retired:
            session.tracks.pop(tid, None)
        for index, detection in enumerate(detections):
            tid = assignments.get(index)
            old = session.tracks.get(tid) if tid else None
            velocity = (0.0, 0.0)
            if old:
                dt = timestamp - old.observed_timestamp
                if dt > 0:
                    velocity = (
                        (
                            detection.bbox[0]
                            + detection.bbox[2]
                            - old.bbox[0]
                            - old.bbox[2]
                        )
                        / (2 * dt),
                        (
                            detection.bbox[1]
                            + detection.bbox[3]
                            - old.bbox[1]
                            - old.bbox[3]
                        )
                        / (2 * dt),
                    )
            else:
                tid = f"sam-{session.epoch}-{session.next_id:05d}"
                session.next_id += 1
            session.tracks[tid] = Track(
                tid,
                detection.label,
                detection.bbox,
                detection.confidence,
                sequence,
                timestamp,
                velocity,
                detection.contours,
            )
        # Keep recent unmatched predictions only when capacity permits.
        if len(session.tracks) > self.settings.max_tracks:
            session.tracks = dict(
                sorted(
                    session.tracks.items(), key=lambda pair: -pair[1].observed_timestamp
                )[: self.settings.max_tracks]
            )


def decode_jpeg(payload: bytes, settings: Settings) -> Any:
    from PIL import Image, UnidentifiedImageError

    if not payload or len(payload) > settings.max_jpeg_bytes:
        raise HTTPException(413, "JPEG size outside allowed bounds")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "JPEG":
                    raise HTTPException(400, "payload is not JPEG")
                if image.width * image.height > settings.max_image_pixels:
                    raise HTTPException(413, "JPEG pixel count exceeds limit")
                # Match source pixel coordinates; do not apply EXIF rotation.
                return image.convert("RGB")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise HTTPException(413, "JPEG pixel count exceeds limit") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "invalid JPEG") from None


def create_app(
    engine: SceneEngine | None = None,
    video_engine=None,
    session_engine=None,
    perception_engines=None,
) -> FastAPI:
    # Both package tests and the standalone Docker /worker/app.py entry point.
    if __package__:
        from .continuous import SessionEngine, SessionRegistry, register_session_routes
        from .perception_routes import (
            build_perception_engines,
            register_perception_routes,
        )
        from .sam31 import VideoEngine, register_video_routes
    else:
        from continuous import SessionEngine, SessionRegistry, register_session_routes
        from perception_routes import (
            build_perception_engines,
            register_perception_routes,
        )
        from sam31 import VideoEngine, register_video_routes

    selected = engine or SceneEngine()
    selected_video = video_engine or VideoEngine()
    selected_session = session_engine or SessionEngine(selected_video)
    sessions = SessionRegistry()
    sessions.add(selected_session)
    selected_routes = (
        build_perception_engines(selected_video, sessions)
        if perception_engines is None
        else perception_engines
    )
    for route_engine in selected_routes.values():
        sessions.add(route_engine)
    production = engine is None
    image_enabled = os.environ.get("SAM_SCENE_IMAGE_ENABLED", "true").lower() == "true"
    admission = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if image_enabled:
            try:
                await asyncio.to_thread(selected.warmup)
            except Exception:
                selected.error_code = "model_initialization_failed"
                LOGGER.exception("SAM scene initialization failed")
        try:
            await asyncio.to_thread(selected_video.warmup)
        except Exception:
            LOGGER.exception("SAM3.1 video initialization failed")
        try:
            await asyncio.to_thread(selected_session.warmup)
        except Exception:
            LOGGER.exception("SAM3.1 session initialization failed")
        for route, route_engine in selected_routes.items():
            try:
                await asyncio.to_thread(route_engine.warmup)
            except Exception:
                LOGGER.exception(
                    "Perception route initialization failed route=%s", route
                )

        async def reap_sessions():
            while True:
                await asyncio.sleep(5)
                if admission.acquire(blocking=False):
                    try:
                        sessions.expire()
                    finally:
                        admission.release()

        reaper = asyncio.create_task(reap_sessions())
        try:
            yield
        finally:
            reaper.cancel()
            try:
                await reaper
            except asyncio.CancelledError:
                pass
        # Avoid model teardown while an abandoned CUDA request still uses it.
        if admission.acquire(blocking=False):
            try:
                selected.close()
                sessions.close()
                selected_video.close()
            finally:
                admission.release()

    app = FastAPI(
        title="Goodvision SAM Scene Worker",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.engine = selected
    app.state.perception_sessions = sessions
    register_video_routes(app, selected_video, admission, decode_jpeg)
    register_session_routes(app, selected_session, admission, decode_jpeg)
    register_perception_routes(app, selected_routes, admission, decode_jpeg)

    @app.get("/health/live")
    async def live():
        return {"live": True}

    @app.get("/health/ready")
    @app.get("/health/tracking")
    async def ready():
        is_ready = selected.ready and (selected.gpu_ready or not production)
        return JSONResponse(
            {
                "ready": is_ready,
                "gpu_ready": selected.gpu_ready,
                "backend": "sam3_image_motion",
                "sam3_commit": SAM3_COMMIT,
                "native_sam_video_tracking": False,
                "error_code": selected.error_code,
            },
            status_code=200 if is_ready else 503,
        )

    @app.post("/v1/tracks")
    @app.post("/v1/labels")
    async def tracks(request: Request, include_attributes: bool = False):
        if not selected.ready or (production and not selected.gpu_ready):
            raise HTTPException(503, "inference unavailable")
        if not admission.acquire(blocking=False):
            raise HTTPException(429, "inference busy", headers={"Retry-After": "1"})
        if sessions.has_active_session:
            admission.release()
            raise HTTPException(409, "active_session_exists")
        task = None
        deadline = time.monotonic() + selected.settings.request_timeout_seconds
        try:
            headers = request.headers
            session_id = headers.get(SESSION_HEADER, "").strip()
            if not session_id or len(session_id) > 160:
                raise HTTPException(400, "invalid stream session")
            try:
                sequence_text = headers.get(SEQUENCE_HEADER, "")
                if len(sequence_text) > 20:
                    raise ValueError()
                sequence = int(sequence_text)
                timestamp = float(headers.get(TIMESTAMP_HEADER, ""))
                if sequence < 0 or not math.isfinite(timestamp) or timestamp < 0:
                    raise ValueError()
            except ValueError:
                raise HTTPException(400, "invalid frame metadata") from None
            if headers.get("content-type", "").split(";", 1)[0].lower() != "image/jpeg":
                raise HTTPException(415, "expected image/jpeg")
            size = headers.get("content-length")
            if size is not None:
                try:
                    size = int(size)
                except ValueError:
                    raise HTTPException(400, "invalid content length") from None
                if size < 1 or size > selected.settings.max_jpeg_bytes:
                    raise HTTPException(413, "JPEG size outside allowed bounds")

            async def read_body():
                payload = bytearray()
                async for part in request.stream():
                    if len(payload) + len(part) > selected.settings.max_jpeg_bytes:
                        raise HTTPException(413, "JPEG size outside allowed bounds")
                    payload.extend(part)
                return bytes(payload)

            payload = await asyncio.wait_for(
                read_body(), max(0.001, deadline - time.monotonic())
            )

            def infer():
                image = decode_jpeg(payload, selected.settings)
                try:
                    return selected.infer(
                        image,
                        stream_session_id=session_id,
                        frame_sequence=sequence,
                        timestamp_seconds=timestamp,
                        deadline=deadline,
                    )
                finally:
                    image.close()

            task = asyncio.create_task(asyncio.to_thread(infer))
            return await asyncio.wait_for(
                asyncio.shield(task), max(0.001, deadline - time.monotonic())
            )
        except FrameOrderError:
            raise HTTPException(
                409, "source frame moved backwards; restart with a new session id"
            ) from None
        except BusyError:
            raise HTTPException(
                429, "inference busy", headers={"Retry-After": "1"}
            ) from None
        except TimeoutError:
            raise HTTPException(504, "inference deadline exceeded") from None
        except HTTPException:
            raise
        except Exception:
            LOGGER.exception("SAM scene inference failed")
            raise HTTPException(503, "inference unavailable") from None
        finally:
            if task is not None and not task.done():

                def finished(background):
                    # Consume abandoned exceptions; keep admission closed until
                    # CUDA actually finishes, bounding outstanding work to one.
                    try:
                        background.result()
                    except (Exception, asyncio.CancelledError):
                        pass
                    admission.release()

                task.add_done_callback(finished)
            else:
                admission.release()

    return app


app = create_app()
