"""Local InsightFace recognition with a small persistent identity gallery.

The module deliberately keeps InsightFace optional. Importing this module does
not import InsightFace or ONNX Runtime; both are resolved only when inference is
first requested. The runtime implements the same ``GroundingClient`` shape as
the vision-language client so it can be passed directly to ``StreamingSession``.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .config import InferenceSettings
from .contracts import NormalizedBBox
from .grounding import (
    GroundedClassification,
    GroundingResponse,
    ModelObject,
)
from .sliding_window import FrameWindow

__all__ = [
    "FaceGallery",
    "FaceMatch",
    "FaceRecognitionRuntime",
    "FaceRecognizerConfig",
    "GalleryPerson",
]

_GALLERY_SCHEMA_VERSION = 1
_ANALYSIS_KIND = "face_recognition"
_EVENT_LABEL = "unknown_person"
_SUPPORTED_PROVIDERS = frozenset(
    {"CPUExecutionProvider", "CUDAExecutionProvider"}
)


@dataclass(frozen=True, slots=True)
class FaceRecognizerConfig:
    """Configuration for local InsightFace inference and gallery storage."""

    model_name: str = "buffalo_l"
    model_root: Path = Path(".models/insightface")
    provider: str = "CPUExecutionProvider"
    det_size: int | tuple[int, int] = 640
    threshold: float = 0.4
    min_face_size: int = 80
    gallery_path: Path = Path(".goodvision/face-gallery.json")

    def __post_init__(self) -> None:
        model_name = str(self.model_name).strip()
        if not model_name:
            raise ValueError("model_name must not be empty")
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "model_root", Path(self.model_root).expanduser())
        object.__setattr__(self, "gallery_path", Path(self.gallery_path).expanduser())

        if self.provider not in _SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(_SUPPORTED_PROVIDERS))
            raise ValueError(f"provider must be one of: {supported}")
        _detector_size(self.det_size)
        if (
            isinstance(self.threshold, bool)
            or not math.isfinite(float(self.threshold))
            or not -1.0 <= float(self.threshold) <= 1.0
        ):
            raise ValueError("threshold must be a finite raw cosine value from -1 to 1")
        if (
            isinstance(self.min_face_size, bool)
            or not isinstance(self.min_face_size, int)
            or self.min_face_size < 1
        ):
            raise ValueError("min_face_size must be a positive integer")


@dataclass(frozen=True, slots=True)
class GalleryPerson:
    """Public, embedding-free view of one enrolled identity."""

    person_id: str
    name: str
    sample_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "person_id": self.person_id,
            "name": self.name,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class FaceMatch:
    """Best gallery identity for one normalized query embedding."""

    person_id: str
    name: str
    similarity: float


@dataclass(slots=True)
class _GalleryEntry:
    name: str
    embeddings: list[np.ndarray[Any, np.dtype[np.float32]]]


class FaceGallery:
    """Thread-safe JSON gallery intended for a small demo identity set.

    Embeddings are L2-normalized before storage. Mutations are persisted through
    a same-directory temporary file followed by ``os.replace``, so readers never
    observe a partially written JSON document.
    """

    def __init__(self, path: str | Path, *, model_name: str) -> None:
        rendered_model = str(model_name).strip()
        if not rendered_model:
            raise ValueError("model_name must not be empty")
        self.path = Path(path).expanduser()
        self.model_name = rendered_model
        self._lock = threading.RLock()
        self._people: dict[str, _GalleryEntry] = {}
        self._embedding_dimension: int | None = None
        self._load()

    @property
    def embedding_dimension(self) -> int | None:
        with self._lock:
            return self._embedding_dimension

    def enroll(
        self,
        person_id: str,
        name: str,
        embedding: Any,
    ) -> GalleryPerson:
        rendered_id = _required_text(person_id, "person_id")
        rendered_name = _required_text(name, "name")
        vector = _normalized_embedding(embedding)
        with self._lock:
            dimension = self._embedding_dimension
            if dimension is None:
                dimension = int(vector.size)
            elif vector.size != dimension:
                raise ValueError(
                    "embedding dimension does not match the existing gallery: "
                    f"expected {dimension}, got {vector.size}"
                )
            previous_people = self._people
            previous_dimension = self._embedding_dimension
            updated_people = {
                key: _GalleryEntry(item.name, list(item.embeddings))
                for key, item in previous_people.items()
            }
            entry = updated_people.get(rendered_id)
            if entry is None:
                entry = _GalleryEntry(name=rendered_name, embeddings=[])
                updated_people[rendered_id] = entry
            else:
                entry.name = rendered_name
            entry.embeddings.append(vector.copy())
            self._people = updated_people
            self._embedding_dimension = dimension
            try:
                self._persist_locked()
            except OSError as exc:
                self._people = previous_people
                self._embedding_dimension = previous_dimension
                raise RuntimeError(f"could not persist face gallery: {exc}") from exc
            return GalleryPerson(rendered_id, entry.name, len(entry.embeddings))

    def list_people(self) -> tuple[GalleryPerson, ...]:
        with self._lock:
            return tuple(
                GalleryPerson(person_id, entry.name, len(entry.embeddings))
                for person_id, entry in sorted(self._people.items())
            )

    def delete(self, person_id: str) -> bool:
        rendered_id = _required_text(person_id, "person_id")
        with self._lock:
            if rendered_id not in self._people:
                return False
            previous_people = self._people
            previous_dimension = self._embedding_dimension
            self._people = dict(previous_people)
            del self._people[rendered_id]
            if not self._people:
                self._embedding_dimension = None
            try:
                self._persist_locked()
            except OSError as exc:
                self._people = previous_people
                self._embedding_dimension = previous_dimension
                raise RuntimeError(f"could not persist face gallery: {exc}") from exc
            return True

    def match(self, embedding: Any) -> FaceMatch | None:
        vector = _normalized_embedding(embedding)
        with self._lock:
            if not self._people:
                return None
            if vector.size != self._embedding_dimension:
                raise RuntimeError(
                    "query embedding dimension does not match the face gallery: "
                    f"expected {self._embedding_dimension}, got {vector.size}"
                )
            best: FaceMatch | None = None
            for person_id, entry in self._people.items():
                similarity = max(
                    float(np.dot(vector, sample)) for sample in entry.embeddings
                )
                similarity = max(-1.0, min(1.0, similarity))
                if best is None or similarity > best.similarity:
                    best = FaceMatch(person_id, entry.name, similarity)
            return best

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._load_payload(payload)
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid face gallery {self.path}: {exc}") from exc

    def _load_payload(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("gallery root must be an object")
        if payload.get("schema_version") != _GALLERY_SCHEMA_VERSION:
            raise ValueError(
                f"gallery schema_version must be {_GALLERY_SCHEMA_VERSION}"
            )
        stored_model = _required_text(payload.get("model_name"), "model_name")
        if stored_model != self.model_name:
            raise ValueError(
                f"gallery model is {stored_model!r}, expected {self.model_name!r}"
            )
        raw_dimension = payload.get("embedding_dimension")
        if raw_dimension is not None and (
            isinstance(raw_dimension, bool)
            or not isinstance(raw_dimension, int)
            or raw_dimension < 1
        ):
            raise ValueError("embedding_dimension must be null or a positive integer")
        raw_people = payload.get("people")
        if not isinstance(raw_people, list):
            raise TypeError("people must be an array")

        people: dict[str, _GalleryEntry] = {}
        inferred_dimension: int | None = raw_dimension
        for index, raw_person in enumerate(raw_people):
            if not isinstance(raw_person, Mapping):
                raise TypeError(f"people[{index}] must be an object")
            person_id = _required_text(raw_person.get("person_id"), "person_id")
            if person_id in people:
                raise ValueError(f"duplicate person_id {person_id!r}")
            name = _required_text(raw_person.get("name"), "name")
            raw_embeddings = raw_person.get("embeddings")
            if not isinstance(raw_embeddings, list) or not raw_embeddings:
                raise ValueError(f"people[{index}].embeddings must not be empty")
            embeddings: list[np.ndarray[Any, np.dtype[np.float32]]] = []
            for raw_embedding in raw_embeddings:
                vector = _normalized_embedding(raw_embedding)
                if inferred_dimension is None:
                    inferred_dimension = int(vector.size)
                elif vector.size != inferred_dimension:
                    raise ValueError(
                        f"embedding for {person_id!r} has dimension {vector.size}; "
                        f"expected {inferred_dimension}"
                    )
                embeddings.append(vector)
            people[person_id] = _GalleryEntry(name=name, embeddings=embeddings)

        if not people and raw_dimension is not None:
            raise ValueError("empty galleries must use null embedding_dimension")
        self._people = people
        self._embedding_dimension = inferred_dimension

    def _persist_locked(self) -> None:
        payload = {
            "schema_version": _GALLERY_SCHEMA_VERSION,
            "model_name": self.model_name,
            "embedding_dimension": self._embedding_dimension,
            "people": [
                {
                    "person_id": person_id,
                    "name": entry.name,
                    "embeddings": [sample.tolist() for sample in entry.embeddings],
                }
                for person_id, entry in sorted(self._people.items())
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


class FaceRecognitionRuntime:
    """InsightFace-backed ``GroundingClient`` plus gallery management APIs."""

    def __init__(
        self,
        config: FaceRecognizerConfig,
        *,
        gallery: FaceGallery | None = None,
    ) -> None:
        self.config = config
        self.settings = InferenceSettings(
            model=config.model_name,
            event_label=_EVENT_LABEL,
            system_prompt="Local InsightFace recognition; no LLM prompt is used.",
            prompt="Classify enrolled and unenrolled faces in the newest frame.",
            request_parameters={},
        )
        self.gallery = gallery or FaceGallery(
            config.gallery_path,
            model_name=config.model_name,
        )
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def event_label(self) -> str:
        return self.settings.event_label

    @property
    def loaded(self) -> bool:
        with self._model_lock:
            return self._model is not None

    def enroll(self, person_id: str, name: str, image_bytes: bytes) -> GalleryPerson:
        image = _decode_image(image_bytes)
        faces = self._infer(image, operation="enrollment")
        usable = [face for face in faces if self._face_is_large_enough(face, image)]
        if not usable:
            if faces:
                raise ValueError(
                    f"no face meets min_face_size={self.config.min_face_size}"
                )
            raise ValueError("no face detected in enrollment image")
        if len(usable) != 1:
            raise ValueError(
                "enrollment image must contain exactly one usable face; "
                f"found {len(usable)}"
            )
        embedding = _face_embedding(usable[0])
        if embedding is None:
            raise ValueError("InsightFace did not return an enrollment embedding")
        return self.gallery.enroll(person_id, name, embedding)

    def list(self) -> tuple[GalleryPerson, ...]:
        return self.gallery.list_people()

    def list_people(self) -> tuple[GalleryPerson, ...]:
        return self.list()

    def delete(self, person_id: str) -> bool:
        return self.gallery.delete(person_id)

    def delete_person(self, person_id: str) -> bool:
        return self.delete(person_id)

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        try:
            image = _decode_image(window.newest_frame.jpeg_bytes)
        except ValueError as exc:
            return GroundingResponse(
                window=window,
                classification=None,
                latency_seconds=time.monotonic() - started,
                error=str(exc),
            )

        faces = self._infer(image, operation="classification")
        height, width = image.shape[:2]
        objects: list[ModelObject] = []
        event_indexes: list[int] = []
        counts = {"known": 0, "unknown": 0, "unusable": 0}

        for face in faces:
            pixel_bbox = _face_bbox(face, width=width, height=height)
            if pixel_bbox is None:
                continue
            x1, y1, x2, y2 = pixel_bbox
            face_width = x2 - x1
            face_height = y2 - y1
            detection_score = _detection_score(face)
            attributes: dict[str, Any] = {
                "detection_score": detection_score,
                "face_width": round(face_width, 3),
                "face_height": round(face_height, 3),
                "threshold": float(self.config.threshold),
            }

            if min(face_width, face_height) < self.config.min_face_size:
                status = "unusable"
                label = "unusable_face"
                attributes["reason"] = "face_too_small"
            else:
                embedding = _face_embedding(face)
                if embedding is None:
                    status = "unusable"
                    label = "unusable_face"
                    attributes["reason"] = "embedding_unavailable"
                else:
                    match = self.gallery.match(embedding)
                    if match is not None:
                        attributes["similarity"] = match.similarity
                    if match is not None and match.similarity >= self.config.threshold:
                        status = "known"
                        label = "known_person"
                        attributes["person_id"] = match.person_id
                        attributes["name"] = match.name
                    else:
                        status = "unknown"
                        label = _EVENT_LABEL

            attributes["status"] = status
            object_index = len(objects)
            objects.append(
                ModelObject(
                    label=label,
                    bbox=NormalizedBBox(
                        x1 / width,
                        y1 / height,
                        x2 / width,
                        y2 / height,
                    ),
                    confidence=detection_score,
                    attributes=attributes,
                )
            )
            counts[status] += 1
            if status == "unknown":
                event_indexes.append(object_index)

        event_confidences = [
            objects[index].confidence
            for index in event_indexes
            if objects[index].confidence is not None
        ]
        all_confidences = [
            item.confidence for item in objects if item.confidence is not None
        ]
        confidence = max(event_confidences or all_confidences or [0.0])
        classification = GroundedClassification(
            detected=bool(event_indexes),
            confidence=confidence,
            description=(
                "Face recognition on the newest frame: "
                f"{counts['known']} known, {counts['unknown']} unknown, "
                f"{counts['unusable']} unusable."
            ),
            objects=tuple(objects),
            analysis_kind=_ANALYSIS_KIND,
            event_object_indexes=tuple(event_indexes),
        )
        return GroundingResponse(
            window=window,
            classification=classification,
            latency_seconds=time.monotonic() - started,
        )

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                self._model = _load_face_analysis(self.config)
            return self._model

    def _infer(
        self,
        image: np.ndarray[Any, Any],
        *,
        operation: str,
    ) -> tuple[Any, ...]:
        model = self._ensure_model()
        try:
            with self._inference_lock:
                return tuple(model.get(image))
        except Exception as exc:
            raise RuntimeError(
                f"InsightFace {operation} inference failed: {exc}"
            ) from exc

    def _face_is_large_enough(self, face: Any, image: np.ndarray[Any, Any]) -> bool:
        height, width = image.shape[:2]
        bbox = _face_bbox(face, width=width, height=height)
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        return min(x2 - x1, y2 - y1) >= self.config.min_face_size


def _load_face_analysis(config: FaceRecognizerConfig) -> Any:
    """Import and initialize InsightFace on first use."""

    try:
        insightface_app = importlib.import_module("insightface.app")
        onnxruntime = importlib.import_module("onnxruntime")
    except ModuleNotFoundError as exc:
        missing = exc.name or "InsightFace dependency"
        raise RuntimeError(
            "InsightFace face recognition is unavailable because "
            f"{missing!r} is not installed. Install 'insightface' and exactly one "
            "of 'onnxruntime' (CPU) or 'onnxruntime-gpu' (CUDA)."
        ) from exc

    available = set(onnxruntime.get_available_providers())
    if config.provider not in available:
        raise RuntimeError(
            f"requested ONNX Runtime provider {config.provider!r} is unavailable; "
            f"available providers: {', '.join(sorted(available)) or 'none'}"
        )
    face_analysis = getattr(insightface_app, "FaceAnalysis", None)
    if face_analysis is None:
        raise RuntimeError("installed InsightFace package does not expose FaceAnalysis")

    config.model_root.mkdir(parents=True, exist_ok=True)
    try:
        model = face_analysis(
            name=config.model_name,
            root=str(config.model_root),
            allowed_modules=["detection", "recognition"],
            providers=[config.provider],
        )
        model.prepare(
            ctx_id=-1 if config.provider == "CPUExecutionProvider" else 0,
            det_size=_detector_size(config.det_size),
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to initialize InsightFace model {config.model_name!r} "
            f"with provider {config.provider!r}: {exc}"
        ) from exc
    return model


def _decode_image(image_bytes: bytes) -> np.ndarray[Any, Any]:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("image_bytes must be non-empty bytes")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image_bytes do not contain a decodable image")
    return image


def _detector_size(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, bool):
        raise ValueError("det_size must be a positive integer or width/height pair")
    if isinstance(value, int):
        dimensions = (value, value)
    elif isinstance(value, tuple) and len(value) == 2:
        dimensions = value
    else:
        raise ValueError("det_size must be a positive integer or width/height pair")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in dimensions
    ):
        raise ValueError("det_size values must be positive integers")
    return dimensions


def _required_text(value: Any, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must not be empty")
    rendered = str(value).strip()
    if not rendered:
        raise ValueError(f"{name} must not be empty")
    return rendered


def _normalized_embedding(value: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding must be a numeric vector") from exc
    if vector.size < 1 or not np.all(np.isfinite(vector)):
        raise ValueError("embedding must contain finite values")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("embedding norm must be positive and finite")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def _face_value(face: Any, name: str) -> Any:
    if isinstance(face, Mapping):
        return face.get(name)
    return getattr(face, name, None)


def _face_embedding(face: Any) -> np.ndarray[Any, np.dtype[np.float32]] | None:
    embedding = _face_value(face, "normed_embedding")
    if embedding is None:
        embedding = _face_value(face, "embedding")
    if embedding is None:
        return None
    try:
        return _normalized_embedding(embedding)
    except ValueError:
        return None


def _face_bbox(
    face: Any,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    raw_bbox = _face_value(face, "bbox")
    if raw_bbox is None:
        return None
    try:
        values = np.asarray(raw_bbox, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 4 or not np.all(np.isfinite(values[:4])):
        return None
    x1 = max(0.0, min(float(width), float(values[0])))
    y1 = max(0.0, min(float(height), float(values[1])))
    x2 = max(0.0, min(float(width), float(values[2])))
    y2 = max(0.0, min(float(height), float(values[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _detection_score(face: Any) -> float | None:
    value = _face_value(face, "det_score")
    if value is None or isinstance(value, bool):
        return None
    try:
        rendered = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rendered):
        return None
    return max(0.0, min(1.0, rendered))
