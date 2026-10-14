"""Local InsightFace recognition with a small persistent identity gallery.

The module deliberately keeps InsightFace optional. Importing this module does
not import InsightFace or ONNX Runtime; both are resolved only when inference is
first requested. The runtime implements the same ``GroundingClient`` shape as
the vision-language client so it can be passed directly to ``StreamingSession``.
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import math
import os
import queue
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import cv2
import numpy as np

from .config import InferenceSettings
from .contracts import PUBLIC_MODEL_NAME, NormalizedBBox
from .face_tracking import FaceIdentityTracker
from .grounding import (
    GroundedClassification,
    GroundingResponse,
    ModelObject,
    build_face_diagnostics,
)
from .person_identity import stranger_evidence
from .privacy import LocalFaceUnavailableError
from .reference_memory import UNKNOWN_FACE_EVENT_LABEL
from .rich_labeling import RichLabelCoordinator, RichLabelSnapshot
from .sliding_window import FrameWindow

if TYPE_CHECKING:
    from .person_registry import PersonRegistry

__all__ = [
    "FaceGallery",
    "FaceMatch",
    "FaceRecognitionRuntime",
    "FaceRecognitionSession",
    "FaceRecognizerConfig",
    "GalleryPerson",
]

_GALLERY_SCHEMA_VERSION = 2
_LEGACY_GALLERY_SCHEMA_VERSION = 1
_EMBEDDING_PROFILE = "local-face-embedding-v1"
_ANALYSIS_KIND = "face_recognition"
_SUPPORTED_PROVIDERS = frozenset(
    {"CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"}
)
_ACCELERATED_PROVIDERS = _SUPPORTED_PROVIDERS - {"CPUExecutionProvider"}
_RUNTIME_OUTPUT_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FaceRecognizerConfig:
    """Configuration for local InsightFace inference and gallery storage."""

    model_name: str = "buffalo_l"
    model_root: Path = Path(".models/insightface")
    allow_model_download: bool = False
    provider: str = "CPUExecutionProvider"
    det_size: int | tuple[int, int] = 640
    threshold: float = 0.4
    min_face_size: int = 20
    enrollment_min_face_size: int = 80
    gallery_path: Path = Path(".goodvision/face-gallery.json")
    inference_instances: int = 2
    inference_threads: int = 2

    def __post_init__(self) -> None:
        for name, upper in (("inference_instances", 4), ("inference_threads", 16)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer between 1 and {upper}")
        model_name = str(self.model_name).strip()
        if not model_name:
            raise ValueError("model_name must not be empty")
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "model_root", Path(self.model_root).expanduser())
        object.__setattr__(self, "gallery_path", Path(self.gallery_path).expanduser())

        if not isinstance(self.allow_model_download, bool):
            raise TypeError("allow_model_download must be true or false")
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
        if (
            isinstance(self.enrollment_min_face_size, bool)
            or not isinstance(self.enrollment_min_face_size, int)
            or self.enrollment_min_face_size < 1
        ):
            raise ValueError("enrollment_min_face_size must be a positive integer")


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


@dataclass(frozen=True, slots=True)
class FaceGallerySnapshot:
    version: str
    entries: tuple[tuple[str, str, tuple[tuple[float, ...], ...]], ...]

    @property
    def people(self):
        return {person_id: name for person_id, name, _ in self.entries}

    def matches(self, embedding):
        vector = _normalized_embedding(embedding)
        result = []
        for person_id, name, samples in self.entries:
            if any(len(sample) != vector.size for sample in samples):
                raise RuntimeError("query embedding dimension differs from gallery")
            similarity = max(float(np.dot(vector, sample)) for sample in samples)
            result.append(FaceMatch(person_id, name, max(-1.0, min(1.0, similarity))))
        return tuple(sorted(result, key=lambda value: value.similarity, reverse=True))

    def as_private_dict(self):
        return {"version": self.version, "entries": self.entries}

    @classmethod
    def from_private_dict(cls, payload):
        return cls(payload["version"], tuple(
            (person_id, name, tuple(tuple(sample) for sample in samples))
            for person_id, name, samples in payload["entries"]
        ))


@dataclass(frozen=True, slots=True)
class FaceObservation:
    window: FrameWindow
    objects: tuple[ModelObject, ...]
    gallery: FaceGallerySnapshot
    started: float
    decode_seconds: float
    inference_seconds: float
    error: str | None = None
    pool_wait_seconds: float = 0.0


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
        self._revision = uuid4().hex
        self._snapshot_cache = None
        self._load()

    def private_snapshot(self) -> FaceGallerySnapshot:
        with self._lock:
            if self._snapshot_cache is None:
                self._snapshot_cache = FaceGallerySnapshot(self._revision, tuple(
                    (person_id, entry.name, tuple(tuple(float(v) for v in sample)
                                                 for sample in entry.embeddings))
                    for person_id, entry in sorted(self._people.items())
                ))
            return self._snapshot_cache

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
                raise RuntimeError("could not persist face gallery") from exc
            self._revision, self._snapshot_cache = uuid4().hex, None
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
                raise RuntimeError("could not persist face gallery") from exc
            self._revision, self._snapshot_cache = uuid4().hex, None
            return True

    def match(self, embedding: Any) -> FaceMatch | None:
        matches = self.matches(embedding)
        return matches[0] if matches else None

    def matches(self, embedding: Any) -> tuple[FaceMatch, ...]:
        """Rank identities for conservative switching without exposing vectors."""
        vector = _normalized_embedding(embedding)
        with self._lock:
            if not self._people:
                return ()
            if vector.size != self._embedding_dimension:
                raise RuntimeError(
                    "query embedding dimension does not match the face gallery: "
                    f"expected {self._embedding_dimension}, got {vector.size}"
                )
            matches: list[FaceMatch] = []
            for person_id, entry in self._people.items():
                similarity = max(
                    float(np.dot(vector, sample)) for sample in entry.embeddings
                )
                similarity = max(-1.0, min(1.0, similarity))
                matches.append(FaceMatch(person_id, entry.name, similarity))
            return tuple(
                sorted(matches, key=lambda item: item.similarity, reverse=True)
            )

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            migrate_legacy = self._load_payload(payload)
            if migrate_legacy:
                self._persist_locked()
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError("face gallery could not be loaded") from exc

    def _load_payload(self, payload: Any) -> bool:
        if not isinstance(payload, Mapping):
            raise TypeError("gallery root must be an object")
        schema_version = payload.get("schema_version")
        migrate_legacy = schema_version == _LEGACY_GALLERY_SCHEMA_VERSION
        if schema_version not in {
            _LEGACY_GALLERY_SCHEMA_VERSION,
            _GALLERY_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported face gallery schema")
        if migrate_legacy:
            stored_model = _required_text(payload.get("model_name"), "model_name")
            profile_matches = stored_model == self.model_name
        else:
            stored_profile = _required_text(
                payload.get("embedding_profile"), "embedding_profile"
            )
            profile_matches = stored_profile == _embedding_profile(self.model_name)
        if not profile_matches:
            raise ValueError("face gallery embedding profile is incompatible")
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
        return migrate_legacy

    def _persist_locked(self) -> None:
        payload = {
            "schema_version": _GALLERY_SCHEMA_VERSION,
            "embedding_profile": _embedding_profile(self.model_name),
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
            model=PUBLIC_MODEL_NAME,
            event_label=UNKNOWN_FACE_EVENT_LABEL,
            system_prompt="Classify faces locally; no remote prompt is used.",
            prompt="Classify enrolled and unenrolled faces in the newest frame.",
            request_parameters={},
        )
        self.gallery = gallery or FaceGallery(
            config.gallery_path,
            model_name=config.model_name,
        )
        self._model: Any | None = None
        self._additional_models = {}
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._instance_locks = [self._inference_lock] + [
            threading.Lock() for _ in range(config.inference_instances - 1)
        ]
        self._available_models = queue.LifoQueue()
        for slot in reversed(range(config.inference_instances)):
            self._available_models.put_nowait(slot)
        self._warmup_lock = threading.Lock()
        self._warmed = False

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

    @property
    def warmed(self) -> bool:
        with self._warmup_lock:
            return self._warmed

    def warmup(self) -> float:
        """Load the model and execute one blank inference before live frames."""

        started = time.monotonic()
        with self._warmup_lock:
            if self._warmed:
                return 0.0
            width, height = _detector_size(self.config.det_size)
            image = np.zeros((height, width, 3), dtype=np.uint8)
            for slot in range(self.config.inference_instances):
                self._infer_slot(slot, image)
            self._warmed = True
        return time.monotonic() - started

    def enroll(self, person_id: str, name: str, image_bytes: bytes) -> GalleryPerson:
        image = _decode_image(image_bytes)
        faces = self._infer(image, operation="enrollment")
        usable = [
            face
            for face in faces
            if self._face_is_large_enough(
                face,
                image,
                minimum=self.config.enrollment_min_face_size,
            )
        ]
        if not usable:
            if faces:
                raise ValueError(
                    "no face meets "
                    "enrollment_min_face_size="
                    f"{self.config.enrollment_min_face_size}"
                )
            raise ValueError("no face detected in enrollment image")
        if len(usable) != 1:
            raise ValueError(
                "enrollment image must contain exactly one usable face; "
                f"found {len(usable)}"
            )
        embedding = _face_embedding(usable[0])
        if embedding is None:
            raise ValueError("face embedding is unavailable")
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
        return self._classify_window(window)

    def _classify_window(
        self,
        window: FrameWindow,
        *,
        tracker: FaceIdentityTracker | PersonRegistry | None = None,
        snapshot: RichLabelSnapshot | None = None,
    ) -> GroundingResponse:
        return self.complete_observation(
            self.observe_window(window), tracker=tracker, snapshot=snapshot,
        )

    def observe_window(
        self, window: FrameWindow, *, gallery: FaceGallerySnapshot | None = None,
    ) -> FaceObservation:
        """Decode/detect/recognize without mutating any run identity history."""
        started = time.monotonic()
        gallery = gallery or self.gallery.private_snapshot()
        try:
            image = _decode_image(window.newest_frame.jpeg_bytes)
        except ValueError as exc:
            return FaceObservation(
                window, (), gallery, started, time.monotonic() - started, 0, str(exc),
            )

        decoded = time.monotonic()
        faces, pool_wait, inference_seconds = self._infer_timed(image)
        height, width = image.shape[:2]
        objects: list[ModelObject] = []

        for face in faces:
            pixel_bbox = _face_bbox(face, width=width, height=height)
            if pixel_bbox is None:
                continue
            x1, y1, x2, y2 = pixel_bbox
            face_width = x2 - x1
            face_height = y2 - y1
            detection_score = _detection_score(face)
            embedding = None
            attributes: dict[str, Any] = {
                "detection_score": detection_score,
                "face_width": round(face_width, 3),
                "face_height": round(face_height, 3),
                "threshold": float(self.config.threshold),
            }

            quality_reason = _face_quality_reason(face, pixel_bbox, detection_score)
            if min(face_width, face_height) < self.config.min_face_size:
                status = "unusable"
                label = "unusable_face"
                attributes["reason"] = "face_too_small"
            elif quality_reason is not None:
                status = "unusable"
                label = "unusable_face"
                attributes["reason"] = quality_reason
            else:
                embedding = _face_embedding(face)
                if embedding is None:
                    status = "unusable"
                    label = "unusable_face"
                    attributes["reason"] = "embedding_unavailable"
                else:
                    matches = gallery.matches(embedding)
                    match = matches[0] if matches else None
                    if match is not None:
                        attributes["similarity"] = match.similarity
                        attributes["match_margin"] = (
                            match.similarity - matches[1].similarity
                            if len(matches) > 1
                            else 2.0
                        )
                    if (
                        match is not None
                        and match.similarity >= self.config.threshold
                        and float(attributes["match_margin"]) >= 0.08
                    ):
                        status = "known"
                        label = "known_person"
                        attributes["person_id"] = match.person_id
                        attributes["name"] = match.name
                    elif (
                        match is not None
                        and match.similarity >= self.config.threshold
                    ):
                        status = "unusable"
                        label = "unusable_face"
                        attributes["reason"] = "ambiguous_gallery_match"
                    else:
                        status = "unknown"
                        label = UNKNOWN_FACE_EVENT_LABEL

            attributes["status"] = status
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
                    identity_embedding=(
                        tuple(float(value) for value in embedding)
                        if embedding is not None
                        else ()
                    ),
                    identity_quality=_strict_identity_quality(
                        face,
                        pixel_bbox,
                        detection_score,
                    ),
                )
            )
        return FaceObservation(
            window, tuple(objects), gallery, started, decoded - started,
            inference_seconds, pool_wait_seconds=pool_wait,
        )

    def complete_observation(
        self, observation: FaceObservation, *,
        tracker=None, snapshot=None, gallery=None,
    ) -> GroundingResponse:
        """Ordered, small identity update after a read-only inference completes."""
        window, started = observation.window, observation.started
        if observation.error is not None:
            return GroundingResponse(window, None, time.monotonic() - started,
                                     error=observation.error)
        objects = list(observation.objects)
        gallery = gallery or self.gallery.private_snapshot()
        if gallery.version != observation.gallery.version:
            # Re-evaluate only the inexpensive gallery lookup if enrollment was
            # edited in flight. Never resurrect a deleted gallery identity.
            objects = [self._rematch_face(item, gallery) for item in objects]
        event_indexes = []
        counts = {"known": 0, "unknown": 0, "unusable": 0}
        # Capture actual observations before identity confirmation/holding can
        # change their status or append a face from recent track history.
        raw_objects = tuple(objects)
        if tracker is not None:
            resolve = getattr(tracker, "resolve_faces", None) or tracker.resolve
            objects = list(
                resolve(
                    objects,
                    timestamp=window.end_timestamp,
                    frame_sequence=window.newest_frame.source_frame_index,
                    people=gallery.people,
                    snapshot=snapshot,
                )
            )
        for object_index, item in enumerate(objects):
            status = str(item.attributes["status"])
            counts[status] += 1
            if stranger_evidence(item.attributes) is True:
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
                f"最新帧人脸识别结果为{counts['known']}名已知人员、"
                f"{counts['unknown']}名陌生人、"
                f"{counts['unusable']}张不可用人脸。 / "
                "Face recognition on the newest frame: "
                f"{counts['known']} known, {counts['unknown']} unknown, "
                f"{counts['unusable']} unusable."
            ),
            objects=tuple(objects),
            analysis_kind=_ANALYSIS_KIND,
            event_object_indexes=tuple(event_indexes),
            face_observations=tuple(objects),
        )
        return GroundingResponse(
            window=window,
            classification=classification,
            latency_seconds=time.monotonic() - started,
            face_diagnostics=build_face_diagnostics(raw_objects, tuple(objects)),
        )

    def _rematch_face(self, item, gallery):
        if (not item.identity_embedding or (
            item.attributes.get("status") == "unusable"
            and item.attributes.get("reason") != "ambiguous_gallery_match"
        )):
            return item
        attributes = {key: value for key, value in item.attributes.items()
                      if key not in {"person_id", "name", "reason", "similarity",
                                     "match_margin", "status"}}
        matches = gallery.matches(item.identity_embedding)
        best = matches[0] if matches else None
        margin = best.similarity - matches[1].similarity if len(matches) > 1 else 2.0
        if best is not None:
            attributes.update(similarity=best.similarity, match_margin=margin)
        if best is not None and best.similarity >= self.config.threshold:
            if margin >= 0.08:
                label = "known_person"
                attributes.update(
                    status="known", person_id=best.person_id, name=best.name,
                )
            else:
                label = "unusable_face"
                attributes.update(status="unusable", reason="ambiguous_gallery_match")
        else:
            label = UNKNOWN_FACE_EVENT_LABEL
            attributes["status"] = "unknown"
        return replace(item, label=label, attributes=attributes)

    def _ensure_model(self, slot=0) -> Any:
        if slot == 0 and self._model is not None:
            return self._model
        with self._model_lock:
            if slot == 0:
                if self._model is None:
                    self._model = _load_face_analysis(self.config)
                return self._model
            if slot not in self._additional_models:
                self._additional_models[slot] = _load_face_analysis(self.config)
            return self._additional_models[slot]

    def _infer_slot(self, slot, image):
        model = self._ensure_model(slot)
        try:
            with self._instance_locks[slot]:
                return tuple(model.get(image))
        except Exception as exc:
            raise RuntimeError("local face analysis failed") from exc

    def _infer(
        self,
        image: np.ndarray[Any, Any],
        *,
        operation: str,
    ) -> tuple[Any, ...]:
        return self._infer_timed(image)[0]

    def _infer_timed(self, image):
        started = time.monotonic()
        slot = self._available_models.get()
        try:
            ready = time.monotonic()
            result = self._infer_slot(slot, image)
            return result, ready - started, time.monotonic() - ready
        finally:
            self._available_models.put(slot)

    def _face_is_large_enough(
        self,
        face: Any,
        image: np.ndarray[Any, Any],
        *,
        minimum: int,
    ) -> bool:
        height, width = image.shape[:2]
        bbox = _face_bbox(face, width=width, height=height)
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        return min(x2 - x1, y2 - y1) >= minimum


class FaceRecognitionSession:
    """Share model/gallery resources while keeping recognition history isolated."""

    def __init__(
        self, runtime: FaceRecognitionRuntime, *,
        observation_store_path: Path | None = None,
    ) -> None:
        self.runtime = runtime
        self.settings = runtime.settings
        self._tracker = FaceIdentityTracker()
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._rich_labels: RichLabelCoordinator | None = None
        self._last_rich_sequence = -1
        self._last_source_sequence = -1
        self._person_registry: PersonRegistry | None = None
        self._worker = None
        self._worker_lock = threading.Lock()
        self._observation_store_path = observation_store_path
        self.fast_results = None

    def bind_person_registry(self, registry: PersonRegistry) -> None:
        with self._lock:
            self._person_registry = registry
            self._session_id = registry.stream_session_id

    def submit_window(self, window: FrameWindow, *, on_admitted=None):
        """Durably register source order before any inference thread can start."""
        from .face_worker import FaceObservationWorker

        with self._worker_lock:
            if self._worker is None:
                run_id = self._session_id or uuid4().hex
                path = Path(self._observation_store_path or (
                    self.runtime.config.gallery_path.parent / "private-face"
                ))
                if path.suffix.lower() not in {".sqlite", ".sqlite3"}:
                    path = path / f"{uuid4().hex}.sqlite3"
                self._worker = FaceObservationWorker(self, path, run_id)
            worker = self._worker
        return worker.submit(window, on_admitted=on_admitted)

    def stats(self):
        with self._worker_lock:
            return self._worker.stats() if self._worker else {
                "scheduler": "durable_ordered_pool",
                "instances": self.runtime.config.inference_instances,
                "inference_threads": self.runtime.config.inference_threads,
                "submitted": 0, "completed": 0, "failed": 0, "active": 0, "backlog": 0,
            }

    def window_for(self, future):
        return self._worker.window_for(future)

    def close(self, *, wait=False):
        with self._worker_lock:
            worker = self._worker
        if worker is not None:
            worker.close(wait=wait)

    def complete_observation(self, observation, *, gallery=None):
        with self._lock:
            return self.runtime.complete_observation(
                observation, tracker=self._person_registry or self._tracker,
                snapshot=self._snapshot_locked(observation.window),
                gallery=gallery,
            )

    def bind_rich_labels(
        self, session_id: str, coordinator: RichLabelCoordinator | None
    ) -> None:
        with self._lock:
            self._session_id = session_id
            self._rich_labels = coordinator
            self._tracker = FaceIdentityTracker()
            self._last_rich_sequence = -1
            self._last_source_sequence = -1

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        with self._lock:
            return self.runtime._classify_window(
                window, tracker=self._person_registry or self._tracker,
                snapshot=self._snapshot_locked(window),
            )

    def _snapshot_locked(self, window):
        snapshot = None
        if self._rich_labels is not None:
            # Raw completed frames only: the display selector may hold or
            # interpolate boxes past the useful lifetime of identity evidence.
            candidates = [
                item
                for item in self._rich_labels.history()
                if item.stream_session_id == self._session_id
                and item.frame_sequence <= window.newest_frame.source_frame_index
                and item.timestamp_seconds <= window.end_timestamp
            ]
            snapshot = max(
                candidates, key=lambda item: item.timestamp_seconds, default=None
            )
            if (
                snapshot is not None
                and window.end_timestamp - snapshot.timestamp_seconds <= 0.75
                and (
                    window.newest_frame.source_frame_index
                    > self._last_source_sequence
                )
            ):
                self._last_rich_sequence = snapshot.frame_sequence
                self._last_source_sequence = window.newest_frame.source_frame_index
        return snapshot


def _strict_identity_quality(
    face: Any,
    bbox: tuple[float, float, float, float],
    score: float | None,
) -> bool:
    """Only a clear near-frontal face can invalidate an established identity."""
    if (
        score is None
        or score < 0.8
        or min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < 80
        or _face_quality_reason(face, bbox, score) is not None
    ):
        return False
    landmarks = _face_value(face, "kps")
    if landmarks is None:
        return False
    points = np.asarray(landmarks, dtype=np.float64)
    eye_span = float(np.linalg.norm(points[1] - points[0]))
    eye_center = (points[0] + points[1]) * 0.5
    return (
        eye_span >= (bbox[2] - bbox[0]) * 0.25
        and abs(float(points[2, 0] - eye_center[0])) <= eye_span * 0.25
    )


def _face_quality_reason(
    face: Any,
    bbox: tuple[float, float, float, float],
    detection_score: float | None,
) -> str | None:
    if detection_score is None or detection_score < 0.65:
        return "low_detection_confidence"
    landmarks = _face_value(face, "kps")
    if landmarks is not None:
        try:
            points = np.asarray(landmarks, dtype=np.float64)
        except (TypeError, ValueError):
            return "invalid_face_landmarks"
        if points.shape != (5, 2) or not np.all(np.isfinite(points)):
            return "invalid_face_landmarks"
        # A collapsed eye span or a nose outside both eyes indicates an extreme
        # profile/false face. Ordinary side views still reach gallery matching.
        eye_span = float(np.linalg.norm(points[1] - points[0]))
        eye_center = (points[0] + points[1]) * 0.5
        if eye_span < (bbox[2] - bbox[0]) * 0.15 or (
            abs(float(points[2, 0] - eye_center[0])) > eye_span * 0.8
        ):
            return "extreme_face_angle"
    return None


def _load_face_analysis(config: FaceRecognizerConfig) -> Any:
    """Import and initialize InsightFace on first use."""

    # Some optional runtime libraries print their provider choice and model
    # paths directly instead of using Python logging. Capture both streams so
    # those implementation details cannot reach container logs.
    try:
        with _silence_runtime_output():
            return _load_face_analysis_inner(config)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("local face analysis could not be initialized") from exc


@contextmanager
def _silence_runtime_output() -> Any:
    """Capture Python and native fd output during optional engine startup."""

    sink = io.StringIO()
    with _RUNTIME_OUTPUT_LOCK:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null_fd, 1)
            os.dup2(null_fd, 2)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            os.close(null_fd)


def _load_face_analysis_inner(config: FaceRecognizerConfig) -> Any:
    """Load the optional local face engine while output capture is active."""

    try:
        insightface_app = importlib.import_module("insightface.app")
        onnxruntime = importlib.import_module("onnxruntime")
    except ModuleNotFoundError as exc:
        raise LocalFaceUnavailableError("local face analysis is unavailable") from exc

    if config.provider == "CUDAExecutionProvider":
        # The GPU distribution can install its CUDA/cuDNN runtime libraries as
        # Python packages. Preload them before InsightFace creates ONNX
        # sessions; the NVIDIA container runtime supplies only the host driver.
        preload_dlls = getattr(onnxruntime, "preload_dlls", None)
        if callable(preload_dlls):
            preload_dlls(directory="")

    available = set(onnxruntime.get_available_providers())
    if config.provider not in available:
        raise RuntimeError("configured local face execution mode is unavailable")
    face_analysis = getattr(insightface_app, "FaceAnalysis", None)
    if face_analysis is None:
        raise RuntimeError("local face analysis is unavailable")

    model_directory = config.model_root / "models" / config.model_name
    if not model_directory.is_dir() and not config.allow_model_download:
        raise RuntimeError(
            "authorized local face analysis files are not installed and automatic "
            "download is disabled"
        )

    config.model_root.mkdir(parents=True, exist_ok=True)
    try:
        model = face_analysis(
            name=config.model_name,
            root=str(config.model_root),
            allowed_modules=["detection", "recognition"],
            providers=[config.provider],
        )
        _configure_face_session_threads(model, onnxruntime, config)
        model.prepare(
            ctx_id=-1 if config.provider == "CPUExecutionProvider" else 0,
            det_size=_detector_size(config.det_size),
        )
        if config.provider in _ACCELERATED_PROVIDERS:
            _require_accelerated_face_sessions(model, config.provider)
    except Exception as exc:
        raise RuntimeError("local face analysis could not be initialized") from exc
    return model


def _configure_face_session_threads(model, onnxruntime, config):
    """InsightFace 1.0.1 drops sess_options in its model_zoo.get_model.

    Replace each selected module's session before prepare; shape-dependent
    module metadata still describes the same ONNX file. Avoid global patches.
    """
    modules = getattr(model, "models", None)
    if not isinstance(modules, Mapping) or not {
        "detection", "recognition",
    }.issubset(modules):
        raise RuntimeError("local face CPU thread limits could not be verified")
    for name in ("detection", "recognition"):
        module = modules[name]
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = config.inference_threads
        options.inter_op_num_threads = 1
        session = onnxruntime.InferenceSession(
            module.model_file, sess_options=options, providers=[config.provider],
        )
        actual = session.get_session_options()
        if (actual.intra_op_num_threads != config.inference_threads
                or actual.inter_op_num_threads != 1):
            raise RuntimeError("local face CPU thread limits could not be verified")
        module.session = session


def _require_accelerated_face_sessions(model: Any, provider: str) -> None:
    """Fail fast when InsightFace silently creates CPU-backed sessions."""

    modules = getattr(model, "models", None)
    if not isinstance(modules, Mapping):
        raise RuntimeError("local face accelerated execution could not be verified")
    required = {"detection", "recognition"}
    if not required.issubset(modules):
        raise RuntimeError("local face accelerated execution could not be verified")
    for name in sorted(required):
        session = getattr(modules[name], "session", None)
        get_providers = getattr(session, "get_providers", None)
        providers = tuple(get_providers()) if callable(get_providers) else ()
        if not providers or providers[0] != provider:
            raise RuntimeError("local face accelerated execution could not be verified")


def _embedding_profile(_model_name: str) -> str:
    """Return the manually versioned public compatibility profile."""

    # Bump this opaque version whenever the internal embedding implementation
    # changes. It intentionally cannot be derived back to an engine/model name.
    return _EMBEDDING_PROFILE


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
