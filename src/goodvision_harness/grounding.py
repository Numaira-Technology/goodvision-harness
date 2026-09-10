"""Grounded event recognition over sampled live-video windows."""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import threading
import time
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import cv2
import numpy as np

from .config import InferenceSettings
from .contracts import PUBLIC_MODEL_NAME, NormalizedBBox, readonly_json_mapping
from .model_api import ModelAPIClient, ModelAPIError, ModelAPISettings
from .privacy import PUBLIC_ANALYSIS_ERROR, PUBLIC_FACE_ERROR
from .reference_memory import (
    FACE_RECOGNITION_PURPOSE,
    UNKNOWN_FACE_EVENT_LABEL,
    MemoryContext,
)
from .sliding_window import FrameWindow
from .video_to_stream import encode_jpeg

FACE_REFERENCE_ANALYSIS_KIND = "face_reference_match"
REFERENCE_MEMORY_ANALYSIS_KIND = "reference_memory"
SAFETY_MULTI_ANALYSIS_KIND = "safety_events"
FIXED_FIVE_ANALYSIS_KIND = "fixed_five_events"
SELECTED_FEATURES_ANALYSIS_KIND = "selected_features"

SAFETY_EVENT_LABELS = ("fight", "smoking", "fall", "weapon")
STRANGER_EVENT_LABEL = "stranger"
DETECTION_EVENT_LABELS = (*SAFETY_EVENT_LABELS, STRANGER_EVENT_LABEL)
MIN_MULTI_EVENT_MAX_TOKENS = 1024
_LOGGER = logging.getLogger(__name__)

# Enforce the smoking evidence floor when building the actual request, including
# deployments that still supply an older, deliberately permissive prompt.
SMOKING_EVIDENCE_RULE = (
    "Mandatory smoking evidence rule (overrides any high-recall preference, "
    "low threshold, ambiguous-positive instruction, or smoking example): "
    "detected=true requires BOTH an identifiable cigarette, cigar, smoking pipe, "
    "or e-cigarette visibly separate from the person's fingers in the NEWEST "
    "frame AND a visible action using, holding, or preparing that item. The item "
    "may be lit or unlit; smoke is not required. One or two fingers held to the "
    "mouth, pinched fingers, a bare hand, skin, fingernails, or miming a puff "
    "are not smoking items and must produce detected=false when no actual "
    "smoking item is identifiable. A pen, straw, food, or a generic small "
    "white/cylindrical object near the lips is insufficient. First distinguish "
    "the item from fingers and ordinary objects using its visible physical "
    "features; posture or context cannot establish what the object is. If the "
    "item is hidden, too small, blurred, or cannot be distinguished from a "
    "finger or pen, use detected=false and objects=[], not a low-confidence "
    "positive. Never invent a concealed cigarette or put a cigarette box on "
    "fingers. For a positive, the description must name the visible item, "
    "its distinguishing visual feature, and the person's visible interaction; "
    "return only the actual smoking item's tight box."
)


def _system_prompt_with_evidence_rules(
    system_prompt: str, event_labels: Collection[str]
) -> str:
    if "smoking" in {label.strip().casefold() for label in event_labels}:
        return f"{system_prompt}\n\n{SMOKING_EVIDENCE_RULE}"
    return system_prompt


BILINGUAL_DESCRIPTION_RULE = (
    "Every description must be a single bilingual string: one concise Simplified "
    "Chinese sentence, followed by ' / ', followed by its equivalent English "
    "sentence. Both sentences must describe the same visible evidence, without "
    "adding details in either language. This language rule also applies to "
    "negative judgments and supersedes earlier monolingual description examples. "
    "Keep JSON keys, event_label values, object labels, and identifiers unchanged."
)
ACTOR_ASSOCIATION_RULE = (
    "Additional optional tracking metadata, without changing any event verdict "
    "or objects rule: add a top-level actors array with at most 8 clearly visible "
    "people in the NEWEST LIVE image, including when every event is negative. "
    "Prioritize event participants, then other clearly visible people. Each actor "
    "has actor_id (a short ID local to this response, such as a1), confidence "
    "(0..1), and a tight visible body bbox_2d on the same 0..1000 grid. "
    "Do not invent people, names, stable tracking IDs, or hidden body extent. "
    "An event object may additionally have actor_ids listing only these local "
    "IDs for people visibly participating in or holding/using that object. "
    "Use actor_ids=[] if ownership is uncertain; proximity alone is insufficient. "
    "Keep smoking/weapon objects as item boxes; put body boxes only in actors. "
    "Negative judgments still use objects=[] but may have top-level actors. "
    "Prioritize complete event judgments and target boxes within the existing "
    "output budget; omit auxiliary actors before omitting required event fields. "
    "Do not describe actors in prose. Reference images are never live actors."
)
_EVENT_NAMES_ZH = {
    "fight": "打架",
    "smoking": "吸烟",
    "fall": "跌倒",
    "weapon": "武器",
    "stranger": "陌生人",
    "unknown_person": "陌生人",
}


@dataclass(frozen=True, slots=True)
class ModelObject:
    """One object grounded by the model on the newest window frame."""

    label: str
    bbox: NormalizedBBox
    confidence: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict, hash=False)
    # Private association data: never serialized into the legacy result objects.
    actor_ids: tuple[str, ...] = ()
    identity_embedding: tuple[float, ...] = field(default=(), repr=False, compare=False)
    identity_quality: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attributes",
            readonly_json_mapping(self.attributes, name="model object attributes"),
        )


@dataclass(frozen=True, slots=True)
class ModelActor:
    """A response-local body observation, not a persistent identity."""

    actor_id: str
    bbox: NormalizedBBox
    confidence: float = 1.0
    matched_memory_id: str | None = None


@dataclass(frozen=True, slots=True)
class ObservationBundle:
    """Private observations accompanying one unchanged public analysis record."""

    actors: tuple[ModelActor, ...] = ()
    object_actor_ids: tuple[tuple[str, ...], ...] = ()
    face_observations: tuple[ModelObject, ...] = ()


def observation_bundle(classification: Any) -> ObservationBundle:
    return ObservationBundle(
        actors=tuple(getattr(classification, "actors", ())),
        object_actor_ids=tuple(
            item.actor_ids for item in getattr(classification, "objects", ())
        ),
        face_observations=tuple(getattr(classification, "face_observations", ())),
    )


@dataclass(frozen=True, slots=True)
class GroundedClassification:
    detected: bool
    confidence: float
    description: str
    objects: tuple[ModelObject, ...] = ()
    analysis_kind: str = "event"
    event_object_indexes: tuple[int, ...] | None = None
    matched_memory_ids: tuple[str, ...] = ()
    actors: tuple[ModelActor, ...] = ()
    face_observations: tuple[ModelObject, ...] = ()

    def __post_init__(self) -> None:
        analysis_kind = str(self.analysis_kind).strip()
        if not analysis_kind:
            raise ValueError("analysis_kind must not be empty")
        object.__setattr__(self, "analysis_kind", analysis_kind)

        memory_ids = tuple(self.matched_memory_ids)
        if any(
            not isinstance(memory_id, str) or not memory_id for memory_id in memory_ids
        ):
            raise TypeError("matched_memory_ids must contain non-empty text")
        if len(set(memory_ids)) != len(memory_ids):
            raise ValueError("matched_memory_ids must not contain duplicates")
        object.__setattr__(self, "matched_memory_ids", memory_ids)

        indexes = self.event_object_indexes
        if indexes is not None:
            try:
                normalized = tuple(indexes)
            except TypeError as exc:
                raise TypeError(
                    "event_object_indexes must be an iterable of integers"
                ) from exc
            if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in normalized
            ):
                raise TypeError("event_object_indexes must contain only integers")
            if any(index < 0 or index >= len(self.objects) for index in normalized):
                raise ValueError("event_object_indexes contains an out-of-range index")
            if len(set(normalized)) != len(normalized):
                raise ValueError("event_object_indexes must not contain duplicates")
            object.__setattr__(self, "event_object_indexes", normalized)


@dataclass(frozen=True, slots=True)
class GroundedEventClassification:
    """One rich grounded judgment inside a multi-event classification."""

    event_label: str
    detected: bool
    confidence: float
    description: str
    objects: tuple[ModelObject, ...] = ()
    matched_memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        event_label = str(self.event_label).strip()
        if not event_label:
            raise ValueError("event_label must not be empty")
        object.__setattr__(self, "event_label", event_label)
        if not isinstance(self.detected, bool):
            raise TypeError("detected must be true or false")
        confidence = _number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        description = str(self.description).strip()
        if not description:
            raise ValueError("description must not be empty")
        object.__setattr__(self, "description", description)

        objects = tuple(self.objects)
        if self.detected and not objects:
            raise ValueError("detected=true requires at least one grounded object")
        if not self.detected and objects:
            raise ValueError("detected=false requires objects=[]")
        object.__setattr__(self, "objects", objects)

        memory_ids = tuple(self.matched_memory_ids)
        if any(
            not isinstance(memory_id, str) or not memory_id for memory_id in memory_ids
        ):
            raise TypeError("matched_memory_ids must contain non-empty text")
        if len(set(memory_ids)) != len(memory_ids):
            raise ValueError("matched_memory_ids must not contain duplicates")
        object.__setattr__(self, "matched_memory_ids", memory_ids)


@dataclass(frozen=True, slots=True)
class MultiGroundedClassification:
    """A set of independently grounded event judgments for one exact frame."""

    judgments: tuple[GroundedEventClassification, ...]
    analysis_kind: str = SAFETY_MULTI_ANALYSIS_KIND
    missing_event_labels: tuple[str, ...] = ()
    actors: tuple[ModelActor, ...] = ()
    face_observations: tuple[ModelObject, ...] = ()
    # Observations such as known faces belong in the frame result without
    # becoming evidence for a positive safety or stranger judgment.
    supporting_objects: tuple[ModelObject, ...] = ()

    def __post_init__(self) -> None:
        judgments = tuple(self.judgments)
        if not judgments:
            raise ValueError("judgments must not be empty")
        event_labels = [judgment.event_label for judgment in judgments]
        if len(set(event_labels)) != len(event_labels):
            raise ValueError("event_label values must be unique")
        object.__setattr__(self, "judgments", judgments)
        object.__setattr__(self, "supporting_objects", tuple(self.supporting_objects))
        analysis_kind = str(self.analysis_kind).strip()
        if not analysis_kind:
            raise ValueError("analysis_kind must not be empty")
        object.__setattr__(self, "analysis_kind", analysis_kind)

    @property
    def detected(self) -> bool:
        return any(judgment.detected for judgment in self.judgments)

    @property
    def confidence(self) -> float:
        positives = tuple(
            judgment.confidence for judgment in self.judgments if judgment.detected
        )
        if positives:
            return max(positives)
        return min(judgment.confidence for judgment in self.judgments)

    @property
    def description(self) -> str:
        positives = tuple(judgment for judgment in self.judgments if judgment.detected)
        if not positives:
            return "No target events detected."
        return "; ".join(
            f"{judgment.event_label}: {judgment.description}" for judgment in positives
        )

    @property
    def objects(self) -> tuple[ModelObject, ...]:
        return tuple(
            item for judgment in self.judgments for item in judgment.objects
        ) + self.supporting_objects

    @property
    def matched_memory_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                memory_id
                for judgment in self.judgments
                for memory_id in judgment.matched_memory_ids
            )
        )


@dataclass(frozen=True, slots=True)
class GroundingResponse:
    window: FrameWindow
    classification: GroundedClassification | MultiGroundedClassification | None
    latency_seconds: float
    raw_text: str | None = None
    response_id: str | None = None
    usage: Mapping[str, Any] | None = None
    error: str | None = None
    # A missing CPU face sample is neither a failure nor a negative judgment.
    # Health is tracked independently so partial success cannot hide an outage.
    branch_errors: tuple[tuple[str, str], ...] = ()
    completed_branches: tuple[str, ...] = ()
    # Bounded local runtime diagnostics, never part of the public v2 result.
    face_diagnostics: Mapping[str, Any] | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.classification is not None


_FACE_DIAGNOSTIC_REASONS = frozenset(
    {
        "face_too_small",
        "face_too_small_for_stranger",
        "low_detection_confidence",
        "invalid_face_landmarks",
        "extreme_face_angle",
        "embedding_unavailable",
        "ambiguous_gallery_match",
        "identity_unconfirmed",
        "identity_removed",
        "out_of_order_frame",
    }
)
_FACE_DIAGNOSTIC_LIMIT = 8


def build_face_diagnostics(
    raw_objects: tuple[ModelObject, ...] | None,
    stabilized_objects: tuple[ModelObject, ...],
) -> dict[str, Any]:
    """Keep observation quality separate from the identity confirmation state.

    Raw observations must be captured before tracking: a tracker can append a
    briefly held identity even when the current detector found no face. Legacy
    clients without that information explicitly report it as unavailable.
    Names, identifiers, embeddings and arbitrary attributes are never copied.
    """

    def status(item: ModelObject) -> str:
        value = item.attributes.get("status")
        return value if value in ("known", "unknown", "unusable") else "other"

    def reason(item: ModelObject) -> str:
        value = item.attributes.get("reason")
        if value is None:
            return "none"
        return (
            value
            if isinstance(value, str) and value in _FACE_DIAGNOSTIC_REASONS
            else "other"
        )

    def counts(items: tuple[ModelObject, ...]) -> dict[str, int]:
        result = dict.fromkeys(("known", "unknown", "unusable", "other"), 0)
        for item in items:
            result[status(item)] += 1
        return result

    def summaries(items: tuple[ModelObject, ...]) -> list[dict[str, Any]]:
        result = []
        for item in items[:_FACE_DIAGNOSTIC_LIMIT]:
            entry: dict[str, Any] = {
                "bbox": [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
                "status": status(item),
                "reason": reason(item),
            }
            for key, maximum in (
                ("face_width", 1_000_000),
                ("face_height", 1_000_000),
                ("detection_score", 1),
                ("unknown_confirmations", 1_000_000),
            ):
                value = item.attributes.get(key)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and 0 <= value <= maximum
                    and math.isfinite(value)
                ):
                    entry[key] = value
            source = item.attributes.get("identity_source")
            if source in ("face_match", "track_history"):
                entry["identity_source"] = source
            result.append(entry)
        return result

    reason_counts: dict[str, int] = {}
    for item in stabilized_objects:
        key = reason(item)
        reason_counts[key] = reason_counts.get(key, 0) + 1
    return {
        "state": (
            "unavailable" if raw_objects is None else "faces" if raw_objects
            else "no_faces"
        ),
        "raw_observations_available": raw_objects is not None,
        "detected_face_count": None if raw_objects is None else len(raw_objects),
        "face_count": len(stabilized_objects),
        "raw_status_counts": None if raw_objects is None else counts(raw_objects),
        "stable_status_counts": counts(stabilized_objects),
        "reason_counts": reason_counts,
        "raw_faces": None if raw_objects is None else summaries(raw_objects),
        "faces": summaries(stabilized_objects),
        "truncated": max(len(raw_objects or ()), len(stabilized_objects))
        > _FACE_DIAGNOSTIC_LIMIT,
    }


def _record_analysis_failure(
    window: FrameWindow,
    *,
    branch: str,
    stage: str,
    error: Exception | None = None,
    code: str = "request_failed",
) -> str:
    """Log only bounded diagnostics, never provider text or camera data."""

    status_code = None
    if isinstance(error, ModelAPIError):
        code = error.code
        status_code = error.status_code
    _LOGGER.warning(
        "analysis_failure branch=%s stage=%s window=%d code=%s http_status=%s",
        branch,
        stage,
        window.index,
        code,
        status_code,
    )
    return code


class GroundingClient(Protocol):
    settings: InferenceSettings

    def classify_window(self, window: FrameWindow) -> GroundingResponse: ...


def _json_object(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("model response did not contain a JSON object") from None
        try:
            payload = json.loads(candidate[start:end])
        except json.JSONDecodeError as exc:
            raise ValueError("model response contained invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("grounded classification must be a JSON object")
    return payload


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError(f"{name} must be finite")
    return rendered


def _matched_memory_ids(
    payload: Mapping[str, Any],
    *,
    allowed_memory_ids: Collection[str] | None,
) -> tuple[str, ...]:
    raw_memory_ids = payload.get("matched_memory_ids", [])
    if not isinstance(raw_memory_ids, list):
        raise TypeError("matched_memory_ids must be an array")

    matched_memory_ids: list[str] = []
    seen_memory_ids: set[str] = set()
    for memory_id in raw_memory_ids:
        if not isinstance(memory_id, str) or not memory_id:
            raise TypeError("matched_memory_ids must contain non-empty text")
        if memory_id in seen_memory_ids:
            raise ValueError("matched_memory_ids must not contain duplicates")
        seen_memory_ids.add(memory_id)
        matched_memory_ids.append(memory_id)

    if allowed_memory_ids is None:
        if matched_memory_ids:
            raise ValueError("model reported memory matches without a memory context")
        return tuple(matched_memory_ids)

    if "matched_memory_ids" not in payload:
        raise ValueError("memory-aware result omitted matched_memory_ids")
    allowed = set(allowed_memory_ids)
    unknown_ids = [
        memory_id for memory_id in matched_memory_ids if memory_id not in allowed
    ]
    if unknown_ids:
        raise ValueError("model reported unknown memory IDs: " + ", ".join(unknown_ids))
    return tuple(matched_memory_ids)


def _parse_model_objects(
    raw_objects: Any,
    *,
    field_name: str = "objects",
) -> tuple[ModelObject, ...]:
    if raw_objects is None:
        raw_objects = []
    if not isinstance(raw_objects, list):
        raise TypeError(f"{field_name} must be an array")

    objects: list[ModelObject] = []
    for index, raw_object in enumerate(raw_objects):
        object_field = f"{field_name}[{index}]"
        if not isinstance(raw_object, Mapping):
            raise TypeError(f"{object_field} must be an object")
        label = raw_object.get("label")
        if not isinstance(label, str) or not label.strip():
            raise TypeError(f"{object_field}.label must be non-empty text")
        raw_bbox = raw_object.get("bbox_2d")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise TypeError(f"{object_field}.bbox_2d must contain four numbers")
        x1, y1, x2, y2 = (
            _number(value, f"{object_field}.bbox_2d") for value in raw_bbox
        )
        if any(value < 0.0 or value > 1000.0 for value in (x1, y1, x2, y2)):
            raise ValueError(f"{object_field}.bbox_2d must use the 0..1000 grid")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{object_field}.bbox_2d must satisfy x2>x1 and y2>y1")
        object_confidence = raw_object.get("confidence")
        parsed_object_confidence: float | None = None
        if object_confidence is not None:
            parsed_object_confidence = _number(
                object_confidence, f"{object_field}.confidence"
            )
            if not 0.0 <= parsed_object_confidence <= 1.0:
                raise ValueError(f"{object_field}.confidence must be between 0 and 1")
        attributes = raw_object.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise TypeError(f"{object_field}.attributes must be an object")
        objects.append(
            ModelObject(
                label=label.strip(),
                bbox=NormalizedBBox(
                    x1 / 1000.0,
                    y1 / 1000.0,
                    x2 / 1000.0,
                    y2 / 1000.0,
                ),
                confidence=parsed_object_confidence,
                attributes=attributes,
                actor_ids=_actor_ids(raw_object.get("actor_ids")),
            )
        )
    return tuple(objects)


def _actor_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        dict.fromkeys(
            item
            for item in value[:8]
            if isinstance(item, str) and item.strip() and len(item) <= 64
        )
    )


def _parse_actors(
    payload: Mapping[str, Any],
    *,
    allowed_memory_ids: Collection[str] | None = None,
    allow_identity: bool = False,
) -> tuple[ModelActor, ...]:
    """Auxiliary metadata is fail-open; a bad actor cannot lose a real alert."""
    raw = payload.get("actors", [])
    if not isinstance(raw, list):
        return ()
    actors: dict[str, ModelActor] = {}
    duplicates: set[str] = set()
    for item in raw[:32]:
        if not isinstance(item, Mapping):
            continue
        actor_id = item.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id.strip() or len(actor_id) > 64:
            continue
        if actor_id in actors:
            duplicates.add(actor_id)
            continue
        try:
            box = _parse_model_objects(
                [
                    {
                        "label": "person",
                        "bbox_2d": item.get("bbox_2d"),
                        "confidence": item.get("confidence", 1.0),
                    }
                ]
            )[0]
        except (TypeError, ValueError):
            continue
        memory_id = item.get("matched_memory_id")
        if not (
            allow_identity
            and isinstance(memory_id, str)
            and memory_id in (allowed_memory_ids or ())
        ):
            memory_id = None
        actors[actor_id] = ModelActor(
            actor_id,
            box.bbox,
            box.confidence or 0.0,
            memory_id,
        )
    return tuple(item for key, item in actors.items() if key not in duplicates)[:8]


def _normalize_bilingual_description(description: str) -> str:
    chinese, separator, english = description.partition(" / ")
    # Some valid model judgments return both translations in reverse order.
    # Reorder text locally; never discard an immediate verdict or issue a
    # second inference request merely to correct this presentation detail.
    if (
        separator
        and description.count(" / ") == 1
        and re.search(r"[A-Za-z]", chinese)
        and not re.search(r"[\u3400-\u9fff]", chinese)
        and re.search(r"[\u3400-\u9fff]", english)
    ):
        chinese, english = english, chinese
    if (
        not separator
        or not re.search(r"[\u3400-\u9fff]", chinese)
        or not re.search(r"[A-Za-z]", english)
        or re.search(r"[\u3400-\u9fff]", english)
    ):
        raise ValueError("description must use Chinese / English format")
    return f"{chinese.strip()} / {english.strip()}"


def parse_grounded_classification(
    text: str,
    *,
    allowed_memory_ids: Collection[str] | None = None,
    analysis_kind: str = "event",
    require_bilingual: bool = False,
) -> GroundedClassification:
    """Parse event judgment and Qwen-relative 2D boxes.

    Qwen grounding coordinates use a relative 0..1000 canvas. The public
    contract deliberately converts them to source-normalized 0..1 xyxy.
    Positive judgments require at least one valid grounded object.
    """

    payload = _json_object(text)
    detected = payload.get("detected")
    confidence = payload.get("confidence")
    description = payload.get("description")
    if not isinstance(detected, bool):
        raise TypeError("detected must be true or false")
    confidence_value = _number(confidence, "confidence")
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not isinstance(description, str) or not description.strip():
        raise TypeError("description must be non-empty text")
    if require_bilingual:
        description = _normalize_bilingual_description(description)
    matched_memory_ids = _matched_memory_ids(
        payload,
        allowed_memory_ids=allowed_memory_ids,
    )

    objects = _parse_model_objects(payload.get("objects", []))
    if detected and not objects:
        raise ValueError("detected=true requires at least one grounded object")
    return GroundedClassification(
        detected=detected,
        confidence=confidence_value,
        description=description.strip(),
        objects=objects,
        analysis_kind=analysis_kind,
        matched_memory_ids=matched_memory_ids,
        actors=_parse_actors(
            payload,
            allowed_memory_ids=allowed_memory_ids,
            allow_identity=analysis_kind == FACE_REFERENCE_ANALYSIS_KIND,
        ),
    )


def parse_multi_grounded_classification(
    text: str,
    *,
    event_labels: tuple[str, ...] = SAFETY_EVENT_LABELS,
    analysis_kind: str = SAFETY_MULTI_ANALYSIS_KIND,
    require_bilingual: bool = False,
) -> MultiGroundedClassification:
    """Parse one strict grounded judgment for every requested event label."""

    requested_labels = tuple(str(label).strip() for label in event_labels)
    if not requested_labels or any(not label for label in requested_labels):
        raise ValueError("event_labels must contain non-empty values")
    if len(set(requested_labels)) != len(requested_labels):
        raise ValueError("event_labels must be unique")

    payload = _json_object(text)
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise TypeError("events must be an array")

    expected = set(requested_labels)
    parsed: dict[str, GroundedEventClassification] = {}
    for index, raw_event in enumerate(raw_events):
        field_name = f"events[{index}]"
        if not isinstance(raw_event, Mapping):
            raise TypeError(f"{field_name} must be an object")
        event_label = raw_event.get("event_label")
        if not isinstance(event_label, str) or not event_label.strip():
            raise TypeError(f"{field_name}.event_label must be non-empty text")
        event_label = event_label.strip()
        if event_label not in expected:
            raise ValueError(f"unknown event_label: {event_label}")
        if event_label in parsed:
            raise ValueError(f"duplicate event_label: {event_label}")

        detected = raw_event.get("detected")
        if not isinstance(detected, bool):
            raise TypeError(f"{field_name}.detected must be true or false")
        confidence = _number(raw_event.get("confidence"), f"{field_name}.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{field_name}.confidence must be between 0 and 1")
        description = raw_event.get("description")
        if not isinstance(description, str) or not description.strip():
            raise TypeError(f"{field_name}.description must be non-empty text")
        if require_bilingual:
            description = _normalize_bilingual_description(description)
        objects = _parse_model_objects(
            raw_event.get("objects", []),
            field_name=f"{field_name}.objects",
        )
        parsed[event_label] = GroundedEventClassification(
            event_label=event_label,
            detected=detected,
            confidence=confidence,
            description=description,
            objects=objects,
        )

    missing = [label for label in requested_labels if label not in parsed]
    if missing:
        raise ValueError(
            "model response is missing event labels: " + ", ".join(missing)
        )
    return MultiGroundedClassification(
        judgments=tuple(parsed[label] for label in requested_labels),
        analysis_kind=analysis_kind,
        actors=_parse_actors(payload),
    )


def build_grounding_messages(
    window: FrameWindow,
    *,
    event_label: str,
    system_prompt: str,
    prompt: str,
    memory_context: MemoryContext | None = None,
) -> list[dict[str, Any]]:
    effective_event_label = (
        UNKNOWN_FACE_EVENT_LABEL
        if memory_context is not None
        and memory_context.purpose == FACE_RECOGNITION_PURPOSE
        else event_label
    )
    rendered_prompt = prompt.replace("{event}", effective_event_label).replace(
        "{window_label}", window.label
    )
    live_frames = (
        window.frames[-1:]
        if memory_context is not None and memory_context.newest_frame_only
        else window.frames
    )
    timestamps = ", ".join(
        f"f{frame.sample_index}={frame.timestamp_seconds:.3f}s" for frame in live_frames
    )
    if memory_context is None:
        text = (
            f"{rendered_prompt}\n\n"
            f"Target event: {event_label}\n"
            f"Window: {window.label}\n"
            f"Sample timestamps: {timestamps}\n"
            "The JPEG images follow in chronological order."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    else:
        decision_prompt = rendered_prompt
        if memory_context.purpose == FACE_RECOGNITION_PURPOSE:
            decision_prompt = (
                "Classify faces in the NEWEST live camera frame, using the supplied "
                "references only as known-identity context."
            )
        text = (
            "BEGIN LIVE CAMERA INPUT\n"
            f"{decision_prompt}\n\n"
            f"Target event: {effective_event_label}\n"
            f"Window: {window.label}\n"
            f"Sample timestamps: {timestamps}\n"
            "The live JPEG images follow in chronological order.\n\n"
            "Authoritative memory decision rule:\n"
            f"{memory_context.instruction}\n\n"
            "Harness result contract:\n"
            f"{memory_context.result_contract}\n\n"
            "For this memory-aware request, respond with only one JSON object. "
            "It must contain detected, confidence, matched_memory_ids, description, "
            "and objects. matched_memory_ids may contain only supplied MEMORY_ID "
            "values. detected and matched_memory_ids are independent. objects must "
            "contain only live targets that caused detected=true; use an empty array "
            "when detected=false. Each object must contain label and "
            "bbox_2d=[x1,y1,x2,y2] on Qwen's 0..1000 grid; detected=true requires "
            "at least one object."
        )
        content = list(memory_context.prompt_prefix)
        content.append({"type": "text", "text": text})
    actor_rule = ACTOR_ASSOCIATION_RULE
    if (
        memory_context is not None
        and memory_context.purpose == FACE_RECOGNITION_PURPOSE
    ):
        actor_rule += (
            " Only for this known-face reference task, an actor may include "
            "matched_memory_id naming one supplied MEMORY_ID confidently matched "
            "to that actor's currently visible face; otherwise omit it. Never "
            "assign a global memory match to an actor without a visible face."
        )
    content[-1]["text"] += f"\n\n{actor_rule}"
    content.extend(
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,"
                + base64.b64encode(frame.jpeg_bytes).decode("ascii")
            },
        }
        for frame in live_frames
    )
    return [
        {
            "role": "system",
            "content": _system_prompt_with_evidence_rules(
                system_prompt, (effective_event_label,)
            )
            + f"\n\n{BILINGUAL_DESCRIPTION_RULE}",
        },
        {"role": "user", "content": content},
    ]


def build_multi_grounding_messages(
    window: FrameWindow,
    *,
    system_prompt: str,
    prompt: str,
    event_labels: tuple[str, ...] = SAFETY_EVENT_LABELS,
    append_contract: bool = True,
) -> list[dict[str, Any]]:
    """Build one Qwen request that judges all fixed safety events at once."""

    rendered_events = ", ".join(event_labels)
    rendered_prompt = (
        prompt.replace("{events}", rendered_events)
        .replace("{event}", rendered_events)
        .replace("{window_label}", window.label)
    )
    timestamps = ", ".join(
        f"f{frame.sample_index}={frame.timestamp_seconds:.3f}s"
        for frame in window.frames
    )
    system_text = (
        f"{_system_prompt_with_evidence_rules(system_prompt, event_labels)}\n"
        f"{BILINGUAL_DESCRIPTION_RULE}\n"
        "For this request, independently judge every fixed "
        "safety event and allow multiple simultaneous positives."
    )
    contract = (
        "Authoritative fixed safety-event output contract (this supersedes any "
        "earlier single-event JSON example): return only one JSON object with an "
        "events array. The array must contain exactly one item for each target "
        "event in the stated order and use its exact event_label. Every item must "
        "contain event_label, detected (boolean), confidence (0..1 confidence in "
        "that verdict), description (one short sentence of visible evidence), and "
        "objects (array). Each object must contain label, confidence (0..1), and "
        "bbox_2d=[x1,y1,x2,y2] on Qwen's relative 0..1000 top-left-origin "
        "xyxy grid for the NEWEST image. A positive item requires at least one "
        "tight target box; a negative item must use objects=[]. For fight and "
        "fall, return only the involved person's body box or boxes. For smoking, "
        "return only the cigarette, cigar, pipe, or e-cigarette box, never the "
        "smoker's person box. For weapon, use the same high-recall policy but "
        "return only the firearm, knife, machete, club, or other weapon box, "
        "never the holder's person box."
    )
    contract_section = f"\n\n{contract}" if append_contract else ""
    text = (
        f"{rendered_prompt}\n\n"
        f"Target safety events (exact order): {rendered_events}\n"
        f"Window: {window.label}\n"
        f"Sample timestamps: {timestamps}"
        f"{contract_section}\n\n{ACTOR_ASSOCIATION_RULE}\n"
        "The live JPEG images follow in chronological order."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,"
                + base64.b64encode(frame.jpeg_bytes).decode("ascii")
            },
        }
        for frame in window.frames
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]


def _memory_analysis_kind(memory_context: MemoryContext | None) -> str:
    if memory_context is None:
        return "event"
    if memory_context.purpose == FACE_RECOGNITION_PURPOSE:
        return FACE_REFERENCE_ANALYSIS_KIND
    return REFERENCE_MEMORY_ANALYSIS_KIND


class AIClubGroundingClient:
    def __init__(
        self,
        settings: InferenceSettings,
        transport: ModelAPIClient,
        memory_context: MemoryContext | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport
        self.memory_context = memory_context

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        try:
            response = self._transport.chat_completion(
                model=self.settings.model,
                messages=build_grounding_messages(
                    window,
                    event_label=self.settings.event_label,
                    system_prompt=self.settings.system_prompt,
                    prompt=self.settings.prompt,
                    memory_context=self.memory_context,
                ),
                **dict(self.settings.request_parameters),
            )
            text = self._transport.assistant_text(response)
        except (ModelAPIError, ValueError) as exc:
            code = _record_analysis_failure(
                window,
                branch="safety",
                stage="request",
                error=exc,
            )
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=None,
                latency_seconds=time.monotonic() - started,
                error=PUBLIC_ANALYSIS_ERROR,
                branch_errors=(("safety", code),),
            )
        latency = time.monotonic() - started
        try:
            classification = parse_grounded_classification(
                text,
                allowed_memory_ids=(
                    self.memory_context.memory_ids
                    if self.memory_context is not None
                    else None
                ),
                analysis_kind=_memory_analysis_kind(self.memory_context),
                require_bilingual=True,
            )
        except (TypeError, ValueError):
            code = _record_analysis_failure(
                window,
                branch="safety",
                stage="classification",
                code="invalid_response",
            )
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=None,
                latency_seconds=latency,
                error=PUBLIC_ANALYSIS_ERROR,
                branch_errors=(("safety", code),),
            )
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=None,
            latency_seconds=latency,
            completed_branches=("safety",),
        )


class DryRunGroundingClient:
    """Deterministic local client used by tests and zero-cost demos."""

    def __init__(
        self,
        settings: InferenceSettings,
        memory_context: MemoryContext | None = None,
    ) -> None:
        self.settings = settings
        self.memory_context = memory_context

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        time.sleep(0.35 + (window.index * 13 % 40) / 100.0)
        detected = window.index % 6 in {2, 3, 4}
        confidence = round(0.6 + (window.index * 37 % 100) / 250.0, 3)
        memory_ids = (
            self.memory_context.memory_ids if self.memory_context is not None else ()
        )
        if self.memory_context is not None and not memory_ids:
            detected = False
        state = "visibly active" if detected else "not visible"
        event_name = _EVENT_NAMES_ZH.get(
            self.settings.event_label, self.settings.event_label
        )
        objects: tuple[ModelObject, ...] = ()
        if detected:
            objects = (
                ModelObject(
                    label="person",
                    bbox=NormalizedBBox(0.18, 0.12, 0.72, 0.9),
                    confidence=confidence,
                    actor_ids=("a1",),
                ),
            )
        classification = GroundedClassification(
            detected=detected,
            confidence=confidence,
            description=(
                f"演示帧{window.label}中"
                f"{'检测到' if detected else '未检测到'}"
                f"{event_name}"
                "事件。 / "
                f"Dry run: {self.settings.event_label} {state} in {window.label}."
            ),
            objects=objects,
            analysis_kind=_memory_analysis_kind(self.memory_context),
            matched_memory_ids=(
                memory_ids[:1]
                if self.memory_context is not None and window.index % 2 == 0
                else ()
            ),
            actors=(ModelActor("a1", NormalizedBBox(0.18, 0.12, 0.72, 0.9)),),
        )
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=None,
            latency_seconds=time.monotonic() - started,
        )


class AIClubMultiGroundingClient:
    """Run one Qwen request for all four visual safety event labels."""

    def __init__(
        self,
        settings: InferenceSettings,
        transport: ModelAPIClient,
        event_labels: tuple[str, ...] = SAFETY_EVENT_LABELS,
    ) -> None:
        labels = tuple(event_labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("event_labels must be non-empty and unique")
        self.settings = settings
        self.event_labels = labels
        self.memory_context = None
        self.model_name = PUBLIC_MODEL_NAME
        self._transport = transport

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        multi_prompt = str(getattr(self.settings, "multi_event_prompt", "")).strip()
        # The configured multi-event prompt contains a fully expanded four-label
        # JSON example. Reuse it only for that exact request; selected subsets use
        # the generic prompt plus the dynamic authoritative contract below.
        use_fixed_prompt = (
            bool(multi_prompt) and self.event_labels == SAFETY_EVENT_LABELS
        )
        prompt = multi_prompt if use_fixed_prompt else self.settings.prompt
        request_parameters = dict(self.settings.request_parameters)
        configured_max_tokens = request_parameters.get("max_tokens")
        if configured_max_tokens is None:
            request_parameters["max_tokens"] = MIN_MULTI_EVENT_MAX_TOKENS
        elif not isinstance(configured_max_tokens, bool) and isinstance(
            configured_max_tokens, int
        ):
            request_parameters["max_tokens"] = max(
                int(configured_max_tokens), MIN_MULTI_EVENT_MAX_TOKENS
            )
        try:
            response = self._transport.chat_completion(
                model=self.settings.model,
                messages=build_multi_grounding_messages(
                    window,
                    system_prompt=self.settings.system_prompt,
                    prompt=prompt,
                    event_labels=self.event_labels,
                    append_contract=not use_fixed_prompt,
                ),
                **request_parameters,
            )
            text = self._transport.assistant_text(response)
        except (ModelAPIError, ValueError) as exc:
            code = _record_analysis_failure(
                window,
                branch="safety",
                stage="request",
                error=exc,
            )
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=None,
                latency_seconds=time.monotonic() - started,
                error=PUBLIC_ANALYSIS_ERROR,
                branch_errors=(("safety", code),),
            )

        latency = time.monotonic() - started
        try:
            classification = parse_multi_grounded_classification(
                text,
                event_labels=self.event_labels,
                require_bilingual=True,
            )
        except (TypeError, ValueError):
            code = _record_analysis_failure(
                window,
                branch="safety",
                stage="classification",
                code="invalid_response",
            )
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=None,
                latency_seconds=latency,
                error=PUBLIC_ANALYSIS_ERROR,
                branch_errors=(("safety", code),),
            )
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=None,
            latency_seconds=latency,
            completed_branches=("safety",),
        )


class DryRunMultiGroundingClient:
    """Deterministic four-event client for local demos and tests."""

    def __init__(
        self,
        settings: InferenceSettings,
        event_labels: tuple[str, ...] = SAFETY_EVENT_LABELS,
    ) -> None:
        self.settings = settings
        self.event_labels = tuple(event_labels)
        self.memory_context = None
        self.model_name = PUBLIC_MODEL_NAME

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        judgments: list[GroundedEventClassification] = []
        active_index = window.index % (len(self.event_labels) + 2)
        for index, event_label in enumerate(self.event_labels):
            detected = index == active_index
            confidence = 0.82 if detected else 0.91
            objects = (
                (
                    ModelObject(
                        label="person",
                        bbox=NormalizedBBox(0.18, 0.12, 0.72, 0.9),
                        confidence=confidence,
                        attributes={"dry_run": True},
                        actor_ids=("a1",),
                    ),
                )
                if detected
                else ()
            )
            judgments.append(
                GroundedEventClassification(
                    event_label=event_label,
                    detected=detected,
                    confidence=confidence,
                    description=(
                        f"演示中{'检测到' if detected else '未检测到'}"
                        f"{_EVENT_NAMES_ZH.get(event_label, event_label)}事件。 / "
                        f"Dry run: {event_label} "
                        f"{'is visible' if detected else 'is not visible'}."
                    ),
                    objects=objects,
                )
            )
        return GroundingResponse(
            window=window,
            classification=MultiGroundedClassification(
                tuple(judgments),
                actors=(ModelActor("a1", NormalizedBBox(0.18, 0.12, 0.72, 0.9)),),
            ),
            latency_seconds=time.monotonic() - started,
        )


class DryRunStrangerGroundingClient:
    """Face-runtime-shaped dry run that never loads InsightFace weights."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings
        self.memory_context = None
        self.model_name = PUBLIC_MODEL_NAME

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        detected = window.index % 4 == 1
        status = "unknown" if detected else "known"
        objects = (
            ModelObject(
                label=UNKNOWN_FACE_EVENT_LABEL if detected else "known_person",
                bbox=NormalizedBBox(0.3, 0.15, 0.62, 0.72),
                confidence=0.88,
                attributes={
                    "status": status,
                    "detection_score": 0.88,
                    "dry_run": True,
                },
            ),
        )
        classification = GroundedClassification(
            detected=detected,
            confidence=0.88,
            description=(
                f"演示人脸识别结果为{0 if detected else 1}名已知人员、"
                f"{1 if detected else 0}名陌生人、0张不可用人脸。 / "
                "Dry-run face recognition: "
                f"{1 if not detected else 0} known, "
                f"{1 if detected else 0} unknown, 0 unusable."
            ),
            objects=objects,
            analysis_kind="face_recognition",
            event_object_indexes=(0,) if detected else (),
        )
        return GroundingResponse(
            window=window,
            classification=classification,
            latency_seconds=time.monotonic() - started,
        )


def _safety_window(
    window: FrameWindow,
    *,
    max_image_side: int,
    jpeg_quality: int,
) -> FrameWindow:
    """Re-encode a face-resolution window for the lower-cost safety model."""

    if max_image_side < 1:
        raise ValueError("safety_max_image_side must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("safety_jpeg_quality must be between 1 and 100")
    frames = []
    for frame in window.frames:
        encoded = np.frombuffer(frame.jpeg_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                f"could not decode sampled frame {frame.sample_index} for safety"
            )
        jpeg_bytes, width, height = encode_jpeg(
            image,
            max_image_side=max_image_side,
            quality=jpeg_quality,
        )
        frames.append(
            replace(
                frame,
                jpeg_bytes=jpeg_bytes,
                width=width,
                height=height,
            )
        )
    return replace(window, frames=tuple(frames))


class SelectedFeaturesGroundingClient:
    """Run only selected safety/stranger detectors and merge one-frame results."""

    def __init__(
        self,
        safety_client: GroundingClient | None,
        stranger_client: GroundingClient | None,
        *,
        safety_event_labels: tuple[str, ...] = SAFETY_EVENT_LABELS,
        include_stranger: bool = True,
        safety_max_image_side: int = 320,
        safety_jpeg_quality: int = 75,
        face_live_scheduler: str = "single_flight",
    ) -> None:
        labels = tuple(safety_event_labels)
        if any(label not in SAFETY_EVENT_LABELS for label in labels):
            raise ValueError("safety_event_labels contains an unsupported label")
        if len(set(labels)) != len(labels):
            raise ValueError("safety_event_labels must be unique")
        canonical_labels = tuple(
            label for label in SAFETY_EVENT_LABELS if label in labels
        )
        if labels != canonical_labels:
            raise ValueError("safety_event_labels must use canonical order")
        if bool(labels) != (safety_client is not None):
            raise ValueError("safety client must match selected safety features")
        if include_stranger != (stranger_client is not None):
            raise ValueError("stranger client must match selected stranger feature")
        if not labels and not include_stranger:
            raise ValueError("at least one feature must be selected")
        if safety_max_image_side < 1:
            raise ValueError("safety_max_image_side must be positive")
        if not 1 <= safety_jpeg_quality <= 100:
            raise ValueError("safety_jpeg_quality must be between 1 and 100")
        if face_live_scheduler not in {"full", "single_flight"}:
            raise ValueError("face_live_scheduler must be full or single_flight")
        selected_client = safety_client or stranger_client
        assert selected_client is not None
        self.settings = selected_client.settings
        self.event_labels = labels + (
            (STRANGER_EVENT_LABEL,) if include_stranger else ()
        )
        self.memory_context = None
        self.model_name = PUBLIC_MODEL_NAME
        self._safety_client = safety_client
        self._stranger_client = stranger_client
        self._safety_event_labels = labels
        self._include_stranger = include_stranger
        self._safety_max_image_side = safety_max_image_side
        self._safety_jpeg_quality = safety_jpeg_quality
        self.face_live_scheduler = face_live_scheduler
        # InsightFace owns one inference session protected by a model lock.  The
        # outer live pipeline can have several model API windows in flight, but
        # queueing every one of those windows at the face-model lock turns a
        # single CPU inference into many seconds of stale backlog.  CPU mode
        # therefore admits at most one live face branch per selected-features
        # session. Accelerated providers deliberately run the face branch for
        # every admitted mixed-detector request so their throughput can be
        # measured without silently reducing stranger sampling.
        self._stranger_live_slot = threading.Lock()

    @property
    def independent_live_branches(self) -> bool:
        """The live pipeline may schedule each selected detector separately."""

        return self._safety_client is not None and self._stranger_client is not None

    def classify_safety_window(self, window: FrameWindow) -> GroundingResponse:
        """Run only safety; the outer pipeline owns concurrency and freshness."""

        if self._safety_client is None:
            raise ValueError("safety branch is not selected")
        started = time.monotonic()
        try:
            resized_window = _safety_window(
                window,
                max_image_side=self._safety_max_image_side,
                jpeg_quality=self._safety_jpeg_quality,
            )
        except (ValueError, cv2.error):
            code = _record_analysis_failure(
                window, branch="safety", stage="input", code="invalid_response"
            )
            return GroundingResponse(
                window=window,
                classification=None,
                latency_seconds=time.monotonic() - started,
                error=PUBLIC_ANALYSIS_ERROR,
                branch_errors=(("safety", code),),
            )
        response = None
        error = None
        try:
            response = self._safety_client.classify_window(resized_window)
        except Exception as exc:
            error = exc
        return self._merge_responses(
            window,
            started=started,
            include_safety=True,
            include_stranger=False,
            safety_response=response,
            safety_error=error,
        )

    def classify_stranger_window(self, window: FrameWindow) -> GroundingResponse:
        """Run local faces on the original image, without a remote-model slot."""

        if self._stranger_client is None:
            raise ValueError("stranger branch is not selected")
        started = time.monotonic()
        response = None
        error = None
        try:
            response = self._stranger_client.classify_window(window)
        except Exception as exc:
            error = exc
        return self._merge_responses(
            window,
            started=started,
            include_safety=False,
            include_stranger=True,
            stranger_response=response,
            stranger_error=error,
        )

    def _classify_stranger_with_slot(
        self,
        window: FrameWindow,
    ) -> GroundingResponse:
        """Run one live face sample and always release its single-flight slot."""

        assert self._stranger_client is not None
        try:
            return self._stranger_client.classify_window(window)
        finally:
            self._stranger_live_slot.release()

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        resized_window: FrameWindow | None = None
        if self._safety_client is not None:
            try:
                resized_window = _safety_window(
                    window,
                    max_image_side=self._safety_max_image_side,
                    jpeg_quality=self._safety_jpeg_quality,
                )
            except ValueError:
                code = _record_analysis_failure(
                    window,
                    branch="safety",
                    stage="input",
                    code="invalid_response",
                )
                return GroundingResponse(
                    window=window,
                    classification=None,
                    latency_seconds=time.monotonic() - started,
                    error=PUBLIC_ANALYSIS_ERROR,
                    branch_errors=(("safety", code),),
                )

        safety_response: GroundingResponse | None = None
        stranger_response: GroundingResponse | None = None
        safety_error: Exception | None = None
        stranger_error: Exception | None = None
        if (
            self._safety_client is not None
            and self._stranger_client is not None
            and (
                self.face_live_scheduler == "full"
                or self._stranger_live_slot.acquire(blocking=False)
            )
        ):
            assert resized_window is not None
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="selected-features-face",
            ) as executor:
                try:
                    stranger_future = executor.submit(
                        (
                            self._classify_stranger_with_slot
                            if self.face_live_scheduler == "single_flight"
                            else self._stranger_client.classify_window
                        ),
                        window,
                    )
                except BaseException:
                    if self.face_live_scheduler == "single_flight":
                        self._stranger_live_slot.release()
                    raise
                try:
                    safety_response = self._safety_client.classify_window(
                        resized_window
                    )
                except Exception as exc:
                    safety_error = exc
                try:
                    stranger_response = stranger_future.result()
                except Exception as exc:
                    stranger_error = exc
        elif self._safety_client is not None and self._stranger_client is not None:
            # Face inference is already running for a nearby frame. Dropping
            # this face branch keeps the live path latest-first without copying
            # an old bbox onto a different frame. Model inference remains fully
            # concurrent and its complete safety result is still delivered.
            assert resized_window is not None
            try:
                safety_response = self._safety_client.classify_window(resized_window)
            except Exception as exc:
                safety_error = exc
        elif self._safety_client is not None:
            assert resized_window is not None
            try:
                safety_response = self._safety_client.classify_window(resized_window)
            except Exception as exc:
                safety_error = exc
        else:
            assert self._stranger_client is not None
            try:
                stranger_response = self._stranger_client.classify_window(window)
            except Exception as exc:
                stranger_error = exc

        return self._merge_responses(
            window,
            started=started,
            include_safety=self._safety_client is not None,
            include_stranger=self._stranger_client is not None,
            safety_response=safety_response,
            stranger_response=stranger_response,
            safety_error=safety_error,
            stranger_error=stranger_error,
            allow_missing_stranger=True,
        )

    def _merge_responses(
        self,
        window: FrameWindow,
        *,
        started: float,
        include_safety: bool,
        include_stranger: bool,
        safety_response: GroundingResponse | None = None,
        stranger_response: GroundingResponse | None = None,
        safety_error: Exception | None = None,
        stranger_error: Exception | None = None,
        allow_missing_stranger: bool = False,
    ) -> GroundingResponse:
        """Share validation for synchronous and independently completed branches."""

        judgments: list[GroundedEventClassification] = []
        supporting_objects: tuple[ModelObject, ...] = ()
        branch_errors: list[tuple[str, str]] = []
        completed_branches: list[str] = []
        safety_failed = False
        face_diagnostics = None
        actors: tuple[ModelActor, ...] = ()
        face_observations: tuple[ModelObject, ...] = ()
        if include_safety:
            safety_classification = (
                safety_response.classification
                if safety_error is None and safety_response is not None
                else None
            )
            if isinstance(safety_classification, MultiGroundedClassification):
                safety_labels = tuple(
                    judgment.event_label for judgment in safety_classification.judgments
                )
                if safety_labels == self._safety_event_labels:
                    judgments.extend(safety_classification.judgments)
                    completed_branches.append("safety")
                    actors = safety_classification.actors
                else:
                    safety_failed = True
            else:
                safety_failed = True
            if safety_failed:
                reported = (
                    dict(safety_response.branch_errors).get("safety")
                    if safety_response is not None
                    else None
                )
                code = reported or _record_analysis_failure(
                    window,
                    branch="safety",
                    stage="branch",
                    error=safety_error,
                    code=(
                        "invalid_response"
                        if safety_classification is not None
                        else "request_failed"
                    ),
                )
                branch_errors.append(("safety", code))

        if include_stranger:
            face_classification = (
                stranger_response.classification
                if stranger_error is None and stranger_response is not None
                else None
            )
            if isinstance(face_classification, GroundedClassification):
                try:
                    indexes = face_classification.event_object_indexes
                    if indexes is None:
                        indexes = (
                            tuple(range(len(face_classification.objects)))
                            if face_classification.detected
                            else ()
                        )
                    stranger_objects = tuple(
                        face_classification.objects[index] for index in indexes
                    )
                    judgments.append(
                        GroundedEventClassification(
                            event_label=STRANGER_EVENT_LABEL,
                            detected=face_classification.detected,
                            confidence=face_classification.confidence,
                            description=face_classification.description,
                            objects=stranger_objects,
                            matched_memory_ids=(face_classification.matched_memory_ids),
                        )
                    )
                    event_indexes = set(indexes)
                    supporting_objects = tuple(
                        item for index, item in enumerate(face_classification.objects)
                        if index not in event_indexes
                        and item.label == "known_person"
                        and item.attributes.get("status") == "known"
                        and item.attributes.get("person_id")
                    )
                    completed_branches.append("stranger")
                    face_observations = (
                        face_classification.face_observations
                        or face_classification.objects
                    )
                    assert stranger_response is not None
                    face_diagnostics = stranger_response.face_diagnostics
                    if face_diagnostics is None:
                        face_diagnostics = build_face_diagnostics(
                            None, face_observations
                        )
                except (TypeError, ValueError, IndexError) as exc:
                    stranger_error = exc
            # No response/error means single-flight deliberately skipped this
            # sample. Do not report it as either healthy, failed, or clear.
            if "stranger" not in completed_branches and (
                not allow_missing_stranger
                or stranger_response is not None
                or stranger_error is not None
            ):
                reported = (
                    dict(stranger_response.branch_errors).get("stranger")
                    if stranger_response is not None
                    else None
                )
                code = reported or _record_analysis_failure(
                    window,
                    branch="stranger",
                    stage="branch",
                    error=stranger_error,
                    code=(
                        "invalid_response"
                        if face_classification is not None
                        else "request_failed"
                    ),
                )
                branch_errors.append(("stranger", code))

        if not judgments:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=None,
                latency_seconds=time.monotonic() - started,
                error=(PUBLIC_ANALYSIS_ERROR if safety_failed else PUBLIC_FACE_ERROR),
                branch_errors=tuple(branch_errors),
            )

        try:
            returned_labels = tuple(judgment.event_label for judgment in judgments)
            classification = MultiGroundedClassification(
                judgments=tuple(judgments),
                actors=tuple(actors),
                face_observations=tuple(face_observations),
                missing_event_labels=tuple(
                    label for label in self.event_labels if label not in returned_labels
                ),
                supporting_objects=supporting_objects,
                analysis_kind=(
                    FIXED_FIVE_ANALYSIS_KIND
                    if returned_labels == DETECTION_EVENT_LABELS
                    else SELECTED_FEATURES_ANALYSIS_KIND
                ),
            )
        except (TypeError, ValueError):
            branch_errors.extend(
                (
                    branch,
                    _record_analysis_failure(
                        window,
                        branch=branch,
                        stage="merge",
                        code="invalid_response",
                    ),
                )
                for branch in completed_branches
            )
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=None,
                latency_seconds=time.monotonic() - started,
                error=(
                    PUBLIC_ANALYSIS_ERROR
                    if include_safety and not include_stranger
                    else PUBLIC_FACE_ERROR
                ),
                branch_errors=tuple(branch_errors),
            )
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=None,
            latency_seconds=time.monotonic() - started,
            branch_errors=tuple(branch_errors),
            completed_branches=tuple(completed_branches),
            face_diagnostics=face_diagnostics,
        )


class FixedFiveGroundingClient(SelectedFeaturesGroundingClient):
    """Backward-compatible all-five composite."""

    def __init__(
        self,
        safety_client: GroundingClient,
        stranger_client: GroundingClient,
        *,
        safety_max_image_side: int = 320,
        safety_jpeg_quality: int = 75,
    ) -> None:
        super().__init__(
            safety_client,
            stranger_client,
            safety_event_labels=SAFETY_EVENT_LABELS,
            include_stranger=True,
            safety_max_image_side=safety_max_image_side,
            safety_jpeg_quality=safety_jpeg_quality,
        )


def build_grounding_client(
    settings: InferenceSettings,
    aiclub_settings: ModelAPISettings,
    *,
    memory_context: MemoryContext | None = None,
    dry_run: bool = False,
    api_key: str | None = None,
    transport: ModelAPIClient | None = None,
) -> GroundingClient:
    if dry_run:
        return DryRunGroundingClient(settings, memory_context)
    return AIClubGroundingClient(
        settings,
        transport or ModelAPIClient(aiclub_settings, api_key=api_key),
        memory_context,
    )


def build_selected_features_grounding_client(
    settings: InferenceSettings,
    aiclub_settings: ModelAPISettings,
    *,
    features: tuple[str, ...],
    stranger_client: GroundingClient | None = None,
    safety_max_image_side: int = 320,
    safety_jpeg_quality: int = 75,
    face_live_scheduler: str = "single_flight",
    dry_run: bool = False,
    api_key: str | None = None,
    transport: ModelAPIClient | None = None,
) -> SelectedFeaturesGroundingClient:
    """Build one detector graph for an arbitrary non-empty feature subset."""

    selected = tuple(features)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("features must be non-empty and unique")
    if any(feature not in DETECTION_EVENT_LABELS for feature in selected):
        raise ValueError("features contains an unsupported label")
    canonical = tuple(
        feature for feature in DETECTION_EVENT_LABELS if feature in selected
    )
    if selected != canonical:
        raise ValueError("features must use canonical order")

    safety_labels = tuple(
        feature for feature in selected if feature in SAFETY_EVENT_LABELS
    )
    include_stranger = STRANGER_EVENT_LABEL in selected
    safety_client: GroundingClient | None = None
    effective_stranger_client: GroundingClient | None = None
    if safety_labels:
        if dry_run:
            safety_client = DryRunMultiGroundingClient(settings, safety_labels)
        else:
            safety_client = AIClubMultiGroundingClient(
                settings,
                transport or ModelAPIClient(aiclub_settings, api_key=api_key),
                safety_labels,
            )
    if include_stranger:
        if dry_run:
            face_settings = (
                stranger_client.settings
                if stranger_client is not None
                else replace(settings, event_label=UNKNOWN_FACE_EVENT_LABEL)
            )
            effective_stranger_client = DryRunStrangerGroundingClient(face_settings)
        else:
            if stranger_client is None:
                raise ValueError("stranger_client is required for stranger detection")
            effective_stranger_client = stranger_client
    return SelectedFeaturesGroundingClient(
        safety_client,
        effective_stranger_client,
        safety_event_labels=safety_labels,
        include_stranger=include_stranger,
        safety_max_image_side=safety_max_image_side,
        safety_jpeg_quality=safety_jpeg_quality,
        face_live_scheduler=face_live_scheduler,
    )


def build_fixed_five_grounding_client(
    settings: InferenceSettings,
    aiclub_settings: ModelAPISettings,
    *,
    stranger_client: GroundingClient,
    safety_max_image_side: int = 320,
    safety_jpeg_quality: int = 75,
    dry_run: bool = False,
    api_key: str | None = None,
) -> FixedFiveGroundingClient:
    """Build the fixed-five composite while keeping legacy clients unchanged."""

    safety_client: GroundingClient
    effective_stranger_client = stranger_client
    if dry_run:
        safety_client = DryRunMultiGroundingClient(settings)
        effective_stranger_client = DryRunStrangerGroundingClient(
            stranger_client.settings
        )
    else:
        safety_client = AIClubMultiGroundingClient(
            settings,
            ModelAPIClient(aiclub_settings, api_key=api_key),
        )
    return FixedFiveGroundingClient(
        safety_client,
        effective_stranger_client,
        safety_max_image_side=safety_max_image_side,
        safety_jpeg_quality=safety_jpeg_quality,
    )
