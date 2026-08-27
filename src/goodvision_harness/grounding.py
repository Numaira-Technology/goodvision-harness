"""Grounded event recognition over sampled live-video windows."""

from __future__ import annotations

import base64
import json
import math
import time
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import cv2
import numpy as np

from .config import InferenceSettings
from .contracts import NormalizedBBox, readonly_json_mapping
from .openrouter import OpenRouterClient, OpenRouterError, OpenRouterSettings
from .reference_memory import (
    FACE_RECOGNITION_PURPOSE,
    MemoryContext,
    UNKNOWN_FACE_EVENT_LABEL,
)
from .sliding_window import FrameWindow
from .video_to_stream import encode_jpeg


FACE_REFERENCE_ANALYSIS_KIND = "face_reference_match"
REFERENCE_MEMORY_ANALYSIS_KIND = "reference_memory"
SAFETY_MULTI_ANALYSIS_KIND = "safety_events"
FIXED_FIVE_ANALYSIS_KIND = "fixed_five_events"

SAFETY_EVENT_LABELS = ("fight", "smoking", "fall", "weapon")
STRANGER_EVENT_LABEL = "stranger"
DETECTION_EVENT_LABELS = SAFETY_EVENT_LABELS + (STRANGER_EVENT_LABEL,)
MIN_MULTI_EVENT_MAX_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class ModelObject:
    """One object grounded by the model on the newest window frame."""

    label: str
    bbox: NormalizedBBox
    confidence: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attributes",
            readonly_json_mapping(self.attributes, name="model object attributes"),
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

    def __post_init__(self) -> None:
        analysis_kind = str(self.analysis_kind).strip()
        if not analysis_kind:
            raise ValueError("analysis_kind must not be empty")
        object.__setattr__(self, "analysis_kind", analysis_kind)

        memory_ids = tuple(self.matched_memory_ids)
        if any(
            not isinstance(memory_id, str) or not memory_id
            for memory_id in memory_ids
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
            not isinstance(memory_id, str) or not memory_id
            for memory_id in memory_ids
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

    def __post_init__(self) -> None:
        judgments = tuple(self.judgments)
        if not judgments:
            raise ValueError("judgments must not be empty")
        event_labels = [judgment.event_label for judgment in judgments]
        if len(set(event_labels)) != len(event_labels):
            raise ValueError("event_label values must be unique")
        object.__setattr__(self, "judgments", judgments)
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
        positives = tuple(
            judgment for judgment in self.judgments if judgment.detected
        )
        if not positives:
            return "No target events detected."
        return "; ".join(
            f"{judgment.event_label}: {judgment.description}"
            for judgment in positives
        )

    @property
    def objects(self) -> tuple[ModelObject, ...]:
        return tuple(
            item for judgment in self.judgments for item in judgment.objects
        )

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

    @property
    def ok(self) -> bool:
        return self.classification is not None


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
                raise ValueError(
                    f"{object_field}.confidence must be between 0 and 1"
                )
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
            )
        )
    return tuple(objects)


def parse_grounded_classification(
    text: str,
    *,
    allowed_memory_ids: Collection[str] | None = None,
    analysis_kind: str = "event",
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
    )


def parse_multi_grounded_classification(
    text: str,
    *,
    event_labels: tuple[str, ...] = SAFETY_EVENT_LABELS,
    analysis_kind: str = SAFETY_MULTI_ANALYSIS_KIND,
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
        confidence = _number(
            raw_event.get("confidence"), f"{field_name}.confidence"
        )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{field_name}.confidence must be between 0 and 1")
        description = raw_event.get("description")
        if not isinstance(description, str) or not description.strip():
            raise TypeError(f"{field_name}.description must be non-empty text")
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
        f"f{frame.sample_index}={frame.timestamp_seconds:.3f}s"
        for frame in live_frames
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
        {"role": "system", "content": system_prompt},
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
        f"{system_prompt}\nFor this request, independently judge every fixed "
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
        "tight evidence box; a negative item must use objects=[]. For weapon, use "
        "the same high-recall policy: detect a person visibly holding, carrying, "
        "drawing, pointing, or brandishing a plausible firearm, knife, machete, "
        "or other clear weapon; box the person and weapon separately when visible."
    )
    contract_section = f"\n\n{contract}" if append_contract else ""
    text = (
        f"{rendered_prompt}\n\n"
        f"Target safety events (exact order): {rendered_events}\n"
        f"Window: {window.label}\n"
        f"Sample timestamps: {timestamps}"
        f"{contract_section}\n"
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


class OpenRouterGroundingClient:
    def __init__(
        self,
        settings: InferenceSettings,
        transport: OpenRouterClient,
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
        except (OpenRouterError, ValueError) as exc:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=None,
                latency_seconds=time.monotonic() - started,
                error=str(exc),
            )
        latency = time.monotonic() - started
        response_id = str(response["id"]) if response.get("id") is not None else None
        usage = response.get("usage")
        try:
            classification = parse_grounded_classification(
                text,
                allowed_memory_ids=(
                    self.memory_context.memory_ids
                    if self.memory_context is not None
                    else None
                ),
                analysis_kind=_memory_analysis_kind(self.memory_context),
            )
        except (TypeError, ValueError) as exc:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=text,
                latency_seconds=latency,
                response_id=response_id,
                usage=usage if isinstance(usage, Mapping) else None,
                error=f"unparseable model reply: {exc}",
            )
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=text,
            latency_seconds=latency,
            response_id=response_id,
            usage=usage if isinstance(usage, Mapping) else None,
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
            self.memory_context.memory_ids
            if self.memory_context is not None
            else ()
        )
        if self.memory_context is not None and not memory_ids:
            detected = False
        state = "visibly active" if detected else "not visible"
        objects: tuple[ModelObject, ...] = ()
        if detected:
            objects = (
                ModelObject(
                    label="person",
                    bbox=NormalizedBBox(0.18, 0.12, 0.72, 0.9),
                    confidence=confidence,
                ),
            )
        classification = GroundedClassification(
            detected=detected,
            confidence=confidence,
            description=(
                f"Dry run: {self.settings.event_label} {state} in {window.label}."
            ),
            objects=objects,
            analysis_kind=_memory_analysis_kind(self.memory_context),
            matched_memory_ids=(
                memory_ids[:1]
                if self.memory_context is not None and window.index % 2 == 0
                else ()
            ),
        )
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=None,
            latency_seconds=time.monotonic() - started,
        )


class OpenRouterMultiGroundingClient:
    """Run one Qwen request for all four visual safety event labels."""

    def __init__(
        self,
        settings: InferenceSettings,
        transport: OpenRouterClient,
        event_labels: tuple[str, ...] = SAFETY_EVENT_LABELS,
    ) -> None:
        labels = tuple(event_labels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("event_labels must be non-empty and unique")
        self.settings = settings
        self.event_labels = labels
        self.memory_context = None
        self.model_name = settings.model
        self._transport = transport

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        multi_prompt = str(getattr(self.settings, "multi_event_prompt", "")).strip()
        prompt = multi_prompt or self.settings.prompt
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
                    append_contract=not bool(multi_prompt),
                ),
                **request_parameters,
            )
            text = self._transport.assistant_text(response)
        except (OpenRouterError, ValueError) as exc:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=None,
                latency_seconds=time.monotonic() - started,
                error=str(exc),
            )

        latency = time.monotonic() - started
        response_id = str(response["id"]) if response.get("id") is not None else None
        usage = response.get("usage")
        try:
            classification = parse_multi_grounded_classification(
                text,
                event_labels=self.event_labels,
            )
        except (TypeError, ValueError) as exc:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=text,
                latency_seconds=latency,
                response_id=response_id,
                usage=usage if isinstance(usage, Mapping) else None,
                error=f"unparseable multi-event model reply: {exc}",
            )
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=text,
            latency_seconds=latency,
            response_id=response_id,
            usage=usage if isinstance(usage, Mapping) else None,
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
        self.model_name = f"dry-run:{settings.model}"

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
                        f"Dry run: {event_label} "
                        f"{'is visible' if detected else 'is not visible'}."
                    ),
                    objects=objects,
                )
            )
        return GroundingResponse(
            window=window,
            classification=MultiGroundedClassification(tuple(judgments)),
            latency_seconds=time.monotonic() - started,
        )


class DryRunStrangerGroundingClient:
    """Face-runtime-shaped dry run that never loads InsightFace weights."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings
        self.memory_context = None
        self.model_name = f"dry-run:{settings.model}"

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


class FixedFiveGroundingClient:
    """Merge four Qwen safety judgments with local stranger recognition."""

    def __init__(
        self,
        safety_client: GroundingClient,
        stranger_client: GroundingClient,
        *,
        safety_max_image_side: int = 320,
        safety_jpeg_quality: int = 75,
    ) -> None:
        if safety_max_image_side < 1:
            raise ValueError("safety_max_image_side must be positive")
        if not 1 <= safety_jpeg_quality <= 100:
            raise ValueError("safety_jpeg_quality must be between 1 and 100")
        self.settings = safety_client.settings
        self.event_labels = DETECTION_EVENT_LABELS
        self.memory_context = None
        self.model_name = (
            f"{safety_client.settings.model} + {stranger_client.settings.model}"
        )
        self._safety_client = safety_client
        self._stranger_client = stranger_client
        self._safety_max_image_side = safety_max_image_side
        self._safety_jpeg_quality = safety_jpeg_quality

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        try:
            resized_window = _safety_window(
                window,
                max_image_side=self._safety_max_image_side,
                jpeg_quality=self._safety_jpeg_quality,
            )
        except ValueError as exc:
            return GroundingResponse(
                window=window,
                classification=None,
                latency_seconds=time.monotonic() - started,
                error=str(exc),
            )

        safety_response: GroundingResponse | None = None
        stranger_response: GroundingResponse | None = None
        safety_error: Exception | None = None
        stranger_error: Exception | None = None
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fixed-five-face",
        ) as executor:
            stranger_future = executor.submit(
                self._stranger_client.classify_window,
                window,
            )
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

        if safety_error is not None:
            return GroundingResponse(
                window=window,
                classification=None,
                latency_seconds=time.monotonic() - started,
                error=f"safety detector crashed: {safety_error}",
            )
        assert safety_response is not None
        if safety_response.classification is None:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=safety_response.raw_text,
                latency_seconds=time.monotonic() - started,
                response_id=safety_response.response_id,
                usage=safety_response.usage,
                error=(
                    "safety detector failed: "
                    f"{safety_response.error or 'unknown error'}"
                ),
            )
        if not isinstance(
            safety_response.classification, MultiGroundedClassification
        ):
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=safety_response.raw_text,
                latency_seconds=time.monotonic() - started,
                response_id=safety_response.response_id,
                usage=safety_response.usage,
                error="safety detector returned a single-event classification",
            )
        safety_labels = tuple(
            judgment.event_label
            for judgment in safety_response.classification.judgments
        )
        if safety_labels != SAFETY_EVENT_LABELS:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=safety_response.raw_text,
                latency_seconds=time.monotonic() - started,
                response_id=safety_response.response_id,
                usage=safety_response.usage,
                error=(
                    "safety detector returned unexpected event labels: "
                    + ", ".join(safety_labels)
                ),
            )

        if stranger_error is not None:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=safety_response.raw_text,
                latency_seconds=time.monotonic() - started,
                response_id=safety_response.response_id,
                usage=safety_response.usage,
                error=f"stranger detector crashed: {stranger_error}",
            )
        assert stranger_response is not None
        face_classification = stranger_response.classification
        if face_classification is None:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=safety_response.raw_text,
                latency_seconds=time.monotonic() - started,
                response_id=safety_response.response_id,
                usage=safety_response.usage,
                error=(
                    "stranger detector failed: "
                    f"{stranger_response.error or 'unknown error'}"
                ),
            )
        if not isinstance(face_classification, GroundedClassification):
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=safety_response.raw_text,
                latency_seconds=time.monotonic() - started,
                response_id=safety_response.response_id,
                usage=safety_response.usage,
                error="stranger detector returned a multi-event classification",
            )

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
        try:
            stranger_judgment = GroundedEventClassification(
                event_label=STRANGER_EVENT_LABEL,
                detected=face_classification.detected,
                confidence=face_classification.confidence,
                description=face_classification.description,
                objects=stranger_objects,
                matched_memory_ids=face_classification.matched_memory_ids,
            )
            classification = MultiGroundedClassification(
                judgments=(
                    *safety_response.classification.judgments,
                    stranger_judgment,
                ),
                analysis_kind=FIXED_FIVE_ANALYSIS_KIND,
            )
        except (TypeError, ValueError) as exc:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=safety_response.raw_text,
                latency_seconds=time.monotonic() - started,
                response_id=safety_response.response_id,
                usage=safety_response.usage,
                error=f"invalid stranger detector result: {exc}",
            )
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=safety_response.raw_text,
            latency_seconds=time.monotonic() - started,
            response_id=safety_response.response_id,
            usage=safety_response.usage,
        )


def build_grounding_client(
    settings: InferenceSettings,
    openrouter_settings: OpenRouterSettings,
    *,
    memory_context: MemoryContext | None = None,
    dry_run: bool = False,
) -> GroundingClient:
    if dry_run:
        return DryRunGroundingClient(settings, memory_context)
    return OpenRouterGroundingClient(
        settings,
        OpenRouterClient(openrouter_settings),
        memory_context,
    )


def build_fixed_five_grounding_client(
    settings: InferenceSettings,
    openrouter_settings: OpenRouterSettings,
    *,
    stranger_client: GroundingClient,
    safety_max_image_side: int = 320,
    safety_jpeg_quality: int = 75,
    dry_run: bool = False,
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
        safety_client = OpenRouterMultiGroundingClient(
            settings,
            OpenRouterClient(openrouter_settings),
        )
    return FixedFiveGroundingClient(
        safety_client,
        effective_stranger_client,
        safety_max_image_side=safety_max_image_side,
        safety_jpeg_quality=safety_jpeg_quality,
    )
