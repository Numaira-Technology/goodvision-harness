"""Grounded event recognition over sampled live-video windows."""

from __future__ import annotations

import base64
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import InferenceSettings
from .contracts import NormalizedBBox, readonly_json_mapping
from .openrouter import OpenRouterClient, OpenRouterError, OpenRouterSettings
from .sliding_window import FrameWindow


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

    def __post_init__(self) -> None:
        analysis_kind = str(self.analysis_kind).strip()
        if not analysis_kind:
            raise ValueError("analysis_kind must not be empty")
        object.__setattr__(self, "analysis_kind", analysis_kind)

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
class GroundingResponse:
    window: FrameWindow
    classification: GroundedClassification | None
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


def parse_grounded_classification(text: str) -> GroundedClassification:
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

    raw_objects = payload.get("objects", [])
    if raw_objects is None:
        raw_objects = []
    if not isinstance(raw_objects, list):
        raise TypeError("objects must be an array")
    objects: list[ModelObject] = []
    for index, raw_object in enumerate(raw_objects):
        if not isinstance(raw_object, Mapping):
            raise TypeError(f"objects[{index}] must be an object")
        label = raw_object.get("label")
        if not isinstance(label, str) or not label.strip():
            raise TypeError(f"objects[{index}].label must be non-empty text")
        raw_bbox = raw_object.get("bbox_2d")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise TypeError(f"objects[{index}].bbox_2d must contain four numbers")
        x1, y1, x2, y2 = (
            _number(value, f"objects[{index}].bbox_2d") for value in raw_bbox
        )
        if any(value < 0.0 or value > 1000.0 for value in (x1, y1, x2, y2)):
            raise ValueError(f"objects[{index}].bbox_2d must use the 0..1000 grid")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"objects[{index}].bbox_2d must satisfy x2>x1 and y2>y1")
        object_confidence = raw_object.get("confidence")
        parsed_object_confidence: float | None = None
        if object_confidence is not None:
            parsed_object_confidence = _number(
                object_confidence, f"objects[{index}].confidence"
            )
            if not 0.0 <= parsed_object_confidence <= 1.0:
                raise ValueError(f"objects[{index}].confidence must be between 0 and 1")
        attributes = raw_object.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise TypeError(f"objects[{index}].attributes must be an object")
        objects.append(
            ModelObject(
                label=label.strip(),
                bbox=NormalizedBBox(x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0),
                confidence=parsed_object_confidence,
                attributes=attributes,
            )
        )
    if detected and not objects:
        raise ValueError("detected=true requires at least one grounded object")
    return GroundedClassification(
        detected=detected,
        confidence=confidence_value,
        description=description.strip(),
        objects=tuple(objects),
    )


def build_grounding_messages(
    window: FrameWindow,
    *,
    event_label: str,
    system_prompt: str,
    prompt: str,
) -> list[dict[str, Any]]:
    rendered_prompt = prompt.replace("{event}", event_label).replace(
        "{window_label}", window.label
    )
    timestamps = ", ".join(
        f"f{frame.sample_index}={frame.timestamp_seconds:.3f}s"
        for frame in window.frames
    )
    text = (
        f"{rendered_prompt}\n\n"
        f"Target event: {event_label}\n"
        f"Window: {window.label}\n"
        f"Sample timestamps: {timestamps}\n"
        "The JPEG images follow in chronological order."
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
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


class OpenRouterGroundingClient:
    def __init__(
        self,
        settings: InferenceSettings,
        transport: OpenRouterClient,
    ) -> None:
        self.settings = settings
        self._transport = transport

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
            classification = parse_grounded_classification(text)
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

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        time.sleep(0.35 + (window.index * 13 % 40) / 100.0)
        detected = window.index % 6 in {2, 3, 4}
        confidence = round(0.6 + (window.index * 37 % 100) / 250.0, 3)
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
        )
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=None,
            latency_seconds=time.monotonic() - started,
        )


def build_grounding_client(
    settings: InferenceSettings,
    openrouter_settings: OpenRouterSettings,
    *,
    dry_run: bool = False,
) -> GroundingClient:
    if dry_run:
        return DryRunGroundingClient(settings)
    return OpenRouterGroundingClient(
        settings,
        OpenRouterClient(openrouter_settings),
    )
