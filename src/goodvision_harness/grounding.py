"""Grounded event recognition over sampled live-video windows."""

from __future__ import annotations

import base64
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .config import InferenceSettings
from .contracts import NormalizedBBox
from .openrouter import OpenRouterClient, OpenRouterError, OpenRouterSettings
from .sliding_window import FrameWindow


@dataclass(frozen=True, slots=True)
class ModelObject:
    """One object grounded by the model on the newest window frame."""

    label: str
    bbox: NormalizedBBox
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class GroundedClassification:
    detected: bool
    confidence: float
    description: str
    objects: tuple[ModelObject, ...] = ()


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
        objects.append(
            ModelObject(
                label=label.strip(),
                bbox=NormalizedBBox(x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0),
                confidence=parsed_object_confidence,
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
    output_contract = (
        "Judge the target event on the NEWEST (last) image. Return only one JSON "
        "object with detected (boolean), confidence (0..1), description (short "
        "text), and objects (array). Each relevant object must be "
        '{"label":"...","confidence":0..1,"bbox_2d":[x1,y1,x2,y2]}. '
        "bbox_2d uses Qwen's relative 0..1000 top-left-origin xyxy canvas for "
        "the NEWEST image. If detected=true, include at least one valid object; "
        "use [] only when detected=false."
    )
    text = (
        f"{rendered_prompt}\n\nTarget event: {event_label}\n"
        f"Window: {window.label}\nSample timestamps: {timestamps}\n"
        f"{output_contract}\nThe JPEG images follow in chronological order."
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
            classification = parse_grounded_classification(text)
        except (OpenRouterError, TypeError, ValueError) as exc:
            return GroundingResponse(
                window=window,
                classification=None,
                raw_text=locals().get("text"),
                latency_seconds=time.monotonic() - started,
                error=str(exc),
            )
        response_id = str(response["id"]) if response.get("id") is not None else None
        usage = response.get("usage")
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=text,
            latency_seconds=time.monotonic() - started,
            response_id=response_id,
            usage=usage if isinstance(usage, Mapping) else None,
        )


class DryRunGroundingClient:
    """Deterministic local client used by tests and zero-cost demos."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings

    def classify_window(self, window: FrameWindow) -> GroundingResponse:
        started = time.monotonic()
        detected = window.index % 4 in {1, 2}
        objects: tuple[ModelObject, ...] = ()
        if detected:
            shift = (window.index % 3) * 0.04
            objects = (
                ModelObject(
                    label="person",
                    bbox=NormalizedBBox(
                        0.18 + shift,
                        0.12,
                        0.72 + shift,
                        0.9,
                    ),
                    confidence=0.91,
                ),
            )
        classification = GroundedClassification(
            detected=detected,
            confidence=0.92 if detected else 0.96,
            description=(
                "Dry-run event and object are visible."
                if detected
                else "Dry-run event is not visible."
            ),
            objects=objects,
        )
        return GroundingResponse(
            window=window,
            classification=classification,
            raw_text=json.dumps(
                {
                    "detected": classification.detected,
                    "confidence": classification.confidence,
                    "description": classification.description,
                }
            ),
            latency_seconds=time.monotonic() - started,
            response_id=f"dry-run-{window.index}",
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
