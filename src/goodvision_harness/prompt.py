"""Build the multimodal prompt sent for one sliding video window."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from .sliding_window import FrameWindow


@dataclass(frozen=True, slots=True)
class BinaryClassification:
    """Validated binary result returned by the live vision model."""

    detected: bool
    confidence: float
    description: str


def parse_binary_classification(text: str) -> BinaryClassification:
    """Parse a strict result while tolerating an accidental Markdown fence."""

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

    if not isinstance(payload, dict):
        raise TypeError("binary classification must be a JSON object")
    detected = payload.get("detected")
    confidence = payload.get("confidence")
    description = payload.get("description")
    if not isinstance(detected, bool):
        raise TypeError("binary classification detected must be true or false")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("binary classification confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("binary classification confidence must be between 0 and 1")
    if not isinstance(description, str) or not description.strip():
        raise TypeError("binary classification description must be non-empty text")
    return BinaryClassification(
        detected=detected,
        confidence=confidence,
        description=description.strip(),
    )


def build_window_messages(
    window: FrameWindow,
    *,
    event_label: str,
    system_prompt: str,
    prompt: str,
) -> list[dict[str, Any]]:
    """Return OpenRouter chat messages with text followed by ordered JPEGs.

    The configurable prompt may contain the literal placeholders ``{event}`` and
    ``{window_label}``. JPEG frames are embedded as base64 data URLs so local
    videos never need to be uploaded to a separate media host.
    """

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
