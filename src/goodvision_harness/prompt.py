"""Build the multimodal prompt sent for one sliding video window."""

from __future__ import annotations

import base64
import json
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from .reference_memory import (
    FACE_RECOGNITION_PURPOSE,
    MemoryContext,
    UNKNOWN_FACE_EVENT_LABEL,
)
from .sliding_window import FrameWindow


@dataclass(frozen=True, slots=True)
class BinaryClassification:
    """Validated binary result returned by the live vision model."""

    detected: bool
    confidence: float
    description: str
    matched_memory_ids: tuple[str, ...] = ()


def parse_binary_classification(
    text: str,
    *,
    allowed_memory_ids: Collection[str] | None = None,
) -> BinaryClassification:
    """Parse a strict result and validate any model-reported memory matches."""

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

    raw_memory_ids = payload.get("matched_memory_ids", [])
    if not isinstance(raw_memory_ids, list):
        raise TypeError("binary classification matched_memory_ids must be a list")
    matched_memory_ids: list[str] = []
    seen_memory_ids: set[str] = set()
    for memory_id in raw_memory_ids:
        if not isinstance(memory_id, str) or not memory_id:
            raise TypeError(
                "binary classification matched_memory_ids must contain non-empty text"
            )
        if memory_id in seen_memory_ids:
            raise ValueError(
                "binary classification matched_memory_ids must not contain duplicates"
            )
        seen_memory_ids.add(memory_id)
        matched_memory_ids.append(memory_id)

    if allowed_memory_ids is None:
        if matched_memory_ids:
            raise ValueError("model reported memory matches without a memory context")
    else:
        if "matched_memory_ids" not in payload:
            raise ValueError("memory-aware result omitted matched_memory_ids")
        allowed = set(allowed_memory_ids)
        unknown_ids = [
            memory_id for memory_id in matched_memory_ids if memory_id not in allowed
        ]
        if unknown_ids:
            raise ValueError(
                "model reported unknown memory IDs: " + ", ".join(unknown_ids)
            )
    return BinaryClassification(
        detected=detected,
        confidence=confidence,
        description=description.strip(),
        matched_memory_ids=tuple(matched_memory_ids),
    )


def build_window_messages(
    window: FrameWindow,
    *,
    event_label: str,
    system_prompt: str,
    prompt: str,
    memory_context: MemoryContext | None = None,
) -> list[dict[str, Any]]:
    """Return OpenAI-compatible chat messages with optional memory and live JPEGs.

    The configurable prompt may contain the literal placeholders ``{event}`` and
    ``{window_label}``. JPEG frames are embedded as base64 data URLs so local
    videos never need to be uploaded to a separate media host.
    """

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
            "For this memory-aware request, respond with only this JSON object; "
            "this contract supersedes any earlier output example:\n"
            '{"detected": true or false, "confidence": 0.0-1.0, '
            '"matched_memory_ids": ["supplied MEMORY_ID"], '
            '"description": "one short sentence of visible evidence"}'
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
