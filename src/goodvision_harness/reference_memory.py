"""Load a compact, immutable multimodal reference context for Qwen inference."""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .video_to_stream import encode_jpeg


FACE_RECOGNITION_PURPOSE = "face_recognition"
DEFAULT_MAX_MEMORY_REFERENCES = 8
DEFAULT_MEMORY_IMAGE_SIDE = 384
DEFAULT_MEMORY_JPEG_QUALITY = 85
MAX_MEMORY_SOURCE_IMAGE_BYTES = 20 * 1024 * 1024
_MEMORY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")

_FACE_RECOGNITION_INSTRUCTION = (
    "The reference images above are known-face memory and are context only; they "
    "must never trigger a detection by themselves. Compare faces visible in the "
    "LIVE CAMERA INPUT against those references. Set detected=true only when at "
    "least one live face is the same person as a supplied MEMORY_ID. A similar-looking "
    "or uncertain face is not a match. Set detected=false when no face is visible, "
    "every visible face is unknown, or identity is uncertain. matched_memory_ids must "
    "contain every matched supplied ID and no other value."
)


@dataclass(frozen=True, slots=True)
class MemoryReference:
    """One prompt-ready reference image with an opaque model-visible ID."""

    memory_id: str
    image_data_url: str = field(repr=False)
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """A fixed memory prefix shared by every inference call in one session."""

    purpose: str
    references: tuple[MemoryReference, ...]
    instruction: str
    newest_frame_only: bool = True

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(reference.memory_id for reference in self.references)

    @property
    def prompt_prefix(self) -> list[dict[str, Any]]:
        """Return the stable multimodal prefix placed before live camera input."""

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"BEGIN MEMORY CONTEXT\nPurpose: {self.purpose}\n"
                    "The following images are references, not live observations."
                ),
            }
        ]
        for reference in self.references:
            content.extend(
                (
                    {
                        "type": "text",
                        "text": f"MEMORY_ID: {reference.memory_id}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": reference.image_data_url},
                    },
                )
            )
        content.append({"type": "text", "text": "END MEMORY CONTEXT"})
        return content


def load_memory_context(
    value: Mapping[str, Any],
    *,
    config_directory: Path,
) -> MemoryContext | None:
    """Validate and load one configured context, or return ``None`` when disabled."""

    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("memory.enabled must be true or false")
    if not enabled:
        return None

    purpose = str(value.get("purpose", FACE_RECOGNITION_PURPOSE)).strip()
    if not purpose:
        raise ValueError("memory.purpose must not be empty")

    instruction_value = value.get("instruction")
    if instruction_value is None:
        if purpose != FACE_RECOGNITION_PURPOSE:
            raise ValueError(
                "memory.instruction is required for non-face-recognition purposes"
            )
        instruction = _FACE_RECOGNITION_INSTRUCTION
    elif not isinstance(instruction_value, str) or not instruction_value.strip():
        raise TypeError("memory.instruction must be non-empty text")
    else:
        instruction = instruction_value.strip()

    newest_frame_only = value.get(
        "newest_frame_only", purpose == FACE_RECOGNITION_PURPOSE
    )
    if not isinstance(newest_frame_only, bool):
        raise TypeError("memory.newest_frame_only must be true or false")
    max_references = _positive_int(
        value.get("max_references", DEFAULT_MAX_MEMORY_REFERENCES),
        "memory.max_references",
    )
    max_image_side = _positive_int(
        value.get("max_image_side", DEFAULT_MEMORY_IMAGE_SIDE),
        "memory.max_image_side",
    )
    jpeg_quality = _bounded_int(
        value.get("jpeg_quality", DEFAULT_MEMORY_JPEG_QUALITY),
        "memory.jpeg_quality",
        minimum=1,
        maximum=100,
    )

    raw_references = value.get("references") or []
    if not isinstance(raw_references, Sequence) or isinstance(
        raw_references, (str, bytes, bytearray)
    ):
        raise TypeError("memory.references must be a list")
    if not raw_references:
        raise ValueError("memory.references must not be empty when memory is enabled")
    if len(raw_references) > max_references:
        raise ValueError(
            "memory.references exceeds memory.max_references "
            f"({len(raw_references)} > {max_references})"
        )

    references: list[MemoryReference] = []
    seen_ids: set[str] = set()
    for index, raw_reference in enumerate(raw_references):
        label = f"memory.references[{index}]"
        if not isinstance(raw_reference, Mapping):
            raise TypeError(f"{label} must be a mapping")
        memory_id = str(raw_reference.get("id", "")).strip()
        if _MEMORY_ID.fullmatch(memory_id) is None:
            raise ValueError(
                f"{label}.id must be 1-64 ASCII letters, digits, '.', '_', ':', or '-'"
            )
        if memory_id in seen_ids:
            raise ValueError(f"duplicate memory reference id: {memory_id}")
        seen_ids.add(memory_id)

        configured_path = raw_reference.get("image")
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise TypeError(f"{label}.image must be a non-empty path")
        image_path = Path(configured_path).expanduser()
        if not image_path.is_absolute():
            image_path = config_directory / image_path
        image_path = image_path.resolve()
        references.append(
            load_memory_reference(
                memory_id,
                image_path,
                max_image_side=max_image_side,
                jpeg_quality=jpeg_quality,
            )
        )

    if purpose == FACE_RECOGNITION_PURPOSE and instruction_value is None:
        return build_face_memory_context(
            references,
            newest_frame_only=newest_frame_only,
        )
    return MemoryContext(
        purpose=purpose,
        references=tuple(references),
        instruction=instruction,
        newest_frame_only=newest_frame_only,
    )


def build_face_memory_context(
    references: Sequence[MemoryReference],
    *,
    newest_frame_only: bool = True,
) -> MemoryContext:
    """Build the standard face-comparison context from prepared references."""

    prepared = tuple(references)
    if not prepared:
        raise ValueError("face memory requires at least one reference")
    memory_ids = [reference.memory_id for reference in prepared]
    if len(set(memory_ids)) != len(memory_ids):
        raise ValueError("face memory reference IDs must be unique")
    return MemoryContext(
        purpose=FACE_RECOGNITION_PURPOSE,
        references=prepared,
        instruction=_FACE_RECOGNITION_INSTRUCTION,
        newest_frame_only=newest_frame_only,
    )


def load_memory_reference(
    memory_id: str,
    image_path: Path,
    *,
    max_image_side: int = DEFAULT_MEMORY_IMAGE_SIDE,
    jpeg_quality: int = DEFAULT_MEMORY_JPEG_QUALITY,
) -> MemoryReference:
    if not image_path.is_file():
        raise FileNotFoundError(f"memory reference image does not exist: {image_path}")
    source_size = image_path.stat().st_size
    if source_size < 1:
        raise ValueError(f"memory reference image is empty: {image_path}")
    if source_size > MAX_MEMORY_SOURCE_IMAGE_BYTES:
        raise ValueError(
            f"memory reference image exceeds {MAX_MEMORY_SOURCE_IMAGE_BYTES} bytes: "
            f"{image_path}"
        )
    image_bytes = image_path.read_bytes()
    if len(image_bytes) > MAX_MEMORY_SOURCE_IMAGE_BYTES:
        raise ValueError(
            f"memory reference image exceeds {MAX_MEMORY_SOURCE_IMAGE_BYTES} bytes: "
            f"{image_path}"
        )
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"memory reference is not a readable image: {image_path}")
    jpeg, width, height = encode_jpeg(
        decoded,
        max_image_side=max_image_side,
        quality=jpeg_quality,
    )
    return MemoryReference(
        memory_id=memory_id,
        image_data_url=(
            "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        ),
        width=width,
        height=height,
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value
