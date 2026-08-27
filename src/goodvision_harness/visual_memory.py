"""Non-blocking handoff contract for optional visual-memory persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import json_mapping_as_dict, readonly_json_mapping


@dataclass(frozen=True, slots=True)
class VisualMemoryRecord:
    """One JSON observation and its optional, exactly correlated evidence JPEG."""

    observation: dict[str, Any]
    evidence_jpeg: bytes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation, dict):
            raise TypeError("observation must be a dict")
        observation = readonly_json_mapping(
            self.observation,
            name="observation",
        )
        object.__setattr__(
            self,
            "observation",
            json_mapping_as_dict(observation),
        )
        if self.evidence_jpeg is not None and not isinstance(
            self.evidence_jpeg, bytes
        ):
            raise TypeError("evidence_jpeg must be bytes or None")


class VisualMemorySink(Protocol):
    """Best-effort sink whose submit method must never block the live pipeline."""

    def submit(self, record: VisualMemoryRecord) -> bool:
        """Enqueue a record immediately, returning whether it was accepted."""

        ...


class NullVisualMemorySink:
    """No-op sink used when visual memory is disabled."""

    def submit(self, record: VisualMemoryRecord) -> bool:
        del record
        return True
