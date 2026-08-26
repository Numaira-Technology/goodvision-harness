"""Stable, transport-neutral output from live visual inference."""

from __future__ import annotations

from dataclasses import dataclass

VISION_RESULT_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class VisionResult:
    """One newest-window visual decision, independent of robot commands."""

    event_label: str
    detected: bool
    confidence: float
    description: str
    window_index: int
    window_start_seconds: float
    window_end_seconds: float

    def as_dict(self) -> dict[str, object]:
        """Render the canonical JSON-compatible record."""

        return {
            "type": "vision_result",
            "schema_version": VISION_RESULT_SCHEMA_VERSION,
            "event_label": self.event_label,
            "detected": self.detected,
            "confidence": round(self.confidence, 3),
            "description": self.description,
            "window_index": self.window_index,
            "window_start_seconds": round(self.window_start_seconds, 3),
            "window_end_seconds": round(self.window_end_seconds, 3),
        }
