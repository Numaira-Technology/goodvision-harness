"""Stable, transport-neutral output from live visual inference."""

from __future__ import annotations

from dataclasses import dataclass

from .privacy import public_safe_value

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

        record = {
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
        safe = public_safe_value(record)
        assert isinstance(safe, dict)
        return safe


@dataclass(frozen=True, slots=True)
class PreliminaryVisionResult:
    """Early model verdict emitted before optional descriptive fields finish."""

    event_label: str
    detected: bool
    confidence: float
    window_index: int
    source_frame_index: int
    window_start_seconds: float
    window_end_seconds: float
    latency_seconds: float

    def as_dict(self) -> dict[str, object]:
        """Render a bounded provisional record that cannot be mistaken for final."""

        record = {
            "type": "preliminary_vision_result",
            "schema_version": VISION_RESULT_SCHEMA_VERSION,
            "event_label": self.event_label,
            "detected": self.detected,
            "confidence": round(self.confidence, 3),
            "window_index": self.window_index,
            "source_frame_index": self.source_frame_index,
            "window_start_seconds": round(self.window_start_seconds, 3),
            "window_end_seconds": round(self.window_end_seconds, 3),
            "latency_seconds": round(self.latency_seconds, 3),
        }
        safe = public_safe_value(record)
        assert isinstance(safe, dict)
        return safe
