"""Canonical JSON contract for live analysis results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .privacy import public_safe_value

ANALYSIS_RESULT_SCHEMA_VERSION = "2"
BBOX_COORDINATE_SYSTEM = "source_normalized_xyxy"
PUBLIC_MODEL_NAME = "numaira-beta-fast"


def _freeze_json(value: Any, name: str) -> Any:
    """Return an immutable representation of a JSON-compatible value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings")
            frozen[key] = _freeze_json(item, f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{name} must contain only JSON-compatible values")


def readonly_json_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str = "attributes",
) -> Mapping[str, Any]:
    """Validate and deeply freeze a JSON object for use on result records."""

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen = _freeze_json(value, name)
    assert isinstance(frozen, Mapping)
    return frozen


def json_mapping_as_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mutable JSON-compatible copy of a frozen mapping."""

    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: thaw(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return [thaw(child) for child in item]
        return item

    return {key: thaw(item) for key, item in value.items()}


def _required_text(value: str, name: str) -> str:
    rendered = str(value).strip()
    if not rendered:
        raise ValueError(f"{name} must not be empty")
    return rendered


def _confidence(value: float, name: str) -> float:
    rendered = float(value)
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return rendered


@dataclass(frozen=True, slots=True)
class NormalizedBBox:
    """Top-left-origin xyxy coordinates relative to the source frame."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in (self.x1, self.y1, self.x2, self.y2))
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("bbox coordinates must be finite values between 0 and 1")
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError("bbox must satisfy x2>x1 and y2>y1")

    def as_dict(self) -> dict[str, object]:
        return {
            "coordinate_system": BBOX_COORDINATE_SYSTEM,
            "x1": round(float(self.x1), 6),
            "y1": round(float(self.y1), 6),
            "x2": round(float(self.x2), 6),
            "y2": round(float(self.y2), 6),
        }

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        if width < 1 or height < 1:
            raise ValueError("frame dimensions must be positive")
        x1 = max(0, min(width - 1, round(self.x1 * width)))
        y1 = max(0, min(height - 1, round(self.y1 * height)))
        x2 = max(x1 + 1, min(width, round(self.x2 * width)))
        y2 = max(y1 + 1, min(height, round(self.y2 * height)))
        return x1, y1, x2, y2


@dataclass(frozen=True, slots=True)
class GroundedObject:
    object_id: str
    label: str
    bbox: NormalizedBBox
    confidence: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        _required_text(self.object_id, "object_id")
        _required_text(self.label, "object label")
        if self.confidence is not None:
            _confidence(self.confidence, "object confidence")
        object.__setattr__(
            self,
            "attributes",
            readonly_json_mapping(self.attributes, name="object attributes"),
        )

    def as_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "object_id": self.object_id,
            "label": self.label,
            "bbox": self.bbox.as_dict(),
            "attributes": json_mapping_as_dict(self.attributes),
        }
        if self.confidence is not None:
            record["confidence"] = round(float(self.confidence), 3)
        return record


@dataclass(frozen=True, slots=True)
class FrameReference:
    source_frame_index: int
    timestamp_seconds: float
    observed_at: str
    width: int
    height: int
    clock_source: str = "ingest_estimated"

    def __post_init__(self) -> None:
        if self.source_frame_index < 0:
            raise ValueError("source_frame_index must be non-negative")
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        _required_text(self.observed_at, "observed_at")
        if self.width < 1 or self.height < 1:
            raise ValueError("frame dimensions must be positive")
        _required_text(self.clock_source, "clock_source")

    def as_dict(self) -> dict[str, object]:
        return {
            "source_frame_index": self.source_frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "observed_at": self.observed_at,
            "width": self.width,
            "height": self.height,
            "clock_source": self.clock_source,
        }


@dataclass(frozen=True, slots=True)
class EventJudgment:
    event_label: str
    detected: bool
    confidence: float
    description: str
    object_ids: tuple[str, ...] = ()
    matched_memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.event_label, "event_label")
        if not isinstance(self.detected, bool):
            raise TypeError("detected must be true or false")
        _confidence(self.confidence, "judgment confidence")
        _required_text(self.description, "description")
        if any(not str(value).strip() for value in self.object_ids):
            raise ValueError("object_ids must not contain empty values")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("object_ids must be unique")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.matched_memory_ids
        ):
            raise ValueError("matched_memory_ids must contain non-empty text")
        if len(set(self.matched_memory_ids)) != len(self.matched_memory_ids):
            raise ValueError("matched_memory_ids must be unique")
        if self.detected and not self.object_ids:
            # 独立Ashley补齐版：已声明的车辆场景事件没有人物框关联。
            # 普通人物/物体阳性仍要求真实对象，不借场景例外放宽其关联。
            from .event_session import requires_subject
            # 地面垃圾沿主线显式scene extent投影，车辆保留空对象发布。
            if self.event_label == "ground_litter" or requires_subject(self.event_label):
                raise ValueError("detected judgments must reference at least one object")

    def as_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "event_label": self.event_label,
            "detected": self.detected,
            "confidence": round(self.confidence, 3),
            "description": self.description,
            "object_ids": list(self.object_ids),
        }
        if self.matched_memory_ids:
            record["matched_memory_ids"] = list(self.matched_memory_ids)
        from .event_session import requires_subject
        if not requires_subject(self.event_label):
            record["grounding_scope"] = "scene"
        return record


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    message_id: str
    source_id: str
    stream_session_id: str
    sequence: int
    emitted_at: str
    frame: FrameReference
    judgments: tuple[EventJudgment, ...]
    objects: tuple[GroundedObject, ...]
    model_name: str
    model_latency_seconds: float
    analysis_kind: str = "event"
    missing_event_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in {
            "message_id": self.message_id,
            "source_id": self.source_id,
            "stream_session_id": self.stream_session_id,
            "emitted_at": self.emitted_at,
            "model_name": self.model_name,
            "analysis_kind": self.analysis_kind,
        }.items():
            _required_text(value, name)
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if not self.judgments:
            raise ValueError("judgments must not be empty")
        if (
            not math.isfinite(self.model_latency_seconds)
            or self.model_latency_seconds < 0
        ):
            raise ValueError("model_latency_seconds must be finite and non-negative")
        object_ids = {item.object_id for item in self.objects}
        if len(object_ids) != len(self.objects):
            raise ValueError("object_id values must be unique")
        referenced = {value for item in self.judgments for value in item.object_ids}
        if not referenced.issubset(object_ids):
            raise ValueError("judgments reference unknown objects")
        labels = tuple(self.missing_event_labels)
        for label in labels:
            _required_text(label, "missing event label")
        if len(set(labels)) != len(labels) or set(labels).intersection(
            judgment.event_label for judgment in self.judgments
        ):
            raise ValueError("missing event labels must be unique and unevaluated")
        object.__setattr__(self, "missing_event_labels", labels)

    @property
    def description(self) -> str:
        """Summarize detected events across this frame's analysis result."""

        summary = "; ".join(
            f"{judgment.event_label}: {judgment.description}"
            for judgment in self.judgments
            if judgment.detected
        )
        if self.missing_event_labels:
            incomplete = (
                "分析未完成。部分能力暂无结果。 / "
                "Analysis incomplete; some capabilities have no result."
            )
            return f"{summary}; {incomplete}" if summary else incomplete
        if not summary and self.analysis_kind == "scene_state":
            return "暂无已确认的事件。 / No confirmed events."
        return summary or "无异常 / No anomalies detected"

    def as_dict(self) -> dict[str, Any]:
        record = {
            "type": "analysis_result",
            "schema_version": ANALYSIS_RESULT_SCHEMA_VERSION,
            "analysis_kind": self.analysis_kind,
            "message_id": self.message_id,
            "observation_id": self.message_id,
            "source_id": self.source_id,
            "stream_session_id": self.stream_session_id,
            "sequence": self.sequence,
            "emitted_at": self.emitted_at,
            "frame": self.frame.as_dict(),
            "description": self.description,
            "analysis_complete": not self.missing_event_labels,
            "missing_event_labels": list(self.missing_event_labels),
            "judgments": [item.as_dict() for item in self.judgments],
            "objects": [item.as_dict() for item in self.objects],
            "model": {
                # Keep the provider route private while preserving the stable
                # customer-facing product name in every serialized result.
                "name": PUBLIC_MODEL_NAME,
                "latency_seconds": round(self.model_latency_seconds, 3),
            },
        }
        safe = public_safe_value(record)
        assert isinstance(safe, dict)
        return safe
