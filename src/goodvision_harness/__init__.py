"""Live perception harness: wall-time streaming into windowed Qwen inference."""

from .contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)

__all__ = [
    "AnalysisResult",
    "EventJudgment",
    "FrameReference",
    "GroundedObject",
    "NormalizedBBox",
]
