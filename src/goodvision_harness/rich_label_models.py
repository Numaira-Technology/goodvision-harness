"""Console-selectable rich-label models and how each maps onto a backend.

The dashboard dropdown and the session-start apply-logic share this one registry
so they never drift. SAM keeps the continuous video path (sam31_stream); the
single-frame detectors (YOLO / RF-DETR) all speak the existing paddle-style
`/v1/labels` contract via RichLabelHTTPClient, so selecting one only changes the
backend + endpoint (the detector worker's DETECTOR_MODEL picks the actual model).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RichLabelModel:
    key: str            # stable id used in settings + dashboard <option>
    label: str          # human label shown in the console
    backend: str        # harness rich_labeling backend
    detector_model: str | None  # worker DETECTOR_MODEL (None for SAM)
    note: str


# Ordered for the dropdown. Picks come from the 2026/09/29 benchmark
# (0.5s-tier + highest 两轮描边率). The console `key`/`label`/`note` are kept
# brand-neutral on purpose: ui_info() must not leak implementation names (the
# harness redacts "sam"/"yolo"/... and a privacy guard test enforces it), so the
# real model names live only in `detector_model`, which is sent to the worker via
# the DETECTOR_MODEL_HEADER and never surfaced in the UI.
RICH_LABEL_MODELS: tuple[RichLabelModel, ...] = (
    RichLabelModel("contour", "描边模式(最准/出描边)", "sam31_stream", None,
                   "person outlines; slowest"),
    RichLabelModel("det-balanced", "均衡检测(0.5s)", "paddle", "rfdetr-l",
                   "balanced detector; boxes"),
    RichLabelModel("det-fast", "快速检测(最快)", "paddle", "yolo26-s",
                   "fastest detector; boxes"),
    RichLabelModel("det-accurate", "高精度检测(最准)", "paddle", "yolov8-m",
                   "most accurate detector; boxes"),
    RichLabelModel("det-segment", "单帧分割(可选)", "paddle", "sam3-onnx",
                   "optional single-image person segmentation; speed varies"),
)

_BY_KEY = {m.key: m for m in RICH_LABEL_MODELS}
DEFAULT_RICH_LABEL_MODEL = "contour"

# Per-request header that tells a multi-model detector worker which model to run
# for this call, so the console selection drives the worker live (the worker
# lazy-loads and caches each model). Non-detector workers ignore it.
DETECTOR_MODEL_HEADER = "X-GoodVision-Detector-Model"


def rich_label_model_keys() -> tuple[str, ...]:
    return tuple(_BY_KEY)


def validate_rich_label_model(value: object) -> str:
    if not isinstance(value, str) or value not in _BY_KEY:
        raise ValueError(
            f"rich_label_model must be one of {sorted(_BY_KEY)}"
        )
    return value


def resolve_rich_label_model(key: str) -> RichLabelModel:
    """The backend/detector mapping for a selected model key."""
    return _BY_KEY[validate_rich_label_model(key)]
