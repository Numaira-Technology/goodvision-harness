"""Render grounded event results on source frames for the demo stream."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2

from .privacy import redact_public_text


def draw_overlay(
    bgr: Any,
    *,
    event_label: str,
    detected: bool,
    confidence: float,
    description: str,
    objects: Sequence[Any],
    stale: bool = False,
    analysis_kind: str = "event",
) -> Any:
    """Return a copy of ``bgr`` with a compact verdict and normalized boxes."""

    rendered = bgr.copy()
    height, width = rendered.shape[:2]
    if analysis_kind == "face_recognition":
        return _draw_face_overlay(
            rendered,
            detected=detected,
            confidence=confidence,
            description=description,
            objects=objects,
            stale=stale,
        )

    positive = bool(detected) and not stale
    color = (40, 40, 220) if positive else (55, 180, 70)
    if stale:
        color = (90, 150, 220)

    for item in objects if positive else ():
        bbox = getattr(item, "bbox", None)
        if bbox is None:
            continue
        try:
            x1, y1, x2, y2 = bbox.to_pixels(width, height)
        except (AttributeError, TypeError, ValueError):
            continue
        cv2.rectangle(rendered, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), color, 2)
        label = redact_public_text(getattr(item, "label", "object"))
        item_confidence = getattr(item, "confidence", None)
        if item_confidence is not None:
            label += f" {float(item_confidence):.2f}"
        _text_with_background(rendered, label[:80], (x1, max(18, y1)), color)

    positive_label = (
        "UNKNOWN"
        if analysis_kind == "face_reference_match"
        else redact_public_text(event_label).upper()
    )
    state = "STALE" if stale else (positive_label if detected else "CLEAR")
    title = f"{state}  {float(confidence):.2f}"
    _text_with_background(rendered, title[:100], (12, 28), color, scale=0.7)
    if description:
        _text_with_background(
            rendered,
            _ascii_safe(redact_public_text(description))[:110],
            (12, 55),
            (45, 45, 45),
            scale=0.48,
        )
    return rendered


def _draw_face_overlay(
    rendered: Any,
    *,
    detected: bool,
    confidence: float,
    description: str,
    objects: Sequence[Any],
    stale: bool,
) -> Any:
    height, width = rendered.shape[:2]
    colors = {
        "known": (55, 180, 70),
        "unknown": (40, 40, 220),
        "unusable": (105, 105, 105),
    }
    statuses: list[str] = []
    for item in objects:
        attributes = getattr(item, "attributes", {})
        if not isinstance(attributes, Mapping):
            attributes = {}
        status = str(attributes.get("status", "unusable"))
        if status not in colors:
            status = "unusable"
        statuses.append(status)
        bbox = getattr(item, "bbox", None)
        if bbox is None:
            continue
        try:
            x1, y1, x2, y2 = bbox.to_pixels(width, height)
        except (AttributeError, TypeError, ValueError):
            continue
        color = colors[status]
        cv2.rectangle(
            rendered,
            (x1, y1),
            (max(x1, x2 - 1), max(y1, y2 - 1)),
            color,
            2,
        )
        if status == "known":
            label = redact_public_text(attributes.get("name") or "known")
        elif status == "unknown":
            label = "UNKNOWN"
        else:
            label = "unusable"
        similarity = attributes.get("similarity")
        if isinstance(similarity, (int, float)) and not isinstance(similarity, bool):
            label += f" {float(similarity):.2f}"
        _text_with_background(rendered, label[:80], (x1, max(18, y1)), color)

    if stale:
        state, color = "STALE", (90, 150, 220)
    elif detected or "unknown" in statuses:
        state, color = "UNKNOWN", colors["unknown"]
    elif not statuses:
        state, color = "NO FACE", colors["unusable"]
    elif all(status == "unusable" for status in statuses):
        state, color = "NO USABLE FACE", colors["unusable"]
    else:
        state, color = "CLEAR", colors["known"]
    _text_with_background(
        rendered,
        f"{state}  {float(confidence):.2f}"[:100],
        (12, 28),
        color,
        scale=0.7,
    )
    if description:
        _text_with_background(
            rendered,
            _ascii_safe(redact_public_text(description))[:110],
            (12, 55),
            (45, 45, 45),
            scale=0.48,
        )
    return rendered


def _text_with_background(
    image: Any,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.55,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(0, min(x, max(0, image.shape[1] - text_width - 8)))
    y = max(text_height + 6, min(y, image.shape[0] - baseline - 2))
    cv2.rectangle(
        image,
        (x, y - text_height - 6),
        (x + text_width + 8, y + baseline + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y - 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _ascii_safe(value: str) -> str:
    return value.encode("ascii", "replace").decode("ascii")
