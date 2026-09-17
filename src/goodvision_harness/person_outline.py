"""Fast person outline strokes for live overlay.

Contour extraction is local and bounded: a small crop, no extra model call.
If the silhouette is unusable, callers draw a stroked rectangle instead.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .contracts import NormalizedBBox

__all__ = [
    "draw_person_stroke",
    "estimate_body_from_face",
    "extract_person_contour",
]


def extract_person_contour(
    image: Any, bbox: tuple[int, int, int, int]
) -> np.ndarray | None:
    """Return a tight silhouette in image coordinates, or None."""

    x1, y1, x2, y2 = (int(value) for value in bbox)
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None
    try:
        height, width = image.shape[:2]
    except (AttributeError, TypeError, ValueError):
        return None
    x1 = max(0, min(x1, width - 2))
    y1 = max(0, min(y1, height - 2))
    x2 = max(x1 + 2, min(x2, width))
    y2 = max(y1 + 2, min(y2, height))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_h, crop_w = crop.shape[:2]
    scale = 1.0
    work = crop
    longest = max(crop_h, crop_w)
    if longest > 96:
        scale = 96.0 / longest
        work = cv2.resize(
            crop,
            (max(8, int(round(crop_w * scale))), max(8, int(round(crop_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    border = np.concatenate(
        (gray[0], gray[-1], gray[:, 0], gray[:, -1])
    )
    background = float(np.median(border))
    mask = (np.abs(gray.astype(np.int16) - background) > 18).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 0.12 * mask.shape[0] * mask.shape[1]:
        return None
    points = contour.reshape(-1, 2).astype(np.float32)
    if scale != 1.0:
        points /= scale
    points[:, 0] += x1
    points[:, 1] += y1
    if len(points) < 4:
        return None
    return np.round(points).astype(np.int32)


def draw_person_stroke(
    image: Any,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    contour: np.ndarray | None = None,
) -> None:
    """Draw a silhouette stroke, or a full rectangle if no contour is available."""

    if contour is not None and len(contour) >= 4:
        pts = contour.reshape(-1, 1, 2)
        cv2.polylines(image, [pts], True, (12, 14, 18), 5, cv2.LINE_AA)
        cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)
        return
    x1, y1, x2, y2 = bbox
    x2 = max(x1, x2 - 1)
    y2 = max(y1, y2 - 1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (12, 14, 18), 5, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def estimate_body_from_face(face: NormalizedBBox) -> NormalizedBBox:
    """Expand a face box into an approximate standing-person box."""

    width = face.x2 - face.x1
    height = face.y2 - face.y1
    cx = (face.x1 + face.x2) / 2.0
    body_width = min(0.95, max(width * 2.2, 0.12))
    body_height = min(0.98, max(height * 6.2, 0.35))
    x1 = max(0.0, min(1.0 - body_width, cx - body_width / 2.0))
    y1 = max(0.0, min(1.0 - body_height, face.y1 - height * 0.15))
    return NormalizedBBox(x1, y1, min(1.0, x1 + body_width), min(1.0, y1 + body_height))
