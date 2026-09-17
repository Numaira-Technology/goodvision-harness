from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.person_outline import (
    draw_person_stroke,
    estimate_body_from_face,
    extract_person_contour,
)
from goodvision_harness import overlay as overlay_module


def test_person_overlay_strokes_the_full_box_not_just_corners() -> None:
    source = np.full((220, 320, 3), 40, dtype=np.uint8)
    rendered = overlay_module.draw_rich_overlay(
        source,
        [
            SimpleNamespace(
                bbox=NormalizedBBox(0.2, 0.35, 0.75, 0.85),
                label="person",
                object_id="person-object",
                confidence=0.92,
                attributes={"track_id": "P0007"},
            )
        ],
    )
    x1, y1, x2, y2 = NormalizedBBox(0.2, 0.35, 0.75, 0.85).to_pixels(
        source.shape[1], source.shape[0]
    )
    assert not np.array_equal(rendered[y1, (x1 + x2) // 2], source[y1, (x1 + x2) // 2])
    assert not np.array_equal(rendered[(y1 + y2) // 2, x1], source[(y1 + y2) // 2, x1])


def test_non_person_overlay_keeps_corner_brackets() -> None:
    source = np.full((220, 320, 3), 40, dtype=np.uint8)
    rendered = overlay_module.draw_rich_overlay(
        source,
        [
            SimpleNamespace(
                bbox=NormalizedBBox(0.2, 0.35, 0.75, 0.85),
                label="bottle",
                object_id="bottle-object",
                confidence=0.92,
                attributes={"track_id": "O0007"},
            )
        ],
    )
    x1, y1, x2, y2 = NormalizedBBox(0.2, 0.35, 0.75, 0.85).to_pixels(
        source.shape[1], source.shape[0]
    )
    assert np.array_equal(rendered[y1, (x1 + x2) // 2], source[y1, (x1 + x2) // 2])
    assert np.array_equal(rendered[(y1 + y2) // 2, x1], source[(y1 + y2) // 2, x1])


def test_extract_person_contour_follows_blob_instead_of_the_full_box() -> None:
    image = np.full((120, 80, 3), 18, dtype=np.uint8)
    image[25:95, 28:52] = (210, 180, 90)
    contour = extract_person_contour(image, (20, 20, 60, 100))
    assert contour is not None
    xs, ys = contour[:, 0], contour[:, 1]
    assert xs.max() - xs.min() < 40
    assert ys.max() - ys.min() > 40


def test_missing_silhouette_falls_back_to_a_stroked_rectangle() -> None:
    image = np.full((80, 80, 3), 40, dtype=np.uint8)
    rendered = image.copy()
    draw_person_stroke(rendered, (10, 10, 70, 70), (40, 40, 220), contour=None)
    assert not np.array_equal(rendered[10, 40], image[10, 40])
    assert np.array_equal(rendered[40, 40], image[40, 40])


def test_estimate_body_from_face_extends_below_the_head() -> None:
    face = NormalizedBBox(0.45, 0.10, 0.55, 0.22)
    body = estimate_body_from_face(face)
    assert body.y1 <= face.y1
    assert body.y2 > face.y2 + 0.2
    assert body.x1 < face.x1
    assert body.x2 > face.x2
