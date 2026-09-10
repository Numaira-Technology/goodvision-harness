from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from goodvision_harness import overlay as overlay_module
from goodvision_harness.contracts import NormalizedBBox


@pytest.mark.parametrize(
    ("person_id", "name", "expected_label"),
    [
        ("employee-007", "Alice", "ID: employee-007 | Alice 0.99"),
        ("employee-007", "", "ID: employee-007 0.99"),
        ("employee-007", "employee-007", "ID: employee-007 0.99"),
        ("insightface-user", "Alice", "ID: insightface-user | Alice 0.99"),
    ],
)
def test_known_face_shows_enrollment_id_and_green_box_without_stranger_alert(
    monkeypatch: pytest.MonkeyPatch,
    person_id: str,
    name: str,
    expected_label: str,
) -> None:
    labels: list[str] = []
    original_put_text = overlay_module.cv2.putText

    def capture(image: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        labels.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(overlay_module.cv2, "putText", capture)
    source = np.zeros((320, 640, 3), dtype=np.uint8)
    face = SimpleNamespace(
        bbox=NormalizedBBox(0.2, 0.4, 0.6, 0.9),
        attributes={
            "status": "known",
            "person_id": person_id,
            "name": name,
            "similarity": 0.99,
        },
    )

    rendered = overlay_module.draw_overlay(
        source,
        event_label="unknown_person",
        detected=False,
        confidence=0.99,
        description="",
        objects=(face,),
        analysis_kind="face_recognition",
    )

    assert labels == [expected_label, "CLEAR  0.99"]
    x1, y1, _x2, y2 = face.bbox.to_pixels(640, 320)
    assert tuple(rendered[(y1 + y2) // 2, x1]) == (55, 180, 70)
    assert rendered is not source
    assert not source.any()


def test_face_boxes_keep_mixed_status_colors_and_preserve_existing_event_hud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = overlay_module.draw_overlay(
        np.zeros((320, 640, 3), dtype=np.uint8),
        event_label="fall",
        detected=True,
        confidence=0.9,
        description="Fall detected.",
        objects=(),
    )
    before = source.copy()
    labels: list[str] = []
    original_put_text = overlay_module.cv2.putText

    def capture(image: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        labels.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(overlay_module.cv2, "putText", capture)
    faces = tuple(
        SimpleNamespace(
            bbox=NormalizedBBox(left, 0.4, left + 0.15, 0.9),
            attributes={
                "status": status,
                "person_id": "employee-007" if status == "known" else "old-id",
                "name": "Alice" if status == "known" else "Old identity",
            },
        )
        for left, status in ((0.1, "known"), (0.4, "unknown"), (0.7, "unusable"))
    )

    rendered = overlay_module.draw_face_boxes(source, faces)

    assert labels == ["ID: employee-007 | Alice", "UNKNOWN", "unusable"]
    for face, color in zip(
        faces, ((55, 180, 70), (40, 40, 220), (105, 105, 105)), strict=True
    ):
        x1, y1, _x2, y2 = face.bbox.to_pixels(640, 320)
        assert tuple(rendered[(y1 + y2) // 2, x1]) == color
    assert np.array_equal(rendered[:80], before[:80])
    assert rendered is not source
    assert np.array_equal(source, before)
