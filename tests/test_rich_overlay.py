from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from goodvision_harness import overlay as overlay_module
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.privacy import contains_private_implementation_detail


def _tracked_object(
    track_id: object,
    *,
    label: str = "person",
    object_id: str = "person-object",
) -> SimpleNamespace:
    return SimpleNamespace(
        bbox=NormalizedBBox(0.2, 0.35, 0.75, 0.85),
        label=label,
        object_id=object_id,
        confidence=0.92,
        attributes={"track_id": track_id},
    )


def test_rich_overlay_draws_brackets_and_a_translucent_card_on_a_copy() -> None:
    source = np.full((220, 320, 3), 40, dtype=np.uint8)
    rendered = overlay_module.draw_rich_overlay(source, [_tracked_object(7)])

    assert np.all(source == 40)
    assert rendered is not source
    assert np.any(rendered != source)

    x1, y1, x2, y2 = NormalizedBBox(0.2, 0.35, 0.75, 0.85).to_pixels(
        source.shape[1], source.shape[0]
    )
    assert not np.array_equal(rendered[y1, x1], source[y1, x1])
    assert not np.array_equal(rendered[y2 - 1, x2 - 1], source[y2 - 1, x2 - 1])
    assert np.array_equal(rendered[y1, (x1 + x2) // 2], source[y1, (x1 + x2) // 2])
    assert np.array_equal(rendered[(y1 + y2) // 2, x1], source[(y1 + y2) // 2, x1])

    card_pixel = rendered[y1 - 38, x1 + 6]
    assert np.all(card_pixel < source[y1 - 38, x1 + 6])
    assert np.all(card_pixel > np.array([16, 20, 26], dtype=np.uint8))


def test_rich_overlay_color_is_stable_for_track_id() -> None:
    source = np.zeros((220, 320, 3), dtype=np.uint8)
    first = overlay_module.draw_rich_overlay(
        source,
        [_tracked_object("camera-a:17", label="person", object_id="old-id")],
    )
    same_track = overlay_module.draw_rich_overlay(
        source,
        [_tracked_object("camera-a:17", label="human", object_id="new-id")],
    )
    x1, y1, _x2, _y2 = NormalizedBBox(0.2, 0.35, 0.75, 0.85).to_pixels(
        source.shape[1], source.shape[0]
    )
    first_corner = first[y1 : y1 + 8, x1 : x1 + 8]
    same_corner = same_track[y1 : y1 + 8, x1 : x1 + 8]

    assert np.array_equal(first_corner, same_corner)
    assert any(
        not np.array_equal(
            first_corner,
            overlay_module.draw_rich_overlay(source, [_tracked_object(candidate)])[
                y1 : y1 + 8, x1 : x1 + 8
            ],
        )
        for candidate in range(1, 20)
    )


def test_rich_overlay_promotes_one_track_card_with_multiple_alerts(
    monkeypatch: Any,
) -> None:
    captured_text: list[str] = []
    captured_colors: list[tuple[int, int, int]] = []
    original_put_text = overlay_module.cv2.putText
    original_corner = overlay_module._draw_corner_bbox

    def capture_text(image: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        captured_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    def capture_corner(
        image: Any,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        captured_colors.append(color)
        original_corner(image, bbox, color)

    monkeypatch.setattr(overlay_module.cv2, "putText", capture_text)
    monkeypatch.setattr(overlay_module, "_draw_corner_bbox", capture_corner)
    image = np.zeros((260, 640, 3), dtype=np.uint8)
    person = SimpleNamespace(
        bbox=NormalizedBBox(0.2, 0.35, 0.75, 0.9),
        label="person",
        confidence=0.92,
        attributes={
            "track_id": "P0007",
            "estimated_age_range": "25-34",
            "estimated_gender": "male-presenting",
        },
    )
    alerts = {
        "P0007": (
            SimpleNamespace(event_label="fight", confidence=0.91),
            SimpleNamespace(event_label="fall", confidence=0.86),
            SimpleNamespace(event_label="stranger", confidence=0.875),
        )
    }

    overlay_module.draw_rich_overlay(
        image,
        [person],
        alerts_by_track_id=alerts,
    )

    assert captured_colors == [overlay_module._ALERT_COLOR]
    assert len(captured_text) == 3
    assert captured_text[0].startswith("P0007")
    assert captured_text[1] == "ALERT  FIGHT 91% / FALL 86% / STRANGER 88%"
    assert "AGE 25-34" in captured_text[2]
    assert "GENDER MALE" in captured_text[2]
    assert not contains_private_implementation_detail(" ".join(captured_text))


def test_unmatched_event_object_uses_rich_corner_card_style(
    monkeypatch: Any,
) -> None:
    captured_text: list[str] = []
    captured_corners: list[
        tuple[tuple[int, int, int, int], tuple[int, int, int]]
    ] = []
    original_put_text = overlay_module.cv2.putText
    original_corner = overlay_module._draw_corner_bbox

    def capture_text(image: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        captured_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    def capture_corner(
        image: Any,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        captured_corners.append((bbox, color))
        original_corner(image, bbox, color)

    monkeypatch.setattr(overlay_module.cv2, "putText", capture_text)
    monkeypatch.setattr(overlay_module, "_draw_corner_bbox", capture_corner)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = NormalizedBBox(0.781, 0.644, 0.862, 0.738)
    cigarette = SimpleNamespace(
        object_id="object-2",
        label="cigarette",
        bbox=bbox,
        confidence=0.92,
    )

    rendered = overlay_module.draw_event_rich_overlay(
        image,
        [cigarette],
        alerts_by_object_id={
            "object-2": (
                SimpleNamespace(event_label="smoking", confidence=0.95),
            )
        },
    )

    assert rendered is not image
    assert captured_corners == [
        (bbox.to_pixels(image.shape[1], image.shape[0]), overlay_module._ALERT_COLOR)
    ]
    assert captured_text == [
        "CIGARETTE",
        "ALERT  SMOKING 95%",
        "CONF 92%",
    ]


def test_rich_overlay_only_displays_supplied_sanitized_attributes(
    monkeypatch: Any,
) -> None:
    captured: list[str] = []
    original_put_text = overlay_module.cv2.putText

    def capture(image: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        captured.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(overlay_module.cv2, "putText", capture)
    image = np.zeros((260, 800, 3), dtype=np.uint8)
    objects = [
        SimpleNamespace(
            bbox=NormalizedBBox(0.05, 0.3, 0.35, 0.9),
            label="person",
            object_id="person-object",
            confidence=0.923,
            attributes={
                "track_id": 7,
                "name": "OpenRouter Alice\nADMIN",
                "estimated_age_range": "25-34",
                "estimated_gender": "feminine",
            },
        ),
        SimpleNamespace(
            bbox=NormalizedBBox(0.55, 0.3, 0.9, 0.9),
            label="backpack",
            object_id="bag-2",
            confidence=0.81,
            attributes={"color": "red"},
        ),
    ]

    overlay_module.draw_rich_overlay(image, objects)

    assert len(captured) == 4
    text = " ".join(captured)
    assert "P7" in text
    assert "25-34" in text
    assert "FEMININE" in text
    assert "BACKPACK" in text
    assert "#bag-2" in text
    assert "ADMIN" in text
    assert "\n" not in text
    assert not contains_private_implementation_detail(text)
    assert "AGE" not in " ".join(captured[2:])
    assert "GENDER" not in " ".join(captured[2:])


def test_rich_overlay_keeps_demographics_and_adds_compact_structured_appearance(
    monkeypatch: Any,
) -> None:
    captured: list[str] = []
    original_put_text = overlay_module.cv2.putText

    def capture(image: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        captured.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(overlay_module.cv2, "putText", capture)
    image = np.zeros((280, 900, 3), dtype=np.uint8)
    person = SimpleNamespace(
        bbox=NormalizedBBox(0.1, 0.35, 0.55, 0.9),
        label="person",
        object_id="person-object",
        confidence=0.923,
        attributes={
            "track_id": 8,
            "estimated_age_range": "18-60",
            "estimated_gender": "male-presenting",
            "estimated_direction": "Front",
            "wearing_glasses": True,
            "wearing_hat": False,
            "holding_object": True,
            "bag_type": "Backpack",
            "sleeve_type": "LongSleeve",
            "upper_style": "UpperLogo",
            "lower_styles": ("Trousers", "Skirt&Dress"),
            "wearing_boots": True,
            # Explicit structured values win over contradictory legacy labels.
            "appearance_attributes": (
                "NoGlasses",
                "Hat",
                "NoBag",
                "ShortSleeve",
            ),
        },
    )

    overlay_module.draw_rich_overlay(image, [person])

    assert len(captured) == 3
    assert "AGE 18-60" in captured[1]
    assert "GENDER MALE" in captured[1]
    assert "CONF 92%" in captured[1]
    assert "GLASSES" in captured[2]
    assert "HOLDING ITEM" in captured[2]
    assert "BACKPACK" in captured[2]
    assert "HAT" not in captured[2]
    assert "SHORT SLEEVE" not in captured[2]
    assert "\n" not in " ".join(captured)
    assert not contains_private_implementation_detail(" ".join(captured))


def test_rich_appearance_decodes_all_structured_fields_and_positive_fallbacks() -> None:
    structured = overlay_module._rich_appearance_tokens(
        {
            "estimated_direction": "side",
            "wearing_glasses": True,
            "wearing_hat": True,
            "holding_object": True,
            "bag_type": "shoulder bag",
            "sleeve_type": "short sleeve",
            "upper_style": "UpperStride",
            "lower_styles": ("lower pattern", "shorts"),
            "wearing_boots": True,
        }
    )

    assert structured == (
        "SIDE",
        "GLASSES",
        "HAT",
        "HOLDING ITEM",
        "SHOULDER BAG",
        "BOOTS",
        "SHORT SLEEVE",
        "TOP STRIPED",
        "BOTTOM PATTERN",
        "SHORTS",
    )

    fallback = overlay_module._rich_appearance_tokens(
        {
            "appearance_attributes": (
                "Back",
                "NoGlasses",
                "Glasses",
                "NoHat",
                "Hat",
                "HoldObjectsInFront",
                "NoBag",
                "HandBag",
                "LongSleeve",
                "UpperStripe",
                "LowerPattern",
                "Boots",
                "OpenRouter SECRET",
            )
        }
    )

    assert fallback == (
        "BACK",
        "GLASSES",
        "HAT",
        "HOLDING ITEM",
        "HANDBAG",
        "BOOTS",
        "LONG SLEEVE",
        "TOP STRIPED",
        "BOTTOM PATTERN",
    )


def test_rich_overlay_skips_bad_boxes_and_tolerates_bad_metadata() -> None:
    class BrokenBBox:
        def to_pixels(self, _width: int, _height: int) -> tuple[float, ...]:
            return (float("nan"), 10.0, 20.0, 30.0)

    image = np.zeros((120, 180, 3), dtype=np.uint8)
    objects = [
        None,
        SimpleNamespace(
            bbox=BrokenBBox(),
            label="person",
            confidence="not-a-number",
            attributes={"track_id": 1},
        ),
        SimpleNamespace(
            bbox=(0.8, 0.8, 0.2, 0.2),
            label="bad",
            attributes={"track_id": 2},
        ),
        {
            "bbox": (0.1, 0.2, 0.5, 0.8),
            "label": "parcel",
            "object_id": "parcel-3",
            "confidence": float("inf"),
            "attributes": ["not", "a", "mapping"],
        },
    ]

    rendered = overlay_module.draw_rich_overlay(image, objects)

    assert np.all(image == 0)
    assert np.any(rendered != image)
