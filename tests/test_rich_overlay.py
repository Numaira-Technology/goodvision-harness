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
    assert not np.array_equal(rendered[y1, (x1 + x2) // 2], source[y1, (x1 + x2) // 2])
    assert not np.array_equal(rendered[(y1 + y2) // 2, x1], source[(y1 + y2) // 2, x1])

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
    original_stroke = overlay_module.draw_person_stroke

    def capture_text(image: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        captured_text.append(text)
        return original_put_text(image, text, *args, **kwargs)

    def capture_stroke(
        image: Any,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
        contour: Any = None,
    ) -> None:
        captured_colors.append(color)
        original_stroke(image, bbox, color, contour)

    monkeypatch.setattr(overlay_module.cv2, "putText", capture_text)
    monkeypatch.setattr(overlay_module, "draw_person_stroke", capture_stroke)
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
    assert "GENDER" not in " ".join(captured_text)
    assert "MALE" not in " ".join(captured_text)
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
    assert "FEMININE" not in text
    assert "GENDER" not in text
    assert "BACKPACK" in text
    assert "#bag-2" in text
    assert "ADMIN" in text
    assert "\n" not in text
    assert not contains_private_implementation_detail(text)
    assert "AGE" not in " ".join(captured[2:])
    assert "GENDER" not in " ".join(captured[2:])


def test_rich_overlay_keeps_age_and_appearance_without_displaying_gender(
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
    assert "GENDER" not in " ".join(captured)
    assert "MALE" not in " ".join(captured)
    assert "CONF 92%" in captured[1]
    assert "GLASSES" not in captured[2]
    assert "HOLDING ITEM" in captured[2]
    assert "BACKPACK" in captured[2]
    assert "HAT" not in captured[2]
    assert "SHORT SLEEVE" not in captured[2]
    assert "LONG SLEEVE" not in captured[2]
    assert "TOP LOGO" not in captured[2]
    assert "TROUSERS" not in captured[2]
    assert "SKIRT/DRESS" not in captured[2]
    assert "BOOTS" not in captured[2]
    assert "\n" not in " ".join(captured)
    assert not contains_private_implementation_detail(" ".join(captured))


def test_rich_appearance_excludes_garments_from_structured_and_fallback_values(
) -> None:
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
        "HOLDING ITEM",
        "SHOULDER BAG",
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
        "HOLDING ITEM",
        "HANDBAG",
    )


def test_vlm_rich_overlay_displays_clothing_colors_and_visible_side(
    monkeypatch: Any,
) -> None:
    captured: list[str] = []
    original_put_text = overlay_module.cv2.putText

    def capture(image: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        captured.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(overlay_module.cv2, "putText", capture)
    person = _tracked_object("P0001")
    person.attributes.update(
        {
            "annotation_source": "vlm",
            "estimated_direction": "Side",
            "upper_style": "UpperLogo",
            "upper_color": "black",
            "lower_styles": ("Trousers",),
            "lower_color": "blue",
        }
    )

    overlay_module.draw_rich_overlay(
        np.zeros((280, 900, 3), dtype=np.uint8), [person]
    )

    assert captured[2] == "SIDE / BLACK TOP LOGO / BLUE TROUSERS"
    assert len(captured) == 3
    assert not contains_private_implementation_detail(" ".join(captured))


def test_vlm_appearance_prefers_structured_values_and_filters_unknown_labels() -> None:
    tokens = overlay_module._rich_appearance_tokens(
        {
            "annotation_source": "vlm",
            "estimated_direction": "Front",
            "wearing_glasses": True,
            "wearing_hat": False,
            "wearing_boots": True,
            "sleeve_type": "LongSleeve",
            "upper_style": "UpperStride",
            "lower_styles": ("Trousers",),
            "appearance_attributes": (
                "Back", "Hat", "ShortSleeve", "UpperLogo", "Shorts",
                "HandBag", "OpenRouter SECRET",
            ),
        }
    )

    assert tokens == (
        "FRONT", "LONG SLEEVE", "STRIPED TOP", "TROUSERS", "HANDBAG",
        "GLASSES", "BOOTS",
    )


def test_vlm_appearance_supports_color_only_and_safe_legacy_clothing() -> None:
    tokens = overlay_module._rich_appearance_tokens(
        {
            "annotation_source": "vlm",
            "upper_color": "dark-green",
            "lower_color": "OpenRouter SECRET",
            "appearance_attributes": ("Back", "ShortSleeve", "Skirt&Dress"),
        }
    )

    assert tokens == ("BACK", "SHORT SLEEVE", "DARK GREEN TOP", "SKIRT/DRESS")
    assert overlay_module._rich_appearance_tokens(
        {"upper_color": "red", "lower_color": "blue"}
    ) == ()


def test_vlm_cards_keep_crowded_people_readable_and_preserve_source_boxes(
    monkeypatch: Any,
) -> None:
    corners: list[tuple[int, int, int, int]] = []
    cards: list[tuple[int, int, int, int]] = []
    texts: list[str] = []
    leaders: list[tuple[Any, Any]] = []
    original_rectangle = overlay_module.cv2.rectangle
    original_put_text = overlay_module.cv2.putText
    original_line = overlay_module.cv2.line

    def capture_corner(image: Any, bbox: Any, color: Any) -> None:
        corners.append(bbox)

    def capture_rectangle(
        image: Any, start: Any, end: Any, color: Any, *args: Any, **kwargs: Any,
    ) -> Any:
        if color == (34, 39, 45):
            cards.append((*start, *end))
        return original_rectangle(image, start, end, color, *args, **kwargs)

    def capture_text(image: Any, text: str, *args: Any, **kwargs: Any) -> Any:
        texts.append(text)
        return original_put_text(image, text, *args, **kwargs)

    def capture_line(image: Any, start: Any, end: Any, *args: Any, **kwargs: Any) -> Any:
        leaders.append((start, end))
        return original_line(image, start, end, *args, **kwargs)

    monkeypatch.setattr(overlay_module, "_draw_corner_bbox", capture_corner)
    monkeypatch.setattr(overlay_module.cv2, "rectangle", capture_rectangle)
    monkeypatch.setattr(overlay_module.cv2, "putText", capture_text)
    monkeypatch.setattr(overlay_module.cv2, "line", capture_line)
    people = [
        SimpleNamespace(
            label="person", track_id=f"P{index:04d}", confidence=0.95,
            bbox=NormalizedBBox(left, 0.01, left + 0.085, 0.28),
            attributes={
                "annotation_source": "vlm", "estimated_age_range": "18-60",
                "estimated_direction": "Back", "sleeve_type": "ShortSleeve",
                "upper_color": "black", "lower_styles": ("Trousers",),
            },
        )
        for index, left in enumerate((0.0, 0.25, 0.445, 0.61), start=1)
    ]
    chairs = [
        SimpleNamespace(
            label="office chair", track_id=f"O{index:04d}", confidence=0.9,
            bbox=NormalizedBBox(left, 0.07, left + 0.1, 0.345),
            attributes={"annotation_source": "vlm"},
        )
        for index, left in enumerate((0.0, 0.24, 0.44, 0.62), start=5)
    ]
    objects = chairs + people  # Persons must be allocated space first.
    source = np.full((405, 720, 3), 80, dtype=np.uint8)

    overlay_module.draw_rich_overlay(source, objects)

    assert np.all(source == 80)
    assert corners == [item.bbox.to_pixels(720, 405) for item in chairs]
    assert len(cards) == len(leaders) == len(objects)
    person_titles = [index for index, text in enumerate(texts) if "PERSON" in text]
    object_titles = [index for index, text in enumerate(texts) if "OFFICE CHAIR" in text]
    assert len(person_titles) == len(object_titles) == 4
    assert max(person_titles) < min(object_titles)
    assert sum("BACK / SHORT SLEEVE / BLACK TOP / TROUSERS" == t for t in texts) == 4
    for index, rect in enumerate(cards):
        assert 0 <= rect[0] < rect[2] < 720
        assert 0 <= rect[1] < rect[3] < 405
        for other in cards[index + 1:]:
            assert (
                rect[2] < other[0] or other[2] < rect[0]
                or rect[3] < other[1] or other[3] < rect[1]
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
