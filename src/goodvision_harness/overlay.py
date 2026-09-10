"""Render grounded event results on source frames for the demo stream."""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .privacy import redact_public_text

_RICH_COLORS: tuple[tuple[int, int, int], ...] = (
    (63, 221, 255),
    (255, 184, 77),
    (113, 232, 104),
    (255, 116, 184),
    (224, 149, 255),
    (255, 216, 91),
    (94, 202, 255),
    (174, 238, 77),
)
_PERSON_LABELS = frozenset({"person", "human", "pedestrian"})
_ALERT_COLOR = (40, 40, 220)
_FACE_COLORS = {
    "known": (55, 180, 70),
    "unknown": (40, 40, 220),
    "unusable": (105, 105, 105),
}

_APPEARANCE_TOKENS = {
    "front": ("direction", "FRONT"),
    "side": ("direction", "SIDE"),
    "back": ("direction", "BACK"),
    "holdobjectsinfront": ("holding", "HOLDING ITEM"),
    "backpack": ("bag", "BACKPACK"),
    "handbag": ("bag", "HANDBAG"),
    "shoulderbag": ("bag", "SHOULDER BAG"),
}
_APPEARANCE_PRIORITY = (
    "direction",
    "holding",
    "bag",
)
_VLM_APPEARANCE_TOKENS = {
    **_APPEARANCE_TOKENS,
    "shortsleeve": ("sleeve", "SHORT SLEEVE"),
    "longsleeve": ("sleeve", "LONG SLEEVE"),
    "upperstride": ("upper", "STRIPED TOP"),
    "upperstripe": ("upper", "STRIPED TOP"),
    "upperlogo": ("upper", "TOP LOGO"),
    "upperplaid": ("upper", "PLAID TOP"),
    "uppersplice": ("upper", "MIXED TOP"),
    "lowerstripe": ("lower", "STRIPED BOTTOM"),
    "lowerpattern": ("lower", "PATTERNED BOTTOM"),
    "longcoat": ("lower", "LONG COAT"),
    "trousers": ("lower", "TROUSERS"),
    "shorts": ("lower", "SHORTS"),
    "skirtdress": ("lower", "SKIRT/DRESS"),
    "glasses": ("glasses", "GLASSES"),
    "hat": ("hat", "HAT"),
    "boots": ("boots", "BOOTS"),
}
_VLM_APPEARANCE_PRIORITY = (
    "direction",
    "sleeve",
    "upper",
    "lower",
    "holding",
    "bag",
    "glasses",
    "hat",
    "boots",
)
_GARMENT_COLORS = frozenset(
    {
        "beige", "black", "blue", "brown", "gold", "gray", "green", "grey",
        "khaki", "maroon", "multicolor", "navy", "olive", "orange", "pink",
        "purple", "red", "silver", "tan", "teal", "turquoise", "white", "yellow",
    }
)


@dataclass(frozen=True)
class _VLMCard:
    bbox: tuple[int, int, int, int]
    primary: str
    detail: str
    appearance: str
    color: tuple[int, int, int]
    alert: str
    is_person: bool


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
    incomplete: bool = False,
    known_faces: Sequence[Any] = (),
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
            incomplete=incomplete,
        )

    positive = bool(detected) and not stale
    color = (40, 40, 220) if positive else (55, 180, 70)
    if stale or (incomplete and not positive):
        color = (90, 150, 220)

    for item in known_faces if not stale else ():
        attributes = getattr(item, "attributes", {})
        person_id = str(attributes.get("person_id", "")).strip()
        if not person_id:
            continue
        try:
            x1, y1, x2, y2 = item.bbox.to_pixels(width, height)
        except (AttributeError, TypeError, ValueError):
            continue
        identity_color = (55, 180, 70)
        cv2.rectangle(
            rendered, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)),
            identity_color, 2,
        )
        _text_with_background(
            rendered, person_id[:120],
            (x1, max(18, y1)), identity_color,
        )

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
    if stale or (incomplete and not detected):
        return rendered
    state = positive_label if detected else "CLEAR"
    title = f"{state}  {float(confidence):.2f}"
    _text_with_background(rendered, title[:100], (12, 28), color, scale=0.7)
    if description:
        _text_with_background(
            rendered,
            redact_public_text(description)[:110],
            (12, 55),
            (45, 45, 45),
            scale=0.48,
        )
    return rendered


def draw_rich_overlay(
    bgr: Any,
    objects: Sequence[Any],
    *,
    alerts_by_track_id: Mapping[str, Sequence[Any]] | None = None,
) -> Any:
    """Return a copy of ``bgr`` with track-stable rich object annotations.

    The renderer only presents metadata supplied by the caller.  In particular,
    it does not derive demographic attributes from pixels or from an object's
    label.  Bad objects are skipped independently so one malformed model result
    cannot prevent the remaining annotations from being drawn.
    """

    rendered = bgr.copy()
    try:
        height, width = rendered.shape[:2]
    except (AttributeError, TypeError, ValueError):
        return rendered
    if width < 1 or height < 1:
        return rendered

    if isinstance(objects, Mapping):
        safe_objects: tuple[Any, ...] = (objects,)
    else:
        try:
            safe_objects = tuple(objects or ())
        except (TypeError, ValueError):
            return rendered

    vlm_cards: list[_VLMCard] = []
    for index, item in enumerate(safe_objects):
        bbox = _rich_bbox_to_pixels(_object_value(item, "bbox"), width, height)
        if bbox is None:
            continue
        attributes = _rich_attributes(item)
        track_id = _first_scalar(
            attributes,
            ("track_id", "tracking_id"),
        )
        if track_id is None:
            track_id = _object_value(item, "track_id")
        object_id = _object_value(item, "object_id")
        color_key = track_id
        if color_key is None:
            color_key = object_id
        if color_key is None:
            color_key = _object_value(item, "label", index)
        alerts = _rich_alerts_for_track(alerts_by_track_id, track_id)
        alert_text = _rich_alert_text(alerts)
        color = _ALERT_COLOR if alert_text else _rich_track_color(color_key)

        _draw_corner_bbox(rendered, bbox, color)
        primary, detail, appearance = _rich_label_lines(
            item,
            attributes=attributes,
            track_id=track_id,
            object_id=object_id,
        )
        if _mapping_value(attributes, "annotation_source") == "vlm":
            label = _public_hud_text(_object_value(item, "label", "object"))
            vlm_cards.append(_VLMCard(
                bbox, primary, detail, appearance, color, alert_text,
                label.casefold() in _PERSON_LABELS,
            ))
            continue
        _draw_rich_card(
            rendered,
            bbox,
            primary,
            detail,
            appearance,
            color,
            alert=alert_text,
        )

    if vlm_cards:
        _draw_vlm_cards(rendered, vlm_cards)
    return rendered


def draw_event_rich_overlay(
    bgr: Any,
    objects: Sequence[Any],
    *,
    alerts_by_object_id: Mapping[str, Sequence[Any]],
) -> Any:
    """Draw unmatched model objects with the same HUD language as Rich boxes."""

    rendered = bgr.copy()
    try:
        height, width = rendered.shape[:2]
    except (AttributeError, TypeError, ValueError):
        return rendered
    if width < 1 or height < 1:
        return rendered

    if isinstance(objects, Mapping):
        safe_objects: tuple[Any, ...] = (objects,)
    else:
        try:
            safe_objects = tuple(objects or ())
        except (TypeError, ValueError):
            return rendered

    for item in safe_objects:
        object_id = _object_value(item, "object_id")
        alerts = _rich_alerts_for_track(alerts_by_object_id, object_id)
        alert_text = _rich_alert_text(alerts)
        if not alert_text:
            continue
        bbox = _rich_bbox_to_pixels(_object_value(item, "bbox"), width, height)
        if bbox is None:
            continue

        label = _public_hud_text(_object_value(item, "label", "object"))
        primary = (label or "object").upper()
        confidence = _rich_confidence(_object_value(item, "confidence"))
        detail = f"CONF {confidence}" if confidence else "DETECTED"
        _draw_corner_bbox(rendered, bbox, _ALERT_COLOR)
        _draw_rich_card(
            rendered,
            bbox,
            primary,
            detail,
            "",
            _ALERT_COLOR,
            alert=alert_text,
        )

    return rendered


def _object_value(item: Any, key: str, default: Any = None) -> Any:
    try:
        if isinstance(item, Mapping):
            return item.get(key, default)
        return getattr(item, key, default)
    except (AttributeError, KeyError, TypeError, ValueError):
        return default


def _rich_attributes(item: Any) -> Mapping[str, Any]:
    attributes = _object_value(item, "attributes", {})
    return attributes if isinstance(attributes, Mapping) else {}


def _rich_alerts_for_track(
    alerts_by_track_id: Mapping[str, Sequence[Any]] | None,
    track_id: Any,
) -> tuple[Any, ...]:
    if not isinstance(alerts_by_track_id, Mapping) or track_id is None:
        return ()
    try:
        alerts = alerts_by_track_id.get(str(track_id), ())
    except (AttributeError, KeyError, TypeError, ValueError):
        return ()
    if isinstance(alerts, Sequence) and not isinstance(
        alerts, (str, bytes, bytearray)
    ):
        try:
            return tuple(alerts)
        except (TypeError, ValueError):
            return ()
    return ()


def _rich_alert_text(alerts: Sequence[Any]) -> str:
    badges: list[str] = []
    for alert in alerts:
        label = _public_hud_text(_object_value(alert, "event_label"), limit=32)
        if not label:
            continue
        badge = label.upper()
        confidence = _rich_confidence(_object_value(alert, "confidence"))
        if confidence:
            badge += f" {confidence}"
        if badge not in badges:
            badges.append(badge)
    return ("ALERT  " + " / ".join(badges))[:96] if badges else ""


def _mapping_value(values: Mapping[str, Any], key: str) -> Any:
    try:
        return values.get(key)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _first_scalar(values: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = _mapping_value(values, key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (str, int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            if _public_hud_text(value):
                return value
    return None


def _rich_bbox_to_pixels(
    bbox: Any,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None

    normalized = False
    try:
        to_pixels = getattr(bbox, "to_pixels", None)
    except (AttributeError, TypeError, ValueError):
        to_pixels = None
    if callable(to_pixels):
        try:
            raw_coordinates = to_pixels(width, height)
        except (AttributeError, OverflowError, TypeError, ValueError):
            return None
    else:
        coordinate_system = ""
        if isinstance(bbox, Mapping):
            try:
                raw_coordinates = tuple(bbox[key] for key in ("x1", "y1", "x2", "y2"))
                coordinate_system = str(bbox.get("coordinate_system", ""))
            except (KeyError, TypeError, ValueError):
                return None
        elif isinstance(bbox, Sequence) and not isinstance(
            bbox, (str, bytes, bytearray)
        ):
            try:
                raw_coordinates = tuple(bbox)
            except (TypeError, ValueError):
                return None
        else:
            try:
                raw_coordinates = tuple(
                    getattr(bbox, key) for key in ("x1", "y1", "x2", "y2")
                )
            except (AttributeError, TypeError, ValueError):
                return None
        normalized = "normalized" in coordinate_system.casefold()

    try:
        if len(raw_coordinates) != 4 or any(
            isinstance(value, bool) for value in raw_coordinates
        ):
            return None
        x1, y1, x2, y2 = (float(value) for value in raw_coordinates)
    except (OverflowError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None

    if not callable(to_pixels) and not normalized:
        normalized = all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2))
    if normalized:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None
    left = max(0, min(width - 1, round(x1)))
    top = max(0, min(height - 1, round(y1)))
    right = max(left + 1, min(width, round(x2)))
    bottom = max(top + 1, min(height, round(y2)))
    return left, top, right, bottom


def _stable_key(value: Any) -> str:
    try:
        rendered = str(value)
    except (TypeError, ValueError):
        return "object"
    return rendered or "object"


def _rich_track_color(value: Any) -> tuple[int, int, int]:
    checksum = zlib.crc32(_stable_key(value).encode("utf-8", "replace"))
    return _RICH_COLORS[checksum % len(_RICH_COLORS)]


def _draw_corner_bbox(
    image: Any,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2_exclusive, y2_exclusive = bbox
    x2 = max(x1, x2_exclusive - 1)
    y2 = max(y1, y2_exclusive - 1)
    corner = max(4, min(28, round(min(x2 - x1 + 1, y2 - y1 + 1) * 0.24)))
    segments = (
        ((x1, y1), (min(x2, x1 + corner), y1)),
        ((x1, y1), (x1, min(y2, y1 + corner))),
        ((x2, y1), (max(x1, x2 - corner), y1)),
        ((x2, y1), (x2, min(y2, y1 + corner))),
        ((x1, y2), (min(x2, x1 + corner), y2)),
        ((x1, y2), (x1, max(y1, y2 - corner))),
        ((x2, y2), (max(x1, x2 - corner), y2)),
        ((x2, y2), (x2, max(y1, y2 - corner))),
    )
    for start, end in segments:
        cv2.line(image, start, end, (12, 14, 18), 5, cv2.LINE_AA)
    for start, end in segments:
        cv2.line(image, start, end, color, 2, cv2.LINE_AA)


def _public_hud_text(value: Any, *, limit: int = 72) -> str:
    if value is None:
        return ""
    try:
        rendered = redact_public_text(value)
    except (TypeError, ValueError):
        return ""
    rendered = "".join(
        character if character.isprintable() else " " for character in rendered
    )
    return " ".join(rendered.split())[:limit]


def _rich_label_lines(
    item: Any,
    *,
    attributes: Mapping[str, Any],
    track_id: Any,
    object_id: Any,
) -> tuple[str, str, str]:
    raw_label = _object_value(item, "label", "object")
    label = _public_hud_text(raw_label) or "object"
    is_person = label.casefold() in _PERSON_LABELS
    name = _first_scalar(attributes, ("name", "display_name", "person_name"))
    person_id = _first_scalar(attributes, ("person_id", "identity_id"))

    if is_person:
        identity = _public_hud_text(name or person_id)
        primary_parts: list[str] = []
        track_text = _public_hud_text(track_id, limit=24)
        if track_text:
            track_token = (
                track_text.upper()
                if track_text.casefold().startswith("p")
                else f"P{track_text}"
            )
            primary_parts.append(track_token)
        primary_parts.append(identity or label.upper())
        if not track_text and person_id is None:
            fallback_id = _public_hud_text(object_id, limit=28)
            if fallback_id:
                primary_parts.append(fallback_id)
    else:
        primary_parts = [label.upper()]
        display_id = _public_hud_text(object_id or track_id, limit=28)
        if display_id and display_id.casefold() != label.casefold():
            primary_parts.append(f"#{display_id.lstrip('#')}")

    details: list[str] = []
    age = _first_scalar(
        attributes,
        (
            "estimated_age_range",
            "age_band",
            "estimated_age_band",
            "estimated_age",
            "age",
        ),
    )
    if age is not None:
        details.append(f"AGE {_public_hud_text(age, limit=24).upper()}")
    confidence = _rich_confidence(_object_value(item, "confidence"))
    if confidence:
        details.append(f"CONF {confidence}")

    primary = "  |  ".join(part for part in primary_parts if part)[:72]
    detail = "  |  ".join(details)[:72] or "DETECTED"
    appearance = " / ".join(_rich_appearance_tokens(attributes))[:72]
    return primary, detail, appearance if is_person else ""


def _attribute_key(value: Any) -> str:
    text = _public_hud_text(value, limit=32)
    return "".join(character for character in text.casefold() if character.isalnum())


def _attribute_items(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def _rich_appearance_tokens(attributes: Mapping[str, Any]) -> tuple[str, ...]:
    """Decode public appearance labels, with garment details for VLM output."""

    found: list[tuple[str, str]] = []
    covered: set[str] = set()
    is_vlm = _mapping_value(attributes, "annotation_source") == "vlm"
    tokens = _VLM_APPEARANCE_TOKENS if is_vlm else _APPEARANCE_TOKENS
    priority = _VLM_APPEARANCE_PRIORITY if is_vlm else _APPEARANCE_PRIORITY
    fields = (
        ("estimated_direction", "direction"),
        ("bag_type", "bag"),
    )
    if is_vlm:
        fields += (
            ("sleeve_type", "sleeve"),
            ("upper_style", "upper"),
            ("lower_styles", "lower"),
        )
    for field, category in fields:
        if field not in attributes:
            continue
        covered.add(category)
        value = _mapping_value(attributes, field)
        for item in _attribute_items(value):
            match = tokens.get(_attribute_key(item))
            if match is not None and match[0] == category:
                found.append(match)

    flags = (
        ("holding_object", "holding", "HOLDING ITEM"),
    )
    if is_vlm:
        flags += (
            ("wearing_glasses", "glasses", "GLASSES"),
            ("wearing_hat", "hat", "HAT"),
            ("wearing_boots", "boots", "BOOTS"),
        )
    for field, category, token in flags:
        if field in attributes:
            covered.add(category)
            if _mapping_value(attributes, field) is True:
                found.append((category, token))

    fallback = _mapping_value(attributes, "appearance_attributes")
    if isinstance(fallback, Sequence) and not isinstance(
        fallback, (str, bytes, bytearray)
    ):
        for item in fallback:
            match = tokens.get(_attribute_key(item))
            if match is not None and match[0] not in covered:
                found.append(match)

    if is_vlm:
        for field, category, garment in (
            ("upper_color", "upper", "TOP"),
            ("lower_color", "lower", "BOTTOM"),
        ):
            color = _rich_garment_color(_mapping_value(attributes, field))
            if not color:
                continue
            for index, (found_category, token) in enumerate(found):
                if found_category == category:
                    found[index] = (category, f"{color} {token}")
                    break
            else:
                found.append((category, f"{color} {garment}"))

    found.sort(key=lambda item: priority.index(item[0]))
    return tuple(dict.fromkeys(token for _category, token in found))


def _rich_garment_color(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    words = _public_hud_text(value, limit=32).casefold().replace("-", " ").split()
    allowed = _GARMENT_COLORS | {"light", "dark", "and", "/"}
    if (
        not 1 <= len(words) <= 4
        or any(word not in allowed for word in words)
        or not any(word in _GARMENT_COLORS for word in words)
    ):
        return ""
    return " ".join(words).upper()


def _rich_confidence(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    try:
        rendered = float(value)
    except (OverflowError, TypeError, ValueError):
        return ""
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        return ""
    return f"{rendered:.0%}"


def _fit_hud_text(text: str, width: int, scale: float) -> str:
    if width <= 0:
        return ""
    if _text_size(text, scale)[0][0] <= width:
        return text
    suffix = "..."
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[:middle].rstrip() + suffix
        if _text_size(candidate, scale)[0][0] <= width:
            low = middle
        else:
            high = middle - 1
    return (text[:low].rstrip() + suffix) if low else ""


def _rich_card_lines(
    primary: str,
    detail: str,
    appearance: str,
    alert: str,
    compact: bool = False,
) -> list[tuple[str, float, tuple[int, int, int]]]:
    if compact:
        return [(primary.replace("  |  ", " "), 0.38, (225, 232, 237))]
    lines = [(primary, 0.48, (245, 248, 250))]
    if alert:
        lines.append((alert, 0.42, (145, 178, 255)))
    lines.append((detail, 0.39, (190, 199, 207)))
    if appearance:
        lines.append((appearance, 0.36, (164, 177, 188)))
    return lines


def _rich_card_size(
    width: int,
    height: int,
    lines: Sequence[tuple[str, float, tuple[int, int, int]]],
) -> tuple[int, int]:
    text_widths = tuple(
        _text_size(text, scale)[0][0]
        for text, scale, _text_color in lines
    )
    card_width = min(
        width,
        max(96, min(360, max(text_widths, default=74) + 22)),
    )
    card_height = min(height, 8 + 17 * len(lines))
    return card_width, card_height


def _card_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    gap: int = 4,
) -> bool:
    return (
        left[0] < right[2] + gap and left[2] + gap > right[0]
        and left[1] < right[3] + gap and left[3] + gap > right[1]
    )


def _place_vlm_card(
    bbox: tuple[int, int, int, int],
    card_size: tuple[int, int],
    frame_size: tuple[int, int],
    occupied: Sequence[tuple[int, int, int, int]],
    object_boxes: Sequence[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Pack nearby cards into free space before allowing them over objects."""

    card_width, card_height = card_size
    width, height = frame_size
    source_x = (bbox[0] + bbox[2]) // 2
    source_y = (bbox[1] + bbox[3]) // 2
    obstacles = (*object_boxes, *occupied)
    xs = {0, width - card_width, bbox[0], source_x - card_width // 2}
    ys = {0, height - card_height, bbox[1] - card_height - 6, bbox[3] + 6}
    for left, top, right, bottom in obstacles:
        xs.update((left - card_width - 6, right + 6))
        ys.update((top - card_height - 6, bottom + 6))
    xs = {max(0, min(x, width - card_width)) for x in xs}
    ys = {max(0, min(y, height - card_height)) for y in ys}
    candidates = sorted(
        ((x, y, x + card_width, y + card_height) for x in xs for y in ys),
        key=lambda rect: (
            abs((rect[0] + rect[2]) // 2 - source_x)
            + abs((rect[1] + rect[3]) // 2 - source_y),
            rect[1], rect[0],
        ),
    )
    for avoid_objects in (True, False):
        for rect in candidates:
            if any(_card_overlap(rect, other) for other in occupied):
                continue
            if avoid_objects and any(
                _card_overlap(rect, other) for other in object_boxes
            ):
                continue
            return rect
    return None


def _draw_vlm_cards(image: Any, cards: Sequence[_VLMCard]) -> None:
    height, width = image.shape[:2]
    occupied: list[tuple[int, int, int, int]] = []
    planned: list[tuple[_VLMCard, tuple[int, int, int, int], bool]] = []
    object_boxes = tuple(card.bbox for card in cards)
    ordered = sorted(
        cards,
        key=lambda card: (
            not card.is_person, not bool(card.alert), card.bbox[1], card.bbox[0],
            card.primary,
        ),
    )
    for card in ordered:
        compact = not card.is_person and not card.alert
        lines = _rich_card_lines(
            card.primary, card.detail, card.appearance, card.alert, compact,
        )
        size = _rich_card_size(width, height, lines)
        rect = _place_vlm_card(
            card.bbox, size, (width, height), occupied, object_boxes,
        )
        if rect is not None:
            occupied.append(rect)
            planned.append((card, rect, compact))

    # Leaders go underneath every card, so dense scenes cannot cross out text.
    for card, rect, _compact in planned:
        target_x = (rect[0] + rect[2]) // 2
        target_y = (rect[1] + rect[3]) // 2
        source = (
            max(card.bbox[0], min(target_x, card.bbox[2] - 1)),
            max(card.bbox[1], min(target_y, card.bbox[3] - 1)),
        )
        target = (
            max(rect[0], min(source[0], rect[2] - 1)),
            max(rect[1], min(source[1], rect[3] - 1)),
        )
        cv2.line(image, source, target, card.color, 1, cv2.LINE_AA)
    for card, rect, compact in planned:
        _draw_rich_card(
            image, card.bbox, card.primary, card.detail, card.appearance, card.color,
            alert=card.alert, position=(rect[0], rect[1]), compact=compact,
        )


def _draw_rich_card(
    image: Any,
    bbox: tuple[int, int, int, int],
    primary: str,
    detail: str,
    appearance: str,
    color: tuple[int, int, int],
    *,
    alert: str = "",
    position: tuple[int, int] | None = None,
    compact: bool = False,
) -> None:
    height, width = image.shape[:2]
    if width < 2 or height < 2:
        return
    x1, y1, _x2, y2 = bbox
    lines = _rich_card_lines(primary, detail, appearance, alert, compact)
    card_width, card_height = _rich_card_size(width, height, lines)
    left = max(0, min(x1, width - card_width))
    if y1 >= card_height + 4:
        top = y1 - card_height - 3
    elif height - y2 >= card_height + 3:
        top = y2 + 3
    else:
        top = max(0, min(y1 + 3, height - card_height))
    if position is not None:
        left, top = position
    right = min(width - 1, left + card_width - 1)
    bottom = min(height - 1, top + card_height - 1)

    region = image[top : bottom + 1, left : right + 1]
    tint = region.copy()
    tint[:] = (16, 20, 26)
    cv2.addWeighted(tint, 0.78, region, 0.22, 0.0, region)
    cv2.rectangle(image, (left, top), (right, bottom), (34, 39, 45), 1)
    cv2.rectangle(
        image,
        (left, top),
        (min(right, left + 3), bottom),
        color,
        -1,
    )

    text_x = min(right, left + 10)
    available_width = max(0, right - text_x - 5)
    for index, (text, scale, text_color) in enumerate(lines):
        fitted = _fit_hud_text(text, available_width, scale)
        _put_hud_text(
            image,
            fitted,
            (text_x, min(bottom - 2, top + 17 * (index + 1))),
            scale,
            text_color,
        )


def draw_face_boxes(bgr: Any, objects: Sequence[Any]) -> Any:
    """Return a copy with face identities, without replacing the event verdict."""

    rendered = bgr.copy()
    _draw_face_boxes(rendered, objects)
    return rendered


def _draw_face_boxes(rendered: Any, objects: Sequence[Any]) -> list[str]:
    height, width = rendered.shape[:2]
    statuses: list[str] = []
    for item in objects:
        attributes = getattr(item, "attributes", {})
        if not isinstance(attributes, Mapping):
            attributes = {}
        status = str(attributes.get("status", "unusable"))
        if status not in _FACE_COLORS:
            status = "unusable"
        statuses.append(status)
        bbox = getattr(item, "bbox", None)
        if bbox is None:
            continue
        try:
            x1, y1, x2, y2 = bbox.to_pixels(width, height)
        except (AttributeError, TypeError, ValueError):
            continue
        color = _FACE_COLORS[status]
        cv2.rectangle(
            rendered,
            (x1, y1),
            (max(x1, x2 - 1), max(y1, y2 - 1)),
            color,
            2,
        )
        if status == "known":
            person_id = attributes.get("person_id")
            name = redact_public_text(attributes.get("name") or "")
            if isinstance(person_id, str) and person_id:
                # Enrollment IDs are business identifiers and must not be redacted.
                label = f"ID: {person_id}"
                if name and name != person_id:
                    label += f" | {name}"
            else:
                label = name or "known"
        elif status == "unknown":
            label = "UNKNOWN"
        else:
            label = "unusable"
        similarity = attributes.get("similarity")
        if isinstance(similarity, (int, float)) and not isinstance(similarity, bool):
            label += f" {float(similarity):.2f}"
        _text_with_background(rendered, label[:80], (x1, max(18, y1)), color)
    return statuses


def _draw_face_overlay(
    rendered: Any,
    *,
    detected: bool,
    confidence: float,
    description: str,
    objects: Sequence[Any],
    stale: bool,
    incomplete: bool = False,
) -> Any:
    if stale:
        return rendered
    statuses = _draw_face_boxes(rendered, objects)
    if detected or "unknown" in statuses:
        state, color = "UNKNOWN", _FACE_COLORS["unknown"]
    elif incomplete or "unusable" in statuses:
        return rendered
    else:
        state, color = "CLEAR", _FACE_COLORS["known"]
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
            redact_public_text(description)[:110],
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
    text = _fit_hud_text(text, image.shape[1] - 8, scale)
    if not text:
        return
    (text_width, text_height), baseline = _text_size(text, scale)
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
    _put_hud_text(
        image,
        text,
        (x + 4, y - 2),
        scale,
        (255, 255, 255),
    )


@lru_cache(maxsize=16)
def _unicode_font(size: int) -> ImageFont.FreeTypeFont:
    # Ship the font with the package so offline Linux and Windows behave alike.
    return ImageFont.truetype(
        str(Path(__file__).with_name("fonts") / "NotoSansCJKsc-Regular.otf"), size,
    )


def _text_size(text: str, scale: float) -> tuple[tuple[int, int], int]:
    if text.isascii():
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    font = _unicode_font(max(10, round(26 * scale)))
    left, top, right, bottom = font.getbbox(text, anchor="ls")
    return (right - min(0, left), -top), max(0, bottom)


def _put_hud_text(
    image: Any,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
) -> None:
    if text.isascii():
        cv2.putText(
            image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
            1, cv2.LINE_AA,
        )
        return
    font = _unicode_font(max(10, round(26 * scale)))
    x, y = origin
    left, top, right, bottom = font.getbbox(text, anchor="ls")
    left, top = max(0, x + left), max(0, y + top)
    right, bottom = min(image.shape[1], x + right), min(image.shape[0], y + bottom)
    if right <= left or bottom <= top:
        return
    # Convert just the text rectangle, not an entire video frame per caption.
    region = image[top:bottom, left:right]
    canvas = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(canvas).text(
        (x - left, y - top), text, font=font, anchor="ls", fill=color[::-1],
    )
    region[:] = np.asarray(canvas)[:, :, ::-1]
