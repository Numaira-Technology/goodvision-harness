"""Validate the complete compact response before null attributes are omitted."""

from __future__ import annotations

import json
from typing import Any

from .rich_labeling import RichLabelProtocolError

_CONTRACT_ERROR = "rich annotation does not satisfy the complete compact contract"
_PERSON_LABELS = {"p", "person", "pedestrian"}


def _invalid(*_args: Any) -> Any:
    raise ValueError(_CONTRACT_ERROR)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _invalid()
        result[key] = value
    return result


def _enum_contains(choices: list[Any], value: Any) -> bool:
    # Python considers True == 1; JSON Schema treats booleans as distinct values.
    return any(
        value == choice and isinstance(value, bool) == isinstance(choice, bool)
        for choice in choices
    )


def _attribute_value(value: Any, schema: dict[str, Any]) -> None:
    if "enum" in schema:
        if not _enum_contains(schema["enum"], value):
            _invalid()
        return
    allowed_types = schema.get("type", [])
    if value is None and "null" in allowed_types:
        return
    if "string" in allowed_types:
        if (
            not isinstance(value, str)
            or not schema.get("minLength", 0) <= len(value) <= schema["maxLength"]
        ):
            _invalid()
        return
    if "array" in allowed_types:
        if not isinstance(value, list) or len(value) > schema["maxItems"]:
            _invalid()
        for item in value:
            _attribute_value(item, schema["items"])
        return
    # Do not silently loosen validation if the wire schema gains a new type.
    _invalid()


def _attribute_object(value: Any, schema: dict[str, Any]) -> None:
    if value is None and "null" in schema["type"]:
        return
    if not isinstance(value, dict):
        _invalid()
    properties = schema["properties"]
    if not set(schema.get("required", [])).issubset(value) or not set(value).issubset(
        properties,
    ):
        _invalid()
    for key, item in value.items():
        _attribute_value(item, properties[key])


def validate_complete_compact_annotation(text: str, scope: str = "all") -> None:
    """Require the full wire contract without claiming visual correctness.

    All thirteen person fields must be present; unknown values may be null.
    This must run on the original response, because the public parser omits
    nulls and intentionally accepts older, partial annotation formats.
    """

    # The client can import this helper without a circular module import.
    from .vlm_rich_labeling import (
        build_compact_rich_annotation_response_format,
        parse_rich_annotation,
    )

    try:
        if not isinstance(text, str) or scope not in ("all", "people", "items"):
            _invalid()
        schema = build_compact_rich_annotation_response_format(scope)["json_schema"][
            "schema"
        ]
        payload = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_invalid,
        )
        if not isinstance(payload, dict) or set(payload) != {"o"}:
            _invalid()
        rows = payload["o"]
        rows_schema = schema["properties"]["o"]
        if not isinstance(rows, list) or len(rows) > rows_schema["maxItems"]:
            _invalid()
        if scope == "all":
            person_schema, item_schema = rows_schema["items"]["anyOf"]
        else:
            person_schema = item_schema = rows_schema["items"]
        item_count = 0
        for row in rows:
            if not isinstance(row, list) or not row or not isinstance(row[0], str):
                _invalid()
            label = row[0]
            person = label.strip().lower() in _PERSON_LABELS
            if person:
                # Aliases or whitespace must not disguise a person as a short item.
                if label != "p" or scope == "items":
                    _invalid()
                row_schema = person_schema
            else:
                if scope == "people":
                    _invalid()
                item_count += 1
                if item_count > 8:
                    _invalid()
                row_schema = item_schema
            if not row_schema["minItems"] <= len(row) <= row_schema["maxItems"]:
                _invalid()
            fields = row_schema["prefixItems"]
            if not person and (
                not label.strip()
                or not fields[0]["minLength"] <= len(label) <= fields[0]["maxLength"]
            ):
                _invalid()
            for value, coordinate in zip(row[1:5], fields[1:5], strict=True):
                if (
                    type(value) is not int
                    or not coordinate["minimum"] <= value <= coordinate["maximum"]
                ):
                    _invalid()
            if len(row) == 6:
                _attribute_object(row[5], fields[5])
        # Retain positive-area and public attribute validation in one place.
        parse_rich_annotation(text)
    except (ValueError, TypeError, KeyError, IndexError, RichLabelProtocolError):
        # Provider text or arbitrary attribute names never enter public errors.
        raise RichLabelProtocolError(_CONTRACT_ERROR) from None
