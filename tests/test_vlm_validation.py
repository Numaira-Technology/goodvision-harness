from __future__ import annotations

import json
from typing import Any

import pytest

from goodvision_harness.rich_labeling import RichLabelProtocolError
from goodvision_harness.vlm_validation import validate_complete_compact_annotation

PERSON_KEYS = (
    "a", "g", "d", "s", "uc", "lc", "b", "us", "ls", "gl", "ht", "bt", "ho",
)


def _person(**attributes: Any) -> list[Any]:
    return ["p", 100, 100, 400, 900, dict.fromkeys(PERSON_KEYS) | attributes]


def _validate(rows: list[Any], scope: str = "all") -> None:
    return validate_complete_compact_annotation(json.dumps({"o": rows}), scope)


@pytest.mark.parametrize("scope", ["all", "people", "items"])
def test_empty_scene_has_a_complete_structure(scope: str) -> None:
    assert _validate([], scope) is None


@pytest.mark.parametrize("scope", ["all", "people"])
def test_all_unknown_fields_are_present_without_inventing_values(scope: str) -> None:
    assert _validate([_person()], scope) is None


def test_complete_mixed_scene_keeps_all_person_fields_and_optional_item_color() -> None:
    assert _validate([
        _person(a="a", g="m", d="s", s="l", uc="white", lc="black", b="n",
                us="st", ls=["tr", "st"], gl=0, ht=1, bt=0, ho=1),
        ["chair", 0, 0, 500, 1000],
        ["desk", 0, 500, 1000, 1000, {"c": "brown"}],
        ["screen", 1, 1, 2, 2, {"c": None}],
        ["bag", 2, 2, 3, 3, None],
    ]) is None


@pytest.mark.parametrize("key", PERSON_KEYS)
def test_each_missing_person_field_is_rejected_before_nulls_are_discarded(
    key: str,
) -> None:
    row = _person()
    del row[5][key]
    with pytest.raises(RichLabelProtocolError):
        _validate([row])


@pytest.mark.parametrize("row", [
    ["p", 1, 2, 3, 4],
    ["p", 1, 2, 3, 4, None],
    ["p", 1, 2, 3, 4, {}],
    ["p", 1, 2, 3, 4, 0.9, dict.fromkeys(PERSON_KEYS)],
    ["chair", 1, 2, 3, 4, {"a": "a"}],
    ["chair", 1, 2, 3, 4, {"c": "brown", "extra": None}],
    ["chair", 1, 2, 3, 4, 0.9],
    ["chair", 1, 2, 3, 4, {}, "extra"],
    ["", 1, 2, 3, 4],
    [" ", 1, 2, 3, 4],
    ["a" * 65, 1, 2, 3, 4],
    [42, 1, 2, 3, 4],
    [],
    {},
    None,
])
def test_incomplete_or_malformed_rows_are_rejected(row: Any) -> None:
    with pytest.raises(RichLabelProtocolError):
        _validate([row])


@pytest.mark.parametrize("label", ["person", "pedestrian", "P", " p ", "Person"])
@pytest.mark.parametrize("complete_attributes", [False, True])
def test_person_aliases_cannot_bypass_the_person_contract(
    label: str, complete_attributes: bool,
) -> None:
    row = _person() if complete_attributes else ["p", 1, 2, 3, 4]
    row[0] = label
    with pytest.raises(RichLabelProtocolError):
        _validate([row])


@pytest.mark.parametrize("attributes", [
    {"extra": None}, {"q": 0.9}, {"a": "adult"}, {"a": "unknown"},
    {"d": "Side"}, {"s": "short"}, {"b": "none"}, {"us": "none"},
    {"ls": "tr"}, {"ls": ["Trousers"]}, {"ls": ["tr"] * 7},
    {"ls": [None]}, {"gl": True}, {"ht": "0"}, {"bt": 2}, {"ho": 0.0},
    {"uc": ""}, {"lc": " "}, {"uc": "x" * 25}, {"uc": 1},
])
def test_invalid_attribute_values_and_unknown_keys_are_rejected(
    attributes: dict[str, Any],
) -> None:
    with pytest.raises(RichLabelProtocolError):
        _validate([_person(**attributes)])


@pytest.mark.parametrize("value", [True, False, 1.0, -1, 1001, "1", None])
@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_coordinates_must_be_integer_grid_values(value: Any, index: int) -> None:
    row = _person()
    row[index] = value
    with pytest.raises(RichLabelProtocolError):
        _validate([row])


@pytest.mark.parametrize("bbox", [[100, 100, 100, 900], [400, 100, 100, 900]])
def test_public_positive_area_check_is_preserved(bbox: list[int]) -> None:
    row = _person()
    row[1:5] = bbox
    with pytest.raises(RichLabelProtocolError):
        _validate([row])


def test_scoped_responses_cannot_return_other_entity_types() -> None:
    with pytest.raises(RichLabelProtocolError):
        _validate([_person()], "items")
    with pytest.raises(RichLabelProtocolError):
        _validate([["chair", 1, 2, 3, 4]], "people")


def test_total_and_nonperson_cardinality_limits_are_independent() -> None:
    item = ["chair", 1, 2, 3, 4]
    assert _validate([_person()] * 4 + [item] * 8) is None
    assert _validate([_person()] * 12, "people") is None
    assert _validate([item] * 8, "items") is None
    for rows, scope in (
        ([_person()] * 13, "all"), ([_person()] * 13, "people"),
        ([item] * 9, "all"), ([item] * 9, "items"),
        ([_person()] * 5 + [item] * 8, "all"),
    ):
        with pytest.raises(RichLabelProtocolError):
            _validate(rows, scope)


@pytest.mark.parametrize("text", [
    '{"objects":[]}', '{"o":[],"extra":null}', '{"o":null}', '[]',
    '{"o":[],"o":[]}', '{"o":[["chair",1,2,3,4,{"c":null,"c":"red"}]]}',
    '{"o":[["chair",NaN,2,3,4]]}', '```json\n{"o":[]}\n```',
    'private provider failure text',
])
def test_original_response_is_strict_json_and_error_does_not_echo_input(
    text: str,
) -> None:
    with pytest.raises(RichLabelProtocolError) as error:
        validate_complete_compact_annotation(text)
    assert str(error.value) == (
        "rich annotation does not satisfy the complete compact contract"
    )


@pytest.mark.parametrize("scope", ["invalid", "ALL", None])
def test_invalid_scope_raises_a_safe_protocol_error(scope: Any) -> None:
    with pytest.raises(RichLabelProtocolError):
        validate_complete_compact_annotation('{"o":[]}', scope)
