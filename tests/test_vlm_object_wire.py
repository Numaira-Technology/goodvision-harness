from __future__ import annotations

import base64
import json
from typing import Any

import jsonschema
import pytest

from goodvision_harness.rich_labeling import RichLabelProtocolError
from goodvision_harness.vlm_object_wire import (
    build_object_wire_messages,
    build_object_wire_response_format,
    object_wire_to_compact,
)
from goodvision_harness.vlm_rich_labeling import parse_rich_annotation

PERSON_FIELDS = (
    "a", "g", "d", "s", "uc", "lc", "b", "us", "ls", "gl", "ht", "bt", "ho",
)


def _person(**attributes: Any) -> dict[str, Any]:
    return {"box": [100, 100, 400, 900], **dict.fromkeys(PERSON_FIELDS), **attributes}


def _item(**updates: Any) -> dict[str, Any]:
    return {"label": "chair", "box": [500, 400, 900, 1000], "c": None, **updates}


def _convert(payload: Any, scope: str = "all") -> dict[str, Any]:
    return json.loads(object_wire_to_compact(json.dumps(payload), scope))


@pytest.mark.parametrize("scope", ["all", "people", "items"])
def test_schema_and_converter_agree_on_valid_scoped_payload(scope: str) -> None:
    payload = {"people": [_person()], "items": [_item()]}
    if scope != "all":
        payload = {scope: payload[scope]}
    response_format = build_object_wire_response_format(scope)
    schema = response_format["json_schema"]["schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)
    assert response_format["json_schema"]["strict"] is True
    converted = _convert(payload, scope)
    assert len(converted["o"]) == (2 if scope == "all" else 1)
    assert len(parse_rich_annotation(json.dumps(converted))) == len(converted["o"])


@pytest.mark.parametrize("scope", ["all", "people", "items"])
def test_schema_has_no_tuple_or_conditional_keywords_and_all_fields_have_types(
    scope: str,
) -> None:
    def check(node: Any) -> None:
        if not isinstance(node, dict):
            return
        assert not {"prefixItems", "anyOf", "oneOf", "not", "const"}.intersection(node)
        assert "type" in node
        if node["type"] == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        elif "array" in node["type"]:
            assert isinstance(node["items"], dict)
            check(node["items"])

    check(build_object_wire_response_format(scope)["json_schema"]["schema"])


def test_conversion_preserves_explicit_values_and_null_without_normalizing() -> None:
    person = _person(
        a="a", g="m", d="s", s="l", uc="WHITE", lc=None, b="n", us="lo",
        ls=["tr", "st"], gl=1, ht=0, bt=None, ho=0,
    )
    item = _item(c="Brown")
    converted = _convert({"people": [person], "items": [item]})
    assert converted == {"o": [
        ["p", *person["box"], {key: person[key] for key in PERSON_FIELDS}],
        ["chair", *item["box"], {"c": "Brown"}],
    ]}
    assert converted["o"][0][5]["lc"] is None
    assert converted["o"][0][5]["uc"] == "WHITE"


@pytest.mark.parametrize("scope", ["all", "people", "items"])
def test_empty_scope_converts_to_empty_compact_rows(scope: str) -> None:
    payload = {"people": [], "items": []} if scope == "all" else {scope: []}
    assert _convert(payload, scope) == {"o": []}


@pytest.mark.parametrize("key", ("box", *PERSON_FIELDS))
def test_person_cannot_omit_any_field_even_when_unknown(key: str) -> None:
    person = _person()
    del person[key]
    with pytest.raises(RichLabelProtocolError):
        _convert({"people": [person]}, "people")


@pytest.mark.parametrize("key", ["label", "box", "c"])
def test_item_cannot_omit_any_field(key: str) -> None:
    item = _item()
    del item[key]
    with pytest.raises(RichLabelProtocolError):
        _convert({"items": [item]}, "items")


@pytest.mark.parametrize("updates", [
    {"gl": True}, {"ht": "0"}, {"bt": 2}, {"ho": 1.0},
    {"a": "18-60"}, {"d": "Side"}, {"ls": "tr"}, {"us": "unknown"},
    {"uc": "x" * 25}, {"lc": ""}, {"extra": None}, {"label": "person"},
    {"box": [100, 100, 400]}, {"box": [100, 100, 400, 900, 1000]},
    {"box": [100.0, 100, 400, 900]}, {"box": [True, 100, 400, 900]},
    {"box": [100, 100, 1001, 900]}, {"box": [400, 100, 100, 900]},
    {"box": None},
])
def test_person_bad_values_are_rejected_without_repair(updates: dict[str, Any]) -> None:
    with pytest.raises(RichLabelProtocolError):
        _convert({"people": [_person(**updates)]}, "people")


@pytest.mark.parametrize("updates", [
    {"label": "p"}, {"label": "person"}, {"label": " Pedestrian "},
    {"label": None}, {"label": ""}, {"c": 5}, {"c": "x" * 25},
    {"box": [1, 2, 3]}, {"box": [1, 2, 3, 4, 5]}, {"a": None},
])
def test_item_bad_values_and_person_aliases_are_rejected(
    updates: dict[str, Any],
) -> None:
    with pytest.raises(RichLabelProtocolError):
        _convert({"items": [_item(**updates)]}, "items")


def test_entity_limits_include_combined_count() -> None:
    assert len(_convert({"people": [_person()] * 4, "items": [_item()] * 8})["o"]) == 12
    for payload, scope in (
        ({"people": [_person()] * 13}, "people"),
        ({"items": [_item()] * 9}, "items"),
        ({"people": [_person()] * 5, "items": [_item()] * 8}, "all"),
    ):
        with pytest.raises(RichLabelProtocolError):
            _convert(payload, scope)


@pytest.mark.parametrize("payload,scope", [
    ({"people": []}, "all"), ({"people": [], "items": []}, "people"),
    ({"people": [], "items": []}, "items"), ({"o": []}, "all"),
    ({"people": None}, "people"), ({"people": {}}, "people"),
    ({"people": [None]}, "people"), ({"people": [[]]}, "people"),
    ([], "all"), (None, "all"),
])
def test_wrong_scope_or_payload_shape_is_rejected(payload: Any, scope: str) -> None:
    with pytest.raises(RichLabelProtocolError):
        _convert(payload, scope)


@pytest.mark.parametrize("text", [
    '{"people":[],"people":[],"items":[]}',
    '{"items":[{"label":"chair","box":[1,2,3,4],"c":null,"c":"red"}]}',
    '{"items":[{"label":"chair","box":[1,2,3,NaN],"c":null}]}',
    '```json\n{"items":[]}\n```', 'private upstream error',
])
def test_malformed_json_has_a_safe_generic_error(text: str) -> None:
    with pytest.raises(RichLabelProtocolError) as error:
        object_wire_to_compact(text, "items")
    assert str(error.value) == (
        "rich annotation does not satisfy the named-object wire contract"
    )


@pytest.mark.parametrize("scope", ["all", "people", "items"])
def test_prompt_carries_original_image_and_explicit_coordinate_axes(scope: str) -> None:
    jpeg = b"original-jpeg"
    messages = build_object_wire_messages(jpeg, scope)
    prompt = messages[1]["content"][0]["text"]
    image = messages[1]["content"][1]["image_url"]["url"]
    assert base64.b64decode(image.partition(",")[2]) == jpeg
    assert "x is horizontal" in prompt
    assert "y is vertical" in prompt
    if scope != "items":
        assert "image edges" in prompt
        assert "all visible people FIRST" in prompt or scope == "people"
        assert "a,g,d,s,uc,lc,b,us,ls,gl,ht,bt,ho" in prompt


def test_invalid_scope_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_object_wire_response_format("invalid")
    with pytest.raises(ValueError):
        build_object_wire_messages(b"image", "invalid")
    with pytest.raises(RichLabelProtocolError):
        object_wire_to_compact('{"people":[]}', "invalid")
