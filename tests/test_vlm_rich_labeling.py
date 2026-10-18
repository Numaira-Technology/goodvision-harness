from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from typing import Any

import cv2
import jsonschema
import numpy as np
import pytest

from goodvision_harness import vlm_rich_labeling as vlm
from goodvision_harness.model_api import ModelAPIClient, ModelAPISettings
from goodvision_harness.rich_labeling import (
    RichLabelCoordinator,
    RichLabelProtocolError,
)


def _jpeg() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((180, 320, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _person(**updates: Any) -> dict[str, Any]:
    return {
        "label": "person", "bbox_2d": [100, 100, 400, 900], "confidence": 0.95,
        "attributes": {
            "estimated_age_range": "18-60", "estimated_gender": "male-presenting",
            "estimated_direction": "Side", "wearing_glasses": True,
            "sleeve_type": "ShortSleeve", "lower_styles": ["Trousers"],
            "upper_color": "white", "lower_color": "black",
            "estimated_age_confidence": 0.8,
        },
        **updates,
    }


def _unknown_compact_attributes() -> dict[str, None]:
    return {key: None for key in (
        "a", "g", "d", "s", "uc", "lc", "b", "us", "ls", "gl", "ht", "bt", "ho",
    )}


def _reply(objects: list[dict[str, Any]]) -> dict[str, Any]:
    return {"choices": [{"message": {"content": json.dumps({"objects": objects})}}]}


def _mock_completion(monkeypatch: pytest.MonkeyPatch, responder: Any) -> None:
    async def stream(_pool: Any, **request: Any) -> dict[str, Any]:
        return responder({
            "model": request["model"], "messages": request["messages"],
            "api_key": request["api_key"], **request["request_parameters"],
        })

    monkeypatch.setattr(vlm.httpx, "AsyncClient", _AsyncPool)
    monkeypatch.setattr(vlm, "complete_stream", stream)


def test_parse_preserves_paddle_attributes_and_normalizes_coordinates() -> None:
    payload = {"objects": [_person(), {
        "label": "Chair", "bbox_2d": [500, 300, 900, 1000], "attributes": {},
    }]}
    objects = vlm.parse_rich_annotation("```json\n" + json.dumps(payload) + "\n```")
    assert objects[0].bbox.x1 == 0.1
    assert objects[0].bbox.y2 == 0.9
    assert objects[0].attributes["estimated_direction"] == "Side"
    assert objects[0].attributes["lower_styles"] == ("Trousers",)
    assert objects[0].attributes["annotation_source"] == "vlm"
    assert objects[1].label == "chair"
    assert objects[1].confidence is None


@pytest.mark.parametrize("item", [
    _person(bbox_2d=[1, 2, 1001, 999]),
    _person(bbox_2d=[400, 100, 100, 900]),
    _person(bbox_2d=[False, 100, 400, 900]),
    _person(bbox_2d=[0, 0, float("nan"), 1000]),
    _person(confidence=1.2),
    _person(attributes={"wearing_hat": "false"}),
    _person(attributes={"estimated_age_range": "25"}),
    _person(attributes={"lower_styles": "Trousers"}),
])
def test_invalid_model_objects_are_rejected(item: dict[str, Any]) -> None:
    with pytest.raises(RichLabelProtocolError):
        vlm.parse_rich_annotation(json.dumps({"objects": [item]}))


@pytest.mark.parametrize("payload", [
    "bad json", "[]", '{"objects": null}', '{"events": []}',
    json.dumps({"objects": [_person()] * 65}),
])
def test_invalid_or_excessive_response_is_rejected(payload: str) -> None:
    with pytest.raises(RichLabelProtocolError):
        vlm.parse_rich_annotation(payload)


def test_uncertain_and_unrecognized_attributes_are_not_published() -> None:
    objects = vlm.parse_rich_annotation(json.dumps({"objects": [_person(attributes={
        "estimated_age_range": None, "estimated_gender": "unknown",
        "annotation_source": "arbitrary", "person_id": "invented-id",
        "photo_url": "https://invalid.example", "wearing_hat": False,
    })]}))
    assert dict(objects[0].attributes) == {
        "annotation_source": "vlm", "wearing_hat": False,
    }


def test_client_uses_separate_prompt_key_and_downscaled_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    key = ["first-key"]

    def post(payload: dict[str, Any]) -> dict[str, Any]:
        captured.append({"key": payload["api_key"], **payload})
        return _reply([_person()])

    _mock_completion(monkeypatch, post)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="configured-vlm", api_key=lambda: key[0],
        max_image_side=160, max_tokens=3000,
    )
    result = client.infer(
        _jpeg(), stream_session_id="stream-a", frame_sequence=17, timestamp_seconds=2.5,
    )
    assert result.stream_session_id == "stream-a"
    assert result.frame_sequence == 17
    assert result.timestamp_seconds == 2.5
    assert result.objects[0].track_id == "P0001"
    assert captured[0]["model"] == "configured-vlm"
    assert captured[0]["max_tokens"] == 3000
    assert captured[0]["temperature"] == 0
    content = captured[0]["messages"][1]["content"]
    assert "Include people even when there is no safety event" in content[0]["text"]
    data = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1])
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[:2] == (90, 160)
    key[0] = "second-key"
    client.infer(
        _jpeg(), stream_session_id="stream-a", frame_sequence=18, timestamp_seconds=3.5,
    )
    assert captured[1]["key"] == "second-key"
    assert client.api_requests == 2
    client.close()


def test_tracks_follow_overlap_not_model_order_and_reset_between_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = _person()
    right = _person(bbox_2d=[600, 100, 900, 900])
    replies = iter([
        _reply([left, right]), _reply([right, left]), _reply([]),
        _reply([left]), _reply([left]), _reply([left]),
    ])
    _mock_completion(monkeypatch, lambda _: next(replies))
    client = vlm.VLMRichLabelClient(ModelAPISettings(), model="vlm", api_key="test")

    def infer(sequence: int, timestamp: float, session: str = "a") -> Any:
        return client.infer(
            _jpeg(), stream_session_id=session, frame_sequence=sequence,
            timestamp_seconds=timestamp,
        )

    assert [o.track_id for o in infer(1, 1).objects] == ["P0001", "P0002"]
    assert [o.track_id for o in infer(2, 2).objects] == ["P0002", "P0001"]
    assert infer(3, 3).objects == ()
    assert infer(4, 4).objects[0].track_id == "P0003"
    assert infer(5, 20).objects[0].track_id == "P0004"  # expired spatial association
    assert infer(1, 1, "b").objects[0].track_id == "P0001"
    client.close()


def test_dry_run_never_requires_credentials_or_calls_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("dry run must not construct an API client")

    monkeypatch.setattr(vlm, "ModelAPIClient", unexpected)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key=None, dry_run=lambda: True,
    )
    assert client.infer(
        _jpeg(), stream_session_id="dry", frame_sequence=1, timestamp_seconds=0,
    ).objects == ()


def test_coordinator_keeps_video_fail_open_on_malformed_api_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_completion(monkeypatch, lambda _: _reply([
        _person(bbox_2d=[100, 100, 50, 50]),
    ]))
    client = vlm.VLMRichLabelClient(ModelAPISettings(), model="vlm", api_key="test")
    coordinator = RichLabelCoordinator(client)
    try:
        assert coordinator.submit(
            _jpeg(), stream_session_id="live", frame_sequence=1, timestamp_seconds=0,
        )
        assert coordinator.wait_until_idle()
        assert coordinator.latest() is None
        assert coordinator.stats()["failed"] == 1
        assert coordinator.stats()["last_error"] == "rich-label inference failed"
    finally:
        coordinator.close()


def test_truncated_completion_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _reply([])
    response["choices"][0]["finish_reason"] = "length"
    _mock_completion(monkeypatch, lambda _: response)
    client = vlm.VLMRichLabelClient(ModelAPISettings(), model="vlm", api_key="test")
    with pytest.raises(RichLabelProtocolError, match="truncated"):
        client.infer(
            _jpeg(), stream_session_id="live", frame_sequence=1, timestamp_seconds=0,
        )
    client.close()


def test_response_observer_receives_result_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _reply([])
    response["choices"][0]["finish_reason"] = "length"
    observed: list[Any] = []
    _mock_completion(monkeypatch, lambda _: response)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="test",
        response_observer=observed.append,
    )
    with pytest.raises(RichLabelProtocolError, match="truncated"):
        client.infer(
            _jpeg(), stream_session_id="live", frame_sequence=1, timestamp_seconds=0,
        )
    assert observed == [response]
    client.close()


def test_configured_request_parameters_preserve_rich_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    def post(payload: Any) -> Any:
        captured.append(payload)
        return _reply([])

    _mock_completion(monkeypatch, post)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="test", max_tokens=3000,
        request_parameters={"response_format": {"type": "json_object"}},
    )
    client.infer(
        _jpeg(), stream_session_id="live", frame_sequence=1, timestamp_seconds=0,
    )
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["model"] == "vlm"
    assert captured[0]["max_tokens"] == 3000
    assert len(captured[0]["messages"][1]["content"]) == 2
    client.close()


@pytest.mark.parametrize("key", ["model", "messages", "max_tokens", "stream"])
def test_request_parameters_cannot_replace_fixed_fields(key: str) -> None:
    with pytest.raises(ValueError, match="fixed field"):
        vlm.VLMRichLabelClient(
            ModelAPISettings(), model="vlm", request_parameters={key: "override"},
        )


def test_compact_rows_restore_paddle_attributes_without_inventing_confidence() -> None:
    objects = vlm.parse_rich_annotation(json.dumps({"o": [
        ["p", 100, 100, 400, 900, {
            "a": "a", "g": "m", "d": "s", "s": "l",
            "uc": "White", "lc": "black", "b": "b", "us": "pl",
            "ls": ["tr", "st"], "gl": 1, "ht": 0, "bt": 0, "ho": 1,
        }],
        ["Chair", 500, 300, 900, 1000, {"c": "Blue"}],
    ]}))

    person, chair = objects
    assert person.label == "person"
    assert person.bbox.x1 == 0.1
    assert person.bbox.y2 == 0.9
    assert person.confidence is None
    assert dict(person.attributes) == {
        "annotation_source": "vlm",
        "estimated_age_range": "18-60",
        "estimated_gender": "male-presenting",
        "estimated_direction": "Side",
        "sleeve_type": "LongSleeve",
        "upper_color": "white",
        "lower_color": "black",
        "bag_type": "Backpack",
        "upper_style": "UpperPlaid",
        "lower_styles": ("Trousers", "LowerStripe"),
        "wearing_glasses": True,
        "wearing_hat": False,
        "wearing_boots": False,
        "holding_object": True,
    }
    assert chair.label == "chair"
    assert chair.confidence is None
    assert dict(chair.attributes) == {"annotation_source": "vlm", "color": "blue"}


@pytest.mark.parametrize("attributes", [None, {}, {
    "a": None, "g": "unknown", "d": None, "s": None, "gl": None, "q": None,
}])
def test_compact_unknown_attributes_stay_unknown(attributes: Any) -> None:
    objects = vlm.parse_rich_annotation(json.dumps({"o": [
        ["p", 100, 100, 400, 900, attributes],
        ["table", 500, 500, 900, 900],
    ]}))
    for item in objects:
        assert item.confidence is None
        assert dict(item.attributes) == {"annotation_source": "vlm"}


def test_compact_reported_confidence_and_legacy_attributes_are_preserved() -> None:
    person, = vlm.parse_rich_annotation(json.dumps({"o": [[
        "p", 100, 100, 400, 900, {
            "q": 0.83, "a": "o", "g": "f", "d": "b", "s": "s",
            "b": "n", "ls": ["sk"], "estimated_age_confidence": 0.6,
        },
    ]]}))
    assert person.confidence == 0.83
    assert person.attributes["estimated_age_range"] == ">60"
    assert person.attributes["estimated_gender"] == "female-presenting"
    assert person.attributes["estimated_direction"] == "Back"
    assert person.attributes["sleeve_type"] == "ShortSleeve"
    assert person.attributes["bag_type"] == "No bag"
    assert person.attributes["lower_styles"] == ("Skirt&Dress",)
    assert person.attributes["estimated_age_confidence"] == 0.6


@pytest.mark.parametrize("confidence", [None, 0, 0.83, 1])
@pytest.mark.parametrize("include_q", [False, True])
def test_compact_seven_item_rows_preserve_explicit_confidence(
    confidence: float | None, include_q: bool,
) -> None:
    attributes: dict[str, Any] = {"a": "18-60", "uc": "white"}
    if include_q:
        attributes["q"] = confidence
    person, = vlm.parse_rich_annotation(json.dumps({"o": [[
        "p", 100, 100, 400, 900, confidence, attributes,
    ]]}))
    assert person.confidence == confidence
    assert person.attributes["estimated_age_range"] == "18-60"
    assert person.attributes["upper_color"] == "white"
    assert person.bbox.x1 == 0.1
    assert person.bbox.y2 == 0.9


@pytest.mark.parametrize("confidence", [None, 0.8])
def test_compact_seven_item_rows_accept_null_attributes(confidence: float | None) -> None:
    chair, = vlm.parse_rich_annotation(json.dumps({"o": [[
        "chair", 100, 100, 400, 900, confidence, None,
    ]]}))
    assert chair.confidence == confidence
    assert dict(chair.attributes) == {"annotation_source": "vlm"}


@pytest.mark.parametrize("tail", [
    [True, {}], [False, {}], [-0.1, {}], [1.1, {}], [float("nan"), {}],
    [float("inf"), {}], ["0.9", {}], [{}, {}], [None, []], [None, "ignored"],
    [0.9, {"q": 0.8}], [None, {"q": 0.8}], [0.8, {"q": None}],
    [0, {"q": False}], [None, {"q": "unknown"}], [None, {}, "ignored"],
])
def test_invalid_compact_seven_item_rows_are_rejected(tail: list[Any]) -> None:
    with pytest.raises(RichLabelProtocolError):
        vlm.parse_rich_annotation(json.dumps({"o": [[
            "p", 100, 100, 400, 900, *tail,
        ]]}))


@pytest.mark.parametrize("key,value,expected", [
    ("a", "<18", "<18"), ("a", "18-60", "18-60"), ("a", ">60", ">60"),
    ("d", "front", "Front"), ("d", "side", "Side"), ("d", "back", "Back"),
    ("d", "Side", "Side"),
    ("s", "short", "ShortSleeve"), ("s", "long", "LongSleeve"),
    ("s", "LongSleeve", "LongSleeve"),
    ("b", "none", "No bag"), ("b", "No bag", "No bag"),
    ("b", "ShoulderBag", "ShoulderBag"),
    ("g", "female-presenting", "female-presenting"),
    ("us", "UpperPlaid", "UpperPlaid"),
    ("ls", ["Trousers", "LowerStripe"], ("Trousers", "LowerStripe")),
])
def test_compact_accepts_known_aliases_and_canonical_values(
    key: str, value: Any, expected: Any,
) -> None:
    person, = vlm.parse_rich_annotation(json.dumps({"o": [[
        "p", 100, 100, 400, 900, {key: value},
    ]]}))
    assert person.attributes[vlm.COMPACT_ATTRIBUTE_NAMES[key]] == expected


def test_compact_none_upper_style_is_unreported() -> None:
    person, = vlm.parse_rich_annotation(json.dumps({"o": [[
        "p", 100, 100, 400, 900, {"us": "none", "uc": "white"},
    ]]}))
    assert dict(person.attributes) == {
        "annotation_source": "vlm", "upper_color": "white",
    }
    assert person.confidence is None


@pytest.mark.parametrize("rows", [
    None,
    [["p", 100, 100, 400]],
    [["p", 100, 100, 400, 900, {}, 0.9]],
    [["p", 100, 100, 400, 900, ["a", "m"]]],
    [["p", 100, 100, 400, 900, {"a": "adult"}]],
    [["p", 100, 100, 400, 900, {"d": "left"}]],
    [["p", 100, 100, 400, 900, {"s": "medium"}]],
    [["p", 100, 100, 400, 900, {"b": "purse"}]],
    [["p", 100, 100, 400, 900, {"us": "plain"}]],
    [["p", 100, 100, 400, 900, {"ls": "tr"}]],
    [["p", 100, 100, 400, 900, {"ls": ["pants"]}]],
    [["p", 100, 100, 400, 900, {"gl": "0"}]],
    [["p", 100, 100, 400, 900, {"ht": 2}]],
    [["p", 100, 100, 400, 900, {"q": True}]],
    [["p", 100, 100, 400, 900, {"q": 2}]],
    [["p", 400, 100, 100, 900]],
    [["p", 100, 100, 1001, 900]],
    [["p", False, 100, 400, 900]],
    [["p", 100, 100, 400, 900]] * 13,
    [_person()],
])
def test_invalid_compact_rows_fail_closed(rows: Any) -> None:
    with pytest.raises(RichLabelProtocolError):
        vlm.parse_rich_annotation(json.dumps({"o": rows}))


def test_compact_empty_scene_and_mixed_envelopes() -> None:
    assert vlm.parse_rich_annotation('{"o":[]}') == ()
    with pytest.raises(RichLabelProtocolError):
        vlm.parse_rich_annotation('{"o":[],"objects":[]}')


@pytest.mark.parametrize("reply_format", ["objects", "compact"])
def test_compact_client_accepts_both_response_formats_and_keeps_source_metadata(
    monkeypatch: pytest.MonkeyPatch,
    reply_format: str,
) -> None:
    captured: list[dict[str, Any]] = []
    content = (
        {"objects": [_person(confidence=None)]}
        if reply_format == "objects"
        else {"o": [["p", 100, 100, 400, 900, {"d": "s"}]]}
    )

    def post(payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(payload)
        return {"choices": [{"message": {"content": json.dumps(content)}}]}

    _mock_completion(monkeypatch, post)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="test", wire_format="compact",
        max_tokens=1024,
        request_parameters={
            "provider": {"only": ["deepinfra"], "allow_fallbacks": False}
        },
    )
    result = client.infer(
        _jpeg(), stream_session_id="compact-test", frame_sequence=7,
        timestamp_seconds=3.5,
    )

    assert result.stream_session_id == "compact-test"
    assert result.frame_sequence == 7
    assert result.timestamp_seconds == 3.5
    assert result.objects[0].track_id == "P0001"
    assert result.objects[0].attributes["estimated_direction"] == "Side"
    assert result.objects[0].confidence is None
    request = captured[0]
    assert request["model"] == "vlm"
    assert request["max_tokens"] == 1024
    assert request["provider"] == {"only": ["deepinfra"], "allow_fallbacks": False}
    prompt = request["messages"][1]["content"][0]["text"]
    # The complete few-shot example must itself satisfy the strict parser.
    example, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"o":'):])
    example_objects = vlm.parse_rich_annotation(json.dumps(example))
    assert [item.label for item in example_objects] == ["person", "chair"]
    assert example_objects[0].attributes["estimated_direction"] == "Side"
    client.close()


@pytest.mark.parametrize("wire_format", ["auto", None, False])
def test_unknown_wire_format_is_rejected(wire_format: Any) -> None:
    with pytest.raises(ValueError, match="wire_format"):
        vlm.VLMRichLabelClient(ModelAPISettings(), model="vlm", wire_format=wire_format)


@pytest.mark.parametrize("scope,region,labels", [
    ("all", None, ["person", "chair"]),
    ("people", None, ["person"]),
    ("people", (0, 500), ["person"]),
    ("people", (500, 1000), ["person"]),
    ("items", None, ["chair"]),
])
def test_scoped_compact_prompt_has_only_applicable_valid_examples(
    scope: str, region: tuple[float, float] | None, labels: list[str],
) -> None:
    jpeg = _jpeg()
    messages = vlm.build_compact_rich_annotation_messages(
        jpeg, scope=scope, person_region=region,
    )
    prompt = messages[1]["content"][0]["text"]
    example, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"o":'):])
    objects = vlm.parse_rich_annotation(json.dumps(example))
    assert [item.label for item in objects] == labels
    schema = vlm.build_compact_rich_annotation_response_format(scope)
    jsonschema.validate(example, schema["json_schema"]["schema"])
    if scope == "items":
        assert "age appearance" not in prompt
        assert "gender presentation" not in prompt
        assert "sleeves" not in prompt
    else:
        person = objects[0]
        assert person.attributes["upper_color"] == "white"
        assert person.attributes["lower_color"] == "black"
        assert person.attributes["lower_styles"] == ("Trousers",)
        if region is not None:
            low, high = region
            assert low <= (person.bbox.x1 + person.bbox.x2) * 500 < high
            assert f"{low:g} <= (x1+x2)/2 < {high:g}" in prompt
            assert "FULL-image coordinates" in prompt
    data_url = messages[1]["content"][1]["image_url"]["url"]
    assert base64.b64decode(data_url.split(",", 1)[1]) == jpeg


@pytest.mark.parametrize("scope,region", [
    ("invalid", None), ("people", (0, 0)), ("people", (800, 200)),
    ("people", (-1, 500)), ("people", (0, 1001)), ("people", (0, float("inf"))),
    ("people", (float("nan"), 1000)), ("people", (False, 500)),
    ("people", (0, "500")), ("people", (0, 500, 1000)),
    ("all", (0, 500)), ("items", (0, 500)),
])
def test_invalid_compact_scope_or_region_is_rejected(scope: Any, region: Any) -> None:
    with pytest.raises(ValueError):
        vlm.build_compact_rich_annotation_messages(
            b"image", scope=scope, person_region=region,
        )
    with pytest.raises(ValueError):
        vlm.VLMRichLabelClient(
            ModelAPISettings(), model="vlm", wire_format="compact",
            scope=scope, person_region=region,
        )


def test_scoped_client_passes_person_region_to_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def post(payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(payload)
        return _reply([])

    _mock_completion(monkeypatch, post)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="test", wire_format="compact",
        scope="people", person_region=(500, 1000),
    )
    client.infer(_jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1)
    prompt = captured[0]["messages"][1]["content"][0]["text"]
    assert "500 <= (x1+x2)/2 < 1000" in prompt
    with pytest.raises(ValueError, match="requires compact"):
        vlm.VLMRichLabelClient(ModelAPISettings(), model="vlm", scope="people")
    client.close()


class _AsyncPool:
    def __init__(self, **_kwargs: Any) -> None:
        self.loop = asyncio.get_running_loop()
        self.close_count = 0

    async def aclose(self) -> None:
        assert asyncio.get_running_loop() is self.loop
        await asyncio.sleep(0)
        self.close_count += 1


def _stream_reply(rows: list[Any], *, tokens: int = 10) -> dict[str, Any]:
    return {
        "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"o": rows}),
        }}],
        "provider": "DeepInfra",
        "usage": {"prompt_tokens": 100, "completion_tokens": tokens,
                  "total_tokens": 100 + tokens,
                  "completion_tokens_details": {"reasoning_tokens": 0}},
        "_timing": {"elapsed_seconds": tokens / 100},
    }


@pytest.mark.parametrize("wire_format", ["objects", "compact"])
def test_single_reuses_pool_with_current_key_and_preserves_stream_metrics(
    monkeypatch: pytest.MonkeyPatch, wire_format: str,
) -> None:
    pools = []
    calls = []
    observed = []
    key = ["first-key"]
    response = _stream_reply([["p", 100, 100, 400, 900, {
        "a": "a", "g": "m", "d": "s", "s": "s", "uc": "white",
        "lc": "black", "ls": ["tr"], "gl": 1, "ht": 0,
    }]])
    response["usage"]["prompt_tokens_details"] = {"cached_tokens": 64}
    response["_timing"]["first_content_token_seconds"] = 0.02

    def pool(**kwargs: Any) -> _AsyncPool:
        result = _AsyncPool(**kwargs)
        pools.append(result)
        return result

    async def stream(http_client: Any, **kwargs: Any) -> Any:
        calls.append((http_client, kwargs))
        return response

    monkeypatch.setattr(vlm.httpx, "AsyncClient", pool)
    monkeypatch.setattr(vlm, "complete_stream", stream)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(timeout_seconds=2, max_retries=9), model="vlm",
        api_key=lambda: key[0], wire_format=wire_format,
        max_tokens=1024, response_observer=observed.append,
        request_parameters={"reasoning": {"enabled": False}},
    )
    try:
        first = client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
        )
        key[0] = "second-key"
        second = client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=2, timestamp_seconds=2,
        )
        assert len(pools) == 1
        assert all(pool is pools[0] for pool, _ in calls)
        assert [call["api_key"] for _, call in calls] == ["first-key", "second-key"]
        assert all(0 < call["deadline_seconds"] <= 2 for _, call in calls)
        assert all(call["request_parameters"] == {
            "temperature": 0.0, "reasoning": {"enabled": False}, "max_tokens": 1024,
        } for _, call in calls)
        assert client.api_requests == 2
        assert first.objects[0].track_id == second.objects[0].track_id
        assert len(first.objects[0].attributes) == 10
        assert observed == [response, response]
        assert observed[1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 64
        assert observed[1]["_timing"]["first_content_token_seconds"] == 0.02
    finally:
        client.close()
    assert pools[0].close_count == 1
    assert pools[0].loop.is_closed()


def test_single_stream_failure_has_no_retry_and_preserves_previous_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def stream(_pool: Any, **_kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise vlm.ModelAPIError("model stream socket timed out")
        return _stream_reply([])

    monkeypatch.setattr(vlm.httpx, "AsyncClient", _AsyncPool)
    monkeypatch.setattr(vlm, "complete_stream", stream)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(max_retries=9), model="vlm", api_key="key",
    )
    try:
        original = client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
        )
        with pytest.raises(vlm.ModelAPIError, match="timed out"):
            client.infer(
                _jpeg(), stream_session_id="test", frame_sequence=2,
                timestamp_seconds=2,
            )
        assert client._previous is original
        assert client.api_requests == len(calls) == 2
        client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=3, timestamp_seconds=3,
        )
        assert client.api_requests == 3
    finally:
        client.close()


@pytest.mark.parametrize("execution_mode", ["single", "parallel"])
def test_frame_deadline_includes_image_preparation(
    monkeypatch: pytest.MonkeyPatch, execution_mode: str,
) -> None:
    encode = vlm.encode_jpeg

    def slow_encode(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.02)
        return encode(*args, **kwargs)

    monkeypatch.setattr(vlm, "encode_jpeg", slow_encode)
    monkeypatch.setattr(vlm.httpx, "AsyncClient", lambda **_: pytest.fail("no pool"))
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(timeout_seconds=0.01), model="vlm", api_key="key",
        wire_format="compact", execution_mode=execution_mode,
    )
    try:
        with pytest.raises(vlm.ModelAPIError, match="wall deadline"):
            client.infer(
                _jpeg(), stream_session_id="test", frame_sequence=1,
                timestamp_seconds=1,
            )
        assert client.api_requests == 0
    finally:
        client.close()


def test_parallel_reuses_pool_and_frame_key_and_restores_crop_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pools = []
    calls = []
    observed = []
    key_calls = []

    def pool(**kwargs: Any) -> _AsyncPool:
        result = _AsyncPool(**kwargs)
        pools.append(result)
        return result

    def key() -> str:
        key_calls.append(1)
        return f"frame-key-{len(key_calls)}"

    async def stream(http_client: Any, **kwargs: Any) -> Any:
        index = len(calls) % 3
        calls.append((http_client, kwargs))
        await asyncio.sleep(0)
        assert len(calls) % 3 == 0  # All three requests started before any finished.
        if index == 0:
            rows = [
                ["p", 100, 100, 300, 900, {"uc": "white"}],
                ["p", 1000 * 2 / 3, 100, 1000, 900, {"uc": "blue"}],
            ]
        elif index == 1:
            rows = [
                ["p", 0, 100, 1000 / 3, 900, {"uc": "red"}],
                ["p", 500, 100, 800, 900, {"uc": "black"}],
            ]
        else:
            rows = [["chair", 20 * i, 500, 20 * i + 10, 900] for i in range(10)]
        return _stream_reply(rows, tokens=(index + 1) * 10)

    monkeypatch.setattr(vlm.httpx, "AsyncClient", pool)
    monkeypatch.setattr(vlm, "complete_stream", stream)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(timeout_seconds=5), model="vlm", api_key=key,
        wire_format="compact", execution_mode="parallel", max_image_side=320,
        max_tokens=400, response_observer=observed.append,
        request_parameters={"temperature": 0.7, "top_p": 0.8},
    )
    try:
        first = client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
        )
        second = client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=2, timestamp_seconds=2,
        )
        people = [item for item in first.objects if item.label == "person"]
        assert len(people) == 3
        assert people[0].bbox.x1 == pytest.approx(0.06)
        assert people[0].bbox.x2 == pytest.approx(0.18)
        assert people[1].bbox.x1 == pytest.approx(0.4)
        assert people[1].bbox.x2 == pytest.approx(0.6)
        assert people[1].attributes["upper_color"] == "red"
        assert people[2].bbox.x1 == pytest.approx(0.7)
        assert people[2].bbox.x2 == pytest.approx(0.88)
        assert len(first.objects) == 11  # Three people, at most eight items.
        assert [item.track_id for item in second.objects] == [
            item.track_id for item in first.objects
        ]
        assert len(pools) == 1
        assert client.api_requests == 6
        assert len(key_calls) == 2
        assert [kwargs["api_key"] for _, kwargs in calls] == (
            ["frame-key-1"] * 3 + ["frame-key-2"] * 3
        )
        assert all(http_client is pools[0] for http_client, _ in calls)
        sizes = []
        for _, kwargs in calls[:3]:
            data_url = kwargs["messages"][1]["content"][1]["image_url"]["url"]
            decoded = cv2.imdecode(np.frombuffer(
                base64.b64decode(data_url.split(",", 1)[1]), dtype=np.uint8,
            ), cv2.IMREAD_COLOR)
            sizes.append(decoded.shape[:2])
            assert 0 < kwargs["deadline_seconds"] <= 5
            assert kwargs["request_parameters"] == {
                "max_tokens": 400, "temperature": 0.7, "top_p": 0.8,
            }
        assert sizes == [(180, 192), (180, 192), (180, 320)]
        assert len(observed) == 2
        assert observed[0]["provider"] == "DeepInfra"
        assert observed[0]["usage"]["completion_tokens"] == 60
        assert observed[0]["usage"]["total_tokens"] == 360
        assert observed[0]["_timing"]["max_part_seconds"] == 0.3
        assert set(observed[0]["parts"]) == {"people_left", "people_right", "items"}
        assert len(vlm.parse_rich_annotation(
            ModelAPIClient.assistant_text(observed[0]),
        )) == 11
    finally:
        client.close()
        client.close()
    assert pools[0].close_count == 1
    assert pools[0].loop.is_closed()


def _merge_person(x1: int, x2: int, color: str) -> Any:
    return vlm.parse_rich_annotation(json.dumps({"o": [[
        "p", x1, 100, x2, 900, {"uc": color},
    ]]}))[0]


@pytest.mark.parametrize("side,x1,x2", [
    ("left", 510, 590), ("right", 454, 541),
])
def test_crop_merge_keeps_unmatched_person_across_center(
    side: str, x1: int, x2: int,
) -> None:
    person = _merge_person(x1, x2, "brown")
    result = vlm._merge_crop_people(
        [person] if side == "left" else [], [person] if side == "right" else [],
    )
    assert result == [person]


def test_crop_merge_matches_true_duplicate_and_preserves_chosen_metadata() -> None:
    left = _merge_person(400, 600, "blue")
    right = _merge_person(400, 600, "red")
    result = vlm._merge_crop_people([left], [right])
    assert result == [right]
    assert result[0].bbox.x1 == 0.4
    assert result[0].attributes["upper_color"] == "red"
    assert result[0].confidence is None


def test_crop_merge_retains_adjacent_people_and_same_crop_duplicates() -> None:
    left = _merge_person(350, 480, "blue")
    right = _merge_person(460, 600, "red")
    assert vlm._merge_crop_people([left], [right]) == [left, right]
    assert vlm._merge_crop_people([left, left], []) == [left, left]


def test_crop_merge_pairs_once_and_prefers_highest_overlap() -> None:
    left = _merge_person(400, 600, "blue")
    near = _merge_person(410, 610, "gray")
    exact = _merge_person(400, 600, "red")
    assert vlm._merge_crop_people([left], [near, exact]) == [exact, near]


def test_crop_merge_iou_threshold_is_inclusive() -> None:
    left = _merge_person(250, 750, "blue")
    half_overlap = _merge_person(250, 500, "red")
    below_half = _merge_person(250, 495, "gray")
    assert len(vlm._merge_crop_people([left], [half_overlap])) == 1
    assert vlm._merge_crop_people([left], [below_half]) == [left, below_half]


def test_parallel_restores_and_keeps_right_only_person_left_of_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def stream(_client: Any, **_kwargs: Any) -> Any:
        index = len(calls)
        calls.append(index)
        return _stream_reply([
            ["p", 90, 0, 235, 288, {"a": "a", "d": "b", "uc": "brown"}],
        ] if index == 1 else [])

    monkeypatch.setattr(vlm.httpx, "AsyncClient", _AsyncPool)
    monkeypatch.setattr(vlm, "complete_stream", stream)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="test", wire_format="compact",
        execution_mode="parallel", structured_output=True,
    )
    try:
        snapshot = client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
        )
        person, = snapshot.objects
        assert person.track_id == "P0001"
        assert person.bbox.x1 == pytest.approx(0.454)
        assert person.bbox.x2 == pytest.approx(0.541)
        assert (person.bbox.x1 + person.bbox.x2) / 2 == pytest.approx(0.4975)
        assert person.bbox.y1 == 0
        assert person.bbox.y2 == 0.288
        assert person.attributes["estimated_age_range"] == "18-60"
        assert person.attributes["estimated_direction"] == "Back"
        assert person.attributes["upper_color"] == "brown"
    finally:
        client.close()


@pytest.mark.parametrize("failure", ["invalid", "cancelled", "timeout"])
def test_parallel_failure_awaits_all_parts_and_recovers_without_partial_results(
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    observed = []
    observed_parts = []
    completed = []
    calls = []
    phase = ["success"]
    pool = []

    def create_pool(**kwargs: Any) -> _AsyncPool:
        result = _AsyncPool(**kwargs)
        pool.append(result)
        return result

    async def stream(_http_client: Any, **_kwargs: Any) -> Any:
        index = len(calls) % 3
        current_phase = phase[0]
        calls.append((current_phase, index))
        try:
            await asyncio.sleep(0)
            if current_phase == "failure":
                if failure == "timeout" or index == 2:
                    await asyncio.Event().wait()
                if index == 1:
                    if failure == "cancelled":
                        raise asyncio.CancelledError()
                    response = _stream_reply([])
                    response["choices"][0]["message"]["content"] = "invalid"
                    return response
            return _stream_reply([])
        finally:
            await asyncio.sleep(0)
            completed.append((current_phase, index))

    monkeypatch.setattr(vlm.httpx, "AsyncClient", create_pool)
    monkeypatch.setattr(vlm, "complete_stream", stream)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(timeout_seconds=0.2), model="vlm", api_key="key",
        wire_format="compact", execution_mode="parallel",
        response_observer=observed.append,
        part_response_observer=lambda name, response: observed_parts.append(
            (phase[0], name, response)
        ),
    )
    try:
        original = client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
        )
        phase[0] = "failure"
        expected = RichLabelProtocolError if failure == "invalid" else vlm.ModelAPIError
        with pytest.raises(expected):
            client.infer(
                _jpeg(), stream_session_id="test", frame_sequence=2, timestamp_seconds=2,
            )
        assert client._previous is original
        assert len(observed) == 1
        assert sorted(i for p, i in completed if p == "failure") == [0, 1, 2]
        assert not client._part_tasks
        failed_parts = {
            name: response for part_phase, name, response in observed_parts
            if part_phase == "failure"
        }
        if failure == "invalid":
            assert set(failed_parts) == {"people_left", "people_right"}
            assert ModelAPIClient.assistant_text(failed_parts["people_right"]) == "invalid"
            assert failed_parts["people_right"]["choices"][0]["finish_reason"] == "stop"
        elif failure == "cancelled":
            assert set(failed_parts) == {"people_left"}
        else:
            assert not failed_parts
        phase[0] = "recovered"
        recovered = client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=3, timestamp_seconds=3,
        )
        assert client._previous is recovered
        assert len(observed) == 2
        assert len(pool) == 1
        assert client.api_requests == 9
        assert {
            name for part_phase, name, _ in observed_parts if part_phase == "recovered"
        } == {"people_left", "people_right", "items"}
    finally:
        client.close()
    assert pool[0].close_count == 1


@pytest.mark.parametrize("execution_mode", ["single", "parallel"])
def test_close_on_worker_loop_awaits_stream_cleanup(
    monkeypatch: pytest.MonkeyPatch, execution_mode: str,
) -> None:
    pools = []
    entered = []
    cleaned = []

    def pool(**kwargs: Any) -> _AsyncPool:
        result = _AsyncPool(**kwargs)
        pools.append(result)
        return result

    async def stream(_http_client: Any, **_kwargs: Any) -> Any:
        index = len(entered)
        entered.append(index)
        if index == 0:
            asyncio.get_running_loop().call_soon(client.close)
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.append(index)

    monkeypatch.setattr(vlm.httpx, "AsyncClient", pool)
    monkeypatch.setattr(vlm, "complete_stream", stream)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="key",
        wire_format="compact", execution_mode=execution_mode,
    )
    with pytest.raises(vlm.ModelAPIError, match="cancelled"):
        client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
        )
    assert sorted(cleaned) == ([0, 1, 2] if execution_mode == "parallel" else [0])
    assert pools[0].close_count == 1
    assert pools[0].loop.is_closed()
    assert client._previous is None
    client.close()
    with pytest.raises(vlm.ModelAPIError, match="closed"):
        client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=2, timestamp_seconds=2,
        )


@pytest.mark.parametrize("execution_mode", ["single", "parallel"])
def test_dry_run_does_not_create_loop_pool_or_read_key(
    monkeypatch: pytest.MonkeyPatch, execution_mode: str,
) -> None:
    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("dry run must not create network resources or resolve credentials")

    monkeypatch.setattr(vlm.httpx, "AsyncClient", unexpected)
    monkeypatch.setattr(vlm.asyncio, "new_event_loop", unexpected)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key=unexpected, dry_run=True,
        wire_format="compact", execution_mode=execution_mode,
    )
    assert client.infer(
        _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
    ).objects == ()
    assert client.api_requests == 0
    client.close()


@pytest.mark.parametrize("kwargs", [
    {"execution_mode": "unknown"}, {"execution_mode": "parallel"},
    {"execution_mode": "parallel", "wire_format": "compact", "scope": "people"},
])
def test_invalid_execution_mode_combination_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        vlm.VLMRichLabelClient(ModelAPISettings(), model="vlm", **kwargs)


def test_parallel_people_take_priority_within_twelve_object_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def stream(_http_client: Any, **_kwargs: Any) -> Any:
        index = len(calls)
        calls.append(index)
        if index == 0:
            rows = [["p", i * 100, 100, i * 100 + 50, 900] for i in range(5)]
        elif index == 1:
            rows = []
        else:
            rows = [["chair", i * 100, 500, i * 100 + 50, 900] for i in range(8)]
        return _stream_reply(rows)

    monkeypatch.setattr(vlm.httpx, "AsyncClient", _AsyncPool)
    monkeypatch.setattr(vlm, "complete_stream", stream)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="key",
        wire_format="compact", execution_mode="parallel",
    )
    try:
        result = client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
        )
        assert [item.label for item in result.objects] == ["person"] * 5 + ["chair"] * 7
    finally:
        client.close()


@pytest.mark.parametrize("execution_mode", ["single", "parallel"])
def test_missing_key_does_not_create_pool_or_count_requests(
    monkeypatch: pytest.MonkeyPatch, execution_mode: str,
) -> None:
    monkeypatch.setattr(vlm, "environment_api_key", lambda _: "")
    monkeypatch.setattr(vlm.httpx, "AsyncClient", lambda **_: pytest.fail("no pool"))
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key=None,
        wire_format="compact", execution_mode=execution_mode,
    )
    with pytest.raises(vlm.ModelAPIError, match="key is missing"):
        client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
        )
    assert client.api_requests == 0
    assert client._loop is None
    client.close()


@pytest.mark.parametrize("execution_mode", ["single", "parallel"])
def test_close_during_decode_prevents_late_client_creation(
    monkeypatch: pytest.MonkeyPatch, execution_mode: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    errors = []
    decode = cv2.imdecode

    def blocked_decode(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(2)
        return decode(*args, **kwargs)

    monkeypatch.setattr(vlm.cv2, "imdecode", blocked_decode)
    monkeypatch.setattr(vlm.httpx, "AsyncClient", lambda **_: pytest.fail("no pool"))
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="key",
        wire_format="compact", execution_mode=execution_mode,
    )

    def infer() -> None:
        try:
            client.infer(
                _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=infer, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        client.close()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], vlm.ModelAPIError)
    assert client.api_requests == 0
    assert client._loop is None


@pytest.mark.parametrize("execution_mode", ["single", "parallel"])
def test_concurrent_close_at_frame_completion_returns_and_closes_pool_once(
    monkeypatch: pytest.MonkeyPatch, execution_mode: str,
) -> None:
    completed = threading.Event()
    release = threading.Event()
    errors = []
    pools = []
    observed = []
    new_loop = asyncio.new_event_loop

    def loop_factory() -> Any:
        loop = new_loop()
        run = loop.run_until_complete

        def run_to_boundary(awaitable: Any) -> Any:
            result = run(awaitable)
            if awaitable is client._frame_task:
                completed.set()
                assert release.wait(2)
            return result

        loop.run_until_complete = run_to_boundary
        return loop

    def pool(**kwargs: Any) -> _AsyncPool:
        result = _AsyncPool(**kwargs)
        pools.append(result)
        return result

    async def stream(*_args: Any, **_kwargs: Any) -> Any:
        return _stream_reply([])

    monkeypatch.setattr(vlm.asyncio, "new_event_loop", loop_factory)
    monkeypatch.setattr(vlm.httpx, "AsyncClient", pool)
    monkeypatch.setattr(vlm, "complete_stream", stream)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="key",
        wire_format="compact", execution_mode=execution_mode,
        response_observer=observed.append,
    )

    def infer() -> None:
        try:
            client.infer(
                _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
            )
        except BaseException as exc:
            errors.append(exc)

    def close() -> None:
        try:
            client.close()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=infer, daemon=True)
    closers = [threading.Thread(target=close, daemon=True) for _ in range(3)]
    worker.start()
    try:
        assert completed.wait(2)
        for closer in closers:
            closer.start()
        deadline = time.monotonic() + 2
        while not client._closed and time.monotonic() < deadline:
            time.sleep(0.001)
        assert client._closed
    finally:
        release.set()
        worker.join(2)
        for closer in closers:
            if closer.ident is not None:
                closer.join(2)
    assert not worker.is_alive()
    assert all(not closer.is_alive() for closer in closers)
    assert all(isinstance(error, vlm.ModelAPIError) for error in errors)
    assert pools[0].close_count == 1
    assert pools[0].loop.is_closed()
    assert not client._part_tasks
    # Single-response diagnostics run before parsing, before this close began.
    assert len(observed) == (1 if execution_mode == "single" else 0)
    assert client._previous is None


@pytest.mark.parametrize("scope", ["all", "people", "items"])
def test_compact_strict_schema_accepts_its_scope_and_null_unknowns(scope: str) -> None:
    response_format = vlm.build_compact_rich_annotation_response_format(scope)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    person = ["p", 100, 100, 400, 900, {
        **_unknown_compact_attributes(),
        "a": "a", "g": "m", "d": "s", "s": "l", "b": None,
        "uc": "white", "lc": None, "ls": ["tr"], "gl": 1, "ht": 0,
    }]
    items = [["chair", 500, 400, 900, 1000], ["desk", 0, 500, 400, 900, {"c": None}]]
    rows = [person] if scope == "people" else items if scope == "items" else [person, *items]
    jsonschema.validate({"o": rows}, schema)
    jsonschema.validate({"o": []}, schema)
    assert len(vlm.parse_rich_annotation(json.dumps({"o": rows}))) == len(rows)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"o": [rows[0]] * (9 if scope == "items" else 13)}, schema)


@pytest.mark.parametrize("row", [
    ["p", 100, 100, 400, {"a": "a"}],
    ["p", 100, 100, 400, 900],
    ["p", 100, 100, 400, 900, {"a": "a"}, {"a": "o"}],
    ["p", 100, 100, 400, 900, None, {"a": "a"}],
    ["p", [100, 100, 400, 900], {"a": "a"}],
    ["p", 100.5, 100, 400, 900, {}],
    ["p", True, 100, 400, 900, {}],
    ["p", 100, 100, 1001, 900, {}],
    ["p", 100, 100, 400, 900, {"a": "adult"}],
    ["p", 100, 100, 400, 900, {"ls": "tr"}],
    ["p", 100, 100, 400, 900, {"ls": ["pants"]}],
    ["p", 100, 100, 400, 900, {"gl": "yes"}],
    ["p", 100, 100, 400, 900, {"extra": "ignored"}],
])
@pytest.mark.parametrize("scope", ["people", "all"])
def test_compact_schema_rejects_missing_boxes_double_attributes_and_bad_values(
    scope: str, row: list[Any],
) -> None:
    schema = vlm.build_compact_rich_annotation_response_format(scope)["json_schema"]["schema"]
    if len(row) == 6 and isinstance(row[-1], dict):
        row = [*row[:-1], {**_unknown_compact_attributes(), **row[-1]}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"o": [row]}, schema)


def test_compact_schema_keeps_local_positive_area_validation() -> None:
    schema = vlm.build_compact_rich_annotation_response_format("people")["json_schema"]["schema"]
    payload = {"o": [["p", 400, 100, 100, 900, _unknown_compact_attributes()]]}
    jsonschema.validate(payload, schema)  # Bounds alone cannot compare x1 and x2.
    with pytest.raises(RichLabelProtocolError, match="positive area"):
        vlm.parse_rich_annotation(json.dumps(payload))


@pytest.mark.parametrize("scope", ["people", "all"])
def test_compact_schema_allows_thirteen_unknown_values_without_invented_attributes(
    scope: str,
) -> None:
    response_format = vlm.build_compact_rich_annotation_response_format(scope)
    schema = response_format["json_schema"]["schema"]
    payload = {"o": [["p", 100, 100, 400, 900, _unknown_compact_attributes()]]}
    jsonschema.validate(payload, schema)
    person, = vlm.parse_rich_annotation(json.dumps(payload))
    assert dict(person.attributes) == {"annotation_source": "vlm"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"o": [["p", 100, 100, 400, 900, None]]}, schema)


@pytest.mark.parametrize("scope", ["people", "all"])
@pytest.mark.parametrize("missing_key", [
    "a", "g", "d", "s", "uc", "lc", "b", "us", "ls", "gl", "ht", "bt", "ho",
])
def test_compact_schema_rejects_each_missing_person_attribute(
    scope: str, missing_key: str,
) -> None:
    response_format = vlm.build_compact_rich_annotation_response_format(scope)
    attributes = _unknown_compact_attributes()
    del attributes[missing_key]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"o": [["p", 100, 100, 400, 900, attributes]]},
            response_format["json_schema"]["schema"],
        )


@pytest.mark.parametrize("scope", ["all", "people", "items"])
def test_single_structured_compact_request_uses_matching_scope_schema(
    monkeypatch: pytest.MonkeyPatch, scope: str,
) -> None:
    captured = []

    def post(payload: Any) -> Any:
        captured.append(payload)
        return _stream_reply([])

    _mock_completion(monkeypatch, post)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="test", scope=scope,
        wire_format="compact", structured_output=True,
        request_parameters={"response_format": {"type": "json_object"}, "temperature": 0.7},
    )
    client.infer(_jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1)
    assert captured[0]["response_format"] == vlm.build_compact_rich_annotation_response_format(scope)
    assert captured[0]["temperature"] == 0.7
    assert client.request_parameters["response_format"] == {"type": "json_object"}
    client.close()


def test_parallel_structured_requests_use_people_people_items_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    async def stream(_client: Any, **kwargs: Any) -> Any:
        captured.append(kwargs["request_parameters"])
        return _stream_reply([])

    monkeypatch.setattr(vlm.httpx, "AsyncClient", _AsyncPool)
    monkeypatch.setattr(vlm, "complete_stream", stream)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="test",
        wire_format="compact", execution_mode="parallel", structured_output=True,
        request_parameters={"response_format": {"type": "json_object"}},
    )
    try:
        client.infer(_jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1)
        assert [params["response_format"] for params in captured] == [
            vlm.build_compact_rich_annotation_response_format(scope)
            for scope in ("people", "people", "items")
        ]
    finally:
        client.close()


@pytest.mark.parametrize("kwargs,exception", [
    ({"structured_output": True}, ValueError),
    ({"structured_output": "true", "wire_format": "compact"}, TypeError),
    ({"structured_output": 1, "wire_format": "compact"}, TypeError),
])
def test_invalid_structured_output_options_are_rejected(
    kwargs: dict[str, Any], exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        vlm.VLMRichLabelClient(ModelAPISettings(), model="vlm", **kwargs)


@pytest.mark.parametrize("wire_format", ["objects", "compact"])
def test_prompt_cache_keeps_only_stable_instruction_prefix_across_changed_images(
    wire_format: str,
) -> None:
    builder = (
        vlm.build_compact_rich_annotation_messages
        if wire_format == "compact" else vlm.build_rich_annotation_messages
    )
    original = builder(b"frame-one")
    cached = vlm._cacheable_annotation_messages(original)
    changed = vlm._cacheable_annotation_messages(builder(b"frame-two"))
    assert cached[:3] == changed[:3]
    assert cached[3] != changed[3]
    assert cached[0] == original[0]
    assert cached[1]["content"] == [{
        **original[1]["content"][0], "cache_control": {"type": "ephemeral"},
    }]
    assert cached[2] == {"role": "assistant", "content": "Ready."}
    assert cached[3] == {"role": "user", "content": [original[1]["content"][1]]}
    assert "cache_control" not in original[1]["content"][0]
    assert "cache_control" not in cached[3]["content"][0]


@pytest.mark.parametrize("execution_mode", ["single", "parallel"])
@pytest.mark.parametrize("prompt_cache", [False, True])
def test_client_applies_optional_prompt_cache_to_every_request(
    monkeypatch: pytest.MonkeyPatch, execution_mode: str, prompt_cache: bool,
) -> None:
    calls = []

    async def stream(_pool: Any, **request: Any) -> Any:
        calls.append(request)
        return _stream_reply([])

    monkeypatch.setattr(vlm.httpx, "AsyncClient", _AsyncPool)
    monkeypatch.setattr(vlm, "complete_stream", stream)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="vlm", api_key="key", wire_format="compact",
        execution_mode=execution_mode, prompt_cache=prompt_cache,
    )
    try:
        client.infer(
            _jpeg(), stream_session_id="test", frame_sequence=1, timestamp_seconds=1,
        )
    finally:
        client.close()
    assert len(calls) == (3 if execution_mode == "parallel" else 1)
    for request in calls:
        messages = request["messages"]
        assert len(messages) == (4 if prompt_cache else 2)
        if prompt_cache:
            assert messages[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
            assert len(messages[1]["content"]) == 1
            assert messages[3]["content"][0]["type"] == "image_url"
        else:
            assert "cache_control" not in messages[1]["content"][0]
            assert messages[1]["content"][1]["type"] == "image_url"


@pytest.mark.parametrize("prompt_cache", [None, "true", 1])
def test_prompt_cache_requires_a_boolean(prompt_cache: Any) -> None:
    with pytest.raises(TypeError, match="prompt_cache must be boolean"):
        vlm.VLMRichLabelClient(
            ModelAPISettings(), model="vlm", prompt_cache=prompt_cache,
        )
