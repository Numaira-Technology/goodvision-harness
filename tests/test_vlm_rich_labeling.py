from __future__ import annotations

import base64
import json
from typing import Any

import cv2
import numpy as np
import pytest

from goodvision_harness import vlm_rich_labeling as vlm
from goodvision_harness.aiclub import AIClubClient, AIClubSettings
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


def _reply(objects: list[dict[str, Any]]) -> dict[str, Any]:
    return {"choices": [{"message": {"content": json.dumps({"objects": objects})}}]}


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

    def post(client: AIClubClient, payload: dict[str, Any]) -> dict[str, Any]:
        captured.append({"key": client._api_key, **payload})
        return _reply([_person()])

    monkeypatch.setattr(AIClubClient, "_post_json", post)
    client = vlm.VLMRichLabelClient(
        AIClubSettings(), model="configured-vlm", api_key=lambda: key[0],
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


def test_tracks_follow_overlap_not_model_order_and_reset_between_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = _person()
    right = _person(bbox_2d=[600, 100, 900, 900])
    replies = iter([
        _reply([left, right]), _reply([right, left]), _reply([]),
        _reply([left]), _reply([left]), _reply([left]),
    ])
    monkeypatch.setattr(AIClubClient, "_post_json", lambda *_: next(replies))
    client = vlm.VLMRichLabelClient(AIClubSettings(), model="vlm", api_key="test")

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


def test_dry_run_never_requires_credentials_or_calls_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("dry run must not construct an API client")

    monkeypatch.setattr(vlm, "AIClubClient", unexpected)
    client = vlm.VLMRichLabelClient(
        AIClubSettings(), model="vlm", api_key=None, dry_run=lambda: True,
    )
    assert client.infer(
        _jpeg(), stream_session_id="dry", frame_sequence=1, timestamp_seconds=0,
    ).objects == ()


def test_coordinator_keeps_video_fail_open_on_malformed_api_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(AIClubClient, "_post_json", lambda *_: _reply([
        _person(bbox_2d=[100, 100, 50, 50]),
    ]))
    client = vlm.VLMRichLabelClient(AIClubSettings(), model="vlm", api_key="test")
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
    monkeypatch.setattr(AIClubClient, "_post_json", lambda *_: response)
    client = vlm.VLMRichLabelClient(AIClubSettings(), model="vlm", api_key="test")
    with pytest.raises(RichLabelProtocolError, match="truncated"):
        client.infer(
            _jpeg(), stream_session_id="live", frame_sequence=1, timestamp_seconds=0,
        )


def test_response_observer_receives_result_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _reply([])
    response["choices"][0]["finish_reason"] = "length"
    observed: list[Any] = []
    monkeypatch.setattr(AIClubClient, "_post_json", lambda *_: response)
    client = vlm.VLMRichLabelClient(
        AIClubSettings(), model="vlm", api_key="test",
        response_observer=observed.append,
    )
    with pytest.raises(RichLabelProtocolError, match="truncated"):
        client.infer(
            _jpeg(), stream_session_id="live", frame_sequence=1, timestamp_seconds=0,
        )
    assert observed == [response]


def test_configured_request_parameters_preserve_rich_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    def post(_client: Any, payload: Any) -> Any:
        captured.append(payload)
        return _reply([])

    monkeypatch.setattr(AIClubClient, "_post_json", post)
    client = vlm.VLMRichLabelClient(
        AIClubSettings(), model="vlm", api_key="test", max_tokens=3000,
        request_parameters={"response_format": {"type": "json_object"}},
    )
    client.infer(
        _jpeg(), stream_session_id="live", frame_sequence=1, timestamp_seconds=0,
    )
    assert captured[0]["response_format"] == {"type": "json_object"}
    assert captured[0]["model"] == "vlm"
    assert captured[0]["max_tokens"] == 3000
    assert len(captured[0]["messages"][1]["content"]) == 2


@pytest.mark.parametrize("key", ["model", "messages", "max_tokens", "stream"])
def test_request_parameters_cannot_replace_fixed_fields(key: str) -> None:
    with pytest.raises(ValueError, match="fixed field"):
        vlm.VLMRichLabelClient(
            AIClubSettings(), model="vlm", request_parameters={key: "override"},
        )
