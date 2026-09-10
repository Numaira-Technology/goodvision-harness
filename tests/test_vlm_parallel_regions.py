from __future__ import annotations

import base64
import json
from typing import Any

import cv2
import jsonschema
import numpy as np
import pytest

from goodvision_harness import vlm_rich_labeling as vlm
from goodvision_harness.model_api import ModelAPIError, ModelAPISettings
from goodvision_harness.vlm_validation import validate_complete_compact_annotation

ATTRIBUTES = dict.fromkeys((
    "a", "g", "d", "s", "uc", "lc", "b", "us", "ls", "gl", "ht", "bt", "ho",
))


class _Pool:
    closed = False

    async def aclose(self) -> None:
        self.closed = True


def _jpeg() -> bytes:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :100, 0] = 255
    frame[:, 100:, 2] = 255
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def _response(rows: list[list[Any]]) -> dict[str, Any]:
    return {
        "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"o": rows}),
        }}],
        "usage": {"completion_tokens": 10},
    }


def _request_parts(request: dict[str, Any]) -> tuple[str, str]:
    blocks = [
        block for message in request["messages"]
        if isinstance(message["content"], list) for block in message["content"]
    ]
    prompt = next(block["text"] for block in blocks if block["type"] == "text")
    image = next(
        block["image_url"]["url"] for block in blocks if block["type"] == "image_url"
    )
    return prompt, image


@pytest.mark.parametrize("prompt_cache", [False, True])
def test_full_frame_shares_pixels_and_preserves_native_coordinates_and_recall(
    monkeypatch: pytest.MonkeyPatch, prompt_cache: bool,
) -> None:
    requests: list[dict[str, Any]] = []
    images: list[str] = []
    encode_count = 0
    encode = vlm.encode_jpeg
    observed: list[dict[str, Any]] = []
    pool = _Pool()

    def count_encode(*args: Any, **kwargs: Any) -> Any:
        nonlocal encode_count
        encode_count += 1
        return encode(*args, **kwargs)

    async def complete(_pool: _Pool, **request: Any) -> dict[str, Any]:
        requests.append(request)
        prompt, image = _request_parts(request)
        images.append(image)
        boundary = ["p", 450, 100, 550, 900, dict(ATTRIBUTES)]
        if "0 <= (x1+x2)/2 < 500" in prompt:
            rows = [["p", 100, 100, 300, 900, dict(ATTRIBUTES)], boundary]
        elif "500 <= (x1+x2)/2 < 1000" in prompt:
            rows = [
                ["p", 700, 100, 900, 900, dict(ATTRIBUTES)], boundary,
                # A model's region mistake remains visible for quality scoring.
                ["p", 310, 100, 410, 900, dict(ATTRIBUTES)],
            ]
        else:
            assert "Only major non-person objects" in prompt
            rows = [["chair", 600, 500, 800, 950]]
        return _response(rows)

    monkeypatch.setattr(vlm, "encode_jpeg", count_encode)
    monkeypatch.setattr(vlm, "create_vlm_http_client", lambda **kwargs: pool)
    monkeypatch.setattr(vlm, "complete_stream", complete)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="test-model", api_key="fake-key",
        wire_format="compact", execution_mode="parallel",
        parallel_strategy="full_frame",
        structured_output=True, require_complete=True, prompt_cache=prompt_cache,
        max_image_side=160, response_observer=observed.append,
    )
    try:
        snapshot = client.infer(
            _jpeg(), stream_session_id="test-session", frame_sequence=7,
            timestamp_seconds=1.5,
        )
        assert encode_count == 1
        assert client.api_requests == len(requests) == 3
        assert len(set(images)) == 1
        pixels = cv2.imdecode(
            np.frombuffer(base64.b64decode(images[0].split(",", 1)[1]), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert pixels.shape[:2] == (80, 160)
        assert pixels[40, 20, 0] > 200 and pixels[40, 140, 2] > 200
        people = [obj for obj in snapshot.objects if obj.label == "person"]
        boxes = sorted((person.bbox.x1, person.bbox.x2) for person in people)
        assert boxes == [(0.1, 0.3), (0.31, 0.41), (0.45, 0.55), (0.7, 0.9)]
        chair = next(obj for obj in snapshot.objects if obj.label == "chair")
        assert (chair.bbox.x1, chair.bbox.x2) == (0.6, 0.8)
        assert snapshot.frame_sequence == 7 and snapshot.timestamp_seconds == 1.5
        assert len(observed) == 1
        assert set(observed[0]["parts"]) == {"people_left", "people_right", "items"}
        assert observed[0]["usage"]["completion_tokens"] == 30
        names = [request["request_parameters"]["response_format"]["json_schema"]["name"]
                 for request in requests]
        assert sorted(names) == ["rich_items", "rich_people", "rich_people"]
    finally:
        client.close()
    assert pool.closed


def test_full_frame_failure_never_publishes_partial_parts_and_can_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fail = [True]
    observed: list[dict[str, Any]] = []

    async def complete(_pool: _Pool, **request: Any) -> dict[str, Any]:
        prompt, _image = _request_parts(request)
        if "Only major non-person objects" in prompt:
            if fail[0]:
                raise ModelAPIError("model stream failed")
            return _response([["chair", 600, 500, 800, 950]])
        return _response([["p", 100, 100, 300, 900, dict(ATTRIBUTES)]])

    monkeypatch.setattr(vlm, "create_vlm_http_client", lambda **kwargs: _Pool())
    monkeypatch.setattr(vlm, "complete_stream", complete)
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="test-model", api_key="fake-key",
        wire_format="compact", execution_mode="parallel",
        parallel_strategy="full_frame", structured_output=True,
        require_complete=True, response_observer=observed.append,
    )
    try:
        with pytest.raises(ModelAPIError):
            client.infer(
                _jpeg(), stream_session_id="test-session", frame_sequence=1,
                timestamp_seconds=1,
            )
        assert client.api_requests == 3
        assert client._previous is None and not observed
        assert not client._part_tasks
        fail[0] = False
        snapshot = client.infer(
            _jpeg(), stream_session_id="test-session", frame_sequence=2,
            timestamp_seconds=2,
        )
        assert len(snapshot.objects) == 2
        assert len(observed) == 1 and client.api_requests == 6
    finally:
        client.close()


@pytest.mark.parametrize("settings", [
    {"parallel_strategy": "unknown"}, {"parallel_strategy": True},
    {"parallel_strategy": "full_frame"},
])
def test_invalid_parallel_strategy_is_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        vlm.VLMRichLabelClient(ModelAPISettings(), model="test-model", **settings)


def test_default_parallel_strategy_preserves_crops() -> None:
    client = vlm.VLMRichLabelClient(
        ModelAPISettings(), model="test-model", wire_format="compact",
        execution_mode="parallel",
    )
    assert client.parallel_strategy == "crops"
    client.close()


@pytest.mark.parametrize("region", [(0, 500), (500, 1000)])
def test_half_frame_example_satisfies_its_schema_and_complete_validator(
    region: tuple[int, int],
) -> None:
    messages = vlm.build_compact_rich_annotation_messages(
        b"test-image", scope="people", person_region=region,
    )
    prompt = messages[1]["content"][0]["text"]
    example_text = prompt.split("(structure only; do not copy values): ", 1)[1]
    example, _end = json.JSONDecoder().raw_decode(example_text)
    coordinates = example["o"][0][1:5]
    assert all(type(value) is int for value in coordinates)
    assert region[0] <= (coordinates[0] + coordinates[2]) / 2 < region[1]
    schema = vlm.build_compact_rich_annotation_response_format("people")[
        "json_schema"
    ]["schema"]
    jsonschema.Draft202012Validator(schema).validate(example)
    validate_complete_compact_annotation(json.dumps(example), "people")
