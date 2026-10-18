"""Validate the event contract without repairing or inventing model judgments."""

import json
from dataclasses import replace

import httpx
import pytest
from test_vlm_independent import config, job

from goodvision_harness import scene_model
from goodvision_harness.config import load_config
from goodvision_harness.event_sensitivity import LITTER_RULE, PPE_RULE
from goodvision_harness.grounding import SAFETY_EVENT_LABELS
from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.scene_model import _ResponseValidationError
from goodvision_harness.vlm_compact_model import (
    CompactGroundingInspector,
    EventJSONInspector,
    event_response_format,
)
from tools.vlm_benchmark.cloud import configure_event_json
from tools.vlm_multistream_benchmark import LABELS, ROOT


def reply(text, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": text}}]}


@pytest.mark.parametrize(
    "wrap", ["{}", "```json\n{}\n```", "```\n{}\n```", "```json\r\n{}\r\n```"]
)
def test_complete_json_preserves_each_judgment(wrap):
    request = job(config(), 1.2, 18)
    for value in (True, False, None):
        text = json.dumps({"smoking": value, "fall": False})
        votes = EventJSONInspector._parse(
            request, reply(" \n" + wrap.format(text) + "\n ")
        )
        assert votes[0].detected is value
        assert votes[1].detected is False
        assert all(not v.subject_ids and not v.subject_boxes for v in votes)


@pytest.mark.parametrize(
    "text",
    [
        '{"smoking": false}',
        '{"smoking": false, "fall": false, "extra": false}',
        '{"smoking": "false", "fall": false}',
        '{"smoking": 0, "fall": false}',
        '{"smoking": false, "fall": false,}',
        "{'smoking': False, 'fall': False}",
        '{"smoking": true, "smoking": false, "fall": false}',
        '{"smoking": NaN, "fall": false}',
        '{"smoking": false, "fall": false}{}',
        'Here is the answer: {"smoking": false, "fall": false}',
        '```json\n{"smoking": false, "fall": false}\n``` done',
        '```json\n{"smoking": false, "fall": false}',
        '```json\n```json\n{"smoking": false, "fall": false}\n```\n```',
        '```json\n{"smoking": false, "fall": fal\n```',
        "[]",
    ],
)
def test_no_partial_or_repaired_answer_is_accepted(text):
    with pytest.raises(_ResponseValidationError):
        EventJSONInspector._parse(job(config(), 1.2, 18), reply(text))


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_even_valid_json_requires_finished_completion(finish):
    with pytest.raises(_ResponseValidationError, match="completion_not_finished"):
        EventJSONInspector._parse(
            job(config(), 1.2, 18), reply('{"smoking": false, "fall": false}', finish)
        )


def test_fenced_json_does_not_relax_grounding_value_validation():
    with pytest.raises(_ResponseValidationError):
        CompactGroundingInspector._parse(
            job(config(), 1.2, 18), reply('```json\n{"no_events": "false"}\n```')
        )


@pytest.fixture
def base(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setenv("INFERENCE_MODEL", "qwen/qwen3.5-35b-a3b:nitro")
    return load_config(ROOT / "tools/vlm_benchmark/config.json")


def test_schema_requires_all_events_and_preserves_route_without_mutating_base(base):
    original = json.loads(json.dumps(dict(base.inference.request_parameters)))
    changed = configure_event_json(base, "schema")
    parameters = changed.inference.request_parameters
    schema = parameters["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"] == {
        "type": "object",
        "required": list(LABELS),
        "additionalProperties": False,
        "properties": {label: {"type": ["boolean", "null"]} for label in LABELS},
    }
    assert parameters["reasoning"] == original["reasoning"]
    assert parameters["provider"] == {
        **original["provider"],
        "require_parameters": True,
    }
    assert dict(base.inference.request_parameters) == original
    assert parameters["provider"] is not base.inference.request_parameters["provider"]
    assert configure_event_json(base, "prompt") is base
    unsupported = replace(base, aiclub=replace(base.model_api, provider="dashscope"))
    with pytest.raises(ValueError, match="unsupported"):
        configure_event_json(unsupported, "schema")
    for labels in ((), ("fall", "fall"), ("arbitrary",), ("stranger",)):
        with pytest.raises(ValueError):
            event_response_format(labels)


def test_schema_reaches_http_and_diagnostics_reset_on_failure(base, monkeypatch):
    base = configure_event_json(base, "schema")
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        if len(seen) == 2:
            raise httpx.ReadTimeout("private connection details")
        text = json.dumps(
            {label: None if label == "fall" else False for label in LABELS}
        )
        return httpx.Response(200, json=reply("```json\n" + text + "\n```"))

    monkeypatch.setattr(
        scene_model,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    inspector = EventJSONInspector(
        base.model_api,
        model=base.inference.model,
        api_key="test-only",
        request_parameters=base.inference.request_parameters,
        max_tokens=160,
    )
    request = job(replace(config(), labels=LABELS), 1.2, 18)
    try:
        votes = inspector.inspect(request)
        assert tuple(v.event_label for v in votes) == SAFETY_EVENT_LABELS
        assert next(v for v in votes if v.event_label == "fall").detected is None
        assert inspector.last_response_diagnostics["content_wrapper"] == "code_fence"
        assert inspector.last_http_timing["request_bytes"] > 0
        assert inspector.last_http_timing["response_read_seconds"] >= 0
        assert inspector.last_inspection_timing["request_prepare_seconds"] >= 0
        assert seen[0]["response_format"] == event_response_format(LABELS)
        assert seen[0]["provider"]["require_parameters"] is True
        assert seen[0]["max_tokens"] == 160
        prompt = seen[0]["messages"][0]["content"]
        assert PPE_RULE in prompt
        assert LITTER_RULE in prompt
        assert seen[0]["reasoning"] == base.inference.request_parameters["reasoning"]
        assert (
            len(
                [
                    p
                    for p in seen[0]["messages"][1]["content"]
                    if p["type"] == "image_url"
                ]
            )
            == 2
        )
        with pytest.raises(ModelAPIError) as failure:
            inspector.inspect(request)
        assert failure.value.code == "timeout"
        assert inspector.last_response_diagnostics == {}
        assert inspector.last_http_timing["response_read_seconds"] is None
        assert inspector.last_http_timing["response_bytes"] == 0
        assert "verdict_parse_seconds" not in inspector.last_inspection_timing
    finally:
        inspector.close()


def test_six_event_request_rejects_legacy_four_event_completion():
    request = job(replace(config(), labels=SAFETY_EVENT_LABELS), 1.2, 18)
    legacy = dict.fromkeys(("fight", "smoking", "fall", "weapon"), False)
    with pytest.raises(_ResponseValidationError, match="invalid_verdict"):
        EventJSONInspector._parse(request, reply(json.dumps(legacy)))
    complete = {**legacy, "ppe_violation": True, "ground_litter": None}
    votes = EventJSONInspector._parse(request, reply(json.dumps(complete)))
    assert {v.event_label: v.detected for v in votes} == complete


def test_response_diagnostics_preserve_only_public_provider_categories():
    from goodvision_harness.scene_model import _response_shape

    assert _response_shape({"choices": [], "provider": "DeepInfra"}) == {
        "choices_count": 0,
        "provider_category": "DeepInfra",
    }
    result = _response_shape({"choices": [], "provider": "private-provider-sentinel"})
    assert result["provider_category"] == "other"
    assert "private-provider-sentinel" not in str(result)
