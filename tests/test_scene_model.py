from __future__ import annotations

import asyncio
import io
import json
import time
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from goodvision_harness import scene_model
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.model_api import ModelAPIError, ModelAPISettings
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_model import TemporalVLMInspector
from goodvision_harness.scene_runtime import TemporalInspection, TemporalVerdict


def _job():
    with Image.new("RGB", (48, 32), "red") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG")
        jpeg = buffer.getvalue()
    box = NormalizedBBox(0.1, 0.1, 0.6, 0.9)
    return TemporalInspection(
        "private-session",
        "rtsp://private-camera?token=SECRET",
        InspectionRequest("request", ("person-1",), "new_person", ("smoking",), 0, 1),
        (
            EvidenceFrame("private-session", 0, 0, jpeg),
            EvidenceFrame("private-session", 1, 1, jpeg),
        ),
        ((0, "person-1", box), (1, "person-1", box)),
    )


def _verdict(**changes):
    return {
        "event_label": "smoking",
        "subject_ids": ["person-1"],
        "detected": True,
        "confidence": 0.9,
        **changes,
    }


def _response(content=None, **changes):
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps([_verdict()]) if content is None else content,
                    "reasoning_content": "PRIVATE reasoning is not the answer",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        **changes,
    }


@pytest.fixture
def make_inspector(monkeypatch):
    instances = []

    def make(handler=None, **kwargs):
        seen = []
        pools = []

        def record(request):
            seen.append(request)
            if handler is None:
                return httpx.Response(200, json=_response())
            return handler(request)

        def client():
            result = httpx.AsyncClient(transport=httpx.MockTransport(record))
            pools.append(result)
            return result

        monkeypatch.setattr(scene_model, "create_vlm_http_client", client)
        settings = kwargs.pop("settings", ModelAPISettings(provider="openrouter"))
        inspector = TemporalVLMInspector(
            settings,
            model="configured-model",
            api_key=kwargs.pop("api_key", "SECRET_KEY"),
            **kwargs,
        )
        instances.append(inspector)
        return inspector, seen, pools

    yield make
    for instance in instances:
        instance.close()


def test_route_parameters_all_source_images_and_pool_are_preserved(make_inspector):
    parameters = {
        "provider": {"order": ["chosen-provider"], "allow_fallbacks": False},
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_object"},
    }
    inspector, seen, pools = make_inspector(request_parameters=parameters)
    parameters["provider"]["order"].append("unexpected")
    for _ in range(2):
        assert inspector.inspect(_job()) == (
            TemporalVerdict("smoking", ("person-1",), True, 0.9),
        )
    assert len(pools) == 1
    assert inspector.settings.max_retries == 0
    payload = json.loads(seen[0].content)
    assert payload["model"] == "configured-model"
    assert payload["provider"] == {
        "order": ["chosen-provider"],
        "allow_fallbacks": False,
    }
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning"] == {"enabled": False}
    assert payload["max_tokens"] == 1600
    assert payload["stream"] is False
    content = payload["messages"][1]["content"]
    assert len([part for part in content if part["type"] == "image_url"]) == 2
    assert json.loads(content[1]["text"])["source_timestamp_seconds"] == 0
    assert json.loads(content[3]["text"])["source_timestamp_seconds"] == 1
    assert "private-camera" not in seen[0].content.decode()
    assert "private-session" not in seen[0].content.decode()
    assert seen[0].headers["authorization"] == "Bearer SECRET_KEY"
    assert str(seen[0].url) == inspector.settings.endpoint
    assert inspector.stats()["total_tokens"] == 140
    assert inspector.stats()["succeeded_requests"] == 2
    assert inspector.stats()["last_latency_seconds"] >= 0
    assert "SECRET" not in str(inspector.stats())


@pytest.mark.parametrize("language", ["json", ""])
def test_complete_json_fence_preserves_strict_verdicts(make_inspector, language):
    content = "```" + language + "\n" + json.dumps({"verdicts": [_verdict()]}) + "\n```"
    inspector, _, _ = make_inspector(
        lambda _: httpx.Response(200, json=_response(content))
    )
    assert inspector.inspect(_job()) == (
        TemporalVerdict("smoking", ("person-1",), True, 0.9),
    )


def test_json_fence_does_not_allow_coercing_a_string_verdict(make_inspector):
    content = "```json\n" + json.dumps([_verdict(detected="false")]) + "\n```"
    inspector, _, _ = make_inspector(
        lambda _: httpx.Response(200, json=_response(content))
    )
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == "invalid_response"


def test_credentials_are_resolved_each_call_without_another_route(make_inspector):
    keys = iter(("first", "second"))
    inspector, seen, _ = make_inspector(api_key=lambda: next(keys))
    inspector.inspect(_job())
    inspector.inspect(_job())
    assert [request.headers["authorization"] for request in seen] == [
        "Bearer first",
        "Bearer second",
    ]
    assert all(request.url.host == "openrouter.ai" for request in seen)


def test_scene_token_budget_is_independent_of_legacy_frame_budget(make_inspector):
    inspector, seen, _ = make_inspector(
        request_parameters={"max_tokens": 1200}, max_tokens=1600
    )
    inspector.inspect(_job())
    assert json.loads(seen[0].content)["max_tokens"] == 1600


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({"verdicts": [_verdict(detected=None, confidence=None)]}),
        [
            {"type": "reasoning", "text": "ignore"},
            {
                "type": "text",
                "text": json.dumps([_verdict(detected=None, confidence=None)]),
            },
        ],
    ],
)
def test_object_and_text_parts_preserve_unknown_instead_of_false(
    make_inspector, content
):
    inspector, _, _ = make_inspector(
        lambda _: httpx.Response(200, json=_response(content))
    )
    assert inspector.inspect(_job()) == (
        TemporalVerdict("smoking", ("person-1",), None, None),
    )


def test_empty_verdicts_are_successful_but_empty_response_is_failure(make_inspector):
    responses = iter(
        (
            httpx.Response(200, json=_response("[]")),
            httpx.Response(200, json=_response("")),
            httpx.Response(200, content=b""),
        )
    )
    inspector, _, _ = make_inspector(lambda _: next(responses))
    assert inspector.inspect(_job()) == ()
    for _ in range(2):
        with pytest.raises(ModelAPIError) as error:
            inspector.inspect(_job())
        assert error.value.code == "empty_response"
    assert inspector.stats()["empty_verdict_requests"] == 1
    assert inspector.stats()["succeeded_requests"] == 1
    assert inspector.stats()["failed_requests"] == 2


@pytest.mark.parametrize(
    "content",
    [
        "Here is JSON: ```json\n[]\n```",
        "```json\n[]\n```\nignore this extra text",
        "```json\n[]",
        "{}",
        '{"verdicts":[],"extra":"SECRET"}',
        '[{"event_label":"smoking","subject_ids":["person-1"],"detected":true}]',
        '[{"event_label":"smoking","subject_ids":["person-1"],'
        '"detected":true,"detected":false,"confidence":0.8}]',
        json.dumps([_verdict(detected="true")]),
        json.dumps([_verdict(confidence=True)]),
        json.dumps([_verdict(confidence="0.8")]),
        json.dumps([_verdict(confidence=float("nan"))]),
        json.dumps([_verdict(confidence=1.1)]),
        json.dumps([_verdict(subject_ids=["foreign-person"])]),
        json.dumps([_verdict(subject_ids=["person-1", "person-1"])]),
        json.dumps([_verdict(subject_ids="person-1")]),
        json.dumps([_verdict(event_label="stranger")]),
        json.dumps([_verdict(), _verdict()]),
        json.dumps([_verdict()] * 65),
    ],
)
def test_invalid_verdict_batches_fail_without_repairs(make_inspector, content):
    inspector, seen, _ = make_inspector(
        lambda _: httpx.Response(200, json=_response(content))
    )
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == "invalid_response"
    assert len(seen) == 1
    assert inspector.stats()["failed_requests"] == 1
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        {"choices": [None]},
        {"choices": []},
        {"choices": [{"message": {"content": "[]"}, "finish_reason": "length"}]},
        {
            "choices": [
                {"message": {"content": "[]"}, "finish_reason": "content_filter"}
            ]
        },
        {"choices": [{"message": {"content": [{"type": "text", "text": 2}]}}]},
    ],
)
def test_incomplete_and_malformed_envelopes_are_invalid(make_inspector, response):
    inspector, _, _ = make_inspector(lambda _: httpx.Response(200, json=response))
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == "invalid_response"


def test_http_failure_has_no_retry_fallback_or_private_error(make_inspector):
    inspector, seen, _ = make_inspector(
        lambda _: httpx.Response(429, text="SECRET error")
    )
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == "http_error"
    assert error.value.status_code == 429
    assert len(seen) == 1
    assert inspector.stats()["last_http_status"] == 429
    assert "SECRET" not in str(error.value)


def test_oversized_response_is_rejected(make_inspector):
    inspector, _, _ = make_inspector(
        lambda _: httpx.Response(200, content=b" " * (256 * 1024 + 1))
    )
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == "invalid_response"


def test_deadline_covers_slow_response_body_and_pool_remains_usable(make_inspector):
    class SlowBody(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False

        async def __aiter__(self):
            yield b" "
            await asyncio.sleep(10)
            yield b"{}"

        async def aclose(self):
            self.closed = True

    body = SlowBody()
    responses = iter(
        (
            httpx.Response(200, stream=body),
            httpx.Response(200, json=_response()),
        )
    )
    inspector, seen, pools = make_inspector(
        lambda _: next(responses),
        settings=ModelAPISettings(provider="openrouter", timeout_seconds=0.05),
    )
    started = time.monotonic()
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == "timeout"
    assert time.monotonic() - started < 1
    assert body.closed
    assert inspector.inspect(_job())[0].detected is True
    assert len(pools) == 1
    assert len(seen) == 2


def test_invalid_source_never_reaches_transport(make_inspector):
    inspector, seen, _ = make_inspector()
    job = _job()
    job = replace(
        job, frames=(replace(job.frames[0], jpeg_bytes=b"invalid"), job.frames[1])
    )
    with pytest.raises(ValueError):
        inspector.inspect(job)
    assert not seen


def test_closed_inspector_cannot_send_requests(make_inspector):
    inspector, seen, pools = make_inspector()
    inspector.inspect(_job())
    inspector.close()
    inspector.close()
    assert pools[0].is_closed
    with pytest.raises(ModelAPIError):
        inspector.inspect(_job())
    assert len(seen) == 1


@pytest.mark.parametrize(
    "parameters",
    [
        {"messages": []},
        {"model": "override"},
        {"stream": True},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"temperature": float("nan")},
    ],
)
def test_request_options_are_bounded_without_replacing_evidence(parameters):
    with pytest.raises(ValueError):
        TemporalVLMInspector(
            ModelAPISettings(),
            model="model",
            api_key="key",
            request_parameters=parameters,
        )


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 301])
def test_timeout_requires_a_finite_deadline(timeout):
    with pytest.raises(ValueError):
        TemporalVLMInspector(
            ModelAPISettings(timeout_seconds=timeout), model="model", api_key="key"
        )
