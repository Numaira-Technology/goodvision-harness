from __future__ import annotations

import asyncio
import base64
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
        client_options = kwargs.pop("client_options", {})

        def record(request):
            seen.append(request)
            if handler is None:
                return httpx.Response(200, json=_response())
            return handler(request)

        def client():
            result = httpx.AsyncClient(
                transport=httpx.MockTransport(record), **client_options,
            )
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
    prompt = payload["messages"][0]["content"]
    assert 'Return only a JSON object with one field "verdicts"' in prompt
    assert 'Return {"verdicts":[]}' in prompt
    assert "Return only a JSON array" not in prompt
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


def test_default_prompt_requests_one_array_contract_with_wrapper_parse_compatibility(
    make_inspector,
):
    inspector, seen, _ = make_inspector(
        lambda _: httpx.Response(200, json=_response(
            json.dumps({"verdicts": [_verdict()]}),
        )),
    )
    assert inspector.inspect(_job())[0].detected is True
    prompt = json.loads(seen[0].content)["messages"][0]["content"]
    assert "Return only a JSON array" in prompt
    assert 'field "verdicts"' not in prompt
    assert "also accepted" not in prompt


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


@pytest.mark.parametrize(
    "code,status",
    [("invalid_response", 502), ("empty_response", 502), ("connection", 502),
     ("timeout", 504), ("overloaded", 429), ("request_failed", 502)],
)
def test_companion_error_classification_survives_boundary(make_inspector, code, status):
    response = httpx.Response(status, json={"code": code, "detail": "PRIVATE_SECRET"})
    inspector, seen, _ = make_inspector(
        lambda _: response,
        settings=ModelAPISettings(
            provider="local", inspection_endpoint="http://companion/v1/inspections"
        ),
    )
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == code
    assert error.value.status_code == status
    assert inspector.stats()["last_error_code"] == code
    assert len(seen) == 1
    assert response.is_closed
    assert "PRIVATE_SECRET" not in str(error.value)


@pytest.mark.parametrize(
    "body",
    [b"not JSON PRIVATE_SECRET", b'[]', b'{"code": []}',
     b'{"code": "PRIVATE_SECRET"}', b'{"code": "connection", "detail": "' +
     b"x" * 4096 + b'"}'],
)
def test_companion_invalid_error_body_has_safe_fallback(make_inspector, body):
    inspector, _, _ = make_inspector(
        lambda _: httpx.Response(502, content=body),
        settings=ModelAPISettings(
            provider="local", inspection_endpoint="http://companion/v1/inspections"
        ),
    )
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == "http_error"
    assert error.value.status_code == 502
    assert "PRIVATE_SECRET" not in str(error.value)


def test_oversized_response_is_rejected(make_inspector):
    inspector, _, _ = make_inspector(
        lambda _: httpx.Response(200, content=b" " * (256 * 1024 + 1))
    )
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == "invalid_response"


@pytest.mark.parametrize(
    "content,stage,reason",
    [
        ("SECRET response prose", "verdict_json", "invalid_json_syntax"),
        ('{"verdicts":[],"verdicts":[]}', "verdict_json", "duplicate_json_field"),
        (
            json.dumps([_verdict(confidence=float("nan"))]),
            "verdict_json",
            "nonfinite_json_number",
        ),
        ("{}", "verdict_schema", "verdict_array_required_or_limit"),
        (json.dumps(["SECRET"]), "verdict_schema", "non_object"),
        (json.dumps([{"event_label": "smoking"}]), "verdict_schema", "missing_fields"),
        (json.dumps([_verdict(extra="SECRET")]), "verdict_schema", "extra_fields"),
        (
            json.dumps([_verdict(subject_ids="SECRET")]),
            "verdict_schema",
            "subject_ids_not_list",
        ),
        (
            json.dumps([_verdict(event_label=123)]),
            "verdict_schema",
            "event_label_not_string",
        ),
        (
            json.dumps([_verdict(subject_ids=["SECRET-person"])]),
            "verdict_grounding",
            "unknown_subject_id",
        ),
        (
            json.dumps([_verdict(subject_ids=[])]),
            "verdict_grounding",
            "empty_subject_ids",
        ),
        (
            json.dumps([_verdict(subject_ids=["person-1", "person-1"])]),
            "verdict_grounding",
            "duplicate_subject_id",
        ),
        (
            json.dumps([_verdict(event_label="SECRET-label")]),
            "verdict_grounding",
            "unrequested_event_label",
        ),
        (
            json.dumps([_verdict(detected="SECRET")]),
            "verdict_grounding",
            "invalid_detected_type",
        ),
        (
            json.dumps([_verdict(confidence="SECRET")]),
            "verdict_grounding",
            "invalid_confidence_type",
        ),
        (
            json.dumps([_verdict(confidence=2)]),
            "verdict_grounding",
            "invalid_confidence_range",
        ),
        (
            json.dumps([_verdict(), _verdict()]),
            "verdict_grounding",
            "duplicate_verdict",
        ),
    ],
)
def test_invalid_response_logs_safe_stage_and_reason_without_changing_rejection(
    make_inspector,
    monkeypatch,
    content,
    stage,
    reason,
):
    logs = []
    monkeypatch.setattr(
        scene_model, "log_timing", lambda name, **fields: logs.append((name, fields))
    )
    inspector, seen, _ = make_inspector(
        lambda _: httpx.Response(200, json=_response(content))
    )
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == "invalid_response"
    assert len(seen) == 1
    validation = [
        (name, fields) for name, fields in logs if name == "vlm_response_validation"
    ]
    assert len(validation) == 1
    name, fields = validation[0]
    assert name == "vlm_response_validation"
    assert fields["request_id"] == "request"
    assert fields["validation_stage"] == stage
    assert fields["validation_reason"] == reason
    assert fields["content_characters"] == len(content)
    assert "SECRET" not in str(logs)
    assert "person-1" not in str(logs)
    assert "private-camera" not in str(logs)
    assert "PRIVATE reasoning" not in str(logs)


@pytest.mark.parametrize(
    "response,stage,reason,code",
    [
        (
            httpx.Response(200, content=b"SECRET invalid JSON"),
            "envelope_json",
            "invalid_json_syntax",
            "invalid_response",
        ),
        (
            httpx.Response(200, json=[]),
            "envelope_schema",
            "envelope_not_object",
            "invalid_response",
        ),
        (
            httpx.Response(200, content=b" " * (256 * 1024 + 1)),
            "envelope_body",
            "response_size_limit",
            "invalid_response",
        ),
        (
            httpx.Response(200, content=b""),
            "envelope_body",
            "empty_response_body",
            "empty_response",
        ),
        (
            httpx.Response(200, json=_response("")),
            "assistant_content",
            "empty_assistant_text",
            "empty_response",
        ),
        (
            httpx.Response(200, json={"choices": []}),
            "envelope_schema",
            "missing_choices",
            "invalid_response",
        ),
        (
            httpx.Response(200, json={"choices": [{"finish_reason": "SECRET"}]}),
            "envelope_schema",
            "completion_not_finished",
            "invalid_response",
        ),
    ],
)
def test_envelope_failures_have_safe_diagnostics_and_preserve_error_code(
    make_inspector,
    monkeypatch,
    response,
    stage,
    reason,
    code,
):
    logs = []
    monkeypatch.setattr(
        scene_model,
        "log_timing",
        lambda name, **fields: (
            logs.append(fields) if name == "vlm_response_validation" else None
        ),
    )
    inspector, _, _ = make_inspector(lambda _: response)
    with pytest.raises(ModelAPIError) as error:
        inspector.inspect(_job())
    assert error.value.code == code
    try:
        decoded_response = response.json()
    except ValueError:
        decoded_response = None
    assert error.value.response_diagnostics == scene_model._response_shape(
        decoded_response, inspector.model
    )
    assert "SECRET" not in str(error.value.response_diagnostics)
    assert logs[0]["validation_stage"] == stage
    assert logs[0]["validation_reason"] == reason
    assert "SECRET" not in str(logs)


@pytest.mark.parametrize("initial_chunks", [(), (b" ", b"\n")])
def test_deadline_covers_slow_response_body_and_pool_remains_usable(
    make_inspector, monkeypatch, initial_chunks
):
    logs = []
    monkeypatch.setattr(
        scene_model, "log_timing", lambda name, **fields: logs.append((name, fields))
    )

    class SlowBody(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False
            self.cancelled = False

        async def __aiter__(self):
            for chunk in initial_chunks:
                yield chunk
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
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
    assert body.cancelled
    (record,) = [fields for name, fields in logs if name == "vlm_http_timing"]
    assert record["outcome"] == "cancelled"
    assert record["http_status"] == 200
    assert 0.025 <= record["response_read_seconds"] < 1
    assert record["response_chunks"] == len(initial_chunks)
    assert record["response_bytes"] == sum(map(len, initial_chunks))
    assert record["response_first_non_whitespace_seconds"] is None
    assert record["response_leading_whitespace_bytes"] == sum(map(len, initial_chunks))
    assert "http_first_non_whitespace_seconds" not in record
    (inspection,) = [fields for name, fields in logs if name == "vlm_inspection_timing"]
    assert inspection["error_code"] == "timeout"
    assert inspection["outcome"] == "error"
    if initial_chunks:
        assert (
            0 <= record["response_first_chunk_seconds"]
            <= record["response_last_chunk_seconds"] < record["response_read_seconds"]
        )
    else:
        assert record["response_first_chunk_seconds"] is None
        assert record["response_last_chunk_seconds"] is None
    assert "SECRET" not in repr(logs)
    assert "PRIVATE reasoning" not in repr(logs)
    assert "openrouter.ai" not in repr(logs)
    assert inspector.inspect(_job())[0].detected is True
    assert len(pools) == 1
    assert len(seen) == 2


def test_response_body_timing_records_chunk_offsets_and_known_usage_only(
    make_inspector, monkeypatch
):
    from types import SimpleNamespace

    now, logs = [100.0], []
    monkeypatch.setattr(scene_model, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(
        scene_model, "log_timing", lambda name, **fields: logs.append((name, fields))
    )
    encoded = json.dumps(
        _response(
            usage={
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "completion_tokens_details": {
                    "reasoning_tokens": 7,
                    "private": "SECRET reasoning",
                },
                "arbitrary": "SECRET usage",
            }
        )
    ).encode()

    class ChunkedBody(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False

        async def __aiter__(self):
            now[0] += 0.25
            yield b" "
            now[0] += 1.5
            yield encoded
            now[0] += 0.25

        async def aclose(self):
            self.closed = True

    body = ChunkedBody()
    inspector, _, _ = make_inspector(lambda _: httpx.Response(200, stream=body))
    assert inspector.inspect(_job())[0].detected is True
    assert body.closed
    (record,) = [fields for name, fields in logs if name == "vlm_http_timing"]
    assert record["outcome"] == "completed"
    assert record["response_read_seconds"] == 2
    assert record["response_first_chunk_seconds"] == 0.25
    assert record["response_last_chunk_seconds"] == 1.75
    assert record["response_first_non_whitespace_seconds"] == 1.75
    assert record["response_leading_whitespace_bytes"] == 1
    assert record["response_max_chunk_gap_seconds"] == 1.5
    assert record["response_chunk_samples"] == [
        {"offset_seconds": 0.25, "bytes": 1, "whitespace_only": True},
        {"offset_seconds": 1.75, "bytes": len(encoded), "whitespace_only": False},
    ]
    (inspection,) = [fields for name, fields in logs if name == "vlm_inspection_timing"]
    assert inspection["total_seconds"] == 2
    assert inspection["verdict_parse_seconds"] == 0
    assert inspection["returned_event_count"] == 1
    assert inspection["requested_event_count"] == 1
    assert inspection["empty_verdicts"] is False
    assert record["response_chunks"] == 2
    assert record["response_bytes"] == len(encoded) + 1
    assert record["prompt_tokens"] == 50
    assert record["completion_tokens"] == 20
    assert record["reasoning_tokens"] == 7
    assert "SECRET" not in repr(logs)
    assert "PRIVATE reasoning" not in repr(logs)
    assert "arbitrary" not in repr(logs)
    assert "openrouter.ai" not in repr(logs)


def test_detailed_body_trace_is_bounded_and_empty_verdicts_are_explicit(
    make_inspector, monkeypatch
):
    logs = []
    monkeypatch.setattr(
        scene_model, "log_timing", lambda name, **fields: logs.append((name, fields))
    )
    encoded = json.dumps(_response("[]", id="gen-SECRET", usage={
        "prompt_tokens_details": {"cached_tokens": 12, "private": "SECRET"},
    })).encode()

    class ManyChunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(12):
                yield b" \r\n"
            yield encoded

    inspector, _, _ = make_inspector(lambda _: httpx.Response(200, stream=ManyChunks()))
    assert inspector.inspect(_job()) == ()
    (record,) = [fields for name, fields in logs if name == "vlm_http_timing"]
    assert len(record["response_chunk_samples"]) == 8
    assert record["response_chunk_samples_dropped"] == 5
    assert record["response_leading_whitespace_bytes"] == 36
    assert record["response_first_non_whitespace_seconds"] is not None
    assert record["cached_prompt_tokens"] == 12
    assert len(record["generation_ref"]) == 64
    (inspection,) = [fields for name, fields in logs if name == "vlm_inspection_timing"]
    assert inspection["empty_verdicts"] is True
    assert inspection["verdict_count"] == inspection["returned_event_count"] == 0
    assert inspection["requested_event_count"] == 1
    assert "SECRET" not in repr(logs)
    assert "PRIVATE reasoning" not in repr(logs)
    assert "arbitrary" not in repr(logs)
    assert "openrouter.ai" not in repr(logs)


@pytest.mark.parametrize("value", [None, True, -1, "SECRET", 1.5])
def test_response_usage_diagnostics_ignore_non_token_values(
    make_inspector, monkeypatch, value
):
    logs = []
    monkeypatch.setattr(
        scene_model, "log_timing", lambda name, **fields: logs.append((name, fields))
    )
    inspector, _, _ = make_inspector(
        lambda _: httpx.Response(
            200,
            json=_response(
                usage={
                    "prompt_tokens": value,
                    "completion_tokens": value,
                    "completion_tokens_details": {"reasoning_tokens": value},
                    "prompt_tokens_details": {"cached_tokens": value},
                }
            ),
        )
    )
    assert inspector.inspect(_job())[0].detected is True
    (record,) = [fields for name, fields in logs if name == "vlm_http_timing"]
    assert not (
        {
            "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "cached_prompt_tokens",
        }
        & record.keys()
    )
    assert "SECRET" not in repr(logs)


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


@pytest.mark.parametrize("status", [200, 401, 405, 307])
def test_warmup_has_no_credentials_body_redirects_or_inference_stats(
    make_inspector, monkeypatch, status,
):
    logs = []
    monkeypatch.setattr(
        scene_model, "log_timing", lambda name, **fields: logs.append((name, fields)),
    )

    def forbidden_key_lookup():
        pytest.fail("Connection warmup must not retrieve inference credentials")

    inspector, seen, _ = make_inspector(
        lambda _: httpx.Response(
            status, headers={"location": "https://redirect-secret.test/"},
        ),
        api_key=forbidden_key_lookup,
        settings=ModelAPISettings(
            provider="openrouter",
            endpoint="https://URL_SECRET:SECRET@private.test/v1/chat/completions"
            "?api_key=QUERY_SECRET#PRIVATE",
            extra_headers={"x-secret": "SETTINGS_SECRET"},
        ),
        client_options={
            "auth": httpx.BasicAuth("CLIENT_SECRET", "SECRET"),
            "headers": {"authorization": "Bearer HEADER_SECRET", "x-secret": "SECRET"},
            "cookies": {"secret": "COOKIE_SECRET"},
            "params": {"api_key": "PARAM_SECRET"},
            "follow_redirects": True,
        },
    )
    previous_stats = inspector.stats()
    assert inspector.warmup() is True
    assert inspector.stats() == previous_stats
    (request,) = seen
    assert request.method == "HEAD"
    assert str(request.url) == "https://private.test/v1/chat/completions"
    assert dict(request.headers) == {"host": "private.test"}
    assert request.content == b""
    assert all(0 < timeout <= 5 for timeout in request.extensions["timeout"].values())
    (record,) = [
        fields for name, fields in logs if name == "vlm_connection_warmup_timing"
    ]
    assert record["outcome"] == "transport_ready"
    assert record["total_seconds"] >= 0
    assert not any(private in repr(logs) for private in ("SECRET", "private.test"))


def test_warmup_consumes_response_and_reuses_pool_until_closed(make_inspector):
    class EmptyHeadBody(httpx.AsyncByteStream):
        exhausted = False
        closed = False

        async def __aiter__(self):
            self.exhausted = True
            if False:
                yield b""

        async def aclose(self):
            self.closed = True

    body = EmptyHeadBody()
    inspector, seen, pools = make_inspector(
        lambda request: (
            httpx.Response(405, stream=body)
            if request.method == "HEAD" else httpx.Response(200, json=_response())
        ),
    )
    assert inspector.warmup()
    assert body.exhausted and body.closed
    owning_loop = inspector._loop
    assert inspector.inspect(_job())[0].detected is True
    assert inspector._loop is owning_loop
    assert len(pools) == 1
    assert [request.method for request in seen] == ["HEAD", "POST"]
    inspector.close()
    inspector.close()
    assert pools[0].is_closed
    assert owning_loop.is_closed()
    assert inspector.warmup() is False
    assert len(seen) == 2


def test_failed_warmup_keeps_pool_usable_and_failure_logs_private(
    make_inspector, monkeypatch,
):
    logs = []
    monkeypatch.setattr(
        scene_model, "log_timing", lambda name, **fields: logs.append((name, fields)),
    )

    def handler(request):
        if request.method == "HEAD":
            raise httpx.ConnectError("https://SECRET.test/ CONNECTION_SECRET")
        return httpx.Response(200, json=_response())

    inspector, seen, pools = make_inspector(handler)
    assert inspector.warmup() is False
    assert inspector.stats()["requests"] == 0
    assert inspector.inspect(_job())[0].detected is True
    assert len(pools) == 1
    assert len(seen) == 2
    (record,) = [
        fields for name, fields in logs if name == "vlm_connection_warmup_timing"
    ]
    assert record["outcome"] == "failed"
    assert "SECRET" not in repr(logs)


def test_warmup_initialization_failure_does_not_disable_later_inference(
    make_inspector, monkeypatch,
):
    inspector, seen, pools = make_inspector()
    create_client = scene_model.create_vlm_http_client
    attempts = 0

    def temporary_failure():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelAPIError("TLS initialization failed: SECRET")
        return create_client()

    monkeypatch.setattr(scene_model, "create_vlm_http_client", temporary_failure)
    assert inspector.warmup() is False
    assert not seen
    assert inspector.stats()["requests"] == 0
    assert inspector.inspect(_job())[0].detected is True
    assert len(pools) == 1
    assert attempts == 2


def test_warmup_deadline_includes_body_and_respects_shorter_request_timeout(
    make_inspector, monkeypatch,
):
    logs = []
    monkeypatch.setattr(
        scene_model, "log_timing", lambda name, **fields: logs.append((name, fields)),
    )

    class SlowHeadBody(httpx.AsyncByteStream):
        cancelled = False
        closed = False

        async def __aiter__(self):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            yield b""

        async def aclose(self):
            self.closed = True

    body = SlowHeadBody()
    inspector, seen, pools = make_inspector(
        lambda request: (
            httpx.Response(200, stream=body)
            if request.method == "HEAD" else httpx.Response(200, json=_response())
        ),
        settings=ModelAPISettings(provider="openrouter", timeout_seconds=0.05),
    )
    started = time.monotonic()
    assert inspector.warmup() is False
    assert time.monotonic() - started < 1
    assert body.cancelled and body.closed
    assert all(
        0 < timeout <= 0.05 for timeout in seen[0].extensions["timeout"].values()
    )
    (record,) = [
        fields for name, fields in logs if name == "vlm_connection_warmup_timing"
    ]
    assert record["outcome"] == "timeout"
    assert inspector.inspect(_job())[0].detected is True
    assert len(pools) == 1


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


@pytest.mark.parametrize("quality", [0, 101, True, 70.5, "70"])
def test_scene_jpeg_quality_is_strictly_bounded(make_inspector, quality):
    with pytest.raises(ValueError, match="jpeg_quality"):
        make_inspector(jpeg_quality=quality)


def test_jpeg_quality_and_exact_request_byte_counts_are_safe(
    make_inspector, monkeypatch
):
    logs = []
    monkeypatch.setattr(
        scene_model, "log_timing", lambda name, **fields: logs.append((name, fields))
    )
    inspector, seen, _ = make_inspector(jpeg_quality=70, max_image_side=32)
    assert inspector.inspect(_job())[0].detected is True
    (record,) = [fields for name, fields in logs if name == "vlm_http_timing"]
    (request,) = seen
    images = [
        base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        for part in json.loads(request.content)["messages"][1]["content"]
        if part["type"] == "image_url"
    ]
    assert record["request_bytes"] == len(request.content)
    assert record["http_request_attempted"] is True
    assert record["encoded_image_bytes"] == sum(map(len, images))
    assert record["request_bytes"] > record["encoded_image_bytes"]
    assert record["frame_count"] == len(images) == 2
    assert record["jpeg_quality"] == 70
    assert record["max_image_side"] == 32
    assert record["outcome"] == "completed"
    assert record["transport_trace_observed"] is False
    assert "send_request_body_seconds" not in record  # unavailable, not a zero
    with Image.open(io.BytesIO(images[0])) as image:
        assert image.size == (32, 21)
        assert image.quantization[0][0] == 10  # standard JPEG quality 70 table
    for private in (
        "SECRET",
        "private-camera",
        "person-1",
        "openrouter.ai",
        "image_url",
    ):
        assert private not in repr(logs)


@pytest.mark.parametrize("failure", [False, True])
def test_http_trace_records_body_headers_and_incomplete_upload_safely(
    make_inspector,
    monkeypatch,
    failure,
):
    from types import SimpleNamespace

    now, logs = [100.0], []
    monkeypatch.setattr(scene_model, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(
        scene_model, "log_timing", lambda name, **fields: logs.append((name, fields))
    )

    async def handler(request):
        trace = request.extensions["trace"]
        private = {"url": "https://SECRET.test", "headers": {"authorization": "SECRET"}}
        await trace("private.unsupported.started", private)
        await trace("http11.send_request_body.started", private)
        now[0] += 0.25
        if failure:
            # An abruptly failed transport need not emit the matching end event.
            raise httpx.WriteError("SECRET body credentials")
        await trace("http11.send_request_body.complete", private)
        await trace("http11.receive_response_headers.started", private)
        now[0] += 1.5
        await trace("http11.receive_response_headers.complete", private)
        return httpx.Response(200, json=_response())

    inspector, seen, _ = make_inspector(handler)
    if failure:
        with pytest.raises(ModelAPIError) as error:
            inspector.inspect(_job())
        assert error.value.code == "connection"
    else:
        assert inspector.inspect(_job())[0].detected is True
    (record,) = [fields for name, fields in logs if name == "vlm_http_timing"]
    assert record["request_bytes"] == len(seen[0].content)
    assert record["http_request_attempted"] is True
    assert record["send_request_body_seconds"] == 0.25
    assert record["request_body_completed"] is not failure
    assert record["transport_trace_observed"] is True
    if not failure:
        assert record["receive_response_headers_seconds"] == 1.5
        assert record["http_status"] == 200
    assert record["outcome"] == ("error" if failure else "completed")
    assert "SECRET" not in repr(logs)


def test_failure_before_send_does_not_reuse_previous_http_attempt(
    make_inspector, monkeypatch
):
    inspector, seen, _ = make_inspector()
    inspector.inspect(_job())
    assert inspector.last_http_timing["http_request_attempted"] is True

    def fail_build(*args, **kwargs):
        raise ValueError("private payload cannot be built")

    monkeypatch.setattr(inspector._http, "build_request", fail_build)
    with pytest.raises(ModelAPIError):
        inspector.inspect(_job())
    assert len(seen) == 1
    assert inspector.last_http_timing["http_request_attempted"] is False


@pytest.mark.parametrize("content,wrapper,reason", [
    ('```json\n{"PRIVATE_SECRET":false,}\n```',
     "complete_fence", "invalid_json_syntax"),
    ('```json\n{"PRIVATE_SECRET":false}', "unrecognized_fence", "invalid_json_syntax"),
    ('{"PRIVATE_SECRET":false,"PRIVATE_SECRET":true}', "none", "duplicate_json_field"),
])
def test_reply_diagnostics_are_bounded_and_do_not_expose_content(
    content, wrapper, reason,
):
    result = scene_model._reply_diagnostics(content)
    assert result["json_wrapper"] == wrapper
    assert result["diagnostic_json_reason"] == reason
    assert "PRIVATE_SECRET" not in str(result)
    assert len(result["content_sha256"]) == 64
    assert len(result["json_token_kinds_prefix"]) <= 80
    if reason == "invalid_json_syntax":
        assert result["json_error_line"] >= 1
        assert result["json_error_column"] >= 1
