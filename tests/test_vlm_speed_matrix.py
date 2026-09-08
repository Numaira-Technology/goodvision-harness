from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from goodvision_harness.model_api import ModelAPIError, ModelAPISettings

SPEC = importlib.util.spec_from_file_location(
    "vlm_speed_matrix",
    Path(__file__).parents[1] / "tools" / "vlm_speed_matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def jpeg() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((90, 160, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def case(**kwargs):
    return {
        "id": "test",
        "model": "test/vision",
        "execution_mode": "single",
        "stream": True,
        "http2": False,
        "trust_env": False,
        "keepalive_expiry": 60,
        **kwargs,
    }


def completion(content='{"o":[]}', finish="stop"):
    return {
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.001,
        },
        "provider": "mock",
    }


def sse(content='{"o":[]}'):
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": content}}]},
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    body = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        content=body + "data: [DONE]\n\n",
        extensions={"http_version": b"HTTP/2"},
    )


def client_for(handler, selected=None):
    selected = selected or case()
    return matrix.MatrixClient(
        ModelAPISettings(provider="openrouter", timeout_seconds=2, max_retries=0),
        model=selected["model"],
        api_key="not-a-real-secret",
        wire_format="compact",
        max_image_side=100,
        execution_mode=selected["execution_mode"],
        parallel_strategy=selected.get("parallel_strategy", "crops"),
        require_complete=selected.get("require_complete", False),
        connection_options=selected,
        transport=httpx.MockTransport(handler),
    )


def test_native_stream_reuses_pool_loop_and_captures_wire_images():
    requests = []
    loops = []

    async def handler(request):
        loops.append(asyncio.get_running_loop())
        requests.append(json.loads(request.content))
        return sse()

    client = client_for(handler)
    original = matrix.vlm.complete_stream
    try:
        first = matrix.run_trial(client, case(), jpeg(), 0, trace_transport=True)
        pool = client._http_client
        second = matrix.run_trial(client, case(), jpeg(), 1)
        assert first["valid"] and second["valid"]
        assert client._http_client is pool
        assert loops[0] is loops[1] is client._loop
        assert first["parts"][0]["image"]["dimensions"] == [100, 56]
        assert first["parts"][0]["response"]["_transport"]["http_version"] == "HTTP/2"
        assert all(request["model"] == "test/vision" for request in requests)
        assert all(
            request["stream_options"] == {"include_usage": True} for request in requests
        )
        assert matrix.vlm.complete_stream is original
    finally:
        loop = client._loop
        pool = client._http_client
        client.close()
    assert loop.is_closed() and pool.is_closed


def test_schema_rejects_incomplete_person_that_native_parser_accepts():
    content = json.dumps({"o": [["p", 100, 100, 300, 900, {"a": "a"}]]})
    client = client_for(lambda request: sse(content))
    try:
        row = matrix.run_trial(client, case(), jpeg(), 0)
    finally:
        client.close()
    assert row["native_success"]
    assert not row["valid"]
    validation = row["parts"][0]["validation"]
    assert validation["parse_valid"] and not validation["schema_valid"]
    assert row["people"] == 1
    assert row["objects"]


def test_require_complete_case_enables_native_validation_inside_inference():
    content = json.dumps({"o": [["p", 100, 100, 300, 900, {"a": "a"}]]})
    selected = case(require_complete=True)
    client = client_for(lambda request: sse(content), selected)
    try:
        row = matrix.run_trial(client, selected, jpeg(), 0)
    finally:
        client.close()
    assert not row["native_success"] and not row["valid"]
    assert row["parts"][0]["response"]
    assert row["error_type"] == "RichLabelProtocolError"


def test_provider_http_error_code_retained_without_response_body():
    client = client_for(
        lambda request: httpx.Response(429, json={"error": "secret-body"})
    )
    try:
        row = matrix.run_trial(client, case(), jpeg(), 0)
    finally:
        client.close()
    assert not row["valid"]
    assert row["parts"][0]["http_status"] == 429
    assert row["parts"][0]["error_code"] == "http_429"
    assert "secret-body" not in json.dumps(row)


def test_parallel_preserves_sibling_responses_when_domain_parse_fails():
    selected = case(execution_mode="parallel")
    call_count = 0

    async def handler(request):
        nonlocal call_count
        call_count += 1
        # Two domain-valid completions and one invalid response complete together.
        return sse("not json" if call_count == 3 else '{"o":[]}')

    client = client_for(handler, selected)
    try:
        row = matrix.run_trial(client, selected, jpeg(), 0)
    finally:
        client.close()
    assert not row["valid"] and not row["native_success"]
    assert {part["name"] for part in row["parts"]} == {
        "people_left",
        "people_right",
        "items",
    }
    assert all("response" in part for part in row["parts"])
    assert row["usage"]["total_tokens"] == 45
    assert sum(part["validation"].get("valid", False) for part in row["parts"]) == 2


def test_nonstream_preserves_contract_and_no_stream_options():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={**completion(), "untrusted": "omit-me"})

    selected = case(stream=False)
    client = client_for(handler, selected)
    try:
        row = matrix.run_trial(client, selected, jpeg(), 0)
    finally:
        client.close()
    assert row["valid"]
    assert requests[0]["stream"] is False and "stream_options" not in requests[0]
    response = row["parts"][0]["response"]
    assert "untrusted" not in response
    assert response["usage"]["cost"] == 0.001
    assert "first_content_token_seconds" not in response["_timing"]


def test_prompt_suffix_does_not_mutate_shared_native_messages(monkeypatch):
    shared = matrix.vlm.build_compact_rich_annotation_messages(jpeg())
    original_text = shared[1]["content"][0]["text"]
    sent = []

    def handler(request):
        sent.append(json.loads(request.content)["messages"][1]["content"][0]["text"])
        return sse()

    monkeypatch.setattr(
        matrix.vlm,
        "build_compact_rich_annotation_messages",
        lambda *args, **kwargs: shared,
    )
    selected = case(prompt_suffix="Boolean fields use JSON 0/1/null.")
    client = client_for(handler, selected)
    try:
        for trial in range(2):
            assert matrix.run_trial(client, selected, jpeg(), trial)["valid"]
    finally:
        client.close()
    assert shared[1]["content"][0]["text"] == original_text
    assert sent == [original_text + " " + selected["prompt_suffix"]] * 2


def test_omit_example_preserves_region_and_row_types_without_mutating_prompts(
    monkeypatch,
):
    shared = {
        ("people", (0.0, 500.0)): matrix.vlm.build_compact_rich_annotation_messages(
            jpeg(),
            scope="people",
            person_region=(0.0, 500.0),
        ),
        ("people", (500.0, 1000.0)): matrix.vlm.build_compact_rich_annotation_messages(
            jpeg(),
            scope="people",
            person_region=(500.0, 1000.0),
        ),
        ("items", None): matrix.vlm.build_compact_rich_annotation_messages(
            jpeg(), scope="items"
        ),
    }
    originals = {key: json.dumps(messages) for key, messages in shared.items()}
    original_prompts = [
        messages[1]["content"][0]["text"] for messages in shared.values()
    ]
    sent = []

    def build(_jpeg, *, scope="all", person_region=None):
        return shared[scope, person_region]

    def handler(request):
        sent.append(json.loads(request.content)["messages"][1]["content"][0]["text"])
        return sse()

    monkeypatch.setattr(matrix.vlm, "build_compact_rich_annotation_messages", build)
    selected = case(
        execution_mode="parallel",
        parallel_strategy="full_frame",
        omit_example=True,
        prompt_suffix="Keep every attribute key.",
    )
    client = client_for(handler, selected)
    try:
        for trial in range(2):
            row = matrix.run_trial(client, selected, jpeg(), trial)
            assert row["protocol_valid"]
            assert all(part["prompt_example_removed"] for part in row["parts"])
    finally:
        client.close()
    assert {key: json.dumps(messages) for key, messages in shared.items()} == originals
    assert sent[:3] == sent[3:]
    for original, prompt in zip(original_prompts, sent[:3], strict=True):
        example = original[
            original.index("Complete valid example (") : original.index("Each row=")
        ]
        assert example not in prompt
        assert "Complete valid example (" not in prompt
        assert (
            "Each row=[label,x1,y1,x2,y2,attributes], a JSON ARRAY, NOT an object."
            in prompt
        )
        assert "four separate coordinate numbers" in prompt
        assert "Top-level JSON has only key o, an array of rows." in prompt
        assert prompt.endswith("Keep every attribute key.")
    assert "0 <= (x1+x2)/2 < 500" in sent[0]
    assert "500 <= (x1+x2)/2 < 1000" in sent[1]
    assert "Only major non-person objects" in sent[2]


@pytest.mark.parametrize(
    "content",
    [
        '{"o":[],"o":[]}',
        json.dumps({"o": [["chair", 10, 20, 100, 200]] * 9}),
    ],
)
def test_protocol_validation_catches_rules_beyond_wire_schema(content):
    client = client_for(lambda request: sse(content))
    try:
        row = matrix.run_trial(client, case(), jpeg(), 0)
    finally:
        client.close()
    assert row["native_success"]
    assert row["parts"][0]["validation"]["schema_valid"]
    assert not row["parts"][0]["validation"]["complete_contract_valid"]
    assert not row["valid"] and not row["protocol_valid"]
    assert row["visual_quality_evaluated"] is False


def test_named_wire_keeps_raw_reply_and_converts_without_losing_attributes():
    person = {
        "box": [100, 100, 300, 900],
        **dict.fromkeys(
            ("a", "g", "d", "s", "uc", "lc", "b", "us", "ls", "gl", "ht", "bt", "ho")
        ),
    }
    person.update(a="a", s="s", gl=0, uc="blue")
    named = {
        "people": [person],
        "items": [{"label": "chair", "box": [500, 500, 900, 950], "c": None}],
    }
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return sse(json.dumps(named))

    selected = case(
        wire_format="named", structured_output=True, prompt_suffix="Extra instruction."
    )
    client = client_for(handler, selected)
    try:
        row = matrix.run_trial(client, selected, jpeg(), 0)
    finally:
        client.close()
    assert row["valid"] and row["people"] == 1 and row["items"] == 1
    part = row["parts"][0]
    assert (
        json.loads(part["wire_response"]["choices"][0]["message"]["content"]) == named
    )
    compact = json.loads(part["response"]["choices"][0]["message"]["content"])
    assert len(compact["o"][0][5]) == 13
    assert compact["o"][0][5]["gl"] == 0
    assert part["wire_validation"]["schema_valid"] and part["wire_conversion_valid"]
    assert part["wire_conversion_seconds"] >= 0
    assert sent[0]["response_format"]["json_schema"]["name"] == "rich_named_all"
    assert "prefixItems" not in json.dumps(sent[0]["response_format"])
    assert sent[0]["messages"][1]["content"][0]["text"].endswith("Extra instruction.")
    assert row["usage"]["total_tokens"] == 15


def test_invalid_named_wire_retains_original_usage_even_when_conversion_fails():
    named = {"people": [{"box": [100, 100, 300, 900], "gl": "yes"}], "items": []}

    def handler(request):
        return httpx.Response(200, json=completion(json.dumps(named)))

    selected = case(wire_format="named", stream=False)
    client = client_for(handler, selected)
    try:
        row = matrix.run_trial(client, selected, jpeg(), 0)
    finally:
        client.close()
    part = row["parts"][0]
    assert not row["valid"] and not row["native_success"]
    assert (
        not part["wire_conversion_valid"]
        and not part["wire_validation"]["schema_valid"]
    )
    assert "response" not in part
    assert (
        json.loads(part["wire_response"]["choices"][0]["message"]["content"]) == named
    )
    assert row["usage"]["total_tokens"] == 15 and row["usage"]["cost"] == 0.001
    summary = matrix.summarize([row])
    assert summary["reported_cost"] == 0.001 and summary["cost_reporting_complete"]
    assert summary["failed"] == 1


def test_named_parallel_keeps_crop_scope_and_validates_each_wire_reply():
    names = []

    def handler(request):
        schema_name = json.loads(request.content)["response_format"]["json_schema"][
            "name"
        ]
        names.append(schema_name)
        return sse(
            json.dumps(
                {"items": []} if schema_name.endswith("items") else {"people": []}
            )
        )

    selected = case(
        wire_format="named", execution_mode="parallel", structured_output=True
    )
    client = client_for(handler, selected)
    try:
        row = matrix.run_trial(client, selected, jpeg(), 0)
    finally:
        client.close()
    assert row["valid"]
    assert names == ["rich_named_people", "rich_named_people", "rich_named_items"]
    assert all(part["wire_validation"]["schema_valid"] for part in row["parts"])
    assert row["usage"]["total_tokens"] == 45


def test_named_full_frame_preserves_regions_and_global_coordinates():
    attributes = dict.fromkeys(
        ("a", "g", "d", "s", "uc", "lc", "b", "us", "ls", "gl", "ht", "bt", "ho")
    )
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        prompt = payload["messages"][1]["content"][0]["text"]
        if "0 <= (x1+x2)/2 < 500" in prompt:
            named = {"people": [{"box": [100, 100, 300, 900], **attributes}]}
        elif "500 <= (x1+x2)/2 < 1000" in prompt:
            named = {"people": [{"box": [600, 100, 800, 900], **attributes}]}
        else:
            named = {
                "items": [{"label": "chair", "box": [100, 500, 400, 900], "c": None}]
            }
        return sse(json.dumps(named))

    selected = case(
        wire_format="named",
        execution_mode="parallel",
        parallel_strategy="full_frame",
        structured_output=True,
        require_complete=True,
        prompt_suffix="Keep all attributes.",
    )
    client = client_for(handler, selected)
    try:
        row = matrix.run_trial(client, selected, jpeg(), 0)
    finally:
        client.close()
    assert row["protocol_valid"] and row["people"] == 2 and row["items"] == 1
    assert len({part["image"]["sha256"] for part in row["parts"]}) == 1
    assert all(part["image"]["dimensions"] == [100, 56] for part in row["parts"])
    prompts = [payload["messages"][1]["content"][0]["text"] for payload in requests]
    assert "0 <= (x1+x2)/2 < 500" in prompts[0]
    assert "500 <= (x1+x2)/2 < 1000" in prompts[1]
    for prompt in prompts[:2]:
        assert "left boundary included, right boundary excluded" in prompt
        assert "Keep FULL-image coordinates" in prompt
        assert "Each person is an OBJECT with box" in prompt
        assert prompt.endswith("Keep all attributes.")
    assert "(x1+x2)/2" not in prompts[2]
    people = [obj for obj in row["objects"] if obj["label"] == "person"]
    assert [(obj["bbox"]["x1"], obj["bbox"]["x2"]) for obj in people] == [
        (0.1, 0.3),
        (0.6, 0.8),
    ]
    assert all(part["wire_validation"]["schema_valid"] for part in row["parts"])
    assert all(part["wire_conversion_valid"] for part in row["parts"])


def test_full_frame_strategy_sends_full_images_and_keeps_global_boxes():
    attributes = dict.fromkeys(
        ("a", "g", "d", "s", "uc", "lc", "b", "us", "ls", "gl", "ht", "bt", "ho")
    )
    prompts = []

    def handler(request):
        prompt = json.loads(request.content)["messages"][1]["content"][0]["text"]
        prompts.append(prompt)
        if "0 <= (x1+x2)/2 < 500" in prompt:
            objects = [["p", 100, 100, 300, 900, attributes]]
        elif "500 <= (x1+x2)/2 < 1000" in prompt:
            objects = [["p", 600, 100, 800, 900, attributes]]
        else:
            objects = [["chair", 100, 500, 400, 900]]
        return sse(json.dumps({"o": objects}))

    selected = case(
        execution_mode="parallel", parallel_strategy="full_frame", require_complete=True
    )
    client = client_for(handler, selected)
    try:
        assert client.parallel_strategy == "full_frame"
        row = matrix.run_trial(client, selected, jpeg(), 0)
    finally:
        client.close()
    assert row["valid"] and row["people"] == 2 and row["items"] == 1
    assert len({part["image"]["sha256"] for part in row["parts"]}) == 1
    assert all(part["image"]["dimensions"] == [100, 56] for part in row["parts"])
    people = [obj for obj in row["objects"] if obj["label"] == "person"]
    assert [(obj["bbox"]["x1"], obj["bbox"]["x2"]) for obj in people] == [
        (0.1, 0.3),
        (0.6, 0.8),
    ]
    assert "0 <= (x1+x2)/2 < 500" in prompts[0]
    assert "500 <= (x1+x2)/2 < 1000" in prompts[1]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": "private-secret"}),
        httpx.Response(200, json=completion(finish="length")),
        httpx.Response(
            200, headers={"content-type": "text/html"}, content="private-secret"
        ),
        httpx.Response(200, json={"error": "private-secret"}),
    ],
)
def test_nonstream_rejects_status_media_type_and_finish_without_remote_text(response):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: response)
        ) as client:
            with pytest.raises(ModelAPIError) as caught:
                await matrix.complete_nonstream(
                    client,
                    settings=ModelAPISettings(provider="openrouter"),
                    model="test",
                    messages=[{"role": "user", "content": "test"}],
                    api_key="private-secret",
                    deadline_seconds=1,
                )
            assert "private-secret" not in str(caught.value)

    asyncio.run(run())


def test_nonstream_wall_deadline_and_no_retry():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.1)
        return httpx.Response(200, json=completion())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ModelAPIError, match="wall deadline"):
                await matrix.complete_nonstream(
                    client,
                    settings=ModelAPISettings(provider="openrouter", max_retries=5),
                    model="test",
                    messages=[{"role": "user", "content": "test"}],
                    api_key="secret",
                    deadline_seconds=0.01,
                )

    asyncio.run(run())
    assert calls == 1


def test_key_loading_does_not_change_route_or_import_other_environment(
    tmp_path, monkeypatch
):
    env = tmp_path / ".env"
    env.write_text(
        "OPENROUTER_API_KEY=only-this-key\nINFERENCE_PROVIDER=bad\nMODEL_API_KEY=wrong\n"
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("INFERENCE_PROVIDER", "original")
    assert matrix.load_key(env) == "only-this-key"
    assert matrix.os.environ["INFERENCE_PROVIDER"] == "original"
    assert "OPENROUTER_API_KEY" not in matrix.os.environ


def test_summary_counts_failures_in_under_three_denominator():
    rows = [
        {"wall_seconds": 2.0, "valid": True, "parts": [], "usage": {}},
        {
            "wall_seconds": 1.0,
            "valid": False,
            "parts": [],
            "usage": {},
            "error_type": "failure",
        },
        {"wall_seconds": 4.0, "valid": True, "parts": [], "usage": {}},
    ]
    summary = matrix.summarize(rows)
    assert summary["requests"] == 3 and summary["failed"] == 1
    assert summary["valid_under_3_rate"] == pytest.approx(1 / 3)
    assert summary["p95_wall_seconds"] == 4


def test_case_defaults_and_rejected_overrides(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text('[{"id":"baseline","model":"test/vision"}]')
    selected = matrix.load_cases(path)[0]
    assert selected["keepalive_expiry"] == 60 and selected["trust_env"] is True
    path.write_text(
        '[{"id":"bad","model":"test/vision","request_parameters":{"stream":false}}]'
    )
    with pytest.raises(ValueError, match="fixed field"):
        matrix.load_cases(path)


def test_full_frame_case_requires_parallel_and_accepts_named(tmp_path):
    path = tmp_path / "cases.json"
    selected = {
        "id": "full",
        "model": "test/vision",
        "parallel_strategy": "full_frame",
        "execution_mode": "parallel",
    }
    path.write_text(json.dumps([selected]))
    assert matrix.load_cases(path)[0]["parallel_strategy"] == "full_frame"
    path.write_text(json.dumps([{**selected, "wire_format": "named"}]))
    assert matrix.load_cases(path)[0]["wire_format"] == "named"
    path.write_text(json.dumps([{**selected, "execution_mode": "single"}]))
    with pytest.raises(ValueError, match="parallel"):
        matrix.load_cases(path)


@pytest.mark.parametrize("status", [401, 403])
def test_preflight_rejects_credentials_without_logging_remote_body(monkeypatch, status):
    actual_client = httpx.AsyncClient
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json={"error": "secret-remote-body"})

    monkeypatch.setattr(
        matrix.httpx,
        "AsyncClient",
        lambda **kwargs: actual_client(
            **kwargs,
            transport=httpx.MockTransport(handler),
        ),
    )
    with pytest.raises(ModelAPIError, match=f"HTTP {status}") as caught:
        asyncio.run(matrix.preflight_key("private-credential", trust_env=False))
    assert len(seen) == 1 and seen[0].method == "GET"
    assert seen[0].url.path == "/api/v1/key"
    assert "secret-remote-body" not in str(caught.value)
    assert "private-credential" not in str(caught.value)


def test_cli_run_rotates_images_and_writes_complete_jsonl(
    tmp_path, monkeypatch, capsys
):
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=private-credential\nINFERENCE_PROVIDER=aiclub\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    inputs = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for path in inputs:
        path.write_bytes(jpeg())
    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {"id": "sse", "model": "test/vision", "trust_env": False},
                {
                    "id": "json",
                    "model": "test/vision",
                    "stream": False,
                    "trust_env": False,
                },
            ]
        )
    )
    output = tmp_path / "results.jsonl"
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        return sse() if body["stream"] else httpx.Response(200, json=completion())

    actual_class = matrix.MatrixClient
    monkeypatch.setattr(
        matrix,
        "MatrixClient",
        lambda *args, **kwargs: actual_class(
            *args,
            **kwargs,
            transport=httpx.MockTransport(handler),
        ),
    )
    args = matrix.build_parser().parse_args(
        [
            "--env-file",
            str(env),
            "--image",
            str(inputs[0]),
            "--image",
            str(inputs[1]),
            "--cases",
            str(cases),
            "--output",
            str(output),
            "--repeats",
            "4",
            "--no-preflight",
        ]
    )
    matrix.run(args)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    trials = [row for row in records if row["type"] == "trial"]
    summaries = [row for row in records if row["type"] == "summary"]
    assert len(trials) == 8 and len(summaries) == 2
    assert all(row["valid"] for row in trials)
    assert all(
        row["protocol_valid"] and not row["visual_quality_evaluated"] for row in trials
    )
    assert all(request.url.host == "openrouter.ai" for request in requests)
    assert all(request.headers["X-OpenRouter-Cache"] == "false" for request in requests)
    assert [(row["image_index"], row["mirrored"]) for row in trials[::2]] == [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ]
    assert all(summary["requests"] == 4 for summary in summaries)
    assert all(
        summary["validity_definition"] == "protocol_and_schema_only"
        for summary in summaries
    )
    assert "private-credential" not in capsys.readouterr().out
    assert "private-credential" not in output.read_text()
