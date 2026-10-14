"""Direct Model Studio wiring, credential separation and real inspector parsing."""

import json
from dataclasses import replace

import httpx
import pytest
from test_vlm_independent import config, job
from test_vlm_lean_grounding import payload

from goodvision_harness import scene_model
from goodvision_harness.model_api import ModelAPIError, ModelAPISettings
from tools.vlm_benchmark import modes, routes


def test_route_credentials_do_not_fall_back_to_another_api(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-only")
    monkeypatch.setenv("INFERENCE_API_KEY", "generic-router")
    monkeypatch.setenv("MODEL_API_KEY", "generic-too")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        routes.read_api_key("dashscope")
    env = tmp_path / ".env"
    env.write_text("DASHSCOPE_API_KEY=direct-only\n", encoding="utf-8")
    assert routes.read_api_key("dashscope", env) == "direct-only"
    assert routes.read_api_key("openrouter", env) == "router-only"
    assert routes.mode_api_key(ModelAPISettings(provider="dashscope")) == ""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "direct-env")
    assert routes.read_api_key("dashscope", env) == "direct-env"
    assert routes.mode_api_key(ModelAPISettings(provider="dashscope")) == "direct-env"


@pytest.mark.parametrize("backend", ["openrouter", "dashscope"])
def test_mode_credentials_match_selected_channel_with_generic_overrides(
    monkeypatch, backend
):
    monkeypatch.setenv("INFERENCE_API_KEY", "wrong-channel")
    monkeypatch.setenv("MODEL_API_KEY", "also-wrong")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-only")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "direct-only")
    settings = ModelAPISettings(provider=backend)
    expected = "router-only" if backend == "openrouter" else "direct-only"
    assert routes.mode_api_key(settings) == expected
    monkeypatch.delenv(routes.KEY_ENV[backend])
    assert routes.mode_api_key(settings) == ""


def test_other_dashscope_models_keep_explicit_output_format():
    from goodvision_harness.vlm_independent import _inspector

    requested = {"type": "json_schema", "json_schema": {"name": "existing"}}
    cfg = replace(
        config(),
        model="qwen3.7-plus",
        output_mode="events",
        model_settings=ModelAPISettings(provider="dashscope"),
        api_key="test-only",
        parameters={"response_format": requested},
    )
    inspector = _inspector(cfg)
    try:
        assert inspector._parameters["response_format"] == requested
    finally:
        inspector.close()


@pytest.mark.parametrize("backend", ["openrouter", "dashscope"])
@pytest.mark.parametrize("present", [True, False])
def test_mode_worker_receives_only_selected_credential(
    tmp_path, monkeypatch, backend, present
):
    from tools import vlm_multistream_benchmark as benchmark

    # Exercise worker configuration after the top-level preflight, including a
    # credential disappearing between stages. No worker or network is started.
    args = modes.parser().parse_args(["--backend", backend, "--real"])
    args.output_mode = "events"
    base = modes.configure(args)
    monkeypatch.setenv("INFERENCE_API_KEY", "generic-other-provider")
    monkeypatch.setenv("MODEL_API_KEY", "generic-other-provider-too")
    for env in routes.KEY_ENV.values():
        monkeypatch.setenv(env, "unselected-credential")
    if present:
        monkeypatch.setenv(routes.KEY_ENV[backend], "selected-credential")
    else:
        monkeypatch.delenv(routes.KEY_ENV[backend])
    seen = []

    class Captured(Exception):
        pass

    def capture(cfg, **kwargs):
        seen.append(cfg)
        raise Captured()

    monkeypatch.setattr(benchmark, "IndependentVLMBranch", capture)
    with pytest.raises(Captured if present else ValueError):
        benchmark.run_stage(args, base, 1, [[b"unused"]], tmp_path, 1)
    if present:
        assert len(seen) == 1 and seen[0].api_key == "selected-credential"
    else:
        assert not seen
        assert not (tmp_path / "streams-1-repeat-1").exists()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://workspace.eu-central-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://workspace.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://workspace.us-east-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
        "https://workspace.cn-hongkong.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
    ],
)
def test_direct_route_and_model_are_in_resolved_configuration(endpoint):
    args = modes.parser().parse_args(["--backend", "dashscope", "--endpoint", endpoint])
    modes.validate(args)
    base = modes.configure(args)
    assert args.model == base.inference.model == routes.DASHSCOPE_MODEL
    assert base.model_api.endpoint == endpoint
    assert base.model_api.api_key_env == "DASHSCOPE_API_KEY"
    assert base.inference.request_parameters == {"temperature": 0}


@pytest.mark.parametrize(
    "extra",
    [
        ["--provider-only", "deepinfra"],
        ["--metrics-url", "https://dashscope.aliyuncs.com/metrics"],
        ["--endpoint", "https://dashscope.aliyuncs.com/compatible-mode/v1"],
        ["--endpoint", "https://openrouter.ai/api/v1/chat/completions"],
        [
            "--endpoint",
            "https://bucket.oss-cn-beijing.aliyuncs.com/compatible-mode/v1/chat/completions",
        ],
        ["--endpoint", "https://aliyuncs.com/compatible-mode/v1/chat/completions"],
        [
            "--endpoint",
            "http://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ],
        [
            "--endpoint",
            "https://key@dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ],
        [
            "--endpoint",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions?key=secret",
        ],
    ],
)
def test_bad_direct_options_rejected_before_calls(extra):
    args = modes.parser().parse_args(["--backend", "dashscope", *extra])
    with pytest.raises(ValueError):
        modes.validate(args)


@pytest.mark.parametrize(
    "mode,wire", [("events", "legacy"), ("compact", "legacy"), ("compact", "lean-v1")]
)
def test_direct_transport_keeps_six_event_and_localization_validation(
    monkeypatch, mode, wire
):
    seen = []
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.host == "dashscope-intl.aliyuncs.com"
        assert request.headers["Authorization"] == "Bearer direct-test-key"
        assert body["model"] == routes.DASHSCOPE_MODEL
        assert body["response_format"] == {"type": "json_object"}
        assert not {
            "provider",
            "reasoning",
            "chat_template_kwargs",
            "models",
        }.intersection(body)
        assert "JSON" in body["messages"][0]["content"]
        answer = (
            dict.fromkeys(modes.LABELS, False)
            if mode == "events"
            else payload()
            if wire == "lean-v1"
            else {"no_events": 0.9}
        )
        if calls == 2:
            # No schema enforcement upstream must not silently admit missing labels.
            answer = {"fight": False}
        elif calls == 3:
            answer = payload()
            answer["e"]["ground_litter"][0]["p"] = ["p1"]  # invalid actor association
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(answer)},
                    }
                ]
            },
        )

    monkeypatch.setattr(
        scene_model,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    cfg = replace(
        config(),
        labels=modes.LABELS,
        output_mode=mode,
        grounding_format=wire,
        event_policy="research-v1",
        model=routes.DASHSCOPE_MODEL,
        api_key="direct-test-key",
        model_settings=ModelAPISettings(
            provider="dashscope",
            endpoint="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
            max_retries=0,
        ),
        parameters={"temperature": 0},
    )
    inspector = modes.mode_inspector(cfg)
    try:
        result = inspector.inspect(job(cfg, 1.2, 18))
        assert {v.event_label for v in result} == set(modes.LABELS)
        with pytest.raises(ModelAPIError) as error:
            inspector.inspect(job(cfg, 1.2, 18))
        assert error.value.code == "invalid_response"
        if wire == "lean-v1":
            with pytest.raises(ModelAPIError) as error:
                inspector.inspect(job(cfg, 1.2, 18))
            assert error.value.code == "invalid_response"
        assert len(seen) == (3 if wire == "lean-v1" else 2)
    finally:
        inspector.close()
