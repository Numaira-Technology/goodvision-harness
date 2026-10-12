"""The fixed experiment must keep routing and presentation on both launch paths."""

import json
from dataclasses import replace

import httpx
import pytest
from test_vlm_independent import config, job
from test_vlm_lean_grounding import payload

from goodvision_harness import scene_model
from goodvision_harness.model_api import ModelAPISettings
from tools.vlm_benchmark import modes, qwen35b_alibaba
from tools.vlm_benchmark.route_audit import audit


@pytest.mark.parametrize("cloud", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_fixed_plan_without_network_or_credentials(
    cloud, fallback, capsys, monkeypatch
):
    monkeypatch.setattr(
        qwen35b_alibaba,
        "read_api_key",
        lambda *_: pytest.fail("planning read a credential"),
    )
    args = ["--real", "--plan-only", *(["--cloud"] if cloud else [])]
    if fallback:
        args.append("--allow-provider-fallback")
    assert qwen35b_alibaba.main(args) == 0
    text = capsys.readouterr().out
    plan = json.loads(text.split("\n", 3)[3] if cloud else text)
    s = plan["settings"]
    assert s["model"] == qwen35b_alibaba.MODEL
    assert s["provider_only"] == ("" if fallback else "alibaba")
    assert s["provider_prefer"] == ("alibaba" if fallback else "")
    assert s["backend"] == "openrouter" and s["require_route_match"]
    assert s["streams"] == [12] and s["repeats"] == 2
    assert s["modes"] == ["events", "compact", "parallel"]
    assert s["monitor_layout"] == "single-video" and s["record"]
    assert s["global_limit"] == 48 and s["per_stream_limit"] == 4
    assert s["request_rate"] == 40 and s["grounding_format"] == "lean-v1"
    assert s["local_concurrency"] == 4 and s["observation_fps"] == 1.25
    assert plan["request_budget"]["measurement_upper"] == 6624


@pytest.mark.parametrize(
    "extra",
    [
        ["--model", "other"],
        ["--provider-only", "deepinfra"],
        ["--streams", "1"],
        ["--monitor-layout", "evidence"],
        ["--no-record"],
    ],
)
def test_fixed_entry_point_rejects_overrides(extra):
    with pytest.raises(SystemExit):
        qwen35b_alibaba.main(["--plan-only", *extra])


@pytest.mark.parametrize("lane", ["events", "compact"])
@pytest.mark.parametrize("fallback", [False, True])
def test_actual_inspector_sends_fixed_model_route_and_keeps_quality_checks(
    lane, fallback, monkeypatch
):
    options = modes.parser().parse_args(
        qwen35b_alibaba.fixed_arguments(allow_provider_fallback=fallback)
    )
    modes.validate(options)
    base = modes.configure(options)
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["model"] == qwen35b_alibaba.MODEL
        assert body["provider"]["order" if fallback else "only"] == ["alibaba"]
        assert body["provider"]["allow_fallbacks"] is fallback
        if fallback:
            assert "only" not in body["provider"]
        assert body["provider"]["require_parameters"] is True
        assert body["reasoning"] == {"enabled": False}
        assert body["response_format"] == {"type": "json_object"}
        assert "models" not in body and "ignore" not in body["provider"]
        answer = dict.fromkeys(modes.LABELS, False) if lane == "events" else payload()
        return httpx.Response(
            200,
            json={
                "model": qwen35b_alibaba.MODEL,
                "provider": "DeepInfra" if fallback else "Alibaba",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(answer)},
                    }
                ],
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
        output_mode=lane,
        grounding_format="lean-v1",
        model=qwen35b_alibaba.MODEL,
        api_key="fixture-only",
        model_settings=ModelAPISettings(provider="openrouter", max_retries=0),
        parameters=base.inference.request_parameters,
    )
    inspector = modes.mode_inspector(cfg)
    try:
        verdicts = inspector.inspect(job(cfg, 1.2, 18))
        assert {v.event_label for v in verdicts} == set(modes.LABELS)
        assert inspector.last_response_diagnostics["model_match"] == "matched"
        assert inspector.last_response_diagnostics["provider_category"] == (
            "DeepInfra" if fallback else "Alibaba"
        )
        assert len(seen) == 1
    finally:
        inspector.close()


@pytest.mark.parametrize("lane", ["events", "compact"])
@pytest.mark.parametrize(
    "model,provider",
    [
        ("qwen/qwen3-vl-30b-a3b-instruct", "alibaba"),
        (qwen35b_alibaba.MODEL, "deepinfra"),
    ],
)
def test_alibaba_multimodal_compatibility_does_not_change_other_routes(
    lane, model, provider
):
    cfg = replace(
        config(),
        model=model,
        output_mode=lane,
        grounding_format="lean-v1",
        model_settings=ModelAPISettings(provider="openrouter"),
        parameters={"provider": {"only": [provider], "allow_fallbacks": False}},
    )
    inspector = modes.mode_inspector(cfg)
    try:
        assert inspector._parameters["response_format"]["type"] == "json_schema"
    finally:
        inspector.close()


@pytest.mark.parametrize(
    "provider,model,valid",
    [
        ("Alibaba", "matched", True),
        ("Alibaba Cloud Int.", "matched", True),
        ("DeepInfra", "matched", False),
        ("Alibaba", "mismatch", False),
        (None, "matched", False),
        ("Alibaba", "unreported", False),
    ],
)
def test_route_audit_requires_positive_evidence_for_every_lane(
    tmp_path, provider, model, valid
):
    records = []
    for lane in ("information", "grounding"):
        records.append(
            {
                "report": {
                    "vlm_lane": lane,
                    "state": "completed",
                    "analysis_complete": True,
                    "timing": {
                        "response_diagnostics": {
                            "provider_category": provider,
                            "model_match": model,
                        }
                    },
                }
            }
        )
    # Timeouts/admission errors are not misreported as provider mismatches.
    records.append(
        {
            "report": {
                "vlm_lane": "grounding",
                "state": "failed",
                "error_code": "timeout",
                "timing": {},
            }
        }
    )
    path = tmp_path / "reports.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert audit(path, {"lanes": {"events": {}, "grounding": {}}})["valid"] is valid
    assert not audit(path, {"lanes": {"events": {}, "grounding": {}, "missing": {}}})[
        "valid"
    ]


def test_wrong_provider_stops_at_preflight_and_marks_trial_invalid(
    tmp_path, monkeypatch
):
    from test_vlm_multistream_benchmark import report

    from tools.vlm_multistream_benchmark import summarize

    options = modes.parser().parse_args(qwen35b_alibaba.fixed_arguments())
    options.real, options.record, options.monitor = True, False, False
    options.output = tmp_path / "run"
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"decode is mocked")
    options.video = [video]
    monkeypatch.setattr(modes, "mode_api_key", lambda _: "fixture-key")
    monkeypatch.setattr(modes, "decode", lambda *_: [b"fixture-frame"])
    # This exercises route validation, not the POSIX cloud execution lock.
    monkeypatch.setattr(modes, "acquire", lambda _: None)

    class QuietTelemetry:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def summary(self):
            return {}

    monkeypatch.setattr(modes, "Telemetry", QuietTelemetry)
    calls = []

    def stage(args, base, count, frames, location, repetition, **_):
        calls.append(location.name)
        r = report()
        r["timing"]["response_diagnostics"] = {
            "provider_category": "DeepInfra",
            "model_match": "matched",
        }
        folder = location / f"streams-{count}-repeat-{repetition}"
        folder.mkdir()
        (folder / "reports.jsonl").write_text(
            json.dumps({"camera": 1, "report": r}) + "\n", encoding="utf-8"
        )
        data = summarize([r], {}, 3, 1, 3)
        return {
            "streams": count,
            "repeat": repetition,
            "lanes": {"events": data},
            "per_stream": [{"lanes": {"events": data}}],
            "all_drained": True,
            "input_timing_valid": True,
            "result_records_missing": 0,
            "paired_windows": None,
            "nominal_requests_per_second": 1.25,
            "rate_limit_below_offered_load": False,
        }

    monkeypatch.setattr(modes, "run_stage", stage)
    with pytest.raises(RuntimeError, match="model/provider"):
        modes.run(options)
    assert calls == ["preflight-events"]
    summary = json.loads((options.output / "summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "failed"
    assert summary["jobs"][0]["route_audit"]["valid"] is False
    assert "无效，保留诊断" in (options.output / "report.html").read_text(  # noqa: RUF001
        encoding="utf-8"
    )


@pytest.mark.parametrize("choices", [None, {}, "invalid-choices"])
def test_malformed_response_cannot_hide_a_route_mismatch(tmp_path, choices):
    # One verified response must not hide a later malformed, misrouted response.
    records = []
    for provider, model, complete in [
        ("Alibaba", qwen35b_alibaba.MODEL, True),
        ("DeepInfra", "arbitrary-server-model-text", False),
    ]:
        diagnostics = scene_model._response_shape(
            {
                "provider": provider,
                "model": model,
                "choices": [] if complete else choices,
            },
            qwen35b_alibaba.MODEL,
        )
        assert "arbitrary-server-model-text" not in json.dumps(diagnostics)
        records.append(
            {
                "report": {
                    "vlm_lane": "information",
                    "state": "completed" if complete else "failed",
                    "analysis_complete": complete,
                    "timing": {"response_diagnostics": diagnostics},
                }
            }
        )
    path = tmp_path / "reports.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    result = audit(path, {"lanes": {"events": {}}})
    assert result["valid"] is False
    assert result["lanes"]["events"]["mismatch"] == 1


@pytest.mark.parametrize(
    "provider,model,valid",
    [
        ("DeepInfra", "matched", True),
        ("Alibaba", "matched", True),
        ("other", "matched", True),
        ("DeepInfra", "mismatch", False),
        (None, "matched", False),
        ("Alibaba", "unreported", False),
    ],
)
def test_preferred_route_keeps_model_validation_and_provider_counts(
    tmp_path, provider, model, valid
):
    path = tmp_path / "reports.jsonl"
    path.write_text(
        json.dumps(
            {
                "report": {
                    "state": "completed",
                    "analysis_complete": True,
                    "timing": {
                        "response_diagnostics": {
                            "provider_category": provider,
                            "model_match": model,
                        }
                    },
                }
            }
        )
        + "\n"
    )
    result = audit(path, {"lanes": {"events": {}}}, allow_other_providers=True)
    assert result["valid"] is valid
    assert result["lanes"]["events"]["complete_providers"] == {
        provider or "unreported": 1,
    }


@pytest.mark.parametrize(
    "extra",
    [
        ["--provider-only", "alibaba"],
        ["--backend", "dashscope"],
        ["--backend", "local"],
    ],
)
def test_preferred_provider_rejects_conflicting_routes(extra):
    args = modes.parser().parse_args(["--provider-prefer", "alibaba", *extra])
    with pytest.raises(ValueError):
        modes.validate(args)


def test_json_object_grounding_example_keeps_policy_and_strict_parser():
    from goodvision_harness.scene_model import _ResponseValidationError
    from goodvision_harness.vlm_event_policy import OUTPUT_FORMAT_MARKER
    from goodvision_harness.vlm_lean_grounding import expand

    options = modes.parser().parse_args(
        qwen35b_alibaba.fixed_arguments(allow_provider_fallback=True)
    )
    modes.validate(options)
    cfg = replace(
        config(),
        model=qwen35b_alibaba.MODEL,
        labels=modes.LABELS,
        output_mode="compact",
        grounding_format="lean-v1",
        model_settings=ModelAPISettings(provider="openrouter"),
        parameters=modes.configure(options).inference.request_parameters,
    )
    inspector = modes.mode_inspector(cfg)
    event_inspector = modes.mode_inspector(replace(cfg, output_mode="events"))
    try:
        evidence = job(cfg, 1.2, 18)
        prompt = inspector.build_messages(evidence)[0]["content"]
        event_prompt = event_inspector.build_messages(evidence)[0]["content"]
        assert (
            prompt.split(OUTPUT_FORMAT_MARKER)[0]
            == event_prompt.split(OUTPUT_FORMAT_MARKER)[0]
        )
        example = prompt.split("JSON shape only (not evidence or preset answers): ")[1]
        example = example.split(". Every e value MUST")[0]
        expanded = expand(json.loads(example), modes.LABELS)
        assert len(expanded["verdicts"]) == 6
        assert all(v["detected"] is None for v in expanded["verdicts"])
        bad = json.loads(example)
        bad["e"]["fight"] = []
        with pytest.raises(_ResponseValidationError):
            expand(bad, modes.LABELS)
        bad = json.loads(example)
        bad["e"]["smoking"][0]["i"] = bad["e"]["smoking"][0].pop("p")
        with pytest.raises(_ResponseValidationError):
            expand(bad, modes.LABELS)
    finally:
        inspector.close()
        event_inspector.close()
