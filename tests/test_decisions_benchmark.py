"""Mock-only Decisions regression. No keys, rentals or paid model requests."""

import asyncio
import json
import os
import tarfile
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from benchmark_runtime_fixtures import config, job

from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.scene_evidence import EvidenceFrame
from tools.benchmark_runtime.vlm_event_policy import event_policy_text
from tools.decisions_benchmark import adapter as a
from tools.decisions_benchmark import profile as p
from tools.decisions_benchmark import suite
from tools.decisions_benchmark.__main__ import main
from tools.decisions_benchmark.report import (
    analyze,
    astra_agreement,
    audit_coverage,
    audit_usage,
)
from tools.vlm_benchmark import modes
from tools.vlm_benchmark.routes import read_api_key, validate_route
from tools.vlm_multistream_benchmark import _stage_api_key, drain_budget_seconds


def response_body():
    return {
        "model": p.MODEL,
        "answers": [
            {
                "type": "choice",
                "name": label,
                "choice": "uncertain",
                "confidence": 0.8,
                "probabilities": [
                    {"value": key, "probability": 1.0 if key == "uncertain" else 0.0}
                    for key in a.VALUES
                ],
            }
            for label in p.LABELS
        ],
        "usage": {"input_tokens": 3221, "output_tokens": 0, "total_tokens": 3221},
    }


def inspection():
    cfg = replace(
        config(),
        labels=p.LABELS,
        model=p.MODEL,
        model_settings=p.DecisionSettings(),
        output_mode="events",
        api_key="test-key",
    )
    value = job(cfg, 1.2, 3)
    value = replace(
        value,
        frames=tuple(
            EvidenceFrame("vlm-run", i, i * 0.4, value.frames[0].jpeg_bytes)
            for i in range(4)
        ),
    )
    value = replace(
        value,
        request=replace(
            value.request,
            start_seconds=value.frames[0].timestamp_seconds,
            end_seconds=value.frames[-1].timestamp_seconds,
        ),
    )
    return cfg, value


@pytest.mark.parametrize("action", ["plan", "preflight", "run", "analyze", "bundle"])
def test_default_never_calls_model_or_reads_credentials(monkeypatch, capsys, action):
    def forbidden(*args, **kwargs):
        pytest.fail("planning must stay offline")

    monkeypatch.setattr(suite, "credential", forbidden)
    monkeypatch.setattr(suite, "preflight", forbidden)
    monkeypatch.setattr(suite, "run", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    assert main([action]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["request_budget"] == {
        "measurement_upper": 1656,
        "preflight_and_warmup_upper": 36,
        "optional_soak_upper": 0,
        "total_upper": 1692,
    }
    assert plan["recordings"] == 2 and plan["execute"] is False
    assert plan["event_policy"]["sha256"] == p.POLICY_SHA256


@pytest.mark.parametrize(
    "key,value",
    [
        ("model", "other"),
        ("modes", ["parallel"]),
        ("modes", ["grounding"]),
        ("streams", [1]),
        ("observation_fps", 1),
        ("http_timeout", 3),
        ("record", False),
        ("provider_prefer", "alibaba"),
        ("endpoint", "https://openrouter.ai/v1/decisions"),
        ("metrics_url", "http://localhost/metrics"),
    ],
)
def test_frozen_contract(key, value):
    args = suite.arguments()
    setattr(args, key, value)
    with pytest.raises(ValueError):
        modes.validate(args)


def test_configuration_and_drain():
    args = suite.arguments()
    base = modes.configure(args)
    assert isinstance(base.model_api, p.DecisionSettings)
    assert base.inference.request_parameters == {}
    args.real, args.output_mode = True, "events"
    assert drain_budget_seconds(args, base) == 25
    assert p.DecisionSettings().endpoint == p.ENDPOINT
    with pytest.raises(ValueError):
        replace(base.model_api, endpoint="https://elsewhere.example/decisions")


def test_credential_isolation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for name in (
        "OPENROUTER_API_KEY",
        "MODEL_API_KEY",
        "INFERENCE_API_KEY",
        "LOCAL_INFERENCE_API_KEY",
    ):
        monkeypatch.setenv(name, "wrong-key")
    with pytest.raises(ValueError):
        read_api_key("openai_decisions")
    args = suite.arguments()
    args.real = True
    base = SimpleNamespace(model_api=p.DecisionSettings())
    with pytest.raises(ValueError):
        _stage_api_key(args, base)
    monkeypatch.setenv("OPENAI_API_KEY", "selected-key")
    assert _stage_api_key(args, base) == "selected-key"
    validate_route("openai_decisions", p.ENDPOINT)
    with pytest.raises(ValueError):
        validate_route("openai_decisions", p.ENDPOINT + "?key=bad")


def test_native_payload_frames_policy_usage_and_audit(monkeypatch, tmp_path):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url == p.ENDPOINT
        assert request.headers["Authorization"] == "Bearer test-key"
        assert set(body) == {"model", "input", "questions"}
        content = body["input"][0]["content"]
        assert content[0]["text"] == event_policy_text(p.LABELS)
        assert len([x for x in content if x["type"] == "input_image"]) == 4
        assert [q["name"] for q in body["questions"]] == list(p.LABELS)
        return httpx.Response(
            200,
            json=response_body(),
            headers={
                "x-request-id": "req_test123",
                "openai-processing-ms": "103.5",
                "x-ratelimit-limit-tokens": "2000000",
                "private-secret": "must-not-log",
            },
        )

    def client(**kwargs):
        assert kwargs == {"trust_env": False}
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(a, "create_vlm_http_client", client)
    cfg, value = inspection()
    inspector = a.inspector(cfg, audit_dir=tmp_path)
    try:
        verdicts = inspector.inspect(value)
        assert len(verdicts) == 6 and all(v.detected is None for v in verdicts)
        assert inspector.last_http_timing["prompt_tokens"] == 3221
        assert inspector.last_http_timing["completion_tokens"] == 0
        assert inspector.settings.timeout_seconds == 20
    finally:
        inspector.close()
    assert len(seen) == 1
    audit_text = next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8")
    assert "test-key" not in audit_text and "must-not-log" not in audit_text
    audit = json.loads(audit_text)
    assert audit["complete"] and audit["headers"]["openai_request_id"] == "req_test123"


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate",
        "refusal",
        "model",
        "choice",
        "probability",
        "nan",
        "bool",
        "sum",
    ],
)
def test_invalid_native_answers_fail(failure):
    body = response_body()
    if failure == "missing":
        body["answers"].pop()
    if failure == "duplicate":
        body["answers"][1] = body["answers"][0]
    if failure == "refusal":
        body["answers"][0] = {"type": "refusal", "name": "fight"}
    if failure == "model":
        body["model"] = "other"
    if failure == "choice":
        body["answers"][0]["choice"] = "maybe"
    if failure == "probability":
        body["answers"][0]["probabilities"].pop()
    if failure == "nan":
        body["answers"][0]["confidence"] = float("nan")
    if failure == "bool":
        body["answers"][0]["confidence"] = True
    if failure == "sum":
        body["answers"][0]["probabilities"][0]["probability"] = 1
    with pytest.raises(ValueError):
        a.parse_answers(body)


@pytest.mark.parametrize("status", [302, 401, 403, 429, 500])
def test_http_failures_no_retry_no_redirect(monkeypatch, tmp_path, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status,
            json={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "test-key must-not-log",
                }
            },
            headers={
                "location": "https://unsafe.example",
                "retry-after": "60",
                "x-request-id": "req_failed",
            },
        )

    monkeypatch.setattr(
        a,
        "create_vlm_http_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    cfg, value = inspection()
    inspector = a.inspector(cfg, audit_dir=tmp_path)
    try:
        with pytest.raises(ModelAPIError) as exc:
            inspector.inspect(value)
        assert exc.value.status_code == status
        assert inspector.last_http_timing["http_request_attempted"] is True
    finally:
        inspector.close()
    assert len(calls) == 1
    text = next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8")
    assert "test-key" not in text and "must-not-log" not in text
    assert json.loads(text)["complete"] is False


def test_total_deadline_cancels_and_audits(monkeypatch, tmp_path):
    async def handler(request):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(
        a,
        "create_vlm_http_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    cfg, value = inspection()
    inspector = a.inspector(cfg, audit_dir=tmp_path)
    try:
        with pytest.raises(ModelAPIError) as exc:
            inspector.inspect(value)
        assert exc.value.code == "timeout"
    finally:
        inspector.close()
    assert (
        json.loads(next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8"))[
            "complete"
        ]
        is False
    )


@pytest.mark.parametrize("status", [401, 403, 429])
@pytest.mark.parametrize("failure", ["read", "close", "deadline"])
def test_received_http_status_survives_body_failures(
    monkeypatch, tmp_path, status, failure
):
    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            if failure == "deadline":
                await asyncio.Event().wait()
            if failure == "read":
                raise httpx.ReadError("private-error-do-not-log")
            yield b"{}"

        async def aclose(self):
            if failure == "close":
                raise httpx.ReadError("private-error-do-not-log")

    if failure == "deadline":
        original_wait_for = asyncio.wait_for

        async def short_deadline(awaitable, timeout):
            return await original_wait_for(awaitable, timeout=0.02)

        monkeypatch.setattr(asyncio, "wait_for", short_deadline)

    monkeypatch.setattr(
        a,
        "create_vlm_http_client",
        lambda **kw: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(status, stream=BrokenBody())
            )
        ),
    )
    cfg, value = inspection()
    inspector = a.inspector(cfg, audit_dir=tmp_path)
    try:
        with pytest.raises(ModelAPIError) as exc:
            inspector.inspect(value)
        assert exc.value.status_code == status
    finally:
        inspector.close()
    text = next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8")
    assert "private-error" not in text
    assert json.loads(text)["http_status"] == status


def test_no_audit_does_not_claim_zero_cost(tmp_path):
    result = audit_usage(tmp_path)
    assert result["estimated_base_cost_usd"] is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only Linux service lifecycle")
def test_preflight_initialization_failure_releases_suite_lock(monkeypatch, tmp_path):
    import os

    from tools import vlm_multistream_benchmark as runner
    from tools.vlm_benchmark.lock import acquire

    monkeypatch.setattr(suite, "check_video", lambda _: None)
    monkeypatch.setattr(suite, "credential", lambda _: "test-key")
    monkeypatch.setattr(modes, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "decode", lambda *args: [])

    def fail(**kw):
        raise OSError("fixture")

    monkeypatch.setattr(a, "DecisionInspector", fail)
    with pytest.raises(OSError):
        suite.preflight(tmp_path / "output.mp4", tmp_path / "probe")
    fd = acquire(tmp_path / "outputs" / "vlm-modes" / "run")
    os.close(fd)
    result = json.loads(
        (tmp_path / "probe" / "preflight.json").read_text(encoding="utf-8")
    )
    assert result["passed"] is False


def test_header_sanitization():
    h = a.safe_headers(
        httpx.Headers(
            {
                "x-request-id": "sk-secret",
                "retry-after": "NaN",
                "openai-processing-ms": "-1",
                "x-ratelimit-limit-tokens": "secret",
                "x-ratelimit-reset-tokens": "1m2.5s",
                "x-ratelimit-reset-requests": "secret",
            }
        )
    )
    assert h == {"x-ratelimit-reset-tokens": "1m2.5s"}


def test_probability_is_not_a_new_threshold():
    body = response_body()
    body["answers"][0]["choice"] = "positive"
    body["answers"][0]["confidence"] = 0.01
    assert a.parse_answers(body)[0].detected is True


def test_native_answers_are_matched_by_name_not_position():
    body = response_body()
    body["answers"][0]["choice"] = "positive"
    body["answers"].reverse()
    parsed = a.parse_answers(body)
    assert parsed[0].event_label == "fight" and parsed[0].detected is True


def test_report_describes_direct_route(tmp_path):
    from tools.vlm_benchmark.mode_report import render_modes

    args = suite.arguments()
    args.real = True
    render_modes(
        tmp_path,
        {
            "state": "preparing",
            "settings": vars(args),
            "event_labels": list(p.LABELS),
            "jobs": [],
            "request_budget": modes.request_budget(args),
        },
    )
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "OpenAI Decisions 直连 API" in page
    assert "固定服务商：OpenAI" in page  # noqa: RUF001
    assert "允许其他供应商接替：False" in page  # noqa: RUF001


def fixture_report():
    return {
        "camera": 1,
        "report": {
            "state": "completed",
            "analysis_complete": True,
            "vlm_lane": "events",
            "window_id": "w1",
            "stream_session_id": "stream1",
            "request_id": "request1",
            "historical_only": True,
            "event_policy": {"sha256": p.POLICY_SHA256},
            "evidence": [
                {"frame_sequence": i, "timestamp_seconds": i / 2.5} for i in range(4)
            ],
            "judgments": [
                {"event_label": label, "observed_detected": False} for label in p.LABELS
            ],
            "timing": {
                "source_to_publish_seconds": 2,
                "http_request_attempted": True,
                "http_timing": {"prompt_tokens": 10, "total_seconds": 1},
            },
        },
    }


def reference():
    return {
        "policy": {"sha256": p.POLICY_SHA256},
        "decoded_evidence": [],
        "reference_sha256": "test-reference",
        "lookup": {(0, 1, 2, 3): "W001"},
        "refs": {
            "W001": {"judgments": {label: {"detected": False} for label in p.LABELS}}
        },
    }


def test_astra_excludes_uncertain_failed_unaligned():
    raw = fixture_report()
    raw["report"]["judgments"][0]["observed_detected"] = None
    result = astra_agreement([raw], reference())
    assert result["binary_comparable"] == 5
    raw["report"]["state"] = "failed"
    assert astra_agreement([raw], reference())["binary_comparable"] == 0


def test_missing_usage_not_zero_and_cost_estimate(tmp_path):
    folder = tmp_path / "measure" / "api-audit"
    folder.mkdir(parents=True)
    (folder / "worker.jsonl").write_text(
        "\n".join(
            map(
                json.dumps,
                [
                    {
                        "http_request_attempted": True,
                        "http_status": 200,
                        "input_tokens": 1000,
                    },
                    {"http_request_attempted": True, "http_status": 429},
                ],
            )
        )
    )
    result = audit_usage(tmp_path)
    assert result["usage_missing_requests"] == 1
    assert result["estimated_base_cost_usd"] is None
    assert result["reported_usage_base_cost_usd"] == pytest.approx(0.0001)
    assert result["billed_cost_verified"] is False


def test_analysis_keeps_planned_denominator_and_history(tmp_path):
    raw = fixture_report()
    (tmp_path / "raw.jsonl").write_text(json.dumps(raw) + "\n")
    manifest = {
        "settings": {**p.FROZEN, "real": True},
        "decisions_profile": p.VERSION,
        "transport": "decisions-json-v1",
        "event_policy": {"sha256": p.POLICY_SHA256},
        "decoded_evidence": [[]],
        "videos": [{"sha256": p.VIDEO_SHA256}],
        "http_timeouts": {"events": 20},
    }
    stage = {
        "lanes": {
            "events": {
                "expected_observations": 828,
                "reports": 1,
                "complete_valid_results": 1,
            }
        },
        "input_schedule_span_seconds": 56,
        "drain_elapsed_seconds": 4,
        "all_drained": True,
        "input_timing_valid": True,
        "result_records_missing": 0,
        "result_retention": {"valid": True},
        "shared_admission": {"active": 0, "waiting": 0},
    }
    summary = {
        "state": "completed",
        "jobs": [
            {
                "id": "measure",
                "phase": "measurement",
                "mode": "events",
                "repeat": 1,
                "streams": 12,
                "raw_reports": "raw.jsonl",
                "result": stage,
                "recording": {"complete": True},
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    result = analyze(tmp_path, reference())
    trial = result["trials"][0]
    assert trial["complete_rate"] == 1 / 828
    assert trial["within_three_seconds"]["on_time_coverage"] == 1 / 828
    assert trial["delivery"]["historical_complete_results"] == 1
    assert trial["tokens"]["output_tps"] is None
    assert not result["passes_target"] and not result["all_two_trials"]
    assert trial["audit_coverage"]["missing_attempts"] == 1
    assert not trial["valid"]
    page = (tmp_path / "decisions-comparison.html").read_text(encoding="utf-8")
    assert "未通过验收" in page and "未知" in page
    assert "不完整或不一致" in page


def audit_record(report):
    return {
        "stream_session_id": report["stream_session_id"],
        "request_id": report["request_id"],
        "http_request_attempted": True,
        "http_status": 200,
        "complete": True,
        "input_tokens": 1000,
    }


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "replaced", "status", "outcome"]
)
def test_audit_reconciles_identities_and_outcomes(problem):
    report = fixture_report()["report"]
    report["timing"]["http_status"] = 200
    rows = [audit_record(report)]
    assert audit_coverage([report], rows)["valid"]
    if problem == "missing":
        rows.clear()
    if problem == "duplicate":
        rows.append(rows[0])
    if problem == "replaced":
        rows[0]["request_id"] = "other"
    if problem == "status":
        rows[0]["http_status"] = 429
    if problem == "outcome":
        rows[0]["complete"] = False
    assert not audit_coverage([report], rows)["valid"]


def test_unknown_attempt_does_not_make_sent_completeness_over_100(tmp_path):
    test_analysis_keeps_planned_denominator_and_history(tmp_path)
    raw = fixture_report()
    raw["report"]["timing"].pop("http_request_attempted")
    (tmp_path / "raw.jsonl").write_text(json.dumps(raw) + "\n")
    result = analyze(tmp_path, reference())
    assert result["trials"][0]["complete_among_sent"] is None
    assert not result["trials"][0]["valid"]


def test_interrupted_suite_cannot_pass_even_with_two_timely_trials(tmp_path):
    import copy

    test_analysis_keeps_planned_denominator_and_history(tmp_path)
    document = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    first = document["jobs"][0]
    document["jobs"] = []
    for repeat in (1, 2):
        stage = copy.deepcopy(first)
        stage.update(
            id=f"measure-{repeat}",
            repeat=repeat,
            raw_reports=f"measure-{repeat}/raw.jsonl",
        )
        lane = stage["result"]["lanes"]["events"]
        lane.update(reports=828, complete_valid_results=828)
        rows = []
        for camera in range(1, 13):
            for tick in range(69):
                row = fixture_report()
                row["camera"] = camera
                row["report"].update(
                    window_id=f"window{tick}",
                    stream_session_id=f"stream{repeat}-{camera}",
                    request_id=f"req{tick}",
                )
                rows.append(row)
        folder = tmp_path / stage["id"]
        (folder / "api-audit").mkdir(parents=True)
        (folder / "raw.jsonl").write_text("\n".join(map(json.dumps, rows)))
        (folder / "api-audit" / "worker.jsonl").write_text(
            "\n".join(json.dumps(audit_record(row["report"])) for row in rows)
        )
        document["jobs"].append(stage)
    (tmp_path / "summary.json").write_text(json.dumps(document))
    assert analyze(tmp_path, reference())["passes_target"] is True
    document["state"] = "interrupted"
    (tmp_path / "summary.json").write_text(json.dumps(document))
    assert analyze(tmp_path, reference())["passes_target"] is False


def test_portable_bundle_excludes_credentials(tmp_path):
    from tools.decisions_benchmark.package import bundle

    target = tmp_path / "source.tar.gz"
    result = bundle(target)
    assert len(result["sha256"]) == 64
    with tarfile.open(target) as archive:
        names = archive.getnames()
        assert "tools/decisions_benchmark/adapter.py" in names
        assert "docs/guides/decisions-api-benchmark.md" in names
        assert not any(n.endswith(".env") or n.endswith(".mp4") for n in names)
    with pytest.raises(ValueError):
        bundle(target)


@pytest.mark.parametrize(
    "status,expected,cleanup_failure",
    [(200, 3, False), (429, 1, False), (200, 3, True)],
)
@pytest.mark.skipif(os.name != "posix", reason="POSIX-only Linux service lifecycle")
def test_visual_preflight_uses_three_native_calls(
    monkeypatch, tmp_path, status, expected, cleanup_failure
):
    import os

    from tools import vlm_multistream_benchmark as runner
    from tools.vlm_benchmark.lock import acquire

    calls = []
    _cfg, value = inspection()
    frames = [value.frames[0].jpeg_bytes] * 140

    class Frames(list):
        media_seconds = tuple(i / 2.5 for i in range(140))

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status, json=response_body(), headers={"retry-after": "60"}
        )

    monkeypatch.setattr(suite, "check_video", lambda _: None)
    monkeypatch.setattr(suite, "credential", lambda _: "test-key")
    monkeypatch.setattr(modes, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "decode", lambda *args: Frames(frames))
    monkeypatch.setattr(
        a,
        "create_vlm_http_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    if cleanup_failure:
        original_close = a.DecisionInspector.close

        def broken_close(self):
            original_close(self)
            raise OSError("fixture cleanup failure")

        monkeypatch.setattr(a.DecisionInspector, "close", broken_close)

    folder = tmp_path / "probe"
    if cleanup_failure:
        with pytest.raises(OSError):
            suite.preflight(tmp_path / "output.mp4", folder)
    elif status == 200:
        result = suite.preflight(tmp_path / "output.mp4", folder)
        assert result["state"] == "passed_preflight"
    else:
        with pytest.raises(ModelAPIError):
            suite.preflight(tmp_path / "output.mp4", folder)
    assert len(calls) == expected
    saved = json.loads((folder / "preflight.json").read_text(encoding="utf-8"))
    assert saved["passed"] is (status == 200 and not cleanup_failure)
    assert saved["formal_test_started"] is False
    audit = (
        next((folder / "api-audit").glob("*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(audit) == expected
    fd = acquire(tmp_path / "outputs" / "vlm-modes" / "run")
    os.close(fd)


def mock_decisions_factory(cfg, *, audit_dir=None):
    """Spawnable native-HTTP mock, with no external network or credentials."""

    def handler(request):
        if request.method == "HEAD":
            return httpx.Response(405)
        assert request.method == "POST" and request.url == p.ENDPOINT
        assert len(json.loads(request.content)["questions"]) == 6
        body = response_body()
        for answer in body["answers"]:
            answer.update(choice="negative")
        return httpx.Response(200, json=body)

    a.create_vlm_http_client = lambda **kw: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    return a.inspector(cfg, audit_dir=audit_dir)


def test_native_decisions_crosses_real_worker_and_result_bus(tmp_path):
    from functools import partial

    from benchmark_runtime_fixtures import jpeg

    from tools.benchmark_runtime.vlm_independent import IndependentVLMBranch

    cfg, _ = inspection()
    branch = IndependentVLMBranch(
        cfg,
        inspector_factory=partial(mock_decisions_factory, audit_dir=tmp_path / "audit"),
    )
    branch.start()
    try:
        for seq, at in enumerate([0, 0.4, 0.8]):
            branch.offer(jpeg(), at, seq)
        assert branch.wait_until_idle(10)
        report = branch.results.latest().payload
        assert report["analysis_complete"] and report["event_scope"] == "scene"
        assert len(report["judgments"]) == 6
        assert all(j["observed_detected"] is False for j in report["judgments"])
        assert report["timing"]["http_request_attempted"] is True
        assert report["timing"]["http_timing"]["prompt_tokens"] == 3221
        assert report["event_policy"]["sha256"] == p.POLICY_SHA256
        assert report["objects"] == []
        audit = [
            json.loads(line)
            for path in (tmp_path / "audit").glob("*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert audit_coverage([report], audit)["valid"]
    finally:
        branch.close(timeout_seconds=5)
