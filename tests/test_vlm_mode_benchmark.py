import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
from benchmark_runtime_fixtures import ROOT, await_condition, config, jpeg
from test_vlm_multistream_benchmark import report

from goodvision_harness.config import load_config
from goodvision_harness.model_api import environment_api_key
from tools.benchmark_runtime.vlm_admission import AdmissionSettings, SharedAdmission
from tools.benchmark_runtime.vlm_independent import IndependentVLMBranch
from tools.vlm_benchmark import lock as suite_lock
from tools.vlm_benchmark import modes
from tools.vlm_benchmark.mode_report import render_modes
from tools.vlm_benchmark.telemetry import parse_metrics, server_deltas
from tools.vlm_multistream_benchmark import (
    LABELS,
    paired_summary,
    release_schedule,
    simulated_factory,
    summarize,
)


def lane_report(lane, window, **kwargs):
    return {**report(**kwargs), "vlm_lane": lane, "window_id": window}


def test_lanes_keep_separate_denominators_and_late_results():
    rows = [
        lane_report("information", "a"),
        lane_report("grounding", "a", seconds=4),
        lane_report("grounding", "b", error="timeout"),
    ]
    event = summarize(rows, {}, 3, 3, 3, lane="events")
    grounding = summarize(rows, {}, 3, 3, 3, lane="grounding")
    assert event["complete_valid_results"] == grounding["complete_valid_results"] == 1
    assert event["without_complete_result"] == grounding["without_complete_result"] == 2
    assert event["within_seconds"]["3"]["count"] == 1
    assert grounding["within_seconds"]["3"]["count"] == 0
    assert grounding["within_seconds"]["5"]["fraction_of_planned"] == 1 / 3
    assert grounding["source_to_publish_seconds"]["p95"] == 4
    assert grounding["errors"] == {"timeout": 1}
    assert grounding["request_phase_metrics"]["reorder_wait_seconds"]["p50"] is None


def test_pairing_requires_same_camera_and_window_not_nearby_frames():
    cameras = [
        [
            lane_report("information", "a"),
            lane_report("grounding", "a", seconds=4),
            lane_report("information", "b"),
            lane_report("grounding", "c", seconds=2),
        ],
        [lane_report("grounding", "b", seconds=2)],
    ]
    pairs = paired_summary(cameras, 8)
    assert pairs["both_complete"] == 1
    assert pairs["both_complete_fraction"] == 1 / 8
    assert pairs["without_both_complete"] == 7
    assert pairs["events_complete_without_grounding"] == 1
    assert pairs["grounding_complete_without_events"] == 2
    assert pairs["grounding_minus_events_seconds"]["p50"] == 3
    assert pairs["both_published_seconds"]["p95"] == 4


def test_mode_specific_schema_never_leaks_into_grounding(monkeypatch):
    monkeypatch.setattr(modes, "_inspector", lambda cfg: cfg)
    cfg = replace(config(), parameters={"temperature": 0})
    event = modes.mode_inspector(replace(cfg, output_mode="events"))
    ground = modes.mode_inspector(replace(cfg, output_mode="compact"))
    assert event.parameters["response_format"]["type"] == "json_schema"
    assert ground.parameters["response_format"] == {"type": "json_object"}
    assert cfg.parameters == {"temperature": 0}


def test_same_window_queue_replacement_is_visible_separately():
    rows = [
        [
            lane_report("information", "a", error="admission_superseded"),
            lane_report("grounding", "a"),
        ]
    ]
    pairs = paired_summary(rows, 1)
    assert pairs["both_complete"] == 0
    assert pairs["paired_admission_superseded"] == 1
    assert pairs["grounding_complete_without_events"] == 1


@pytest.mark.parametrize("failure", ["429", "invalid_soak"])
def test_runner_stops_and_preserves_report_on_failure(tmp_path, monkeypatch, failure):
    args = modes.parser().parse_args(
        [
            "--streams",
            "12",
            "--modes",
            "events",
            "--repeats",
            "1",
            "--duration",
            "3",
            "--soak-seconds",
            "30",
            "--output",
            str(tmp_path / "run"),
        ]
    )

    class NoTelemetry:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def summary(self):
            return {}

    calls = []

    def fake_stage(options, base, count, frames, location, repeat, **kwargs):
        calls.append(location.name)
        data = summarize([report()], {}, 3, 1, 3)
        if failure == "429":
            data["http_status_counts"] = {"429": 1}
        return {
            "streams": count,
            "repeat": repeat,
            "lanes": {"events": data},
            "per_stream": [{"lanes": {"events": data}}] * count,
            "input_timing_valid": True,
            "all_drained": options.duration != 30,
            "result_records_missing": 0,
            "paired_windows": None,
            "nominal_requests_per_second": 15,
            "rate_limit_below_offered_load": False,
        }

    monkeypatch.setattr(modes, "Telemetry", NoTelemetry)
    monkeypatch.setattr(modes, "run_stage", fake_stage)
    with pytest.raises(RuntimeError, match=r"429|invalid input"):
        modes.run(args)
    result = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert result["state"] == "failed"
    assert result["jobs"]
    if failure == "429":
        assert len(calls) == 1
    else:
        assert result["jobs"][-1]["phase"] == "soak"


def test_manifest_fingerprints_nested_scene_implementations(tmp_path, monkeypatch):
    args = modes.parser().parse_args(["--output", str(tmp_path / "run")])

    def stop_after_manifest(*args, **kwargs):
        raise RuntimeError("stop after manifest")

    monkeypatch.setattr(modes, "Telemetry", lambda *_: nullcontext())
    monkeypatch.setattr(modes, "run_stage", stop_after_manifest)
    with pytest.raises(RuntimeError, match="stop after manifest"):
        modes.run(args)
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest["source_hashes"]
    for name in (
        "runtime/scene.py",
        "runtime/scheduling.py",
        "events/state_machine.py",
        "actions/alerts.py",
        "actions/results.py",
        "evaluation/replay.py",
    ):
        source = ROOT / "src/goodvision_harness" / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashes[str(source.relative_to(ROOT))] == digest


def test_three_arm_order_budget_and_rate_limit_guard():
    args = modes.parser().parse_args([])
    modes.validate(args)
    schedule = list(modes.jobs(args))
    assert [j["mode"] for j in schedule[:6]] == [
        "events",
        "compact",
        "parallel",
        "parallel",
        "compact",
        "events",
    ]
    assert len(schedule) == 24
    assert modes.request_budget(args)["measurement_upper"] == 14800
    args.request_rate = 20
    with pytest.raises(ValueError, match="30/s"):
        modes.validate(args)
    args.allow_rate_limited = True
    modes.validate(args)


def test_request_budget_counts_integer_windows_for_each_short_trial():
    args = modes.parser().parse_args(
        [
            "--streams",
            "1",
            "--duration",
            "3",
            "--repeats",
            "1",
            "--observation-fps",
            "0.25",
        ]
    )
    # Each arm still has one window per lane, even at 0.25 Hz for 3 seconds.
    assert modes.request_budget(args) == {
        "measurement_upper": 4,
        "preflight_and_warmup_upper": 8,
        "optional_soak_upper": 0,
        "total_upper": 12,
    }


def test_missing_video_leaves_a_failed_report(tmp_path):
    args = modes.parser().parse_args(
        ["--video", str(tmp_path / "missing.mp4"), "--output", str(tmp_path / "run")]
    )
    with pytest.raises(FileNotFoundError):
        modes.run(args)
    result = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert result["state"] == "failed"
    assert result["reason"] == "FileNotFoundError"
    assert not result["jobs"]


def test_stage_exception_keeps_failed_stage_visible(tmp_path, monkeypatch):
    args = modes.parser().parse_args(["--output", str(tmp_path / "run")])

    def broken_stage(*args, **kwargs):
        raise RuntimeError("test startup failure")

    monkeypatch.setattr(modes, "run_stage", broken_stage)
    with pytest.raises(RuntimeError, match="test startup failure"):
        modes.run(args)
    result = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert result["state"] == "failed"
    assert result["active_stage"] == "preflight-events"
    assert "preflight-events" in (args.output / "report.html").read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(suite_lock.fcntl is None, reason="real runner requires POSIX")
def test_real_runner_releases_lock_on_error_and_rejects_overlapping_runs(
    tmp_path, monkeypatch
):
    args = modes.parser().parse_args(
        ["--real", "--backend", "openrouter", "--video", "example.mp4"]
    )
    monkeypatch.setattr(modes, "ROOT", tmp_path)
    monkeypatch.delenv(suite_lock.LOCK_ENV, raising=False)
    lock_root = tmp_path / "outputs/vlm-modes/run"

    def broken_run(_):
        with pytest.raises(RuntimeError, match="another VLM benchmark"):
            suite_lock.acquire(lock_root)
        raise RuntimeError("test run failure")

    monkeypatch.setattr(modes, "_run", broken_run)
    with pytest.raises(RuntimeError, match="test run failure"):
        modes.run(args)
    fd = suite_lock.acquire(lock_root)
    try:
        with pytest.raises(RuntimeError, match="another VLM benchmark"):
            modes.run(args)
    finally:
        os.close(fd)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_nonfinite_load_cannot_start(value):
    args = modes.parser().parse_args([])
    args.duration = value
    with pytest.raises(ValueError, match="duration"):
        modes.validate(args)


def test_real_local_run_requires_deployment_record_and_safe_metrics_origin():
    args = modes.parser().parse_args(
        ["--real", "--backend", "local", "--video", "example.mp4"]
    )
    with pytest.raises(ValueError, match="quantization"):
        modes.validate(args)
    args.model_revision, args.quantization = "revision", "4bit"
    args.engine_version, args.cache_policy = "test-engine", "disabled"
    args.metrics_url = "http://elsewhere:8000/metrics"
    with pytest.raises(ValueError, match="origin"):
        modes.validate(args)
    args.metrics_url = "http://127.0.0.1:8000/metrics"
    modes.validate(args)


def test_local_configuration_does_not_inherit_cloud_auth_or_routing(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "cloud-secret-test-only")
    args = modes.parser().parse_args(["--backend", "local"])
    base = modes.configure(args)
    assert base.model_api.provider == "local"
    assert base.model_api.api_key_env == "LOCAL_INFERENCE_API_KEY"
    assert base.inference.request_parameters == {
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
    }


def acknowledged_mode_results(args, base, tmp_path, *, inspector_factory=None):
    """Exercise real processes/IPC without measuring this CI host's capacity.

    Acknowledge both lanes before advancing each logical observation window.
    Real-time sender lag remains covered by the benchmark scheduler tests.
    """
    gate_settings = AdmissionSettings(
        str(tmp_path / "admission.sqlite"),
        args.global_limit,
        args.per_stream_limit,
        args.request_rate,
        args.queue_timeout,
    )
    cfg = replace(
        config(tmp_path.name),
        labels=LABELS,
        model_settings=base.model_api,
        model=base.inference.model,
        api_key=environment_api_key(base.model_api) if args.real else "",
        parameters=dict(base.inference.request_parameters)
        if args.real
        else {
            "simulation_seconds": args.simulated_latency,
            "simulation_grounding_seconds": args.simulated_grounding_latency,
        },
        options={"max_tokens": base.runtime.scene_max_tokens},
        observation_fps=args.observation_fps,
        max_in_flight=args.local_concurrency,
        output_mode=args.output_mode,
        admission=gate_settings,
    )
    branch = IndependentVLMBranch(
        cfg,
        inspector_factory=inspector_factory or simulated_factory,
    )
    lanes = (
        ("events", "grounding") if args.output_mode == "parallel" else ("grounding",)
    )
    sample_fps = max(2.0, args.observation_fps * 2)
    _, releases = release_schedule(
        1, args.duration, sample_fps, args.observation_fps, "aligned"
    )
    # These cases use three 0.8-second windows at 1.25 observations/s.
    assert args.duration == 3 and args.observation_fps == 1.25
    endpoints = (2, 4, 6)
    try:
        branch.start()
        await_condition(lambda: branch.health()["state"] == "running")
        buses = [branch.results]
        if args.output_mode == "parallel":
            buses.append(branch.grounding_results)
        for at, actions in releases:
            ((_, sequence),) = actions
            assert branch.offer(jpeg(), at, sequence), branch.stats()
            if sequence in endpoints:
                count = endpoints.index(sequence) + 1
                await_condition(
                    lambda count=count: all(
                        len(bus.records_after(0)) >= count for bus in buses
                    )
                )
        branch.finish_source()
        await_condition(lambda: branch.stats()["state"] == "drained")
        stats = branch.stats()
        records = [bus.records_after(0) for bus in buses]
        assert all([r.cursor for r in rows] == [1, 2, 3] for rows in records)
        rows = [r.payload for rr in records for r in rr]
    finally:
        branch.close(timeout_seconds=6)
    lane_results = {
        lane: summarize(rows, stats, args.duration, 3, args.deadline, lane=lane)
        for lane in lanes
    }
    return {
        "all_drained": stats["state"] == "drained",
        # This acknowledged fixture is not a real-time capacity measurement.
        "input_timing_valid": False,
        "result_records_missing": 0,
        "streams": 1,
        "repeat": 1,
        "nominal_requests_per_second": args.observation_fps * len(lanes),
        "rate_limit_below_offered_load": False,
        "per_stream": [
            {
                **summarize(rows, stats, args.duration, 3, args.deadline),
                "lanes": lane_results,
            }
        ],
        "lanes": lane_results,
        "paired_windows": paired_summary([rows], 3)
        if args.output_mode == "parallel"
        else None,
        "shared_admission": SharedAdmission(gate_settings).snapshot(),
    }


def test_local_http_endpoint_receives_correct_schemas_and_records_timings(
    tmp_path, monkeypatch
):
    """Real transport/parse/IPC against a local stub, never paid model inference."""
    monkeypatch.setenv("INFERENCE_API_KEY", "cloud-secret-test-only")
    monkeypatch.delenv("LOCAL_INFERENCE_API_KEY", raising=False)
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((body, self.headers.get("Authorization")))
            fmt = body["response_format"]
            answer = (
                {label: False for label in fmt["json_schema"]["schema"]["required"]}
                if fmt["type"] == "json_schema"
                else {"no_events": 0.9}
            )
            raw = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(answer),
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        args = modes.parser().parse_args(["--backend", "local"])
        args.endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
        args.real, args.duration, args.output_mode, args.deadline = (
            True,
            3,
            "parallel",
            3,
        )
        stage = acknowledged_mode_results(
            args,
            modes.configure(args),
            tmp_path,
            inspector_factory=modes.mode_inspector,
        )
        assert stage["all_drained"]
        assert stage["paired_windows"]["both_complete"] == 3
        assert len(seen) == 6
        assert all(key is None for _, key in seen)
        for lane in ("events", "grounding"):
            phases = stage["lanes"][lane]["request_phase_metrics"]
            assert phases["inspection_timing.request_prepare_seconds"]["count"] == 3
            assert phases["http_timing.completion_tokens"]["p50"] == 20
        assert sum(b["max_tokens"] == 160 for b, _ in seen) == 3
        assert sum(b["max_tokens"] == 1600 for b, _ in seen) == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def snapshot(count, seconds):
    return parse_metrics(
        f'vllm:request_queue_time_seconds_count{{model_name="test"}} {count}\n'
        f'vllm:request_queue_time_seconds_sum{{model_name="test"}} {seconds}\n'
        "unrelated_credentials_or_messages 100\n"
        "vllm:num_requests_waiting 2\n"
        "vllm:request_decode_time_seconds_sum NaN\n"
    )


def test_default_uses_openrouter_model_without_local_deployment():
    args = modes.parser().parse_args([])
    assert not args.real
    assert args.backend == "openrouter"
    assert args.model == "qwen/qwen3-vl-30b-a3b-instruct"
    base = modes.configure(args)
    assert base.model_api.provider == "openrouter"
    assert base.inference.model == args.model
    assert "reasoning" not in base.inference.request_parameters
    assert base.inference.request_parameters["provider"]["sort"] == "latency"
    assert base.inference.request_parameters["provider"]["require_parameters"]
    hybrid = modes.configure(modes.parser().parse_args(["--model", "qwen/qwen3.8-27b"]))
    assert hybrid.inference.request_parameters["reasoning"] == {"enabled": False}
    args.real, args.video = True, ["example.mp4"]
    modes.validate(args)  # No local weights/revision/GPU metadata needed.
    args.metrics_url = "http://127.0.0.1:8000/metrics"
    with pytest.raises(ValueError, match="external API"):
        modes.validate(args)


def test_server_metrics_use_stage_delta_not_lifetime_mean():
    before, after = snapshot(100, 1000), snapshot(102, 1001)
    assert len(before) == 3
    phases = server_deltas([before, after])
    queue = phases["vllm:request_queue_time_seconds"]
    assert queue["available"]
    assert queue["finished_request_count"] == 2
    assert queue["mean_seconds"] == 0.5
    assert not phases["vllm:request_decode_time_seconds"]["available"]


def test_missing_metrics_and_resets_are_not_zero_latency():
    name = "vllm:request_queue_time_seconds"
    assert (
        server_deltas([None, snapshot(1, 1)])[name]["reason"]
        == "missing_boundary_metrics"
    )
    assert (
        server_deltas([snapshot(10, 10), snapshot(1, 1), snapshot(20, 20)])[name][
            "reason"
        ]
        == "counter_reset"
    )
    assert (
        server_deltas([snapshot(1, 1), {}])[name]["reason"] == "metric_series_changed"
    )
    assert (
        server_deltas([snapshot(1, 1), None, snapshot(2, 2)])[name]["reason"]
        == "sampling_gap"
    )


def test_different_sum_and_count_series_are_not_combined():
    name = "vllm:request_queue_time_seconds"
    before = parse_metrics(f'{name}_sum{{worker="a"}} 1\n{name}_count{{worker="b"}} 1')
    after = {key: value + 1 for key, value in before.items()}
    assert server_deltas([before, after])[name]["reason"] == "unpaired_sum_count"


@pytest.mark.parametrize("mode", ["compact", "parallel"])
def test_production_branch_collects_grounding_bus_and_preserves_old_event_view(
    tmp_path, mode
):
    args = SimpleNamespace(
        real=False,
        duration=3,
        simulated_latency=0.01,
        simulated_grounding_latency=0.15,
        observation_fps=1.25,
        output_mode=mode,
        local_concurrency=4,
        global_limit=16,
        per_stream_limit=4,
        request_rate=40,
        queue_timeout=1,
        deadline=3,
    )
    stage = acknowledged_mode_results(args, load_config(ROOT / "config.yaml"), tmp_path)
    assert stage["all_drained"]
    assert stage["result_records_missing"] == 0
    assert stage["lanes"]["grounding"]["complete_valid_results"] == 3
    assert stage["per_stream"][0]["complete_valid_results"] == 3
    if mode == "parallel":
        assert stage["lanes"]["events"]["complete_valid_results"] == 3
        assert stage["paired_windows"]["both_complete"] == 3
        # The host scheduler can delay either process independently. Exact
        # pairing arithmetic is checked with fixed timestamps above.
        assert (
            stage["paired_windows"]["grounding_minus_events_seconds"]["p50"] is not None
        )
        assert stage["shared_admission"]["grants"] == 6
    document = {
        "settings": {"real": False, "backend": "local"},
        "state": "completed",
        "event_labels": ["fight"],
        "jobs": [
            {
                "id": "example",
                "mode": mode,
                "phase": "measurement",
                "result": stage,
                "telemetry": {},
                "raw_reports": "reports.jsonl",
                "telemetry_file": "telemetry.jsonl",
            }
        ],
    }
    render_modes(tmp_path, document)
    assert "grounding" in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "模拟调度" in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert (
        json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["jobs"][0][
            "result"
        ]
        == stage
    )
