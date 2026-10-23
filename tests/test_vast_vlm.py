"""Offline regression: no rental, downloaded weights or real inference."""

import json
import os
import tarfile
from dataclasses import replace

import httpx
import pytest
from benchmark_runtime_fixtures import config, job
from test_vlm_lean_grounding import payload

from goodvision_harness.model_api import ModelAPIError, ModelAPISettings
from goodvision_harness.runtime.scheduling import EvidenceFrame
from tools.vast_vlm import benchmark as b
from tools.vast_vlm import package, runtime
from tools.vast_vlm import profile as p
from tools.vast_vlm.__main__ import main
from tools.vast_vlm.report import astra_rows
from tools.vlm_benchmark import modes
from tools.vlm_benchmark.vision_timeliness import assess_timeliness
from tools.vlm_multistream_benchmark import LABELS, drain_budget_seconds


@pytest.mark.parametrize("action", ["plan", "prepare", "start", "run", "preflight"])
def test_default_is_offline_plan(monkeypatch, capsys, action):
    def forbidden(*a, **kw):
        pytest.fail("default must not execute/network/download")

    for name in ("prepare", "start"):
        monkeypatch.setattr(runtime, name, forbidden)
    for name in ("run", "preflight", "readiness"):
        monkeypatch.setattr(b, name, forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    assert main([action]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["request_budget"]["measurement_upper"] == 6624
    assert plan["total_request_upper_including_probe"] == 6770
    assert plan["policy"]["sha256"] == p.POLICY_SHA256
    assert plan["recordings"] == 6 and plan["execute"] is False


@pytest.mark.parametrize(
    "name,value",
    [
        ("model", "other"),
        ("backend", "openrouter"),
        ("streams", [1]),
        ("observation_fps", 1),
        ("endpoint", "https://openrouter.ai/api/v1/chat/completions"),
        ("global_limit", 72),
        ("vision_profile", "qwen35"),
        ("record", False),
        ("parallel_scheduling", "independent"),
    ],
)
def test_contract_drift_rejected(name, value):
    args = b.arguments()
    setattr(args, name, value)
    with pytest.raises(ValueError):
        modes.validate(args)


def worker(mode):
    return replace(
        config(),
        labels=LABELS,
        model=p.SERVED_MODEL,
        model_settings=ModelAPISettings(provider="local"),
        output_mode=mode,
        grounding_format="lean-v1",
    )


def stream(answer, finish="stop", done=True):
    def encode(data):
        return "data: " + json.dumps(data) + "\n\n"

    value = encode(
        {
            "model": p.SERVED_MODEL,
            "choices": [{"index": 0, "delta": {"content": json.dumps(answer)}}],
        }
    )
    value += encode({"choices": [{"delta": {}, "finish_reason": finish}]})
    value += encode(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 2400,
                "completion_tokens": 80,
            },
        }
    )
    return (value + ("data: [DONE]\n\n" if done else "")).encode()


@pytest.mark.parametrize("mode", ["events", "compact"])
def test_original_four_frames_sse_and_twenty_second_deadline(monkeypatch, mode):
    requests = []
    answer = dict.fromkeys(LABELS, False) if mode == "events" else payload()

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url == p.BASE_URL + "/v1/chat/completions"
        assert "Authorization" not in request.headers
        assert body["stream_options"] == {"include_usage": True}
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["max_tokens"] == (160 if mode == "events" else 1600)
        assert "provider" not in body and "reasoning" not in body
        images = [c for c in body["messages"][1]["content"] if c["type"] == "image_url"]
        assert len(images) == 4
        assert p.policy_check()["sha256"] == p.POLICY_SHA256
        return httpx.Response(
            200,
            content=stream(answer),
            headers={"content-type": "text/event-stream"},
        )

    def client(**kwargs):
        assert kwargs == {"trust_env": False}
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(b, "create_vlm_http_client", client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-used")
    monkeypatch.setenv("LOCAL_INFERENCE_API_KEY", "also-not-used")
    cfg = worker(mode)
    inspection = job(cfg, 1.2, 3)
    inspection = replace(
        inspection,
        frames=tuple(
            EvidenceFrame("vlm-run", i, i * 0.4, inspection.frames[0].jpeg_bytes)
            for i in range(4)
        ),
    )
    inspection = replace(
        inspection,
        request=replace(
            inspection.request,
            start_seconds=0,
            end_seconds=inspection.frames[-1].timestamp_seconds,
        ),
    )
    inspector = b.inspector(cfg)
    try:
        assert inspector.settings.timeout_seconds == 20
        assert len(inspector.inspect(inspection)) == 6
        assert inspector.last_http_timing["completion_tokens"] == 80
        assert inspector.last_http_timing["stream_first_content_seconds"] >= 0
        # Same six-class policy prose as the existing Alibaba inspector.
        original = modes.mode_inspector(
            replace(
                cfg,
                model_settings=ModelAPISettings(provider="openrouter"),
                parameters={"provider": {"order": ["alibaba"]}},
            )
        )
        try:
            assert (
                inspector.build_messages(inspection)[0]
                == original.build_messages(inspection)[0]
            )
        finally:
            original.close()
    finally:
        inspector.close()
    assert len(requests) == 1


@pytest.mark.parametrize("mode", ["events", "compact"])
@pytest.mark.parametrize("failure", ["429", "length", "truncated", "missing-label"])
def test_failures_are_not_repaired_or_retried(monkeypatch, mode, failure):
    answer = dict.fromkeys(LABELS, False) if mode == "events" else payload()
    if failure == "missing-label":
        (answer if mode == "events" else answer["e"]).pop("fall")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            429 if failure == "429" else 200,
            headers={"content-type": "text/event-stream"},
            content=stream(
                answer,
                "length" if failure == "length" else "stop",
                failure != "truncated",
            ),
        )

    monkeypatch.setattr(
        b,
        "create_vlm_http_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    cfg = worker(mode)
    inspector = b.inspector(cfg)
    try:
        with pytest.raises((ModelAPIError, ValueError)):
            inspector.inspect(job(cfg, 1.2, 3))
        assert len(calls) == 1
    finally:
        inspector.close()


def test_events_drain_budget_preserves_late_results():
    args = b.arguments()
    args.real, args.output_mode = True, "events"
    assert drain_budget_seconds(args, modes.configure(args)) == 25
    args.self_hosted_qwen35 = False
    assert drain_budget_seconds(args, modes.configure(args)) == 15


def model_fixture(tmp_path):
    files = {
        "config.json": "{}",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "preprocessor_config.json": "{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"x": "weights.safetensors"}}
        ),
        "weights.safetensors": "fake weights",
    }
    for name, value in files.items():
        (tmp_path / name).write_text(value)
    manifest = {
        "repo": p.MODEL_REPO,
        "revision": p.MODEL_REVISION,
        "files": {name: runtime.sha(tmp_path / name) for name in files},
    }
    runtime.write_json(tmp_path / "verified.json", manifest)
    return manifest


def test_model_integrity_and_identity(tmp_path):
    model_fixture(tmp_path)
    runtime.verify_model(tmp_path)
    (tmp_path / "weights.safetensors").write_text("truncated")
    with pytest.raises(ValueError, match="integrity"):
        runtime.verify_model(tmp_path)


def test_missing_manifest_never_means_ready(tmp_path):
    with pytest.raises(FileNotFoundError):
        runtime.verify_model(tmp_path)


def test_mount_check_does_not_accept_container_disk_or_tmpfs(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    for mounts in ("", f"1 0 0:1 / {tmp_path} rw - tmpfs tmpfs rw"):
        with pytest.raises(ValueError, match="mount"):
            runtime.volume_root(tmp_path, mounts)
    runtime.volume_root(tmp_path, f"1 0 0:1 / {tmp_path} rw - ext4 /dev/x rw")
    assert (tmp_path / ".goodvision-vlm-volume.json").exists()


def test_server_plan_cannot_load_other_models_or_offload(tmp_path):
    command = runtime.serve_command(tmp_path)
    assert p.MODEL_REVISION in command[command.index("--model") + 1]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--no-enable-prefix-caching" in command
    assert "--cpu-offload-gb" not in command
    assert "--language-model-only" not in command


def test_source_bundle_excludes_credentials_and_weights(tmp_path):
    target = tmp_path / "bundle.tar.gz"
    result = package.bundle(target)
    assert result["files"] > 50
    with tarfile.open(target) as archive:
        names = archive.getnames()
        assert "tools/vast_vlm/runtime.py" in names
        assert not any(".env" in n or n.endswith(".safetensors") for n in names)
        manifest = json.load(archive.extractfile("bundle-manifest.json"))
        for name, digest in manifest["files"].items():
            import hashlib

            assert (
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
            )


def test_export_is_results_only_and_never_overwrites(tmp_path):
    source = tmp_path / "result"
    source.mkdir()
    for name in ("manifest.json", "summary.json"):
        (source / name).write_text("{}")
    (source / ".env").write_text("secret")
    (source / "viewport.mp4").write_bytes(b"offline fixture")
    dest = tmp_path / "result.tar.gz"
    package.collect(source, dest)
    with tarfile.open(dest) as archive:
        assert set(archive.getnames()) == {
            "manifest.json",
            "summary.json",
            "viewport.mp4",
        }
    with pytest.raises(ValueError):
        package.collect(source, dest)


def report_row(lane="events", seconds=2, historical=True):
    return {
        "camera": 1,
        "report": {
            "state": "completed",
            "analysis_complete": True,
            "vlm_lane": lane,
            "window_id": "w1",
            "historical_only": historical,
            "timing": {
                "source_to_publish_seconds": seconds,
                "http_request_attempted": True,
            },
            "event_policy": {"sha256": p.POLICY_SHA256},
            "evidence": [{"frame_sequence": 3, "timestamp_seconds": 1.2}],
            "judgments": [
                {"event_label": label, "observed_detected": False} for label in LABELS
            ],
        },
    }


def test_parallel_requires_both_on_time_and_counts_historical():
    raw = [report_row(), report_row("grounding", 4)]
    result = assess_timeliness(raw, {"events", "grounding"})
    assert result["joint"]["all_cameras_pass"] is False
    assert result["lanes"]["events"] != result["joint"]
    assert result["joint"]["per_camera"][0]["planned_windows"] == 69


def test_astra_only_scores_complete_aligned_decided_rows():
    reference = {
        "lookup": {(3,): "w1"},
        "refs": {"w1": {"judgments": {label: {"detected": False} for label in LABELS}}},
    }
    raw = [report_row()]
    raw[0]["report"]["judgments"][0]["observed_detected"] = None
    result = astra_rows(raw, reference)
    assert result["binary_comparable"] == 5 and result["agreement"] == 1
    raw[0]["report"]["state"] = "failed"
    assert astra_rows(raw, reference)["binary_comparable"] == 0


def test_preflight_reuses_original_four_frames_and_two_parsers(monkeypatch):
    from benchmark_runtime_fixtures import jpeg

    from tools import vlm_multistream_benchmark as multistream

    class Frames(list):
        media_seconds = tuple(i / 2.5 for i in range(14))

    calls = []
    monkeypatch.setattr(b, "check_video", lambda video: None)
    monkeypatch.setattr(b, "readiness", lambda: None)
    monkeypatch.setattr(multistream, "decode", lambda *a: Frames([jpeg()] * 14))

    def handler(request):
        body = json.loads(request.content)
        images = [c for c in body["messages"][1]["content"] if c["type"] == "image_url"]
        assert len(images) == 4
        answer = dict.fromkeys(LABELS, False) if not calls else payload()
        calls.append(body)
        return httpx.Response(
            200,
            content=stream(answer),
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        b,
        "create_vlm_http_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert b.preflight("unused") == [
        {"mode": "events", "complete": True},
        {"mode": "compact", "complete": True},
    ]
    assert len(calls) == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only Linux service lifecycle")
def test_prepare_reuses_verified_files_without_network(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    root = runtime.model_dir(tmp_path)
    root.mkdir(parents=True)
    expected = model_fixture(root)
    monkeypatch.setattr(runtime, "volume_root", lambda path: path)

    def forbidden(*a, **kw):
        pytest.fail("a complete model must not be downloaded again")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=forbidden,
            snapshot_download=forbidden,
        ),
    )
    assert runtime.prepare(tmp_path) == expected


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only Linux service lifecycle")
def test_bad_model_stays_failed_and_is_not_silently_repaired(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    root = runtime.model_dir(tmp_path)
    root.mkdir(parents=True)
    model_fixture(root)
    (root / "weights.safetensors").unlink()
    monkeypatch.setattr(runtime, "volume_root", lambda path: path)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: pytest.fail("do not silently change a validated model"),
            snapshot_download=lambda **kw: pytest.fail("unexpected download"),
        ),
    )
    with pytest.raises(ValueError, match="integrity"):
        runtime.prepare(tmp_path)


def test_startup_failure_is_not_reported_as_ready():
    from types import SimpleNamespace

    with pytest.raises(RuntimeError, match="before readiness"):
        runtime.wait_for_server(SimpleNamespace(poll=lambda: 1))


def test_preflight_runs_under_the_same_suite_lock(monkeypatch, tmp_path):
    observed = []
    args = b.arguments()
    args.real, args.video = True, [tmp_path / "output.mp4"]
    monkeypatch.setattr(modes, "ROOT", tmp_path)
    monkeypatch.setattr(modes, "validate", lambda args: None)
    monkeypatch.setattr(modes, "acquire", lambda root: observed.append("locked") or 123)
    monkeypatch.setattr(modes.os, "close", lambda fd: observed.append("closed"))
    monkeypatch.setattr(modes, "_run", lambda args: observed.append("run"))
    modes.run(args, before_run=lambda: observed.append("preflight"))
    assert observed == ["locked", "preflight", "run", "closed"]


def test_report_uses_828_denominator_even_with_only_one_reply(tmp_path):
    from tools.vast_vlm.report import analyze

    manifest = {
        "settings": {**p.FROZEN, "real": True},
        "event_policy": {"sha256": p.POLICY_SHA256},
        "decoded_evidence": [[]],
        "videos": [{"sha256": p.VIDEO_SHA256}],
        "http_timeouts": {"events": 20, "compact": 20},
        "transport": "sse-v1",
    }
    reference = {
        "policy": {"sha256": p.POLICY_SHA256},
        "decoded_evidence": [],
        "reference_sha256": "fixture",
        "lookup": {(3,): "w1"},
        "refs": {"w1": {"judgments": {k: {"detected": False} for k in LABELS}}},
    }
    raw = report_row()
    (tmp_path / "raw.jsonl").write_text(json.dumps(raw) + "\n")
    summary = {
        "state": "complete",
        "jobs": [
            {
                "phase": "measurement",
                "mode": "events",
                "repeat": 1,
                "streams": 12,
                "raw_reports": "raw.jsonl",
                "recording": {"complete": True},
                "telemetry": {},
                "result": {
                    "all_drained": True,
                    "input_timing_valid": True,
                    "result_records_missing": 0,
                    "result_retention": {"valid": True},
                    "shared_admission": {"active": 0, "waiting": 0},
                    "input_schedule_span_seconds": 55.8,
                    "drain_elapsed_seconds": 3,
                    "lanes": {
                        "events": {
                            "expected_observations": 828,
                            "reports": 1,
                            "complete_valid_results": 1,
                        }
                    },
                },
            }
        ],
    }
    for name, data in (("manifest.json", manifest), ("summary.json", summary)):
        runtime.write_json(tmp_path / name, data)
    result = analyze(tmp_path, reference)
    assert result["all_six_trials"] is False
    row = result["trials"][0]
    assert row["sent_rate"] == row["complete_rate"] == 1 / 828
    assert row["astra"]["binary_comparable"] == 6
    assert row["timeliness"]["lanes"]["events"]["all_cameras_pass"] is False
    assert row["tokens"]["completion_tokens"]["reported_requests"] == 0
    assert row["tokens"]["api_completion_tps"]["count"] == 0
