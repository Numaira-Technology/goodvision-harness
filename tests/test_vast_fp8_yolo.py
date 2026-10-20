"""Offline FP8 routing, lifecycle guards and unchanged tracked schema."""

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from benchmark_runtime_fixtures import job
from test_tracked_grounding import answer
from test_vast_vlm import stream, worker

from tools.tracked_grounding.detector import DetectorBroker
from tools.vast_vlm import benchmark, profile, runtime, yolo_runtime
from tools.vast_vlm import fp8_adapter as fp8
from tools.vast_vlm import fp8_profile as p
from tools.vast_vlm.fp8 import main
from tools.vlm_benchmark import modes
from tools.vlm_multistream_benchmark import drain_budget_seconds


@pytest.mark.parametrize(
    "action",
    [
        "plan",
        "prepare",
        "start",
        "start-detector",
        "preflight",
        "run",
        "bundle",
        "collect",
    ],
)
def test_offline_unless_explicit(action, monkeypatch, capsys):
    def forbidden(*a, **k):
        pytest.fail("plan accessed network/runtime/credentials")

    monkeypatch.setattr(runtime, "volume_root", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("tools.tracked_grounding.suite.read_api_key", forbidden)
    main([action])
    plan = json.loads(capsys.readouterr().out)
    assert plan["execute"] is False
    assert plan["formal_vlm_requests"] == 6624
    assert plan["event_policy"]["sha256"] == profile.POLICY_SHA256
    assert plan["revision"] == p.MODEL_REVISION
    assert plan["global_limit"] == 48


def test_fp8_isolated_from_bf16_and_drain_budget(tmp_path):
    cmd = runtime.serve_command(tmp_path, p=p)
    assert cmd[cmd.index("--quantization") + 1] == "fp8"
    assert cmd[cmd.index("--max-num-seqs") + 1] == "16"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.85"
    assert p.MODEL_REVISION in cmd[cmd.index("--model") + 1]
    assert "--quantization" not in runtime.serve_command(tmp_path)
    args = fp8.arguments(video="output.mp4", real=True)
    args.output_mode = "events"
    assert drain_budget_seconds(args, modes.configure(args)) >= 25
    assert args.person_ids_only and args.record and args.streams == [12]
    assert modes.configure(args).aiclub.provider == "local"


@pytest.mark.parametrize(
    "memory,name,arch,ok",
    [
        (48, "RTX 4090", (8, 9), True),
        (24, "RTX 4090", (8, 9), False),
        (48, "RTX 6000", (8, 9), False),
        (48, "RTX 4090", (8, 0), False),
    ],
)
def test_real_hardware_guard(memory, name, arch, ok):
    gpu = SimpleNamespace(
        total_memory=memory * 1024**3, name=name, major=arch[0], minor=arch[1]
    )
    if ok:
        p.validate_gpu(gpu)
    else:
        with pytest.raises(ValueError):
            p.validate_gpu(gpu)


@pytest.mark.parametrize("mode", ["events", "compact"])
def test_local_sse_person_pipeline_without_cloud_key(tmp_path, monkeypatch, mode):
    cfg = worker(mode)
    task = job(cfg, 0.8, 1)

    def transport(row, jpeg):
        return [
            {"track_id": "P0001", "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9}}
        ]

    captured = []

    def response(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert str(request.url) == p.BASE_URL + "/v1/chat/completions"
        assert "authorization" not in request.headers
        assert "provider" not in payload and "reasoning" not in payload
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream(
                answer() if mode == "compact" else dict.fromkeys(cfg.labels, False)
            ),
        )

    monkeypatch.setenv("LOCAL_INFERENCE_API_KEY", "not-to-be-forwarded")
    monkeypatch.setattr(
        benchmark,
        "create_vlm_http_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(response)),
    )
    with DetectorBroker(
        "http://127.0.0.1:8092", tmp_path, transport=transport
    ) as broker:
        for f in task.frames:
            broker.submit(
                0,
                task.stream_session_id,
                f.frame_sequence,
                f.timestamp_seconds,
                f.jpeg_bytes,
            )
        obj = fp8.inspector(cfg, broker_url=broker.endpoint)
        try:
            assert len(obj.inspect(task)) == 6
            assert obj.last_http_timing["completion_tokens"] == 80
            if mode == "compact":
                assert "P0001" in json.dumps(captured[0]["messages"])
        finally:
            obj.close()


def test_local_rejects_cloud_endpoint():
    cfg = worker("compact")
    cfg = replace(
        cfg,
        model_settings=replace(
            cfg.model_settings, endpoint="https://openrouter.ai/api/v1/chat/completions"
        ),
    )
    with pytest.raises(ValueError, match="frozen local"):
        fp8.inspector(cfg, broker_url="http://127.0.0.1:1/token")


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only Linux service lifecycle")
def test_yolo_tamper_not_silently_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "volume_root", lambda path: path)
    path = yolo_runtime.weights(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bad weights")
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("must not redownload")
    )
    with pytest.raises(ValueError, match="integrity"):
        yolo_runtime.prepare(tmp_path)


def test_stale_detector_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "volume_root", lambda path: path)
    (tmp_path / "runtime").mkdir()
    runtime.write_json(
        tmp_path / "runtime/detector-active.json",
        {
            "version": p.VERSION,
            "weights_sha256": p.YOLO_SHA256,
            "pid": 123,
            "start_ticks": "1",
        },
    )
    monkeypatch.setattr(runtime, "process_start_ticks", lambda pid: "2")
    monkeypatch.setattr(runtime, "boot_id", lambda: "new-boot")
    with pytest.raises(ValueError, match="matching live"):
        yolo_runtime.verify_active(tmp_path)


@pytest.mark.parametrize("preflight_only", [True, False])
@pytest.mark.parametrize("profile", ["fp8", "decisions"])
def test_model_suite_credentials_and_preflight_boundary(
    tmp_path, monkeypatch, preflight_only, profile
):
    import os

    from tools.tracked_decisions import suite as decisions
    from tools.tracked_grounding import suite

    adapter = fp8 if profile == "fp8" else decisions
    credentials = []

    def read_key(provider, env_file, *, required):
        assert provider == "openai_decisions" and required
        credentials.append(provider)
        return "offline-key"

    monkeypatch.setattr("tools.decisions_benchmark.suite.read_api_key", read_key)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-key-before-read")

    class Resource:
        endpoint = "http://127.0.0.1:1/token"

        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def check_ready(self):
            return {"gpu_ready": True}

        def begin(self, *a):
            pass

        def finish(self, *a):
            return {"complete": True}

        def drain(self):
            return {"ready": 1}

        def summary(self):
            return {"server_phases": {}}

        def submit(self, *a):
            pass

    calls = []

    def stage(options, base, *a, **kw):
        calls.append((options.output_mode, options.duration))
        assert kw["inspector_factory"].func is adapter.inspector
        assert base.aiclub.provider == (
            "local" if profile == "fp8" else "openai_decisions"
        )
        if profile == "decisions":
            assert os.environ["OPENAI_API_KEY"] == "offline-key"
            assert "audit_dir" in kw["inspector_factory"].keywords
        return {
            "all_drained": True,
            "input_timing_valid": True,
            "result_records_missing": 0,
            "result_retention": {"valid": True},
            "lanes": {
                "grounding": {"complete_valid_results": 1, "http_status_counts": {}}
            },
        }

    monkeypatch.setattr(
        suite, "read_api_key", lambda *a, **k: pytest.fail("cloud credentials")
    )
    monkeypatch.setattr(suite, "sha", lambda path: suite.VIDEO_SHA)
    monkeypatch.setattr(
        suite,
        "load_reference",
        lambda *a: {"policy": {"sha256": suite.POLICY_SHA}, "decoded_evidence": {}},
    )
    monkeypatch.setattr(
        "tools.vlm_benchmark.admission_storage.memory_root", lambda: tmp_path
    )
    monkeypatch.setattr(suite, "DetectorBroker", Resource)
    monkeypatch.setattr(suite, "PersonMonitor", Resource)
    monkeypatch.setattr(suite, "Telemetry", Resource)
    monkeypatch.setattr(suite, "decode", lambda *a: [])
    monkeypatch.setattr(suite, "run_stage", stage)
    monkeypatch.setattr(suite, "acquire", lambda *a: os.open(os.devnull, os.O_RDONLY))
    monkeypatch.setattr(suite.time, "sleep", lambda *a: None)
    monkeypatch.setattr("tools.tracked_grounding.report.analyze", lambda *a: {})
    output = tmp_path / "results"
    result = suite.run(
        tmp_path / "output.mp4",
        output,
        p.DETECTOR_URL,
        tmp_path,
        model_adapter=adapter,
        preflight_only=preflight_only,
    )
    assert result["state"] == "completed"
    assert len(calls) == (2 if preflight_only else 14)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["execute"] is True
    if profile == "fp8":
        assert manifest["settings"]["quantization"] == "FP8"
        assert credentials == []
    else:
        assert credentials == ["openai_decisions"]
        assert manifest["settings"]["decisions_person_tracking"] is True
    assert manifest["settings"]["global_limit"] == 48


def test_fp8_manifest_cannot_masquerade_as_bf16(tmp_path):
    from test_vast_vlm import model_fixture

    manifest = model_fixture(tmp_path)
    manifest.update(repo=p.MODEL_REPO, revision=p.MODEL_REVISION)
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "awq"}})
    )
    manifest["files"]["config.json"] = runtime.sha(tmp_path / "config.json")
    runtime.write_json(tmp_path / "verified.json", manifest)
    with pytest.raises(ValueError, match="FP8"):
        runtime.verify_model(tmp_path, p=p)
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "fp8"}})
    )
    manifest["files"]["config.json"] = runtime.sha(tmp_path / "config.json")
    runtime.write_json(tmp_path / "verified.json", manifest)
    assert runtime.verify_model(tmp_path, p=p)["revision"] == p.MODEL_REVISION
    with pytest.raises(ValueError, match="identity"):
        runtime.verify_model(tmp_path)


def test_exported_bundle_imports_without_original_checkout(tmp_path):
    import os
    import subprocess
    import sys

    from tools.vast_vlm import package

    target = tmp_path / "source.tar.gz"
    package.bundle(target)
    stage = tmp_path / "isolated"
    stage.mkdir()
    package.safe_extract(target, stage)
    env = dict(os.environ, PYTHONPATH=str(stage / "src") + os.pathsep + str(stage))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.vast_vlm.fp8_adapter import plan; "
            "from tools.tracked_grounding.report import analyze; "
            "assert plan()['formal_vlm_requests']==6624",
        ],
        cwd=stage,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (stage / "workers/detector_rich_label/tracker.py").is_file()
    assert (stage / "tools/vast_vlm/setup-yolo.sh").is_file()
    manifest = json.loads((stage / "bundle-manifest.json").read_text())
    assert manifest["source_identity"]["integration"] == (
        "main-grounding-isolated-benchmarks-v3"
    )
    for name in ("person_grounding.py", "person_detector.py"):
        relative = "tools/benchmark_runtime/" + name
        assert manifest["files"][relative] == runtime.sha(stage / relative)


def test_server_entrypoint_does_not_require_benchmark_dependencies(tmp_path):
    import os
    import subprocess
    import sys

    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('tools.tracked_grounding', 'goodvision_harness', 'av', 'httpx')
        if fullname.startswith(blocked):
            raise RuntimeError('server imported client dependency: ' + fullname)
sys.meta_path.insert(0, Guard())
import tools.vast_vlm.fp8
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_budget_counts_twelve_camera_warmups():
    args = fp8.arguments()
    budget = fp8.request_budget(args)
    assert budget["measurement_upper"] == 6624
    assert budget["warmup_upper"] == 12 * 8 * modes.planned_observations(10, 1.25)
    assert budget["preflight_upper"] == 2 * modes.planned_observations(2.4, 1.25)
    assert budget["total_upper"] == sum(
        budget[k] for k in ("measurement_upper", "warmup_upper", "preflight_upper")
    )
