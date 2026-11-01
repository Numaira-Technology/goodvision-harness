"""Local HTTP/process control tests; no downloaded weights, GPU or paid requests."""

import base64
import io
import json
import os
import socket
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from goodvision_harness.grounding import SAFETY_EVENT_LABELS
from tools.vlm_benchmark import local_model as model
from tools.vlm_benchmark import local_model_probe as probe
from tools.vlm_benchmark import modes, run_remote


def deployment(tmp_path, port=8000):
    return {
        "state": "installed",
        "profile": model.profile(),
        "repo": str(model.ROOT),
        "gpu_index": 0,
        "port": port,
        "model_path": str(tmp_path / "weights"),
        "engine_requirements_sha256": model.digest(
            model.HERE / "model-requirements.txt"
        ),
    }


def test_pinned_launch_keeps_vision_and_bounded_single_gpu(tmp_path):
    data = deployment(tmp_path)
    cmd = model.serve_command(tmp_path, data)
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert json.loads(cmd[cmd.index("--limit-mm-per-prompt") + 1]) == {
        "image": 4,
        "video": 0,
    }
    assert "--trust-remote-code" not in cmd
    assert cmd[cmd.index("--cpu-offload-gb") + 1] == "0"
    assert "--no-enable-prefix-caching" in cmd
    assert cmd[cmd.index("--max-num-seqs") + 1] == "8"
    assert cmd[cmd.index("--quantization") + 1] == "compressed-tensors"
    assert json.loads(cmd[cmd.index("--default-chat-template-kwargs") + 1]) == {
        "enable_thinking": False
    }


@pytest.mark.parametrize(
    "name,total,free,driver,error",
    [
        ("RTX 4090", 24564, 23000, "580.82.09", None),
        ("RTX 4090", 24564, 23000, "570.82", "driver"),
        ("RTX 4090", 24564, 5000, "580.82", "memory"),
        ("RTX 3090", 24564, 23000, "580.82", "4090"),
    ],
)
def test_hardware_rejection_before_download(name, total, free, driver, error):
    line = f"0, NVIDIA GeForce {name}, GPU-example, {total}, {free}, {driver}"
    if error:
        with pytest.raises(RuntimeError, match=error):
            model.parse_gpu(line, 0, model.profile())
    else:
        assert model.parse_gpu(line, 0, model.profile())["uuid"] == "GPU-example"


def test_clean_local_environment_does_not_inherit_keys_or_engine_overrides(monkeypatch):
    for key in (
        "VLLM_API_KEY",
        "VLLM_USE_V1",
        "OPENROUTER_API_KEY",
        "LOCAL_INFERENCE_API_KEY",
    ):
        monkeypatch.setenv(key, "not-a-real-key")
    env = model.clean_environment()
    assert not any(k.startswith("VLLM_") for k in env)
    assert "OPENROUTER_API_KEY" not in env and "LOCAL_INFERENCE_API_KEY" not in env


def test_available_memory_accounts_for_container_limit(tmp_path):
    proc = tmp_path / "meminfo"
    proc.write_text("MemAvailable: 104857600 kB\n")
    (tmp_path / "memory.max").write_text(str(24 * 1024**3))
    (tmp_path / "memory.current").write_text(str(4 * 1024**3))
    assert model.available_memory(proc, tmp_path) == 20 * 1024**3
    (tmp_path / "memory.max").write_text("max")
    assert model.available_memory(proc, tmp_path) == 100 * 1024**3


def test_preflight_failure_is_recorded_before_any_install(tmp_path, monkeypatch):
    def reject(*_):
        raise RuntimeError("driver too old")

    monkeypatch.setattr(model, "preflight", reject)
    monkeypatch.setattr(model, "run", lambda *_a, **_kw: pytest.fail("started install"))
    with pytest.raises(RuntimeError, match="driver too old"):
        model.install(SimpleNamespace(root=tmp_path, gpu=0, port=8000))
    assert (
        json.loads((tmp_path / "install-state.json").read_text())["state"] == "failed"
    )
    assert not (tmp_path / "deployment.json").exists()


@pytest.mark.parametrize("state", ["STOPPED", "RUNNING", "unreachable"])
def test_stop_requires_confirmed_process_state(tmp_path, monkeypatch, state):
    monkeypatch.setattr(
        model,
        "control",
        lambda *_a, **_kw: SimpleNamespace(returncode=3, stdout=f"qwen-vlm {state}"),
    )
    if state == "STOPPED":
        model.stop(tmp_path)
        assert json.loads((tmp_path / "health.json").read_text())["state"] == "stopped"
    else:
        with pytest.raises(RuntimeError, match="could not confirm"):
            model.stop(tmp_path)
        assert not (tmp_path / "health.json").exists()


def test_checkpoint_accepts_single_file_but_rejects_missing_vision_or_shards(tmp_path):
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "vision_config": {"depth": 27},
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {"group0": {"weights": {"num_bits": 4}}},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    for name in (
        "model.safetensors",
        "tokenizer_config.json",
        "preprocessor_config.json",
    ):
        (tmp_path / name).write_text("fixture")
    model.validate_checkpoint(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="shards"):
        model.validate_checkpoint(tmp_path)
    cfg["vision_config"] = None
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match="multimodal"):
        model.validate_checkpoint(tmp_path)


def test_existing_deployment_cannot_silently_change_profile(tmp_path):
    data = deployment(tmp_path)
    model.write_json(tmp_path / "deployment.json", data)
    assert model.load_install(tmp_path) == data
    data["profile"]["max_num_seqs"] = 100
    model.write_json(tmp_path / "deployment.json", data)
    with pytest.raises(RuntimeError, match="profile changed"):
        model.load_install(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="Linux model operations use flock")
def test_operations_are_serialized_and_release_after_error(tmp_path):
    with pytest.raises(ValueError):
        with model.operation_lock(tmp_path):
            with pytest.raises(RuntimeError, match="another model"):
                with model.operation_lock(tmp_path):
                    pytest.fail("overlapping operation")
            raise ValueError("fake failure")
    with model.operation_lock(tmp_path):
        pass


def test_start_failure_stops_owned_service_and_preserves_probe_details(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(root=tmp_path, wait_seconds=1)
    calls = []

    def control(root, *cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="qwen-vlm RUNNING pid 123")

    def health(*_):
        model.write_json(
            tmp_path / "health.json", {"state": "failed", "failed_check": "vision_red"}
        )
        raise RuntimeError("bad vision")

    monkeypatch.setattr(model, "control", control)
    monkeypatch.setattr(model, "healthy_http", lambda _: True)
    monkeypatch.setattr(model, "health", health)
    with pytest.raises(RuntimeError, match="bad vision"):
        model.start(args, deployment(tmp_path))
    assert calls[-1] == ("stop", "qwen-vlm")
    assert (
        json.loads((tmp_path / "health.json").read_text())["failed_check"]
        == "vision_red"
    )


def test_occupied_port_does_not_start_or_probe_unrelated_service(tmp_path, monkeypatch):
    calls = []

    def control(root, *cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="qwen-vlm STOPPED")

    monkeypatch.setattr(model, "control", control)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        data = deployment(tmp_path, sock.getsockname()[1])
        with pytest.raises(RuntimeError, match="port occupied"):
            model.start(SimpleNamespace(root=tmp_path, wait_seconds=1), data)
    assert not any(cmd[0] in {"start", "stop"} for cmd in calls)
    assert json.loads((tmp_path / "health.json").read_text())["state"] == "failed"


@pytest.mark.parametrize(
    "failure,failed_check,request_count",
    [
        (None, None, 4),
        ("wrong_model", "model_identity", 0),
        ("wrong_image", "vision_red", 1),
        ("partial_json", "events_production_schema", 3),
        ("partial_grounding", "compact_production_schema", 4),
        ("http_error", "vision_red", 1),
        ("no_metrics", None, 4),
    ],
)
def test_real_local_http_probe_checks_images_and_production_contracts(
    tmp_path, failure, failed_check, request_count
):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, body, status=200, content_type="application/json"):
            encoded = (
                body.encode() if isinstance(body, str) else json.dumps(body).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply(
                    {
                        "data": [
                            {
                                "id": "wrong"
                                if failure == "wrong_model"
                                else "test-model"
                            }
                        ]
                    }
                )
            else:
                self.reply(
                    "vllm:num_requests_waiting 0\n",
                    status=404 if failure == "no_metrics" else 200,
                    content_type="text/plain",
                )

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            assert self.headers.get("Authorization") is None
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            images = [
                part
                for msg in body["messages"]
                for part in (msg["content"] if isinstance(msg["content"], list) else [])
                if part["type"] == "image_url"
            ]
            assert len(images) == 4
            if failure == "http_error":
                self.reply({"error": "secret-provider-body"}, 500)
                return
            fmt = body["response_format"]
            if fmt.get("json_schema", {}).get("name") == "visual_health":
                data = base64.b64decode(images[-1]["image_url"]["url"].split(",")[1])
                with Image.open(io.BytesIO(data)) as image:
                    red, _, blue = image.getpixel((1, 1))
                answer = {"color": "red" if red > blue else "blue"}
                if failure == "wrong_image":
                    answer = {"color": "green"}
            elif fmt["type"] == "json_schema":
                answer = {label: False for label in SAFETY_EVENT_LABELS}
                if failure == "partial_json":
                    answer.pop("fall")
            else:
                answer = {
                    "bbox_coordinate_system": "normalized_1",
                    "subjects": [],
                    "verdicts": [
                        {
                            "event_label": label,
                            "subject_ids": [],
                            "detected": False,
                            "confidence": 0.9,
                        }
                        for label in SAFETY_EVENT_LABELS
                    ],
                }
                if failure == "partial_grounding":
                    answer["verdicts"].pop()
            self.reply(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(answer),
                            },
                        }
                    ]
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "health.json"
    try:
        if failed_check:
            with pytest.raises((ValueError, RuntimeError, httpx.HTTPStatusError)):
                probe.probe(
                    f"http://127.0.0.1:{server.server_port}", "test-model", output
                )
            result = json.loads(output.read_text())
            assert result["state"] == "failed"
            assert result["failed_check"] == failed_check
            assert "secret-provider-body" not in output.read_text()
        else:
            result = probe.probe(
                f"http://127.0.0.1:{server.server_port}", "test-model", output
            )
            assert result["state"] == "ready"
            assert result["metrics_exposed"] == (failure != "no_metrics")
            assert len(seen) == len(result["checks"]) == 4
        assert len(seen) == request_count
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org",
        "http://localhost:8000",
        "http://user:pass@127.0.0.1:8000",
        "http://127.0.0.1:8000/?key=x",
    ],
)
def test_probe_cannot_send_health_images_to_external_provider(url):
    with pytest.raises(ValueError):
        probe.validate_endpoint(url)


def test_package_is_self_contained_and_uses_matching_benchmark_metadata(
    tmp_path, monkeypatch
):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    archive = tmp_path / "bundle.tar.gz"
    manifest = run_remote.bundle(model.ROOT, [video], archive, scope="local_gpu_model")
    assert manifest["scope"] == "local_gpu_model"
    app = tmp_path / "app"
    run_remote.safe_extract(archive, app)
    for name in (
        "local_model.py",
        "local_model_probe.py",
        "local_model_profile.json",
        "model-requirements.txt",
        "monitor.py",
        "monitor.html",
        "monitor.js",
        "recording.py",
        "monitor-requirements.txt",
    ):
        assert (app / "tools/vlm_benchmark" / name).is_file()
    monkeypatch.setattr(modes, "ROOT", app)
    cfg = modes.configure(modes.parser().parse_args(["--backend", "local"]))
    assert cfg.model_api.provider == "local"
    args = SimpleNamespace(
        root=tmp_path,
        video=video,
        streams=[1, 12],
        duration=3,
        repeats=1,
        output=tmp_path / "report",
    )
    data = deployment(tmp_path)
    cmd = model.benchmark_command(args, data)
    assert cmd[cmd.index("--model") + 1] == data["profile"]["served_model"]
    assert cmd[cmd.index("--model-revision") + 1] == data["profile"]["model_revision"]
    assert "--real" in cmd and "openrouter" not in cmd


def test_plan_needs_no_gpu_or_download():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.vlm_benchmark.local_model",
            "plan",
        ],
        cwd=model.ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout)["base_model"] == "Qwen/Qwen3.8-27B"


def test_packaged_benchmark_does_not_require_git(monkeypatch):
    monkeypatch.setattr(modes.shutil, "which", lambda _: None)
    monkeypatch.setattr(
        modes.subprocess, "run", lambda *_a, **_kw: pytest.fail("needs git")
    )
    assert modes.checkout_revision() == ""
