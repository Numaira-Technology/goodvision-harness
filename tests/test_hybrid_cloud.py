import io
import json
import tarfile
from unittest.mock import Mock

import pytest

from tools import hybrid_cloud as hybrid


@pytest.mark.parametrize(
    "streams,port,sam",
    [(2, 65535, 8092), (0, 8280, 8092), (2, 8091, 8092), (60000, 1, 65000)],
)
def test_invalid_ports_fail_before_renting(streams, port, sam):
    with pytest.raises(ValueError):
        hybrid.validate_ports(streams, port, sam)


def test_cloud_account_and_model_tokens_do_not_reach_cpu_settings():
    environment = {
        "AICLUB_API_KEY": "event-token",
        "VAST_API_KEY": "cloud-token",
        "HF_TOKEN": "model-token",
        "SAM_SSH_KEY": "private-key",
        "INFERENCE_PROVIDER": "aiclub",
        "SCENE_VLM_OUTPUT_MODE": "parallel",
    }
    assert hybrid.inference_environment(environment) == {
        "AICLUB_API_KEY": "event-token",
        "INFERENCE_PROVIDER": "aiclub",
        "SCENE_VLM_OUTPUT_MODE": "parallel",
    }


def test_cpu_lanes_keep_independent_state_and_exact_sam_camera(tmp_path):
    root = tmp_path / "source"
    first = hybrid.lane_environment({}, {}, root, tmp_path / "one", 1, 2)
    second = hybrid.lane_environment({}, {}, root, tmp_path / "two", 2, 2)
    assert (
        first["HARNESS_DEPLOYMENT_SETTINGS_PATH"]
        != second["HARNESS_DEPLOYMENT_SETTINGS_PATH"]
    )
    assert "/camera-01/" in first["RICH_LABEL_ENDPOINT"]
    assert "/camera-02/" in second["RICH_LABEL_ENDPOINT"]
    assert first["SAM31_TRANSPORT_CODEC"] == "jpeg"
    assert first["CUDA_VISIBLE_DEVICES"] == ""
    assert first["LOCAL_CAMERA_DISPLAY_ENABLED"] == "true"
    assert "FACE_SAMPLE_FPS" not in first  # Preserve event cadence from main.


def test_bundle_excludes_local_secrets_and_generated_bytecode(tmp_path):
    for name in [
        "pyproject.toml",
        "config.yaml",
        "tools/hybrid_cloud.py",
        "src/goodvision_harness/service.py",
        "src/goodvision_harness/assets/runtime.wasm",
        ".env",
        "src/goodvision_harness/.env",
        "src/goodvision_harness/__pycache__/service.pyc",
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("test")
    archive = tmp_path / "out.tgz"
    hybrid.build_bundle(tmp_path, archive)
    with tarfile.open(archive) as f:
        names = f.getnames()
    assert "src/goodvision_harness/assets/runtime.wasm" in names
    assert not any(".env" in n or "__pycache__" in n for n in names)


def test_forwarded_consoles_are_loopback_only():
    assert hybrid.forwards(2, 8280) == [
        "-L",
        "127.0.0.1:8280:127.0.0.1:8200",
        "-L",
        "127.0.0.1:8281:127.0.0.1:8201",
    ]


def test_readiness_requires_every_cpu_lane(monkeypatch):
    opener = Mock()
    opener.open.side_effect = [
        io.BytesIO(json.dumps({"local_camera_display": {"enabled": True}}).encode())
        for _ in range(2)
    ]
    monkeypatch.setattr(hybrid, "build_opener", lambda _: opener)
    worker = Mock()
    worker.poll.return_value = None
    hybrid.wait_ready(worker, 2, 8280)
    assert [c.args[0] for c in opener.open.call_args_list] == [
        "http://127.0.0.1:8280/v1/ui/info",
        "http://127.0.0.1:8281/v1/ui/info",
    ]


def test_readiness_fails_if_supervisor_exits():
    worker = Mock()
    worker.poll.return_value = 1
    with pytest.raises(RuntimeError, match="supervisor exited"):
        hybrid.wait_ready(worker, 2, 8280)


def test_failed_child_stops_siblings_and_removes_secret_file(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"AICLUB_API_KEY": "test-token"}))
    monkeypatch.setattr(hybrid, "REMOTE", str(tmp_path))
    monkeypatch.setattr(hybrid.signal, "SIGHUP", 31, raising=False)
    monkeypatch.setattr(hybrid.signal, "signal", lambda *args: None)
    killed, launched = [], []
    monkeypatch.setattr(
        hybrid.os, "killpg", lambda pid, sig: killed.append(pid), raising=False
    )

    class Child:
        def __init__(self, pid, code):
            self.pid, self.code = pid, code

        def poll(self):
            return self.code

        def wait(self, timeout=None):
            self.code = 0
            return 0

    def launch(command, **kwargs):
        launched.append(kwargs)
        return Child(len(launched), 1 if len(launched) == 1 else None)

    monkeypatch.setattr(hybrid.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="child exited"):
        hybrid.serve(2, "sam-test-command")
    assert not runtime.exists()
    assert killed == [2, 3]
    assert all(item["stdout"].closed for item in launched[1:])
    assert all(item["env"]["CUDA_VISIBLE_DEVICES"] == "" for item in launched[1:])


@pytest.mark.parametrize("hybrid_port,bind", [(8280, "127.0.0.1"), (None, "0.0.0.0")])
def test_sam_forward_is_private_for_direct_hybrid_mode(monkeypatch, hybrid_port, bind):
    from tools import start_sam

    monkeypatch.setattr(start_sam, "wait_for_ssh", lambda *args: ("test-host", 22))
    monkeypatch.setattr(start_sam, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(hybrid, "deploy", lambda *args: None)
    launch = Mock()
    monkeypatch.setattr(start_sam.subprocess, "Popen", launch)
    start_sam.start_worker(
        Mock(), 1, [], hf_token="test-token", sam_port=8092, hybrid_port=hybrid_port
    )
    command = launch.call_args.args[0]
    assert f"{bind}:8092:127.0.0.1:8092" in command
    if hybrid_port is not None:
        assert "127.0.0.1:8280:127.0.0.1:8200" in command
        assert not any("0.0.0.0:" in argument for argument in command)
