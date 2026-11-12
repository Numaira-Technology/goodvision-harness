"""No-GPU tests for the qwen8b R2 host launcher and local eval transport."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import registry
from tools.qwen8b.eval_local import LocalTransport
from tools.qwen8b.serve import require_packed_prefix
from tools.start_qwen8b_r2 import _packed_python, _serve_remote, attach_key, wait_for_ssh
from tools.unified_benchmark.preflight import model_configs


def test_qwen8b_registry_pins():
    repo, rev = registry.weights_of("qwen8b")
    assert repo == "Qwen/Qwen3-VL-8B-Instruct"
    assert rev == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    env = registry.env_of("qwen8b")
    assert env["r2_key"] == "envs/qwen8b/cu128-py312-torch280-ada.tar.gz"
    assert env["build"]["flash_attn"] is True


def test_serve_remote_uses_absolute_packed_python_without_activate():
    remote = _serve_remote("/opt/env/qwen8b/bin/activate")
    assert "source " not in remote
    assert "conda activate" not in remote
    assert "python -m " not in remote
    assert "/opt/env/qwen8b/bin/python -s -c" in remote
    assert "PYTHONNOUSERSITE=1" in remote
    assert "sys.prefix" in remote
    assert "tools.qwen8b.serve" in remote
    assert "packed interpreter missing" in remote
    assert _packed_python("/opt/env/qwen8b/bin/activate") == (
        "/opt/env/qwen8b/bin/python"
    )


def test_serve_require_packed_prefix_rejects_host_python():
    with pytest.raises(SystemExit, match="sys.prefix"):
        require_packed_prefix("qwen8b")


def test_self_hosted_profile_skips_openrouter_pin():
    configs = model_configs(["qwen8b_r2_host"])
    cfg = configs["qwen8b_r2_host"]
    assert cfg["execution"] == "self_hosted_r2"
    assert cfg["weight_revision"] == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert cfg["prompt_policy"] == "current-image-v1"
    env = registry.env_of("qwen8b")
    assert env["sha256"] in cfg["env_source"]
    assert cfg["serving"].startswith("tools.qwen8b.serve")
    assert "flash_attention_2" in cfg["serving"]
    assert not Path("tools/unified_benchmark/profiles/qwen8b_selfhost.json").exists()


def test_attach_key_sends_public_material(tmp_path, monkeypatch):
    key = tmp_path / "id"
    key.write_text("secret")
    pub = Mock(return_value=Mock(stdout="ssh-ed25519 AAAA test\n", stderr="", returncode=0))
    monkeypatch.setattr("tools.start_qwen8b_r2.subprocess.run", pub)
    vast = Mock()
    vast.attach_ssh.return_value = {"msg": "ok"}
    attach_key(vast, 99, key)
    vast.attach_ssh.assert_called_once()
    kwargs = vast.attach_ssh.call_args.kwargs
    assert kwargs["instance_id"] == 99
    assert kwargs["ssh_key"].startswith("ssh-ed25519")
    assert "secret" not in kwargs["ssh_key"]


def test_wait_for_ssh_fails_fast_on_publickey_denial(monkeypatch):
    calls = {"attach": 0, "ssh": 0}

    def attach(vast, instance_id, key_path):
        calls["attach"] += 1

    def run(command, **kwargs):
        calls["ssh"] += 1
        return Mock(returncode=255, stderr="Permission denied (publickey).\n")

    vast = Mock()
    vast.show_instance.return_value = {
        "actual_status": "running",
        "ssh_host": "gw.vast.ai",
        "ssh_port": 12345,
        "ports": {},
    }
    monkeypatch.setattr("tools.start_qwen8b_r2.attach_key", attach)
    monkeypatch.setattr("tools.start_qwen8b_r2.subprocess.run", run)
    monkeypatch.setattr("tools.start_qwen8b_r2.time.sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="publickey denied"):
        wait_for_ssh(vast, 7, ["-o", "BatchMode=yes"], Path("id"),
                     deny_budget=0, boot_budget=30)
    assert calls["attach"] == 1
    assert calls["ssh"] >= 1
    assert calls["ssh"] < 40  # must not burn a 20-minute poll


def test_wait_for_ssh_returns_on_first_success(monkeypatch):
    vast = Mock()
    vast.show_instance.return_value = {
        "actual_status": "running",
        "ssh_host": "gw.vast.ai",
        "ssh_port": 2222,
        "ports": {},
    }
    monkeypatch.setattr("tools.start_qwen8b_r2.subprocess.run",
                        lambda *a, **k: Mock(returncode=0, stderr=""))
    monkeypatch.setattr("tools.start_qwen8b_r2.time.sleep", lambda _s: None)
    host, port = wait_for_ssh(vast, 3, ["-o", "BatchMode=yes"], Path("id"))
    assert (host, port) == ("gw.vast.ai", 2222)


def test_local_transport_rejects_fingerprint_mismatch(tmp_path):
    import hashlib
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(buf, format="JPEG")
    raw = buf.getvalue()
    frame = tmp_path / "f.jpg"
    frame.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    config = json.loads(Path("tools/unified_benchmark/profiles/qwen8b_r2_host.json").read_text())
    unit = {
        "id": "u1", "targets": ["fight"],
        "question": "Is any fighting visible in the current image?",
        "frames": [{"path": "f.jpg", "sha256": digest}],
        "request_sha256": "deadbeef",
    }
    ledger = Mock()
    ledger.state.return_value = (1, {})
    transport = LocalTransport(tmp_path, config, "http://127.0.0.1:9", tmp_path, ledger)
    from tools.unified_benchmark.retry import RequestFailure
    with pytest.raises(RequestFailure, match="fingerprint"):
        transport(unit)


def test_provider_body_skips_selfhost_routing():
    from tools.event_benchmark.evidence import provider_body
    from tools.event_benchmark.run_eval import build_request

    assert provider_body(None) is None
    assert provider_body({"config": {"transport": "selfhost", "provider": {"only": ["x"]}}}) is None
    url, body = build_request(
        "language", "qwen/qwen3-vl-8b-instruct", "Is fighting visible?",
        "data:image/jpeg;base64,xx",
        {"config": {"transport": "selfhost", "provider": {"only": ["x"]}}},
    )
    assert "provider" not in body
    _, routed = build_request(
        "language", "qwen/qwen3-vl-8b-instruct", "Is fighting visible?",
        "data:image/jpeg;base64,xx",
        {"config": {"transport": "openrouter", "provider": {"only": ["parasail"], "allow_fallbacks": False}}},
    )
    assert routed["provider"]["only"] == ["parasail"]


def test_run_side_by_side_overlaps_env_and_weights():
    """Env restore and weight prefetch must run at the same time."""
    import threading

    from tools import start_qwen8b_r2 as host

    both = threading.Barrier(2, timeout=5)

    def env():
        both.wait()  # times out if the steps run one after another
        return "/opt/env/bin/activate"

    def weights():
        both.wait()

    assert host.run_side_by_side(env=env, weights=weights) == {
        "env": "/opt/env/bin/activate",
        "weights": None,
    }


def test_run_side_by_side_raises_first_failure_without_waiting():
    import threading
    import time

    import pytest

    from tools import start_qwen8b_r2 as host

    release = threading.Event()

    def weights():
        raise RuntimeError("prefetch failed")

    def env():
        release.wait(5)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="prefetch failed"):
        host.run_side_by_side(env=env, weights=weights)
    # The caller gets the error at once and can release the instance.
    assert time.monotonic() - started < 2
    release.set()
