"""Bind preflight's artifact selection through the real SSH handoff validator."""

import copy
import json
import sys

import pytest
from test_qwen8b_host_preflight import host_bundle
from test_qwen8b_release import fake_runtime
from test_qwen8b_vllm_preflight import frozen

from tools import cloud_env, env_source, registry
from tools import start_qwen8b_r2 as host
from tools import start_qwen8b_vllm as vllm
from tools.qwen8b import eval_local

__all__ = ["frozen", "host_bundle"]


def ack(name, env):
    return json.dumps(
        {
            "name": name,
            "source": "r2",
            "python": env["unpack_to"] + "/bin/python",
            "activate": env["unpack_to"] + "/" + env["activate_rel"],
            "identity": env_source.restore_identity(env),
        }
    )


@pytest.mark.parametrize("phase", ["before", "during", "unchanged", "malformed"])
def test_caller_identity_is_captured_before_any_upload(tmp_path, monkeypatch, phase):
    selected = copy.deepcopy(registry.env_of("qwen8b"))
    newer = copy.deepcopy(selected)
    newer["sha256"] = "a" * 64
    events = []
    monkeypatch.setattr(
        registry, "env_of", lambda name: newer if phase == "before" else selected
    )

    def upload(*args):
        events.append("upload")
        if phase == "during":
            # Mutation of a supplied dictionary must not mutate the expected reply.
            selected["compat"]["torch"] = "9.9.9"

    def run(*args, **kwargs):
        events.append("run")
        return ack("qwen8b", selected)

    kwargs = dict(
        ssh_options=[],
        host="synthetic.invalid",
        port=22,
        name="qwen8b",
        expected_env={} if phase == "malformed" else selected,
        r2_env={key: "synthetic" for key in cloud_env.R2_KEYS},
        uploader=upload,
        runner=run,
        cleaner=lambda *args: events.append("cleanup"),
    )
    if phase == "unchanged":
        assert cloud_env.restore_env(**kwargs) == selected["unpack_to"] + "/bin/python"
    else:
        with pytest.raises(
            (ValueError, RuntimeError), match=r"identity|safe relative path"
        ):
            cloud_env.restore_env(**kwargs)
    assert events == (
        [] if phase in ("before", "malformed") else ["upload", "run", "cleanup"]
    )


@pytest.mark.parametrize("module", [host, vllm], ids=["hf", "vllm"])
@pytest.mark.parametrize("phase", ["before", "during", "unchanged"])
def test_main_keeps_preflight_identity_through_restore(
    tmp_path, monkeypatch, host_bundle, frozen, module, phase
):
    bundle = host_bundle if module is host else frozen
    name = "qwen8b" if module is host else "qwen8b-vllm"
    selected = copy.deepcopy(registry.env_of(name))
    newer = copy.deepcopy(selected)
    newer["sha256"] = "a" * 64
    verify = module._verify_eval_bundle
    vast, status, stopped = fake_runtime(monkeypatch, tmp_path, module, None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--eval-bundle",
            str(bundle),
            "--eval-output",
            str(tmp_path / "result"),
            "--status-file",
            str(status),
        ],
    )
    lookups = []

    def lookup(requested):
        assert requested == name
        lookups.append(requested)
        return copy.deepcopy(
            newer if phase == "before" and len(lookups) > 1 else selected
        )

    monkeypatch.setattr(registry, "env_of", lookup)
    monkeypatch.setattr(module, "_verify_eval_bundle", verify)
    events = []

    def restore(**kwargs):
        return cloud_env.restore_env(
            **kwargs,
            uploader=lambda *a: events.append("upload"),
            runner=lambda *a, **k: ack(name, newer if phase == "during" else selected),
            cleaner=lambda *a: events.append("clean-reader"),
        )

    monkeypatch.setattr(module, "restore_env", restore)
    monkeypatch.setattr(
        module, "_server_process", lambda *a: events.append("server") or object()
    )
    monkeypatch.setattr(eval_local, "main", lambda *a: events.append("eval") or 0)
    assert module.main() == (0 if phase == "unchanged" else 1)
    assert ("eval" in events) is (phase == "unchanged")
    assert ("server" in events) is (phase == "unchanged")
    assert ("upload" in events) is (phase != "before")
    assert ("clean-reader" in events) is (phase != "before")
    assert stopped == [True] and vast.destroyed == [123]


def test_legacy_hf_profile_without_artifact_stops_before_credentials(
    host_bundle, monkeypatch
):
    from test_qwen8b_host_preflight import seal

    config = host_bundle / "configs/qwen8b_r2_host.json"
    body = json.loads(config.read_text())
    del body["env_artifact"]
    config.write_text(json.dumps(body))
    seal(host_bundle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "host",
            "--eval-bundle",
            str(host_bundle),
            "--eval-output",
            str(host_bundle.parent / "result"),
        ],
    )
    monkeypatch.setitem(sys.modules, "dotenv", None)
    with pytest.raises(ValueError, match="profile differs"):
        host.main()
