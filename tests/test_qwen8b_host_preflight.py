"""Offline HF launcher admission checks; all remote/runtime operations are fake."""

import json
import sys
from types import SimpleNamespace

import pytest
from test_qwen8b_bundle_conversion import host_bundle, seal

from tools import start_qwen8b_r2 as host
from tools.qwen8b import eval_local
from tools.unified_benchmark.contract import file_hash

# Reuse the same synthetic sealed HF fixture as the bundle converter tests.
__all__ = ["host_bundle"]


@pytest.mark.parametrize(
    "change",
    ["missing", "tampered", "profile", "plan", "existing_output", "output_symlink"],
)
def test_invalid_eval_stops_before_credentials_and_rental(
    host_bundle, tmp_path, monkeypatch, change
):
    bundle, output = host_bundle, tmp_path / "result"
    if change == "missing":
        bundle = tmp_path / "absent"
    elif change == "tampered":
        (bundle / "frames/synthetic.jpg").write_bytes(b"changed")
    elif change in ("profile", "plan"):
        name = (
            "configs/qwen8b_r2_host.json"
            if change == "profile"
            else "plans/qwen8b_r2_host-test.json"
        )
        path = bundle / name
        value = json.loads(path.read_text())
        value["model" if change == "profile" else "config_sha256"] = "wrong"
        path.write_text(json.dumps(value))
        seal(bundle)
    elif change == "existing_output":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "absent-target")
    monkeypatch.setattr(
        sys,
        "argv",
        ["host", "--eval-bundle", str(bundle), "--eval-output", str(output)],
    )
    monkeypatch.setattr(
        host.S, "required", lambda *a: pytest.fail("credentials accessed")
    )
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        SimpleNamespace(
            load_dotenv=lambda *a, **k: pytest.fail(
                "dotenv accessed before offline rejection"
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vastai",
        SimpleNamespace(VastAI=lambda **k: pytest.fail("Vast accessed")),
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        host.main()


@pytest.mark.parametrize("change_after_ready", [None, "tamper", "reseal", "output"])
def test_hf_rechecks_after_preparation_and_always_releases(
    host_bundle, tmp_path, monkeypatch, change_after_ready
):
    events = []
    output = tmp_path / "result"
    key = tmp_path / "synthetic-key"
    key.write_text("never used by SSH")
    monkeypatch.setattr(
        sys,
        "argv",
        ["host", "--eval-bundle", str(host_bundle), "--eval-output", str(output)],
    )
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        SimpleNamespace(load_dotenv=lambda *a, **k: events.append("dotenv")),
    )
    vast = SimpleNamespace(search_offers=lambda **k: [1], show_instances=lambda: [])
    monkeypatch.setitem(sys.modules, "vastai", SimpleNamespace(VastAI=lambda **k: vast))
    monkeypatch.setattr(host.S, "IMAGE", "synthetic")
    monkeypatch.setattr(
        host.S,
        "required",
        lambda name: str(key) if name == "SAM_SSH_KEY" else "synthetic",
    )
    for name in host.R2_KEYS:
        monkeypatch.setenv(name, "synthetic")
    verify = host._verify_eval_bundle

    def preflight(*args):
        events.append("preflight")
        return verify(*args)

    monkeypatch.setattr(host, "_verify_eval_bundle", preflight)
    monkeypatch.setattr(host.S, "prepare_ssh_key", lambda *a: key)
    monkeypatch.setattr(
        host.S, "create_from_offers", lambda *a, **k: events.append("rent") or 123
    )
    monkeypatch.setattr(host, "attach_key", lambda *a: None)
    monkeypatch.setattr(host, "wait_for_ssh", lambda *a: ("synthetic.invalid", 22))
    monkeypatch.setattr(host.S, "run", lambda *a, **k: None)
    monkeypatch.setattr(host, "_upload_repo", lambda *a: None)
    monkeypatch.setattr(host, "restore_env", lambda **k: "/synthetic/bin/activate")
    monkeypatch.setattr(host, "prefetch_models", lambda **k: None)
    monkeypatch.setattr(host, "_server_process", lambda *a: object())

    def ready(*args):
        events.append("ready")
        if change_after_ready == "tamper":
            (host_bundle / "frames/synthetic.jpg").write_bytes(b"changed")
        elif change_after_ready == "reseal":
            (host_bundle / "new-note.txt").write_text("bundle identity changed")
            seal(host_bundle)
        elif change_after_ready == "output":
            output.mkdir()
        return {}

    monkeypatch.setattr(host, "_wait_ready", ready)
    monkeypatch.setattr(eval_local, "main", lambda argv: events.append("eval") or 0)
    monkeypatch.setattr(host.S, "stop_process", lambda *a: events.append("stop"))
    monkeypatch.setattr(host.S, "release_instance", lambda *a: events.append("release"))
    assert host.main() == (0 if change_after_ready is None else 1)
    assert events.index("preflight") < events.index("dotenv") < events.index("rent")
    assert ("eval" in events) is (change_after_ready is None)
    assert events[-2:] == ["stop", "release"]
    if change_after_ready is None:
        assert host._verify_eval_bundle(host_bundle, "both") == file_hash(
            host_bundle / "bundle-files-sha256.json"
        )
