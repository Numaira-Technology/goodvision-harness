"""Persistence and service lifetime regressions for the main integration."""

import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_vast_vlm import model_fixture

from tools.vast_vlm import fp8_adapter, fp8_profile, profile, runtime, yolo_runtime
from tools.vlm_benchmark import modes


@pytest.mark.parametrize(
    "field,value",
    [
        ("streams", [1]),
        ("observation_fps", 1),
        ("modes", ["events"]),
        ("model_revision", profile.MODEL_REVISION),
        ("backend", "openrouter"),
        ("http_timeout", 5),
        ("parallel_scheduling", "independent"),
        ("person_ids_only", False),
        ("monitor_layout", "evidence"),
        ("monitor", False),
        ("provider_prefer", "alibaba"),
        ("record", False),
    ],
)
def test_fp8_frozen_drift_rejected(field, value):
    args = fp8_adapter.arguments()
    setattr(args, field, value)
    with pytest.raises(ValueError, match="contract changed"):
        modes.validate(args)


def test_unlisted_weights_are_rejected(tmp_path):
    model_fixture(tmp_path)
    (tmp_path / "extra.safetensors").write_bytes(b"unverified")
    with pytest.raises(ValueError, match="unexpected"):
        runtime.verify_model(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only Linux service lifecycle")
def test_model_directory_cannot_escape_volume(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "volume"
    root.mkdir()
    (root / "models").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="volume"):
        runtime.model_dir(root)


@pytest.mark.parametrize("detector", [False, True])
def test_old_boot_cannot_reuse_same_process_identity(tmp_path, monkeypatch, detector):
    monkeypatch.setattr(runtime, "volume_root", lambda path: path)
    monkeypatch.setattr(runtime, "boot_id", lambda: "new-boot")
    monkeypatch.setattr(runtime, "process_start_ticks", lambda pid: "10")
    (tmp_path / "runtime").mkdir()
    data = dict(
        version=fp8_profile.VERSION,
        revision=fp8_profile.MODEL_REVISION,
        pid=123,
        start_ticks="10",
        boot_id="old-boot",
        weights_sha256=fp8_profile.YOLO_SHA256,
    )
    runtime.write_json(
        tmp_path
        / ("runtime/detector-active.json" if detector else "runtime/active.json"),
        data,
    )
    with pytest.raises(ValueError, match="matching live"):
        if detector:
            yolo_runtime.verify_active(tmp_path)
        else:
            runtime.verify_active(tmp_path, p=fp8_profile)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only Linux service lifecycle")
def test_disk_check_does_not_credit_unrelated_or_corrupt_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "volume_root", lambda path: path)
    target = runtime.model_dir(tmp_path)
    target.mkdir(parents=True)
    (target / "unrelated.bin").write_bytes(b"x" * 200)
    (target / "weights.safetensors").write_bytes(b"x" * 100)
    info = SimpleNamespace(
        sha=profile.MODEL_REVISION,
        siblings=[
            SimpleNamespace(
                rfilename="weights.safetensors",
                size=100,
                lfs=SimpleNamespace(sha256="wrong"),
            )
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: SimpleNamespace(model_info=lambda *a, **kw: info),
            snapshot_download=lambda *a, **kw: pytest.fail(
                "disk check must precede download"
            ),
        ),
    )
    monkeypatch.setattr(
        runtime.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=10 * 1024**3 + 50),
    )
    with pytest.raises(ValueError, match="insufficient"):
        runtime.prepare(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only Linux service lifecycle")
def test_cleanup_kills_own_workers_even_when_parent_already_exited(monkeypatch):
    signals = []
    monkeypatch.setattr(
        runtime.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    waited = []
    parent = SimpleNamespace(pid=4123, wait=lambda **kw: waited.append(kw) or 0)
    runtime.stop_process_group(parent)
    assert signals == [(4123, signal.SIGTERM), (4123, signal.SIGKILL)]
    assert waited == [{"timeout": 30}, {}]


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only Linux service lifecycle")
def test_cleanup_escalates_when_gpu_process_does_not_exit(monkeypatch):
    signals = []
    monkeypatch.setattr(runtime.os, "killpg", lambda pid, sig: signals.append(sig))

    def wait(**kw):
        if kw:
            raise subprocess.TimeoutExpired("test-worker", 30)
        return 0

    runtime.stop_process_group(SimpleNamespace(pid=4123, wait=wait))
    assert signals == [signal.SIGTERM, signal.SIGKILL]
