"""Diagnostic acceptance and cleanup tests with no cloud or GPU operations."""

import copy
import sys
from types import SimpleNamespace

import pytest
from test_qwen8b_release import FakeVast

from tools import diag_qwen8b_vllm_import as diag


@pytest.mark.parametrize("available,count", [(False, 0), (True, 0), (True, 1)])
def test_cuda_probe_requires_a_visible_cuda_device(monkeypatch, available, count):
    torch = SimpleNamespace(
        __version__="synthetic",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: count,
            get_device_name=lambda i: "synthetic 4090",
            get_device_capability=lambda i: (8, 9),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="synthetic"))
    if available and count:
        exec(diag.CUDA_PROBE, {})
    else:
        with pytest.raises(AssertionError, match=r"CUDA unavailable|No CUDA device"):
            exec(diag.CUDA_PROBE, {})


def test_modern_probe_requires_deep_serving_imports(monkeypatch):
    # Top-level imports succeeding must not hide a missing torch extension module.
    torch = SimpleNamespace(
        __version__="synthetic", version=SimpleNamespace(cuda="13.0")
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="synthetic"))
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", None)
    with pytest.raises(ModuleNotFoundError):
        exec(diag.NEW_PROBE, {})


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "cuda",
        "deep",
        "rejected",
        "exception",
        "still-live",
        "list-error",
        "malformed-list",
        "unrelated",
        "interrupt",
        "interrupt-report",
        "failure-report",
    ],
)
def test_main_success_requires_all_probes_and_confirmed_release(
    tmp_path, monkeypatch, fault
):
    vast = FakeVast(fault)
    key = tmp_path / "synthetic-key"
    key.write_text("not a real key")
    monkeypatch.setitem(
        sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda *a, **k: None)
    )
    monkeypatch.setitem(sys.modules, "vastai", SimpleNamespace(VastAI=lambda **k: vast))
    monkeypatch.setattr(
        diag.S,
        "required",
        lambda name: str(key) if name == "SAM_SSH_KEY" else "SYNTHETIC",
    )
    monkeypatch.setattr(diag.S, "IMAGE", "synthetic")
    for name in diag.H.R2_KEYS:
        monkeypatch.setenv(name, "synthetic")
    monkeypatch.setattr(diag.S, "prepare_ssh_key", lambda *a: key)

    def create(*args, **kwargs):
        vast.alive = True
        return 123

    monkeypatch.setattr(diag.S, "create_from_offers", create)
    monkeypatch.setattr(diag.H, "attach_key", lambda *a: None)
    monkeypatch.setattr(diag.H, "wait_for_ssh", lambda *a: ("synthetic.invalid", 22))
    monkeypatch.setattr(diag.V, "_verify_host_driver", lambda *a: None)
    monkeypatch.setattr(diag.S, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        diag, "restore_env", lambda **k: "/opt/env/qwen8b-vllm/bin/python"
    )

    def ssh_run(ssh, remote, **kwargs):
        cuda = "cuda_is_available" in remote
        deep = "api_server" in remote
        if cuda and fault in ("interrupt", "interrupt-report"):
            raise KeyboardInterrupt
        if cuda and fault == "failure-report":
            raise RuntimeError("synthetic runtime failure")
        rc = 17 if (cuda and fault == "cuda") or (deep and fault == "deep") else 0
        return {"rc": rc, "stdout": "synthetic", "stderr": "", "seconds": 0}

    monkeypatch.setattr(diag, "_ssh_run", ssh_run)
    reports = []

    def write(path, report):
        if (fault == "interrupt-report" and report["phase"] == "interrupted") or (
            fault == "failure-report" and report["phase"] == "failed"
        ):
            # Only the first phase write fails; the final outcome can be retained.
            if "exit_code" not in report:
                raise OSError("synthetic phase-report write failure")
        reports.append(copy.deepcopy(report))

    monkeypatch.setattr(diag, "_write", write)
    monkeypatch.setattr(diag, "OUT_DEFAULT", tmp_path / "report.json")
    result = diag.main()
    assert result == (
        0
        if fault in (None, "unrelated")
        else 130
        if fault in ("interrupt", "interrupt-report")
        else 1
    )
    assert vast.destroyed == [123]
    assert vast.client.timeout == (3, 4)
    report = reports[-1]
    assert report["ok"] is (result == 0)
    assert report["exit_code"] == result
    if fault in ("rejected", "exception", "still-live", "list-error", "malformed-list"):
        assert report["released"] is False
    if fault in ("cuda", "deep"):
        assert report["checks_passed"] is False
        assert report["released"] is True
    if fault == "unrelated":
        assert report["live_after"] == [{"id": 987, "label": "unrelated"}]
