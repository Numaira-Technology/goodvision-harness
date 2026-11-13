"""Diagnostic acceptance and cleanup tests with no cloud or GPU operations."""

import copy
import io
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_qwen8b_release import FakeVast

from tools import diag_qwen8b_vllm_import as diag


@pytest.mark.parametrize("default_encoding", ["cp1252", "utf-16"])
def test_report_and_mirror_use_utf8(tmp_path, monkeypatch, default_encoding):
    report = tmp_path / "reports" / "diagnostic.json"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(
        diag,
        "Path",
        lambda value: artifacts if value == "/opt/cursor/artifacts" else Path(value),
    )
    payload = {"phase": "failed", "error": "诊断失败 — café ☃", "检查": ["CUDA 不可用"]}
    text_encoding = io.text_encoding
    # Model a non-UTF-8 text default without changing the process locale.
    # JSON's existing ASCII escapes already work under cp1252; UTF-16 exposes
    # an implicit file encoding even when every serialized character is ASCII.
    with monkeypatch.context() as encoding_patch:
        encoding_patch.setattr(
            io,
            "text_encoding",
            lambda encoding, stacklevel=2: (
                default_encoding
                if encoding is None
                else text_encoding(encoding, stacklevel)
            ),
        )
        diag._write(report, payload)

    expected = json.dumps(payload, indent=2) + "\n"
    for path in (report, artifacts / report.name):
        text = path.read_bytes().decode("utf-8")
        assert text.replace("\r\n", "\n") == expected
        assert json.loads(text) == payload
    assert (artifacts / report.name).read_bytes() == report.read_bytes()


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


@pytest.fixture
def reporting_runtime(tmp_path, monkeypatch):
    """Real shared creation and report writer; all external work is synthetic."""
    snapshots, writes, attempted, shared_calls = [], [], [], []

    class ReportVast(FakeVast):
        def __init__(self):
            super().__init__(None)
            self.label = None
            self.script = ["success"]

        def search_offers(self, **kwargs):
            first = super().search_offers(**kwargs)[0] | {"machine_id": 101}
            return [first, first | {"id": 2, "machine_id": 102, "dph_total": 1.4}]

        def create_instance(self, **kwargs):
            attempted.append(kwargs["id"])
            self.label = kwargs["label"]
            snapshots.append(json.loads(report_path.read_text()))
            outcome = self.script.pop(0)
            if outcome == "unavailable":
                return {"success": False, "error": "offer_unavailable"}
            self.alive = True
            if outcome == "lost":
                raise TimeoutError("synthetic lost response")
            return {"success": True, "new_contract": 123}

        def show_instances(self):
            return [{"id": 123, "label": self.label}] if self.alive else []

    vast = ReportVast()
    key = tmp_path / "synthetic-key"
    key.write_text("not a real key")
    report_path = tmp_path / "report.json"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr(diag, "OUT_DEFAULT", report_path)
    # Exercise the existing mirror too, without touching /opt/cursor/artifacts.
    monkeypatch.setattr(
        diag,
        "Path",
        lambda value: artifacts if value == "/opt/cursor/artifacts" else Path(value),
    )
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
    monkeypatch.setattr(diag.H, "attach_key", lambda *a: None)
    monkeypatch.setattr(diag.H, "wait_for_ssh", lambda *a: ("synthetic.invalid", 22))
    monkeypatch.setattr(diag.V, "_verify_host_driver", lambda *a: None)
    monkeypatch.setattr(diag.S, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        diag, "restore_env", lambda **k: "/opt/env/qwen8b-vllm/bin/python"
    )
    monkeypatch.setattr(
        diag,
        "_ssh_run",
        lambda *a, **k: {"rc": 0, "stdout": "synthetic", "stderr": "", "seconds": 0},
    )

    def forbidden(*a, **k):
        raise AssertionError("external operation forbidden in reporting tests")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(diag.subprocess, "run", forbidden)
    monkeypatch.setattr(diag.subprocess, "Popen", forbidden)
    original_create = diag.S.create_from_offers

    def create(*args, **kwargs):
        shared_calls.append(json.loads(report_path.read_text()))
        return original_create(*args, **kwargs)

    monkeypatch.setattr(diag.S, "create_from_offers", create)
    original_write = diag._write

    def write(path, report):
        original_write(path, report)
        writes.append(copy.deepcopy(report))

    monkeypatch.setattr(diag, "_write", write)
    return SimpleNamespace(
        vast=vast,
        path=report_path,
        mirror=artifacts / report_path.name,
        snapshots=snapshots,
        writes=writes,
        attempted=attempted,
        shared_calls=shared_calls,
        write=write,
    )


def test_create_intent_report_precedes_shared_dispatch(reporting_runtime):
    runtime = reporting_runtime
    assert diag.main() == 0
    before = runtime.snapshots[0]
    assert runtime.writes[0]["label"] == runtime.vast.label
    assert runtime.writes[0]["ok"] is False
    assert runtime.writes[0]["dph_selected"] is None
    assert before["phase"] == "creating"
    assert runtime.shared_calls == [before]
    assert before["label"] == runtime.vast.label
    assert before["create_intent"] == {
        "label": runtime.vast.label,
        "state": "prepared",
    }
    assert before["dph_selected"] is None
    assert before["dph_cheapest_eligible"] == 1.1
    assert "instance_id" not in before
    assert runtime.attempted == [1]
    assert runtime.vast.destroyed == [123]
    assert json.loads(runtime.mirror.read_text()) == json.loads(
        runtime.path.read_text()
    )


def test_failed_create_checkpoint_prevents_dispatch(reporting_runtime, monkeypatch):
    runtime = reporting_runtime

    def write(path, report):
        if report["phase"] == "creating":
            raise OSError("synthetic checkpoint failure")
        runtime.write(path, report)

    monkeypatch.setattr(diag, "_write", write)
    assert diag.main() == 1
    assert runtime.attempted == []
    assert runtime.shared_calls == []
    assert runtime.snapshots == []
    assert runtime.vast.destroyed == []
    assert runtime.writes[0]["label"] == runtime.writes[-1]["label"]
    assert runtime.writes[-1]["ok"] is False
    assert "synthetic checkpoint failure" in runtime.writes[-1]["error"]


def test_fallback_does_not_claim_cheapest_quote_as_selected(reporting_runtime):
    runtime = reporting_runtime
    runtime.vast.script = ["unavailable", "success"]
    assert diag.main() == 0
    assert runtime.attempted == [1, 2]
    report = json.loads(runtime.path.read_text())
    assert report["instance_id"] == 123
    assert report["dph_cheapest_eligible"] == 1.1
    assert report["dph_selected"] is None
    assert all(
        snapshot["create_intent"]["label"] == runtime.vast.label
        for snapshot in runtime.snapshots
    )
    assert runtime.vast.destroyed == [123]


def test_ambiguous_create_retains_intent_without_replacement(reporting_runtime):
    runtime = reporting_runtime
    runtime.vast.script = ["lost", "success"]
    assert diag.main() == 1
    assert runtime.attempted == [1]
    assert len(runtime.shared_calls) == 1
    assert runtime.vast.script == ["success"]
    before = runtime.snapshots[0]
    report = json.loads(runtime.path.read_text())
    assert before["phase"] == "creating"
    assert before["create_intent"]["label"] == runtime.vast.label
    assert report["label"] == runtime.vast.label
    assert report["instance_id"] == 123  # Existing exact-label reconciliation.
    assert report["dph_selected"] is None
    assert "Creation outcome unknown" in report["error"]
    assert runtime.vast.destroyed == [123]
    assert report["released"] is True
    assert report["ok"] is False
