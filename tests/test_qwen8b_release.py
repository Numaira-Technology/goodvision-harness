"""Run real launcher cleanup with synthetic cloud/SSH/evaluation operations."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import start_qwen8b_r2 as host
from tools import start_qwen8b_vllm as vllm
from tools.qwen8b import eval_local


class FakeVast:
    def __init__(self, fault):
        self.fault = fault
        self.client = SimpleNamespace(timeout=(3, 4))
        self.alive = False
        self.destroyed = []

    def search_offers(self, **kwargs):
        return [
            {
                "id": 1,
                "num_gpus": 1,
                "gpu_ram": 24564,
                "dph_total": 1.1,
                "cuda_max_good": 13.0,
                "driver_version": "580.65.06",
            }
        ]

    def destroy_instance(self, *, id):
        self.destroyed.append(id)
        if self.fault == "rejected":
            return {"success": False}
        if self.fault == "exception":
            raise OSError("synthetic destroy failure")
        if self.fault != "still-live":
            self.alive = False
        return {"success": True}

    def show_instances(self):
        if self.fault == "list-error":
            raise OSError("synthetic list failure")
        if self.fault == "malformed-list":
            return None
        if self.alive:
            return [
                {"id": "123", "label": "synthetic-owned", "actual_status": "running"}
            ]
        return [{"id": 987, "label": "unrelated"}] if self.fault == "unrelated" else []


def fake_runtime(monkeypatch, tmp_path, module, fault, *, interrupted=False):
    vast = FakeVast(fault)
    key = tmp_path / "synthetic-key"
    key.write_text("not a real SSH key")
    status = tmp_path / "status.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--eval-bundle",
            str(tmp_path / "bundle"),
            "--eval-output",
            str(tmp_path / "eval"),
            "--status-file",
            str(status),
        ],
    )
    monkeypatch.setitem(
        sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda *a, **k: None)
    )
    monkeypatch.setitem(sys.modules, "vastai", SimpleNamespace(VastAI=lambda **k: vast))
    monkeypatch.setattr(
        host.S,
        "required",
        lambda name: str(key) if name == "SAM_SSH_KEY" else "SYNTHETIC",
    )
    monkeypatch.setattr(host.S, "IMAGE", "synthetic")
    for name in host.R2_KEYS:
        monkeypatch.setenv(name, "synthetic")
    monkeypatch.setattr(host.S, "prepare_ssh_key", lambda *a: key)

    def create(*args, **kwargs):
        vast.alive = True
        return 123

    monkeypatch.setattr(host.S, "create_from_offers", create)
    monkeypatch.setattr(host.S, "run", lambda *a, **k: None)
    stopped = []
    monkeypatch.setattr(host.S, "stop_process", lambda *a: stopped.append(True))
    monkeypatch.setattr(host, "attach_key", lambda *a: None)
    monkeypatch.setattr(host, "wait_for_ssh", lambda *a: ("synthetic.invalid", 22))
    monkeypatch.setattr(host, "_upload_repo", lambda *a: None)
    monkeypatch.setattr(module, "prefetch_models", lambda **k: None)
    monkeypatch.setattr(
        module, "restore_env", lambda **k: "/opt/env/synthetic/bin/activate"
    )
    monkeypatch.setattr(
        module, "_verify_eval_bundle", lambda *a: "synthetic-bundle-sha"
    )
    monkeypatch.setattr(
        module, "_server_process", lambda *a: SimpleNamespace(_log_file=None)
    )
    monkeypatch.setattr(module, "_wait_ready", lambda *a: {})
    if module is vllm:
        # Applicable to both the source cu128 branch and its cu130 successor.
        monkeypatch.setattr(
            module, "_verify_host_driver", lambda *a: None, raising=False
        )
        monkeypatch.setattr(module, "_verify_restored_env", lambda *a: None)
        monkeypatch.setattr(
            module, "_remote_snapshot", lambda *a: "/synthetic/snapshot"
        )
        monkeypatch.setattr(module, "_smoke", lambda *a: None)
        monkeypatch.setattr(module, "_status", lambda path, text: path.write_text(text))
        monkeypatch.setattr(
            module.subprocess, "check_output", lambda *a, **k: "synthetic GPU status\n"
        )

    def evaluate(argv):
        if interrupted:
            raise KeyboardInterrupt
        return 0

    monkeypatch.setattr(eval_local, "main", evaluate)
    return vast, status, stopped


@pytest.mark.parametrize("module", [host, vllm], ids=["hf", "vllm"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "rejected",
        "exception",
        "still-live",
        "list-error",
        "malformed-list",
        "unrelated",
    ],
)
def test_success_requires_confirmed_owned_release(tmp_path, monkeypatch, module, fault):
    vast, status, stopped = fake_runtime(monkeypatch, tmp_path, module, fault)
    assert module.main() == (0 if fault in (None, "unrelated") else 1)
    assert stopped == [True] and vast.destroyed == [123]
    assert vast.client.timeout == (3, 4)
    if fault not in (None, "unrelated"):
        assert "release-unconfirmed" in status.read_text()
        assert "123" in status.read_text()


@pytest.mark.parametrize("module", [host, vllm], ids=["hf", "vllm"])
@pytest.mark.parametrize("fault", [None, "rejected"])
def test_interrupt_still_attempts_release_and_stays_nonzero(
    tmp_path, monkeypatch, module, fault
):
    vast, status, stopped = fake_runtime(
        monkeypatch, tmp_path, module, fault, interrupted=True
    )
    assert module.main() == 130
    assert stopped == [True] and vast.destroyed == [123]
    if fault:
        assert "release-unconfirmed" in status.read_text()


@pytest.mark.parametrize("module", [host, vllm], ids=["hf", "vllm"])
@pytest.mark.parametrize("location", ["inside", "same", "ancestor", "symlink"])
def test_overlapping_status_rejected_before_credentials(
    tmp_path, monkeypatch, module, location
):
    vast, _, _ = fake_runtime(monkeypatch, tmp_path, module, None)
    output = tmp_path / "eval"
    status = output / "STATUS.md"
    if location == "same":
        status = output
    elif location == "ancestor":
        status = tmp_path
    elif location == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(output, target_is_directory=True)
        status = alias / "STATUS.md"
    monkeypatch.setattr(sys, "argv", sys.argv[:-2] + ["--status-file", str(status)])
    monkeypatch.setattr(
        sys.modules["dotenv"], "load_dotenv",
        lambda *a, **k: pytest.fail("credentials read before offline rejection"),
    )
    with pytest.raises(ValueError, match="status.*evaluation output"):
        module.main()
    assert not vast.alive and not vast.destroyed
    assert not output.exists()


def test_default_vllm_status_allows_documented_fresh_output(tmp_path, monkeypatch):
    real_status = vllm._status
    vast, _, _ = fake_runtime(monkeypatch, tmp_path, vllm, None)
    monkeypatch.chdir(tmp_path)
    output = Path("outputs/qwen8b-r2-vllm")
    monkeypatch.setattr(sys, "argv", [
        "launcher", "--eval-bundle", str(tmp_path / "bundle"),
        "--eval-output", str(output),
    ])
    # Exercise real status writes, keeping its optional artifact mirror local.
    monkeypatch.setattr(vllm, "Path", lambda value: (
        tmp_path / "mirror-status.md"
        if str(value) == "/opt/cursor/artifacts/qwen8b-r2-vllm-STATUS.md"
        else Path(value)
    ))
    monkeypatch.setattr(vllm, "_status", real_status)
    evaluated = []

    def evaluate(argv):
        assert not output.exists()
        output.mkdir(parents=True)
        evaluated.append(argv)
        return 0

    monkeypatch.setattr(eval_local, "main", evaluate)
    assert vllm.main() == 0
    assert len(evaluated) == 1 and vast.destroyed == [123]
    assert "Phase: released" in vllm.STATUS_DEFAULT.read_text()
