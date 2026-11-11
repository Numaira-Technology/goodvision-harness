"""Retain server evidence and reject an output occupied during preparation."""

import builtins
from types import SimpleNamespace

import pytest
from test_qwen8b_vllm_integrity import fake_main_runtime, frozen

from tools import start_qwen8b_vllm as launcher

__all__ = ["frozen"]


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling"])
def test_existing_server_log_is_never_replaced(tmp_path, monkeypatch, kind):
    target = tmp_path / "retained.txt"
    target.write_bytes(b"historical server evidence")
    log = tmp_path / "server.log"
    if kind == "file":
        log.write_bytes(b"historical server evidence")
    else:
        log.symlink_to(target if kind == "symlink" else tmp_path / "missing")
    monkeypatch.setattr(
        launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("process started")
    )
    with pytest.raises(FileExistsError):
        launcher._server_process(
            ["ssh", "synthetic.invalid"],
            "/opt/env/qwen8b-vllm/bin/python",
            "/weights",
            log,
        )
    assert target.read_bytes() == b"historical server evidence"
    if kind != "dangling":
        assert log.read_bytes() == b"historical server evidence"
    else:
        assert log.is_symlink()


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_failed_process_creation_closes_attempt_log(tmp_path, monkeypatch, failure):
    opened = []
    real_open = builtins.open

    def capture(*args, **kwargs):
        stream = real_open(*args, **kwargs)
        opened.append(stream)
        return stream

    def fail(*args, **kwargs):
        kwargs["stdout"].write("attempt failed before process creation\n")
        raise failure("synthetic failure")

    monkeypatch.setattr(launcher, "open", capture, raising=False)
    monkeypatch.setattr(launcher.subprocess, "Popen", fail)
    log = tmp_path / "new.log"
    with pytest.raises(failure):
        launcher._server_process(
            ["ssh", "synthetic.invalid"],
            "/opt/env/qwen8b-vllm/bin/python",
            "/weights",
            log,
        )
    assert len(opened) == 1 and opened[0].closed
    assert log.read_text() == "attempt failed before process creation\n"


def test_successful_process_owns_new_log_until_cleanup(tmp_path, monkeypatch):
    process = SimpleNamespace()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: process)
    log = tmp_path / "new.log"
    result = launcher._server_process(
        ["ssh", "synthetic.invalid"], "/opt/env/qwen8b-vllm/bin/python", "/weights", log
    )
    assert result is process and not result._log_file.closed
    result._log_file.write("retained log\n")
    result._log_file.close()
    assert log.read_text() == "retained log\n"


@pytest.mark.parametrize("kind", ["directory", "dangling-symlink"])
def test_output_created_during_boot_stops_before_smoke(
    tmp_path, monkeypatch, frozen, kind
):
    events = fake_main_runtime(monkeypatch, tmp_path, frozen)
    output = tmp_path / "result"

    def ready(*args):
        if kind == "directory":
            output.mkdir()
            (output / "retained").write_text("original evidence")
        else:
            output.symlink_to(tmp_path / "absent")

    monkeypatch.setattr(launcher, "_wait_ready", ready)
    assert launcher.main() == 1
    assert "smoke" not in events and "eval" not in events
    assert "release" in events
    if kind == "directory":
        assert (output / "retained").read_text() == "original evidence"
    else:
        assert output.is_symlink()
