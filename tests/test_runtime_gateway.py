from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gateway() -> Any:
    return _load(ROOT / "tools" / "runtime_gateway.py")


@pytest.mark.parametrize(
    ("text", "level"),
    (
        ("ERROR Paddle backend failed", "ERROR"),
        ("WARNING InsightFace model missing", "WARNING"),
        ("WARN Qwen response unavailable", "WARNING"),
        ("INFO OpenRouter route selected", "INFO"),
        ("ERROR https://unlisted-provider.invalid/private", "ERROR"),
        ("WARNING rtsp://camera-user:camera-secret@camera.invalid/live", "WARNING"),
        ('  File "/opt/private_engine/session.py", line 1, in start', "ERROR"),
        ("RuntimeError: internal implementation path", "ERROR"),
        ("ERROR \x1b[31mPad\x1b[0mdle backend", "ERROR"),
    ),
)
def test_private_diagnostics_are_withheld_without_losing_severity(
    gateway: Any, text: str, level: str
) -> None:
    assert gateway.PublicLogFilter().project(text) == f"{level} diagnostic withheld"


def test_traceback_continuations_are_withheld_until_normal_logging_resumes(gateway: Any) -> None:
    project = gateway.PublicLogFilter().project
    assert project("Traceback (most recent call last):") == "ERROR diagnostic withheld"
    assert project("    hidden_internal_function()") == "ERROR diagnostic withheld"
    assert project("ValueError: implementation details") == "ERROR diagnostic withheld"
    assert project("INFO service ready") == "INFO service ready"
    assert project("processed frame 12") == "processed frame 12"


def test_interleaved_normal_log_cannot_expose_later_stack_source(gateway: Any) -> None:
    project = gateway.PublicLogFilter().project
    assert project("Traceback (most recent call last):") == "ERROR diagnostic withheld"
    assert project("INFO other thread ready") == "INFO other thread ready"
    assert project("    hidden_internal_function('private parameter')") == (
        "ERROR diagnostic withheld"
    )
    assert project("ValueError: private details") == "ERROR diagnostic withheld"
    assert project("INFO service recovered") == "INFO service recovered"


def test_native_stdout_and_stderr_are_combined_and_filtered(
    gateway: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    command = [sys.executable, "-c", (
        "import os; "
        "os.write(1, b'INFO service ready\\n'); "
        "os.write(2, b'ERROR Paddle native failure\\n'); "
        "os.write(1, b'WARNING https://unlisted.invalid/model\\n')"
    )]
    assert gateway.run_child(command, publish_output=True) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines() == [
        "INFO service ready", "ERROR diagnostic withheld", "WARNING diagnostic withheld"
    ]


def test_long_native_lines_are_not_emitted_in_fragments(
    gateway: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    command = [sys.executable, "-c", (
        "import os; os.write(2, b'ERROR ' + b'x' * 20000 + b'Paddle\\n')"
    )]
    assert gateway.run_child(command, publish_output=True) == 0
    assert capsys.readouterr().out == "ERROR diagnostic withheld\n"


@pytest.mark.parametrize("returncode", (0, 7))
def test_check_mode_captures_everything_and_preserves_child_exit_code(
    gateway: Any, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], returncode: int,
) -> None:
    real_run = gateway.run_child
    calls: list[tuple[list[str], bool]] = []

    def run(command: list[str], *, publish_output: bool) -> int:
        calls.append((command, publish_output))
        return real_run([sys.executable, "-c", (
            "import os,sys; os.write(1,b'Paddle stdout\\n'); "
            "os.write(2,b'InsightFace stderr\\n'); sys.exit(" + str(returncode) + ")"
        )], publish_output=publish_output)

    monkeypatch.setattr(gateway, "run_child", run)
    monkeypatch.setenv("GOODVISION_COMPONENT", "main")
    assert gateway.main(["runtime"]) == returncode
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert captured.err == ""
    assert record["check"] == "runtime"
    assert record["status"] == ("passed" if returncode == 0 else "failed")
    assert "paddle" not in captured.out.lower()
    assert "insightface" not in captured.out.lower()
    assert calls == [([
        sys.executable, str(gateway.CHECK_ROOT / "face_check.py"), "--runtime"
    ], False)]


def test_check_spawn_failure_never_reports_success(
    gateway: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> int:
        raise FileNotFoundError("Paddle https://private.invalid/internal.py")

    monkeypatch.setattr(gateway, "run_child", fail)
    monkeypatch.setenv("GOODVISION_COMPONENT", "main")
    assert gateway.main(["face"]) == 127
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "failed"
    assert "Paddle" not in captured.out
    assert "https://" not in captured.out


@pytest.mark.parametrize("component", ("main", "perception"))
def test_service_command_uses_fixed_component_entrypoint(
    gateway: Any, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], component: str,
) -> None:
    calls: list[tuple[list[str], bool]] = []
    monkeypatch.setenv("GOODVISION_COMPONENT", component)
    monkeypatch.setattr(gateway, "run_child", lambda command, *, publish_output: calls.append((command, publish_output)) or 19)
    assert gateway.main(["serve"]) == 19
    assert calls == [(gateway._serve_command(component), True)]
    assert "ERROR service process stopped unsuccessfully" in capsys.readouterr().out
    if component == "main":
        assert calls[0][0] == ["goodvision-harness"]
    else:
        assert calls[0][0] == [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091", "--no-server-header"]


@pytest.mark.parametrize("check", ("runtime", "face", "perception"))
def test_customer_wrapper_has_fixed_gateway_path_and_returns_failure(
    monkeypatch: pytest.MonkeyPatch, check: str,
) -> None:
    path = ROOT / "tools" / "customer_check.py"
    module = _load(path)
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "__file__", f"/app/tools/check_{check}.py")

    def replace_process(executable: str, command: list[str]) -> None:
        assert executable == sys.executable
        calls.append(command)
        raise SystemExit(23)

    monkeypatch.setattr(module.os, "execv", replace_process)
    with pytest.raises(SystemExit) as stopped:
        module.main([])
    assert stopped.value.code == 23
    assert calls == [[sys.executable, "/opt/goodvision/runtime_gateway.py", check]]
    text = path.read_text(encoding="utf-8").lower()
    assert not any(marker in text for marker in ("paddle", "insightface", "qwen", "openrouter", "onnx", "cuda"))


@pytest.mark.parametrize(
    ("error", "expected"),
    [(FileNotFoundError("private details"), 127), (OSError("private details"), 126)],
)
def test_customer_launcher_spawn_errors_have_neutral_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    error: OSError, expected: int,
) -> None:
    module = _load(ROOT / "tools" / "customer_check.py")
    monkeypatch.setattr(module, "__file__", "/app/tools/check_runtime.py")

    def fail(*_args: Any) -> None:
        raise error

    monkeypatch.setattr(module.os, "execv", fail)
    assert module.main([]) == expected
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "failed"
    assert "private details" not in str(record)


@pytest.mark.parametrize("component", ("main", "perception"))
def test_runtime_check_selects_the_component_script_without_gpu_mode(
    gateway: Any, component: str,
) -> None:
    script = "face" if component == "main" else "perception"
    assert gateway._check_command(component, "runtime") == [
        sys.executable, str(gateway.CHECK_ROOT / f"{script}_check.py"), "--runtime"
    ]
    assert gateway._check_command(component, script) == [
        sys.executable, str(gateway.CHECK_ROOT / f"{script}_check.py")
    ]


def test_reader_start_failure_cleans_up_owned_child_and_restores_handlers(
    gateway: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_DFL}
    forwarded: list[int] = []

    class Process:
        pid = 43210
        stdout = io.BytesIO()
        stopped = False

        def poll(self) -> int | None:
            return 0 if self.stopped else None

        def wait(self, **_kwargs: Any) -> int:
            assert self.stopped
            return 0

    class Reader:
        def start(self) -> None:
            raise RuntimeError("reader startup failed")

        def join(self, **_kwargs: Any) -> None:
            raise AssertionError("must not join a thread that never started")

    process = Process()

    def stop(_process: object, signum: int, **_kwargs: Any) -> None:
        forwarded.append(signum)
        process.stopped = True

    monkeypatch.setattr(gateway.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(gateway.threading, "Thread", lambda **_kwargs: Reader())
    monkeypatch.setattr(gateway.signal, "getsignal", lambda signum: handlers[signum])
    monkeypatch.setattr(gateway.signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler))
    monkeypatch.setattr(gateway, "_signal_owned_process", stop)
    with pytest.raises(RuntimeError, match="reader startup failed"):
        gateway.run_child(["owned-child"], publish_output=True)
    assert forwarded[0] == signal.SIGTERM
    assert process.stdout.closed
    assert handlers == {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_DFL}


def test_signal_targets_only_the_created_process_group(gateway: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.name != "posix":
        pytest.skip("The shipped Linux container uses process-group signaling")
    calls: list[tuple[int, int]] = []
    process = type("Process", (), {"pid": 43210, "poll": lambda self: None})()
    monkeypatch.setattr(gateway.os, "killpg", lambda pid, signum: calls.append((pid, signum)))
    gateway._signal_owned_process(process, signal.SIGTERM)
    gateway._signal_owned_process(process, signal.SIGINT)
    gateway._signal_owned_process(process, signal.SIGTERM, force=True)
    assert calls == [(43210, signal.SIGTERM), (43210, signal.SIGINT), (43210, signal.SIGKILL)]


def test_signal_exit_code_uses_standard_shell_status(gateway: Any) -> None:
    assert gateway._exit_code(7) == 7
    assert gateway._exit_code(-signal.SIGTERM) == 128 + signal.SIGTERM


@pytest.mark.parametrize("publish_output", (False, True))
def test_signals_are_forwarded_and_cancelled_checks_never_pass(
    gateway: Any, monkeypatch: pytest.MonkeyPatch, publish_output: bool,
) -> None:
    handlers: dict[int, Any] = {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_DFL}
    forwarded: list[int] = []

    class Process:
        pid = 43210
        stdout = io.BytesIO()
        stopped = False

        def poll(self) -> int | None:
            if not self.stopped and callable(handlers[signal.SIGTERM]):
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return 0 if self.stopped else None

        def wait(self, **_kwargs: Any) -> int:
            assert self.stopped
            return 0

    process = Process()

    def forward(_process: object, signum: int, **_kwargs: Any) -> None:
        forwarded.append(signum)
        process.stopped = True

    monkeypatch.setattr(gateway.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(gateway.signal, "getsignal", lambda signum: handlers[signum])
    monkeypatch.setattr(gateway.signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler))
    monkeypatch.setattr(gateway, "_signal_owned_process", forward)
    result = gateway.run_child(["test-owned-child"], publish_output=publish_output)
    assert result == (0 if publish_output else 128 + signal.SIGTERM)
    assert forwarded[0] == signal.SIGTERM
    assert handlers == {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_DFL}
