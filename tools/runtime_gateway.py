"""Supervise one application process and its customer-visible diagnostics."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    from public_privacy import contains_private_implementation_detail
except ImportError:
    try:
        from goodvision_harness.privacy import contains_private_implementation_detail
    except ImportError:
        # An incomplete installation must not expose unfiltered diagnostics.
        def contains_private_implementation_detail(_value: object) -> bool:
            return True


CHECK_ROOT = Path("/opt/goodvision/internal")
CHECKS = {name: f"{name}_check.py" for name in ("runtime", "face", "perception")}
# Leave the application its normal session/persistence shutdown window, with
# a small margin before the delivery container's 90-second stop deadline.
STOP_GRACE_SECONDS = 85.0
MAX_LINE_BYTES = 16_384
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://|\bwww\.", re.IGNORECASE)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_LEVEL = re.compile(r"\b(CRITICAL|ERROR|WARNING|WARN|INFO|DEBUG)\b", re.IGNORECASE)
_LOG_START = re.compile(
    r"^\s*(?:\[?\d{4}-\d{2}-\d{2}[^\n]*?\s+)?\[?"
    r"(?:CRITICAL|ERROR|WARNING|WARN|INFO|DEBUG)(?:\]|:|\s)",
    re.IGNORECASE,
)
_TRACE_LINE = re.compile(
    r"^\s*(?:Traceback\b|Exception Group Traceback\b|File [\"']|"
    r"During handling of the above exception|The above exception|"
    r"[\w.]*(?:Error|Exception|Warning):|\^+$|[+| ]*\+-+)",
    re.IGNORECASE,
)


def _withheld(line: str, *, traceback: bool = False) -> str:
    level = _LEVEL.search(line)
    severity = level.group(1).upper() if level else ("ERROR" if traceback else "WARNING")
    if severity == "WARN":
        severity = "WARNING"
    return f"{severity} diagnostic withheld"


class PublicLogFilter:
    """Hide implementation diagnostics, including continuation stack lines."""

    def __init__(self) -> None:
        self._in_traceback = False

    def project(self, line: str) -> str:
        line = _ANSI.sub("", line.rstrip("\r\n"))
        trace_line = bool(_TRACE_LINE.search(line))
        if trace_line:
            self._in_traceback = True
        elif self._in_traceback and _LOG_START.search(line):
            self._in_traceback = False
        try:
            private = contains_private_implementation_detail(line)
        except Exception:
            private = True
        if (
            self._in_traceback or private or _URL.search(line)
            or any(not char.isprintable() and char != "\t" for char in line)
        ):
            return _withheld(line, traceback=self._in_traceback)
        return line


def _read_output(pipe: object, pending: queue.Queue[str | None]) -> None:
    try:
        while True:
            raw = pipe.readline(MAX_LINE_BYTES + 1)  # type: ignore[attr-defined]
            if not raw:
                break
            if len(raw) > MAX_LINE_BYTES:
                # Do not emit partial fragments: a private marker could cross
                # the length boundary or follow an otherwise harmless prefix.
                diagnostic = _withheld(raw.decode("utf-8", errors="replace"))
                while raw and not raw.endswith(b"\n"):
                    raw = pipe.readline(MAX_LINE_BYTES + 1)  # type: ignore[attr-defined]
                pending.put(diagnostic + "\n")
            else:
                pending.put(raw.decode("utf-8", errors="replace"))
    except Exception:
        pending.put("ERROR diagnostic withheld\n")
    finally:
        pending.put(None)


def _signal_owned_process(process: subprocess.Popen[bytes], signum: int, *, force: bool = False) -> None:
    try:
        if os.name == "posix":
            # The child created this fresh session/group. Never signal a
            # caller's group, another service, or an enumerated system process.
            os.killpg(process.pid, signal.SIGKILL if force else signum)
        elif process.poll() is None:
            if force:
                process.kill()
            else:
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                except (OSError, ValueError):
                    process.terminate()
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.kill() if force else process.terminate()


def _exit_code(returncode: int) -> int:
    return returncode if returncode >= 0 else 128 - returncode


def run_child(command: list[str], *, publish_output: bool) -> int:
    """Capture native descriptors without altering this process's stdout."""

    options: dict[str, object] = {"start_new_session": True} if os.name == "posix" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
    }
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **options,
    )
    assert process.stdout is not None
    pending: queue.Queue[str | None] = queue.Queue(maxsize=256)
    reader = threading.Thread(target=_read_output, args=(process.stdout, pending), daemon=True)
    reader.start()
    previous: dict[int, signal.Handlers] = {}
    interrupted: list[tuple[int, float]] = []

    def forward(signum: int, _frame: object) -> None:
        if not interrupted:
            interrupted.append((signum, time.monotonic()))
        _signal_owned_process(process, signum)

    output_done = False
    exited_at: float | None = None
    log_filter = PublicLogFilter()
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, forward)
        while True:
            if interrupted and time.monotonic() - interrupted[0][1] >= STOP_GRACE_SECONDS:
                _signal_owned_process(process, interrupted[0][0], force=True)
            if process.poll() is not None:
                if exited_at is None:
                    exited_at = time.monotonic()
                if output_done or time.monotonic() - exited_at > 1.0:
                    break
            try:
                line = pending.get(timeout=0.05)
            except queue.Empty:
                continue
            if line is None:
                output_done = True
            elif publish_output:
                print(log_filter.project(line), flush=True)
        returncode = _exit_code(process.wait())
        # A cancelled successful-looking check has not completed its audit.
        if interrupted and returncode == 0 and not publish_output:
            returncode = 128 + interrupted[0][0]
        return returncode
    finally:
        if process.poll() is None:
            _signal_owned_process(process, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _signal_owned_process(process, signal.SIGTERM, force=True)
                process.wait(timeout=2)
        # Also retire any descendants that kept the output pipe after their
        # immediate parent exited. The Linux delivery uses a dedicated group.
        if os.name == "posix":
            _signal_owned_process(process, signal.SIGTERM, force=True)
        reader.join(timeout=2)
        if not reader.is_alive():
            process.stdout.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _serve_command(component: str) -> list[str]:
    if component == "main":
        return ["goodvision-harness"]
    return [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091", "--no-server-header"]


def _check_record(check: str, returncode: int, message: str | None = None) -> None:
    record = {"check": check, "status": "passed" if returncode == 0 else "failed"}
    if message is not None:
        record["message"] = message
    print(json.dumps(record, separators=(",", ":")), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in {"serve", *CHECKS}:
        _check_record("request", 2, "invalid runtime request")
        return 2
    mode = args[0]
    component = os.environ.get("GOODVISION_COMPONENT", "main")
    if component not in {"main", "perception"}:
        _check_record(mode, 2, "runtime component is unavailable")
        return 2
    command = _serve_command(component) if mode == "serve" else [
        sys.executable, str(CHECK_ROOT / CHECKS[mode])
    ]
    try:
        returncode = run_child(command, publish_output=mode == "serve")
    except FileNotFoundError:
        returncode = 127
    except Exception:
        returncode = 1
    if mode != "serve":
        _check_record(mode, returncode, "check did not complete successfully" if returncode else None)
    elif returncode:
        print("ERROR service process stopped unsuccessfully", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
