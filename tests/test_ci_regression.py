"""CI partition/gate contracts and a real bounded subprocess, all offline."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from tools.ci_regression import shard_for, verify_coverage

ROOT = Path(__file__).resolve().parents[1]


def manifests(tmp_path):
    nodes = [
        f"tests/test_example_{n}.py::test_case[{p}]"
        for n in range(20)
        for p in range(3)
    ]
    for index in range(2):
        path = tmp_path / str(index)
        path.mkdir()
        (path / "collection.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "shard": index,
                    "shards": 2,
                    "collected": nodes,
                    "selected": [node for node in nodes if shard_for(node, 2) == index],
                }
            )
        )
        (path / "run.json").write_text(json.dumps({"exit_code": 0, "timed_out": False}))
        rows = [
            {"event": "result", "nodeid": node, "stage": stage, "outcome": "passed"}
            for node in nodes
            if shard_for(node, 2) == index
            for stage in ("setup", "call", "teardown")
        ] + [{"event": "finished", "exit_code": 0}]
        (path / "progress.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    return nodes


def test_partition_preserves_modules_and_exact_once(tmp_path):
    nodes = manifests(tmp_path)
    assert verify_coverage(tmp_path, 2, "success") == len(nodes)
    assert shard_for("tests/a.py::test_one", 2) == shard_for(
        "tests/a.py::test_two[1]", 2
    )
    with pytest.raises(ValueError):
        shard_for("tests/a.py::test_one", 0)


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", "", "pending"])
def test_gate_rejects_every_non_success_job_result(tmp_path, result):
    manifests(tmp_path)
    with pytest.raises(ValueError, match="did not succeed"):
        verify_coverage(tmp_path, 2, result)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "extra",
        "duplicate-shard",
        "wrong-count",
        "wrong-universe",
        "reordered",
        "duplicate-node",
        "missing-selected",
        "duplicate-selected",
        "misassigned",
        "empty",
        "failed",
        "timeout",
        "missing-run",
        "truncated",
    ],
)
def test_gate_fails_closed_on_incomplete_or_corrupt_evidence(tmp_path, damage):
    nodes = manifests(tmp_path)
    path = tmp_path / "0" / "collection.json"
    data = json.loads(path.read_text())
    if damage == "missing":
        path.unlink()
    elif damage == "extra":
        (tmp_path / "extra").mkdir()
        (tmp_path / "extra" / "collection.json").write_text(path.read_text())
    elif damage == "duplicate-shard":
        data["shard"] = 1
    elif damage == "wrong-count":
        data["shards"] = 3
    elif damage == "wrong-universe":
        data["collected"] = nodes[:-1]
    elif damage == "reordered":
        data["collected"].reverse()
    elif damage == "duplicate-node":
        data["collected"].append(nodes[0])
    elif damage == "missing-selected":
        data["selected"].pop()
    elif damage == "duplicate-selected":
        data["selected"].append(data["selected"][0])
    elif damage == "misassigned":
        data["selected"].append(next(n for n in nodes if shard_for(n, 2) == 1))
    elif damage == "empty":
        data["collected"] = []
    elif damage in {"failed", "timeout"}:
        (path.parent / "run.json").write_text(
            json.dumps(
                {
                    "exit_code": 1 if damage == "failed" else 124,
                    "timed_out": damage == "timeout",
                }
            )
        )
    elif damage == "missing-run":
        (path.parent / "run.json").unlink()
    elif damage == "truncated":
        path.write_text('{"schema":')
    if damage not in {"missing", "truncated"}:
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_coverage(tmp_path, 2, "success")


def invoke(tmp_path, index, count, tests, *, seconds=30):
    output = tmp_path / f"evidence-{index}"
    command = [
        sys.executable,
        str(ROOT / "tools/ci_regression.py"),
        "run",
        "--shard",
        str(index),
        "--shards",
        str(count),
        "--seconds",
        str(seconds),
        "--output",
        str(output),
        *map(str, tests),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=seconds + 20,
    )
    return result, output


@pytest.mark.skipif(sys.platform != "linux", reason="Linux CI process-group runner")
def test_real_shards_emit_complete_collection_progress_and_junit(tmp_path):
    files = {}
    for n in range(40):
        name = f"test_synthetic_{n}.py"
        files.setdefault(shard_for(name + "::test_ok", 2), name)
        if len(files) == 2:
            break
    assert len(files) == 2
    for name in files.values():
        (tmp_path / name).write_text(
            "def test_ok(): assert True\ndef test_another(): assert True\n"
        )
    for index in range(2):
        result, output = invoke(tmp_path, index, 2, list(files.values()))
        assert result.returncode == 0, result.stdout + result.stderr
        assert "2 passed" in (output / "stdout.log").read_text()
        assert (output / "junit.xml").exists()
        progress = [
            json.loads(line)
            for line in (output / "progress.jsonl").read_text().splitlines()
        ]
        assert progress[-1]["event"] == "finished"
        assert {row["stage"] for row in progress if row["event"] == "stage"} == {
            "setup",
            "call",
            "teardown",
        }
        assert "TIME " in (output / "processes.log").read_text()
    assert verify_coverage(tmp_path, 2, "success") == 4


@pytest.mark.skipif(sys.platform != "linux", reason="Linux CI process-group runner")
def test_failure_is_not_changed_into_success(tmp_path):
    (tmp_path / "test_failure.py").write_text(
        "def test_fail(): assert False, 'intentional evidence'\n"
    )
    result, output = invoke(tmp_path, 0, 1, ["test_failure.py"])
    assert result.returncode == 1
    assert "intentional evidence" in (output / "stdout.log").read_text()
    assert json.loads((output / "run.json").read_text())["exit_code"] == 1
    assert (output / "junit.xml").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group watchdog")
@pytest.mark.parametrize("detached", [False, True])
def test_timeout_keeps_stack_progress_and_kills_test_children(tmp_path, detached):
    (tmp_path / "test_hang.py").write_text(
        """import subprocess, sys, time
from pathlib import Path
def test_hang():
    child = subprocess.Popen([sys.executable, '-c',
        'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN);'
        'time.sleep(60)'], start_new_session=DETACHED)
    Path('child.pid').write_text(str(child.pid))
    time.sleep(60)
""".replace("DETACHED", repr(detached))
    )
    result, output = invoke(tmp_path, 0, 1, ["test_hang.py"], seconds=4)
    assert result.returncode == 124
    record = json.loads((output / "run.json").read_text())
    assert record["timed_out"] is True and record["elapsed_seconds"] < 15
    assert "test_hang" in (output / "stacks.log").read_text()
    assert '"stage": "call"' in (output / "progress.jsonl").read_text()
    pid = int((tmp_path / "child.pid").read_text())
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        assert stat.read_text().split(") ", 1)[1].split()[0] == "Z"


@pytest.mark.parametrize("selection", ["-k", "-m", "--deselect"])
def test_ci_plugin_rejects_hidden_selection(tmp_path, selection):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tools.ci_pytest",
            "--ci-shard=0",
            "--ci-shards=2",
            "--ci-evidence=" + str(tmp_path / "evidence"),
            selection,
            "anything",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode != 0
    assert "CI collection cannot use" in proc.stderr


def test_workflow_preserves_required_names_limits_and_optional_jobs():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    jobs = workflow["jobs"]
    assert jobs["tests"]["name"] == "Tests (ubuntu-latest, Python 3.11)"
    assert jobs["tests"]["if"] == "always()" and jobs["tests"]["needs"] == "linux-tests"
    assert jobs["windows-tests"]["name"] == "Tests (windows-latest, Python 3.12)"
    assert jobs["windows-tests"]["timeout-minutes"] == 25
    linux = jobs["linux-tests"]
    assert linux["timeout-minutes"] == 20 and linux["strategy"]["fail-fast"] is False
    assert linux["strategy"]["matrix"]["shard"] == [0, 1]
    assert not any(job.get("continue-on-error") for job in jobs.values())
    for name in ["linux-tests", "windows-tests"]:
        steps = jobs[name]["steps"]
        assert any('".[dev,webrtc,vlm]"' in step.get("run", "") for step in steps)
        assert any("Fetch pinned historical" in step.get("name", "") for step in steps)
    uploads = [
        step
        for step in linux["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1 and uploads[0]["if"] == "always()"
    assert {
        "sam3-onnx-contract",
        "preview-browser",
        "gateway-video-integration",
    } <= jobs.keys()


@pytest.mark.parametrize(
    "damage",
    [
        "missing-finish",
        "missing-node",
        "missing-call",
        "duplicate-result",
        "unexpected-node",
        "failed-result",
        "early-finish",
        "failed-session",
    ],
)
def test_gate_requires_complete_execution_ledger(tmp_path, damage):
    manifests(tmp_path)
    path = tmp_path / "0/progress.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == "missing-finish":
        rows.pop()
    elif damage == "missing-node":
        rows = [r for r in rows if r.get("nodeid") != rows[0]["nodeid"]]
    elif damage == "missing-call":
        rows.pop(1)
    elif damage == "duplicate-result":
        rows.insert(0, dict(rows[0]))
    elif damage == "unexpected-node":
        rows[0]["nodeid"] = "not-selected.py::test_other"
    elif damage == "failed-result":
        rows[0]["outcome"] = "failed"
    elif damage == "early-finish":
        rows.insert(0, {"event": "finished", "exit_code": 0})
    elif damage == "failed-session":
        rows[-1]["exit_code"] = 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        verify_coverage(tmp_path, 2, "success")


def test_gate_allows_recorded_setup_skip(tmp_path):
    nodes = manifests(tmp_path)
    path = tmp_path / "0/progress.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["outcome"] = "skipped"
    rows.pop(1)  # A legitimate setup skip never calls the test body.
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert verify_coverage(tmp_path, 2, "success") == len(nodes)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux CI subprocess runner")
@pytest.mark.parametrize(
    "exit_code",
    ["import os; os._exit(0)", "import pytest; pytest.exit('early', returncode=0)"],
)
def test_successful_early_process_exit_cannot_pass_gate(tmp_path, exit_code):
    (tmp_path / "test_early.py").write_text(
        f"def test_early():\n    {exit_code}\ndef test_never_run(): assert False\n"
    )
    result, _ = invoke(tmp_path, 0, 1, ["test_early.py"])
    assert result.returncode == 0  # Neither pytest nor the process reports failure.
    with pytest.raises(ValueError):
        verify_coverage(tmp_path, 1, "success")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux cancellation cleanup")
def test_sigterm_cancels_run_cleans_detached_child_and_preserves_unrelated(tmp_path):
    (tmp_path / "test_cancel.py").write_text("""import subprocess, sys, time
from pathlib import Path
def test_cancel():
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'],
                             start_new_session=True)
    Path('child.pid').write_text(str(child.pid))
    time.sleep(60)
""")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    output = tmp_path / "cancel-evidence"
    command = [
        sys.executable,
        str(ROOT / "tools/ci_regression.py"),
        "run",
        "--shard=0",
        "--shards=1",
        "--seconds=30",
        "--output",
        str(output),
        "test_cancel.py",
    ]
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    with (tmp_path / "runner.log").open("w") as log:
        runner = subprocess.Popen(
            command, cwd=tmp_path, env=env, stdout=log, stderr=log
        )
        try:
            deadline = time.monotonic() + 10
            while not (tmp_path / "child.pid").exists() and time.monotonic() < deadline:
                assert runner.poll() is None
                time.sleep(0.05)
            assert (tmp_path / "child.pid").exists()
            runner.send_signal(signal.SIGTERM)
            assert runner.wait(timeout=15) == 143
            record = json.loads((output / "run.json").read_text())
            assert (
                record["cancelled_signal"] == signal.SIGTERM
                and record["exit_code"] == 143
            )
            pid = int((tmp_path / "child.pid").read_text())
            assert not Path(f"/proc/{pid}").exists()
            assert unrelated.poll() is None
        finally:
            if runner.poll() is None:
                runner.send_signal(signal.SIGTERM)
                runner.wait(timeout=15)
            unrelated.terminate()
            unrelated.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux shutdown watchdog")
def test_timeout_retains_stacks_after_pytest_session_finishes(tmp_path):
    (tmp_path / "test_shutdown.py").write_text("""import threading, time
def wait_forever():
    time.sleep(60)
def test_pass_then_hang_at_shutdown():
    threading.Thread(target=wait_forever, daemon=False).start()
""")
    # Preserve this node's real three-second watchdog and fail-closed gate.
    # Startup may consume that budget, so finished-session/shutdown-stack
    # assertions live in the readiness-gated bootstrap lifecycle test below.
    result, output = invoke(tmp_path, 0, 1, ["test_shutdown.py"], seconds=3)
    assert result.returncode == 124
    record = json.loads((output / "run.json").read_text())
    assert record["timed_out"] is True and record["timeout_seconds"] == 3
    with pytest.raises(ValueError):
        verify_coverage(tmp_path, 1, "success")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux shutdown signal")
def test_bootstrap_retains_stacks_after_pytest_session_finishes(tmp_path):
    (tmp_path / "test_shutdown.py").write_text("""import threading, time
from pathlib import Path

def wait_forever():
    time.sleep(60)

def test_pass_then_hang_at_shutdown():
    # This callback runs inside the real threading._shutdown, before it joins
    # our non-daemon thread. Normal atexit callbacks would run too late.
    threading._register_atexit(lambda: Path("shutdown-ready").touch())
    threading.Thread(target=wait_forever, daemon=False).start()
""")
    output = tmp_path / "evidence"
    output.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    ready, stack = tmp_path / "shutdown-ready", output / "stacks.log"
    with (
        (output / "stdout.log").open("wb") as stdout,
        (output / "stderr.log").open("wb") as stderr,
    ):
        child = subprocess.Popen(
            [
                sys.executable, "-u", "-m", "tools.ci_regression", "_pytest",
                str(output), "-q", "-p", "tools.ci_pytest", "--ci-shard=0",
                "--ci-shards=1", "--ci-evidence=" + str(output),
                "--junitxml=" + str(output / "junit.xml"), "test_shutdown.py",
            ],
            cwd=tmp_path, env=env, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                assert child.poll() is None, "bootstrap exited before shutdown"
                time.sleep(0.01)
            assert ready.exists(), "bootstrap did not reach shutdown within 10s"
            progress = [
                json.loads(line)
                for line in (output / "progress.jsonl").read_text().splitlines()
            ]
            assert (
                progress[-1]["event"] == "finished"
                and progress[-1]["exit_code"] == 0
            )
            assert child.poll() is None
            os.kill(child.pid, signal.SIGUSR1)
            deadline = time.monotonic() + 3
            stacks, frames = "", set()
            while time.monotonic() < deadline:
                stacks = stack.read_text()
                frames = {
                    line.rsplit(" in ", 1)[-1]
                    for line in stacks.splitlines()
                    if line.lstrip().startswith("File ") and " in " in line
                }
                if {"wait_forever", "_shutdown"} <= frames:
                    break
                assert child.poll() is None, "bootstrap exited before stack dump"
                time.sleep(0.01)
            assert "wait_forever" in stacks and "_shutdown" in stacks
            assert {"wait_forever", "_shutdown"} <= frames
        finally:
            # The owned session contains only this bootstrap and its thread.
            # Keep cleanup bounded even when readiness or stack assertions fail.
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=5)
            else:
                child.wait(timeout=5)


@pytest.mark.skipif(
    sys.platform != "linux", reason="Linux stdout backpressure regression"
)
@pytest.mark.parametrize("finish", ["failure", "timeout"])
def test_stalled_console_cannot_block_capture_or_watchdog(tmp_path, finish):
    (tmp_path / "test_output.py").write_text('''import os, time
def test_large_output(capfd):
    with capfd.disabled():
        os.write(1, b"stdout-marker:" + b"X" * 400000)
        os.write(2, b"stderr-marker:" + b"Y" * 400000)
    ENDING
'''.replace(
        "ENDING",
        "time.sleep(60)"
        if finish == "timeout"
        else "assert False, 'synthetic failure'",
    ))
    output = tmp_path / "stalled-console"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    runner = subprocess.Popen(
        [sys.executable, str(ROOT / "tools/ci_regression.py"), "run",
         "--shard=0", "--shards=1", "--seconds=3", "--output",
         str(output), "test_output.py"],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Deliberately do not drain either console pipe until the wrapper exits.
        assert runner.wait(timeout=15) == (124 if finish == "timeout" else 1)
        record = json.loads((output / "run.json").read_text())
        assert record["timed_out"] == (finish == "timeout")
        assert record["elapsed_seconds"] < 15
        assert b"stdout-marker:" + b"X" * 400000 in (output / "stdout.log").read_bytes()
        assert b"stderr-marker:" + b"Y" * 400000 in (output / "stderr.log").read_bytes()
        if finish == "timeout":
            assert "test_large_output" in (output / "stacks.log").read_text()
        else:
            assert (output / "junit.xml").exists()
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait(timeout=5)
        runner.stdout.close()
        runner.stderr.close()

@pytest.mark.skipif(sys.platform != "linux", reason="Linux diagnostic signal")
def test_stack_handler_defers_dump_to_python_and_keeps_descriptor(
    tmp_path, monkeypatch
):
    from tools import ci_regression as runner

    installed, calls = [], []
    exit_callbacks = []
    monkeypatch.setattr(
        runner.atexit, "register", lambda *args: exit_callbacks.append(args)
    )
    monkeypatch.setattr(
        runner.signal, "signal", lambda sig, fn: installed.append((sig, fn))
    )
    monkeypatch.setattr(
        runner.faulthandler, "dump_traceback", lambda **kw: calls.append(kw)
    )
    monkeypatch.setattr(
        runner.faulthandler,
        "register",
        lambda *a, **kw: pytest.fail("asynchronous dump handler is unsafe"),
    )
    fd = runner._install_stack_dump(tmp_path / "stacks.log")
    try:
        assert installed[0][0] == signal.SIGUSR1 and callable(installed[0][1])
        # Registration survives return through non-daemon joins and atexit.
        os.fstat(fd)
        for _ in range(3):
            installed[0][1](signal.SIGUSR1, None)
        assert calls == [{"file": fd, "all_threads": True}] * 3
        assert exit_callbacks == [
            (runner.signal.signal, signal.SIGUSR1, signal.SIG_IGN)
        ]
    finally:
        os.close(fd)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux diagnostic signal")
def test_stack_registration_failure_closes_its_descriptor(tmp_path, monkeypatch):
    from tools import ci_regression as runner

    opened, closed = [], []
    real_open, real_close = os.open, os.close

    def open_stack(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def close_stack(fd):
        closed.append(fd)
        real_close(fd)

    def fail(*args):
        raise OSError("synthetic signal registration failure")

    monkeypatch.setattr(runner.os, "open", open_stack)
    monkeypatch.setattr(runner.os, "close", close_stack)
    monkeypatch.setattr(runner.signal, "signal", fail)
    with pytest.raises(OSError, match="signal registration failure"):
        runner._install_stack_dump(tmp_path / "stacks.log")
    assert len(opened) == 1 and closed == opened
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux diagnostic signal")
def test_repeated_stack_requests_during_worker_frame_churn(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\nfaulthandler_timeout = 0.03\n")
    (tmp_path / "test_workers.py").write_text("""import threading, time
from pathlib import Path

def recurse(depth):
    one, two, three = 1, 2, 3
    return recurse(depth - 1) if depth else one + two + three

def busy_worker(stop):
    while not stop.is_set():
        recurse(123)

def test_active_workers(pytestconfig):
    assert float(pytestconfig.getini("faulthandler_timeout")) == 0
    stop = threading.Event()
    worker = threading.Thread(target=busy_worker, args=(stop,))
    worker.start()
    try:
        Path("ready").touch()
        deadline = time.monotonic() + 20
        while not Path("finish").exists():
            assert time.monotonic() < deadline
            time.sleep(.005)
    finally:
        stop.set()
        worker.join(3)
    assert not worker.is_alive()
""")
    output = tmp_path / "evidence"
    output.mkdir()
    env = dict(os.environ)
    env["PYTEST_ADDOPTS"] = "-o faulthandler_timeout=0.02"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    stack = output / "stacks.log"
    with (
        (output / "stdout.log").open("wb") as stdout,
        (output / "stderr.log").open("wb") as stderr,
    ):
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "tools.ci_regression",
                "_pytest",
                str(output),
                "-q",
                "test_workers.py",
                "-o",
                "faulthandler_timeout=0.01",
            ],
            cwd=tmp_path,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        try:
            deadline = time.monotonic() + 10
            while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
                assert child.poll() is None
                time.sleep(0.01)
            assert (tmp_path / "ready").exists()
            for _ in range(20):
                previous = stack.read_text().count("Current thread ")
                assert child.poll() is None
                os.kill(child.pid, signal.SIGUSR1)
                deadline = time.monotonic() + 2
                while (
                    stack.read_text().count("Current thread ") == previous
                    and time.monotonic() < deadline
                ):
                    assert child.poll() is None
                    time.sleep(0.005)
                assert stack.read_text().count("Current thread ") > previous
            # Stop requesting dumps before the child begins interpreter shutdown.
            (tmp_path / "finish").touch()
            assert child.wait(timeout=5) == 0
        finally:
            (tmp_path / "finish").touch()
            if child.poll() is None:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
    assert "1 passed" in (output / "stdout.log").read_text()
    assert (output / "stderr.log").read_bytes() == b""
    assert stack.read_text().count("Current thread ") >= 20
    assert "in recurse" in stack.read_text()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux diagnostic signal")
@pytest.mark.parametrize("terminator", [False, True])
def test_pytest_bootstrap_installs_handler_before_collection(
    tmp_path, monkeypatch, terminator
):
    from tools import ci_regression as runner

    calls = []
    monkeypatch.setattr(runner, "_install_stack_dump", lambda path: calls.append(path))

    def collect(args):
        assert calls == [tmp_path / "stacks.log"]
        expected = ["-q"]
        if not terminator:
            expected += ["test_example.py"]
        expected += ["-o", "faulthandler_timeout=3"]
        expected += ["-o", "faulthandler_timeout=0"]
        if terminator:
            expected += ["--", "test_example.py"]
        assert args == expected
        return 4

    monkeypatch.setattr(pytest, "main", collect)
    args = ["_pytest", str(tmp_path), "-q", "-o", "faulthandler_timeout=3"]
    args += (["--"] if terminator else []) + ["test_example.py"]
    # No terminator: injection belongs at the end of all caller arguments.
    if not terminator:
        args.remove("test_example.py")
        args.insert(3, "test_example.py")

    assert runner.main(args) == 4


@pytest.mark.skipif(sys.platform != "linux", reason="Linux diagnostic signal")
def test_stack_atexit_registration_failure_restores_handler(tmp_path, monkeypatch):
    from tools import ci_regression as runner

    installed, opened = [], []
    real_open = runner.os.open
    previous = object()

    def install(sig, handler):
        installed.append((sig, handler))
        return previous

    def open_stack(*args):
        fd = real_open(*args)
        opened.append(fd)
        return fd

    def fail(*args):
        raise RuntimeError("synthetic atexit registration failure")

    monkeypatch.setattr(runner.os, "open", open_stack)
    monkeypatch.setattr(runner.signal, "signal", install)
    monkeypatch.setattr(runner.atexit, "register", fail)
    with pytest.raises(RuntimeError, match="atexit registration failure"):
        runner._install_stack_dump(tmp_path / "stacks.log")
    assert callable(installed[0][1])
    assert installed[-1] == (signal.SIGUSR1, previous)
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux finalization signal")
def test_stack_signal_survives_final_gc_after_pytest_finishes(tmp_path):
    (tmp_path / "test_final_gc.py").write_text('''import atexit, gc, os, sys, time

def test_pass_then_finalize():
    ready = os.open('late-ready', os.O_WRONLY | os.O_CREAT, 0o600)
    done = os.open('late-complete', os.O_WRONLY | os.O_CREAT, 0o600)
    entered = [False]
    def early_cleanup():
        with open('atexit-ready', 'w') as marker:
            marker.write('ready')
        deadline = time.monotonic() + 5
        while not os.path.exists('atexit-release'):
            assert time.monotonic() < deadline
            time.sleep(.005)
    atexit.register(early_cleanup)
    def final_gc(phase, info, finalizing=sys.is_finalizing, write=os.write,
                 stat=os.stat, clock=time.monotonic, sleep=time.sleep):
        if phase != 'start' or not finalizing() or entered[0]:
            return
        entered[0] = True
        write(ready, b'after-signal-fini')
        deadline = clock() + 5
        while clock() < deadline:
            try:
                stat('release')
                write(done, b'finalizer-completed')
                return
            except FileNotFoundError:
                sleep(.005)
        write(done, b'barrier-timeout')
    gc.callbacks.append(final_gc)
''')
    output = tmp_path / "evidence"
    output.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with (
        (output / "stdout.log").open("wb") as stdout,
        (output / "stderr.log").open("wb") as stderr,
    ):
        child = subprocess.Popen(
            [sys.executable, "-u", "-m", "tools.ci_regression", "_pytest",
             str(output), "-q", "-p", "tools.ci_pytest", "--ci-shard=0",
             "--ci-shards=1", "--ci-evidence=" + str(output),
             "test_final_gc.py"],
            cwd=tmp_path, env=env, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        try:
            early = tmp_path / "atexit-ready"
            deadline = time.monotonic() + 10
            while not early.exists() and time.monotonic() < deadline:
                assert child.poll() is None
                time.sleep(.005)
            assert early.exists()
            # Later-registered Python cleanup still has the safe stack handler.
            os.kill(child.pid, signal.SIGUSR1)
            stack = output / "stacks.log"
            deadline = time.monotonic() + 3
            while "in early_cleanup" not in stack.read_text():
                assert time.monotonic() < deadline and child.poll() is None
                time.sleep(.005)
            (tmp_path / "atexit-release").touch()
            ready = tmp_path / "late-ready"
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if ready.exists() and ready.stat().st_size:
                    break
                assert child.poll() is None
                time.sleep(.005)
            assert ready.read_bytes() == b"after-signal-fini"
            rows = [
                json.loads(line)
                for line in (output / "progress.jsonl").read_text().splitlines()
            ]
            assert rows[-1]["event"] == "finished" and rows[-1]["exit_code"] == 0
            # This is later than both threading._shutdown and normal atexit.
            status = Path(f"/proc/{child.pid}/status").read_text()
            ignored = next(line.split()[1] for line in status.splitlines()
                           if line.startswith("SigIgn:"))
            assert int(ignored, 16) & (1 << (signal.SIGUSR1 - 1))
            os.kill(child.pid, signal.SIGUSR1)
            (tmp_path / "release").touch()
            assert child.wait(timeout=5) == 0
            assert (tmp_path / "late-complete").read_bytes() == b"finalizer-completed"
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
    assert "1 passed" in (output / "stdout.log").read_text()
    assert (output / "stderr.log").read_bytes() == b""


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process exit evidence")
def test_finished_pytest_does_not_waive_real_signal_exit(tmp_path):
    (tmp_path / "test_signal_exit.py").write_text('''import atexit, os, signal, time
from pathlib import Path
def test_pass_then_die():
    def ready_to_terminate():
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)
        Path('signal-ready.tmp').write_text(str(os.getpid()))
        os.replace('signal-ready.tmp', 'signal-ready')
        time.sleep(60)
    atexit.register(ready_to_terminate)
''')
    output = tmp_path / "evidence"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with (tmp_path / "runner.log").open("wb") as log:
        runner = subprocess.Popen(
            [sys.executable, str(ROOT / "tools/ci_regression.py"), "run",
             "--shard=0", "--shards=1", "--seconds=30", "--output",
             str(output), "test_signal_exit.py"],
            cwd=tmp_path, env=env, stdout=log, stderr=log,
            start_new_session=True,
        )
        child_pid, child_started = None, None
        try:
            # Bound startup separately from the post-pytest failure under test.
            # The runner's 30s backstop exceeds these 10s + 5s phase limits;
            # it is not a replacement for the existing three-second watchdog test.
            ready = tmp_path / "signal-ready"
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                assert runner.poll() is None
                time.sleep(.005)
            assert ready.exists(), "child did not reach post-pytest signal barrier"
            rows = [
                json.loads(line)
                for line in (output / "progress.jsonl").read_text().splitlines()
            ]
            assert rows[-1]["event"] == "finished" and rows[-1]["exit_code"] == 0
            child_pid = int(ready.read_text())
            child_stat = Path(f"/proc/{child_pid}/stat").read_text()
            child_started = child_stat.rsplit(") ", 1)[1].split()[19]
            os.kill(child_pid, signal.SIGUSR1)
            assert runner.wait(timeout=5) != 0
        finally:
            if runner.poll() is None:
                runner.terminate()
                try:
                    runner.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    # Do not abandon the readiness-blocked child if the wrapper
                    # itself cannot finish. Only signal its verified generation.
                    if child_pid is not None:
                        try:
                            stat = Path(f"/proc/{child_pid}/stat").read_text()
                            if stat.rsplit(") ", 1)[1].split()[19] == child_started:
                                os.killpg(child_pid, signal.SIGKILL)
                        except (ProcessLookupError, FileNotFoundError):
                            pass
                    try:
                        runner.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        runner.kill()
                        runner.wait(timeout=3)
    record = json.loads((output / "run.json").read_text())
    assert record["exit_code"] == -signal.SIGUSR1
    assert record["timed_out"] is False and record["cancelled_signal"] is None
    rows = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["event"] == "finished" and rows[-1]["exit_code"] == 0
    with pytest.raises(ValueError, match="shard did not finish successfully"):
        verify_coverage(tmp_path, 1, "success")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux diagnostic failure")
def test_stack_request_failure_still_cleans_and_records_failure(tmp_path, monkeypatch):
    from tools import ci_regression as runner

    cleaned = []

    class Child:
        pid = 12345
        returncode = None

        def poll(self):
            return None

    child = Child()

    def cleanup(process):
        cleaned.append(process)
        process.returncode = -signal.SIGTERM

    def fail(pid, sig):
        assert (pid, sig) == (child.pid, signal.SIGUSR1)
        raise OSError("synthetic diagnostic failure")

    ticks = iter([0, 961, 961])
    monkeypatch.setattr(runner, "_become_subreaper", lambda: None)
    monkeypatch.setattr(runner, "_snapshot", lambda stream: None)
    monkeypatch.setattr(runner, "_cleanup", cleanup)
    monkeypatch.setattr(runner.platform, "platform", lambda: "synthetic-platform")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runner.os, "kill", fail)
    output = tmp_path / "evidence"
    with pytest.raises(OSError, match="synthetic diagnostic failure"):
        runner.run_shard(output, 0, 1, 960)
    record = json.loads((output / "run.json").read_text())
    assert cleaned == [child]
    assert record["exit_code"] == -signal.SIGTERM
    assert record["timed_out"] is False and record["timeout_seconds"] == 960
