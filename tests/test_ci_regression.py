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
    result, output = invoke(tmp_path, 0, 1, ["test_shutdown.py"], seconds=3)
    assert result.returncode == 124
    progress = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text().splitlines()
    ]
    assert progress[-1]["event"] == "finished" and progress[-1]["exit_code"] == 0
    stacks = (output / "stacks.log").read_text()
    assert "wait_forever" in stacks and "_shutdown" in stacks
    with pytest.raises(ValueError):
        verify_coverage(tmp_path, 1, "success")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux stdout backpressure regression")
@pytest.mark.parametrize("finish", ["failure", "timeout"])
def test_stalled_console_cannot_block_capture_or_watchdog(tmp_path, finish):
    (tmp_path / "test_output.py").write_text('''import os, time
def test_large_output(capfd):
    with capfd.disabled():
        os.write(1, b"stdout-marker:" + b"X" * 400000)
        os.write(2, b"stderr-marker:" + b"Y" * 400000)
    ENDING
'''.replace("ENDING", "time.sleep(60)" if finish == "timeout" else "assert False, 'synthetic failure'"))
    output = tmp_path / "stalled-console"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    runner = subprocess.Popen(
        [sys.executable, str(ROOT / "tools/ci_regression.py"), "run",
         "--shard=0", "--shards=1", "--seconds=3", "--output", str(output), "test_output.py"],
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
