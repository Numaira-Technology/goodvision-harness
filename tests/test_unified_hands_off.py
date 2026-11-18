"""Synthetic offline execution and crash tests. No paid model or GPU calls."""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from helpers.bag_validation import bag_manifest
from helpers.qwen_bundle import frozen_host_bundle
from test_unified_dataset202 import inputs
from tools.qwen8b import eval_local
from tools.unified_benchmark import api_run, batch, dataset202, http_execution, preflight, workflow
from tools.unified_benchmark.contract import (EVENTS, CURRENT_VERSION, atomic_json, digest, file_hash,
                                             rules_for, select_thresholds, write_new)
from tools.unified_benchmark.retry import Ledger, Queue


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("external HTTP/model calls forbidden in offline tests")
    monkeypatch.setattr(api_run.urllib.request, "urlopen", forbidden)


def payload(config, yes=True):
    provider = ("self-hosted-r2-4090" if config.get("execution") == "self_hosted_r2"
                else config["provider_context"]["config"]["expected_provider_name"])
    result = {"model": config["model"], "provider": provider, "usage": {"cost": .001}}
    if config["kind"] == "probability":
        result["answers"] = {"answer": {"type": "noul", "noul": .8 if yes else .2}}
    else:
        result["choices"] = [{"finish_reason": "stop",
                              "message": {"content": json.dumps({"answer": yes})}}]
    return result


def frozen44(root, inputs, aliases):
    root.mkdir()
    (root / "inputs").mkdir()
    parts = dataset202.extend_test(*deepcopy(inputs))
    manifest = root.parent / "synthetic-bags.json"
    manifest.write_text(json.dumps(bag_manifest()))
    dataset202.add_bag_validation(root, parts, manifest, file_hash(manifest))
    models, inspected = {}, {}
    for split, part in parts.items():
        videos = {}
        for filename in {r["filename"] for r in part["rows"]}:
            frames = []
            for n in range(2):
                p = root / "frames" / split / (filename + f"-{n}.jpg")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(f"synthetic, never decode or send: {filename}/{n}".encode())
                frames.append({"path": p.relative_to(root).as_posix(), "sha256": file_hash(p), "time": n})
            videos[filename] = {"frames": frames}
        inspected[split] = videos
    for alias, cfg in preflight.model_configs(aliases).items():
        write_new(root / f"configs/{alias}.json", cfg)
        models[alias] = {}
        for split, part in parts.items():
            plan = preflight.build_plan(split, part["rows"], inspected[split], cfg, root,
                                        CURRENT_VERSION, part["dataset_context"])
            write_new(root / f"plans/{alias}-{split}.json", plan)
            models[alias][split] = {"plan_sha256": digest(plan)}
    write_new(root / "rules.json", rules_for(CURRENT_VERSION))
    sources = preflight.freeze_sources(root)
    write_new(root / "preflight.json", {"offline_passed": True, "models": models,
        "sources_sha256": digest(sources), "dataset_context": parts["test"]["dataset_context"]})
    write_new(root / "bundle-files-sha256.json", {p.relative_to(root).as_posix(): file_hash(p)
        for p in root.rglob("*") if p.is_file()})
    workflow.verify_bundle(root)
    return root


@pytest.fixture
def mandatory_ledger_locks(monkeypatch):
    """Exercise Windows' second-handle restriction on every test platform."""
    owners = set()
    original_init, original_close, original_open = Ledger.__init__, Ledger.close, Path.open

    def ledger_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.file is not None:
            owners.add(self.path.resolve())

    def ledger_close(self):
        original_close(self)
        owners.discard(self.path.resolve())

    def guarded_open(path, *args, **kwargs):
        if path.resolve() in owners:
            raise PermissionError("synthetic mandatory lock denies second handle")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Ledger, "__init__", ledger_init)
    monkeypatch.setattr(Ledger, "close", ledger_close)
    monkeypatch.setattr(Path, "open", guarded_open)
    yield
    assert not owners


def test_luna_validation44_selects_six_thresholds_and_binds_artifact(
        tmp_path, inputs, monkeypatch, mandatory_ledger_locks):
    bundle = frozen44(tmp_path / "bundle", inputs, ["luna"])
    called = []
    def fake(self, uri, body):
        called.append(body)
        return payload(self.config)
    monkeypatch.setattr(api_run.Transport, "call", fake)
    root = tmp_path / "result"
    root.mkdir()
    sha = workflow.verify_bundle(bundle)
    api_run.execute_model("luna", bundle, root, "synthetic", threading.Event(), sha, phase="validation")
    artifact_path = root / "luna/thresholds.json"
    artifact = workflow.load_threshold_artifact(artifact_path, file_hash(artifact_path),
        bundle=bundle, alias="luna", config=preflight.model_configs(["luna"])["luna"], bundle_digest=sha)
    assert artifact["calibration"] == "selected_on_complete_validation44"
    assert set(artifact["details"]) == set(EVENTS)
    assert artifact["validation_summary"]["complete_valid_questions"] == 44
    assert len(called) == 88
    before = len(called)
    api_run.execute_model("luna", bundle, root, "synthetic", threading.Event(), sha,
                          phase="validation", resume=True)
    assert len(called) == before
    bad = artifact | {"calibration": "selected_on_complete_validation24"}
    atomic_json(tmp_path / "old.json", bad)
    with pytest.raises(ValueError, match="validation-only"):
        workflow.load_threshold_artifact(tmp_path / "old.json", file_hash(tmp_path / "old.json"),
            bundle=bundle, alias="luna", config=preflight.model_configs(["luna"])["luna"], bundle_digest=sha)


@pytest.mark.parametrize("cut", ["outcome", "result", "summary_seal"])
def test_qwen_crash_resume_preserves_first_answer_without_resend(
        tmp_path, monkeypatch, cut, mandatory_ledger_locks):
    bundle = tmp_path / "bundle"
    cfg, unit = frozen_host_bundle(bundle)
    sha = workflow.verify_bundle(bundle)
    calls = []
    monkeypatch.setattr(eval_local.LocalTransport, "call",
                        lambda self, body: calls.append(body) or payload(cfg))
    original_write, original_append = http_execution.write_new, Ledger.append
    crashed = False
    def write(path, data):
        nonlocal crashed
        target = (cut == "outcome" and Path(path).parent.name == "outcomes"
                  or cut == "summary_seal" and Path(path).name == "completion.json")
        if target and not crashed:
            crashed = True
            raise SystemExit("simulated power loss")
        return original_write(path, data)
    def append(self, row):
        nonlocal crashed
        if cut == "result" and row["type"] == "result" and not crashed:
            crashed = True
            raise SystemExit("simulated power loss")
        return original_append(self, row)
    monkeypatch.setattr(http_execution, "write_new", write)
    monkeypatch.setattr(Ledger, "append", append)
    root = tmp_path / "result"
    with pytest.raises(SystemExit):
        eval_local.execute(cfg["alias"], bundle, root, "http://127.0.0.1:9", sha, "test")
    assert len(calls) == 1
    eval_local.execute(cfg["alias"], bundle, root, "http://127.0.0.1:9", sha, "test", resume=True)
    assert len(calls) == 1
    summary = json.loads((root / cfg["alias"] / "test/summary.json").read_text())
    assert summary["correct"] == summary["complete_valid_questions"] == 1
    if cut == "outcome":
        assert summary["test_attempt_latency"]["n"] == 0
        assert summary["test_attempt_latency"]["p95_seconds"] is None
    # A second resume verifies the sealed files and never sends again.
    eval_local.execute(cfg["alias"], bundle, root, "http://127.0.0.1:9", sha, "test", resume=True)
    assert len(calls) == 1


def api_batch_config(bundle, key):
    return {"schema": batch.SCHEMA, "repeats": 10, "max_process_restarts": 1, "max_controller_restarts": 1,
        "phase_timeout_seconds": 300, "stall_timeout_seconds": 120,
        "batch_timeout_seconds": 3600, "cleanup_timeout_seconds": 2, "budget_usd": 100,
        "models": [{"alias": "qwen8b", "route": "api", "bundle": str(bundle),
            "bundle_sha256": workflow.verify_bundle(bundle), "python": sys.executable,
            "env_file": str(key), "credential_file_sha256": file_hash(key),
            "attempt_cost_upper_usd": .001, "cost_upper_bound_basis": "synthetic tests only"}]}


@pytest.mark.skipif(os.name != "posix", reason="POSIX batch controller")
def test_ten_rounds_make_new_calls_and_completed_resume_makes_none(tmp_path, inputs, monkeypatch):
    bundle = frozen44(tmp_path / "bundle", inputs, ["qwen8b"])
    key = tmp_path / "synthetic.env"
    key.write_text("OPENROUTER_API_KEY=synthetic-not-a-key\n")
    config = api_batch_config(bundle, key)
    config_path = tmp_path / "batch.json"
    write_new(config_path, config)
    calls = Counter()
    def fake(self, uri, body):
        calls[str(self.output)] += 1
        return payload(self.config)
    monkeypatch.setattr(api_run.Transport, "call", fake)
    # Execute the real worker/runner synchronously; process supervision is tested separately.
    class Worker:
        def __init__(self, argv, **kwargs):
            args = argv[argv.index("worker") + 1:]
            root = Path(args[args.index("--output") + 1])
            task = args[args.index("--task-id") + 1]
            self.returncode = batch.worker(root, task, "synthetic offline test")
        def poll(self):
            return self.returncode
        def wait(self, **kwargs):
            return self.returncode
    monkeypatch.setattr(batch.subprocess, "Popen", Worker)
    root = tmp_path / "runs"
    assert batch.run(config_path, root, "synthetic offline test") == 0
    assert len(calls) == 11
    assert sorted(calls.values()) == [88] + [404] * 10
    summary = batch.read(root / "summary.json")["models"]["qwen8b"]
    assert summary["metrics"]["accuracy"]["n"] == 10
    assert summary["states"] == {"completed": 10}
    before = dict(calls)
    assert batch.run(config_path, root, "synthetic offline test", resume=True) == 0
    assert dict(calls) == before
    assert len(batch.read(root / "state.json")["matrix"]) == 11
    assert batch.main(["report", "--output", str(root)]) == 0
    previous_summary = (root / "summary.json").read_bytes()
    receipt = next((root / "qwen8b/round-01/qwen8b/test/receipts").glob("*.json"))
    receipt.unlink()
    with pytest.raises(FileNotFoundError):
        batch.main(["report", "--output", str(root)])
    assert (root / "summary.json").read_bytes() == previous_summary
    assert dict(calls) == before


def test_budget_and_changed_validation_rejected_before_any_worker(tmp_path, inputs):
    bundle = frozen44(tmp_path / "bundle", inputs, ["qwen8b"])
    key = tmp_path / "synthetic.env"
    key.write_text("OPENROUTER_API_KEY=synthetic\n")
    config = api_batch_config(bundle, key)
    with pytest.raises(ValueError, match="exceeds budget"):
        batch.validate_config(config | {"budget_usd": .001})
    with pytest.raises(ValueError, match="ten"):
        batch.validate_config(config | {"repeats": 11})
    with pytest.raises(ValueError, match="positive"):
        batch.validate_config(config | {"batch_timeout_seconds": 0})


def test_failed_receipts_resume_with_only_remaining_three_attempt_budget(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    cfg, _ = frozen_host_bundle(bundle)
    sha = workflow.verify_bundle(bundle)
    calls = []
    def malformed(self, body):
        calls.append(body)
        result = payload(cfg)
        result["choices"][0]["message"]["content"] = "malformed"
        return result
    monkeypatch.setattr(eval_local.LocalTransport, "call", malformed)
    monkeypatch.setattr(Queue, "wait", lambda self: None)
    original = Ledger.append
    crashed = False
    def append(self, row):
        nonlocal crashed
        if row["type"] == "result" and not crashed:
            crashed = True
            raise SystemExit("crash before ledger result")
        return original(self, row)
    monkeypatch.setattr(Ledger, "append", append)
    root = tmp_path / "runs"
    with pytest.raises(SystemExit):
        eval_local.execute(cfg["alias"], bundle, root, "http://127.0.0.1:9", sha, "test")
    eval_local.execute(cfg["alias"], bundle, root, "http://127.0.0.1:9", sha, "test", resume=True)
    assert len(calls) == 3
    result = batch.read(root / cfg["alias"] / "test/summary.json")
    assert result["accuracy"] == 0
    assert result["incomplete_questions"] == 1
    assert result["test_attempt_latency"]["p95_seconds"] is None
    eval_local.execute(cfg["alias"], bundle, root, "http://127.0.0.1:9", sha, "test", resume=True)
    assert len(calls) == 3


def test_resume_rejects_changed_matrix_and_outcome(tmp_path, inputs, monkeypatch):
    bundle = frozen44(tmp_path / "bundle", inputs, ["qwen8b"])
    key = tmp_path / "synthetic.env"
    key.write_text("OPENROUTER_API_KEY=synthetic\n")
    config = api_batch_config(bundle, key)
    checked = batch.validate_config(config)
    state = deepcopy(checked) | {"config_sha256": digest(config), "started_at_unix": 1}
    state["matrix"].pop()
    with pytest.raises(ValueError, match="matrix"):
        batch.validate_state(state, checked, config)
    root = tmp_path / "runs"
    monkeypatch.setattr(api_run.Transport, "call", lambda self, *_: payload(self.config))
    api_run.execute_model("qwen8b", bundle, root, "synthetic", threading.Event(),
                          workflow.verify_bundle(bundle), phase="validation")
    path = next((root / "qwen8b/validation/outcomes").glob("*.json"))
    saved = batch.read(path)
    saved["answers"] = {}
    atomic_json(path, saved)
    with pytest.raises(ValueError, match="changed"):
        api_run.execute_model("qwen8b", bundle, root, "synthetic", threading.Event(),
                              workflow.verify_bundle(bundle), phase="validation", resume=True)


def test_report_keeps_failed_missing_and_not_started_rounds(tmp_path):
    write_new(tmp_path / "plans/synthetic-test.json", {"questions": [None] * 202})
    item = {"alias": "synthetic", "route": "api", "bundle": str(tmp_path)}
    config = {"models": [item]}
    rows = [{"alias": "synthetic", "split": "test", "round": i, "output": f"r{i}",
             "state": "not_started" if i == 10 else "completed_incomplete" if i == 1 else "completed",
             "invocations": 0 if i == 10 else 1} for i in range(1, 11)]
    for row in rows[:-1]:
        write_new(tmp_path / row["output"] / "synthetic/test/summary.json",
            {"accuracy": 0 if row["round"] == 1 else 1, "complete_valid_questions": 0,
             "final_missing_unique_units": 1, "test_attempt_latency": {"n": 0,
             "p50_seconds": None, "p95_seconds": None}})
    report = batch.summarize(tmp_path, config, {"batch_id": "synthetic", "matrix": rows, "reserved_upper_usd": 0})
    result = report["models"]["synthetic"]
    assert len(result["rounds"]) == 10
    assert result["metrics"]["accuracy"]["mean"] == pytest.approx(8 / 9)
    assert result["metrics"]["accuracy"]["n"] == 9
    assert result["metrics"]["p95_seconds"]["mean"] is None
    assert result["rounds"][-1]["accuracy"] is None


@pytest.mark.parametrize("entrypoint", ["run", "watch"])
def test_non_posix_controller_rejected_before_creating_output(tmp_path, monkeypatch, entrypoint):
    from types import SimpleNamespace
    monkeypatch.setattr(batch, "os", SimpleNamespace(name="nt"))
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="requires POSIX"):
        getattr(batch, entrypoint)(tmp_path / "absent-config.json", output, "synthetic test")
    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor")
def test_supervisor_stops_worker_when_controller_pipe_closes(tmp_path):
    read_fd, write_fd = os.pipe()
    with (tmp_path / "lock").open("a+") as lock:
        command = [sys.executable, "-m", "tools.unified_benchmark.supervise",
                   "--lifetime-fd", str(read_fd), "--lock-fd", str(lock.fileno()),
                   "--timeout", "5", "--cleanup-timeout", "1",
                   sys.executable, "-c", "import time; time.sleep(60)"]
        child = subprocess.Popen(command, pass_fds=(read_fd, lock.fileno()),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.close(read_fd)
        os.close(write_fd)
        stdout, stderr = child.communicate(timeout=8)
    assert child.returncode == 124
    assert b"controller disconnected" in stdout


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor")
@pytest.mark.parametrize("exit_codes, expected, launches", [
    ([-9, 0], 0, 2),
    ([1], 1, 1),
    ([-9, -9], 1, 2),
])
def test_watch_recovers_crash_but_preserves_normal_failure_and_restart_limit(
        tmp_path, inputs, monkeypatch, exit_codes, expected, launches):
    bundle = frozen44(tmp_path / "bundle", inputs, ["qwen8b"])
    key = tmp_path / "synthetic.env"
    key.write_text("OPENROUTER_API_KEY=synthetic\n")
    config_path = tmp_path / "batch.json"
    config = api_batch_config(bundle, key)
    write_new(config_path, config)
    root = tmp_path / "runs"
    calls = []
    class Controller:
        pid = 12345
        def __init__(self, command, **kwargs):
            calls.append(command)
            root.mkdir(exist_ok=True)
        def wait(self, **kwargs):
            return exit_codes[len(calls) - 1]
    monkeypatch.setattr(batch.subprocess, "Popen", Controller)
    assert batch.watch(config_path, root, "synthetic test") == expected
    assert len(calls) == launches
    assert "--resume" not in calls[0]
    if launches > 1:
        assert "--resume" in calls[1]
    state = batch.read(tmp_path / "runs-supervisor.json")
    assert state["invocations"] == launches
    assert state["state"] == ("completed" if expected == 0 else "failed")
    if exit_codes == [-9, -9]:
        # Restarting the watcher cannot reset its persisted restart allowance.
        assert batch.watch(config_path, root, "synthetic test", resume=True) == 1
        assert len(calls) == launches
