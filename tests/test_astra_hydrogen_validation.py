"""Offline scope, routing and recovery checks for the effort comparison."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from helpers.bag_validation import bag_manifest
from test_unified_dataset202 import inputs
from tools.event_benchmark.prepare_provider import select_provider
from tools.unified_benchmark import adapters, api_run, batch, dataset202, preflight, workflow
from tools.unified_benchmark.contract import digest, file_hash, write_new
from tools.unified_benchmark.budget import BudgetGuard
from tools.unified_benchmark.retry import PausedRun

ALIASES = ["astra_low", "astra_medium", "hydrogen_low", "hydrogen_medium"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("no live requests in offline tests")
    monkeypatch.setattr(api_run.urllib.request, "urlopen", forbidden)


@pytest.fixture
def validation_bundle(tmp_path, inputs, monkeypatch):
    def load(out, *_args):
        (out / "inputs").mkdir()
        return dataset202.extend_test(*deepcopy(inputs))

    def inspect(split, row, out, media, protocol):
        assert split == "validation", "test media must never be accessed"
        frames = []
        for n in range(2):
            p = out / "frames" / split / (row["filename"] + f"-{n}.jpg")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"synthetic offline image")
            frames.append({"path": p.relative_to(out).as_posix(), "time": n,
                           "sha256": file_hash(p), "size": [720, 480]})
        return {"frames": frames, "full_decode_passed": True}

    monkeypatch.setattr(preflight, "load_inputs", load)
    monkeypatch.setattr(preflight, "inspect_video", inspect)
    original_check = preflight.subprocess.check_output
    monkeypatch.setattr(preflight.subprocess, "check_output", lambda argv, **kw:
                        "synthetic ffmpeg\n" if argv == ["ffmpeg", "-version"] else original_check(argv, **kw))
    manifest = tmp_path / "synthetic-bag.json"
    manifest.write_text(json.dumps(bag_manifest()))
    out = tmp_path / "bundle"
    assert preflight.main([
        "--inputs", str(tmp_path), "--media", str(tmp_path), "--output", str(out),
        "--phase", "validation", "--require-validation44", "--models", *ALIASES,
        "--bag-validation", str(manifest), "--bag-validation-sha256", file_hash(manifest)]) == 0
    return out


def test_effort_requests_omit_unsupported_temperature_and_change_only_effort(validation_bundle):
    configs = preflight.model_configs(ALIASES)
    bodies = {}
    for alias, cfg in configs.items():
        plan = batch.read(validation_bundle / f"plans/{alias}-validation.json")
        _, body = adapters.request_body(cfg, plan["units"][0], validation_bundle)
        assert body["reasoning"] == {"effort": alias.rsplit("_", 1)[1]}
        assert body["provider"]["allow_fallbacks"] is False
        if alias.startswith("astra"):
            assert "temperature" not in body
            assert body["provider"]["only"] == ["openai"]
        else:
            assert body["temperature"] == 1
            assert body["provider"]["only"] == ["google-ai-studio"]
        bodies[alias] = body
    for model in ("astra", "hydrogen"):
        low, medium = bodies[model + "_low"], bodies[model + "_medium"]
        assert digest(low) != digest(medium)
        assert {k: v for k, v in low.items() if k != "reasoning"} == {
            k: v for k, v in medium.items() if k != "reasoning"}


def test_explicit_provider_must_support_every_sent_parameter():
    catalog = {"data": {"id": "openai/gpt-6-astra", "endpoints": [{
        "tag": "openai", "provider_name": "OpenAI", "status": 0,
        "supported_parameters": ["max_tokens", "reasoning"]}]}}
    with pytest.raises(ValueError, match="no active endpoint"):
        select_provider(catalog, "openai/gpt-6-astra", "language", "a" * 64, "synthetic", "openai")
    selected = select_provider(catalog, "openai/gpt-6-astra", "language", "a" * 64,
                               "synthetic", "openai", required_parameters=["max_tokens", "reasoning"])
    assert selected["provider"]["require_parameters"] is True
    with pytest.raises(ValueError, match="no active endpoint"):
        select_provider(catalog, "openai/gpt-6-astra", "language", "a" * 64, "synthetic", "openai",
                        required_parameters=["max_tokens", "reasoning", "temperature"])


def test_validation_only_bundle_has_no_test_plans_or_test_media(validation_bundle):
    workflow.verify_bundle(validation_bundle)
    meta = batch.read(validation_bundle / "preflight.json")
    assert meta["prepared_splits"] == ["validation"]
    assert meta["unique_videos"] == 34
    assert all(set(phases) == {"validation"} for phases in meta["models"].values())
    assert not list((validation_bundle / "plans").glob("*-test.json"))
    assert not (validation_bundle / "frames/test").exists()


@pytest.mark.parametrize("phase", ["test", "both"])
def test_unprepared_test_rejected_before_any_request(validation_bundle, tmp_path, phase):
    out = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="not prepared"):
        api_run.execute_model("astra_low", validation_bundle, out, "synthetic", threading.Event(),
                              workflow.verify_bundle(validation_bundle), phase=phase)
    assert not out.exists()


def test_four_validation_runs_are_independent_and_resume_does_not_resend(validation_bundle, tmp_path, monkeypatch):
    key = tmp_path / "synthetic.env"
    key.write_text("OPENROUTER_API_KEY=synthetic-not-a-key\n")
    config = {"schema": batch.VALIDATION_SCHEMA, "repeats": 0,
        "max_process_restarts": 1, "max_controller_restarts": 1,
        "phase_timeout_seconds": 300, "stall_timeout_seconds": 120,
        "batch_timeout_seconds": 1800, "cleanup_timeout_seconds": 2, "budget_usd": 2,
        "models": [{"alias": alias, "route": "api", "bundle": str(validation_bundle),
            "bundle_sha256": workflow.verify_bundle(validation_bundle), "python": sys.executable,
            "env_file": str(key), "credential_file_sha256": file_hash(key),
            "attempt_cost_upper_usd": .001, "cost_upper_bound_basis": "synthetic offline bounds"}
            for alias in ALIASES]}
    checked = batch.validate_config(config)
    assert len(checked["matrix"]) == 4
    assert {r["split"] for r in checked["matrix"]} == {"validation"}
    assert checked["reserved_upper_usd"] == pytest.approx(4 * 88 * 3 * .001)
    with pytest.raises(ValueError, match="zero planned test"):
        batch.validate_config(config | {"repeats": 1})
    with pytest.raises(ValueError, match="exceeds budget"):
        batch.validate_config(config | {"budget_usd": .01})
    config.update(budget_mode="rolling_reservation", budget_usd=.5, initial_spent_usd=.01)
    assert batch.validate_config(config)["reserved_upper_usd"] == .5
    calls = []
    def fake(self, uri, body):
        calls.append((self.config["alias"], self.output, body["reasoning"]["effort"]))
        return {"model": self.config["model"],
            "provider": self.config["provider_context"]["config"]["expected_provider_name"],
            "usage": {"cost": .001},
            "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":true}'}}]}
    monkeypatch.setattr(api_run.Transport, "call", fake)
    class Worker:
        def __init__(self, argv, **kwargs):
            root = Path(argv[argv.index("--output") + 1])
            task = argv[argv.index("--task-id") + 1]
            self.returncode = batch.worker(root, task, "synthetic validation-only run")
        def poll(self):
            return self.returncode
        def wait(self, **kwargs):
            return self.returncode
    monkeypatch.setattr(batch.subprocess, "Popen", Worker)
    path = tmp_path / "config.json"
    write_new(path, config)
    out = tmp_path / "run"
    assert batch.run(path, out, "synthetic validation-only run") == 0
    assert len(calls) == 4 * 88
    assert len({str(c[1]) for c in calls}) == 4
    summary = batch.read(out / "summary.json")
    assert summary["validation_only"] and summary["test_questions"] == summary["planned_repeats"] == 0
    assert all(m["rounds"][0]["complete_valid_questions"] == 44 for m in summary["models"].values())
    assert (out / "validation.csv").exists() and not (out / "rounds.csv").exists()
    assert batch.run(path, out, "synthetic validation-only run", resume=True) == 0
    assert len(calls) == 4 * 88


def test_budget_reserves_concurrent_requests_and_survives_restart(tmp_path):
    cfg = {"budget_usd": 1, "initial_spent_usd": .1}
    guard = BudgetGuard(tmp_path, cfg, "astra/validation", tmp_path / "phase", .3)
    def intent(n):
        return {"unit": str(n), "attempt": 1, "request_sha256": "a" * 64}
    def reserve(n):
        try:
            guard(intent(n))
            return True
        except PausedRun:
            return False
    with ThreadPoolExecutor(max_workers=12) as pool:
        accepted = list(pool.map(reserve, range(12)))
    assert sum(accepted) == 3
    state = batch.read(tmp_path / "api-budget.json")
    assert state["charged_or_reserved_usd"] <= 1
    assert len(state["reservations"]) == 3
    resumed = BudgetGuard(tmp_path, cfg, "astra/validation", tmp_path / "phase", .3)
    with pytest.raises(PausedRun, match="budget cap"):
        resumed(intent(12))
    with pytest.raises(ValueError, match="configuration differs"):
        BudgetGuard(tmp_path, {"budget_usd": 20}, "astra/validation", tmp_path / "phase", .3)


def test_budget_releases_only_verified_finite_cost_and_keeps_orphan_reservation(tmp_path):
    cfg = {"budget_usd": 1}
    phase = tmp_path / "phase"
    guard = BudgetGuard(tmp_path, cfg, "astra/validation", phase, .4)
    intent = {"unit": "one", "attempt": 1, "request_sha256": "b" * 64}
    guard(intent)
    # A crash before the intent was appended reuses that same reservation.
    guard = BudgetGuard(tmp_path, cfg, "astra/validation", phase, .4)
    guard(intent)
    assert len(guard.state["reservations"]) == 1
    guard(intent | {"unit": "two"})
    assert guard.state["charged_or_reserved_usd"] == .8
    outcome = phase / "outcomes" / (digest("one") + "-attempt-1.json")
    write_new(outcome, intent | {"metadata": {"usage": {"cost": .05}}})
    guard(intent | {"unit": "three"})
    assert guard.state["charged_or_reserved_usd"] == pytest.approx(.85)
    write_new(phase / "outcomes" / (digest("two") + "-attempt-1.json"),
              (intent | {"unit": "two", "metadata": {"usage": {}}}))
    with pytest.raises(PausedRun, match="budget cap"):
        guard(intent | {"unit": "four"})


def test_budget_stops_on_provider_charge_above_bound(tmp_path):
    cfg = {"budget_usd": 5}
    phase = tmp_path / "phase"
    guard = BudgetGuard(tmp_path, cfg, "astra/validation", phase, .4)
    intent = {"unit": "one", "attempt": 1, "request_sha256": "b" * 64}
    guard(intent)
    write_new(phase / "outcomes" / (digest("one") + "-attempt-1.json"),
              intent | {"metadata": {"usage": {"cost": .5}}})
    with pytest.raises(PausedRun, match="exceeded"):
        guard(intent | {"unit": "two"})


def test_stop_file_prevents_new_requests_without_editing_evidence(tmp_path):
    guard = BudgetGuard(tmp_path, {"budget_usd": 5}, "astra/validation", tmp_path / "phase", .4)
    intent = {"unit": "one", "attempt": 1, "request_sha256": "b" * 64}
    (tmp_path / "STOP").write_text("operator requested stop\n")
    before = file_hash(tmp_path / "api-budget.json")
    with pytest.raises(PausedRun, match="operator stop"):
        guard(intent)
    assert file_hash(tmp_path / "api-budget.json") == before


def test_required_validation44_never_falls_back_to_24(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(preflight, "load_inputs", lambda *args: dataset202.extend_test(*deepcopy(inputs)))
    def forbidden(*args):
        pytest.fail("media processing must not begin without the required supplement")
    monkeypatch.setattr(preflight, "inspect_video", forbidden)
    out = tmp_path / "missing-supplement"
    assert preflight.main(["--inputs", str(tmp_path), "--media", str(tmp_path), "--output", str(out),
                           "--phase", "validation", "--require-validation44", "--models", *ALIASES]) == 1
    assert "bag-validation10-v1" in batch.read(out / "preflight.json")["error"]
