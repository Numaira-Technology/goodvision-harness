"""Review regressions: synthetic evidence only; all network access is forbidden."""
import json
import threading
from pathlib import Path

import pytest
pytest.importorskip("fcntl", reason="Durable benchmark execution requires Linux/macOS")

from tools.unified_benchmark import api_run, preflight, workflow
from tools.unified_benchmark.contract import EVENTS, RULES, digest, file_hash, score, select_thresholds, write_new
from tools.unified_benchmark.retry import Ledger


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network/model calls are forbidden in offline regressions")
    monkeypatch.setattr(api_run.urllib.request, "urlopen", forbidden)


def seal(bundle):
    path = bundle / "bundle-files-sha256.json"
    path.write_text(json.dumps({str(p.relative_to(bundle)): file_hash(p)
                               for p in bundle.rglob("*") if p.is_file() and p != path}))


@pytest.fixture
def frozen(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = {"alias": "luna", "model": "synthetic/luna", "kind": "probability", "backend": "decision",
              "provider_context": {"config": {"expected_provider_name": "Synthetic"}}}
    write_new(bundle / "configs/luna.json", config)
    choice = config | {"alias": "choice", "kind": "choice", "backend": "language", "model": "synthetic/choice"}
    write_new(bundle / "configs/choice.json", choice)
    observed = {}
    for split, counts in (("validation", (6, 6, 6, 6, 0, 0)), ("test", (32, 22, 26, 20, 10, 10))):
        units, questions = [], []
        for event, count in zip(EVENTS, counts):
            for i in range(count):
                uid = f"{split}/{event}/{i}"
                units.append({"id": uid, "targets": [event], "request_sha256": digest(uid)})
                questions.append({"number": len(questions) + 1, "event": event,
                                  "reference": "yes" if i < count // 2 else "no", "unit_ids": [uid]})
                if split == "validation":
                    observed[uid] = {event: {"score": .8 if i < 3 else .2}}
        plan = {"split": split, "config_sha256": digest(config), "rules_sha256": digest(RULES),
                "units": units, "questions": questions}
        write_new(bundle / f"plans/luna-{split}.json", plan)
        write_new(bundle / f"plans/choice-{split}.json", plan | {"config_sha256": digest(choice)})
    write_new(bundle / "rules.json", RULES)
    sources = preflight.freeze_sources(bundle)
    models = {}
    for path in (bundle / "plans").glob("*.json"):
        alias, split = path.stem.rsplit("-", 1)
        models.setdefault(alias, {})[split] = {"plan_sha256": digest(json.loads(path.read_text()))}
    write_new(bundle / "preflight.json", {"offline_passed": True, "sources_sha256": digest(sources), "models": models})
    seal(bundle)
    bundle_digest = workflow.verify_bundle(bundle)
    val = json.loads((bundle / "plans/luna-validation.json").read_text())
    artifact = select_thresholds(val, observed, "probability", config["model"])
    artifact.update(calibration="selected_on_complete_validation24", config_sha256=digest(config),
                    bundle_sha256=bundle_digest, rules_sha256=digest(RULES), model=config["model"])
    return bundle, bundle_digest, config, artifact, observed


def save_artifact(tmp_path, artifact):
    path = tmp_path / "thresholds.json"
    # Allows a nonfinite fixture so that the application, not this helper, rejects it.
    path.write_text(json.dumps(artifact))
    return path, file_hash(path)


@pytest.mark.parametrize("mutation", ["failed", "null", "empty", "missing_event", "nan", "range", "bool",
    "no_summary", "incomplete", "no_validation_only", "wrong_plan", "no_evidence", "wrong_config", "bag", "comparator"])
def test_bad_calibration_blocks_all_selected_models_before_credentials(frozen, tmp_path, monkeypatch, mutation):
    bundle, _, _, artifact, _ = frozen
    if mutation == "failed":
        artifact.update(thresholds=None, calibration="not_calibrated; test blocked")
    elif mutation in ("null", "empty"):
        artifact["thresholds"] = None if mutation == "null" else {}
    elif mutation == "missing_event": artifact["thresholds"].pop("fight")
    elif mutation in ("nan", "range", "bool"):
        artifact["thresholds"]["fight"] = {"nan": float("nan"), "range": 1.1, "bool": True}[mutation]
    elif mutation == "no_summary": artifact.pop("validation_summary")
    elif mutation == "incomplete": artifact["validation_summary"]["complete_valid_questions"] = 23
    elif mutation == "no_validation_only": artifact.pop("validation_only")
    elif mutation == "wrong_plan": artifact["validation_plan_sha256"] = "0" * 64
    elif mutation == "no_evidence": artifact.pop("validation_evidence_sha256")
    elif mutation == "wrong_config": artifact["config_sha256"] = "0" * 64
    elif mutation == "bag": artifact["thresholds"]["bag_knife"] = .1
    else: artifact["comparator"] = ">"
    path, sha = save_artifact(tmp_path, artifact)
    def forbidden(*args, **kwargs): pytest.fail("invalid calibration must block before credentials or transport")
    monkeypatch.setattr(api_run, "load_explicit_key", forbidden)
    monkeypatch.setattr(api_run, "Transport", forbidden)
    with pytest.raises(ValueError):
        api_run.main(["--bundle", str(bundle), "--output", str(tmp_path / "out"),
            "--models", "choice", "luna", "--phase", "test", "--env-file", "absent-private.env",
            "--thresholds", str(path), "--thresholds-sha256", sha,
            "--authorization-note", "synthetic offline test", "--execute"])
    assert not (tmp_path / "out").exists()


def test_failed_calibration_cannot_be_used_for_offline_report(frozen, tmp_path):
    bundle, _, _, artifact, _ = frozen
    artifact.update(thresholds=None, calibration="not_calibrated; test blocked")
    path, sha = save_artifact(tmp_path, artifact)
    with pytest.raises(ValueError, match="test blocked"):
        workflow.main(["report", "--bundle", str(bundle), "--model", "luna", "--ledger", "absent.jsonl",
            "--thresholds", str(path), "--thresholds-sha256", sha, "--output", str(tmp_path / "report.json")])
    assert not (tmp_path / "report.json").exists()


def test_successful_selection_flows_through_test_and_offline_report(frozen, tmp_path, monkeypatch):
    bundle, bundle_digest, config, artifact, observed = frozen
    # Use the public offline selector, not a hand-written success marker.
    val = json.loads((bundle / "plans/luna-validation.json").read_text())
    val_ledger = tmp_path / "validation.jsonl"
    ledger = Ledger(val_ledger, workflow.context_for(val, config, bundle_digest))
    try:
        for u in val["units"]:
            base = {"unit": u["id"], "request_sha256": u["request_sha256"], "attempt": 1}
            ledger.append(base | {"type": "intent"})
            ledger.append(base | {"type": "result", "answers": observed[u["id"]], "complete": True})
    finally: ledger.close()
    path = tmp_path / "selected.json"
    workflow.main(["select", "--bundle", str(bundle), "--model", "luna", "--ledger", str(val_ledger), "--output", str(path)])
    selected = json.loads(path.read_text())
    assert selected["thresholds"] == artifact["thresholds"]
    calls = []
    class SyntheticTransport:
        def __init__(self, *args): pass
        def __call__(self, unit):
            calls.append(unit["id"])
            return {unit["targets"][0]: {"score": .6}}, {"usage": {"cost": 0}, "actual_provider": "Synthetic"}
    monkeypatch.setattr(api_run, "Transport", SyntheticTransport)
    monkeypatch.setattr(api_run, "load_explicit_key", lambda _: "synthetic-not-a-key")
    output = tmp_path / "run"
    assert api_run.main(["--bundle", str(bundle), "--models", "luna", "--phase", "test",
        "--thresholds", str(path), "--thresholds-sha256", file_hash(path), "--output", str(output),
        "--env-file", "synthetic.env", "--authorization-note", "synthetic offline test", "--execute"]) == 0
    result = json.loads((output / "luna/test/summary.json").read_text())
    assert len(calls) == 120 and result["accuracy_denominator"] == 120
    assert result["thresholds"] == selected["thresholds"] and result["final_missing_unique_units"] == 0
    workflow.main(["report", "--bundle", str(bundle), "--model", "luna",
        "--ledger", str(output / "luna/test/attempts.jsonl"), "--thresholds", str(path),
        "--thresholds-sha256", file_hash(path), "--output", str(tmp_path / "report.json")])
    assert json.loads((tmp_path / "report.json").read_text())["correct"] == result["correct"]


def change_frozen_code(bundle, remove=False):
    name = "tools/unified_benchmark/contract.py"
    sources_path = bundle / "source-hashes.json"
    sources = json.loads(sources_path.read_text())
    snapshot = bundle / "source-snapshot" / name
    if remove:
        sources.pop(name)
        snapshot.unlink()
    else:
        snapshot.write_text(snapshot.read_text().replace('return "yes" if "yes" in answers', 'return "no" if "yes" in answers'))
        sources[name] = file_hash(snapshot)
    sources_path.write_text(json.dumps(sources))
    report = bundle / "preflight.json"
    report.write_text(json.dumps(json.loads(report.read_text()) | {"sources_sha256": digest(sources)}))
    seal(bundle)  # old snapshot is internally consistent, but not the active code


@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("entrypoint", ["api", "select", "report"])
def test_changed_or_omitted_source_blocks_every_entrypoint(frozen, tmp_path, monkeypatch, remove, entrypoint):
    bundle, _, _, _, _ = frozen
    change_frozen_code(bundle, remove)
    def forbidden(*args, **kwargs): pytest.fail("code mismatch must fail before reading credentials or ledgers")
    monkeypatch.setattr(api_run, "load_explicit_key", forbidden)
    monkeypatch.setattr("tools.unified_benchmark.retry.Ledger", forbidden)
    with pytest.raises(ValueError, match="source"):
        if entrypoint == "api":
            api_run.main(["--bundle", str(bundle), "--models", "luna", "--phase", "validation",
                "--env-file", "absent.env", "--output", str(tmp_path / "out"),
                "--authorization-note", "synthetic offline test", "--execute"])
        else:
            workflow.main([entrypoint, "--bundle", str(bundle), "--model", "luna", "--ledger", "absent.jsonl",
                           "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_empty_threshold_mapping_is_not_a_fixed_baseline(frozen):
    bundle, _, _, _, evidence = frozen
    plan = json.loads((bundle / "plans/luna-validation.json").read_text())
    with pytest.raises(ValueError, match="six finite"):
        score(plan, evidence, "probability", {})


@pytest.mark.parametrize("entrypoint", ["api", "select", "report"])
@pytest.mark.parametrize("mutation", ["plan", "config", "preflight_digest", "unrecorded_model"])
def test_unsealed_or_changed_plan_blocks_before_any_work(frozen, tmp_path, monkeypatch, mutation, entrypoint):
    bundle, _, _, _, _ = frozen
    manifest = bundle / "bundle-files-sha256.json"
    hashes = json.loads(manifest.read_text())
    if mutation in ("plan", "config"):
        hashes.pop("plans/luna-test.json" if mutation == "plan" else "configs/luna.json")
        manifest.write_text(json.dumps(hashes))
    elif mutation == "preflight_digest":
        path = bundle / "plans/luna-test.json"
        plan = json.loads(path.read_text())
        plan["questions"][0]["reference"] = "no"
        path.write_text(json.dumps(plan))
        seal(bundle)  # Even an updated inventory cannot bypass the preflight plan digest.
    else:
        path = bundle / "preflight.json"
        report = json.loads(path.read_text())
        report["models"].pop("luna")
        path.write_text(json.dumps(report))
        seal(bundle)
    def forbidden(*args, **kwargs):
        pytest.fail("unsealed plans must fail before credentials or ledger access")
    monkeypatch.setattr(api_run, "load_explicit_key", forbidden)
    monkeypatch.setattr("tools.unified_benchmark.retry.Ledger", forbidden)
    with pytest.raises(ValueError, match="unsealed|plan identity|unrecorded"):
        if entrypoint == "api":
            api_run.main(["--bundle", str(bundle), "--models", "luna", "--phase", "validation",
                "--env-file", "absent.env", "--output", str(tmp_path / "out"),
                "--authorization-note", "synthetic", "--execute"])
        else:
            workflow.main([entrypoint, "--bundle", str(bundle), "--model", "luna", "--ledger", "absent.jsonl",
                "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
