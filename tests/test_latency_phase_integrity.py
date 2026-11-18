"""Offline replay export must preserve the identity of its recorded evidence."""
import json

import pytest

from helpers.qwen_bundle import frozen_host_bundle
from tools.latency_investigation.export_replay import records_from_phase
from tools.qwen8b import eval_local
from tools.unified_benchmark import workflow
from tools.unified_benchmark.contract import digest, file_hash


@pytest.fixture
def recorded_phase(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    config, unit = frozen_host_bundle(bundle)

    def reply(self, body):
        return {"model": config["model"], "usage": {
            "prompt_tokens": 800, "completion_tokens": 4},
            "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":true}'}}]}

    monkeypatch.setattr(eval_local.LocalTransport, "call", reply)
    sha = workflow.verify_bundle(bundle)
    eval_local.execute(config["alias"], bundle, tmp_path / "runs",
                       "synthetic-no-network", sha, "test")
    return bundle, tmp_path / "runs" / config["alias"] / "test", config, unit


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_export_phase_reuses_recorded_body_and_usage_without_mutation(recorded_phase):
    bundle, phase, config, unit = recorded_phase
    before = snapshot(bundle.parent)
    records = records_from_phase(bundle, phase, config["alias"])
    assert len(records) == 1
    assert digest(records[0]["body"]) == unit["request_sha256"]
    assert records[0]["prompt_tokens"] == 800
    assert records[0]["arrival_offset_seconds"] == 0
    assert snapshot(bundle.parent) == before


@pytest.mark.parametrize("change", [
    "config", "receipt", "ledger_context", "ledger_request", "run_identity",
    "receipt_anchor", "historical_source",
])
def test_export_phase_rejects_changed_evidence_without_resealing(recorded_phase, change):
    bundle, phase, config, unit = recorded_phase
    if change == "config":
        path = bundle / "configs" / (config["alias"] + ".json")
        path.write_text(json.dumps(config | {"max_tokens": 999}))
    elif change == "receipt":
        path = next((phase / "receipts").glob("*.json"))
        receipt = json.loads(path.read_text())
        receipt["response"]["usage"]["prompt_tokens"] = 100
        path.write_text(json.dumps(receipt))
    elif change.startswith("ledger_") or change == "receipt_anchor":
        path = phase / "attempts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "ledger_context":
            rows[0]["context"]["bundle_sha256"] = "changed"
        elif change == "ledger_request":
            rows[-1]["request_sha256"] = "changed"
        else:
            rows[-1]["metadata"]["receipt_sha256"] = "changed"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif change == "run_identity":
        path = phase.parent / "run.json"
        path.write_text(json.dumps(json.loads(path.read_text()) | {"alias": "different"}))
    else:
        # A self-consistent older source snapshot still cannot be replayed by
        # current adapters. Never repair/reseal the actual historical bundle.
        name = "tools/unified_benchmark/adapters.py"
        path = bundle / "source-snapshot" / name
        path.write_text(path.read_text() + "\n# synthetic historical source\n")
        sources_path = bundle / "source-hashes.json"
        sources = json.loads(sources_path.read_text())
        sources[name] = file_hash(path)
        sources_path.write_text(json.dumps(sources))
        report_path = bundle / "preflight.json"
        report = json.loads(report_path.read_text())
        report["sources_sha256"] = digest(sources)
        report_path.write_text(json.dumps(report))
        hashes_path = bundle / "bundle-files-sha256.json"
        hashes = json.loads(hashes_path.read_text())
        for changed in (path, sources_path, report_path):
            hashes[changed.relative_to(bundle).as_posix()] = file_hash(changed)
        hashes_path.write_text(json.dumps(hashes))
    before = snapshot(bundle.parent)
    with pytest.raises(ValueError):
        records_from_phase(bundle, phase, config["alias"])
    assert snapshot(bundle.parent) == before


def test_export_phase_checks_receipt_hash_without_completion_seal(recorded_phase):
    bundle, phase, config, _ = recorded_phase
    (phase / "completion.json").unlink()
    path = next((phase / "receipts").glob("*.json"))
    record = json.loads(path.read_text())
    record["response"]["usage"]["prompt_tokens"] = 100
    path.write_text(json.dumps(record))
    before = snapshot(bundle.parent)
    with pytest.raises(ValueError, match="raw receipt missing integrity anchor or changed"):
        records_from_phase(bundle, phase, config["alias"])
    assert snapshot(bundle.parent) == before


def test_export_phase_binds_arrival_to_successful_attempt(recorded_phase):
    bundle, phase, config, _ = recorded_phase
    (phase / "completion.json").unlink()
    ledger = phase / "attempts.jsonl"
    intent, result = [json.loads(line) for line in ledger.read_text().splitlines()]
    receipt = phase / result["metadata"]["receipt"]
    outcome = phase / result["metadata"]["outcome"]
    second_receipt = receipt.with_name(receipt.name.replace("attempt-1", "attempt-2"))
    second_outcome = outcome.with_name(outcome.name.replace("attempt-1", "attempt-2"))
    receipt.rename(second_receipt)
    saved = json.loads(outcome.read_text())
    saved["attempt"] = 2
    saved["metadata"]["receipt"] = second_receipt.relative_to(phase).as_posix()
    outcome.unlink()
    second_outcome.write_text(json.dumps(saved))
    result["attempt"] = 2
    result["metadata"].update(receipt=saved["metadata"]["receipt"],
                              outcome=second_outcome.relative_to(phase).as_posix(),
                              outcome_sha256=file_hash(second_outcome))
    # An unresolved first intent is valid crash evidence and consumes an attempt.
    rows = [intent | {"at_unix": 100}, intent | {"attempt": 2, "at_unix": 105}, result]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = snapshot(bundle.parent)
    records = records_from_phase(bundle, phase, config["alias"])
    assert records[0]["arrival_offset_seconds"] == 5
    assert records[0]["prompt_tokens"] == 800
    assert snapshot(bundle.parent) == before
