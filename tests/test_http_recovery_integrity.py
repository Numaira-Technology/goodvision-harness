"""Offline integrity checks for retained HTTP outcomes; no transport calls."""
import json

import pytest

from tools.unified_benchmark import http_execution
from tools.unified_benchmark.contract import write_new
from tools.unified_benchmark.retry import Ledger


@pytest.mark.parametrize("bad", [True, False, -1, float("nan"), float("inf"), 10**400])
def test_raw_orphan_rejects_invalid_timestamp_without_writes(tmp_path, bad):
    unit = {"id": "synthetic", "request_sha256": "hash", "targets": ["fight"]}
    config = {"kind": "choice", "backend": "language", "model": "synthetic/model"}
    ledger = Ledger(tmp_path / "attempts.jsonl", {})
    try:
        ledger.append({"type": "intent", "unit": unit["id"], "attempt": 1,
                       "request_sha256": unit["request_sha256"]})
        raw = http_execution.receipt_path(tmp_path, unit, 1)
        # JSON permits these literals in historical files; the recovery validator
        # must reject them before materializing a new outcome or ledger result.
        raw.parent.mkdir(parents=True)
        raw.write_text(json.dumps({"unit": unit["id"], "request_sha256": "hash",
                                   "http_status": 503, "recorded_at_unix": bad}))
        before = ledger.file_hash(), raw.read_bytes()
        with pytest.raises(ValueError, match="durable completion timestamp|nonfinite value"):
            http_execution.reconcile(tmp_path, ledger, {"units": [unit]}, config, local=True)
        assert (ledger.file_hash(), raw.read_bytes()) == before
        assert len(ledger.rows) == 1
        assert not http_execution.outcome_path(tmp_path, unit, 1).exists()
    finally:
        ledger.close()


@pytest.mark.parametrize("corruption, message", [
    ({"unit": "wrong-unit"}, "raw receipt request identity differs"),
    *[({"http_status": status}, "invalid raw receipt http_status")
      for status in (True, False, "503", 99, 600, 200.0)],
])
def test_resume_checks_later_phase_before_any_transport(tmp_path, monkeypatch, corruption, message):
    from helpers.qwen_bundle import frozen_host_bundle
    from tools.qwen8b import eval_local
    from tools.unified_benchmark.contract import digest
    from tools.unified_benchmark.workflow import context_for

    bundle = tmp_path / "bundle"
    config, unit = frozen_host_bundle(bundle)
    root = tmp_path / "run"
    output = root / config["alias"]
    endpoint, bundle_hash = "http://127.0.0.1:9", "synthetic-bundle"
    write_new(output / "run.json", {
        "alias": config["alias"], "phase": "both", "bundle_sha256": bundle_hash,
        "config_sha256": digest(config), "connection": {"endpoint": endpoint},
        "threshold_sha256": None,
    })
    plan = json.loads((bundle / "plans" / f'{config["alias"]}-test.json').read_text())
    directory = output / "test"
    ledger = Ledger(directory / "attempts.jsonl", context_for(plan, config, bundle_hash))
    try:
        ledger.append({"type": "intent", "unit": unit["id"], "attempt": 1,
                       "request_sha256": unit["request_sha256"]})
    finally:
        ledger.close()
    write_new(http_execution.receipt_path(directory, unit, 1), {
        "unit": unit["id"], "request_sha256": unit["request_sha256"],
        "http_status": 503, "recorded_at_unix": 1,
        **corruption,
    })
    calls = []
    def answer(self, body):
        calls.append(body)
        return {"model": config["model"], "choices": [{"finish_reason": "stop",
                "message": {"content": '{"answer": true}'}}]}
    monkeypatch.setattr(eval_local.LocalTransport, "call", answer)
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match=message):
        eval_local.execute(config["alias"], bundle, root, endpoint, bundle_hash, "both", resume=True)
    assert calls == []
    assert not (output / "validation").exists()
    assert {name: (output / name).read_bytes() for name in before} == before


@pytest.mark.parametrize("field", ["elapsed_seconds", "completed_at_unix", "retry_after"])
def test_saved_outcome_rejects_overflowing_numeric_evidence(tmp_path, field):
    unit = {"id": "synthetic", "request_sha256": "hash", "targets": ["fight"]}
    config = {"kind": "choice", "backend": "language", "model": "synthetic/model"}
    ledger = Ledger(tmp_path / "attempts.jsonl", {})
    try:
        ledger.append({"type": "intent", "unit": unit["id"], "attempt": 1,
                       "request_sha256": unit["request_sha256"]})
        outcome = http_execution.outcome_path(tmp_path, unit, 1)
        saved = {"unit": unit["id"], "attempt": 1, "request_sha256": "hash",
                 "context": {}, "answers": {}, "metadata": {}, "fatal": True,
                 "elapsed_seconds": 1, "completed_at_unix": 1, "retry_after": 0,
                 "http_status": 403, "error": "synthetic"}
        write_new(outcome, saved | {field: 10**400})
        before = ledger.file_hash(), outcome.read_bytes()
        with pytest.raises(ValueError, match="invalid saved outcome " + field):
            http_execution.reconcile(tmp_path, ledger, {"units": [unit]}, config, local=True)
        assert (ledger.file_hash(), outcome.read_bytes()) == before
        assert len(ledger.rows) == 1
    finally:
        ledger.close()


@pytest.mark.parametrize("saved", [{}, [], None])
@pytest.mark.parametrize("validate_only", [True, False])
def test_existing_invalid_outcome_is_never_recovered(tmp_path, saved, validate_only):
    unit = {"id": "synthetic", "request_sha256": "hash", "targets": ["fight"]}
    config = {"kind": "choice", "backend": "language", "model": "synthetic/model"}
    ledger = Ledger(tmp_path / "attempts.jsonl", {})
    try:
        ledger.append({"type": "intent", "unit": unit["id"], "attempt": 1,
                       "request_sha256": unit["request_sha256"]})
        raw = http_execution.receipt_path(tmp_path, unit, 1)
        write_new(raw, {"unit": unit["id"], "request_sha256": "hash",
                        "http_status": 503, "recorded_at_unix": 100})
        outcome = http_execution.outcome_path(tmp_path, unit, 1)
        write_new(outcome, saved)
        before = ledger.file_hash(), raw.read_bytes(), outcome.read_bytes()
        with pytest.raises(ValueError, match="invalid saved outcome"):
            http_execution.reconcile(tmp_path, ledger, {"units": [unit]}, config,
                                     local=True, validate_only=validate_only)
        assert (ledger.file_hash(), raw.read_bytes(), outcome.read_bytes()) == before
        assert len(ledger.rows) == 1
    finally:
        ledger.close()


def test_read_only_orphan_validation_preserves_recovery_for_execution(tmp_path):
    unit = {"id": "synthetic", "request_sha256": "hash", "targets": ["fight"]}
    config = {"kind": "choice", "backend": "language", "model": "synthetic/model"}
    ledger = Ledger(tmp_path / "attempts.jsonl", {})
    try:
        ledger.append({"type": "intent", "unit": unit["id"], "attempt": 1,
                       "request_sha256": unit["request_sha256"]})
        raw = http_execution.receipt_path(tmp_path, unit, 1)
        write_new(raw, {"unit": unit["id"], "request_sha256": "hash",
                        "http_status": 503, "recorded_at_unix": 100})
        before = ledger.file_hash(), raw.read_bytes()
        plan = {"units": [unit]}
        http_execution.reconcile(tmp_path, ledger, plan, config, local=True, validate_only=True)
        assert len(ledger.rows) == 1
        assert (ledger.file_hash(), raw.read_bytes()) == before
        assert not http_execution.outcome_path(tmp_path, unit, 1).exists()
        http_execution.reconcile(tmp_path, ledger, plan, config, local=True)
        assert ledger.rows[-1]["cooldown_until"] == 130
        after = ledger.file_hash()
        http_execution.reconcile(tmp_path, ledger, plan, config, local=True)
        assert ledger.file_hash() == after
    finally:
        ledger.close()
