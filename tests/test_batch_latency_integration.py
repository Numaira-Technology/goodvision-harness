"""Batch and interrupted HTTP reports retain the shared latency qualifications."""

import csv
import json
import urllib.error

import pytest
from test_benchmark_latency_definition import api_attempt as api_attempt

from tools.unified_benchmark import batch, http_execution, workflow
from tools.unified_benchmark.contract import write_new
from tools.unified_benchmark.retry import Queue


@pytest.mark.parametrize("gap", [None, "unresolved", "missing_timing", "legacy"])
def test_batch_json_and_csv_keep_latency_qualifications(tmp_path, gap):
    rows = [
        {"type": "result", "complete": True, "elapsed_seconds": 2},
        {"type": "result", "complete": False, "error": "timeout"},
    ]
    if gap == "unresolved":
        rows.append({"type": "intent", "unit": "crashed", "attempt": 1})
    elif gap == "missing_timing":
        rows.append({"type": "result", "complete": True})
    latency = workflow.timings(rows, split="test")
    if gap == "legacy":
        latency = {"n": 1, "p50_seconds": 2, "p95_seconds": 2}
    write_new(tmp_path / "r1/synthetic/test/summary.json", {
        "accuracy": 1, "test_attempt_latency": latency,
    })
    config = {"models": [{"alias": "synthetic", "route": "api"}]}
    matrix = [{
        "alias": "synthetic", "split": "test", "round": i, "output": f"r{i}",
        "state": "completed" if i == 1 else "not_started",
        "invocations": 1 if i == 1 else 0,
    } for i in (1, 2)]
    batch.summarize(tmp_path, config, {
        "batch_id": "synthetic", "matrix": matrix, "reserved_upper_usd": 0,
    })
    report = json.loads((tmp_path / "summary.json").read_text())
    rounds = report["models"]["synthetic"]["rounds"]
    with (tmp_path / "rounds.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    fields = (
        "failed_attempts", "timeout_attempts", "p95_with_failures_seconds",
        "failure_penalty_seconds", "percentile_method", "attempts",
        "attempts_without_result", "successful_attempts_without_timing",
        "p95_with_failures_n", "attempt_elapsed_fallback",
    )
    for name in fields:
        expected = latency.get(name)
        assert rounds[0][name] == expected
        assert csv_rows[0][name] == ("" if expected is None else str(expected))
        assert rounds[1][name] is None
        assert csv_rows[1][name] == ""
    assert rounds[0]["latency_source"] == latency.get("source")
    assert rounds[0]["latency_n"] == 1
    assert rounds[1]["latency_n"] is None


def test_orphan_http_timeout_keeps_failure_classification(api_attempt, monkeypatch):
    transport, ledger, unit = api_attempt
    calls = []

    def timeout(url, body=None):
        calls.append(url)
        raise urllib.error.URLError(TimeoutError("timed out"))

    transport.call = timeout
    durable = http_execution.DurableTransport(
        transport, transport.output, ledger, transport.config,
    )
    original = http_execution.write_new

    def crash_before_outcome(path, row):
        if path.parent.name == "outcomes":
            raise SystemExit("synthetic interruption after raw timeout receipt")
        return original(path, row)

    monkeypatch.setattr(http_execution, "write_new", crash_before_outcome)
    with pytest.raises(SystemExit, match="synthetic interruption"):
        Queue(ledger, "choice", durable).one(unit)
    monkeypatch.setattr(http_execution, "write_new", original)
    http_execution.reconcile(
        transport.output, ledger, {"units": [unit]}, transport.config,
    )
    assert len(calls) == 1
    result = ledger.rows[-1]
    assert result["type"] == "result" and result["error"] == "timeout"
    latency = workflow.timings(ledger.rows, split="test")
    assert latency["attempts"] == latency["failed_attempts"] == 1
    assert latency["timeout_attempts"] == 1
    assert latency["attempts_without_result"] == 0
    assert latency["p95_with_failures_seconds"] == 90
