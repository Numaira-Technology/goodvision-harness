"""Keep latency qualifications next to conditional percentiles in run logs."""

import json

import pytest

from tools.unified_benchmark.workflow import timings
from tools.vast_run_log import RunLog


@pytest.mark.parametrize("gap", [None, "unresolved", "missing_timing"])
def test_saved_run_log_preserves_latency_denominator_and_fallback(
    tmp_path, monkeypatch, gap
):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    rows = [
        {"type": "result", "complete": True, "elapsed_seconds": 2},
        {"type": "result", "complete": False, "error": "timeout"},
        {"type": "result", "complete": False, "error": "HTTP 503"},
    ]
    if gap == "unresolved":
        rows.append({"type": "intent", "unit": "crashed", "attempt": 1})
    elif gap == "missing_timing":
        rows.append({"type": "result", "complete": True})
    latency = timings(rows, split="test")
    output = tmp_path / "evaluation"
    report = output / "test-model/test/summary.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"test_attempt_latency": latency}))
    session = RunLog("synthetic", "test-model", "synthetic", False)
    session.eval_output = str(output)
    session._owns_eval_output = True
    session.capture_score()
    session.finish_once()
    entry = json.loads((session.run_dir / "entry.json").read_text())
    score = entry["score"]["test"]
    assert score["p50_seconds"] == score["p95_seconds"] == 2
    assert score["latency_n"] == 1
    details = score["latency_details"]
    assert details["failed_attempts"] == 2
    assert details["timeout_attempts"] == 1
    assert details["attempts"] == (3 if gap is None else 4)
    assert details["attempts_without_result"] == (1 if gap == "unresolved" else 0)
    assert details["successful_attempts_without_timing"] == (
        1 if gap == "missing_timing" else 0
    )
    assert details["p95_with_failures_seconds"] == (90 if gap is None else "unknown")
    assert details["p95_with_failures_n"] == 3
    assert details["failure_penalty_seconds"] == 90
    assert details["attempt_elapsed_fallback"] == 1
    assert details["split"] == "test"
    assert details["percentile_method"].startswith("nearest-rank")


def test_old_run_log_report_does_not_invent_failure_counts():
    from tools.vast_run_log import _score_split

    score = _score_split({"test_attempt_latency": {"p50_seconds": 2, "n": 1}})
    assert score["p50_seconds"] == 2
    assert set(score["latency_details"].values()) == {"unknown"}
    assert _score_split({})["latency_details"] == "unknown"
