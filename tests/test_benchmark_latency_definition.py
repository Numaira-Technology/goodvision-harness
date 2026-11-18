"""Benchmark P50/P95: timing boundaries, denominators and missing evidence."""

import threading
import time
from types import SimpleNamespace

import pytest

from tools.unified_benchmark import adapters as a
from tools.unified_benchmark import api_run, latency
from tools.unified_benchmark.contract import digest, file_hash
from tools.unified_benchmark.retry import Ledger, PausedRun, Queue
from tools.unified_benchmark.workflow import timings


def result(seconds=None, *, complete=True, error=None, transport=None):
    row = {"type": "result", "complete": complete, "elapsed_seconds": seconds}
    if error:
        row["error"] = error
    if transport is not None:
        row["metadata"] = {"transport_seconds": transport}
    return row


def test_nearest_rank_matches_the_documented_index():
    values = list(range(1, 21))
    assert latency.nearest_rank(values, 0.5) == 10
    assert latency.nearest_rank(values, 0.95) == 19
    assert latency.nearest_rank([3, 1], 0.5) == 1
    assert latency.nearest_rank([3, 1], 0.95) == 3
    assert latency.nearest_rank([], 0.5) is None
    assert latency.nearest_rank([7], 0) == 7


def test_failures_are_counted_and_raise_the_failure_p95():
    rows = [result(transport=t, seconds=t + 5) for t in (1,) * 9 + (2,)]
    rows += [
        result(1, complete=False, error="timeout"),
        result(0.2, complete=False, error="HTTP 503"),
    ]
    report = timings(rows)
    assert (report["n"], report["p50_seconds"], report["p95_seconds"]) == (10, 1, 2)
    assert (report["failed_attempts"], report["timeout_attempts"]) == (2, 1)
    assert report["p95_with_failures_seconds"] == latency.REQUEST_TIMEOUT_SECONDS
    assert report["attempts"] == report["p95_with_failures_n"] == 12
    assert report["attempt_elapsed_fallback"] == 0
    assert report["percentile_method"].startswith("nearest-rank")


def test_old_ledgers_without_transport_time_fall_back_and_say_so():
    report = timings([result(1.5), result(2.5)])
    assert (report["n"], report["p50_seconds"], report["attempt_elapsed_fallback"]) == (
        2,
        1.5,
        2,
    )
    assert report["failed_attempts"] == 0 and report["p95_with_failures_seconds"] == 2.5


def test_interrupted_attempt_stays_visible_until_its_late_result_arrives():
    first = {"unit": "u", "attempt": 1}
    retry = {"unit": "u", "attempt": 2}
    rows = [
        first | {"type": "intent"},
        retry | {"type": "intent"},
        retry | result(transport=2),
    ]
    report = timings(rows, split="test")
    assert report["attempts"] == 2 and report["n"] == 1
    assert report["attempts_without_result"] == 1
    assert report["failed_attempts"] == report["timeout_attempts"] == 0
    assert report["p95_with_failures_n"] == 1
    assert report["p95_with_failures_seconds"] is None
    # Recovered results can be appended after a later attempt's result.
    rows.append(first | result(transport=3))
    recovered = timings(iter(rows), split="test")
    assert recovered["attempts"] == recovered["p95_with_failures_n"] == 2
    assert recovered["attempts_without_result"] == 0
    assert recovered["p95_with_failures_seconds"] == 3
    assert recovered["split"] == "test"


@pytest.mark.parametrize("duration", [None, -1, True, "2", float("nan"), float("inf")])
@pytest.mark.parametrize("legacy", [False, True])
def test_missing_or_invalid_success_timing_cannot_shrink_the_denominator(
    duration, legacy
):
    row = result(duration) if legacy else result(5)
    if not legacy:
        row["metadata"] = {"transport_seconds": duration}
    report = timings([result(transport=0), row])
    assert report["attempts"] == 2 and report["n"] == 1
    assert report["p50_seconds"] == 0
    assert report["successful_attempts_without_timing"] == 1
    assert report["attempt_elapsed_fallback"] == 0
    assert report["p95_with_failures_seconds"] is None


def test_empty_ledger_has_no_percentiles_or_attempts():
    report = timings([])
    assert report["attempts"] == report["p95_with_failures_n"] == 0
    assert report["p50_seconds"] is report["p95_seconds"] is None
    assert report["p95_with_failures_seconds"] is None


@pytest.mark.parametrize(
    "error",
    [
        "timeout",
        "URLError: <urlopen error timed out>",
        "TimeoutError: timed out",
        "ReadTimeout",
    ],
)
def test_timeout_errors_are_recognised(error):
    assert latency.is_timeout(error)


@pytest.mark.parametrize("error", [None, "HTTP 503", "ConnectionError: reset", 5])
def test_other_errors_are_not_timeouts(error):
    assert not latency.is_timeout(error)


@pytest.fixture
def api_attempt(tmp_path):
    (tmp_path / "f.jpg").write_bytes(b"synthetic-image")
    config = {
        "model": "test/model",
        "backend": "language",
        "kind": "choice",
        "max_tokens": 30,
        "temperature": 0,
        "provider_context": {
            "config": {"expected_provider_name": "Mock", "provider": {"only": ["mock"]}}
        },
    }
    unit = {
        "id": "u",
        "source_id": "s",
        "targets": ["fight"],
        "question": "Fight?",
        "frames": [
            {"path": "f.jpg", "sha256": file_hash(tmp_path / "f.jpg"), "time": 0}
        ],
    }
    unit["request_sha256"] = digest(a.request_body(config, unit, tmp_path)[1])
    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    transport = api_run.Transport(
        tmp_path, config, "synthetic-key", tmp_path / "out", ledger, threading.Event()
    )
    try:
        yield transport, ledger, unit
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "invalid_answer",
        "provider_mismatch",
        "model_mismatch",
        "lookup_failure",
    ],
)
def test_api_transport_timing_excludes_overhead_and_survives_identity_failure(
    api_attempt,
    monkeypatch,
    outcome,
):
    transport, ledger, unit = api_attempt
    clock = [0.0]
    monkeypatch.setattr(
        api_run, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
    )

    def delayed(original, seconds):
        def wrapped(*args, **kwargs):
            clock[0] += seconds
            return original(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(api_run, "request_body", delayed(api_run.request_body, 4))
    monkeypatch.setattr(api_run, "write_receipt", delayed(api_run.write_receipt, 5))
    monkeypatch.setattr(api_run, "file_hash", delayed(api_run.file_hash, 3))
    monkeypatch.setattr(api_run, "parse_response", delayed(api_run.parse_response, 1))
    answer = {
        "id": "g",
        "model": "test/model",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "invalid"
                    if outcome == "invalid_answer"
                    else '{"answer":true}'
                },
            }
        ],
    }
    if outcome == "model_mismatch":
        answer["model"] = "wrong/model"

    def call(url, body=None):
        if body is None:
            clock[0] += 9
            if outcome == "lookup_failure":
                raise TimeoutError("metadata lookup timeout")
            return {
                "data": {
                    "id": "g",
                    "provider_name": (
                        "Wrong" if outcome == "provider_mismatch" else "Mock"
                    ),
                    "model": "test/model",
                }
            }
        clock[0] += 2
        return answer

    transport.call = call
    queue = Queue(
        ledger,
        "choice",
        transport,
        clock=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    identity_failure = outcome in {
        "provider_mismatch",
        "model_mismatch",
        "lookup_failure",
    }
    if identity_failure:
        with pytest.raises(PausedRun):
            queue.one(unit)
    else:
        queue.one(unit)
    rows = [row for row in ledger.rows if row["type"] == "result"]
    assert len(rows) == (3 if outcome == "invalid_answer" else 1)
    for row in rows:
        assert row["complete"] is (outcome == "success")
        assert row["metadata"]["metadata_lookup_seconds"] == 9
        assert row["metadata"]["transport_seconds"] == (2 if identity_failure else 3)
        assert row["elapsed_seconds"] == (23 if identity_failure else 24)
    report = timings(rows)
    assert (
        report["timeout_attempts"] == 0
    )  # Metadata GET timeout is not inference timeout.
    if outcome == "success":
        assert report["p50_seconds"] == 3


def test_api_timeout_is_labelled_and_keeps_its_time(api_attempt, monkeypatch):
    transport, ledger, unit = api_attempt
    clock = [0.0]
    monkeypatch.setattr(
        api_run, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
    )

    def call(url, body=None):
        clock[0] += 90
        raise TimeoutError("timed out")

    transport.call = call
    queue = Queue(
        ledger,
        "choice",
        transport,
        slots=1,
        clock=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    queue.one(unit)
    rows = [r for r in ledger.rows if r["type"] == "result"]
    assert len(rows) == 3 and all(r["error"] == "timeout" for r in rows)
    assert all(r["metadata"]["transport_seconds"] == 90 for r in rows)
    report = timings(rows)
    assert report["n"] == 0
    assert (report["failed_attempts"], report["timeout_attempts"]) == (3, 3)
    assert report["p95_with_failures_seconds"] == latency.REQUEST_TIMEOUT_SECONDS
