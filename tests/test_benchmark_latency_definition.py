"""Benchmark P50/P95: what is timed, which percentile, failures kept."""

import threading
import time

import pytest

from tools.unified_benchmark import adapters as a
from tools.unified_benchmark import latency
from tools.unified_benchmark.contract import digest, file_hash
from tools.unified_benchmark.retry import Ledger, Queue
from tools.unified_benchmark.workflow import timings


def result(seconds=None, *, complete=True, error=None, transport=None):
    row = {"type": "result", "complete": complete, "elapsed_seconds": seconds}
    if error:
        row["error"] = error
    if transport is not None:
        row["metadata"] = {"transport_seconds": transport}
    return row


def test_nearest_rank_matches_the_documented_index():
    values = list(range(1, 21))  # 1..20
    assert latency.nearest_rank(values, 0.5) == 10
    assert latency.nearest_rank(values, 0.95) == 19
    assert latency.nearest_rank([3, 1], 0.95) == 3
    assert latency.nearest_rank([], 0.5) is None
    assert latency.nearest_rank([7], 0) == 7


def test_failures_are_counted_and_raise_the_failure_p95():
    rows = [result(transport=t, seconds=t + 5) for t in (1, 1, 1, 1, 1, 1, 1, 1, 1, 2)]
    rows += [
        {"type": "intent"},
        result(1, complete=False, error="timeout"),
        result(0.2, complete=False, error="HTTP 503"),
    ]
    report = timings(rows)
    # Success-only percentiles use transport time, not whole-attempt elapsed.
    assert (report["n"], report["p50_seconds"], report["p95_seconds"]) == (10, 1, 2)
    assert (report["failed_attempts"], report["timeout_attempts"]) == (2, 1)
    # 12 samples, two at the 90 s timeout: P95 is index ceil(11.4) - 1 = 11.
    assert report["p95_with_failures_seconds"] == latency.REQUEST_TIMEOUT_SECONDS
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


def test_api_transport_time_excludes_the_metadata_lookup(tmp_path):
    from tools.unified_benchmark.api_run import Transport

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
    u = {
        "id": "u",
        "source_id": "s",
        "targets": ["fight"],
        "question": "Fight?",
        "frames": [
            {"path": "f.jpg", "sha256": file_hash(tmp_path / "f.jpg"), "time": 0}
        ],
    }
    u["request_sha256"] = digest(a.request_body(config, u, tmp_path)[1])
    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    transport = Transport(
        tmp_path, config, "synthetic-key", tmp_path / "out", ledger, threading.Event()
    )
    # No provider in the answer, so the runner looks it up; that GET is slow here.
    answer = {
        "id": "g",
        "model": "test/model",
        "choices": [
            {"finish_reason": "stop", "message": {"content": '{"answer":true}'}}
        ],
    }

    def call(url, body=None):
        if body is None:
            time.sleep(0.3)
            return {"data": {"id": "g", "provider_name": "Mock", "model": "test/model"}}
        return answer

    transport.call = call
    try:
        Queue(ledger, "choice", transport).one(u)
        row = ledger.rows[-1]
        assert row["complete"]
        assert row["metadata"]["metadata_lookup_seconds"] >= 0.3
        assert row["metadata"]["transport_seconds"] < 0.3
        assert row["elapsed_seconds"] >= 0.3  # The whole attempt still includes it.
        assert (
            timings(ledger.rows)["p50_seconds"] == row["metadata"]["transport_seconds"]
        )
    finally:
        ledger.close()


def test_api_timeout_is_labelled_and_keeps_its_time(tmp_path):
    from tools.unified_benchmark.api_run import Transport

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
    u = {
        "id": "u",
        "source_id": "s",
        "targets": ["fight"],
        "question": "Fight?",
        "frames": [
            {"path": "f.jpg", "sha256": file_hash(tmp_path / "f.jpg"), "time": 0}
        ],
    }
    u["request_sha256"] = digest(a.request_body(config, u, tmp_path)[1])
    ledger = Ledger(tmp_path / "attempts.jsonl", {"test": True})
    transport = Transport(
        tmp_path, config, "synthetic-key", tmp_path / "out", ledger, threading.Event()
    )

    def call(url, body=None):
        raise TimeoutError("timed out")

    transport.call = call
    clock = [0.0]
    queue = Queue(
        ledger,
        "choice",
        transport,
        slots=1,
        clock=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    try:
        queue.one(u)
        rows = [r for r in ledger.rows if r["type"] == "result"]
    finally:
        ledger.close()
    assert len(rows) == 3 and all(r["error"] == "timeout" for r in rows)
    assert all("transport_seconds" in r["metadata"] for r in rows)
    report = timings(rows)
    assert report["n"] == 0 and (
        report["failed_attempts"],
        report["timeout_attempts"],
    ) == (3, 3)
    assert report["p95_with_failures_seconds"] == latency.REQUEST_TIMEOUT_SECONDS
