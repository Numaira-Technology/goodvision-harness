"""Offline checks for the Qwen latency measurement tools. No GPU and no Vast."""

from __future__ import annotations

import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.latency_investigation.export_replay import command_plan, export_subset
from tools.latency_investigation.http_split import begin_open, record_split
from tools.latency_investigation.metrics_diff import diff_snapshots, snapshot_from_text
from tools.latency_investigation.percentiles import p50, p95, ratio
from tools.latency_investigation.results_table import (
    COLUMNS,
    render_csv,
    rows_from_aiperf_jsonl,
    rows_from_vllm_bench,
)
from tools.latency_investigation.summarize_attempts import summarize_phase
from tools.qwen8b import eval_local
from tools.unified_benchmark import native_run
from tools.vast_run_log import _PROBE, parse_probe


def _jpeg() -> bytes:
    return b"\xff\xd8\xff\xd9"


def _record(index: int, tokens: int, arrival: float = 0.0) -> dict:
    body = {
        "model": "qwen/qwen3-vl-8b-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"question {index}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(_jpeg()).decode()
                        },
                    },
                ],
            }
        ],
        "max_tokens": 30,
        "temperature": 0,
    }
    return {
        "request_id": f"req-{index:04d}",
        "prompt_tokens": tokens,
        "arrival_offset_seconds": arrival,
        "output_text": '{"answer": true}',
        "output_tokens": 4,
        "body": body,
    }


def test_ceil_rank_matches_the_formal_benchmark_and_has_no_p99_helper():
    sample = list(range(1, 101))
    assert p50(sample) == 50
    assert p95(sample) == 95
    assert ratio(95, 50) == 1.9
    assert ratio(1, 0) is None
    assert not hasattr(
        __import__("tools.latency_investigation.percentiles", fromlist=["p99"]), "p99"
    )


def test_export_preserves_long_share_and_keeps_warmup_disjoint(tmp_path):
    records = [_record(i, 800 if i < 20 else 100, arrival=i * 0.01) for i in range(400)]
    manifest = export_subset(records, tmp_path / "set", size=250, seed=7)
    assert manifest["timed_count"] == 250
    assert manifest["source_500_999"] == 20
    assert manifest["timed_500_999"] == 12
    timed = set(manifest["timed_request_ids"])
    warm = set(manifest["warmup_request_ids"])
    assert timed.isdisjoint(warm)
    assert warm
    digest = (tmp_path / "set" / "manifest.sha256").read_text().strip()
    assert len(digest) == 64
    for rel, hashed in manifest["files_sha256"].items():
        assert len(hashed) == 64
        assert (tmp_path / "set" / rel).is_file()
    commands = json.loads((tmp_path / "set" / "commands.json").read_text())
    for step in commands["steps"]:
        argv = step.get("argv") or []
        assert "--ignore-eos" not in argv
        assert "ignore_eos" not in argv
    assert commands["tool_versions"] == {"vllm": "0.31.0", "aiperf": "0.13.0"}
    assert commands["do_not_run_until_authorized"] is True
    closed = next(step for step in commands["steps"] if step["name"] == "closed-loop")
    assert closed["latency_starts"] == "actual_send"
    assert closed["argv"].count("12") >= 1
    assert "--max-concurrency" in closed["argv"]
    assert "--custom-output-len" in closed["argv"]
    assert closed["argv"][closed["argv"].index("--custom-output-len") + 1] == "30"
    image = next(iter((tmp_path / "set" / "images").iterdir()))
    assert image.read_bytes() == _jpeg()
    plan = command_plan(
        tmp_path / "set", max_tokens=30, temperature=0, timed_count=250, warmup_count=12
    )
    assert plan["steps"][0]["name"] == "restart-vllm"
    assert (
        "rent" not in plan["steps"][0]["action"].lower()
        or "does not rent" in plan["steps"][0]["action"]
    )


def test_vllm_table_uses_actual_send_latencies_and_skips_p99(tmp_path):
    records = [_record(i, 800 if i == 0 else 100) for i in range(20)]
    manifest = export_subset(records, tmp_path / "set", size=20, seed=1)
    latencies = [1.0 + index * 0.01 for index in range(manifest["timed_count"])]
    result = {
        "date": "20261112",
        "max_concurrency": 12,
        "latencies": latencies,
        "errors": [""] * len(latencies),
        "failed": 0,
        "completed": len(latencies),
        "generated_texts": ['{"answer": true}'] * len(latencies),
        "output_lens": [4] * len(latencies),
        "p99_e2el_ms": 999,
    }
    rows = rows_from_vllm_bench(
        result,
        manifest,
        host={
            "run_date": "2026-11-12",
            "location": "Hungary, HU",
            "gpu": "RTX 4090",
            "cpu_model": "AMD EPYC",
        },
    )
    assert rows[0]["input_size_bucket"] == "overall"
    assert "p99" not in COLUMNS
    csv_text = render_csv(rows)
    assert "p99" not in csv_text
    assert "actual_send" in rows[0]["notes"]
    assert rows[0]["p50_seconds"] == p50(latencies)
    assert rows[0]["concurrency"] == 12


def test_metrics_diff_means_and_reset_are_not_zero():
    before = snapshot_from_text(
        "\n".join(
            [
                "# HELP vllm:request_queue_time_seconds Queue",
                "vllm:request_queue_time_seconds_sum 1.0",
                "vllm:request_queue_time_seconds_count 4",
                "vllm:time_to_first_token_seconds_sum 2",
                "vllm:time_to_first_token_seconds_count 4",
                "vllm:request_prefill_time_seconds_sum 3",
                "vllm:request_prefill_time_seconds_count 4",
                "vllm:request_decode_time_seconds_sum 4",
                "vllm:request_decode_time_seconds_count 4",
                "vllm:num_preemptions_total 1",
            ]
        )
    )
    after = snapshot_from_text(
        "\n".join(
            [
                "vllm:request_queue_time_seconds_sum 3.0",
                "vllm:request_queue_time_seconds_count 8",
                "vllm:time_to_first_token_seconds_sum 2",
                "vllm:time_to_first_token_seconds_count 4",
                "vllm:request_prefill_time_seconds_sum 5",
                "vllm:request_prefill_time_seconds_count 8",
                "vllm:request_decode_time_seconds_sum 9",
                "vllm:request_decode_time_seconds_count 8",
                "vllm:num_preemptions_total 3",
            ]
        )
    )
    diff = diff_snapshots(before, after)
    assert diff["queue_time_seconds"]["mean_seconds"] == 0.5
    assert diff["ttft_seconds"]["count_delta"] == 0
    assert diff["preemptions"]["delta"] == 2
    reset = diff_snapshots(after, before)
    assert reset["queue_time_seconds"]["reason"] == "counter_reset"
    assert reset["queue_time_seconds"]["available"] is False
    missing = snapshot_from_text("vllm:num_requests_running 1\n")
    assert missing["queue_time_seconds"] == "unknown"
    assert diff_snapshots(missing, missing)["preemptions"]["reason"] == "not_exposed"


def test_probe_extra_line_parses_and_old_probes_stay_intact():
    assert "scaling_governor" in _PROBE
    assert "numa_node" in _PROBE
    assert "query-compute-apps" in _PROBE
    assert "[[" not in _PROBE
    text = (
        "RTX 4090, 0000:81:00.0, 4, 4, 8, 16, 450.00 W, 2520 MHz, 3105 MHz, "
        "10251 MHz, 550.90, 24564 MiB, 0 %\n"
        "HOST\tAMD EPYC 7763 64-Core Processor\t32\t257564 kB\t12.4\n"
        "EXTRA\tperformance\tenabled\t2\t0@0000:81:00.0=1\tnone\n"
    )
    parsed = parse_probe(text)
    assert parsed["host"]["cpu_model"] == "AMD EPYC 7763 64-Core Processor"
    assert parsed["investigation"]["cpu_governor"] == "performance"
    assert parsed["investigation"]["cpu_turbo"] == "enabled"
    assert parsed["investigation"]["numa_node_count"] == "2"
    assert parsed["investigation"]["gpu_numa_nodes"] == "0@0000:81:00.0=1"
    assert parsed["investigation"]["gpu_compute_apps"] == "none"
    old = parse_probe(
        "RTX 4090, 0000:81:00.0, 4, 4, 8, 16, 450.00 W, 2520 MHz, 3105 MHz, "
        "10251 MHz, 550.90, 24564 MiB, 0 %\n"
        "HOST\tAMD EPYC\t32\t257564 kB\t12.4\n"
    )
    assert "investigation" not in old


def test_summarize_phase_buckets_complete_attempts(tmp_path):
    phase = tmp_path / "test"
    receipts = phase / "receipts"
    receipts.mkdir(parents=True)
    receipt = receipts / "unit-a.json"
    receipt.write_text(
        json.dumps(
            {
                "response": {"usage": {"prompt_tokens": 640}},
            }
        )
    )
    short = receipts / "unit-b.json"
    short.write_text(json.dumps({"response": {"usage": {"prompt_tokens": 80}}}))
    rows = [
        {
            "type": "result",
            "complete": True,
            "elapsed_seconds": 1.0,
            "metadata": {"receipt": "test/receipts/unit-a.json"},
        },
        {
            "type": "result",
            "complete": True,
            "elapsed_seconds": 2.0,
            "metadata": {"receipt": "test/receipts/unit-b.json"},
        },
        {
            "type": "result",
            "complete": False,
            "elapsed_seconds": 9.0,
            "metadata": {"receipt": "test/receipts/unit-a.json"},
        },
    ]
    (phase / "attempts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    summary = summarize_phase(phase)
    assert summary["complete_attempts"] == 2
    overall = summary["table"][0]
    assert overall["n"] == 2
    assert overall["p50_seconds"] == 1.0
    buckets = {row["input_size_bucket"]: row["n"] for row in summary["table"]}
    assert buckets["500-999"] == 1
    assert buckets["<500"] == 1


def test_split_timing_is_off_by_default_and_records_three_marks(tmp_path, monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(size)
            body = b'{"ok": true, "pad": "' + b"x" * 64 + b'"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        monkeypatch.delenv("GOODVISION_SPLIT_TIMING", raising=False)
        import urllib.request

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        response, timings = begin_open(request, 5)
        with response:
            assert json.load(response)["ok"] is True
        assert timings is None
        record_split(tmp_path, timings, http_status=200)
        assert not (tmp_path / "transport-splits.jsonl").exists()

        monkeypatch.setenv("GOODVISION_SPLIT_TIMING", "1")
        response, timings = begin_open(request, 5)
        with response:
            json.load(response)
        assert timings["upload_done_seconds"] <= timings["first_byte_seconds"]
        assert timings["first_byte_seconds"] <= timings["read_done_seconds"]
        record_split(tmp_path, timings, http_status=200)
        saved = json.loads(
            (tmp_path / "transport-splits.jsonl").read_text().splitlines()[0]
        )
        assert set(saved) >= {
            "upload_done_seconds",
            "first_byte_seconds",
            "read_done_seconds",
        }
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_default_clients_still_call_urlopen(monkeypatch):
    monkeypatch.delenv("GOODVISION_SPLIT_TIMING", raising=False)
    calls = []

    def urlopen(request, timeout=0):
        calls.append(request.full_url)
        raise eval_local.urllib.error.URLError("stopped")

    monkeypatch.setattr(eval_local.urllib.request, "urlopen", urlopen)
    client = eval_local.LocalTransport(
        Path("."),
        {"model": "qwen/qwen3-vl-8b-instruct"},
        "http://127.0.0.1:9",
        Path("."),
        None,
    )
    with pytest.raises(eval_local.urllib.error.URLError):
        client.call({"messages": []})
    assert calls == ["http://127.0.0.1:9/v1/chat/completions"]
    assert "begin_open" in Path(native_run.__file__).read_text()
    assert os.environ.get("GOODVISION_SPLIT_TIMING", "") != "1"


def test_open_loop_adds_schedule_lag_and_keeps_actual_send(tmp_path):
    manifest = {
        "vllm_version": "0.31.0",
        "requests": [
            {"request_id": "a", "bucket": "<500", "arrival_offset_seconds": 0.0},
            {"request_id": "b", "bucket": "<500", "arrival_offset_seconds": 1.0},
        ],
    }
    # B is sent 0.5s after its schedule. A is the least-late request, lag 0.
    lines = [
        {
            "metadata": {
                "conversation_id": "a",
                "request_start_ns": 1_000_000_000,
            },
            "metrics": {"request_latency": {"value": 500, "unit": "ms"}},
            "error": None,
        },
        {
            "metadata": {
                "conversation_id": "b",
                "request_start_ns": 2_500_000_000,
            },
            "metrics": {"request_latency": {"value": 500, "unit": "ms"}},
            "error": None,
        },
    ]
    path = tmp_path / "profile_export.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in lines))
    rows = rows_from_aiperf_jsonl(path, manifest)
    overall = rows[0]
    assert overall["input_size_bucket"] == "overall"
    assert "scheduled_send" in overall["notes"]
    # Scheduled latencies are 0.5s and 1.0s. Ceil-rank P50 of two values is the first.
    assert overall["p50_seconds"] == 0.5
    assert overall["p95_seconds"] == 1.0
    assert overall["p50_actual_send_seconds"] == 0.5
    assert overall["p95_actual_send_seconds"] == 0.5


@pytest.mark.parametrize(
    "result",
    [
        {
            "latencies": [0, 0, 2],
            "errors": ["HTTP 500", "timeout", ""],
            "failed": 2,
            "completed": 1,
        },
        {"latencies": [0, 0, 0], "errors": ["error"] * 3, "failed": 3, "completed": 0},
        {"latencies": [1, 2, 3], "errors": [""], "failed": 0, "completed": 3},
        {"latencies": [1, 2, 3], "errors": [""] * 3, "failed": 0, "completed": 2},
        {"latencies": [1, 2, 3]},
        {
            "latencies": [1, float("nan"), 3],
            "errors": [""] * 3,
            "failed": 0,
            "completed": 3,
        },
    ],
)
def test_vllm_incomplete_or_inconsistent_runs_cannot_publish_percentiles(result):
    manifest = {"requests": [{"request_id": str(i)} for i in range(3)]}
    with pytest.raises(ValueError, match="bench"):
        rows_from_vllm_bench(result, manifest)


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{"error": "HTTP 500"}],
        [{"metrics": {}}],
        [{"metrics": {"request_latency": {"value": float("nan"), "unit": "s"}}}],
    ],
)
def test_aiperf_unavailable_run_has_clear_error(tmp_path, records):
    path = tmp_path / "profile_export.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="no successful requests with valid latency"):
        rows_from_aiperf_jsonl(path, {"requests": []})


def test_relative_export_paths_and_no_zero_request_warmup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = export_subset([_record(0, 100)], Path("replay"))
    assert manifest["warmup_count"] == 0
    plan = json.loads(Path("replay/commands.json").read_text())
    assert Path(plan["cwd"]) == tmp_path / "replay"
    assert "warmup" not in [step["name"] for step in plan["steps"]]
    for step in plan["steps"]:
        argv = step.get("argv", [])
        for flag, directory in [
            ("--result-dir", "results"),
            ("--output-artifact-dir", "aiperf"),
        ]:
            if flag in argv:
                assert Path(plan["cwd"]) / argv[argv.index(flag) + 1] == (
                    tmp_path / "replay" / directory
                )


def test_metric_boundaries_exclude_warmup_and_scope_each_replay(tmp_path):
    plan = command_plan(
        tmp_path, max_tokens=30, temperature=0, timed_count=250, warmup_count=12
    )
    assert [step["name"] for step in plan["steps"]] == [
        "restart-vllm",
        "warmup",
        "closed-loop-metrics-before",
        "closed-loop",
        "closed-loop-metrics-after",
        "open-loop-restart-vllm",
        "open-loop-warmup",
        "open-loop-metrics-before",
        "open-loop",
        "open-loop-metrics-after",
    ]
    outputs = [
        step["argv"][-1] for step in plan["steps"] if "-metrics-" in step["name"]
    ]
    assert len(set(outputs)) == 4


def test_created_timestamp_is_not_preemption_count():
    before = snapshot_from_text("vllm:num_preemptions_created 1000")
    after = snapshot_from_text("vllm:num_preemptions_created 1100")
    assert diff_snapshots(before, after)["preemptions"] == {
        "available": False,
        "reason": "not_exposed",
    }


@pytest.mark.parametrize(
    "changed,reason",
    [
        ({"a": 1, "b": 100}, "counter_reset"),
        ({"b": 100}, "metric_series_changed"),
    ],
)
def test_metric_series_reset_or_loss_cannot_hide_in_aggregate(changed, reason):
    def scrape(values):
        return "\n".join(
            f'{metric}{{worker="{worker}"}} {value}'
            for worker, value in values.items()
            for metric in [
                "vllm:request_queue_time_seconds_sum",
                "vllm:request_queue_time_seconds_count",
                "vllm:num_preemptions_total",
            ]
        )

    diff = diff_snapshots(
        snapshot_from_text(scrape({"a": 10, "b": 10})),
        snapshot_from_text(scrape(changed)),
    )
    for metric in ("queue_time_seconds", "preemptions"):
        assert diff[metric] == {"available": False, "reason": reason}


@pytest.mark.parametrize("status", [200, 429, 401])
@pytest.mark.parametrize("client_kind", ["qwen", "native"])
def test_sidecar_write_failure_preserves_transport_result(
    tmp_path,
    monkeypatch,
    status,
    client_kind,
):
    import io
    import urllib.error

    from tools.latency_investigation import http_split
    from tools.unified_benchmark.contract import digest
    from tools.unified_benchmark.retry import RequestFailure

    monkeypatch.setenv("GOODVISION_SPLIT_TIMING", "1")
    # Only the optional sidecar is unwritable. Receipts remain writable.
    (tmp_path / http_split.SIDECAR).mkdir()
    calls = []
    payload = {"model": "test-model", "ok": True}

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            if status != 200:
                raise urllib.error.HTTPError(
                    request.full_url,
                    status,
                    "failure",
                    {"Retry-After": "17"},
                    io.BytesIO(b"error"),
                )
            response = io.BytesIO(json.dumps(payload).encode())
            response.status = status
            return response

    monkeypatch.setattr(http_split.urllib.request, "build_opener", lambda *a: Opener())
    if client_kind == "native":
        client = native_run.NativeBackend("test", {}, tmp_path, tmp_path, tmp_path, 9)
        code, body = client.http({})
        assert code == status
        assert body == (payload if status == 200 else {"error": "error"})
    else:

        class Ledger:
            def state(self, *args):
                return (1,)

        client = eval_local.LocalTransport(
            tmp_path,
            {"model": "test-model", "kind": "choice"},
            "http://127.0.0.1:9",
            tmp_path,
            Ledger(),
        )
        monkeypatch.setattr(eval_local, "request_body", lambda *a: (None, {}))
        monkeypatch.setattr(eval_local, "parse_response", lambda *a: {"ok": True})
        unit = {"id": "a", "request_sha256": digest({})}
        if status == 200:
            answers, metadata = client(unit)
            assert answers == {"ok": True}
            assert metadata["actual_model"] == "test-model"
        else:
            with pytest.raises(RequestFailure) as caught:
                client(unit)
            assert caught.value.status == status
            assert caught.value.retry_after == 17
            assert caught.value.fatal == (status == 401)
    assert len(calls) == 1
    assert http_split._UNIT.get() is None


def test_sidecar_mkdir_failure_is_best_effort(tmp_path):
    path = tmp_path / "not-a-directory"
    path.write_text("file")
    record_split(path, {"upload_done_seconds": 1}, http_status=200)


def test_summary_reuses_formal_transport_and_failure_accounting(tmp_path):
    from tools.unified_benchmark.workflow import timings

    rows = [
        {
            "type": "result",
            "unit": "a",
            "attempt": 1,
            "complete": True,
            "elapsed_seconds": 20,
            "metadata": {"transport_seconds": 2},
        },
        {
            "type": "result",
            "unit": "b",
            "attempt": 1,
            "complete": False,
            "error": "TimeoutError",
            "elapsed_seconds": 90,
        },
        {
            "type": "result",
            "unit": "c",
            "attempt": 1,
            "complete": True,
            "elapsed_seconds": 3,
        },
        {"type": "intent", "unit": "d", "attempt": 1},
    ]
    (tmp_path / "attempts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    table = summarize_phase(tmp_path)["table"][0]
    expected = timings(rows, split=tmp_path.name)
    assert all(table[key] == value for key, value in expected.items())
    assert table["p50_seconds"] == 2
    assert table["failed_attempts"] == 1
    assert table["timeout_attempts"] == 1
    assert table["attempt_elapsed_fallback"] == 1
    assert table["attempts_without_result"] == 1
    assert table["p95_with_failures_seconds"] is None


@pytest.mark.parametrize(
    "sessions", [["a"], ["a", "a"], ["a", "unknown"], ["a", "failed"]]
)
def test_aiperf_rejects_partial_failed_duplicate_and_unknown_sessions(
    tmp_path, sessions
):
    manifest = {"requests": [{"request_id": "a"}, {"request_id": "b"}]}
    records = [
        {
            "metadata": {"conversation_id": session},
            "metrics": {"request_latency": {"value": 1, "unit": "s"}},
            "error": "HTTP 500" if session == "failed" else None,
        }
        for session in sessions
    ]
    path = tmp_path / "aiperf.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="every manifest request"):
        rows_from_aiperf_jsonl(path, manifest)
