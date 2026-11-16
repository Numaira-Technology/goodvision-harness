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
