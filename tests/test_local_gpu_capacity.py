"""The benchmark tests use mock transports, never a cloud or GPU service."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location(
    "local_gpu_capacity",
    Path(__file__).resolve().parents[1] / "tools/local_gpu_capacity.py",
)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def request_payload():
    return {
        "model": "local-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the visible event."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                            )
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1600,
    }


def response_payload():
    return {
        "choices": [{"message": {"content": "An observed event."}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }


def records_for(config, latency=0.05):
    return [
        {
            "scheduled": index / config.target_rps,
            "finished": index / config.target_rps + latency,
            "camera": index % config.cameras,
            "outcome": "ok",
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        }
        for index in range(
            benchmark.arrival_count(config.warmup + config.duration, config.target_rps)
        )
    ]


def report(config, records, **kwargs):
    return benchmark.summarize(
        config,
        records,
        pending_at_measurement_start=kwargs.get("start", 1),
        pending_at_load_end=kwargs.get("end", 1),
        pending_after_drain=kwargs.get("pending", 0),
        elapsed=config.warmup + config.duration + 0.1,
    )


def test_defaults_offer_twelve_fair_camera_streams_and_report_tolerance():
    config = benchmark.BenchmarkConfig()
    assert config.target_rps == 15
    assert config.deadline == 15
    assert config.timeout == 15
    assert config.max_in_flight == 48
    result = report(config, records_for(config))
    assert result["passed"]
    assert result["required_completed_rps"] == pytest.approx(14.7)
    assert result["completed_rps"] == pytest.approx(15)
    assert all(row["scheduled"] == 150 for row in result["per_camera"])
    assert result["usage_completed_in_window"]["prompt_tokens"] == 180_000
    assert result["usage_completed_in_window"]["total_tokens"] is None


def test_completing_only_during_drain_cannot_claim_sustained_capacity():
    config = benchmark.BenchmarkConfig(cameras=1, fps=2, warmup=1, duration=2)
    records = records_for(config)
    for row in records:
        row["finished"] = 3.5
    result = report(config, records)
    assert result["outcomes_including_warmup_and_drain"] == {"ok": 6}
    assert result["completed_rps"] == 0
    assert not result["passed"]
    assert "completed_throughput_below_target" in result["failure_reasons"]


def test_latency_failure_rejections_errors_and_growing_queue_fail_independently():
    config = benchmark.BenchmarkConfig(cameras=1, fps=2, warmup=1, duration=2)
    records = records_for(config)
    assert report(config, records)["passed"]
    assert not report(replace(config, deadline=0.01), records)["passed"]
    growing = report(config, records, start=1, end=4)
    assert growing["queue_growth"] == 3
    assert "outstanding_requests_grew_during_measurement" in growing["failure_reasons"]
    assert not report(config, records, pending=1)["passed"]
    for outcome in ("rejected_in_flight_limit", "http_error", "invalid_response"):
        broken = [dict(row) for row in records]
        broken[0]["outcome"] = outcome  # Warmup failures are disclosed too.
        result = report(config, broken)
        assert result["errors_or_rejections"] == 1
        assert not result["passed"]


def test_cli_input_requires_local_endpoint_and_real_multimodal_request_shape(tmp_path):
    for endpoint in (
        "http://127.0.0.1:8000/v1/chat/completions",
        "http://[::1]:8000/v1/chat/completions",
        "http://localhost/v1",
    ):
        assert benchmark.validate_endpoint(endpoint) == endpoint
    with pytest.raises(ValueError, match="loopback"):
        benchmark.validate_endpoint("https://cloud.example/v1")
    assert benchmark.validate_endpoint("http://worker/v1", allow_remote=True)
    with pytest.raises(ValueError, match="credentials"):
        benchmark.validate_endpoint("http://user:password@localhost/v1")
    file = tmp_path / "requests.json"
    file.write_text(
        json.dumps([request_payload(), request_payload()]), encoding="utf-8"
    )
    assert len(benchmark.load_requests(file)) == 2
    for payload, expected in [
        ({"model": "local-vision", "messages": [{"content": "text only"}]}, "image"),
        ({**request_payload(), "stream": True}, "streaming"),
        ({**request_payload(), "model": ""}, "local model"),
    ]:
        file.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            benchmark.load_requests(file)
    payload = request_payload()
    payload["messages"][0]["content"][1]["image_url"]["url"] = (
        "https://external/image.png"
    )
    file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="embedded"):
        benchmark.load_requests(file)


def test_paced_mock_requests_keep_camera_order_and_actual_payload():
    received = []

    async def respond(request):
        received.append(
            (int(request.headers["X-Capacity-Camera"]), json.loads(request.content))
        )
        return httpx.Response(200, json=response_payload())

    config = benchmark.BenchmarkConfig(
        cameras=2,
        fps=5,
        warmup=0.1,
        duration=0.4,
        timeout=1,
        drain=1,
    )
    result = asyncio.run(
        benchmark.run_benchmark(
            config,
            [request_payload()],
            transport=httpx.MockTransport(respond),
        )
    )
    assert [camera for camera, _ in received] == [0, 1, 0, 1, 0]
    assert all(payload == request_payload() for _, payload in received)
    assert result["errors_or_rejections"] == 0
    assert result["pending_after_drain"] == 0
    assert result["distinct_request_templates"] == 1
    assert "cache" in result["workload_note"]


def test_overload_is_rejected_explicitly_and_pending_drain_is_bounded():
    async def blocked(_request):
        await asyncio.sleep(1)
        return httpx.Response(200, json=response_payload())

    config = benchmark.BenchmarkConfig(
        cameras=2,
        fps=5,
        warmup=0,
        duration=0.4,
        max_in_flight=1,
        timeout=2,
        drain=0.01,
    )
    result = asyncio.run(
        benchmark.run_benchmark(
            config,
            [request_payload()],
            transport=httpx.MockTransport(blocked),
        )
    )
    outcomes = result["outcomes_including_warmup_and_drain"]
    assert outcomes == {"drain_timeout": 1, "rejected_in_flight_limit": 3}
    assert result["pending_at_load_end"] == result["pending_after_drain"] == 1
    assert not result["passed"]


@pytest.mark.parametrize("failure", ["timeout", "http_error", "invalid_response"])
def test_failed_requests_never_count_as_completed_capacity(failure):
    async def respond(_request):
        if failure == "timeout":
            await asyncio.sleep(0.1)
        if failure == "http_error":
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={"choices": []})

    config = benchmark.BenchmarkConfig(
        cameras=1,
        fps=10,
        warmup=0,
        duration=0.04,
        timeout=0.01,
        drain=0.1,
    )
    result = asyncio.run(
        benchmark.run_benchmark(
            config,
            [request_payload()],
            transport=httpx.MockTransport(respond),
        )
    )
    assert result["outcomes_including_warmup_and_drain"] == {failure: 1}
    assert result["completed_in_window"] == 0
    assert not result["passed"]
