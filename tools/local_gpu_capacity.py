#!/usr/bin/env python3
"""Measure a local VLM with paced camera arrivals; no model quality claim.

Supply production-like, non-streaming chat requests with embedded image data.
An array rotates different frames/prompts, which reduces repeated-input cache bias.
Default load is 12 cameras at 1.25 requests/second each. Warmup uses the same
load. Completion throughput is measured before drain, so a growing backlog
cannot pass merely because the server eventually finishes every request.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True)
class BenchmarkConfig:
    endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    cameras: int = 12
    fps: float = 1.25
    duration: float = 120.0
    warmup: float = 10.0
    max_in_flight: int = 48
    timeout: float = 15.0
    drain: float = 30.0
    deadline: float = 15.0
    min_throughput_ratio: float = 0.98
    max_queue_growth: int = 1

    @property
    def target_rps(self) -> float:
        return self.cameras * self.fps

    def validate(self) -> None:
        for name in ("cameras", "max_in_flight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_queue_growth, bool)
            or not isinstance(self.max_queue_growth, int)
            or self.max_queue_growth < 0
        ):
            raise ValueError("max_queue_growth must be a non-negative integer")
        if not 0 < self.min_throughput_ratio <= 1:
            raise ValueError("min_throughput_ratio must be between 0 and 1")
        for name in ("fps", "duration", "timeout", "drain", "deadline", "warmup"):
            value = getattr(self, name)
            if (
                not math.isfinite(value)
                or value < 0
                or (name != "warmup" and not value)
            ):
                raise ValueError(
                    f"{name} must be finite and positive (warmup may be 0)"
                )


def validate_endpoint(endpoint: str, *, allow_remote: bool = False) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an HTTP(S) chat-completions URL")
    if parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise ValueError("endpoint must not contain credentials, query, or fragment")
    try:
        local = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = parsed.hostname.lower() == "localhost"
    if not local and not allow_remote:
        raise ValueError("endpoint must be loopback; use --allow-remote explicitly")
    return endpoint


def load_requests(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    requests = value if isinstance(value, list) else [value]
    if not requests:
        raise ValueError("request file must contain at least one chat request")
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"request {index} must be a JSON object")
        if not isinstance(request.get("model"), str) or not request["model"].strip():
            raise ValueError(f"request {index} must specify its local model")
        if request.get("stream", False) is not False:
            raise ValueError("streaming requests are not supported")
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"request {index} requires chat messages")
        images = 0
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("chat messages must be objects")
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image = part.get("image_url")
                url = image.get("url") if isinstance(image, dict) else image
                if not isinstance(url, str) or not url.startswith("data:image/"):
                    raise ValueError(
                        "use embedded image data URLs for offline requests"
                    )
                if ";base64," not in url or not url.split(";base64,", 1)[1].strip():
                    raise ValueError("image data URL must contain base64 image data")
                images += 1
        if not images:
            raise ValueError(f"request {index} needs a representative image input")
    return requests


def arrival_count(seconds: float, rate: float) -> int:
    return math.ceil(seconds * rate - 1e-9)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(
    config: BenchmarkConfig,
    records: list[dict[str, Any]],
    *,
    pending_at_measurement_start: int,
    pending_at_load_end: int,
    pending_after_drain: int,
    elapsed: float,
) -> dict[str, Any]:
    begin, end = config.warmup, config.warmup + config.duration
    arrivals = [row for row in records if begin <= row["scheduled"] < end]
    completed = [
        row
        for row in records
        if row["outcome"] == "ok" and begin <= row["finished"] < end
    ]
    outcomes = Counter(row["outcome"] for row in records)
    latencies = [
        row["finished"] - row["scheduled"] for row in arrivals if row["outcome"] == "ok"
    ]
    errors = sum(count for outcome, count in outcomes.items() if outcome != "ok")
    completed_rps = len(completed) / config.duration
    required_rps = config.target_rps * config.min_throughput_ratio
    queue_growth = pending_at_load_end - pending_at_measurement_start
    p95 = percentile(latencies, 0.95)
    failures = []
    if len(arrivals) != arrival_count(end, config.target_rps) - arrival_count(
        begin,
        config.target_rps,
    ):
        failures.append("scheduled_arrivals_missing")
    if completed_rps + 1e-9 < required_rps:
        failures.append("completed_throughput_below_target")
    if queue_growth > config.max_queue_growth:
        failures.append("outstanding_requests_grew_during_measurement")
    if errors:
        failures.append("requests_failed_or_rejected")
    if pending_after_drain:
        failures.append("requests_pending_after_drain")
    if p95 is None or p95 > config.deadline:
        failures.append("p95_deadline_exceeded")
    usage_rows = [
        row["usage"] for row in completed if isinstance(row.get("usage"), dict)
    ]
    usage: dict[str, int | None] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [
            row[name]
            for row in usage_rows
            if isinstance(row.get(name), int)
            and not isinstance(row[name], bool)
            and row[name] >= 0
        ]
        usage[name] = sum(values) if values else None
        usage[name + "_reported_requests"] = len(values)
    per_camera = []
    for camera in range(config.cameras):
        started = [row for row in arrivals if row["camera"] == camera]
        finished = [row for row in completed if row["camera"] == camera]
        per_camera.append(
            {
                "camera": camera,
                "scheduled": len(started),
                "rejected": sum(
                    row["outcome"].startswith("rejected") for row in started
                ),
                "completed_in_window": len(finished),
                "completed_rps": len(finished) / config.duration,
            }
        )
    return {
        "scope": "VLM capacity only; excludes model accuracy and tracking capacity",
        "passed": not failures,
        "failure_reasons": failures,
        "config": asdict(config),
        "target_rps": config.target_rps,
        "required_completed_rps": required_rps,
        "scheduled_in_window": len(arrivals),
        "completed_in_window": len(completed),
        "completed_rps": completed_rps,
        "latency_seconds": {
            "p50": percentile(latencies, 0.50),
            "p95": p95,
            "max": max(latencies, default=None),
            "basis": "scheduled arrival to response, including scheduler delay",
        },
        "outcomes_including_warmup_and_drain": dict(outcomes),
        "errors_or_rejections": errors,
        "pending_at_measurement_start": pending_at_measurement_start,
        "pending_at_load_end": pending_at_load_end,
        "queue_growth": queue_growth,
        "pending_after_drain": pending_after_drain,
        "elapsed_including_warmup_and_drain": elapsed,
        "usage_completed_in_window": usage,
        "per_camera": per_camera,
    }


async def run_benchmark(
    config: BenchmarkConfig,
    requests: list[dict[str, Any]],
    *,
    api_key: str = "",
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    config.validate()
    if not requests:
        raise ValueError("at least one representative request is required")
    records: list[dict[str, Any]] = []
    tasks: set[asyncio.Task[None]] = set()
    start = time.perf_counter()
    end = config.warmup + config.duration
    pending_at_measurement_start: int | None = None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(
        headers=headers,
        timeout=config.timeout,
        transport=transport,
        limits=httpx.Limits(max_connections=config.max_in_flight),
        trust_env=False,
        follow_redirects=False,
    ) as client:

        async def send(row: dict[str, Any], payload: dict[str, Any]) -> None:
            row["started"] = time.perf_counter() - start
            try:
                response = await asyncio.wait_for(
                    client.post(
                        config.endpoint,
                        json=payload,
                        headers={
                            "X-Capacity-Camera": str(row["camera"]),
                            "X-Capacity-Sequence": str(row["sequence"]),
                        },
                    ),
                    timeout=config.timeout,
                )
                row["http_status"] = response.status_code
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") if isinstance(data, dict) else None
                content = (
                    choices[0].get("message", {}).get("content")
                    if isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                    else None
                )
                if not isinstance(content, str) or not content.strip():
                    row["outcome"] = "invalid_response"
                else:
                    row["outcome"] = "ok"
                    row["usage"] = data.get("usage")
            except (asyncio.TimeoutError, httpx.TimeoutException):
                row["outcome"] = "timeout"
            except httpx.HTTPStatusError:
                row["outcome"] = "http_error"
            except asyncio.CancelledError:
                row["outcome"] = "drain_timeout"
            except (ValueError, AttributeError, TypeError, httpx.HTTPError):
                row["outcome"] = "request_error"
            finally:
                row["finished"] = time.perf_counter() - start

        for sequence in range(arrival_count(end, config.target_rps)):
            scheduled = sequence / config.target_rps
            await asyncio.sleep(max(0.0, start + scheduled - time.perf_counter()))
            if scheduled >= config.warmup and pending_at_measurement_start is None:
                pending_at_measurement_start = sum(not task.done() for task in tasks)
            row = {
                "sequence": sequence,
                "camera": sequence % config.cameras,
                "scheduled": scheduled,
                "outcome": "pending",
            }
            records.append(row)
            now = time.perf_counter() - start
            if now >= end:
                row.update(outcome="rejected_scheduler_late", finished=now)
                continue
            if sum(not task.done() for task in tasks) >= config.max_in_flight:
                row.update(outcome="rejected_in_flight_limit", finished=now)
                continue
            task = asyncio.create_task(send(row, requests[sequence % len(requests)]))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        await asyncio.sleep(max(0.0, start + end - time.perf_counter()))
        pending_at_load_end = sum(not task.done() for task in tasks)
        pending_after_drain = 0
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=config.drain)
            pending_after_drain = len(pending)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    result = summarize(
        config,
        records,
        pending_at_load_end=pending_at_load_end,
        pending_at_measurement_start=pending_at_measurement_start or 0,
        pending_after_drain=pending_after_drain,
        elapsed=time.perf_counter() - start,
    )
    distinct_templates = []
    for request in requests:
        if not any(request == previous for previous in distinct_templates):
            distinct_templates.append(request)
    result["distinct_request_templates"] = len(distinct_templates)
    result["workload_note"] = (
        "Repeated request; prefix/image caches may overstate diverse camera throughput."
        if len(distinct_templates) == 1
        else "Templates rotate; use production prompts, image sizes, and output limits."
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--endpoint", default=BenchmarkConfig.endpoint)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--api-key-env", default="LOCAL_INFERENCE_API_KEY")
    parser.add_argument("--cameras", type=int, default=12)
    parser.add_argument("--fps", type=float, default=1.25)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--warmup", type=float, default=10)
    parser.add_argument("--max-in-flight", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--drain", type=float, default=30)
    parser.add_argument("--deadline", type=float, default=15)
    parser.add_argument(
        "--min-throughput-ratio",
        type=float,
        default=0.98,
        help="minimum completed / target rate; 0.98 allows window jitter",
    )
    parser.add_argument(
        "--max-queue-growth",
        type=int,
        default=1,
        help="allowed increase in outstanding requests during measurement",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        endpoint = validate_endpoint(args.endpoint, allow_remote=args.allow_remote)
        config = BenchmarkConfig(
            **{
                name: endpoint if name == "endpoint" else getattr(args, name)
                for name in BenchmarkConfig.__dataclass_fields__
            }
        )
        config.validate()
        requests = load_requests(args.request_file)
        result = asyncio.run(
            run_benchmark(
                config,
                requests,
                api_key=os.environ.get(args.api_key_env, ""),
            )
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
