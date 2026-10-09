"""Optional host/GPU and vLLM measurements, never fabricated per-lane timings."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from itertools import pairwise
from pathlib import Path
from threading import Event, Thread

import httpx

HISTOGRAMS = (
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:e2e_request_latency_seconds",
)
GAUGES = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
}
ALLOWED = GAUGES | {h + suffix for h in HISTOGRAMS for suffix in ("_sum", "_count")}
LINE = re.compile(r"^([\w:]+)(\{.*\})?\s+(\S+)(?:\s+\S+)?$")


def parse_metrics(text):
    """Keep individual series so a worker restart cannot hide in an aggregate."""
    result = {}
    for line in text.splitlines():
        match = LINE.match(line)
        if not match or match[1] not in ALLOWED:
            continue
        try:
            value = float(match[3])
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            result[match[1] + (match[2] or "")] = value
    return result


def server_deltas(snapshots):
    result = {}
    for name in HISTOGRAMS:
        count_prefix, sum_prefix = name + "_count", name + "_sum"
        if len(snapshots) < 2 or any(s is None for s in (snapshots[0], snapshots[-1])):
            result[name] = {"available": False, "reason": "missing_boundary_metrics"}
            continue
        if any(s is None for s in snapshots):
            result[name] = {"available": False, "reason": "sampling_gap"}
            continue
        valid = [s for s in snapshots if s is not None]
        keys = {k for k in valid[0] if k.split("{", 1)[0] in {count_prefix, sum_prefix}}
        if (
            not keys
            or not any(k.startswith(count_prefix) for k in keys)
            or not any(k.startswith(sum_prefix) for k in keys)
        ):
            result[name] = {"available": False, "reason": "not_exposed"}
            continue
        count_labels = {
            k[len(count_prefix) :] for k in keys if k.startswith(count_prefix)
        }
        sum_labels = {k[len(sum_prefix) :] for k in keys if k.startswith(sum_prefix)}
        if count_labels != sum_labels:
            result[name] = {"available": False, "reason": "unpaired_sum_count"}
            continue
        if any(
            {k for k in s if k.split("{", 1)[0] in {count_prefix, sum_prefix}} != keys
            for s in valid
        ):
            result[name] = {"available": False, "reason": "metric_series_changed"}
            continue
        if any(b[k] < a[k] for a, b in pairwise(valid) for k in keys):
            result[name] = {"available": False, "reason": "counter_reset"}
            continue
        count = sum(
            valid[-1][k] - valid[0][k] for k in keys if k.startswith(count_prefix)
        )
        seconds = sum(
            valid[-1][k] - valid[0][k] for k in keys if k.startswith(sum_prefix)
        )
        result[name] = {
            "available": True,
            "finished_request_count": count,
            "total_seconds": seconds,
            "mean_seconds": seconds / count if count else None,
            "scope": "whole model server during this stage; not per lane; not P95",
        }
    return result


class Telemetry:
    def __init__(self, output, metrics_url=None, api_key="", interval=1):
        self.output = Path(output)
        self.metrics_url, self.api_key, self.interval = metrics_url, api_key, interval
        self.stop = Event()
        self.samples = []
        self.thread = None

    def sample(self):
        row = {
            "monotonic_seconds": time.monotonic(),
            "server_metrics": None,
            "host_load_average": list(os.getloadavg())
            if hasattr(os, "getloadavg")
            else None,
        }
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            row["gpu_csv"] = completed.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            row["gpu_csv"] = None
        if self.metrics_url:
            try:
                headers = (
                    {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                )
                with httpx.Client(
                    timeout=2, trust_env=False, follow_redirects=False
                ) as client:
                    response = client.get(self.metrics_url, headers=headers)
                    response.raise_for_status()
                row["server_metrics"] = parse_metrics(response.text)
            except httpx.HTTPError:
                row["metrics_error"] = "metrics_unavailable"
        self.samples.append(row)
        with self.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")

    def __enter__(self):
        self.sample()

        def poll():
            while not self.stop.wait(self.interval):
                self.sample()

        self.thread = Thread(target=poll, name="vlm-benchmark-telemetry", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()

    def summary(self):
        return {
            "samples": len(self.samples),
            "gpu_sample_count": sum(s["gpu_csv"] is not None for s in self.samples),
            "server_metric_sample_count": sum(
                s["server_metrics"] is not None for s in self.samples
            ),
            "server_phases": server_deltas([s["server_metrics"] for s in self.samples]),
            "limits": "GPU telemetry is for the runner host. "
            "Server phases are aggregate means, "
            "not per-request/per-lane attribution. Use a dedicated server. "
            "Missing metrics are unavailable, not zero. "
            "Host load is not CPU utilization.",
        }
