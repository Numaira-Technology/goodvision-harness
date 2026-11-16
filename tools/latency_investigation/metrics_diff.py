"""Before/after snapshot of the vLLM ``/metrics`` series used for this study.

The diff is count and sum (mean = sum/count) for queue time, time to first
token, prefill, and decode, plus the preemption counter. It is not a P95.
A counter that goes backwards is reported as unknown rather than a negative
delta. Missing series stay unknown rather than 0.
"""

from __future__ import annotations

import math
import re
import urllib.request

HISTOGRAMS = (
    ("queue_time_seconds", "vllm:request_queue_time_seconds"),
    ("ttft_seconds", "vllm:time_to_first_token_seconds"),
    ("prefill_seconds", "vllm:request_prefill_time_seconds"),
    ("decode_seconds", "vllm:request_decode_time_seconds"),
)
PREEMPTION_NAMES = (
    "vllm:num_preemptions_total",
    "vllm:num_preemptions",
    "vllm:num_preemptions_created",
)
_LINE = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+(\S+)(?:\s+\S+)?$")


def parse_metrics(text: str) -> dict:
    totals: dict[str, float] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        totals[match.group(1)] = totals.get(match.group(1), 0.0) + value
    return totals


def fetch_metrics(url: str, timeout: float = 5) -> str:
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _pair(parsed: dict, name: str):
    count = parsed.get(name + "_count")
    total = parsed.get(name + "_sum")
    if count is None or total is None:
        return None
    return {"count": count, "sum": total}


def snapshot_from_text(text: str) -> dict:
    parsed = parse_metrics(text)
    out = {}
    for key, name in HISTOGRAMS:
        pair = _pair(parsed, name)
        out[key] = pair if pair is not None else "unknown"
    preemptions = "unknown"
    for name in PREEMPTION_NAMES:
        if name in parsed:
            preemptions = parsed[name]
            break
    out["preemptions"] = preemptions
    return out


def _delta(before, after):
    if before == "unknown" or after == "unknown":
        return {"available": False, "reason": "not_exposed"}
    if after["count"] < before["count"] or after["sum"] < before["sum"]:
        return {"available": False, "reason": "counter_reset"}
    count = after["count"] - before["count"]
    total = after["sum"] - before["sum"]
    mean = total / count if count else None
    return {
        "available": True,
        "count_delta": count,
        "sum_seconds": total,
        "mean_seconds": mean,
    }


def diff_snapshots(before: dict, after: dict) -> dict:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {"available": False, "reason": "missing_snapshot"}
    out = {}
    for key, _name in HISTOGRAMS:
        out[key] = _delta(before.get(key, "unknown"), after.get(key, "unknown"))
    left = before.get("preemptions", "unknown")
    right = after.get("preemptions", "unknown")
    if left == "unknown" or right == "unknown":
        out["preemptions"] = {"available": False, "reason": "not_exposed"}
    elif not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        out["preemptions"] = {"available": False, "reason": "not_exposed"}
    elif right < left:
        out["preemptions"] = {"available": False, "reason": "counter_reset"}
    else:
        out["preemptions"] = {
            "available": True,
            "before": left,
            "after": right,
            "delta": right - left,
        }
    return out
