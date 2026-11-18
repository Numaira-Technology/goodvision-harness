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
)
_LINE = re.compile(r"^([^\s{]+)(\{.*\})?\s+(\S+)(?:\s+\S+)?$")


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
            value = float(match.group(3))
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        totals[match.group(1) + (match.group(2) or "")] = value
    return totals


def fetch_metrics(url: str, timeout: float = 5) -> str:
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _series(parsed, name):
    return {key: value for key, value in parsed.items() if key.split("{", 1)[0] == name}


def _pair(parsed: dict, name: str):
    counts = _series(parsed, name + "_count")
    sums = _series(parsed, name + "_sum")
    if not counts or {key[len(name + "_count") :] for key in counts} != {
        key[len(name + "_sum") :] for key in sums
    }:
        return None
    return {
        "count": sum(counts.values()),
        "sum": sum(sums.values()),
        "series": counts | sums,
    }


def _series_error(before, after):
    if not before or not after:
        return "not_exposed"
    if before.keys() != after.keys():
        return "metric_series_changed"
    if any(after[key] < value for key, value in before.items()):
        return "counter_reset"
    return None


def snapshot_from_text(text: str) -> dict:
    parsed = parse_metrics(text)
    out = {}
    for key, name in HISTOGRAMS:
        pair = _pair(parsed, name)
        out[key] = pair if pair is not None else "unknown"
    preemptions = "unknown"
    for name in PREEMPTION_NAMES:
        series = _series(parsed, name)
        if series:
            preemptions = sum(series.values())
            out["preemption_series"] = series
            break
    out["preemptions"] = preemptions
    return out


def _delta(before, after):
    if before == "unknown" or after == "unknown":
        return {"available": False, "reason": "not_exposed"}
    reason = _series_error(before.get("series"), after.get("series"))
    if reason:
        return {"available": False, "reason": reason}
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
    reason = _series_error(
        before.get("preemption_series"), after.get("preemption_series")
    )
    if reason:
        out["preemptions"] = {"available": False, "reason": reason}
    elif left == "unknown" or right == "unknown":
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
