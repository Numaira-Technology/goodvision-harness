"""Usage coverage and client-observed token rates, distinct from GPU decode speed."""

import math

from tools.vlm_benchmark.model_comparison import ratio
from tools.vlm_multistream_benchmark import timing


def distribution(values):
    values = [
        v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0
    ]
    return {"count": len(values), **timing(values)}


def token_metrics(records, seconds):
    attempted = [
        r for r in records if r["timing"].get("http_request_attempted") is True
    ]
    usage = [r["timing"].get("http_timing", {}) for r in attempted]
    result = {"http_attempted": len(attempted), "accounting_span_seconds": seconds}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_prompt_tokens",
        "cost_usd",
    ):
        values = [
            u[key]
            for u in usage
            if type(u.get(key)) in (int, float)
            and math.isfinite(u[key])
            and u[key] >= 0
        ]
        result[key] = {
            "reported_requests": len(values),
            "sum_reported": sum(values),
            "coverage": ratio(len(values), len(attempted)),
        }
        if key.endswith("tokens"):
            result[key].update(
                per_second=ratio(sum(values), seconds),
                per_minute=ratio(sum(values) * 60, seconds),
            )
    result["api_completion_tps"] = distribution(
        [
            u["completion_tokens"]
            / u.get("http_elapsed_seconds", u.get("total_seconds"))
            for u in usage
            if type(u.get("completion_tokens")) is int
            and u.get("http_elapsed_seconds", u.get("total_seconds", 0)) > 0
        ]
    )
    result["first_content_seconds"] = distribution(
        [u.get("stream_first_content_seconds") for u in usage]
    )
    result["visible_generation_tps_estimate"] = distribution(
        [
            u["visible_completion_tokens"] / u["stream_content_span_seconds"]
            for u in usage
            if type(u.get("visible_completion_tokens")) is int
            and u.get("stream_content_span_seconds", 0) > 0
            and u.get("stream_content_chunks", 0) > 1
        ]
    )
    result["scope"] = (
        "API completion TPS includes wait and reasoning tokens. Visible generation "
        "TPS is a client estimate, not engine decode TPS; omitted without explicit "
        "reasoning accounting or multiple timed chunks. Aggregate rates include "
        "drain and only reported usage; missing usage is unknown, not zero."
    )
    return result
