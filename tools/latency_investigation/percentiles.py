"""Ceil-rank percentiles matching the formal benchmark's P50/P95 definition.

The rank is ``ceil(fraction * n) - 1`` on the sorted sample, the same rule as
``tools.unified_benchmark.workflow.timings``. This module has no P99.
"""

from __future__ import annotations

import math


def percentile(values, fraction: float):
    ordered = sorted(
        value
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
    if not ordered:
        return None
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[index]


def p50(values):
    return percentile(values, 0.50)


def p95(values):
    return percentile(values, 0.95)


def ratio(p95_value, p50_value):
    if (
        not isinstance(p95_value, (int, float))
        or not isinstance(p50_value, (int, float))
        or isinstance(p95_value, bool)
        or isinstance(p50_value, bool)
        or p50_value == 0
        or not math.isfinite(p95_value)
        or not math.isfinite(p50_value)
    ):
        return None
    return p95_value / p50_value


def token_bucket(prompt_tokens) -> str:
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 0
    ):
        return "unknown"
    if prompt_tokens < 500:
        return "<500"
    if prompt_tokens <= 999:
        return "500-999"
    return ">=1000"
