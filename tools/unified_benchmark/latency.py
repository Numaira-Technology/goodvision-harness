"""One definition of benchmark request latency, shared by the runners and latency_bench.

A sample is one request attempt. Its time runs from sending the request to having the
parsed response; building the request, writing receipts, metadata lookups, queue and
retry backoff are outside it. P50/P95 use the nearest-rank method. Failed attempts are
never silently dropped: they are counted, timeouts separately, and a second P95 counts
every failed attempt at the request timeout.
"""

import math
import re

REQUEST_TIMEOUT_SECONDS = 90
PERCENTILE_METHOD = "nearest-rank: sorted ascending, value at index ceil(p * n) - 1"
_TIMEOUT = re.compile(r"timeout|timed out", re.IGNORECASE)


def nearest_rank(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(math.ceil(fraction * len(ordered)), 1) - 1]


def is_timeout(error):
    """True for a recorded error string that names a timeout."""
    return isinstance(error, str) and bool(_TIMEOUT.search(error))


def raised_timeout(error):
    """True for a timeout exception, including urllib's URLError wrapping one."""
    reason = getattr(error, "reason", None)
    return (
        isinstance(error, TimeoutError)
        or isinstance(reason, TimeoutError)
        or is_timeout(str(error))
    )


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def summary(successes, failed, timeouts, *, penalty_seconds=REQUEST_TIMEOUT_SECONDS):
    """successes: seconds of successful attempts; failed includes timeouts."""
    if timeouts > failed:
        raise ValueError("timeouts are a subset of failed attempts")
    successes = list(successes)
    return {
        "n": len(successes),
        "p50_seconds": nearest_rank(successes, 0.5),
        "p95_seconds": nearest_rank(successes, 0.95),
        "failed_attempts": failed,
        "timeout_attempts": timeouts,
        "p95_with_failures_seconds": nearest_rank(
            successes + [penalty_seconds] * failed, 0.95
        ),
        "failure_penalty_seconds": penalty_seconds,
        "percentile_method": PERCENTILE_METHOD,
    }
