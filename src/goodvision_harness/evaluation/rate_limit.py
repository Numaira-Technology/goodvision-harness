"""Explicit provider cooldown, separate from model latency and retry budgets."""
import math

from ..model_api import ModelAPIError


def wait_after_429(error, *, ledger, context, receipt, clock, stop=None):
    """Record every cooldown, including the exhausted attempt, before more work.

    Long provider waits end this window's retry sequence after cooling down;
    they never cause an early retry that violates Retry-After.
    """
    retry_after = getattr(error, "retry_after_seconds", None)
    if retry_after is None:
        retry_after = getattr(error, "evaluation_diagnostic", {}).get("retry_after_seconds")
    if type(retry_after) not in (int, float) or not math.isfinite(retry_after):
        retry_after = 0.0
    delay = max(60.0, retry_after)
    receipt = receipt or {}
    row = {"kind": "vlm_rate_limit_backoff", **context,
           "call_id": receipt.get("call_id"), "operation": receipt.get("operation"),
           "seconds": delay, "started_epoch": clock.time(), "same_job_and_parameters": True}
    with ledger.lock:
        ledger._append("rate-limit-backoff.jsonl", row)
    waiting = clock.monotonic()
    if stop is None:
        clock.sleep(delay)
    elif stop.wait(delay):
        raise ModelAPIError("evaluation backoff stopped", code="cancelled")
    return clock.monotonic() - waiting, delay <= 300.0
