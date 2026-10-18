"""Separate source skips from shared recovery; do not turn unknown errors into batch question failures."""
import httpx

from ..model_api import ModelAPIError
from .model_ledger import LedgerBlocked
from .offline_observer import ObservationFailure


def native_report_failure_policy(report):
    """Classify one archived window, never the worker's cumulative error count.

    One request's connection or provider error does not establish a shared
    outage. Unknown worker errors, credentials and ledger failures remain
    shared interruptions; they must not become terminal failures for every video.
    """
    code = report.get("error_code")
    if report.get("state") == "completed" and not code:
        return None
    status = report.get("timing", {}).get("http_status")
    local = status not in {401, 402, 403} and (
        code in {"timeout", "connection", "invalid_response", "empty_response",
                 "admission_timeout", "admission_superseded"}
        or code == "http_error" and type(status) is int and 400 <= status < 600
    )
    return {"scope": "window" if local else "shared", "code": code,
            "status_code": status,
            "action": "preserve_and_continue" if local else "interrupt_before_more_sources"}


def failure_policy(error):
    if isinstance(error, ObservationFailure):
        causes = [failure_policy(item) for item in error.failures]
        shared = next((row for row in causes if row["scope"] == "shared"), None)
        return {**(shared or causes[0]), "causes": causes}
    row = {"error_type": type(error).__name__, "code": getattr(error, "code", None),
           "status_code": getattr(error, "status_code", None),
           "scope": "shared", "action": "recover_before_next_source"}
    if isinstance(error, LedgerBlocked):
        row["scope"] = error.scope
    elif isinstance(error, ModelAPIError):
        if error.status_code not in {401, 402, 403} and error.code in {
            "timeout", "invalid_response", "empty_response", "overloaded"
        }:
            row["scope"] = "source"
        elif error.code == "http_error" and error.status_code in {400, 413, 422}:
            row["scope"] = "source"
    elif isinstance(error, httpx.HTTPStatusError):
        row["status_code"] = error.response.status_code
        if error.response.status_code in {400, 413, 422, 429}:
            row["scope"] = "source"
        if error.response.status_code in {500, 503} and "out of memory" in error.response.text.lower():
            row["scope"] = "source"
            row["code"] = "model_out_of_memory"
        elif error.response.status_code == 503 and "MODEL_OPERATION_TIMEOUT" in error.response.text:
            row["scope"] = "source"
            row["code"] = "model_operation_timeout"
    elif isinstance(error, httpx.ReadTimeout):
        row["scope"] = "source"
    if row["scope"] == "source":
        row["action"] = "preserve_and_skip_after_cleanup"
    return row
