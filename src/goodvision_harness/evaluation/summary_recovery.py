"""Explicit, bounded recovery of historical final-answer failures.

Authorization names old calls, not a new payable identity. Old rows and costs
remain immutable; all old sends still count toward the three-send ceiling.
"""
from __future__ import annotations

from pathlib import Path

from .model_operation import operation_key, question_for_request, validate_operation


SCHEMA = "explicit-failed-summary-recovery-v1"


def recovery_calls(approval, *, requests, responses, terminal, operations, root):
    """Return only the historical calls explicitly superseded for dispatch.

    A subsequent call in this same run keeps ordinary cache/unknown protection.
    Reusing the authorization from another run cannot ignore that later call.
    """
    if approval is None:
        return set(), {}
    if (approval.get("schema_version") != SCHEMA
            or approval.get("max_cumulative_sends") != 3
            or not approval.get("authorization") or not approval.get("questions")):
        raise ValueError("Explicit failed-answer recovery authorization is required")
    ignored, selections = set(), {}
    for entry in approval["questions"]:
        qid, prior = entry["qa_id"], entry["prior_call_ids"]
        operation = validate_operation(entry["operation"])
        if (operation["kind"] != "summary" or operation["question"] != qid
                or operation["video"] != entry["video_id"] or qid in selections
                or not isinstance(prior, list) or len(prior) not in {1, 2}
                or len(set(prior)) != len(prior)):
            raise ValueError("Recovery selection has invalid scope or exhausted sends")
        key = operation_key(operation)
        scoped = {call for call, row in requests.items()
                  if (operations[call] or {}).get("question") == qid
                  or row.get("context", {}).get("qa_id") == qid}
        if not set(prior) <= scoped:
            raise ValueError("Authorized historical calls are missing from the cumulative ledger")
        for call in scoped:
            if operations[call] != operation:
                raise ValueError("Recovery cannot switch the saved observation operation")
            if question_for_request(requests[call]["request"]) != question_for_request(entry):
                raise ValueError("Recovery cannot change the original public question")
            if call not in prior:
                if (Path(call).parent != root
                        or requests[call].get("context", {}).get("recovery_authorization")
                        != approval["authorization"]):
                    raise ValueError("A later answer dispatch is not covered by this authorization")
                continue
            receipt = responses.get(call, {})
            response, error = receipt.get("response") or {}, receipt.get("error") or {}
            truncated = any(choice.get("finish_reason") == "length"
                            for choice in response.get("choices", []))
            timeout = error.get("code") == "timeout" and error.get("status_code") is None
            if (terminal.get(call) not in {"recorded", "outcome_unknown"}
                    or not (truncated or error.get("status_code") in {400, 429}
                            or timeout and entry.get("allow_unknown_timeout") is True)):
                raise ValueError("Recovery selected an unapproved or still in-flight response")
        selections[qid] = {**entry, "operation_key": key,
                           "authorization": approval["authorization"]}
        ignored.update(prior)
    return ignored, selections
