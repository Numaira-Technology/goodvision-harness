"""Explicit local operation scopes, independent of prompt representation."""
from __future__ import annotations

import json

from ..benchmark_evidence_encoding import expand_repeated_evidence
from .summary_encoding import expand_window_evidence


def summary_operation(evidence, qa_id):
    return validate_operation({
        "kind": "summary", "session": evidence.get("stream_session_id"),
        "source": evidence.get("source_id"), "video": evidence.get("video_id"),
        "question": qa_id,
    })


def validate_operation(operation):
    """Reject incomplete scopes instead of treating them as new payable work."""
    if not isinstance(operation, dict):
        raise ValueError("Explicit model operation required")
    kind = operation.get("kind")
    fields = ({"kind", "session", "source", "video", "question"} if kind == "summary"
              else {"kind", "session", "source", "lane", "frames"} if kind == "window"
              else set())
    if not fields or set(operation) != fields:
        raise ValueError("Unsupported model operation scope")
    if any(not isinstance(operation[key], str) or not operation[key]
           for key in fields - {"frames"}):
        raise ValueError("Model operation scope contains an empty identity")
    if kind == "window":
        frames = operation["frames"]
        if (not isinstance(frames, list) or not frames
                or any(type(frame) is not int or frame < 0 for frame in frames)
                or any(left >= right for left, right in zip(frames, frames[1:]))):
            raise ValueError("Model window requires ordered source frame sequences")
    return json.loads(json.dumps(operation, allow_nan=False))


def operation_key(operation):
    """Use the small structured scope itself, not a content-derived identifier."""
    return json.dumps(validate_operation(operation), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def question_for_request(request):
    """Compare answer-relevant question content across old and compact prompts."""
    question = request.get("public_question")
    if question is None:
        question = json.loads(request["messages"][1]["content"])["question"]
    return {key: value for key, value in question.items()
            if key not in {"qa_id", "video_id", "source_group"}}


def operation_for_request(context, request):
    """Read new explicit scope or recover it from an archived legacy request.

    Legacy source/session names remain opaque historical values. No digest is
    calculated or compared. Unrecoverable scope is handled by the ledger as an
    unresolved legacy operation, never as a fresh request.
    """
    if "operation" in context:
        operation = validate_operation(context["operation"])
        checks = ({"qa_id": "question", "video_id": "video"} if operation["kind"] == "summary"
                  else {"stream_session_id": "session", "source_id": "source",
                        "lane": "lane", "source_frame_sequences": "frames"})
        if any(key in context and context[key] != operation[field] for key, field in checks.items()):
            raise ValueError("Model operation disagrees with its request context")
        return operation
    if context.get("phase") == "same_evidence_summary":
        messages = request["messages"]
        payload = json.loads(messages[1]["content"])
        evidence = expand_window_evidence(expand_repeated_evidence(payload["evidence"]))
        question = payload["question"]
        if (question.get("qa_id") != context.get("qa_id")
                or question.get("video_id") != context.get("video_id")
                or evidence.get("video_id") != context.get("video_id")):
            raise ValueError("Legacy answer scope does not match its saved question")
        return summary_operation(evidence, context["qa_id"])
    return validate_operation({
        "kind": "window", "session": context.get("stream_session_id"),
        "source": context.get("source_id"), "lane": context.get("lane"),
        "frames": context.get("source_frame_sequences"),
    })
