"""Reuse lossless benchmark encodings for oversized same-evidence answers."""
from __future__ import annotations

import json

from ..benchmark_evidence_encoding import (
    compact_repeated_evidence,
    compact_schema_evidence,
)


def message_bytes(messages):
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()


def prepare_summary_messages(messages, *, max_input_bytes):
    """Keep questions and observations intact; select only a smaller representation."""
    original_bytes = len(message_bytes(messages))
    if original_bytes <= max_input_bytes:
        return messages, None

    payload = json.loads(messages[1]["content"])
    evidence = payload["evidence"]
    selected, selected_bytes, encoding = messages, original_bytes, "original"
    candidates = []
    for compact in (compact_repeated_evidence, compact_schema_evidence):
        try:
            candidate_evidence = compact(evidence)
        except ValueError as exc:
            # Reserved marker collisions retain the preceding lossless representation.
            candidates.append({"encoder": compact.__name__, "status": "retained_previous",
                               "reason": str(exc)})
            continue
        evidence = candidate_evidence
        candidate = [messages[0], {**messages[1], "content": json.dumps(
            {**payload, "evidence": evidence}, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False)}]
        size = len(message_bytes(candidate))
        candidates.append({"encoding": evidence["encoding"], "input_bytes": size})
        if size < selected_bytes:
            selected, selected_bytes, encoding = candidate, size, evidence["encoding"]
        if selected_bytes <= max_input_bytes:
            break

    return selected, {
        "method": "lossless_answer_input_v1",
        "encoding": encoding,
        "original_input_bytes": original_bytes,
        "prepared_input_bytes": selected_bytes,
        "max_input_bytes": max_input_bytes,
        "records_omitted": 0,
        "question_unchanged": True,
        "truncated": False,
        "candidates": candidates,
    }
