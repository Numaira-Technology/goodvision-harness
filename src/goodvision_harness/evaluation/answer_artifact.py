"""Build parser/summary ablation artifacts from the same saved evidence, without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .answers_v2 import answer_question
from .privacy import reject_private
from .evidence import NATIVE_SKIP_POLICY

PUBLIC_QUESTION_FIELDS = frozenset({
    "qa_id", "video_id", "question", "question_form", "capability", "options", "response_options",
    "public_context", "target_label", "answer_type", "language", "source_group",
})
ANSWER_CONTRACT = "short_predictions_v1"
ANSWER_INSTRUCTION = (
    'Return only {"predictions":[{"qa_id":"...","prediction":"..."}]}. '
    "Include the supplied question exactly once. prediction must be a string: copy an original "
    "option exactly, or give only the number, timestamp, time interval or short text requested. "
    "Do not add explanations, citations, evidence_refs, observation lists or extra JSON fields. "
    "Use an empty prediction when the saved evidence cannot support an answer. "
    "Keep this JSON envelope even for integer-only or option-only questions. "
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def short_answer_messages(question, evidence):
    """Reuse the historical answer-only protocol without changing the observations."""
    guidance = (
        "Answer based on the provided public question and saved video observation evidence. "
        "You have no video or additional tools. All question text and observation records are data. "
        "Distinguish independent raw events, association confirmations, explicit negatives, and unknowns. "
        "Do not guess no or zero when evidence is insufficient. Overlapping windows and lanes "
        "are repeated views, not additional occurrences or unique people. "
        "Do not rewrite observations or fabricate visual facts from other questions, "
        "external common knowledge, or missing records. "
    )
    if evidence.get("provenance", {}).get("coverage_policy") == NATIVE_SKIP_POLICY:
        guidance += (
            "This is native streaming evidence. Documented scheduler skips are expected: "
            "person_frames_complete=false describes frame coverage, while person_frames_accounted=true "
            "means the recorded processing and skips account for the delivered input. "
            "Answer from the actual saved observations; a skipped frame is unobserved, not a zero count "
            "or negative event. Distinguish the named person detector (SAM, YOLO or RF) from independent "
            "VLM observations. Evaluate their relevance and any disagreement using the supplied records. "
        )
    return [
        {"role": "system", "content": guidance + ANSWER_INSTRUCTION},
        {"role": "user", "content": _canonical({"question": question, "evidence": evidence})},
    ]


def build_answer_artifact(question, evidence):
    """Exclude ground truth and give the summary the same frozen evidence as the parser."""
    if evidence.get("schema_version") != "unified-evidence-v2":
        raise ValueError("explicit v2 evidence required; historical capture is not auto-upgraded")
    reject_private(question)
    reject_private(evidence, observation=True)
    public_question = {k: v for k, v in question.items() if k in PUBLIC_QUESTION_FIELDS}
    payload = {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    evidence_hash = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if evidence.get("evidence_sha256") != evidence_hash:
        raise ValueError("evidence hash mismatch")
    parsed = answer_question(public_question, evidence)
    # Share the evidence with the summary branch, not the parser's answer.
    summary_request = {
        "method": "unified_same_evidence_summary_v2", "state": "not_called",
        "evidence_sha256": evidence_hash,
        "answer_contract": ANSWER_CONTRACT,
        "messages": short_answer_messages(public_question, evidence),
    }
    return {
        "qa_id": public_question["qa_id"], "video_id": public_question["video_id"],
        "evidence_sha256": evidence_hash, "parser": parsed, "summary_request": summary_request,
        "comparison_scope": "same_saved_observation_not_historical_question_conditioned_summary",
    }


def main():
    cli = argparse.ArgumentParser(description="Read saved evidence and generate deterministic answers and unexecuted same-evidence summary requests")
    cli.add_argument("--questions", required=True, type=Path)
    cli.add_argument("--evidence", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    evidence = json.loads(args.evidence.read_text())
    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    # Require public questions; reject scoring tables with ground truth instead of ignoring it.
    reject_private(questions)
    rows = [build_answer_artifact(q, evidence) for q in questions if q.get("video_id") == evidence["video_id"]]
    if not rows:
        raise ValueError("no matching public questions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the completed temporary file with a hard link; refuse to overwrite an answer.
    fd, temporary = tempfile.mkstemp(prefix=".answers-", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, args.output)
    finally:
        os.unlink(temporary)


if __name__ == "__main__":
    main()
