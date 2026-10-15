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

PUBLIC_QUESTION_FIELDS = frozenset({
    "qa_id", "video_id", "question", "question_form", "capability", "options", "response_options",
    "public_context", "target_label", "answer_type", "language", "source_group",
})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
        "messages": [
            {"role": "system", "content": (
                "根据提供的公开问题和已保存视频观察证据回答。没有视频或额外工具。"
                "区分独立原始事件、关联确认、明确阴性和未知；未完成或证据不足不能猜成no或零。"
                "返回JSON：final_answer（字符串或null）、status（answered/insufficient_evidence/"
                "incomplete_video/request_failure/unsupported）、evidence_refs。"
                "不要改写观察，也不要从其他题目、外部常识或缺失记录补造视觉事实。")},
            {"role": "user", "content": _canonical({"question": public_question, "evidence": evidence})},
        ],
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
