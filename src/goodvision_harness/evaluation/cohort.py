"""Generate parser answers and unexecuted summary requests from shared saved evidence.

This entry point makes no model calls, performs no scoring, and does not upgrade
historical Ashley captures into three-lane evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .answer_artifact import build_answer_artifact
from .answers_v2 import PARSER_VERSION
from .recording import publish_once
from .privacy import reject_private


def generate_cohort(*, questions_path, index_path, output):
    questions_path, index_path = Path(questions_path), Path(index_path)
    question_bytes, index_bytes = questions_path.read_bytes(), index_path.read_bytes()
    questions = [json.loads(line) for line in question_bytes.decode("utf-8").splitlines() if line.strip()]
    reject_private(questions)
    ids = [q["qa_id"] for q in questions]
    if not questions or any(not isinstance(qid, str) or not qid for qid in ids) or len(ids) != len(set(ids)):
        raise ValueError("nonempty cohort with unique question IDs required")
    videos = {q["video_id"] for q in questions}
    index = json.loads(index_bytes)
    if index.get("schema_version") != "unified-observation-index-v1":
        raise ValueError("explicit unified observation index required")
    method = index["method"]
    if not isinstance(method, str) or not method:
        raise ValueError("explicit method identity required")
    entries = index["observations"]
    processed = {row["video_id"]: row for row in index.get("processed_sources", [])}
    seen, evidence_by_video, sources = set(), {}, {}
    for entry in entries:
        vid = entry["video_id"]
        if vid not in videos or vid in seen:
            raise ValueError("index contains duplicate or out-of-cohort video")
        seen.add(vid)
        capture_path = Path(entry["capture_path"])
        if not capture_path.is_absolute():
            capture_path = index_path.parent / capture_path
        raw = capture_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["capture_file_sha256"]:
            raise ValueError("capture file hash mismatch")
        evidence = json.loads(raw)
        reject_private(evidence, observation=True)
        if evidence.get("schema_version") != "unified-evidence-v2" or evidence.get("video_id") != vid:
            raise ValueError("capture schema or video identity mismatch")
        if evidence.get("face_enabled") is not False:
            raise ValueError("face-disabled capture required")
        if evidence.get("provenance", {}).get("question_conditioned") is not False:
            raise ValueError("explicit question-independent observation required")
        evidence_by_video[vid] = evidence
        sources[vid] = {"capture_path": str(capture_path.resolve()),
                        "capture_file_sha256": entry["capture_file_sha256"],
                        "evidence_sha256": evidence.get("evidence_sha256")}
    # Use a new output directory and record input identities before publishing answers.
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "answers").mkdir()
    publish_once(output / "run.json", {
        "schema_version": "unified-cohort-artifacts-v1", "method": method,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions_sha256": hashlib.sha256(question_bytes).hexdigest(),
        "observation_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "questions": len(questions), "videos": len(videos), "parser_version": PARSER_VERSION,
        "new_model_calls": 0, "summary_execution": "not_called", "scoring": "not_run",
        "scope": "saved_observations_only_no_automatic_historical_upgrade",
    })
    result_rows = []
    for position, question in enumerate(questions):
        vid, qid = question["video_id"], question["qa_id"]
        evidence = evidence_by_video.get(vid)
        if evidence is None:
            prior = processed.get(vid)
            # Shared interruption is recoverable, not a terminal question failure.
            missing_status = ("interrupted" if prior and prior.get("status") == "shared_failure"
                              else "request_failure" if prior else "not_started")
            artifact = {"qa_id": qid, "video_id": vid, "method": method,
                        "parser": {"qa_id": qid, "status": missing_status, "final_answer": None,
                                   "reason": "source_processed_without_complete_capture" if prior else "no_saved_observation",
                                   "source_outcome": prior, "parser_version": PARSER_VERSION,
                                   "evidence_refs": []},
                        "summary_request": {"state": "not_called", "reason": "no_saved_observation"}}
        else:
            artifact = build_answer_artifact(question, evidence)
            artifact.update(method=method, observation_source=sources[vid])
        # Keep question IDs in the content; do not interpret them as filename paths.
        path = f"answers/{position + 1:04d}-{hashlib.sha256(qid.encode()).hexdigest()[:16]}.json"
        digest = publish_once(output / path, artifact)
        result_rows.append({"qa_id": qid, "video_id": vid, "capability": question.get("capability"),
                            "question_form": question.get("question_form"), "answer_path": path,
                            "answer_file_sha256": digest, "status": artifact["parser"]["status"]})
    totals = dict(Counter(row["status"] for row in result_rows))
    per_capability = {
        str(capability): dict(Counter(row["status"] for row in result_rows if row["capability"] == capability))
        for capability in sorted({row["capability"] for row in result_rows}, key=str)
    }
    completion = {"method": method, "questions": len(questions), "states": totals,
                  "capability_states": per_capability, "parser_version": PARSER_VERSION,
                  "rows": result_rows, "new_model_calls": 0, "summary_execution": "not_called",
                  "scoring": "not_run", "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    publish_once(output / "completion.json", completion)
    return completion


def main():
    cli = argparse.ArgumentParser(description="Generate answers from saved three-lane evidence without model calls or scoring")
    cli.add_argument("--questions", type=Path, required=True)
    cli.add_argument("--index", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    result = generate_cohort(questions_path=args.questions, index_path=args.index, output=args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
