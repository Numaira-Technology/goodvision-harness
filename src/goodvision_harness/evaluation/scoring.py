"""Score immutable three-lane answers locally using the unchanged, hash-pinned PR63 scorer.

Ground truth is read only here. This module does not import or call model clients.
Terminal failures contribute zero quality, never a fabricated numeric prediction.
Pending questions prevent publication of a completed full-cohort score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

from .recording import publish_once

SCORER_SHA256 = "956982bd59015fbd8591c9e8e5e5c58397189ab4d9e908428668094a84c53d43"
PENDING = frozenset({"not_started", "interrupted", "shared_failure"})
TERMINAL = frozenset({"answered", "insufficient_evidence", "unsupported", "invalid_answer",
                      "request_failure", "incomplete_video", "skipped", "not_called"})


def _read(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"input file hash mismatch: {Path(path).name}")
    return raw


def _jsonl(path, expected):
    values = [json.loads(line) for line in _read(path, expected).splitlines() if line.strip()]
    result = {row["qa_id"]: row for row in values}
    if not values or len(result) != len(values):
        raise ValueError("nonempty unique question IDs required")
    return result


def _aggregate(rows):
    pending = sum(row["score_state"] == "pending" for row in rows)
    totals = {"target": len(rows), "terminal": len(rows) - pending, "pending": pending,
              "states": dict(Counter(row["status"] for row in rows)),
              "overall_score_100": None if pending or not rows else
                  100 * sum(row["metrics"]["Overall_Score"] for row in rows) / len(rows),
              "overall_is_accuracy": False, "metrics": {}}
    names = sorted({name for row in rows for name in row["metrics"]} - {"Overall_Score"})
    for name in names:
        applicable = [row for row in rows if name in row["metrics"]]
        values = [row["metrics"][name] for row in applicable if row["metrics"][name] is not None]
        detail = {"denominator": len(applicable), "numeric_answers": len(values),
                  "mean": sum(values) / len(applicable) if len(values) == len(applicable) else None}
        if name.endswith("_MAE") and len(values) != len(applicable):
            detail["reason"] = "full_cohort_MAE_unavailable_due_to_missing_numeric_answers"
        if name.endswith("_Accuracy") or name == "Counting_Strict_Equality":
            detail.update(correct=sum(value == 1 for value in values),
                          percent=100 * sum(values) / len(applicable) if len(values) == len(applicable) else None)
        totals["metrics"][name] = detail
    return totals


def score_cohort(*, questions_path, questions_sha256, truth_path, truth_sha256,
                 cohort_path, cohort_sha256, mode, output, scorer_path, summary_root=None):
    if mode not in {"rule", "model"}:
        raise ValueError("mode must be rule or model")
    if mode == "model" and summary_root is None:
        raise ValueError("model scoring requires the final-answer directory")
    scorer_path = Path(scorer_path).resolve()
    _read(scorer_path, SCORER_SHA256)
    spec = importlib.util.spec_from_file_location("unified_pr63_scoring", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    questions = _jsonl(questions_path, questions_sha256)
    truths = _jsonl(truth_path, truth_sha256)
    if not questions.keys() <= truths.keys():
        raise ValueError("missing private ground truth")
    cohort_path = Path(cohort_path).resolve()
    cohort = json.loads(_read(cohort_path, cohort_sha256))
    indexed = {row["qa_id"]: row for row in cohort["rows"]}
    if len(indexed) != len(cohort["rows"]) or indexed.keys() != questions.keys():
        raise ValueError("cohort must retain every question exactly once")
    run = json.loads((cohort_path.parent / "run.json").read_bytes())
    if run["questions_sha256"] != questions_sha256:
        raise ValueError("cohort was generated from different questions")
    rows = []
    for qid, question in questions.items():
        entry = indexed[qid]
        artifact_path = (cohort_path.parent / entry["answer_path"]).resolve()
        if not artifact_path.is_relative_to(cohort_path.parent):
            raise ValueError("answer path escapes cohort directory")
        artifact = json.loads(_read(artifact_path, entry["answer_file_sha256"]))
        if (artifact["qa_id"] != qid or artifact["video_id"] != question["video_id"]
                or entry["video_id"] != question["video_id"]):
            raise ValueError("answer/question identity mismatch")
        answer, source = artifact["parser"], artifact_path
        if mode == "model":
            source = Path(summary_root) / artifact_path.stem / "outcome.json"
            if source.exists():
                answer = json.loads(source.read_bytes())
                if (answer["qa_id"] != qid or answer["video_id"] != question["video_id"]
                        or answer.get("artifact_sha256") != entry["answer_file_sha256"]
                        or answer.get("evidence_sha256") != artifact.get("evidence_sha256")):
                    raise ValueError("final-model answer does not belong to this exact artifact")
            else:
                observation = artifact["parser"]["status"]
                answer = {"status": "not_called" if observation in {"request_failure", "incomplete_video"}
                          else "not_started", "final_answer": None, "observation_status": observation,
                          "reason": "no_saved_final_model_answer"}
        status = answer["status"]
        if status not in PENDING | TERMINAL:
            raise ValueError(f"unknown answer state: {status}")
        pending = (status in PENDING or answer.get("observation_status") in PENDING
                   or (answer.get("failure_policy") or {}).get("scope") == "shared")
        prediction = answer.get("final_answer")
        if prediction is not None and not isinstance(prediction, (str, int, float)):
            raise ValueError("answer must be text, a number, or null")
        truth = truths[qid]["ground_truth"]
        doc = {**question, "ground_truth_json": json.dumps(truth, ensure_ascii=False),
               "public_context_json": json.dumps(question.get("public_context") or {}, ensure_ascii=False)}
        # Failed answers contribute zero quality; retain their original payload below.
        scoring_text = str(prediction) if status == "answered" and prediction is not None else ""
        metrics = scorer.benchmark_v1_process_results(doc, [scoring_text])
        if "Counting_MRA" in metrics:
            error = metrics.get("Counting_MAE")
            metrics["Counting_MAE"] = error
            metrics["Counting_Strict_Equality"] = float(error == 0) if error is not None else 0.0
        for prefix in ("Temporal_TimePoint", "Temporal_Duration"):
            if prefix + "_Valid_Rate" in metrics:
                metrics.setdefault(prefix + "_MAE", None)
        if pending:
            metrics = {key: None for key in metrics}
        rows.append({"qa_id": qid, "video_id": question["video_id"], "question": question,
                     "ground_truth": truth, "status": status, "score_state": "pending" if pending else "terminal",
                     "prediction": prediction, "original_answer": answer, "metrics": metrics,
                     "artifact_path": str(artifact_path), "artifact_sha256": entry["answer_file_sha256"],
                     "answer_path": str(source),
                     "answer_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None})
    groups = {}
    for field in ("capability", "question_form", "source_group"):
        values = sorted({str(row["question"].get(field, "unspecified")) for row in rows})
        groups[field] = {value: _aggregate([row for row in rows
                           if str(row["question"].get(field, "unspecified")) == value]) for value in values}
    groups["ground_truth_label"] = {}
    for value in sorted({str(row["ground_truth"]["value"]) for row in rows
                         if row["question"]["capability"] == "event_recognition"}):
        groups["ground_truth_label"][value] = _aggregate([row for row in rows
            if row["question"]["capability"] == "event_recognition" and str(row["ground_truth"]["value"]) == value])
    result = {"schema_version": "unified-pr63-results-v1", "method": cohort["method"], "mode": mode,
              "scorer_sha256": SCORER_SHA256, "scorer_merge_commit": "253935fc944fd8ff05ae84c1f3a987c035141578",
              "questions_sha256": questions_sha256, "truth_sha256": truth_sha256, "cohort_sha256": cohort_sha256,
              "full_cohort": _aggregate(rows), "breakdowns": groups, "questions": rows,
              "new_model_calls": 0, "historical_answers_modified": False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_once(output, result)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "truth", "cohort"):
        cli.add_argument("--" + name, type=Path, required=True)
        cli.add_argument("--" + name + "-sha256", required=True)
    cli.add_argument("--mode", choices=("rule", "model"), required=True)
    cli.add_argument("--summary-root", type=Path)
    cli.add_argument("--scorer", type=Path, default=Path(__file__).resolve().parents[3] /
                     "experiments/ashley-detector182-20261012/scoring/scoring.py")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    result = score_cohort(questions_path=args.questions, questions_sha256=args.questions_sha256,
        truth_path=args.truth, truth_sha256=args.truth_sha256, cohort_path=args.cohort,
        cohort_sha256=args.cohort_sha256, mode=args.mode, summary_root=args.summary_root,
        scorer_path=args.scorer, output=args.output)
    print(json.dumps(result["full_cohort"], ensure_ascii=False))


if __name__ == "__main__":
    main()
