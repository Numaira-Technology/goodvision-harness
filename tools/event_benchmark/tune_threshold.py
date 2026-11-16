#!/usr/bin/env python3
"""Per-class threshold tuning for the decision model (luna), offline.

Protocol (validation-set README + SPEC.md §5/§8):
  pick a per-class threshold on a HELD-OUT validation set, FIX it, then evaluate the
  test set. Never use the test set to pick the threshold. The frozen main table keeps
  gate = 0.5; this is a labelled side-study, not a replacement for it.

Only the decision backend (luna) has continuous per-clip scores worth thresholding
(language models emit hard 0/1). run_eval.py already records per-clip `score` + `gt` +
`event` in `per_video`, so this is pure offline arithmetic — no API calls.

Usage:
  python -I tune_threshold.py select --val outputs/luna_validation_v2.json \
      --val-manifest /reviewed/validation.json --out outputs/thresholds.json
  python -I tune_threshold.py evaluate --thresholds outputs/thresholds.json \
      --thresholds-sha256 <hash-printed-by-select> --test outputs/luna_test_v2.json \
      --test-manifest /reviewed/test.json --out outputs/threshold-report.json

Historical v1 archives are deliberately rejected. Validation requires a complete
reviewed lossless v2 run with six questions and both classes per event; test errors
are explicitly counted. Manifests must include reviewed source_video_group values.
Selection cannot read test predictions; evaluation requires the frozen artifact hash.
Only decision probabilities are supported until a threshold backend defines its
score range and compatibility contract.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EVENTS = ["fight", "smoking", "fall", "weapon"]
_spec = importlib.util.spec_from_file_location("threshold_evidence", Path(__file__).with_name("evidence.py"))
evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evidence)


def read_json(path):
    """Hash the same bytes we parse, so provenance cannot name a later revision."""
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_new_json(path, data):
    raw = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Selection artifacts and reports are immutable outputs, never input replacements.
    with path.open("xb") as handle:
        handle.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_run(per_video, metadata, *, validation=False):
    """Only compatible, complete v2 evidence can enter threshold selection."""
    if metadata.get("spec_version") not in ("v2", "v2.1"):
        raise ValueError("v2/v2.1 run output required; historical/provisional experiments are not eligible")
    if metadata.get("frozen", True) is not True or metadata.get("provisional", False) is not False:
        raise ValueError("frozen, non-provisional evidence required")
    if "scored_events" in metadata and metadata["scored_events"] != EVENTS:
        raise ValueError("the complete frozen base-4 event set is required")
    if metadata.get("score_precision") != "lossless":
        raise ValueError("lossless scores required; rounded v2 archives are not eligible")
    if (metadata.get("backend") != "decision"
            or not isinstance(metadata.get("model"), str) or not metadata["model"].strip()):
        raise ValueError("a named decision model is required; language/threshold backends are unsupported")
    for key in ("samples_json_sha256", "manifest_sha256", "runner_sha256", "spec_sha256"):
        if not _digest(metadata.get(key)):
            raise ValueError(f"missing or invalid {key}")
    config = metadata.get("config")
    required = {"gate", "interval", "frame_rule", "frame_width", "temperature", "retries", "timeout", "decision"}
    if not isinstance(config, dict) or not required <= set(config) or config["gate"] != 0.5:
        raise ValueError("complete frozen gate=0.5 configuration required")
    if type(config["retries"]) is not int or config["retries"] != 3:
        raise ValueError("frozen v2 requires three total request attempts")
    if not isinstance(per_video, list) or not per_video:
        raise ValueError("per_video must contain scored questions")
    numbers, identities = set(), set()
    errors = 0
    for row in per_video:
        if not isinstance(row, dict):
            raise ValueError("each question must be an object")
        number = row.get("number")
        if type(number) is not int or number in numbers:
            raise ValueError("unique integer question numbers required")
        numbers.add(number)
        if row.get("event") not in EVENTS or row.get("gt") not in ("yes", "no"):
            raise ValueError("unsupported event or non-binary ground truth")
        if not isinstance(row.get("filename"), str) or not row["filename"] or not _digest(row.get("sha256")):
            raise ValueError("every question requires a filename and media SHA-256")
        identity = (row["event"], row["sha256"])
        if identity in identities:
            raise ValueError("duplicate event/media question")
        identities.add(identity)
        if "error" in row:
            if validation or "score" in row:
                raise ValueError("validation must be complete and error rows cannot carry scores")
            errors += 1
            continue
        score = row.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("decision scores must be finite probabilities")
        predicted = "yes" if score >= 0.5 else "no"
        if (row.get("pred") != predicted or type(row.get("correct")) is not bool
                or row["correct"] != (predicted == row["gt"])):
            raise ValueError("saved baseline verdict disagrees with the lossless score")
        planned = row.get("frames_planned")
        if (type(planned) is not int or planned <= 0
                or type(row.get("frames_ok")) is not int or row["frames_ok"] != planned
                or type(row.get("frames_failed")) is not int or row["frames_failed"] != 0):
            raise ValueError("scored questions require complete frame evidence")
    scored = len(per_video) - errors
    expected = {"attempted": len(per_video), "total": len(per_video) if metadata["spec_version"] == "v2.1" else scored, "errors": errors}
    overall = metadata.get("overall", {})
    if not isinstance(overall, dict) or any(overall.get(key) != value for key, value in expected.items()):
        raise ValueError("run attempted/scored/error totals disagree with its questions")
    if not scored:
        raise ValueError("run has no eligible scored questions")
    if validation:
        for event in EVENTS:
            if {row["gt"] for row in per_video if row["event"] == event} != {"yes", "no"}:
                raise ValueError(f"validation requires positive and negative examples for {event}")
            if sum(row["event"] == event for row in per_video) < 6:
                raise ValueError(f"validation requires six complete questions for {event}")
    if metadata.get("protocol_variant") != evidence.VARIANT:
        raise ValueError("controlled provider-pinned protocol variant required")
    if not _digest(metadata.get("evidence_sha256")):
        raise ValueError("evidence helper fingerprint required")
    evidence.verify_provider_evidence(metadata, per_video)
    return expected


def validate_manifest(per_video, metadata, path, split_manifest=None, split=None):
    """Match the complete scored cohort to its reviewed, hashed source manifest."""
    manifest, digest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get("status") != "human_reviewed":
        raise ValueError("a human_reviewed manifest is required")
    if (digest != metadata["manifest_sha256"]
            or manifest.get("samples_json_sha256") != metadata["samples_json_sha256"]):
        raise ValueError("run/manifest provenance differs")
    files = manifest.get("files")
    if not isinstance(files, list) or any(not isinstance(row, dict) for row in files):
        raise ValueError("manifest files must be question objects")
    groups = {}
    if split_manifest:
        groups, context = evidence.split_groups(split_manifest, path, split)
        if metadata.get("split_context") != context:
            raise ValueError("run/source-group sidecar provenance differs")
    elif metadata.get("split_context"):
        raise ValueError("supply the frozen --split-manifest used by the run")
    expected = {}
    groups_by_media = {}
    questions_by_event = {}
    for row in files:
        if row.get("event") not in EVENTS:
            continue
        number = row.get("number")
        if type(number) is not int or number in expected:
            raise ValueError("manifest requires unique integer question numbers")
        group = groups.get(number, row.get("source_video_group"))
        if not isinstance(group, str) or not group.strip() or group != group.strip():
            raise ValueError("reviewed source_video_group required for each question")
        if row.get("human_review") != "user_reviewed":
            raise ValueError("every scored manifest question must be user_reviewed")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("manifest question text required")
        event = row["event"]
        if event in questions_by_event and questions_by_event[event] != question:
            raise ValueError("each event must use one frozen question text")
        questions_by_event[event] = question
        for key in ("filename", "sha256"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"manifest question requires {key}")
            identity = (key, value)
            if identity in groups_by_media and groups_by_media[identity] != group:
                raise ValueError("one media identity cannot belong to different source groups")
            groups_by_media[identity] = group
        expected[number] = {**row, "source_video_group": group}
    if set(expected) != {row["number"] for row in per_video}:
        raise ValueError("run does not cover the exact reviewed manifest questions")
    identities = []
    for row in per_video:
        source = expected[row["number"]]
        if (any(row[key] != source.get(key) for key in ("event", "filename", "sha256"))
                or row["gt"] != source.get("proposed_answer")):
            raise ValueError("run question differs from reviewed manifest")
        if row.get("source_video_group", source["source_video_group"]) != source["source_video_group"]:
            raise ValueError("run/manifest source group differs")
        identities.append({**{key: row[key] for key in ("number", "event", "filename", "sha256", "gt")},
                           "question": source["question"],
                           "source_video_group": source["source_video_group"]})
    return identities


def validate_pair(val_rows, val_meta, test_rows, test_meta):
    for key in ("spec_version", "backend", "model", "runner_sha256", "spec_sha256", "protocol_variant", "evidence_sha256", "provider_context"):
        if val_meta[key] != test_meta[key]:
            raise ValueError(f"validation/test {key} differs")
    for key in ("actual_providers", "actual_models"):
        if val_meta["provider_observed"][key] != test_meta["provider_observed"][key]:
            raise ValueError("validation/test actual provider/model differs")
    if val_meta.get("split_context", {}).get("sha256") != test_meta.get("split_context", {}).get("sha256"):
        raise ValueError("validation/test split manifest differs")
    if val_meta["config"] != test_meta["config"]:
        raise ValueError("validation/test sampling or scoring configuration differs")
    for key in ("filename", "sha256", "source_video_group"):
        if key == "source_video_group" and any(key not in row for row in val_rows + test_rows):
            raise ValueError("reviewed source groups are required before testing disjointness")
        if {row[key] for row in val_rows} & {row[key] for row in test_rows}:
            raise ValueError(f"validation/test media overlap by {key}")
    questions = {row["event"]: row["question"] for row in val_rows}
    if any(questions.get(row["event"]) != row["question"] for row in test_rows):
        raise ValueError("validation/test question text differs")


def confusion(items, tau):
    """items: list of (score, gt_yes). pred_yes = score >= tau."""
    tp = fn = tn = fp = 0
    for score, gt in items:
        pred = (score is not None) and (score >= tau)
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif (not gt) and pred:
            fp += 1
        else:
            tn += 1
    n = tp + fn + tn + fp
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "n": n,
            "accuracy": round(100 * acc, 1) if n else None, "precision": prec, "recall": rec,
            "youden_j": round(tpr - fpr, 4) if (tp + fn) and (tn + fp) else None}


def pick_tau(items):
    """Select the longest optimal grid run; verify its midpoint stays optimal.

    Rank exact correct counts, never the rounded display percentages. Equal-length
    runs prefer the midpoint closest to 0.5, then the lower midpoint. If scores
    between grid points make that midpoint worse, use the nearest optimal grid
    point in the original PR. This controlled variant stops instead.
    All decisions depend on validation only.
    """
    if not items:
        raise ValueError("threshold selection requires validation questions")
    grid = [i / 100 for i in range(101)]
    scored = [confusion(items, tau) for tau in grid]
    correct = [c["TP"] + c["TN"] for c in scored]
    best = max(correct)
    winners = [i for i, count in enumerate(correct) if count == best]
    runs = []
    for i in winners:
        if not runs or i != runs[-1][-1] + 1:
            runs.append([i])
        else:
            runs[-1].append(i)
    # Integer grid indices keep mathematically equal distances tied exactly.
    chosen = min(runs, key=lambda run: (-len(run), abs(run[0] + run[-1] - 100),
                                       run[0] + run[-1]))
    midpoint_sum = chosen[0] + chosen[-1]
    tau = midpoint_sum / 200
    at_tau = confusion(items, tau)
    fallback = at_tau["TP"] + at_tau["TN"] != best
    if fallback:
        raise ValueError("optimal-grid midpoint is not optimal; stop for review instead of changing the threshold rule")
    selection = {"tie_rule": "longest-run-midpoint-stop-v2", "grid": grid,
                 "correct_counts": correct, "best_correct": best,
                 "chosen_run": [grid[chosen[0]], grid[chosen[-1]]],
                 "midpoint": midpoint_sum / 200, "fallback_to_grid": fallback}
    return tau, at_tau, selection


def collect(per_video, event):
    return [(x.get("score"), x.get("gt") == "yes")
            for x in per_video if x.get("event") == event and x.get("score") is not None]


def select_thresholds(val_path, manifest_path, split_manifest=None, reference_path=None):
    val_meta, val_digest = read_json(val_path)
    if not isinstance(val_meta, dict):
        raise ValueError("expected a validation run object")
    val_pv = val_meta.get("per_video")
    counts = validate_run(val_pv, val_meta, validation=True)
    identities = validate_manifest(val_pv, val_meta, manifest_path, split_manifest, "validation")
    reference = None
    if reference_path:
        reference, reference_digest = read_json(reference_path)
        if (reference.get("model") != val_meta["model"] or reference.get("backend") != "decision"
                or set(reference.get("thresholds", {})) != set(EVENTS)
                or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1
                       for t in reference["thresholds"].values())):
            raise ValueError("historical reference threshold model or values differ")
        reference = {**reference, "source_sha256": reference_digest}
    per_class = {}
    for event in EVENTS:
        items = collect(val_pv, event)
        tau, at_tau, selection = pick_tau(items)
        per_class[event] = {
            "tau": tau, "val_at_tau": at_tau, "val_at_0.5": confusion(items, 0.5),
            "n_val": len(items), "positive": sum(gt for _, gt in items),
            "negative": sum(not gt for _, gt in items), "selection": selection,
        }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "artifact_type": "decision-threshold-selection-v1", "comparator": ">=",
        "protocol_variant": evidence.VARIANT,
        "reference_thresholds": reference,
        "split_limits": val_meta.get("split_context", {}).get("limits", []),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "tuner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "val_source": str(val_path), "val_result_sha256": val_digest,
        "val_counts": counts, "validation_questions": identities,
        "validation_run": {**{"split_context": val_meta.get("split_context", {})}, **{key: val_meta[key] for key in (
            "spec_version", "score_precision", "backend", "model", "config",
            "samples_json_sha256", "manifest_sha256", "runner_sha256", "spec_sha256",
            "protocol_variant", "evidence_sha256", "provider_context", "provider_observed")}},
        "split_context": val_meta.get("split_context"),
        "thresholds": {event: per_class[event]["tau"] for event in EVENTS},
        "per_class_val": per_class,
        "limitations": "Reviewed groups must cover reused/overlapping footage. "
                       "Training/development disjointness requires independent review; "
                       "it is not established by this tool.",
    }


def evaluate_thresholds(artifact_path, expected_digest, test_paths, manifest_paths, split_manifest=None):
    artifact, digest = read_json(artifact_path)
    if not _digest(expected_digest) or digest != expected_digest:
        raise ValueError("frozen threshold artifact SHA-256 mismatch")
    if (not isinstance(artifact, dict)
            or artifact.get("artifact_type") != "decision-threshold-selection-v1"
            or artifact.get("comparator") != ">="):
        raise ValueError("unsupported threshold artifact")
    thresholds = artifact.get("thresholds")
    if (not isinstance(thresholds, dict) or set(thresholds) != set(EVENTS)
            or any(type(tau) not in (int, float) or not math.isfinite(tau) or not 0 <= tau <= 1
                   for tau in thresholds.values())):
        raise ValueError("invalid frozen thresholds")
    if len(test_paths) != len(manifest_paths):
        raise ValueError("provide one --test-manifest for each --test, in matching order")
    if len({Path(path).resolve() for path in test_paths}) != len(test_paths):
        raise ValueError("duplicate test input")
    report = {
        "study": "decision per-class threshold, validation-selected, held-out evaluation; "
                 "NOT the fixed 0.5 main table",
        "threshold_artifact": str(artifact_path), "threshold_artifact_sha256": digest,
        "thresholds": thresholds, "val_counts": artifact["val_counts"],
        "per_class_val": artifact["per_class_val"],
        "limitations": artifact["limitations"], "split_limits": artifact.get("split_limits", []),
        "reference_thresholds": artifact.get("reference_thresholds"), "test": {},
    }
    for test_path, manifest_path in zip(test_paths, manifest_paths):
        test_meta, result_digest = read_json(test_path)
        if not isinstance(test_meta, dict):
            raise ValueError("expected a test run object")
        test_pv = test_meta.get("per_video")
        counts = validate_run(test_pv, test_meta)
        identities = validate_manifest(test_pv, test_meta, manifest_path, split_manifest, "test")
        validate_pair(artifact["validation_questions"], artifact["validation_run"],
                      identities, test_meta)
        events = {}
        all_tuned, all_fixed, all_previous = [], [], []
        for event in EVENTS:
            items = collect(test_pv, event)
            attempted = sum(row["event"] == event for row in test_pv)
            if not attempted:
                continue
            tau = thresholds[event]
            events[event] = {
                "tau": tau, "at_tau": confusion(items, tau),
                "at_0.5": confusion(items, 0.5),
                "counts": {"attempted": attempted, "total": len(items),
                           "errors": attempted - len(items)},
            }
            if artifact.get("reference_thresholds"):
                previous_tau = artifact["reference_thresholds"]["thresholds"][event]
                events[event]["previous_tau"] = previous_tau
                events[event]["at_previous_tau"] = confusion(items, previous_tau)
                all_previous.extend((float(score >= previous_tau), gt) for score, gt in items)
            # Confusion over hard predictions combines per-event gates without
            # introducing a second score threshold or rounding the original score.
            all_tuned.extend((float(score >= tau), gt) for score, gt in items)
            all_fixed.extend(items)
        total = len(test_pv)
        all_metrics = lambda items: {"correct": sum(s >= .5 if gt else s < .5 for s, gt in items), "total": total}
        report["test"][str(test_path)] = {
            "n_total": counts["total"], "counts": counts,
            "all_questions_at_tau": all_metrics(all_tuned),
            "all_questions_at_0.5": all_metrics(all_fixed),
            "legacy_confusion_note": "overall_at_* accuracy is complete-only; use all_questions_* for headline denominators",
            "result_sha256": result_digest, "manifest_sha256": test_meta["manifest_sha256"],
            "overall_at_tau": confusion(all_tuned, 0.5),
            "overall_at_0.5": confusion(all_fixed, 0.5), "per_event": events,
            "overall_at_previous_tau": confusion(all_previous, 0.5) if artifact.get("reference_thresholds") else None,
        }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="freeze thresholds using validation only")
    select.add_argument("--val", required=True)
    select.add_argument("--val-manifest", required=True)
    select.add_argument("--split-manifest")
    select.add_argument("--reference-thresholds", help="historical thresholds for comparison only, frozen before test")
    select.add_argument("--out", required=True, help="new frozen threshold artifact")
    evaluate = commands.add_parser("evaluate", help="apply an already frozen threshold artifact")
    evaluate.add_argument("--thresholds", required=True)
    evaluate.add_argument("--thresholds-sha256", required=True, help="hash printed by select")
    evaluate.add_argument("--test", action="append", required=True)
    evaluate.add_argument("--test-manifest", action="append", required=True)
    evaluate.add_argument("--split-manifest")
    evaluate.add_argument("--out", required=True, help="new evaluation report")
    args = ap.parse_args()
    try:
        if args.command == "select":
            result = select_thresholds(args.val, args.val_manifest, args.split_manifest, args.reference_thresholds)
        else:
            result = evaluate_thresholds(args.thresholds, args.thresholds_sha256,
                                         args.test, args.test_manifest, args.split_manifest)
        digest = write_new_json(args.out, result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        ap.error(str(error))
    print(f"{args.command}: {args.out}\nSHA-256: {digest}")
    return result


if __name__ == "__main__":
    main()
