"""Select validation thresholds and score saved CLIP observations offline."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import statistics


BASE_EVENTS = ("fight", "smoking", "fall", "weapon")
BAG_EVENTS = ("bag_knife", "bag_alcohol_bottle")
EVENTS = BASE_EVENTS + BAG_EVENTS
SELECTION_RULE = "max(accuracy, -FP, -FN); midpoint of widest finite optimal interval; lower midpoint tie"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _read_records(path):
    records, issues = [], []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("record is not an object")
                    records.append(row)
                except (ValueError, UnicodeError) as error:
                    issues.append({"line": number, "error": str(error)})
    except (OSError, UnicodeError) as error:
        issues.append({"line": None, "error": str(error)})
    return records, issues


def _margin(record, event):
    if record.get("errors") is not None:
        return None
    predictions = record.get("predictions")
    item = predictions.get(event) if isinstance(predictions, dict) else None
    value = item.get("margin") if isinstance(item, dict) else None
    return float(value) if _finite(value) else None


def _index(records):
    indexed = defaultdict(list)
    for record in records:
        filename = record.get("filename")
        if record.get("quality_reference") is True and isinstance(filename, str):
            indexed[filename].append(record)
    return indexed


def _question_rows(samples, indexed, thresholds, model_id):
    rows = []
    for sample in samples:
        event = sample.get("event")
        records = indexed.get(sample.get("filename"), [])
        margins = [_margin(record, event) for record in records]
        valid = [value for value in margins if value is not None]
        score = max(valid) if valid else None
        threshold = thresholds.get(event, 0.0)
        prediction = "unknown" if score is None else "yes" if score > threshold else "no"
        reference = sample.get("proposed_answer")
        correct = prediction == reference if reference in ("yes", "no") else None
        failed = sum(record.get("errors") is not None for record in records)
        status = ("unknown" if prediction == "unknown" else
                  "reference_unknown" if correct is None else "correct" if correct else "incorrect")
        reason = (None if valid else "no_quality_observations" if not records else
                  "request_failures_without_valid_target_margin" if failed == len(records) else
                  "no_valid_target_margin")
        rows.append({**sample, "model": model_id, "reference": reference,
                     "margin": score, "threshold": threshold, "prediction": prediction,
                     "status": status, "correct": correct, "unknown_reason": reason,
                     "saved_quality_records": len(records), "valid_target_records": len(valid),
                     "request_failure_records": failed,
                     "unavailable_target_records": len(records) - len(valid)})
    return rows


def _quality(rows):
    counts = Counter(tp=0, tn=0, fp=0, fn=0, unknown_yes=0, unknown_no=0,
                     unknown_reference=0)
    for row in rows:
        reference, prediction = row.get("reference"), row.get("prediction")
        if reference not in ("yes", "no"):
            counts["unknown_reference"] += 1
        elif prediction == "unknown":
            counts["unknown_" + reference] += 1
        else:
            counts[("tp" if reference == "yes" else "fp") if prediction == "yes" else
                   ("fn" if reference == "yes" else "tn")] += 1
    positive_total = sum(row.get("reference") == "yes" for row in rows)
    negative_total = sum(row.get("reference") == "no" for row in rows)
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    return {**counts, "denominator": len(rows), "positive_total": positive_total,
            "negative_total": negative_total, "correct": tp + tn,
            "unknown": sum(row.get("prediction") == "unknown" for row in rows),
            "quality_failures": fp + fn,
            "questions_with_request_failures": sum(row.get("request_failure_records", 0) > 0
                                                    for row in rows),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / positive_total if positive_total else None,
            "accuracy": (tp + tn) / len(rows) if rows else None}


def _optimal_midpoint(subset, indexed, model_id):
    initial = _question_rows(subset, indexed, {}, model_id)
    if (len(initial) < 6 or any(row["margin"] is None for row in initial) or
            {row["reference"] for row in initial} != {"yes", "no"}):
        raise ValueError("complete validation with six questions and both labels is required")
    values = sorted({row["margin"] for row in initial})
    # Strict > gives constant decisions in [score_i, score_(i+1)).
    bounds = [None, *values, None]
    candidates = []
    for low, high in zip(bounds, bounds[1:]):
        tau = (math.nextafter(high, -math.inf) if low is None else
               low if high is None else low / 2 + high / 2)
        metrics = _quality(_question_rows(subset, indexed, {subset[0]["event"]: tau}, model_id))
        candidates.append({"lower": low, "upper": high, "threshold": tau, **metrics})
    rank = lambda row: (row["correct"], -row["fp"], -row["fn"])
    optimum = max(map(rank, candidates))
    intervals = []
    for row in candidates:
        if rank(row) != optimum:
            continue
        if intervals and intervals[-1]["upper"] == row["lower"]:
            intervals[-1]["upper"] = row["upper"]
        else:
            intervals.append({"lower": row["lower"], "upper": row["upper"]})
    finite = [r for r in intervals if r["lower"] is not None and r["upper"] is not None]
    if not finite:
        raise ValueError("optimal interval is unbounded: no finite midpoint; review required")
    selected = max(finite, key=lambda r: (r["upper"] - r["lower"], -(r["lower"] / 2 + r["upper"] / 2)))
    tau = selected["lower"] / 2 + selected["upper"] / 2
    metrics = _quality(_question_rows(subset, indexed, {subset[0]["event"]: tau}, model_id))
    if rank(metrics) != optimum or not selected["lower"] <= tau < selected["upper"]:
        raise ValueError("selected midpoint is not optimal; review required")
    return tau, selected | {"lower_inclusive": True, "upper_inclusive": False}, intervals, candidates


def select_thresholds(records_path: Path, samples: list, output: Path, model_id: str) -> dict:
    """Fit four per-class thresholds from validation references only."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, issues = _read_records(records_path)
    if issues:
        raise ValueError("validation records unreadable")
    indexed = _index(records)
    thresholds = {event: 0.0 for event in EVENTS}
    details, sweep = {}, []
    for event in EVENTS:
        subset = [sample for sample in samples if sample.get("event") == event]
        initial = _question_rows(subset, indexed, thresholds, model_id)
        candidates = []
        selected_interval, optimal_intervals = None, []
        if event in BASE_EVENTS:
            tau, selected_interval, optimal_intervals, candidates = _optimal_midpoint(subset, indexed, model_id)
            thresholds[event] = tau
            calibration = "selected_on_validation"
        else:
            chosen = {"event": event, "threshold": 0.0, **_quality(initial)}
            candidates.append(chosen)
            calibration = ("uncalibrated_no_bag_validation" if event in BAG_EVENTS else
                           "uncalibrated_no_valid_result")
        for row in candidates:
            sweep.append(row | {"event": event, "selected": row["threshold"] == thresholds[event],
                                "calibration_status": calibration})
        details[event] = {"threshold": thresholds[event], "calibrated": calibration == "selected_on_validation",
                          "selected_interval": selected_interval, "optimal_intervals": optimal_intervals,
                          "calibration_status": calibration, "validation_count": len(subset),
                          "expected_validation_count": 6 if event in BASE_EVENTS else 0,
                          "validation_metrics": _quality(_question_rows(
                              subset, indexed, thresholds, model_id))}
    _write_csv(output / "threshold-sweep.csv", sweep)
    _write_json(output / "thresholds.json", {
        "model": model_id, "selected_at": _timestamp(), "selection_rule": SELECTION_RULE,
        "selection_metric": "(TP + TN) / all validation questions for the class, including unknown results",
        "decision": "per-video maximum finite valid target margin > threshold",
        "validation_only": True, "thresholds": thresholds, "events": details,
        "record_failures": {"count": len(issues), "details": issues},
        "limits": ["Only each video's labeled target class is a scoring reference.",
                   "Bag thresholds remain zero because no bag calibration data were supplied.",
                   "Unknown results remain in the class denominator; no missing result is filled with false."]})
    return thresholds


def _distribution(values):
    values = sorted(value for value in values if _finite(value))
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {"n": len(values), "min": values[0], "p50": values[math.ceil(.5 * len(values)) - 1],
            "p95": values[math.ceil(.95 * len(values)) - 1], "max": values[-1],
            "mean": statistics.mean(values)}


def _latency(record):
    start, end = record.get("last_frame_ingress"), record.get("classification_ready")
    if not _finite(end):
        end = record.get("readable")
    if (_finite(start) and _finite(end) and end >= start and
            any(_margin(record, event) is not None for event in EVENTS)):
        return end - start
    return None


def _grouped(rows, field):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        groups[str(value) if value is not None else "not_recorded"].append(row)
    return {key: _quality(group) for key, group in sorted(groups.items())}


def evaluate(records_path: Path, samples: list, thresholds: dict, output: Path, model_id: str) -> dict:
    """Score all supplied questions using already frozen thresholds."""
    if not isinstance(thresholds, dict):
        raise ValueError("Frozen thresholds must be an event-keyed object")
    for event in EVENTS:
        if not _finite(thresholds.get(event)):
            raise ValueError(f"Missing or invalid frozen threshold for {event}")
    frozen = {event: float(thresholds[event]) for event in EVENTS}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, issues = _read_records(records_path)
    indexed = _index(records)
    rows = _question_rows(samples, indexed, frozen, model_id)
    quality_records = [record for record in records if record.get("quality_reference") is True]
    latencies = [_latency(record) for record in records]
    quality_latencies = [_latency(record) for record in quality_records]
    baseline = _question_rows(samples, indexed, {event: 0.0 for event in EVENTS}, model_id)
    summary = {
        "model": model_id, "generated_at": _timestamp(), "thresholds": frozen,
        "question_count": len(samples), "video_count": len({sample.get("filename") for sample in samples}),
        "quality": _quality(rows), "by_event": {event: _quality([row for row in rows if row.get("event") == event])
                                                for event in EVENTS},
        "by_source_dataset": _grouped(rows, "source_dataset"),
        "by_source_group": _grouped(rows, "source_group"),
        "by_source_video_group": _grouped(rows, "source_video_group"),
        "by_human_review": _grouped(rows, "human_review"),
        "request_failures": {
            "all_window_records": sum(record.get("errors") is not None for record in records),
            "quality_window_records": sum(record.get("errors") is not None for record in quality_records),
            "affected_questions": sum(row["request_failure_records"] > 0 for row in rows)},
        "record_failures": {"count": len(issues), "details": issues},
        "observations": {
            "all_window_records": len(records), "quality_window_records": len(quality_records),
            "quality_videos_with_records": len(set(indexed) & {sample.get("filename") for sample in samples}),
            "valid_result_window_records": sum(any(_margin(record, event) is not None for event in EVENTS)
                                               for record in records),
            "quality_videos_without_records": sorted({sample.get("filename") for sample in samples
                                                       if sample.get("filename") not in indexed})},
        "last_frame_input_to_classification_ready_seconds": _distribution(latencies),
        "quality_last_frame_input_to_classification_ready_seconds": _distribution(quality_latencies),
        "latency_unavailable_window_records": sum(value is None for value in latencies),
        "zero_threshold_baseline": {"quality": _quality(baseline),
                                    "by_event": {event: _quality([row for row in baseline if row.get("event") == event])
                                                 for event in EVENTS}},
        "limits": [
            "Test observations never select or change a threshold; the zero-threshold baseline reuses the same saved scores.",
            "Question accuracy retains every supplied question, including unknown results; recall is TP divided by every GT-yes question.",
            "TP/TN/FP/FN and unknown are prediction outcomes. Request-failure counts describe execution and may overlap valid question predictions.",
            "A partial source can still produce a valid prediction. Raw records alone do not establish EOF, queue drain or opportunity completion rate.",
            "Shared bag observations score two different questions but count only once per recorded media window for latency and execution counts.",
            "Only quality_reference=true observations contribute to quality. Load-only companions contribute to the separately named all-window latency distribution.",
            "Timing ends when classification is ready and does not measure natural event-onset response time. Percentiles use nearest rank.",
            "Reference and review fields are preserved from the supplied samples. Missing source_dataset remains not_recorded rather than inferred.",
        ],
    }
    _write_json(output / "per-question.json", {"model": model_id, "questions": rows})
    _write_csv(output / "per-question.csv", rows)
    _write_json(output / "summary.json", summary)
    return summary
