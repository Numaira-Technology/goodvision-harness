"""Pure planning/scoring primitives. No network, credentials or model imports."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

VERSION = "unified-frame-120-v2"
EVENTS = ("fight", "smoking", "fall", "weapon", "bag_knife", "bag_alcohol_bottle")
RULES = {
    "version": VERSION,
    "sampling": "t=0,1,... seconds, strictly before container duration; common decoded JPEGs",
    "fps": 1, "frame_width": 720, "slots": 12,
    "window": "native VideoJev/Decision3: adjacent two distinct sampled frames; others: single frame",
    "attempts_total": 3, "retry_wait_seconds": [30, 60],
    "retry_after": "respect longer server delay; share provider cooldown; never change request",
    "complete": "every planned target response is valid; partial yes does not complete a question",
    "aggregate": "any yes -> yes; otherwise any no -> no; otherwise all unknown -> unknown",
    "unknown": "valid native unknown is not retried; complete all-unknown is incorrect against binary truth",
    "accuracy": "correct / all planned questions; missing evidence stays in denominator",
    "report": "combined test120 headline; final unique missing units and incomplete questions separately",
    "backpressure": "wait, never drop planned input; not a 12-camera real-time throughput claim",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_files():
    """The execution/selection code required by every frozen bundle."""
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "tools/unified_benchmark").glob("*.py"))
    paths += [p for p in (root / "tools/unified_benchmark/native_assets").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".patch")]
    paths += [root / name for name in (
        "tools/event_benchmark/run_eval.py", "tools/event_benchmark/evidence.py",
        "tools/event_benchmark/tune_threshold.py", "tools/clip_benchmark/scoring.py",
        "tools/clip_benchmark/runtime.py", "tools/clip_benchmark/base_timing.py",
        "tools/clip_benchmark/protocol.py",
    )]
    return {str(path.relative_to(root)): path for path in sorted(paths)}


def validate_threshold_values(thresholds, kind):
    if (not isinstance(thresholds, dict) or set(thresholds) != set(EVENTS)
            or any(type(t) not in (int, float) or not math.isfinite(t)
                   for t in thresholds.values())):
        raise ValueError("six finite thresholds required")
    if kind == "probability" and any(not 0 <= t <= 1 for t in thresholds.values()):
        raise ValueError("probability thresholds must be in [0, 1]")


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(canonical(value) + b"\n")


def sample_times(duration):
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("positive finite duration required")
    return [float(i) for i in range(math.ceil(duration)) if i < duration]


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def valid_value(value, kind):
    if not isinstance(value, dict):
        return False
    if kind == "choice":
        return value.get("answer") in ("yes", "no", "unknown")
    score = value.get("score")
    return (finite_number(score)
            and (kind == "margin" or kind == "probability" and 0 <= score <= 1))


def verdict(value, kind, threshold):
    if not valid_value(value, kind):
        raise ValueError("invalid evidence")
    if kind == "choice":
        return value["answer"]
    return "yes" if (value["score"] > threshold if kind == "margin" else value["score"] >= threshold) else "no"


def aggregate(answers):
    if not answers or any(a not in ("yes", "no", "unknown") for a in answers):
        raise ValueError("nonempty valid answers required")
    return "yes" if "yes" in answers else "no" if "no" in answers else "unknown"


def score(plan, evidence, kind, thresholds=None):
    """evidence[unit_id][event] contains the first valid response, never attempts."""
    units = {u["id"]: u for u in plan["units"]}
    if len(units) != len(plan["units"]) or set(evidence) - set(units):
        raise ValueError("duplicate planned units or unexpected evidence")
    numbers = [q["number"] for q in plan["questions"]]
    if len(numbers) != len(set(numbers)):
        raise ValueError("duplicate question ID")
    default = 0 if kind == "margin" else .5
    thresholds = {e: default for e in EVENTS} if thresholds is None else thresholds
    validate_threshold_values(thresholds, kind)
    rows, missing_units = [], set()
    for q in plan["questions"]:
        ids, event = q["unit_ids"], q["event"]
        if (not ids or len(ids) != len(set(ids)) or set(ids) - set(units)
                or event not in EVENTS or q["reference"] not in ("yes", "no")):
            raise ValueError("invalid question plan")
        if any(event not in units[uid]["targets"] for uid in ids):
            raise ValueError("question target missing from planned unit")
        missing = [uid for uid in ids if not valid_value(evidence.get(uid, {}).get(event), kind)]
        missing_units.update(missing)
        pred = None if missing else aggregate([verdict(evidence[uid][event], kind, thresholds[event]) for uid in ids])
        rows.append({"number": q["number"], "event": event, "reference": q["reference"],
                     "complete": not missing, "prediction": pred,
                     "correct": None if missing else pred == q["reference"], "missing_unit_ids": missing})
    def metrics(group):
        complete = [r for r in group if r["complete"]]
        correct = sum(r["correct"] is True for r in complete)
        return {"planned_questions": len(group), "complete_valid_questions": len(complete),
                "correct": correct, "accuracy": correct / len(group) if group else None,
                "accuracy_denominator": len(group),
                "complete_only_accuracy": correct / len(complete) if complete else None,
                "incomplete_questions": len(group) - len(complete),
                "final_unknown_questions": sum(r["prediction"] == "unknown" for r in complete)}
    return {"protocol": VERSION, **metrics(rows), "planned_unique_units": len(units),
            "final_missing_unique_units": len(missing_units),
            "valid_unit_coverage": (len(units) - len(missing_units)) / len(units) if units else None,
            "per_event": {e: metrics([r for r in rows if r["event"] == e]) for e in EVENTS},
            "questions": rows}


def select_thresholds(plan, evidence, kind, model):
    """Reuse the existing midpoint algorithms; never accept incomplete validation."""
    from tools.event_benchmark.tune_threshold import pick_tau
    from tools.clip_benchmark.scoring import _optimal_midpoint
    if plan["split"] != "validation" or len(plan["questions"]) != 24 or kind not in ("margin", "probability"):
        raise ValueError("threshold selection requires frozen validation24 and continuous scores")
    baseline = score(plan, evidence, kind)
    if baseline["complete_valid_questions"] != 24:
        raise ValueError("incomplete validation blocks threshold selection; no silent fallback")
    values, details = {e: 0.0 if kind == "margin" else .5 for e in EVENTS}, {}
    for event in EVENTS[:4]:
        subset = [q for q in plan["questions"] if q["event"] == event]
        if Counter(q["reference"] for q in subset) != {"yes": 3, "no": 3}:
            raise ValueError("expected three positive and three negative validation questions per class")
        maxima = [max(evidence[uid][event]["score"] for uid in q["unit_ids"]) for q in subset]
        if kind == "probability":
            result = pick_tau(list(zip(maxima, [q["reference"] == "yes" for q in subset])))
            values[event] = result[0]
            details[event] = result
        else:
            samples = [{"filename": str(q["number"]), "event": event, "proposed_answer": q["reference"]} for q in subset]
            indexed = {str(q["number"]): [{"errors": None, "predictions": {event: {"margin": val}}}] for q, val in zip(subset, maxima)}
            result = _optimal_midpoint(samples, indexed, model)
            values[event], details[event] = result[0], result[1:]
    return {"thresholds": values, "details": details, "comparator": ">" if kind == "margin" else ">=",
            "validation_plan_sha256": digest(plan), "validation_evidence_sha256": digest(evidence),
            "validation_summary": {k: baseline[k] for k in ("planned_questions", "complete_valid_questions",
                "incomplete_questions", "final_missing_unique_units")},
            "bag_calibration": "uncalibrated_no_bag_validation", "validation_only": True}
