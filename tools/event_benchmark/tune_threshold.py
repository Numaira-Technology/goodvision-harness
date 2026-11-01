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
  python -I tune_threshold.py --val outputs/luna_validation_v2.json \
      --test outputs/luna_test_v2.json --out outputs/luna_thresholds_v2.json

Historical v1 archives are deliberately rejected. Validation requires a complete
reviewed v2 run with both classes per event; test errors are explicitly counted.
Only decision probabilities are supported until a threshold backend defines its
score range and compatibility contract.
"""
import argparse
import json
import math
from pathlib import Path

EVENTS = ["fight", "smoking", "fall", "weapon"]


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a run object")
    return data.get("per_video"), data


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def validate_run(per_video, metadata, *, validation=False):
    """Only compatible, complete v2 evidence can enter threshold selection."""
    if metadata.get("spec_version") != "v2":
        raise ValueError("v2 run output required; historical v1 archives are not eligible")
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
        planned = row.get("frames_planned")
        if (type(planned) is not int or planned <= 0
                or type(row.get("frames_ok")) is not int or row["frames_ok"] != planned
                or type(row.get("frames_failed")) is not int or row["frames_failed"] != 0):
            raise ValueError("scored questions require complete frame evidence")
    scored = len(per_video) - errors
    expected = {"attempted": len(per_video), "total": scored, "errors": errors}
    overall = metadata.get("overall", {})
    if not isinstance(overall, dict) or any(overall.get(key) != value for key, value in expected.items()):
        raise ValueError("run attempted/scored/error totals disagree with its questions")
    if not scored:
        raise ValueError("run has no eligible scored questions")
    if validation:
        for event in EVENTS:
            if {row["gt"] for row in per_video if row["event"] == event} != {"yes", "no"}:
                raise ValueError(f"validation requires positive and negative examples for {event}")
    return expected


def validate_pair(val_rows, val_meta, test_rows, test_meta):
    for key in ("backend", "model", "runner_sha256", "spec_sha256"):
        if val_meta[key] != test_meta[key]:
            raise ValueError(f"validation/test {key} differs")
    # Concurrency does not change the frozen scoring contract.
    config = lambda data: {key: value for key, value in data["config"].items() if key != "workers"}
    if config(val_meta) != config(test_meta):
        raise ValueError("validation/test sampling or scoring configuration differs")
    for key in ("filename", "sha256"):
        if {row[key] for row in val_rows} & {row[key] for row in test_rows}:
            raise ValueError(f"validation/test media overlap by {key}")


def is_yes(v):
    return str(v).strip().lower() in ("yes", "true", "1")


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
            "accuracy": round(100 * acc, 1), "precision": prec, "recall": rec,
            "youden_j": round(tpr - fpr, 4)}


def pick_tau(items):
    """Grid-sweep tau in [0,1], then select the median of best-scoring grid points.
    Tied points need not form one interval, so this is not a max-margin guarantee."""
    grid = [round(i / 100, 2) for i in range(0, 101)]
    scored = [(tau, confusion(items, tau)) for tau in grid]
    best_acc = max(c["accuracy"] for _, c in scored)
    winners = [tau for tau, c in scored if c["accuracy"] == best_acc]
    tau_star = round(winners[len(winners) // 2], 2)  # median of tied grid points
    return tau_star, confusion(items, tau_star)


def collect(per_video, event):
    return [(x.get("score"), is_yes(x.get("gt")))
            for x in per_video if x.get("event") == event and x.get("score") is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="scored validation JSON (run_eval --out)")
    ap.add_argument("--test", action="append", required=True,
                    help="scored test JSON(s) (run_eval --out); repeatable")
    ap.add_argument("--out", help="write the tuning report JSON here")
    a = ap.parse_args()

    try:
        val_pv, val_meta = rows(a.val)
        val_counts = validate_run(val_pv, val_meta, validation=True)
        test_runs = []
        for test_path in a.test:
            test_pv, test_meta = rows(test_path)
            counts = validate_run(test_pv, test_meta)
            validate_pair(val_pv, val_meta, test_pv, test_meta)
            test_runs.append((test_path, test_pv, test_meta, counts))
    except (OSError, ValueError) as error:
        ap.error(str(error))

    per_class = {}
    for ev in EVENTS:
        items = collect(val_pv, ev)
        if not items:
            per_class[ev] = {"tau": None, "note": "no validation clips for this event"}
            continue
        tau, val_at_tau = pick_tau(items)
        val_at_half = confusion(items, 0.5)
        per_class[ev] = {"tau": tau, "val_at_tau": val_at_tau, "val_at_0.5": val_at_half,
                         "n_val": len(items)}

    report = {
        "study": "luna per-class threshold, val-tuned (held-out), SPEC §5/§8 footnote — "
                 "NOT the frozen 0.5 main table",
        "val_source": a.val,
        "val_model": val_meta.get("model"),
        "val_counts": val_counts,
        "eligibility": "v2 complete validation; test errors reported separately",
        "val_manifest_sha256": val_meta["manifest_sha256"],
        "thresholds": {ev: per_class[ev].get("tau") for ev in EVENTS},
        "per_class_val": per_class,
        "test": {},
    }

    for test_path, test_pv, test_meta, counts in test_runs:
        n_total = counts["total"]
        ev_block = {}
        agg_tau = {"TP": 0, "FN": 0, "TN": 0, "FP": 0, "n": 0}
        agg_half = {"TP": 0, "FN": 0, "TN": 0, "FP": 0, "n": 0}
        for ev in EVENTS:
            items = collect(test_pv, ev)
            if not items:
                continue
            tau = per_class[ev].get("tau")
            at_tau = confusion(items, tau if tau is not None else 0.5)
            at_half = confusion(items, 0.5)
            ev_block[ev] = {"tau": tau, "at_tau": at_tau, "at_0.5": at_half}
            for k in agg_tau:
                agg_tau[k] += at_tau[k]
                agg_half[k] += at_half[k]

        def finish(agg):
            n = agg["n"]
            tp, fp, fn, tn = agg["TP"], agg["FP"], agg["FN"], agg["TN"]
            agg["accuracy"] = round(100 * (tp + tn) / n, 1) if n else 0.0
            agg["precision"] = round(tp / (tp + fp), 3) if (tp + fp) else None
            agg["recall"] = round(tp / (tp + fn), 3) if (tp + fn) else None
            return agg

        report["test"][test_path] = {
            "n_total": n_total,
            "counts": counts,
            "manifest_sha256": test_meta["manifest_sha256"],
            "overall_at_tau": finish(agg_tau),
            "overall_at_0.5": finish(agg_half),
            "per_event": ev_block,
        }

    out_str = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(out_str + "\n")

    # readable summary
    print("Per-class thresholds (val-tuned, held-out):")
    for ev in EVENTS:
        pc = per_class[ev]
        if pc.get("tau") is None:
            print("  %-8s  tau=  n/a  (%s)" % (ev, pc.get("note", "")))
            continue
        print("  %-8s  tau=%.2f   val acc %.1f%% (vs %.1f%% @0.5)  J=%.3f"
              % (ev, pc["tau"], pc["val_at_tau"]["accuracy"],
                 pc["val_at_0.5"]["accuracy"], pc["val_at_tau"]["youden_j"]))
    for test_path, tb in report["test"].items():
        print("\nTest %s (scored=%s, attempted=%s, errors=%s):"
              % (test_path, tb["n_total"], tb["counts"]["attempted"], tb["counts"]["errors"]))
        print("  overall acc  @tuned %.1f%%   vs  @0.5 %.1f%%"
              % (tb["overall_at_tau"]["accuracy"], tb["overall_at_0.5"]["accuracy"]))
    if not a.out:
        print("\n(no --out; report not written)")


if __name__ == "__main__":
    main()
