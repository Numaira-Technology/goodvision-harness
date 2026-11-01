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
  python -I tune_threshold.py --val results/luna_val24.json \
      --test results/luna_1fps.json [--test results/luna_100base.json] \
      --out results/luna_threshold_tuning.json
"""
import argparse
import json
import sys

EVENTS = ["fight", "smoking", "fall", "weapon"]


def rows(path):
    d = json.load(open(path))
    return d.get("per_video", []), d


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
    """Grid-sweep tau in [0,1]; maximise val accuracy, and among all tau that tie at the
    best accuracy pick the MIDPOINT of that region (max-margin / most robust threshold)."""
    grid = [round(i / 100, 2) for i in range(0, 101)]
    scored = [(tau, confusion(items, tau)) for tau in grid]
    best_acc = max(c["accuracy"] for _, c in scored)
    winners = [tau for tau, c in scored if c["accuracy"] == best_acc]
    tau_star = round(winners[len(winners) // 2], 2)  # median of the tied region
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

    val_pv, val_meta = rows(a.val)

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
        "thresholds": {ev: per_class[ev].get("tau") for ev in EVENTS},
        "per_class_val": per_class,
        "test": {},
    }

    for test_path in a.test:
        test_pv, test_meta = rows(test_path)
        n_total = test_meta.get("overall", {}).get("total")
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
            "overall_at_tau": finish(agg_tau),
            "overall_at_0.5": finish(agg_half),
            "per_event": ev_block,
        }

    out_str = json.dumps(report, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w") as f:
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
        print("\nTest %s (n=%s):" % (test_path, tb["n_total"]))
        print("  overall acc  @tuned %.1f%%   vs  @0.5 %.1f%%"
              % (tb["overall_at_tau"]["accuracy"], tb["overall_at_0.5"]["accuracy"]))
    if not a.out:
        print("\n(no --out; report not written)")


if __name__ == "__main__":
    main()
