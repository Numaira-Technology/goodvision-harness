"""Summarize fixed-window trials; clip votes are not independent video samples."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

LABELS = ("fight", "smoking", "fall", "weapon")


def dist(values):
    return {
        "count": len(values),
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
    }


def analyze(directory):
    all_rows = [
        json.loads(line)
        for line in (directory / "trials.jsonl").read_text().splitlines()
    ]
    rows = [r for r in all_rows if r["repeat"] >= 0]
    by_case = {}
    clips = []
    for case in ("full", "compact", "events", "dual_events", "dual_compact"):
        rr = [r for r in rows if r["case"] == case]
        if not rr:
            continue
        for sid in sorted({r["sample_id"] for r in rr}):
            cr = [r for r in rr if r["sample_id"] == sid]

            def vote(label, cr=cr):
                counts = Counter(
                    r.get("decisions", {}).get(label) for r in cr if r["ok"]
                )
                return (
                    True if counts[True] >= 2 else False if counts[False] >= 2 else None
                )

            decisions = {label: vote(label) for label in LABELS}
            clips.append(
                {
                    "case": case,
                    "sample_id": sid,
                    "group": cr[0]["group"],
                    "attempts": len(cr),
                    "decisions": decisions,
                    "votes": [
                        {
                            k: r.get(k)
                            for k in (
                                "repeat",
                                "ok",
                                "decisions",
                                "compact_negative",
                                "grounded_labels",
                            )
                        }
                        for r in cr
                    ],
                }
            )
        cc = [r for r in clips if r["case"] == case]
        positive = [r for r in rr if r["group"] != "adl"]
        by_case[case] = {
            "attempts": len(rr),
            "successes": sum(r["ok"] for r in rr),
            "seconds": dist([r["total_seconds"] for r in rr]),
            "completion_tokens": dist(
                [
                    r["usage"]["completion_tokens"]
                    for r in rr
                    if r.get("usage", {}).get("completion_tokens") is not None
                ]
            ),
            "compact_negative_count": sum(r.get("compact_negative", False) for r in rr),
            "smoking_clips_hit": sum(
                r["decisions"]["smoking"] is True for r in cc if r["group"] == "smoking"
            ),
            "fall_clips_hit": sum(
                r["decisions"]["fall"] is True for r in cc if r["group"] == "fall"
            ),
            "adl_clips_false_positive": sum(
                any(v is True for v in r["decisions"].values())
                for r in cc
                if r["group"] == "adl"
            ),
            "adl_clips_any_unknown": sum(
                any(v is None for v in r["decisions"].values())
                for r in cc
                if r["group"] == "adl"
            ),
            "target_positive_calls": sum(
                r.get("decisions", {}).get(
                    "fall" if r["group"] == "fall" else "smoking"
                )
                is True
                for r in positive
            ),
            "target_positive_grounded_calls": sum(
                ("fall" if r["group"] == "fall" else "smoking")
                in r.get("grounded_labels", [])
                for r in positive
            ),
            "per_group_seconds": {
                g: dist([r["total_seconds"] for r in rr if r["group"] == g])
                for g in ("fall", "smoking", "adl")
            },
            "omitted_labels": sum(
                sum(
                    label
                    not in set(
                        j["event_label"]
                        for j in json.loads(r["choices"][0]["message"]["content"]).get(
                            "verdicts", []
                        )
                    )
                    for label in LABELS
                )
                for r in rr
                if r["ok"] and r["kind"] != "events" and not r.get("compact_negative")
            ),
        }
    paired = []
    for sid, repeat in sorted({(r["sample_id"], r["repeat"]) for r in rows}):
        match = {
            r["case"]: r for r in rows if (r["sample_id"], r["repeat"]) == (sid, repeat)
        }
        if not {"dual_events", "dual_compact", "compact", "events"} <= match.keys():
            continue
        a, b = match["dual_events"], match["dual_compact"]
        paired.append(
            {
                "sample_id": sid,
                "repeat": repeat,
                "information_lead_seconds": b["completion_offset_seconds"]
                - a["completion_offset_seconds"],
                "events_slowdown_ratio": a["total_seconds"]
                / match["events"]["total_seconds"],
                "grounding_slowdown_ratio": b["total_seconds"]
                / match["compact"]["total_seconds"],
                "dispatch_skew_seconds": abs(
                    b["dispatch_offset_seconds"] - a["dispatch_offset_seconds"]
                ),
                "any_decision_disagreement": a.get("decisions") != b.get("decisions"),
                "positive_negative_disagreement": any(
                    {a.get("decisions", {}).get(k), b.get("decisions", {}).get(k)}
                    == {True, False}
                    for k in LABELS
                ),
            }
        )
    result = {
        "completed_requests": len(all_rows),
        "formal_requests": len(rows),
        "successes": sum(r["ok"] for r in all_rows),
        "openrouter_cost_usd": sum(
            r.get("usage", {}).get("cost", 0) or 0 for r in all_rows
        ),
        "by_case": by_case,
        "paired": {
            "count": len(paired),
            "information_first": sum(r["information_lead_seconds"] > 0 for r in paired),
            "information_lead_seconds": dist(
                [r["information_lead_seconds"] for r in paired]
            ),
            "events_slowdown_ratio": dist([r["events_slowdown_ratio"] for r in paired]),
            "grounding_slowdown_ratio": dist(
                [r["grounding_slowdown_ratio"] for r in paired]
            ),
            "dispatch_skew_seconds": dist([r["dispatch_skew_seconds"] for r in paired]),
            "any_decision_disagreements": sum(
                r["any_decision_disagreement"] for r in paired
            ),
            "positive_negative_disagreements": sum(
                r["positive_negative_disagreement"] for r in paired
            ),
        },
        "clips": clips,
        "pairs": paired,
    }
    (directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    r = analyze(p.parse_args().directory)
    print(
        json.dumps(
            {k: v for k, v in r.items() if k not in ("clips", "pairs")}, indent=2
        )
    )
