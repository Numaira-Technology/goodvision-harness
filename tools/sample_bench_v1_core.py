#!/usr/bin/env python3
"""Optimized core sampler for Benchmark V1 (harness-answerable, balanced).

Supersedes the diversity-then-filter approach (sample_bench_v1_80 +
filter_answerable_54), which sampled blind to harness capability and produced an
unbalanced survivor set with a silent mis-answer bug (relative-position location
questions scored as busiest-third).

This sampler is capability- and SUB-FORM-aware from the start, and optimizes for
a fair, statistically usable score:

1. Answerability at the SUB-FORM level: keep only questions whose exact sub-form
   has a working extractor (bench.extract) —
     counting: peak count · event: fight yes/no · location: busiest-third +
     relative displacement (h/v) · temporal: fight-end + MOT count-over-time.
2. Per-capability quotas sized to the independent-video budget (temporal is
   video-starved: ~28 independent videos, so it is capped below 30).
3. Answer-prior balancing for choice questions (uses the private answers), so a
   constant-guess baseline cannot score well. Reports the majority baseline.
4. Difficulty spread: counting binned by crowd density; location/temporal spread
   across sub-forms.
5. Independence: at most one question per video (temporal relaxes to two, since
   its answerable videos are scarce).

Deterministic given SEED. Reads the private bank (questions + answers + media);
writes public questions/media (no answers) + a selection report.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
from bench.extract import parse_count_temporal  # noqa: E402

DEFAULT_BANK = os.environ.get(
    "BENCH_V1_BANK", str(ROOT.parent / "goodvision" / "data" / "v1"))
DEFAULT_OUT = str(ROOT / "bench_v1_core")
SEED = 20260926


# ---- sub-form classification (must match extract.answer_question dispatch) ----

def loc_subform(q: dict) -> str | None:
    o = set(q.get("response_options") or ())
    if o == {"左侧", "中央", "右侧"}:
        return "busiest_third"
    if "approximately_same_horizontal_position" in o:
        return "rel_h"
    if "approximately_same_vertical_position" in o:
        return "rel_v"
    return None


def temporal_subform(q: dict) -> str | None:
    if q.get("target_label") == "physical_conflict":
        return "fight_end"
    return "count" if parse_count_temporal(q["question"]) is not None else None


def is_answerable(q: dict) -> bool:
    cap = q["capability"]
    if cap == "counting":
        return True
    if cap == "event_recognition":
        return q.get("target_label") == "physical_conflict"
    if cap == "location_tracking":
        return loc_subform(q) is not None
    if cap == "temporal_questions":
        return temporal_subform(q) is not None
    return False


# ---- balanced greedy picker --------------------------------------------------

def pick(cands, n, used, gt, rng, gt_quota=None, allow_reuse=False):
    """Pick n questions, balancing GT values and preferring unused videos."""
    rng.shuffle(cands)
    chosen, taken_gt = [], collections.Counter()
    ds_count = collections.Counter()

    def ok_video(q):
        return allow_reuse or q["video_id"] not in used

    def score(q):
        # lower is better: honor gt quota, then fresh video, then dataset spread
        over = gt_quota is not None and taken_gt[gt(q)] >= gt_quota.get(gt(q), 0)
        return (over, q["video_id"] in used, ds_count[q["source_dataset"]])

    pool = [q for q in cands if ok_video(q)]
    while len(chosen) < n and pool:
        pool.sort(key=score)
        q = pool.pop(0)
        if gt_quota is not None and taken_gt[gt(q)] >= gt_quota.get(gt(q), 0) \
                and sum(gt_quota.values()) >= n:
            # quota for this value full and quotas can still satisfy n -> skip it
            remaining = [c for c in pool if taken_gt[gt(c)] < gt_quota.get(gt(c), 0)]
            if remaining:
                pool = remaining
                continue
        chosen.append(q)
        taken_gt[gt(q)] += 1
        ds_count[q["source_dataset"]] += 1
        used.add(q["video_id"])
        pool = [c for c in pool if ok_video(c) and c["qa_id"] != q["qa_id"]]
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Optimized core sampler for Benchmark V1")
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    src = pathlib.Path(args.bank)
    out = pathlib.Path(args.out)
    rng = random.Random(SEED)

    Q = [json.loads(l) for l in open(src / "questions.jsonl") if l.strip()]
    ANS = {json.loads(l)["qa_id"]: json.loads(l) for l in open(src / "answers.jsonl") if l.strip()}
    M = {json.loads(l)["video_id"]: json.loads(l) for l in open(src / "media.jsonl") if l.strip()}
    A = [q for q in Q if is_answerable(q)]
    gt = lambda q: str(ANS[q["qa_id"]]["private_ground_truth"]["value"])  # noqa: E731

    picked: list[dict] = []
    # Independence is enforced WITHIN each capability (that is what a per-capability
    # metric needs); a video may still serve two different capabilities, which is
    # unavoidable given the scarce MOT/ntu pools.

    # counting: density buckets, lean vscrowd for independence
    used_c: set[str] = set()
    cnt = [q for q in A if q["capability"] == "counting"]
    def bucket(q):
        v = float(gt(q))
        return "low" if v < 10 else "med" if v <= 40 else "high"
    for b, k in (("low", 3), ("med", 13), ("high", 14)):
        picked += pick([q for q in cnt if bucket(q) == b], k, used_c, gt, rng)

    # event: fight yes/no balanced
    used_e: set[str] = set()
    ev = [q for q in A if q["capability"] == "event_recognition"]
    picked += pick(ev, 30, used_e, gt, rng, gt_quota={"yes": 15, "no": 15})

    # location: busiest-third + relative displacement, balanced within sub-form
    used_l: set[str] = set()
    loc = [q for q in A if q["capability"] == "location_tracking"]
    picked += pick([q for q in loc if loc_subform(q) == "busiest_third"], 8, used_l, gt, rng,
                   gt_quota={"左侧": 3, "中央": 2, "右侧": 3})
    picked += pick([q for q in loc if loc_subform(q) == "rel_h"], 16, used_l, gt, rng,
                   gt_quota={"left": 6, "right": 6, "approximately_same_horizontal_position": 4})
    picked += pick([q for q in loc if loc_subform(q) == "rel_v"], 6, used_l, gt, rng,
                   gt_quota={"up": 2, "down": 2, "approximately_same_vertical_position": 2})

    # temporal: fight-end + count-over-time; video-starved -> allow 2/video for count
    used_t: set[str] = set()
    temp = [q for q in A if q["capability"] == "temporal_questions"]
    picked += pick([q for q in temp if temporal_subform(q) == "fight_end"], 16, used_t, gt, rng)
    picked += pick([q for q in temp if temporal_subform(q) == "count"], 10, used_t, gt, rng,
                   allow_reuse=True)

    # write outputs
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "questions.jsonl", "w") as f:
        for q in picked:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    vids = {q["video_id"] for q in picked}
    with open(out / "media.jsonl", "w") as f:
        for v in sorted(vids):
            f.write(json.dumps(M[v], ensure_ascii=False) + "\n")

    # report
    print(f"picked {len(picked)} questions over {len(vids)} videos")
    for cap in ["counting", "event_recognition", "location_tracking", "temporal_questions"]:
        qs = [q for q in picked if q["capability"] == cap]
        prior = collections.Counter(gt(q) for q in qs)
        top = max(prior.values()) / len(qs) if qs else 0
        print(f"  {cap:20} n={len(qs):3d}  majority-baseline={top:.0%}  GT={dict(prior)}")
    dsc = collections.Counter(q["source_dataset"] for q in picked)
    print("  datasets:", dict(dsc))
    reuse = {v: c for v, c in collections.Counter(q["video_id"] for q in picked).items() if c > 1}
    print("  videos reused:", reuse or "none")


if __name__ == "__main__":
    main()
