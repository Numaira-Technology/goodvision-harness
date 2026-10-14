#!/usr/bin/env python3
"""Stratified diversity sampler for Benchmark V1.

Selects ~80 questions from the full bank so that every source dataset, every
capability, every answer format, and every feasible (dataset x capability) cell
is represented. Within each cell it spreads across target_labels (event
sub-types) and prefers distinct videos. Deterministic given SEED.

Reads the full private bank from the goodvision repo; writes a harness-ready
sample (questions.jsonl + media.jsonl, no answers) into goodvision-harness.
"""
import argparse, json, collections, os, random, pathlib

# The full private bank (questions.jsonl + media.jsonl) lives in the goodvision
# data repo, not here. Override with --bank / BENCH_V1_BANK; the default matches
# the sibling checkout layout.
DEFAULT_BANK = os.environ.get(
    "BENCH_V1_BANK",
    str(pathlib.Path(__file__).resolve().parents[2] / "goodvision" / "data" / "v1"),
)
DEFAULT_OUT = str(pathlib.Path(__file__).resolve().parents[1] / "bench_v1_sample80")
SEED = 20260924

# Per (dataset, capability) quota. Sums to 80. Covers all 20 feasible cells.
ALLOC = {
    ("meva", "event_recognition"): 9,
    ("meva", "location_tracking"): 7,
    ("meva", "temporal_questions"): 7,
    ("vscrowd", "counting"): 11,
    ("caviar", "event_recognition"): 6,
    ("caviar", "location_tracking"): 4,
    ("caviar", "temporal_questions"): 4,
    ("mot17", "counting"): 3,
    ("mot17", "location_tracking"): 3,
    ("mot17", "temporal_questions"): 3,
    ("mot20", "counting"): 2,
    ("mot20", "location_tracking"): 3,
    ("mot20", "temporal_questions"): 3,
    ("ntu_cctv_fights", "event_recognition"): 3,
    ("ntu_cctv_fights", "temporal_questions"): 3,
    ("rwf2000", "event_recognition"): 3,
    ("ubi_fights", "event_recognition"): 3,
    ("surveillance_camera_fight", "event_recognition"): 3,
}

def main():
    ap = argparse.ArgumentParser(description="Stratified diversity sampler for Benchmark V1")
    ap.add_argument("--bank", default=DEFAULT_BANK, help="dir with questions.jsonl + media.jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output sample dir")
    args = ap.parse_args()
    src = pathlib.Path(args.bank)
    global OUT
    OUT = pathlib.Path(args.out)

    rng = random.Random(SEED)
    Q = [json.loads(l) for l in open(src / "questions.jsonl") if l.strip()]
    M = {json.loads(l)["video_id"]: json.loads(l)
         for l in open(src / "media.jsonl") if l.strip()}

    cells = collections.defaultdict(list)
    for q in Q:
        cells[(q["source_dataset"], q["capability"])].append(q)

    used_videos = set()
    picked = []
    for cell, n in ALLOC.items():
        cands = list(cells[cell])
        rng.shuffle(cands)  # seeded tie-break
        seen_labels = set()
        chosen = []
        # Greedy: prefer new target_label, then a video not used anywhere yet.
        while len(chosen) < n and cands:
            def score(q):
                return (q["target_label"] in seen_labels,      # new label first
                        q["video_id"] in used_videos,          # new video next
                        q["video_id"] in {c["video_id"] for c in chosen})
            cands.sort(key=score)
            q = cands.pop(0)
            chosen.append(q)
            seen_labels.add(q["target_label"])
            used_videos.add(q["video_id"])
        if len(chosen) < n:
            print(f"WARN {cell}: wanted {n}, only {len(chosen)} available")
        picked.extend(chosen)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "questions.jsonl", "w") as f:
        for q in picked:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    media_vids = {q["video_id"] for q in picked}
    with open(OUT / "media.jsonl", "w") as f:
        for vid in sorted(media_vids):
            f.write(json.dumps(M[vid], ensure_ascii=False) + "\n")

    # Console summary
    print(f"picked {len(picked)} questions over {len(media_vids)} videos")
    for k in ["source_dataset", "capability", "answer_type"]:
        c = collections.Counter(q[k] for q in picked)
        print(f"\n{k}: " + ", ".join(f"{kk}={vv}" for kk, vv in c.most_common()))

if __name__ == "__main__":
    main()
