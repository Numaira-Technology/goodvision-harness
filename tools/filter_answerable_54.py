#!/usr/bin/env python3
"""Derive the harness-answerable 54-question subset from the 80-question set.

The 80-question diversity set contains 26 questions this harness cannot answer:
meva/caviar semantic events (talks-to-person, opens-door, walking, vehicle
turns, ...) that fall outside the harness's fixed safety-event taxonomy
(fight / smoking / fall / weapon, see config.yaml multi_event_prompt). This
script drops those 26 (bench.manifest.classify == "no_detector") and writes the
54 the harness can actually answer, so the evaluation set matches the harness's
real capability. It is a pure subset — same rows, no re-sampling.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
from bench.manifest import classify  # noqa: E402
from bench.schema import Question  # noqa: E402

SRC = ROOT / "bench_v1_sample80"
OUT = ROOT / "bench_v1_sample54"


def main():
    OUT.mkdir(exist_ok=True)
    qs = [json.loads(l) for l in open(SRC / "questions.jsonl") if l.strip()]
    kept = [q for q in qs if classify(Question.from_json(q)) != "no_detector"]
    with open(OUT / "questions.jsonl", "w") as f:
        for q in kept:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    keep_vids = {q["video_id"] for q in kept}
    media = [json.loads(l) for l in open(SRC / "media.jsonl") if l.strip()]
    with open(OUT / "media.jsonl", "w") as f:
        for m in media:
            if m["video_id"] in keep_vids:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"kept {len(kept)}/{len(qs)} questions over {len(keep_vids)} videos")


if __name__ == "__main__":
    main()
