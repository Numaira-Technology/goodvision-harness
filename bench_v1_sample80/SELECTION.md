# Benchmark V1 — 80-question diversity sample

Stratified subset of the full Benchmark V1 bank (1351 QA over 650 videos),
selected to feed the in-house streaming harness. Same row schema and feed
protocol as `bench_v1_sample/` (see that dir's `how_to_feed.md`); only the
number of items differs. Answers stay private on the eval side.

## Files
- `questions.jsonl` — 80 public questions (no answers), one JSON object per line.
- `media.jsonl` — the 74 media rows those questions reference.
- Regenerate with `tools/sample_bench_v1_80.py` (deterministic, `SEED=20260924`).

## Selection method
Full bank = 9 datasets × 4 capabilities, but only **18 (dataset × capability)
cells are non-empty** (e.g. meva has no counting; the three fight datasets have
only event_recognition). We assign a quota to each of the 18 feasible cells so
that coverage is guaranteed, then within each cell greedily pick to (1) spread
`target_label` / event sub-types and (2) prefer videos not yet used.

Answer format is determined by capability, so covering all 4 capabilities
covers all 4 answer formats automatically.

## What the 80 items cover
- **All 9 source datasets**: meva 23, caviar 14, vscrowd 11, mot17 9, mot20 8,
  ntu_cctv_fights 6, rwf2000 3, ubi_fights 3, surveillance_camera_fight 3.
- **All 4 capabilities**: event_recognition 27, temporal_questions 20,
  location_tracking 17, counting 16.
- **All 4 answer formats**: single_choice 27, numeric_seconds 20,
  binary_choice 17, numeric_count 16.
- **All 18 feasible (dataset × capability) cells.**
- **74 distinct videos** (reuse only in mot17/mot20, whose whole pools are 7 and
  4 videos); **22 distinct target_labels** including meva/caviar sub-events
  (vehicle turns, door/phone/reading events, walking/shop-enter, etc.).

## Feeding the harness
Identical to the 4-question sample: join `question.video_id → media.video_id`,
replay each video as the input stream (files directly; `image_sequence` folders
assembled at their stated fps), use the question text (+ options block when
`response_options` is present) as the prompt, emit one
`{"qa_id": ..., "prediction": ...}` per line. Do not feed answers to the model.
