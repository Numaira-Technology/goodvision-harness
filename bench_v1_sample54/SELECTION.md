# Benchmark V1 — 54-question harness-answerable set

The **harness-answerable subset** of the 80-question diversity sample: the 54
questions this streaming harness can actually answer from its own detectors and
tracks. The 26 dropped questions are meva/caviar semantic events outside the
harness's safety-event taxonomy — see `benchmark/CAPABILITY_GAP.md` for the full
rationale. Regenerate with `tools/filter_answerable_54.py` (a pure subset of
`bench_v1_sample80/`, no re-sampling).

## Files
- `questions.jsonl` — 54 public questions (no answers).
- `media.jsonl` — the 48 media rows they reference.
- Feed protocol and scoring: identical to `bench_v1_sample/how_to_feed.md`.

## Coverage (diversity retained)
- **All 9 datasets**: vscrowd 11, mot17 9, mot20 8, meva 7, ntu_cctv_fights 6,
  caviar 4, rwf2000 3, ubi_fights 3, surveillance_camera_fight 3.
- **All 4 capabilities**: location_tracking 17, counting 16,
  event_recognition 12 (fight only), temporal_questions 9 (3 fight-end + 6 MOT
  count-over-time).
- **All 4 answer formats**: single_choice 17, numeric_count 16, binary_choice 12,
  numeric_seconds 9.

meva and caviar remain represented through `location_tracking` (person tracking
is event-agnostic); their semantic-event questions are the ones dropped.

## How this maps to the harness
| bucket | n | source |
| --- | --- | --- |
| fight event + fight-end temporal | 15 | scene engine `fight` (CLI path, no GPU) |
| counting / location / MOT count-temporal | 39 | SAM per-frame tracks (GPU) |

Run it exactly like the 80-set (`benchmark/README.md` → "Batch run"), pointing
`--questions`/`--media` at this directory. Every one of the 54 is answerable, so
a complete run should emit 54 predictions.
