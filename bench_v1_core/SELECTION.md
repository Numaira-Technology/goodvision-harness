# Benchmark V1 — core scored set (116 questions)

The optimized evaluation set: harness-answerable, sub-form-aligned, balanced, and
sized for a per-capability score. Regenerate with `tools/sample_bench_v1_core.py`
(deterministic, SEED=20260926). Supersedes the earlier diversity-then-filter sets
(`bench_v1_sample80/`, `bench_v1_sample54/`), which sampled blind to capability
and contained relative-position location questions the runner silently
mis-answered.

## Why this set differs
Sampling is **sub-form aware**, not just capability aware. `location_tracking` is
three different tasks — busiest-third, relative-horizontal, relative-vertical —
and only questions whose sub-form has a working extractor
(`benchmark/bench/extract.py`) are eligible. It also optimizes for a fair score:

- **Answer-prior balancing** (uses the private answers): choice questions are
  balanced so a constant-guess baseline cannot win.
- **Difficulty spread**: counting is binned by crowd density; location/temporal
  spread across sub-forms.
- **Independence**: at most one question per video *within* a capability (temporal
  relaxes, as its answerable videos are scarce).

## Coverage (116 questions / 91 videos)
| capability | n | majority-class baseline | notes |
| --- | --- | --- | --- |
| counting | 30 | ~10% | density buckets low/med/high, leans vscrowd (independent) |
| event_recognition | 30 | 50% | fight yes/no, exactly 15/15 |
| location_tracking | 30 | ~20% | busiest-third 8 + rel-horizontal 16 + rel-vertical 6, options balanced |
| temporal_questions | 26 | ~8% | fight-end 16 + MOT count-over-time 10 (only ~28 answerable videos) |

All 9 source datasets are represented (vscrowd, mot17, mot20, meva, caviar, ntu,
rwf2000, ubi_fights, surveillance_camera_fight). The majority-class baseline is
the score a constant guesser would get — report the harness against it.

## Scoring caveat — normalize choice vocabularies
location ground truth mixes vocabularies: Chinese `左侧/中央/右侧` for busiest-third,
English `left/right/up/down` + `approximately_same_*` for relative displacement.
The extractor emits the matching vocabulary per sub-form, but the eval side must
apply casefolded exact match **within the same vocabulary** — do not compare a
Chinese prediction to an English label.

## Running it
Identical to `benchmark/README.md` → "Batch run", pointing `--questions`/`--media`
at this directory. Relative-displacement location needs the capture to carry the
source frame `width`/`height` (to normalize the pixel reference box); counting and
event do not. Every one of the 116 is answerable.
