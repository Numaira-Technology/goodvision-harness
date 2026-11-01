# Threshold calibration protocol (SPEC §5/§8, expanded)

## Why 0.5 is unfair to continuous-score models

A clip is scored "take the max over frames, call it yes if that exceeds the gate."

qwen/gemini emit 0 or 1 per frame, so gate 0.5 just means "any frame was true" — there is nothing else to move, and tuning it is meaningless. luna's `noul` and CLIP's margin are different: each frame gives a continuous score, and 0.5 is a boundary borrowed from binary logic, not necessarily where this model should sit. That has to come from data.

The "take the max" step cuts both ways. Sampling more frames makes a positive easier to catch (higher recall) and a negative easier to trip (higher false positives). If one event runs hot on per-frame false positives — luna's fall does — 0.5 amplifies them, and raising that event's threshold brings them back down.

## The method: pick on held-out → freeze → evaluate on test

1. Held-out set: clips that are in no test set (`validation_split.json`, `disjoint_assert` all 0). Thresholds are picked only here.
2. Sweep per event: try τ from 0 to 1 in steps of 0.01, score validation accuracy, take the midpoint of the tied-best run. That is what `pick_tau` in `tune_threshold.py` does.
3. Freeze the chosen per-class τ.
4. Score the test set with the frozen τ and report it next to the 0.5 baseline.
5. The main table stays at 0.5. The calibrated result is a footnote or a variant — it never replaces the main table, and never enters the Drive frozen set's headline.

The one rule that matters most: thresholds may only be picked on the held-out set, never on the test set. Picking the τ that maximizes the test score is reading the answers into the model config — the number comes out inflated and falls apart on the next batch of data. The point of a held-out set is to keep "data used to pick τ" and "data used to report" strictly apart.

## Worked example (luna, v1)

Held-out set of 24 clips (6 each of fight/smoking/fall/weapon, 12 yes / 12 no, 1fps), zero overlap with the 50- and 100-clip test sets. Thresholds picked:

| Event | τ | vs 0.5 | Effect |
|---|---|---|---|
| fight | 0.29 | lower | catch more fights |
| smoking | 0.53 | ~same | — |
| fall | 0.58 | higher | cut everyday motion misread as a fall |
| weapon | 0.46 | ~same | — |

Test set (frozen τ vs 0.5): 50-clip 84% → 94%, 100-question 85% → 92%. Almost all of the gain is fall: at 0.5 it had many false positives (4 on the 50 set, 7 on the 100 set), and raising τ to 0.58 removed them.

## Commands

```sh
# 1) Score the held-out set (skip if results/luna_val24.json exists and matches the frozen split)
python tools/event_benchmark/run_eval.py --candidate luna \
  --samples <val_split>/samples.json --media <val_split>/videos \
  --out tools/event_benchmark/results/luna_val24.json

# 2) Score the test set(s) (50 / 100 / 120-base; reuse if already present)

# 3) Calibrate + evaluate (offline, reads score, no API calls)
python -I tools/event_benchmark/tune_threshold.py \
  --val  tools/event_benchmark/results/luna_val24.json \
  --test tools/event_benchmark/results/luna_1fps.json \
  --test tools/event_benchmark/results/luna_100base.json \
  --out  tools/event_benchmark/results/luna_threshold_tuning.json
```

## Small-sample caveat

The fewer held-out clips an event has, the shakier its threshold. Six per class is the current floor; always state the val n per event in the report. If an event has fewer than six, mark its τ as n/a, fall back to 0.5, and don't report a tuned number for it. After the held-out set grows, re-run and re-check `disjoint_assert`.
