---
name: bench-threshold-tune
description: >-
  Held-out calibration of a benchmark model's per-class decision threshold — the tool for onboarding
  a CONTINUOUS-score model (luna 0–1 `noul`, or CLIP/SigLIP2 cosine margin) or when a dataset/event
  changes. Fits each event's yes/no cut on a frozen held-out split and reports it as a labeled variant
  beside the gate-0.5 main table, never replacing it. Use WHENEVER asked to tune / calibrate / 调阈值 /
  校准 luna / per-class threshold / "为什么 0.5 不公平", or when a continuous-score model's fixed-0.5
  number looks unfairly low. Tunes ONLY the threshold (not prompts, sampling, or weights). Does NOT
  apply where there is no continuous score: language (qwen, gemini → hard 0/1) or VideoJev (native
  verdict). Companion to `event-benchmark` and `bench-sheet-push`/`bench-sheet-contribute`.
---

# bench-threshold-tune

Fits the per-class yes/no threshold for a **continuous-score** benchmark model on a **held-out**
validation split, per SPEC §5/§8. It makes the one legitimate post-hoc tweak possible without
breaking the controlled exam: the gate-0.5 main table stays untouched; the tuned number is a
clearly-labeled variant.

**Scope — it tunes ONLY the decision threshold** (each event's score cut), never prompts, sampling,
or model weights.

## When it applies — by what the model emits

| Class | Model | Output | This skill? |
|---|---|---|---|
| language | qwen (baseline), gemini | hard 0/1 (JSON yes/no) | **No** — no continuous score, always 0.5 |
| decision | luna decisions | 0–1 prob (`noul`) | **Yes** — `tune_threshold.py` |
| decision | VideoJev-4B | native verdict (present/absent/insufficient) | **No** — no scalar to threshold |
| threshold | CLIP / SigLIP2 | per-frame cosine margin | **Yes** — via CLIP's own `scoring.py` (not `tune_threshold.py`; never gate 0.5) |

If asked to "tune qwen/gemini" or VideoJev, decline and say why (no continuous score to cut).

## Non-negotiable protocol (full detail: `references/protocol.md`)

1. **Pick on a FROZEN held-out validation split only**, never the test set. Split:
   `tools/event_benchmark/datasets/validation_split.json`, verified disjoint (`disjoint_assert` all 0).
2. **Freeze** the chosen thresholds, then evaluate the test set with them fixed. Don't re-pick per run.
3. **The main table always stays at gate 0.5.** The tuned result is a separate labeled variant
   (`<model> (calibrated)`), shown beside the 0.5 baseline — never replacing it.
4. **Report honestly**: state val n per event; <~6 clips → `tau=n/a`, fall back to 0.5 for that event.

Baseline reference model for the benchmark: **`qwen-vl-8b`** (order/compare against it).

## How to run

**luna (decision)** — offline arithmetic over `run_eval.py`'s saved per-clip `score`, no API calls:

```sh
# val + test score JSONs come from event-benchmark's runner (reuse if already present)
python -I tools/event_benchmark/tune_threshold.py \
  --val  tools/event_benchmark/results/luna_val24.json \
  --test tools/event_benchmark/results/luna_100base.json \
  --out  tools/event_benchmark/results/luna_threshold_tuning.json
```
Grid-sweeps τ per event on the val split, takes the tied-best midpoint, prints per-class τ and test
accuracy @tuned vs @0.5. (Known-good v1 values + numbers: `references/protocol.md`.)

**CLIP / SigLIP2 (threshold)** — reuse its OWN held-out selector, do NOT route through
`tune_threshold.py`:
- Score is a **cosine margin, not a probability** → gate 0.5 is meaningless; its untuned baseline is
  **threshold = 0** (`evaluate()`'s `zero_threshold_baseline`).
- `clip_benchmark/scoring.py::select_thresholds` already fits per-class thresholds on validation
  references only (base-4 selected; bag/repackage classes with no val data stay τ=0, uncalibrated).
  Report the tuned-vs-τ=0 pair labeled "held-out per-class (base-4) / bag uncalibrated τ=0."

`run_eval.py` has no CLIP/VideoJev backend (PR #59 = language + decision only). Don't build one here;
cross-class tables sync each runner's own output — see `event-benchmark`/`references/model-classes.md`.

## Publishing

Push `<model> (calibrated)` as its OWN row (never overwriting the 0.5 baseline) via `bench-sheet-push`
or `bench-sheet-contribute`. Label: "held-out-tuned (val n=…/event), SPEC §5/§8 — not the 0.5 main table."

## Guardrails (have bitten before)

- Never tune on the test set (val/test disjoint, asserted in the split).
- Never tune a model with no continuous score (language, VideoJev).
- Never let the tuned number replace the 0.5 main table or a frozen deliverable's headline.
- Once accepted, freeze thresholds into `validation_split.json` / SPEC — don't re-pick.

## Files

- Core (do not rewrite): repo `tools/event_benchmark/tune_threshold.py` (luna) · `tools/clip_benchmark/scoring.py` (CLIP).
- Held-out split: `tools/event_benchmark/datasets/validation_split.json`.
- Protocol + worked example: `references/protocol.md`. Authoritative rules: repo `SPEC.md` §5/§8.
