---
name: bench-threshold-tune
description: >-
  Calibrate the per-class decision threshold for a DECISION-class (Luna decisions, videojev)
  or THRESHOLD-class (CLIP, YOLO) model on the event-recognition benchmark — the right,
  held-out way. Use this WHENEVER asked to tune / calibrate / 调阈值 / 校准 luna / "给 decision
  模型调 threshold" / per-class threshold / "为什么 decision 在 0.5 下吃亏 / decision 模型用 0.5
  不公平", or right after running event-benchmark when a decision/threshold model's fixed-0.5
  score looks unfairly low and you want its true operating point. It picks per-event thresholds
  on a FROZEN held-out validation split, freezes them, evaluates on the test set, and ALWAYS
  keeps gate 0.5 as the main table — the calibrated result is a separate labeled variant, never
  a replacement. NEVER applies to language-class models (qwen, gemini): they emit hard 0/1 and
  have no tunable threshold. Companion to `event-benchmark` (produces the score JSON) and
  `bench-sheet-push` (publishes the calibrated variant row).
---

# bench-threshold-tune

Picks a per-class threshold for a **decision** or **threshold**-class model, on a **held-out**
validation set, the way SPEC §5/§8 requires. The whole value of this skill is that it makes the
one legitimate "optimization" possible **without breaking the controlled experiment**: the frozen
0.5 main table stays untouched; the calibrated number is a clearly-labeled side result.

**Why this exists.** The frozen exam scores every model at a uniform `gate = 0.5`. That is fair for
"same input, compare the model," but **0.5 is a borrowed boundary** for models that emit a
*continuous calibrated score* — decision-class (Luna `noul` probability) and threshold-class
(CLIP/YOLO logit). Their true operating point is usually not 0.5. Calibrating it on a held-out set
is the legitimate fix (it took luna from **85%→92%** on the 100-set, almost entirely by raising the
fall threshold to cut its false positives). Language-class models (qwen, gemini) emit hard 0/1 —
there is nothing to tune, and you must never pretend otherwise.

## When this applies — and when it NEVER does

| Class | Members | Output | Tune threshold? |
|---|---|---|---|
| language | qwen (baseline), gemini | hard 0/1 (JSON yes/no) | **NO — never. Always 0.5.** |
| decision | luna decisions | calibrated prob 0–1 (`noul`) | **Yes — held-out only** (this skill, `tune_threshold.py`) |
| decision | VideoJev-4B | native multi-label verdict (present/absent/insufficient) | **No scalar to threshold** — uncalibrated, no val split; use the native verdict as-is |
| threshold | CLIP / SigLIP2 | per-frame **cosine margin** (not a probability) | **Yes — held-out, in its OWN code** (`clip_benchmark/scoring.py`); never gate 0.5 |

If someone asks to "tune qwen/gemini," explain they're language-class (hard 0/1) and decline — tuning
them is meaningless and would corrupt the comparison.

## The protocol you must not break (SPEC §5/§8)

Read `references/protocol.md` for the full rules. The non-negotiables:

1. **Pick thresholds on a FROZEN held-out validation split only** — never on the test set. The split
   is `bench/event_benchmark/datasets/validation_split.json`, verified **disjoint** from every test
   set (`disjoint_assert` must be all 0).
2. **Freeze the chosen thresholds**, then evaluate the test set with them fixed. Do not re-pick per run.
3. **The main table always stays at gate 0.5.** The calibrated result is a **separate labeled variant**
   (`<model> (calibrated)`), presented alongside the 0.5 baseline — never replacing it.
4. **Report honestly.** State the val set + n per event; if an event has fewer than ~6 val clips,
   its threshold is unreliable → report `tau=n/a` and fall back to 0.5 for that event.

Baseline reference model for the whole benchmark: **`qwen-vl-8b`** (order/compare against it).

## How to run (decision class — works today)

Pure offline arithmetic — it reads `run_eval.py`'s saved per-clip `score`, **no API calls**.

1. **Get a scored validation JSON** for the model, from `event-benchmark`'s runner, over the held-out
   split (the clips in `validation_split.json`):
   ```sh
   python bench/event_benchmark/run_eval.py --candidate luna \
     --samples <val_split>/samples.json --media <val_split>/videos \
     --out bench/event_benchmark/results/luna_val24.json
   ```
   (If `results/luna_val24.json` already exists and matches the frozen split, reuse it.)

2. **Get scored test JSON(s)** the same way (the main test set, 50 / 100 / 120-base).

3. **Tune + evaluate** with the repo script (wrap it, don't rewrite it):
   ```sh
   python -I bench/event_benchmark/tune_threshold.py \
     --val  bench/event_benchmark/results/luna_val24.json \
     --test bench/event_benchmark/results/luna_1fps.json \
     --test bench/event_benchmark/results/luna_100base.json \
     --out  bench/event_benchmark/results/luna_threshold_tuning.json
   ```
   It grid-sweeps τ∈[0,1] per event, maximizes val accuracy, takes the **midpoint of the tied-best
   region** (max-margin), and prints per-class τ + test accuracy @tuned vs @0.5.

4. **Read it**: per-class τ, val @τ vs @0.5, and for each test set overall @tuned vs @0.5. Known-good
   v1: τ = fight 0.29 · smoking 0.53 · fall 0.58 · weapon 0.46 → 100-set **92% vs 85%**, 50-set **94% vs 84%**.

## Reporting & publishing

- Present `luna (calibrated)` **next to** `luna (0.5)` — both rows, baseline `qwen-vl-8b` as reference.
- Label it: "held-out-tuned (val n=…/event), SPEC §5/§8 footnote — not the frozen 0.5 main table."
- To publish, use **`bench-sheet-push`**: push the calibrated result as its OWN row (a distinct
  variant), never overwriting the 0.5 baseline row. The sheet has a `threshold-tuning` tab for this.

## Threshold class (CLIP / SigLIP2) — reuse its OWN held-out selector, don't rebuild

The CLIP/SigLIP2 benchmark (`tools/clip_benchmark/`, PR #62 draft) **already implements held-out
calibration correctly** — don't point `tune_threshold.py` at it, and don't apply gate 0.5.

- **Its score is a cosine margin, not a probability.** Per frame: `margin = cos(img, positive_text)
  − cos(img, negative_text)`; per clip: the max finite valid margin. `score > threshold` → yes. 0.5
  is meaningless here; the **untuned baseline is threshold = 0** (`evaluate()`'s `zero_threshold_baseline`),
  the analogue of our "@0.5 main table" for this class.
- **`clip_benchmark/scoring.py::select_thresholds`** fits per-class thresholds **on validation
  references only** (SPEC §5/§8 done right): it sweeps candidate cut points and picks
  `max(accuracy, -FP, -FN, -abs(τ), -τ)`. Base events (fight/smoking/fall/weapon) get selected
  thresholds; **bag/repackage classes with no validation data stay τ = 0, uncalibrated** — labeled
  as such, never reported as "calibrated."
- **`evaluate()`** freezes those thresholds and scores the test set, keeping the zero-threshold
  baseline alongside — same discipline as this skill, in CLIP's own quantity.
- **So for CLIP: reuse `select_thresholds` / `evaluate`.** `tune_threshold.py` reads a 0–1 `score`
  (luna's `noul`); a cosine margin is a different quantity and must not go through it. Report CLIP's
  tuned-vs-τ=0 pair, labeled "held-out per-class (base-4) / bag uncalibrated τ=0."

**VideoJev (decision class)** emits a native multi-label verdict with uncalibrated option scores and
**no confidence threshold** — there is no scalar to tune and no validation split for its repackage
classes. Use its verdict as-is; never invent a threshold for it.

`run_eval.py` still has no `threshold`/VideoJev backend (PR #59 = language + decision only). **Do not
build those backends from this skill.** Cross-class results come from each runner's own output,
synced into the report format — see `event-benchmark`'s `references/model-classes.md`.

## Guardrails (these have bitten before)

- **Never tune on the test set.** Val and test must be disjoint (`validation_split.json` asserts it).
- **Never tune a language model.** qwen/gemini are hard 0/1.
- **Never let the calibrated number replace the 0.5 main table** or the frozen Drive deliverable's
  headline. It's a footnote/variant.
- A tiny val set gives rough thresholds — report n per event; <6 → fall back to 0.5 for that event.
- Once thresholds are accepted, **freeze them** (into `validation_split.json` / SPEC), don't re-pick.

## Files

- Wraps repo `bench/event_benchmark/tune_threshold.py` (the tuning core — do not rewrite).
- Reads `run_eval.py` per-clip `score` from `bench/event_benchmark/results/*.json`.
- Held-out split: `bench/event_benchmark/datasets/validation_split.json`.
- Protocol detail + worked example: `references/protocol.md`.
- Authoritative rules: repo `bench/event_benchmark/SPEC.md` §5/§8.
