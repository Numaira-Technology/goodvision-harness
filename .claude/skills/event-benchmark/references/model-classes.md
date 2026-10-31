# Three model classes on one 120-question exam (cross-contributor data sync)

The exam has to hold three very different kinds of model. Fair does not mean "force them through one
runner and one output." It means: same clips, same GT, every model answer-blind; each runs at its real
production operating point, with the output envelope fixed within its class and not aligned across classes.
Below is each class's real implementation, real cadence, output shape, how to sync someone else's results
into our report, and what you must never claim.

> Current state (2026-10, state it honestly): PR #59's 1fps `run_eval.py` only wires language + decision
> (luna). It does NOT wire the CLIP / VideoJev backends. CLIP and VideoJev run in their authors' own
> runners. So "three classes on one table" today means collecting each runner's own output into one report
> format — not one runner over three classes. Old 120 results (including the 10/29 VideoJev run) are not
> new-runner results; don't mix them in. Contributors shipped source only this round — no data, no results.

---

## A. language — qwen / gemini (our `run_eval.py`)

- **Location**: `tools/event_benchmark/run_eval.py` (PR #59 branch), backend `language`.
- **Models**: `qwen/qwen3-vl-8b-instruct` (**baseline / reference row**), `google/gemini-2.5-flash`. OpenRouter chat.
- **Output**: per frame → JSON boolean `{"answer": true/false}` → 1.0 / 0.0. Hard 0/1, no tunable threshold.
- **Cadence**: 1 fps (`--interval 1.0`), 720px wide, one frame one independent call, `--workers 12`.
- **Decision**: clip = max over frames `≥ gate 0.5`. For 0/1 that just means "any frame true," nothing to tune.
- **Threshold**: never tuned. See `bench-threshold-tune`.

## B. decision — Luna decisions / VideoJev (both give a verdict directly, but differ)

Both are system-one models that hand back a verdict, but they work differently — don't conflate them.

### B1. Luna decisions — continuous probability, held-out calibratable
- **Location**: `run_eval.py` backend `decision`; model `openai/gpt-6-luna-decisions`.
- **Output**: `noul` question type, a 0–1 calibrated probability.
- **Cadence / decision**: same as language (1fps, clip = max, gate 0.5 into the main table).
- **Threshold**: 0.5 is a borrowed boundary and unfair to a continuous score, so calibrate it per class on a
  held-out set (`tune_threshold.py`), reported as a separate `luna (calibrated)` variant with the main table
  still at 0.5. See `bench-threshold-tune`.

### B2. VideoJev-4B — native multi-label verdict, no scalar threshold
- **Location (public impl, merged to main)**: `tools/videojev_probe/` (entry
  `python -m tools.videojev_probe` `plan|prepare|download|run`), `tools/videojev_benchmark/`;
  `src/goodvision_harness/videojev_{service,runtime,questions,core,model,contract}.py`.
  Guide: `docs/guides/videojev-candidate.md`.
- **Boundary note**: the 10/29 real run used PR56's dcea006 plus a local "repackaging" six-question wiring
  (private player). The links above are NOT that full frozen package. The private wiring is not in the public
  source — don't touch colleagues' private wiring / Vast instances.
- **Model**: `ZhuYuhan/VideoJev-4B` (pinned revision), official Transformers/FlashAttention, not vLLM. BF16 + FP32 head.
- **Output**: one encode per window, **six** present / absent / **insufficient-evidence** questions grouped,
  **multi-label** (several classes can be positive at once). Option scores are uncalibrated, with no extra
  confidence threshold; the uncertain option is kept; a missing/invalid answer fails the whole window.
- **Cadence**: **2 frames per window** (original JPEGs, 2–4, no mosaic/crop), target once per 0.8s; native
  positional time uses a nominal 2.5 fps.
- **Threshold**: not tuned — keep 0 / the native verdict. There is no scalar probability to cut and no
  validation split for this class; don't hard-code one.
- **Answer-blind**: the reference is an independent Astra packet, and its verdicts never enter model input.
- **Timing boundary**: the probe's timer excludes model load / packet validation and includes preprocessing /
  inference / answer validation (plus YOLO when enabled). It is not source-to-publication time, and it is not
  a 12-stream throughput test. Label the speed column accordingly; don't present it as the product pipeline.

## C. threshold — CLIP / SigLIP2 (colleague's `tools/clip_benchmark/`, PR #62 draft)

- **Location**: `tools/clip_benchmark/` (PR #62, draft, commit `47f96db`; 5 scripts, doesn't touch main, no
  data/results attached).
  - `run.py`: validate → wait for fixed thresholds → test. Args `--config --output --thresholds --cache-dir`.
  - `runtime.py`: `ClipEncoder` loads model + fixed text; `run_streams` does native-speed input, 12-slot
    frame rotation, single-frame inference + timing.
  - `scoring.py`: `select_thresholds` picks thresholds offline, `evaluate` computes metrics + latency
    (function interface, no standalone analysis CLI).
  - `base_timing.py`: feeds frames on the real clock; `prepare.sh`: installs deps, needs a CUDA Torch first.
- **Models**: `openai/clip-vit-large-patch14`, `google/siglip2-base-patch16-224` (pinned revisions).
- **Output**: a fixed positive/negative text pair per class; per frame **margin = cos(img, positive) −
  cos(img, negative)**; per clip score = the max valid margin for that class; `score > threshold` → yes,
  else no; **no valid margin → unknown** (unknown stays in the denominator, not filled as false).
- **Key**: the margin is a **cosine difference, not a probability** — never apply gate 0.5. Its untuned
  baseline is **threshold = 0** (`evaluate`'s `zero_threshold_baseline`), not 0.5.
- **Threshold**: `scoring.py::select_thresholds` is already the correct held-out method — it picks per-class
  thresholds **on validation references only** for the four base events (fight/smoking/fall/weapon); the bag
  classes have no validation data, so their threshold stays 0, uncalibrated. Selection rule
  `max(accuracy, -FP, -FN, -abs(threshold), -threshold)`. Reuse it; don't recompute with our
  `tune_threshold.py` (that reads a 0–1 probability `score`, a different quantity from a margin). See the
  threshold-class section of `bench-threshold-tune`.
- **Cadence**: **1 frame per clip**, target once per 0.8s, 12-slot paced. Latency =
  `last_frame_ingress → classification_ready`.

---

## What makes the exam fair — invariants vs what may differ per class

**Shared invariants, never change them:**
1. **Same 120 clips + same GT.**
2. **Answer-blind**: the standard answer is never in model input (run_eval sends only the question; CLIP
   references are used only offline in scoring; VideoJev's Astra verdicts never enter model input).
3. **Scored events**: the binary main table is fight / smoking / fall / weapon only (did it happen in the clip).
4. **Growing the sample set doesn't change the exam**; a new question type (repackaging bag / a 6th class with
   `unknown`) does — version it separately, keep it out of the binary main table.

**Allowed to differ per class, and this does NOT count as "changing the exam":**
- **Output envelope**: JSON 0/1 (language) / noul probability (luna) / native multi-label verdict (VideoJev) /
  cosine margin (CLIP). Fixed within a class, not aligned across classes. Don't bend a decision model into
  yes/no to match a chat model.
- **Operating cadence**: VLM 1fps · CLIP 1 frame/clip @0.8s · VideoJev 2 frames/window @0.8s. This is the
  honest "each model at its real operating point" axis — state each one's cadence in the report; don't pretend
  a single uniform fps.
- **Threshold handling**: language fixed at 0.5 (= any frame true); luna held-out calibrated (separate variant,
  main table still 0.5); CLIP base-4 held-out, bag classes uncalibrated (τ 0); VideoJev native verdict, no threshold.

---

## Data sync: folding someone else's results into our report (no re-run, no faking)

Contributors shipped source only. To produce one cross-class table:

1. **Each runner produces its own result JSON** (we don't re-run theirs on Vast):
   - language / luna → our `run_eval.py` (+ speed via `latency_bench.py`).
   - CLIP → `clip_benchmark`'s `evaluate` output (`summary.json` / `per-question.json`, with per-event quality,
     `zero_threshold_baseline`, latency distribution).
   - VideoJev → `videojev_probe` / `videojev_benchmark` output (six present/absent/insufficient + timing).
2. **Map to the five report columns** (see `report-format.md`): per-subtask success = each runner's per-event
   x/N; overall accuracy = each runner's OVERALL. For speed/latency, state which timing it is (our wall-clock
   vs CLIP's last-frame→ready vs VideoJev's probe timer); don't merge three different timings into one column
   pretending they're comparable — note the basis in the column or split the rows.
3. **Label honestly** (each of these has bitten before):
   - A class that hasn't gone through PR #59's new runner: mark "own runner, not the unified 1fps runner";
     don't write it as a same-runner result.
   - A cell with no fresh run this round: leave it blank / mark "not run"; never pass off old 120 numbers
     (including 10/29 VideoJev) as new, and never make a number up.
   - CLIP's score is a margin — when reporting its threshold, write "held-out per-class (base-4) / bag
     uncalibrated τ=0," not 0.5.
   - A class with no validation samples (VideoJev, CLIP bag): mark the threshold uncalibrated; don't report a
     "calibrated" number.
   - To push a class's scores into the shared Google Sheet, use `bench-sheet-contribute` (self-contained, for
     any contributor), one row per model+dataset+variant; don't overwrite anyone else's row.
