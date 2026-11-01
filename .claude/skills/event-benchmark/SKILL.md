---
name: event-benchmark
description: >-
  Run the frozen VLM event-recognition benchmark (fight/smoking/fall/weapon on
  surveillance clips) and report every model comparison in the team's confirmed
  column format. Use this WHENEVER comparing or benchmarking vision models (Luna
  decisions, qwen-vl, gemini-flash, CLIP/YOLO) on event detection, running the
  50/120/250-clip eval set, measuring single-inference speed or 12-concurrency
  P50/P95 latency, or presenting accuracy/per-subtask success-rate tables — even
  if the user just says "run the benchmark", "compare the models", "跑一下测评",
  "测速", or "填一下那个表". The column format (single speed · P50/P95 @12 ·
  per-subtask success · overall accuracy · hardware+price) is Edwin-confirmed and
  must be used for ALL such results so tables stay consistent across meetings.
---

# Event-recognition benchmark

A controlled "exam": the ONLY variable is the model under test. Prompt, frame
sampling, gate, and scoring are frozen identically for every candidate, so a
table only ever compares models — never a model-plus-a-tweaked-setup. Two things
make this skill valuable: it keeps that exam frozen, and it forces every result
into the one reporting format the team agreed on, so numbers from different runs
and different meetings line up.

## Three model classes, one exam (read this first for a cross-class table)

The 120-question exam must hold three very different kinds of model. "Fair" is
NOT "run them all through one runner / one output" — it's **same clips, same GT,
every model answer-blind; each runs at its real production operating point, output
envelope fixed within its class.** The runners are separate and live in different
places:

| Class | Model(s) | Runner | Cadence | Output |
|---|---|---|---|---|
| language | qwen (**baseline**), gemini | our `run_eval.py` (`language`) | 1 fps | JSON 0/1 |
| decision | luna | our `run_eval.py` (`decision`) | 1 fps | noul 0–1 prob |
| decision | VideoJev-4B | `tools/videojev_probe/` + `videojev_benchmark/` (on main) | 2 frames/window @0.8s | native 6-q present/absent/insufficient |
| threshold | CLIP / SigLIP2 | `tools/clip_benchmark/` (PR #62, draft) | 1 frame @0.8s | cosine margin |

**`run_eval.py` is NOT wired to the CLIP/VideoJev backends** (PR #59 only has
language + decision). A cross-class table is each runner's own output mapped into
the one report format — not one runner over three classes. **Old 120 results
(incl. the 10/29 VideoJev run) are not new-runner results; never present them as
current.** Contributors shipped source only — no data. Full contract, cadence
caveats, and the data-sync / honesty rules: **`references/model-classes.md`**.

## When you're asked to benchmark / compare models, do this

1. **Confirm the current dataset + candidates.** The dataset path moves as the
   set grows (50 → 120 → ~250) — don't assume. Ask or check which `samples.json`
   and which model(s). Canonical runner and rules live in the repo:
   `goodvision-harness/tools/event_benchmark/`. For CLIP/VideoJev, their runners
   are separate — see the table above and `references/model-classes.md`.

2. **Run accuracy** with the repo's frozen runner (don't rewrite it):
   ```sh
   cd ~/Desktop/goodvision-harness
   python tools/event_benchmark/run_eval.py --candidate luna \
     --samples <set>/samples.json \
     --media <dir1> <dir2> <dir3> \
     --out <scratch>/<set>_luna.json
   # repeat for qwen, gemini. --media takes several dirs (set may be split).
   ```
   It auto-filters to the scored events and prints per-event TP/FN/TN/FP +
   OVERALL. This gives report columns **3 (per-subtask success)** and
   **4 (overall accuracy)**.

3. **Run latency** with this skill's harness for columns **1 and 2**. It is a
   SEPARATE measurement from accuracy — one frame, one predicate, wall-clock
   client→OpenRouter→model→back, same frame/question across models:
   ```sh
   python ~/.claude/skills/event-benchmark/scripts/latency_bench.py \
     --candidate luna --video <any representative clip> \
     --concurrency 12 --rounds 6 --out <scratch>/lat_luna.json
   # repeat for qwen, gemini. Prints single-inference median + P50/P95 @12.
   ```

4. **Report in the confirmed column format — always.** See
   `references/report-format.md`. The five columns (hardware+price merged, per
   Edwin) are fixed; fill them from the JSON outputs above. Look up current API
   prices at report time (they change). Mark provisional GT and no-negative
   events honestly.

## The rules you must not quietly change

Summarized in `references/methodology.md`; authoritative in the repo `SPEC.md`.
The ones most likely to get violated by accident:

- **1 fps sampling, gate 0.5, question from `samples.json`, `--workers 12`** —
  frozen. `--interval 4` (0.25 fps) and `--dense` exist only for
  sampling-sensitivity side-checks, never as the headline number.
- **Output format is fixed within a model class** (language = JSON yes/no,
  decision = noul probability **or** native multi-label verdict, threshold =
  cosine margin vs the model's own fixed threshold), **not aligned across
  classes**. Don't "harmonize" a decision model into yes/no to match a chat
  model, and never apply gate 0.5 to a CLIP margin (it's a cosine difference,
  not a probability). Full per-class contract: `references/model-classes.md`.
- **Expanding the dataset is fine and does NOT change the exam.** Adding a new
  *event type* (especially one with an `unknown` answer, like the bag-search
  clips) DOES change it — that needs a spec version bump and an explicit
  unknown-handling rule, and must not be folded into the binary main table.
- **Latency ≠ accuracy run.** Never quote `run_eval.py` wall-clock as the speed
  column; use `latency_bench.py`. Never claim the full product pipeline latency
  from this harness — it skips YOLO / track-binding / chips.

## Honesty guardrails (these have bitten before)

- An event with no negatives (only "yes" clips) can't have a false-positive
  rate — mark specificity `—(无负例)`, don't infer "zero false positives."
- If ground-truth answers are still `pending_user_review`, label the table
  **provisional** and offer to re-run once the GT is finalized.
- Keep meeting "example" numbers (illustrative x/N) out of the measured table;
  every measured cell should trace to a `run_eval.py` / `latency_bench.py` JSON.

## Files

- `scripts/latency_bench.py` — speed columns (single + P50/P95 @12), self-contained.
- `references/report-format.md` — the Edwin-confirmed column format + template.
- `references/methodology.md` — frozen-exam quick reference; repo `SPEC.md` is authoritative.
- `references/model-classes.md` — the three model classes (language / decision /
  threshold) on one 120-question exam: real runners + paths (CLIP `tools/clip_benchmark`,
  VideoJev `tools/videojev_probe`), per-class cadence & output, what makes it fair,
  and how to sync a contributor's results in without re-running or mis-stating them.
