---
name: bench-threshold-tune
description: >-
  Select per-event thresholds for continuous benchmark scores on reviewed validation
  data, freeze the artifact, and compare tuned versus baseline test results offline.
  Use for tune/calibrate/调阈值/校准 requests. Does not tune hard booleans or native argmax.
---

# Validation-only threshold selection

Use the runner matching the saved artifact. For current120 ledgers, read
`tools/unified_benchmark/README.md` and use `tools.unified_benchmark.workflow select`
and `report`. For the older base-four `per_video` schema, use
`tools/event_benchmark/tune_threshold.py` (v2 and v2.1 accepted separately; never
mix their validation/test versions). The latter retains historical complete-only
confusion metrics and explicitly adds all-question headline counts.

Require reviewed data, lossless scores, full validation coverage, and compatible
model/provider, prompts, effort, precision, sampling, preprocessing and source
fingerprints. Validate source-video groups across splits. Training/development,
actor/scene and prior test-use limits must remain explicit; do not invent a clean
training split or claim the reused test is untouched.

Validation is 24 questions: six per base event, three positive and three negative.
There is no bag validation. Incomplete validation stops calibration and test;
a fixed baseline fallback needs an explicit separate experiment decision.
The test and report entrypoints reject failed/null threshold artifacts before
execution: successful selection, complete validation24 counts, matching frozen
plan/evidence identity and six finite event thresholds are required. The current
execution/selection code must also match the bundle's frozen source fingerprints.

| Model | Selection and decision rule |
|---|---|
| Luna | Grid 0..1 in 0.01 steps; highest validation correct count; longest contiguous optimal grid run; midpoint; equal-length runs prefer closest to 0.5 then lower midpoint. Recheck midpoint optimality; stop if worse. Predict yes at score >= threshold. |
| CLIP / SigLIP2 | Positive-minus-contrast cosine margin; maximize validation accuracy then minimize FP and FN; midpoint of widest finite optimal interval, lower midpoint on ties. Predict yes at margin > threshold. |
| Qwen / Gemini / Llama | Hard boolean output; no numerical threshold to tune. Validation compares prompt/effort settings. |
| VideoJev / Decision3 | Native yes/no/unknown argmax; no scalar threshold. |

Bag thresholds stay Luna 0.5 and CLIP/SigLIP2 0, explicitly uncalibrated. Do not
borrow a base-class threshold or label these bag settings validation-selected.
See [algorithm details](references/protocol.md) for interval boundaries.

Select using validation only, save/hash thresholds before the test phase, and
apply them unchanged. For authorized re-scoring of already available test scores,
freeze selection before reading those scores and label the comparison retrospective.
Do not choose a published threshold by whichever has better test accuracy.

Preserve fixed 0.5 for Luna and zero for margin models as comparisons; a user may
publish the labelled tuned variant in the existing row. The latest recorded Luna
base thresholds were fight 0.21, smoking 0.505, fall 0.34, weapon 0.12. These are
historical calibration outputs, not defaults for another model/provider/prompt run.
The recorded comparison was validation 20/24 -> 24/24 and test 104/120 -> 106/120;
raw evidence remains in the originating run, not bundled here as new results.

Headline accuracy uses all planned questions, with incomplete/unknown counts
separate. Preserve original artifacts; write new reports without model calls for
an offline replay. Publishing is a separate requested step via bench-sheet-contribute.
