---
name: bench-threshold-tune
description: >-
  Select per-event thresholds for continuous benchmark scores on a frozen validation
  split, then evaluate a separately labeled fixed-threshold variant on held-out test
  data. Use for tune/calibrate/调阈值/校准 requests. Keeps the fixed baseline and never
  selects on test results; language booleans and native VideoJev verdicts are out of scope.
---

# Held-out threshold selection

This skill defines an **offline analysis protocol**, not permission to run models
or publish results. Read the current `tools/event_benchmark/SPEC.md` (v2) and
`references/protocol.md` first. The main Luna/language result remains gate 0.5;
CLIP margins require their own fixed baseline (normally zero, never borrowed 0.5).

## Availability and inputs

`tools/event_benchmark/tune_threshold.py` provides separate `select` and `evaluate`
commands. The historical `datasets/validation_split.json` is a candidate list,
not a reviewed v2 manifest. A complete reviewed calibration split is still missing.
CLIP tooling is a separate PR, not a dependency installed by this skill.

Require a reviewed split manifest and saved score artifacts with their original
hashes. If these are missing, report exactly which input is needed and stop tuning;
you can still prepare the offline analysis plan. Historical v1 scores are not v2
calibration inputs. No worked threshold or claimed accuracy gain is supplied here.

## Workflow

1. Freeze train/development, validation and test membership before looking at scores.
   Check media hashes and source-video groups, not just filenames or a saved
   `disjoint_assert` flag. No overlap or near-duplicate excerpts across splits.
2. Verify exact validation question coverage, reviewed labels, positive/negative
   support and complete frame evidence. Missing/error clips are blockers, not
   permission to fit a smaller successful subset. This CLI requires six complete
   questions and both labels for every event and refuses unsupported events; report
   their fixed baseline separately rather than fitting an incomplete subset.
3. Select using validation scores only, following the grid/comparator/tie rule in
   `references/protocol.md`. Keep test scores unavailable during selection.
4. Save and hash the thresholds plus validation provenance **before** opening test
   predictions. Freeze this artifact. Apply it unchanged; do not repick after seeing
   test outcomes. A later selection round needs new untouched evaluation evidence.
5. Report fixed-vs-tuned test results and coverage, validation support, threshold
   provenance and uncertainty. The result is a variant, not a replacement baseline
   and not evidence that the score scale is probabilistically calibrated.

For Luna, require lossless clip-max scores for an exact threshold analysis. The
current runner saves lossless scores and records `score_precision=lossless`.
Earlier v2 outputs rounded `score` to four decimals after prediction; they are
rejected, even if their rounded baseline happens to agree. Never add the marker
to an old archive. See the precision prerequisite in the protocol. Do not tune
qwen/gemini hard booleans or native VideoJev
verdicts. For margins, verify the actual backend comparator and candidate grid;
never feed margins into a probability-grid selector.

Use the commands in `tools/event_benchmark/RESULTS.md`: `select --val ...
--val-manifest ... --out ...` reads no test scores and creates a new immutable
artifact. Save its printed hash before obtaining/opening test predictions, then
run `evaluate --thresholds ... --thresholds-sha256 ... --test ...
--test-manifest ... --out ...`. Both manifests need reviewed `source_video_group`
values. Exact question coverage, labels, hashes, prompts and group disjointness
are checked; training/development membership still requires independent review.

## Publishing

Only when requested, prepare a distinct model/protocol/variant identity through
`bench-sheet-contribute`, e.g. `luna / v2 / val-tuned <artifact hash>` beside
`luna / v2 / fixed 0.5`. Include fallback events and validation n/positive/negative
counts. Do not edit the frozen manifest, spec or baseline row to store a new fit.
Store analysis and threshold artifacts in ignored `outputs/`.
