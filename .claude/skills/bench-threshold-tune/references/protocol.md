# Validation-only threshold protocol

This is a separately labeled analysis protocol, not a change to the v2.1 fixed exam.
The tuning CLI implements separate selection/evaluation phases; a complete reviewed
validation split is still missing. Missing artifacts block real calibration, not
offline synthetic regressions or documentation.

## Input acceptance, before selection

- Freeze a split manifest identifying every question, event, reviewed binary label,
  media SHA-256 and source-video group, and the train/development/validation/test
  assignment. Confirm validation and test were not used for training, prompt/text
  pair selection or model selection. If training membership is unknown (e.g. a
  hosted model), disclose it; do not claim a proven training-disjoint benchmark.
- Verify source-video group and content-hash disjointness across development,
  validation and test. Distinct filenames alone are insufficient. Reused footage or
  overlapping excerpts require grouping before assignment. A literal assertion field
  is not evidence. Unresolved membership/provenance blocks a leakage-free claim.
- Match every expected validation question exactly once. Reject duplicate IDs,
  unexpected questions, label/hash mismatch, nonfinite/out-of-range scores and
  any missing/error/partial-frame validation record. For v2, `frames_ok` must equal
  `frames_planned > 0` and `frames_failed` must be zero. Do not silently drop errors.
- Require reviewed labels and matching model/config/sampling/aggregation/parser
  hashes between validation and eventual test artifacts. Historical v1 or provisional
  outputs cannot be used for a v2 tuning claim.
- Record planned and observed counts per event, including positive and negative
  counts. This protocol's minimum is **six complete validation questions per event
  and at least one of each label**; six is a minimum, not statistical assurance.
  The CLI refuses a run lacking this support for any event. Such events can still
  be reported separately as fixed-baseline/uncalibrated; they must not enter a
  tuned report. A missing question always blocks selection. Never label an
  incomplete run as the full frozen validation split.

## Score precision prerequisite

Earlier v2 runners predicted from the full-precision clip maximum, then rounded
the stored `score` to four decimals. A raw 0.49996 predicted no but was saved as
0.5; applying `>= 0.5` to that value changes the answer. The current runner saves
the exact maximum and records `score_precision=lossless`; the tuner requires this
marker and verifies the saved baseline verdict. Old v1 and rounded v2 outputs are
ineligible. Do not infer lost precision from `pred` or relabel historical artifacts.

An explicitly requested quantized-score experiment must freeze the quantization
as part of its own protocol and use it on both validation and test. First verify
its recomputed 0.5 predictions match every saved baseline prediction; stop on any
mismatch. Even with agreement at 0.5, other thresholds can differ from lossless
scores. Label the experiment quantized and do not claim exact v2 equivalence.

## Luna probability selection: deterministic, validation-only

1. Input is finite numeric clip-max score in [0,1], not a boolean or string.
   Prediction is **score >= tau**, exactly as in the baseline runner.
2. Freeze candidate thresholds `{i/100 : i=0,...,100}` before selection. Compute
   each candidate's validation correct count (all expected questions, no omissions).
3. Find all grid indices with the maximum correct count. Partition them into
   **contiguous runs**. Choose the longest run; break equal-length ties by midpoint
   closest to 0.5, then lower midpoint. Set tau to the arithmetic midpoint of that
   chosen run's endpoints. Do not take the midpoint of disconnected maxima.
4. Re-evaluate validation at the chosen midpoint and verify it still attains the
   maximum. Scores may fall between grid points, so a midpoint is not guaranteed to
   remain optimal. If it is worse, use the best grid point nearest that midpoint,
   breaking ties toward the lower threshold. Store the full candidate scores,
   chosen run, final tau and tie-rule version. No test metric enters this decision.
5. Freeze a JSON artifact with per-event thresholds/support, comparator, candidate
   grid, tie rule, validation counts, split/label/media/result hashes, source commit,
   model/config/runner/spec hashes and timestamp. Hash the artifact itself. Only
   then load held-out test scores and apply it without modification.

The `select` command takes a validation result and its exact reviewed runner
manifest. For every scored question the manifest includes `human_review=user_reviewed`
and a reviewed `source_video_group`. It checks the full cohort and freezes the
selection using exclusive file creation, printing the artifact's SHA-256.
`evaluate` requires that saved hash and the held-out result/manifest pair(s), checks
matching prompts/model/config and disjoint media/source groups, and never repicks.
Manifests group reused/overlapping footage by actual origin, not by renamed files.
These checks cannot prove review truth or unknown hosted-model training membership;
report those limits and verify development disjointness independently.

A baseline of 0.5 need not optimize empirical accuracy, but that does not establish
that it is unfair. Selecting thresholds optimizes validation accuracy; it does not
make the probabilities calibrated. Do not promise a gain on held-out data.

## Margin models

A cosine margin is not constrained to [0,1]. Inspect the reviewed runner and record
its comparison operator (often strict `>`), zero baseline, text pairs, aggregation,
unknown policy and candidate generation. If using a selector with the tie tuple
`(accuracy, -FP, -FN, -abs(tau), -tau)`, freeze that rule and verify it with local
fixtures, including equal scores and no valid scores. It differs from the Luna
rule above. Validate complete held-out coverage before calling it. Events without
planned validation support stay zero/uncalibrated; missing scheduled evidence blocks
selection. Never label bag-search thresholds calibrated without validation data.

## Evaluation and reporting

Apply the frozen artifact to test results once. Publish the fixed and tuned variants
with the same test cohort, v2 eligibility policy and errors/attempted counts.
For current v2.1, report correct/all attempted questions, including errors in the
accuracy denominator but never in TP/FN/TN/FP. A nonempty all-error cohort has 0%
accuracy; no questions means no accuracy. Eligible legacy v2 retains correct/scored
accuracy and no accuracy for an event with no scored questions. Record the protocol
version, explicit denominator, attempted/scored/error counts, per-event validation
n/positives/negatives, fallbacks, threshold hash and comparator. The CLI checks totals
against rows and forbids mixing v2 and v2.1 validation/test artifacts, even if their
other metadata match. Do not relabel old archives to make them compatible. For other
runners preserve their unknown/error denominators.
Repeatedly adjusting the threshold/protocol after inspecting test outcomes consumes
the test set as development data; obtain a new untouched test set for the next claim.
