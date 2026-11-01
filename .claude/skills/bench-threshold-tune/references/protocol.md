# Validation-only threshold protocol

This is a separately labeled analysis protocol, not a change to the v2 fixed exam.
The current repository has no committed tuning CLI or reviewed validation split.
The following inputs and algorithm must be implemented/verified before producing
numbers; missing artifacts block execution, not this documentation's installation.

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
  If a deliberately frozen complete split lacks this support, mark that event
  `tau=n/a`, retain the fixed baseline and report it as uncalibrated. A missing
  question in a planned six-question event blocks the run instead of triggering
  fallback. Never label an incomplete run as the full frozen validation split.

## Score precision prerequisite

The current v2 runner predicts from the full-precision clip maximum, then rounds
the stored `score` to four decimals. For example, a raw 0.49996 predicts no but
is saved as 0.5. Applying `>= 0.5` to that stored value changes the answer.
Require lossless scores or full-precision evidence for an exact v2 threshold
analysis; the current saved score alone is insufficient. Do not infer the lost
precision from `pred`, or change the runner as an implicit tuning step.

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
5. Freeze a JSON artifact with per-event thresholds/fallbacks, comparator, candidate
   grid, tie rule, validation counts, split/label/media/result hashes, source commit,
   model/config/runner/spec hashes and timestamp. Hash the artifact itself. Only
   then load held-out test scores and apply it without modification.

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
Report correct/valid and coverage, per-event validation n/positives/negatives,
fallbacks, threshold hash and comparator. Show no accuracy when valid=0. For other
runners preserve unknown/error denominators; do not silently switch to v2 math.
Repeatedly adjusting the threshold/protocol after inspecting test outcomes consumes
the test set as development data; obtain a new untouched test set for the next claim.
