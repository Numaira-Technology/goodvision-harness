# threshold-tuning-latest: latest verified calibration

`threshold-tuning-latest` explains each model's adopted threshold and the requested
comparison. It is not a second leaderboard or an append-only run log.
Keep historical artifacts and pre-write snapshots locally. Do not delete them
when replacing a displayed record, and do not recreate the retired evaluation tab.
The original `threshold-tuning` retains the historical records: it was restored
after the user requested the organized version in a separate tab. Do not replace
the historical page with the latest overview or synchronize it implicitly.

## Identity and sources

Choose the latest verified adopted calibration for the requested model/variant,
not the artifact with the highest test accuracy or the latest file modification
time. Match model revision, dataset/split fingerprints, prompt/parser, provider,
effort, scoring rule, sampling, precision and source run to the saved reports.
Different scores or configurations are distinct experiments, not threshold-only
comparisons. For numerically calibrated models, a new run without verified
calibration does not silently replace the last valid one. Mark missing evidence
instead of inventing a threshold or result.

For numerically calibrated models, require the frozen threshold artifact,
validation selection report, matching test report when available, and source hashes.
For boolean/native-argmax models, use the frozen configuration and decision rule,
available validation/test reports, and source hashes; no threshold artifact or
numerical selection report exists or is required. For an uncalibrated fixed
baseline, cite its explicit configured threshold and label it uncalibrated.
A threshold-only comparison
needs the named baseline scored on the same saved predictions/scores. A requested
historical-test comparison instead cites the actual earlier run/report and states
configuration and scoring differences; do not relabel it as threshold-only.
Publication does not run models or select thresholds again.

## Sections and fields

The live layout organized on 2026-11-09 has the following sections. Re-read their
headers and model identities before each write; row/column positions are not an API.

| Section | Fill from evidence |
| --- | --- |
| Latest overview | Model/variant, exact decision rule (`>` versus `>=`), inference date, per-event thresholds, calibration status, validation correct/planned, test correct/planned, test accuracy |
| Requested comparison | Explicit historical owner/run/version or same-score baseline, correct counts/accuracy for both, difference in percentage points, FP/FN and completeness/scoring limitations |
| No numerical threshold | Models with boolean outputs or native verdict/argmax, their actual decision method and validation purpose; leave threshold values inapplicable |
| Collapsed event details | Selected threshold, optimal interval bounds and endpoint inclusion, selection rule/tie-break, validation count, GT yes/no counts, correct count, accuracy, FP/FN |
| Notes | Artifact paths and SHA-256, code/model identity, inference versus calibration dates, completeness/unknown counts, limitations and archive location |

In numeric threshold cells, store full-precision values and use display formatting
to shorten them. If the existing layout uses a combined text field (such as
`old → new`), preserve that layout and keep full values and selection details in
the artifact and notes; do not add columns to force a numeric-cell layout.
Merge bag thresholds into one display field only if both events really share the
same value; otherwise show both explicitly in that field/notes without adding
columns automatically. Missing bag validation means fixed/un-calibrated, not tuned.

For current202, test accuracy is `correct / 202 * 100`; historical120 keeps
`correct / 120 * 100`. Both include
incomplete questions in the planned denominator. Validation uses its planned24
or the explicitly documented subset; event counts use that event's planned count.
Show complete-valid counts separately. Do not turn legal unknown into a request
failure or silently count it as no. Derive correctness and confusion counts from
the run's documented scorer; FP/FN need not account for unknown/incomplete units.

Comparison delta is `current accuracy - reference accuracy`, in percentage points.
The user currently requested Ashley/Pengyu's published calibrated historical
tests as the reference (CLIP/SigLIP2 midpoint reanalysis; Luna historical tuned
table). Keep that identity and its provenance explicit. Historical Luna accepted
partial frame evidence, while the current run is complete; its old table score is
not proof of a complete historical run or a matched-protocol comparison.
Do not use independently run old test results as a threshold-only baseline. Record
fixed0/fixed0.5, a named historical threshold, or another requested baseline clearly.
Never choose a threshold because its test result wins; report decreases as well as
increases. Preserve small-validation, prompt-selection and retrospective-calibration
limitations from the source record.

## Updating the page

1. Read the live sections, identities, notes, formulas, formats and groups. Save
   before-values/notes/formats and source hashes locally.
2. Prepare a scoped mapping for the overview, comparison and detail rows together.
   Do not update only the headline while leaving a different threshold below it.
3. Verify correct/planned arithmetic, delta, completeness, confusion counts and
   full-precision threshold equality with the frozen artifact. Inapplicable or
   unavailable evidence stays labelled as such; it is not zero.
4. Re-read before writing. Update the latest record in `threshold-tuning-latest` within the user's
   requested scope; retain history locally. Only append a historical section or
   restructure the tab when explicitly requested. Do not sync `results` implicitly.
5. Follow [publication formatting](publication-formatting.md): Calibri9pt, current
   comparable metric ranges, preserve ordinary styles and collapsed details.
   Threshold magnitude has no good/bad color. For change in accuracy, anchor zero
   neutrally and distinguish negative/positive changes; do not rank different models'
   threshold values. Record color anchors with the local change plan.
6. Read back values, notes, raw threshold precision, effective colors/font and group
   state. Check that the target overview and comparison agree and unrelated tabs
   are unchanged; visually inspect any changed layout before delivery.

No generic writer currently enforces this contract. The historical row validator
and saved column indexes do not validate the live threshold page.
