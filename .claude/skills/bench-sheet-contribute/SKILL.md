---
name: bench-sheet-contribute
description: >-
  Prepare, validate and publish explicitly requested GoodVision benchmark cell
  updates against live Google Sheets headers. Preserve result identity, old
  artifacts and measurement provenance; apply the agreed publication formatting.
---

# Benchmark result publication

Spreadsheet: `1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA`.
Resolve the requested tab (`results`, `threshold-tuning-latest` or historical
`threshold-tuning`),
then read live metadata, headers, model rows, formulas, notes and formatting.
Never infer current positions from a saved column letter or a model's first match.
`eval-2026-11-04` is retired by the user: do not read it for publication, synchronize
to it or recreate it. Existing local snapshots remain historical evidence only.

Read [results semantics](references/results-column-contract.md) for result updates.
Read [threshold publication semantics](references/threshold-tuning-column-contract.md)
for the latest-threshold overview, same-score comparisons and calibration details.
The [historical A:W map](references/column-contract.md), bundled `validate_row.py`,
and repository A:W `sheet_config.json` do not describe the current live layout.
There is no current-layout reusable writer/validator; retain a range-specific
mapping and arithmetic audit instead of claiming a legacy validator covers it.

This skill covers the named GoodVision workbook, not arbitrary spreadsheets.
Results and threshold publication have field contracts. Map the saved evidence
to live headers: calibrated numerical thresholds require a tuning artifact;
boolean/native-argmax rows require their frozen decision configuration, not a
nonexistent threshold artifact. An unrecognized
tab/metric needs its meaning and source resolved before any write. Do not fill
unsupported fields, missing measurements or unrun cohorts just to complete a row.

## Prepare the change

- Use saved verified reports and source fingerprints. No model calls to fill blanks.
- Current202 headline is **correct / all 202 planned questions**, including missing
  questions in the denominator. Report original120/robot82 separately; retain the
  inherited-label limitation. Never put a202 headline under an all120 header.
  Historical120 and Base100 and each event use their planned subset
  denominator. Report complete-valid counts separately. Older success-only results
  keep their original labelled identity; never silently reinterpret them.
- Model names include selected effort, e.g. Gemini 2.5 Flash off / low. A display
  name alone is insufficient: match dataset, prompt/parser, effort, provider,
  threshold artifact, source run and date. Keep variants separate when both requested.
- Hardware and VRAM share one display value where the current header supports it,
  e.g. `RTX 4090 - 24G`. Capacity is not measured peak VRAM. API hardware is unknown.
- Price is the actual run spend in USD, with scope (validation/test/retries) and
  receipt completeness in notes. Unknown charges are not zero. GPU spend needs a
  real bill/rate-duration basis; do not put an old token quote in an actual-cost cell.
- Completeness counts unique planned units, not retry attempts. Successful retry
  removes final missing; historical failures remain in logs. Preserve missing units,
  incomplete questions and valid unknown as distinct quantities.
- Latency must identify the corresponding prompt/effort/provider, workload, sample
  count and timing boundary. Prefer the matching measured run; when the user asks
  to retain existing speed, keep its provenance and disclose any mismatch. Twelve
  workers are not proof of twelve-camera real-time throughput.
- In `results`, threshold values and calibration status belong in notes; in
  `threshold-tuning-latest`, populate its explicit threshold fields and provenance notes.
  Don't turn Luna 0.5 or a previous tuned value into an unexplained default.
  Use bench-threshold-tune for selection, not the publication step.

## Write and verify

Read [publication formatting](references/publication-formatting.md): written cells
use **Calibri, 9 pt**, and metric colors are recalculated from the current
comparable result block on each publication. This does not authorize a live write
when the user requested only documentation or a preview.

Save before-values and a sparse proposed cell/note/format diff locally, including
the color-scale ranges and anchors. Follow the user's
publication authorization: prepare a preview first when requested; otherwise update
only the agreed cells/rows. Do not add columns, tabs or duplicate rows automatically.
Publish evaluation results to `results`; do not maintain a second evaluation tab.
For `threshold-tuning-latest`, maintain the latest verified record for each model/variant
and its aligned comparison/detail rows. Back up the replaced values and notes.
Append history only when specifically requested; do not grow another experiment
block on each routine update. The original `threshold-tuning` was restored as the
historical page; leave it unchanged unless that page is explicitly requested.

Re-read target cells immediately before the write and stop on competing changes.
Preserve formulas, ordinary backgrounds, header layout, grouping, freezes and
dimensions. Apply only the font and metric-color changes defined in the formatting
reference; preserve unrelated conditional-format rules and historical blocks. Write
literal values with RAW input and narrow note/format masks. Re-read the saved cells
and verify values, formulas, notes, font, effective colors and identity. Preserve
all previous local results.
