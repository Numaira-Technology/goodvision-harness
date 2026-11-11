---
name: bench-sheet-contribute
description: >-
  Prepare, validate and publish explicitly requested GoodVision benchmark cell
  updates against live Google Sheets headers. Preserve result identity, old
  artifacts, measurement provenance and existing formatting.
---

# Benchmark result publication

Spreadsheet: `1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA`.
Resolve the requested tab (`results`, `eval-2026-11-04`, or `threshold-tuning`),
then read live metadata, headers, model rows, formulas, notes and formatting.
Never infer current positions from a saved column letter or a model's first match.

Read [eval semantics](references/eval-column-contract.md) or
[results semantics](references/results-column-contract.md) for the destination.
The [historical A:W map](references/column-contract.md), bundled `validate_row.py`,
and repository A:Q `sheet_config.json` do not describe the current live layout.
There is no current-layout reusable writer/validator; retain a range-specific
mapping and arithmetic audit instead of claiming a legacy validator covers it.

## Prepare the change

- Use saved verified reports and source fingerprints. No model calls to fill blanks.
- Current120 headline is **correct / all 120 planned questions**, including missing
  questions in the denominator. Base100 and each event use their planned subset
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
- Threshold values and calibration status belong in notes; don't turn Luna 0.5
  or a previous tuned value into an unexplained default. Use bench-threshold-tune.

## Write and verify

Save before-values and a sparse proposed cell/note diff locally. Follow the user's
publication authorization: prepare a preview first when requested; otherwise update
only the agreed cells/rows. Do not add columns, tabs or duplicate rows automatically.
`results` and `eval` are separate destinations; update both only when requested.
For `threshold-tuning`, append below existing records when that is the requested scope.

Re-read target cells immediately before the write and stop on competing changes.
Preserve formulas, existing color rules, ordinary backgrounds, headers, grouping,
freezes and dimensions unless that specific layout change was requested. Write
literal values with RAW input and narrow note/format masks. Re-read the saved cells
and verify values, formulas, notes and identity. Preserve all previous local results.
