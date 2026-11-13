# Benchmark report fields

Use the named protocol's saved, verified results. For current202, the headline is
correct / all 202 planned questions; separately show complete-valid questions,
missing unique frames/windows, incomplete questions, unknowns and send/valid coverage.
Per-event denominators are 32/22/62/66/10/10; bag classes are not Weapon.

Also report same-run original120 and robot82 cohorts, retaining the inherited,
pending single-view review status for robot82. Historical120 results keep their
120 denominator and 32/22/26/20/10/10 event counts.

Report model/effort, protocol/parser, provider, threshold artifact, prompt and data
hashes, source commit, run date, sampling, hardware/capacity/peak usage, actual
precision and unresolved provider-side version limits. Historical v2 outputs keep
their original interpretation; no comparison is established merely by a table format.

P50/P95 must identify their source and timing boundary. Current API reports measure
successful complete transport attempts including parsing and any metadata lookup,
excluding queue/retry wait; report successful sample count and missing separately.
An independent latency bench is a different workload and must be labelled. Keep
old speed only when explicitly requested, preserving its date/variant. Twelve slots
or workers do not prove twelve-camera end-to-end throughput.

For the retained legacy v2/v2.1 report, preserve the following measurement rules:

- Speed comes from `scripts/latency_bench.py`, not accuracy-run wall time. Same
  normalized JPEG hash and question across candidates; one request per sample,
  no retry hiding. Report success/attempt counts, failures and whether the run is
  complete. Incomplete success-only percentiles are diagnostic, not ranked headline
  speed. This measures client-to-provider round trips, not full product latency.
- Accuracy: per event and overall `correct/all questions`, errors/all questions,
  coverage. Missing or failed answers stay in the denominator. With questions but
  no valid answers, accuracy is zero; only an empty cohort is unavailable.
  No-negative events have unavailable specificity. Unknown is not false.
- Separate other runners or label their non-equivalent cadence, cohort, denominator
  and timing. Do not put sequential probe latency in @12 cells.
- Price: record date/source, currency and unit. API input/output prices ($/M tokens)
  and total GPU run cost ($) are distinct quantities; do not rank them as one cost
  metric or invent a blended token price. Unknown price stays blank.
- Attach dataset/manifest/result hashes, source commit, model revision/config,
  spec/parser version and per-block run dates. Fixed and tuned rows need distinct
  names and threshold-artifact hashes; tuning reports validation counts too.
- Historical/provisional evidence is explicitly labeled and separated from reviewed
  v2 comparisons. An existing draft manifest blocks an official v2 run.

For unified current-frame runs, price means actual run spend (USD), including the stated validation/test/retry scope.
Record known receipt cost and unknown-cost attempts separately. Provider token
quotes and estimated GPU rent are estimates, not an actual bill. Unknown is not zero.

Sheet positions are resolved from live headers using bench-sheet-contribute. Do not
assume A:W or add columns to fit a generic report. Retain before/after snapshots and
raw old results; publish only under the user's requested scope.
