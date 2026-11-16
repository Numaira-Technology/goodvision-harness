# Benchmark report fields

Use the named protocol's saved, verified results. For current202, the headline is
correct / all planned questions (202 baseline, 222 with the current supplement); separately show complete-valid questions,
missing unique frames/windows, incomplete questions, unknowns and send/valid coverage.
Baseline per-event denominators are 32/22/62/66/10/10; the current test supplement
changes them to 32/22/62/66/20/20. Use the frozen plan counts; bag classes are not Weapon.

Also report same-run original120 and robot82 cohorts, retaining the inherited,
pending single-view review status for robot82. Include the separate test_supplement
cohort (20 for the current supplement), retaining its assistant-annotation provenance.
Original API base100/bag20 remain 100/20. Historical120 results keep their
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

Price means actual run spend (USD), including the stated validation/test/retry scope.
Record known receipt cost and unknown-cost attempts separately. Provider token
quotes and estimated GPU rent are estimates, not an actual bill. Unknown is not zero.

Sheet positions are resolved from live headers using bench-sheet-contribute. Do not
assume A:W or add columns to fit a generic report. Retain before/after snapshots and
raw old results; publish only under the user's requested scope.
