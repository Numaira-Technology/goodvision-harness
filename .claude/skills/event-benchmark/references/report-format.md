# Benchmark report fields

Use the named protocol's saved, verified results. For current120, the headline is
correct / all 120 planned questions; separately show complete-valid questions,
missing unique frames/windows, incomplete questions, unknowns and send/valid coverage.
Per-event denominators are 32/22/26/20/10/10; bag classes are not Weapon.

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
