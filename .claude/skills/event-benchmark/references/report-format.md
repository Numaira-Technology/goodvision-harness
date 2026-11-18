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

P50/P95 must identify their source and timing boundary. Current reports follow the
latency definition in `tools/unified_benchmark/README.md`: request sent to response
parsed, nearest-rank, metadata lookup and receipt writes excluded. Report successful
n, failed and timeout attempts, unresolved attempts and missing successful timings,
and `p95_with_failures_seconds` with its sample count beside P50/P95. The penalized
P95 is unavailable when outcomes or successful timings are missing; old-ledger
fallback counts must accompany timings that include whole-attempt overhead.
An independent latency bench is a different workload and must be labelled. Keep
old speed only when explicitly requested, preserving its date/variant. Twelve slots
or workers do not prove twelve-camera end-to-end throughput.

Price means actual run spend (USD), including the stated validation/test/retry scope.
Record known receipt cost and unknown-cost attempts separately. Provider token
quotes and estimated GPU rent are estimates, not an actual bill. Unknown is not zero.

Sheet positions are resolved from live headers using bench-sheet-contribute. Do not
assume A:W or add columns to fit a generic report. Retain before/after snapshots and
raw old results; publish only under the user's requested scope.
