# Recovering a stopped event-timing cohort

The runtime controller can retain completed members of the fixed four-event
50-video cohort after a confirmed shared-resource interruption. This is not a
new model profile, a frame-seek resume, or permission to replay paid work.

## Recovery boundary

Preparation requires the previous controller to be terminal and its exact GPU,
containers, and dedicated network to have confirmed release records. The new
round has its own once-only preparation and launch. Its cohort, media order,
model, sampling, windows, fight rules, output limit, HTTP deadline, and scoring
inputs remain unchanged.

Before reading a video, deployment also requires the detector process's normal
startup message to report `gpu=True`. A running rental or HTTP server alone
does not establish CUDA availability. This uses saved startup output, not a
separate health request, warmup, or model probe.

Completed recordings are referenced through read-only directory links. Their
source configurations, original clocks, reports, and model receipts remain in
the previous round. The controller does not call their models again.

An interrupted recording cannot be silently continued or replayed. Select only
the members the human has asked to replay. No approval text is required in the
source configuration. A replay starts the original media from zero with a new
clock domain. The old partial recording stays intact as a separate failed attempt.
Only the approved new recording contributes that video's primary observation
results; both attempts remain in request and cost totals.

## Runtime state

- `plan.event_recovery` contains the previous round and the member mappings used
  to read completed recordings and select unfinished members.
- `model_ledger.window_replay` contains only `source_id`, `prior_session`, and
  `new_session`. These fields carry prior sends into the cumulative
  three-attempt window budget.

Source configurations and model requests contain no replay approval quote,
replay proof, or repeated replay context. No separate `event-recovery.json` is
generated. The existing credential and private-GT protection remains unchanged.
Historical records are not rewritten.

Matching uses the explicit source, lane, and ordered frame-sequence window.
Old successful replies are not substituted for the newly approved measurement.
Old unknown receipts are not reconciled, assigned zero cost, or silently
retried under their old identities. Unrelated sources retain the existing
success-cache and unknown-request policy.

The cumulative seed includes the previous seed and its live ledger. The new
live ledger also retains the previous event calls so the existing offline
extractor can link inherited reports and report the complete measurement cost.
The ledger loads identical repeated call records once; original call identities
and receipts are preserved.

## Entry

Use `streaming_walltime_trial.py prepare` with the original paired baseline
and the additional `--resume-event-round` directory. Select each explicitly
selected interrupted member with `--replay-interrupted-clip`. Launch only the
newly frozen entry.

The benchmark-side matching and scoring remain in the evaluation repository.
Resource-provider differences, restarts, inherited recordings, and the old
partial attempt must be reported with the final comparison. Controller capture
completion does not turn an old interrupted attempt into complete evidence.

No independent software tests, syntax checks, builds, or health probes were
authorized for this change. Static review is not a test-pass claim.
