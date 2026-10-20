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

Completed recordings are referenced through read-only directory links. Their
source configurations, original clocks, reports, and model receipts remain in
the previous round. The controller does not call their models again.

An interrupted recording cannot be silently continued or replayed. A replay
requires an explicit member selection and the human approval recorded in the
private staging file. It starts the original media from zero with a new clock
domain. The old partial recording stays intact as a separate failed attempt.
Only the approved new recording contributes that video's primary observation
results; both attempts remain in request and cost totals.

## Local fields and why they are retained

- `event_recovery`: preserves the prior round, inherited member paths, replay
  decision, and newly captured members. It prevents a new round from presenting
  an inherited recording as fresh inference.
- `model_ledger.window_replay`: contains `approval_quote`, `source_id`,
  `prior_session`, and `new_session`. It permits the named new measurement while
  carrying prior sends into the cumulative three-attempt window budget.
- `context.window_replay`: records that approval with each new paid request.
  It is local accounting metadata, not model input.

The approval field is deliberately not named `authorization`: that name is
reserved for credentials and remains forbidden by the observation-input privacy
guard. Offline preparation applies the same existing guard to newly generated
source configurations before launch can rent a GPU. Historical rejected source
files are retained unchanged, not migrated or accepted through a privacy bypass.

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
approved interrupted member with `--replay-interrupted-clip`. Keep the approval
in `staging.json` as `replay_approval_quote`. Launch only the newly frozen entry.

The benchmark-side matching and scoring remain in the evaluation repository.
Resource-provider differences, restarts, inherited recordings, and the old
partial attempt must be reported with the final comparison. Controller capture
completion does not turn an old interrupted attempt into complete evidence.

No independent software tests, syntax checks, builds, or health probes were
authorized for this change. Static review is not a test-pass claim.
