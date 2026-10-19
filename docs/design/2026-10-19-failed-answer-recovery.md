# Explicit recovery of 41 historical final answers

## Human decision and scope

The human requested rerunning the historical request failures and truncated
outputs, then repeated the request when recovery had not started. The earlier
"design approval pending" note was an agent-created delay, not missing authority.
This bounded change extends the existing saved-answer runner; it does not add
an observation path, model, scoring rule, GPU run or wall-time trial.

Select 41 questions / 30 saved videos from the historical 182-question report:
24 HTTP400 responses, six timeouts without a response, 11 truncated outputs.
Preserve 138 valid-format answers and three insufficient-evidence answers.
These are execution/format classifications, not correctness claims.

## Implementation contract

- `saved_streaming_recovery.py`: an explicit `--recover-failed-answers` selection
  freezes old outcome paths and call lists. The six timeouts require the explicit
  `--allow-selected-timeout-recovery` switch. The human's 41-question request
  includes these six cases; old response/payment uncertainty remains unchanged.
  Bind each selection to the old outcome's final call and saved request. When a
  legacy outcome has no receipt, accept only one historical send with the exact
  saved model, parameters and messages. A later send prevents this fallback;
  it cannot be added to the old approval by preparing another directory.
- `summary_recovery.py`: check the exact historical call list, original public
  question and observation operation. Ignore only those old calls for cache and
  unknown blocking, never for cumulative send counts. Later calls in another
  recovery cannot be ignored by reusing this authorization. In-flight requests
  and successful, nontruncated historical responses are not recovery candidates.
- `model_ledger.py`: keep every historical row and all three-send limits. Cache
  and unknown protection apply normally to new calls. Distinguish a received
  HTTP error from no response, and keep cost knowledge separate. A received
  non-rate-limit HTTP error requires another explicit recovery, not auto-retry.
- `summary.py`: preserve the recovery link in local request/outcome metadata,
  never in the model prompt. Use the existing timestamp observation view and
  single `prediction` response. No explanations or generated citation lists.

The 41 questions have 42 historical sends. Forty questions have two remaining
slots; question 0215 counting has one. A 429 consumes a slot and always completes
its required cooldown, even at the ceiling. A new timeout stays unknown without
automatic replay. No change to the 75-second final-answer timeout, 1024-token
output budget, model, full evidence, original questions or local PR63 scorer.

## Execution order and verification boundary

1. Read the latest team branch and preserve unrelated files.
2. Review the source changes and publish to `codex/research-testing`.
3. Prepare one independent recovery from the latest cumulative seed plus new
   ledger rows; freeze the published source.
4. Run the existing network-isolated CPU input phase once. Report measured
   input bytes and local tokenizer counts for all 41 selected questions.
5. After its container exits and is removed, use the existing single controller
   to answer fitting questions, then retain outcomes and subset scores.

No independent tests, test edits, lint, syntax, compilation, build or health
checks are authorized. Source review is not a passing test. Actual authorized
input preparation and final-answer dispatch are reported separately.

Read-only review caught two implementation issues before publication: missing
legacy outcome receipts and nullable response usage. The former now uses the
single-dispatch exact-request fallback above; the latter records absent cost as
unknown without preventing a received response from being saved.

## Baseline and limitations

This is an incremental recovery change relative to the preceding integration
release `d106dd9`; it is not an assertion that this is the task's original
Harness baseline or the native Ashley method. It changes no weights, video
inputs, capture scheduling or scoring. The existing cumulative original-Harness
comparison remains in the weekly report. Old HTTP400 bodies are unavailable;
their original causes and missing costs cannot be reconstructed by this fix.
