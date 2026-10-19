# Prompt and request-ledger migration implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` inline. The user's explicit test restriction overrides test-writing and test-running defaults.

**Goal:** Remove model-visible bookkeeping and replace paid-request content hashes with explicit observation operations without losing previous dispatch state.

**Architecture:** Keep the full archive local and project only observations into the final-answer prompt. Identify an answer by its archived observation session/source/video and question, and a window by session/source/lane/frame sequence. Read legacy records without recalculating checksums; decode their saved evidence only to recover explicit scope.

**Tech Stack:** Existing Python, JSONL ledger, filesystem writer locks, existing lossless evidence codecs.

**Spec:** `docs/design/2026-10-19-prompt-and-hash-removal.md`

## Global constraints

- Human authorization covers prompt cleanup, hash removal, repair, normal branch push and subsequent evaluation. It does not authorize unrelated tests.
- Preserve historical files, successful/invalid/unknown receipts, three-send limits, locking, atomic publication and private error redaction.
- Do not change model, original questions, scoring, frozen runtime or observation semantics.
- New source is not deployed until the repair and dependent producer/consumer migration are finished and pushed.
- The user approved retiring signed webhooks and customer automatic updates. Remove their hash implementations and reject use before side effects. Keep official checkpoint and maintainer deployment protocols outside these two retirements; do not make them a new evaluation-resume gate.

## Review focus

- Old compressed answer requests must recover the observation session, not collapse all answers with one QA ID.
- Unknown receipt must take precedence over cached success for the same operation.
- An unresolvable old operation must not become a fresh, payable request under a new prompt.
- Evidence references can use either published IDs or raw IDs; timestamp projection must retain ambiguity rather than silently select a lane.
- Null, missing, explicit negative and repeated observations must remain distinct.

## Task 1: Finish the model-visible projection

Files: `evaluation/summary_view.py`, `summary_context.py`, `summary.py`.

- [x] Preserve subject nulls; translate published and raw references to times without merging records.
- [x] Apply metadata filtering recursively within permitted method/source fields.
- [x] Retain old request messages and use the single-prediction contract only for never-dispatched answers.
- [x] Read the diff and downstream encoding/parser call sites. Do not execute tests.

## Task 2: Replace the ledger's request hash boundary

Files: new `evaluation/model_operation.py`; `model_ledger.py`, `summary.py`, `rate_limit.py`.

Interfaces: `summary_operation(evidence, qa_id)` returns an explicit scope dictionary; `operation_for_request(context, request)` returns a scope or an unresolved legacy marker. `prior_summary_request(operation=..., model=..., parameters=...)` retains paid payloads. Receipts carry `operation`, not a request checksum.

- [x] Recover legacy answer scope through the existing reversible codecs and legacy window scope from explicit context.
- [x] Keep unmapped legacy requests and prohibit new dispatch only within their identifiable question/video or session scope.
- [x] Count unique call rows per operation; preserve success, unknown, rejection and in-flight distinctions.
- [x] Use run-directory-scoped sequential call names; remove image/error-body hashing and seed checksum checks.
- [x] Update summary and cooldown consumers together; retain a temporary ignored legacy constructor keyword until frozen launcher migration is complete.
- [x] Read all changed callers and stored receipt consumers. No runtime correctness claim without authorized execution.

## Remaining branch migration (not completed by these two tasks)

Capture/recording/cohort/scoring artifact interfaces, native worker frame contracts, configuration caches and generated display/scene identities have received grouped producer/consumer edits and bounded static review. Independent recovery launchers are implemented below but unexecuted. The approved signed-webhook and customer automatic-update retirements are implemented. Test-only legacy contracts and frozen historical copies are unchanged. Do not call this branch-wide removal. See the [integration status](../../design/2026-10-19-hash-migration-integration-status.md). After publication, use the full 182-question recovery inventory for authorized input preparation and continue only never-sent fitting answers, then the authorized small wall-time trial.

## Execution record

- Latest human decision: retire signed webhook delivery and customer automatic
  updates, including their builder, with explicit errors and no downgrade.
  Earlier unresolved-choice entries below are historical. Ruling: retain
  official checkpoint loading and maintainer deployment because the human's
  two-feature approval does not retire them; report those remaining hashes
  instead of claiming repository-wide removal. The cost is that the earlier
  blanket hash-removal request remains only partially fulfilled.

- Reused the existing linked integration worktree; no dependencies installed and no baseline tests run.
- The current plan records the already requested core repair. No extra model call, GPU rental, production deployment or external security-protocol change is authorized by this document.
- Tests and test-code edits are omitted under the user's explicit restriction. Static inspection is not a test result.
- The first static review found three important issues: partially missing legacy answer scope, raw/published temporal-reference ambiguity, and new local ledger refusals being escalated downstream. All three were fixed; scoped re-review reported no remaining Critical/Important findings. Sparse sequential call names also use the highest existing local sequence before allocation.
- Independent runtime hash cleanup covered seven non-evaluation files, including the shared camera-display alias map. Bounded static review reported no new Critical/Important findings. Artifact, worker/client and tool migrations have also completed bounded reviews with the reported findings repaired. This is not a complete branch review or a test result.
- No benchmarks or other tests have been executed. The scoped retirement review found no Critical/Important issue. Documentation was updated, and the startup handler now exposes only the fixed, public-safe signed-webhook retirement error instead of hiding it behind a generic error. No commit, push, deployment, input preparation or new model dispatch has occurred yet.

## Task 3: Independent full-scope saved-answer recovery

Files: `tools/saved_streaming_recovery.py`, `tools/saved_streaming_controller.py`. The old experiment directories remain immutable. Checkmarks below mean local implementation and source review, not executed correctness checks.

- [x] Build a question-by-question inventory from the original public question file, completed streaming references, current round captures and explicit failure/interruption references. Do not select from the previous 72-candidate union.
- [x] Copy the latest cumulative seed plus that answer round's new ledger rows without changing historical records. Any prior final dispatch for an original question excludes it from this never-sent recovery.
- [x] Freeze only a committed, published source tree; save explicit plan and source records, without application hashes. Do not prepare or launch it before the outstanding repair/publication decisions are resolved.
- [x] Implement input-only and answer phases using the new cohort/summary/scoring interfaces. Preserve original model/settings and print actual preparation bytes/tokens. Score available outcomes even if another input is over capacity; do not mislabel unstarted questions as model failures.
- [x] Document exact controller responsibilities and perform read-only producer/consumer review. Do not execute the new entry point or tests during implementation.

Ruling: the explicit user prohibition on unrelated tests overrides the skill's TDD execution and fixture-edit defaults. This task is implementation for the authorized evaluation, not authorization to run a new test. Runtime behavior remains unverified until the separately authorized post-push preparation and answer stages.

Pre-flight: the current cohort uses numbered answer paths and no `answer_file_sha256`; the new runner must consume those rows directly. `ModelLedger` accepts a v2 seed with explicit file paths. `execute_summary(..., prepare_only=True)` is the authoritative per-question eligibility/capacity decision after the coarse source-lifecycle inventory. The source collector must not convert legacy `defer_postprocessing` or `complete=false` into a rejection.

Task 3 implementation record: the [recovery guide](../../guides/saved-streaming-recovery.md) records stages, resource ownership, accounting and subset denominators. Static inspection corrected legacy cleanup-field assumptions, the explicit operation's `question` field, inherited scoring counts and unchecked log retrieval before removal. No new entry point was executed.

Ruling: the local-GPU launcher's old source checksum was removed rather than replaced with a large embedded copy of every source file. Existing required-file and writable-mount checks remain, but file presence is explicitly not source identity proof. This avoids replacing a removed hash with equally unnecessary payload work; the cost is that this preflight no longer detects a stale file with the same name. The independent evaluation runner freezes and mounts committed source instead of relying on that preflight.

Final review: one fresh read-only whole-change review, followed by one source-fix
pass. The directory-valued interruption reference and cloud proxy's stale helper
import were corrected. No tests were run or written under the explicit user
restriction; these fixes are not runtime verified. See the
[review record](../../design/2026-10-19-hash-migration-review.md).

Final: minor (deferred): the association identity map can grow during an
indefinitely running session; a later bounded design must preserve identity after
result-cache eviction. No arbitrary eviction was introduced here.

Final: Ruling: runtime fit/quality/cost and current cloud release were outside
this source review; do not infer them from it. The authorized post-push phases
must supply actual measurements and exact resource reconciliation. External
official-checkpoint and maintainer protocols, and protected historical/test-only
contracts, remain outside the two approved retirements and are not claimed removed.
