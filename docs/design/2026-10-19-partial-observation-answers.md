# Approved answers from partial saved observations

## Decision and scope

The user approved answering the remaining 31 original questions from 23 interrupted
native-streaming sources without recapture. This is a bounded extension of the
saved-answer recovery route, not a new observation method or a completed-capture
claim. Sixteen sources have an archived EOF; seven do not. None has a normal
runtime-final drain record. Historical failures, raw observations, paid replies,
unknown receipts, original questions and scoring remain unchanged.

## Implementation

- Export archived source, tracking, independent-report and scene callbacks through
  the existing evidence adapter into a separate directory. Never call recorder
  finish or invent drain statistics. Missing drain states remain null; EOF is true
  only where an archived source-end record says EOF. The export remains incomplete.
- Require an explicit human-approved video, original question and archive-directory
  list for partial-answer eligibility. Default complete-source behavior is unchanged.
- Tell the final model that this is partial evidence. Include saved input bounds,
  unavailable person-result times and failed published-window intervals. Unpublished
  window times remain unknown. Original metadata duration is advisory, not proof of
  observation coverage. Do not provide HTTP diagnostics, hashes, UUIDs or local paths
  as model evidence. Subject aliases preserve relationships.
- Use the existing short single-prediction output and lossless input compression.
  Preserve the latest cumulative ledger; select only questions never dispatched.
  Scope checks and the existing successful/invalid/unknown receipt handling remain.
- Prepare inputs once, inspect actual capacity, then answer once in the existing
  CPU controller. No capture, observation replay, GPU rental or extra model stage.
  Score new final-answer outcomes through the original local PR63 scorer; preserve
  old failures and scores as history. An answered outcome is not necessarily correct.

## Execution record

Base: `fe968898e8e4111bdd785df8eb544fe4cb53598e` on the integration worktree;
publish to `codex/research-testing`. Existing application behavior before this
bounded extension is the published `bedcb78` implementation.

Ruling: the user prohibits independent tests, test edits, lint, compilation and
build checks. Static code review and actual authorized input preparation/answering
are used here; no independent tests are run or claimed passing. The existing
three-video wall-time authorization is separate and remains pending.

One independent read-only review traced export, eligibility, prompt compression,
scoring and CPU lifecycle. It found no actionable Critical, Important or Minor
issue in this bounded change. This is not a runtime or test result. Actual
23-source export, input capacity and provider outcomes remain execution evidence
to collect after publication.
