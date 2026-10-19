# Saved streaming answer recovery

Status: implemented locally and source-reviewed; not executed or deployed. This
entry point is for the original 182 questions / 157 videos, not just the previous
72-question recovery selection. It does not capture video or rent a GPU.

## Scope and state

`tools/saved_streaming_recovery.py prepare` reads the original public question
file, completed streaming references, the final capture round and explicit source
failure/interruption records. Every original question appears in `plan.json`.

- Any earlier final-answer dispatch is preserved, including invalid answers and
  unknown receipts. A new prompt does not make that question payable again.
- Never-sent questions with saved, safely ended streaming sources enter input
  preparation. Local window failures and duration notes are not blanket gates.
- Terminal capture failure, shared interruption, missing capture and unresolved
  source lifecycle remain separate inventory states. This runner does not repair
  those sources or silently score their unstarted answers as zero.
- The latest answer round's cumulative seed **and** its new ledger rows are
  copied verbatim into the new seed. Historical files are not rewritten.

The exact original method is face-disabled native YOLO26-S streaming with VLM.
Original questions, final model/settings, local PR63 scorer and offline truth
are retained. The model receives the reduced observation view and original
question, not GT or private HTTP diagnostics. No additional content hashes are
computed. Git revisions and existing Docker resource handles identify releases
and resources; they are not prompt content.

## Three separate stages

1. **Prepare:** create a new directory, inventory the full scope and freeze the
   committed source. The chosen commit must be HEAD and the fetched team-branch
   ref; relevant uncommitted source is refused. This step calls no model and
   creates no container. Fetch/read the team branch before selecting the commit.
2. **Inputs:** generate local cohorts and call `execute_summary` in prepare-only
   mode. Save each complete prepared request plus its actual bytes, tokenizer
   usage, encoding, eligibility and capacity result. The container has no
   network. No dispatch or new observation occurs.
3. **Answer:** after inspecting preparation results, dispatch only fitting,
   never-sent candidates through the existing ledger. The current summary
   interface prepares the representation again at dispatch; it does not treat
   a saved request file as an unchecked executable request. Old success,
   invalid, unknown and attempt-limit states still take precedence.

Example command forms (placeholders, not a record of execution):

```text
python -B tools/saved_streaming_recovery.py prepare --root NEW_ROUND --source-round CAPTURE_ROUND --last-answer LATEST_ANSWER_ROUND --worktree INTEGRATION_WORKTREE --commit PUBLISHED_COMMIT
python -B NEW_ROUND/checkout/tools/saved_streaming_controller.py inputs --root NEW_ROUND
python -B NEW_ROUND/checkout/tools/saved_streaming_controller.py answer --root NEW_ROUND
```

Do not rerun an entry point after a partial/unknown launch. Inspect the exact
launch, intent, container and atomic outputs first; use a separate recovery
decision rather than removing markers or sending again.

## Controller responsibilities

`tools/saved_streaming_controller.py` owns the existing two host-wide evaluation
locks. Before launch it confirms the prior capture/answer terminal records, the
exact task container labels, the dedicated task network and absence of the exact
previous GPU instance/label. Provider inventory is read-only; no creation or
release operation is called by this CPU-only runner.

The frozen source, assets, seed, public questions, indexes and original capture
directories are mounted read-only. The answer phase alone receives the API key
and task network. Offline scorer and truth files are mounted only for that
phase's local scoring; the request builder never reads truth. Credentials are
not included in command arguments, manifests or whole-container inspection.

Cancellation uses `cancel-requested.json` in the new experiment or a controller
termination signal, followed by an ownership-checked stop of its exact container.
Repeated container-state read failure does not trigger a restart or removal:
the resource is retained for reconciliation. After an observed exit, logs are
retrieved successfully and synced before removing that exact owned container.
Failed log retrieval or ambiguous cleanup leaves `requires_reconciliation`.

## Results and denominators

`inputs-completion.json` lists prepared candidates, not all 182 answers.
`answer-completion.json` lists this recovery's outcomes and retained unsent
questions; `original182_complete` remains false. These reports do not replace
the full inventory or historical results.

Per-video scoring uses only rows with a saved final outcome, with a separate
subset question file and matching counts. A capacity-blocked sibling does not
prevent scoring an existing answer. Conversely, that subset's denominator must
not be presented as the full 182-question result. Full-scope reporting must join
the preserved prior answers, new outcomes and still-pending/source-failure states.

## Inspection findings and limits

Read-only producer/consumer review covered the current cohort paths, seed schema,
summary prepare-only output, source lifecycle fields and local scorer inputs.
It corrected inherited full-group counts in subset scoring and unchecked log
retrieval before container removal. A fresh integration review also found the
directory-valued 0013 interruption reference; it is now retained as pending using
its source manifest and separate reconciliation, not rejected or made answerable.
The local-GPU launcher now checks required
file presence, not content identity; removing its source checksum must not be
reported as proof that a prebuilt image matches the checkout.

No tests, syntax checks, build checks or runtime probes were run. No new input
preparation, API dispatch, GPU rental or wall-time trial was run. The outstanding
security-signing/update-protocol decision has since been resolved by the user:
retire the signed webhook and customer automatic updater with explicit refusals.
Official checkpoint and maintainer protocols remain outside that retirement.
The fresh static
review and source fixes are recorded in
[the review report](../design/2026-10-19-hash-migration-review.md).
