# Hash migration integration status

Status at 2026-10-19 15:05 UTC: the user approved retiring signed webhooks and customer automatic updates. Their implementations now reject use before side effects, and their hash calculations are removed. The earlier migration and two integration fixes remain local pending publication. No runtime verification or claim of repository-wide hash removal.

## Method and baselines

The checkout is `worktrees/harness-eval-integration-20261015`, local branch `codex/eval-integration-20261015`. The authorized publication target is `Numaira-Technology/goodvision-harness:codex/research-testing`. The immediate implementation base is `33a70faa52e9597b7915d907fe0af31c3d4010ad`.

The pre-task Harness baseline remains `6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3`, recorded in the project status and the existing [component guide](../guides/unified-eval-components.md) and [streaming answer guide](../guides/streaming-answer-entry.md). Helper reports describe their incremental edits against 33a70fa; they must not substitute that commit for the pre-task baseline. This is the Harness plus final-answer method, not historical native SAM-Ashley.

Cumulative categories remain distinct: source and worker adapters; file/stream input preparation; explicit inference parameters; loading and calls; native scheduling plus persisted request state; archived observations; final answers and local PR63 scoring. This repair adds the identity/projection changes below. It does not change chosen weights, original questions, inference thresholds, native busy-frame scheduling or scoring formulas. A fresh complete baseline-to-branch line audit was not performed here; the earlier guides remain the cumulative source.

## Implemented contracts

| Boundary | Local implementation |
| --- | --- |
| Model evidence | Timestamped observations and short subject aliases; bookkeeping UUID/hash/path fields removed. Raw, published and confirmed records stay distinct. Temporal references preserve unresolved/ambiguous states. The archive stays unchanged; the model projection is reduced. |
| Final output | One `prediction` string: original option, value, time, interval or requested short text. No copied observations, citations or explanation required. Model and output ceiling unchanged. |
| Paid requests | Explicit session/source/video/question or session/source/lane/frame operation. Prior successful, invalid, unknown and attempted requests retain saved payload/state. Legacy scopes are decoded without recalculating hashes. Three-send limits, locks and atomic receipts remain. |
| Capture reuse | Declared source configuration, resolved Harness settings and profile values are compared. Profile definitions use JSON-native nested lists. Source/session/video identity, record sequence, lifecycle and exact resource ownership checks remain. |
| Cohort reuse | Stores normalized observation-index content and each capture's actual source/session/video, not just the index path. Public question content stays local. |
| Answer/score reuse | Compares answer-relevant question content across legacy/new requests. Existing scores must match the original public question and offline truth rows. GT is not model input. |
| Recovery imports | Preserves source/config/terminal/owner records and inventories; normalizes legacy bookkeeping when comparing semantic settings. Inventory is not byte-content proof. Missing explicit legacy settings require evidence-backed recovery, not automatic recapture. |
| Worker replay | Direct immutable prepared-body comparison; named SQLite columns support the old table. SAM retains at most its last bounded request body per session. Capability tokens are not logged. Contour IDs now contain epoch/object number, with matching producer/schema. |
| Runtime identities | Source-local maps preserve person/event relations; settings are compared privately; bounded JPEG receipts compare original bytes. No credential-bearing settings become persisted cache keys. |
| Overlay | Locked 4096-entry color map; keys above 256 characters are not cached. Palette choices can change after restart/eviction; inference and track identity do not change. |
| Remote media | Reuse requires matching original source path, size and modification time. Legacy bundles without that record must upload the selected video. These fields are provenance, not content-integrity proof. |
| Event policy | Behavior comparison accepts only the existing `research-v1` policy actually used for inference. |
| Equality measurements | Complete lossless mask values with shape/dtype; bounded original JPEG sidecars; temporary replay inputs compared byte-for-byte with bounded buffers. These are observations, not IDs. |
| Customer runtime | Private settings comparison without persisted credentials. Manual installation names use a locked sequence registry; installed labels are imported, conflicts rejected, and live/state-directory names reserved. Package authentication is unchanged. |
| Native orchestration | Current runner records explicit source inventory and frame times. Frozen engines are unchanged and not claimed hash-free. |
| Saved-answer recovery | New independent prepare/input/answer stages inventory all 182 original questions, inherit the latest cumulative ledger and score only saved outcomes with explicit subset counts. No capture or GPU creation. The old controllers remain unchanged. |
| Local-GPU launch | Removed the embedded source checksum. Required-file and writable-mount checks remain; file presence is not a claim that the image matches this checkout. The new evaluation runner freezes and mounts committed source instead. |
| Media fetch manifest | Exports original download handles, paths, recorded size and assembly FPS without copying legacy media checksums. Size is provenance, not content-integrity proof. Original dataset manifests are unchanged. |

## Bounded review record

Findings repaired and independently re-read without execution:

- Ledger: incomplete legacy scope mapping, ambiguous raw/published references, and local refusals incorrectly escalated downstream.
- Artifacts: missing source-setting checks, path-only cohort reuse, legacy setting normalization, and tuple/list profile serialization mismatch.
- Remote tools: positional filename reuse selecting another video; validation accepting a policy different from the one run.
- Overlay: entry count did not bound retained key length.
- Manual project registry: installed labels were not registered before allocation, risking reuse of another installation's project number.

Bounded reviews reported no remaining Critical/Important findings in those reviewed groups, seven runtime files, eight native-worker/client files, both evaluation-tool groups and the native orchestrator. This is not a complete branch review or a test result.

The subsequent fresh whole-change review found two additional integration issues:
a directory-valued 0013 interruption reference rejected by the new planner, and
the cloud proxy's stale import of a removed session helper. Both received source
fixes. One long-lived association-map memory issue is recorded as a deferred
minor. See the [review record](2026-10-19-hash-migration-review.md) for evidence,
limits and unresolved external contracts. This does not turn source review into
a runtime correctness claim.

Reports: [runtime](2026-10-19-runtime-hash-removal-report.md), [first tools](2026-10-19-eval-tools-hash-removal-report.md), [remaining metadata](2026-10-19-remaining-eval-metadata-report.md), [native orchestration](2026-10-19-native-eval-hash-removal-report.md).

## Remaining boundaries

1. The user approved retiring signed webhook delivery. A nonempty signing secret now raises before sender construction. The signing code is removed. Existing unsigned configurations remain a separate mode; configured signed delivery does not fall back to it.
2. The user approved retiring the customer automatic-update protocol and its builder. Both now raise before package/Docker/build operations; the GUI displays the retirement notice without retrying. Official checkpoint loading, maintainer-operated source deployment and third-party packaging protocols were not among the two approved retirements and remain unchanged. They are not a new evaluation-resume gate. No update, download or deployment was performed by this change.
3. Test/QA-only scripts and fixtures retain old contracts under the prohibition on unrelated test edits/execution, including the synthetic face/LLM interop and camera replay/trace tools. These are not claimed removed. Third-party vendored code and package metadata protocols are not silently rewritten.
4. Historical frozen source/manifests remain unchanged. Git/Docker-native object references and constant-time secret comparisons are not application-generated audit hashes and are not deleted.
5. Frozen external preparation/answer controllers still use old fields and gates. New independent recovery scripts are implemented and statically reviewed against actual old records, but not executed. Their input capacity and runtime compatibility remain unmeasured. Do not edit or relaunch completed historical directories. See the [recovery guide](../guides/saved-streaming-recovery.md).
6. Explicit current settings do not establish every legacy deployment's settings. Missing provenance is not a model failure or permission to erase paid state.

## Execution and next actions

No tests were run (no user authorization for unrelated tests). No tests or fixtures were modified. The authorized small wall-time trial has not started. No input preparation, model dispatch, GPU creation, commit, push or deployment occurred during this migration.

After the scoped retirement review, publish normally to the team branch and prepare never-dispatched inputs from the full 182-question inventory. Report actual bytes/tokens; do not substitute the last 72-candidate subset for the full task. Preserve the completed 13 short answers, all prior dispatched requests and unknown receipts.

Resume fitting never-sent answers from saved observations without capture or GPU rental. Genuine source failures/interruption need their own authorized recovery. Then run the authorized 0018/0028/0002 small wall-time trial, keeping GT offline and separating request time, first correct event publication delay and whole-video wall time. The existing event table is not a new timing result.
