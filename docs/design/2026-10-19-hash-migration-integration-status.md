# Hash migration integration status

Status at 2026-10-19 15:48 UTC: the migration and approved signed-webhook/customer-auto-update retirement were published as `bedcb7807692bdf1479cbce2aa9e0cda08fa87d4`. Both retired features reject use before side effects. The subsequent authorized saved-answer evaluation completed 70 question outcomes with no input-capacity or output-truncation failure. This is evidence for that evaluation path, not repository-wide runtime verification or a claim of repository-wide hash removal.

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
5. Frozen external preparation/answer controllers still use old fields and gates. The new independent saved-answer recovery executed successfully as described below. This does not validate an old GPU capture launcher against the new settings contract. Do not edit or relaunch completed historical directories. See the [recovery guide](../guides/saved-streaming-recovery.md).
6. Explicit current settings do not establish every legacy deployment's settings. Missing provenance is not a model failure or permission to erase paid state.

## Execution and next actions

No unrelated tests were run (no user authorization). No tests or fixtures were modified. The authorized small wall-time trial has not started. The approved code was pushed and the saved-answer controller deployed that committed checkout; no GPU was created for input preparation or answering.

The full inventory contained 81 questions with prior final dispatches, 70 never-dispatched candidates, and 31 questions with old source failures/interruption. All 70 candidates fit: 22,368–437,279 input bytes and 7,587–232,606 locally counted Qwen prompt tokens. All 59 formerly oversized questions were present; their aggregate local token count fell from 21,530,579 to 2,179,359. Eleven additional never-dispatched questions omitted from the earlier subset were included. Original observations and questions were preserved.

The run finished at 15:31 UTC with 69 valid-format answers and one final-answer timeout, preserved as unknown without replay. It recorded 76 sends: 69 successes, six explicit 429 responses subsequently recovered, and one timeout. Successful outputs were 6–16 tokens with `finish_reason=stop`. Known cost is a $0.65937543 lower bound; the unknown request cost is not zero. The controller exited with code 0 and removed its container. Historical successes, invalid answers and unknown requests were not resent.

The remaining 31 questions span 23 old interrupted sources: 16 sources have saved EOF records and seven do not; none has a normal final-drain record or sealed capture. Preserve that distinction and obtain a specific partial-observation recovery decision rather than relabeling stopped workers as successfully drained. The independently authorized 0018/0028/0002 wall-time work can proceed without that decision, using a new frozen operational adapter, offline GT, and separate request, first-event-publication and whole-video measurements. The event table and historical timing analysis are not new trial results.
