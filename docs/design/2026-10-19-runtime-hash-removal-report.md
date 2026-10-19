# Runtime hash removal implementation report

Status: DONE_WITH_CONCERNS. Implementation and static source review only; no tests or runtime validation were run.

## Scope and baseline

Worktree: `/Users/pengyuch/goodVision/worktrees/harness-eval-integration-20261015`.
Branch: `codex/eval-integration-20261015`.
Task source baseline: `33a70faa52e9597b7915d907fe0af31c3d4010ad`.
Method: Harness runtime/display and saved-observation evaluation support; this is not historical native SAM-Ashley.

The parent assigned six runtime files and subsequently explicitly extended ownership to `camera_display.py` to keep the shared display-ID interface consistent. Existing parent changes were preserved. This report describes the cumulative assigned delta from the task source baseline, not an independently reconstructed comparison with every earlier historical Harness version. Earlier cumulative changes remain the parent's reporting responsibility.

Changed source files:

- `src/goodvision_harness/scene_model.py`
- `src/goodvision_harness/sam_display_fusion.py`
- `src/goodvision_harness/native_event_audit.py`
- `src/goodvision_harness/vlm_event_policy.py`
- `src/goodvision_harness/runtime/scene.py`
- `src/goodvision_harness/service.py`
- `src/goodvision_harness/camera_display.py`

No commit, push, deployment, historical-result rewrite, model call, GPU operation or production operation was performed by this subtask.

## Implemented field and identity contract

| Area | Previous behavior | New behavior |
| --- | --- | --- |
| Native image audit | SHA-256 of image base64/data URL | `image_omitted: true`, preserving `base64_characters` or `data_url_characters` |
| Reply diagnostics | `content_sha256` | `content_characters`; existing bounded JSON-shape diagnostics remain |
| Provider generation timing | Hashed `generation_ref` | The original `generation_id`, only after the existing bounded `gen-...` validation |
| Policy snapshot | SHA-256 of event decision text | Explicit `decision_rules`, alongside existing version/profile/labels/sensitivities/comparison field |
| Annotation guidance | Instructions to record video/policy hashes | Instructions to retain source reference, timestamps and explicit policy values/rules |
| SAM display person | Hash of epoch/track | `sam-<process-local source-session scope>-<subject sequence>` from a shared session map |
| Stranger whole-frame group slot | Hash of subject tuple | Unambiguous JSON array of the original ordered subject IDs, used only as an in-memory slot key |
| Stranger association evidence | Hash of message/label/slot | Stable `association-<sequence>` from a SceneRuntime-local key-to-ID map |
| Temporal client reuse | Hashed JSON configuration | Private copied configuration values compared directly by the existing pool |
| Camera positive receipt | Header tuple plus image digest | Header tuple mapped to original JPEG bytes; exact byte equality determines duplicate retries |
| Camera capacity refusal | Header tuple plus image digest | Header tuple plus original JPEG bytes compared directly |

The model event-decision text and selected sensitivities are unchanged. Annotation bookkeeping instructions changed; event criteria did not. No long native identifiers, source-session maps, credential-bearing cache keys or image data were added to model prompts by this patch. Raw audit generation IDs remain provider bookkeeping and must stay outside the answer projection.

## Relationships, concurrency and lifecycle

`DisplaySubjectIDs` owns full typed epoch/track values locally and assigns short IDs under a lock. A process-local scope counter has its own lock. `SAMDisplayFusion.subject_ids` and `CameraDisplayMetadata._display_subject_ids` share the same instance when fusion is active. Metadata reset installs the new fusion's map or a fresh standalone source-session map. `display_snapshot` captures the map under the same metadata lock as its anchor, so a concurrent reset cannot relabel an old anchor using the new session's map.

All active `display_subject_id` callers were adapted. Its signature now requires the keyword argument `subject_ids`; downstream callers outside this repository must pass the shared map. IDs are scoped to active runtime instances, not portable identities across process restarts; existing `stream_session_id`/source context remains required. Historical IDs/results are unchanged.

Association keys retain the original message/label/slot relationships, including repeated revisions and accepted-cache eviction. The key-to-short-ID map is intentionally retained until `SceneRuntime.close()`, then cleared. Evicting this map after 90 seconds would allow a retimed/replayed old key to receive a new evidence identity while SceneState could still remember the old one. The existing result ordering, request ordering, revision numbers, locks and SceneState acceptance logic are unchanged.

Temporal pool revision/lease/thread ownership remains in `TemporalClientPool`. Settings, parameters and options are private snapshots; credentials and proxy environment are compared in memory without serialization or diagnostic logging. Existing size selection and acceptance predicates remain.

Camera receipts remain under the existing control lock and keep the existing maximum of 64 positive entries, plus at most one blocked input. The original immutable JPEG object is referenced without re-encoding or an additional bytes copy. Same-size but different JPEG contents do not count as the same retry. Source metadata admission, queue refusal, ACK-loss handling and session reset boundaries remain.

## Unchanged evaluation dimensions

- Source and weights: source identity/audit/cache code changed; no model or weight choice changed and no weight file was modified.
- Input preprocessing: original frames, JPEG encoding, frame times, sampling and geometry were not changed.
- Parameters: event thresholds, tracking behavior, sensitivities and model request options were not changed.
- Loading and calls: only client-cache comparison values changed; request routing, pool size and lifecycle remain.
- Scheduling and state: request/source order and existing retry limits were not changed; new local ID maps retain relationships and release on their described lifecycle boundaries.
- Evidence export: hash metadata was removed/replaced as listed above; images remain omitted from audit output.
- Answers and scoring: no answer protocol, original question/label/option, historical answer or scoring code was modified by this subtask.

## Parent follow-up and unresolved concerns

1. `vlm_independent.py` consumes the policy snapshot and has its own hash-producing custom-policy path. The parent owns that migration. Explicit `decision_rules` must remain audit policy information, not an extra repeated prompt template.
2. `tools/vlm_benchmark/behavior.py:load_packet` still requests `expected['sha256']`, `policy_sha256` and media/frame hashes. With the new snapshot its old policy path will fail until the parent migrates this producer/consumer family. `tools/vlm_benchmark/modes.py` exports `annotation_policy` and therefore receives the new snapshot automatically; associated manifests/reference contracts require coordinated treatment.
3. The parent-owned model ledger already normalizes the historical data-URL hash omission record. Native base64 omission records now use `image_omitted` plus `base64_characters`; consumers that strictly match the former base64-hash shape need the analogous compatibility treatment. Historical files were not rewritten.
4. The 64 positive JPEG receipts can retain at most 128,000,000 JPEG bytes at the existing 2,000,000-byte camera-input limit. A distinct blocked JPEG can add 2,000,000 bytes: approximately 124 MiB total JPEG payload, plus container/header overhead. These are bounds from the existing input limit, not measured runtime allocation. References may share bytes already retained by other components.
5. Source-session identity maps grow with distinct native tracks/association slots until session termination. They do not retain images. Retaining full keys is the explicit memory tradeoff for collision-free stable local IDs without hash computation. No runtime memory measurement was authorized or performed.
6. Existing tests still assert the removed `generation_ref` and `content_sha256` fields, including `tests/test_scene_model.py`. No test code was edited or run; compatibility updates require separate explicit user authorization.
7. Signed webhook delivery and customer updater integrity protocols remain outside this subtask and unresolved. No unsigned fallback or replacement integrity claim was introduced.

## Review evidence and verification limits

Read-only review covered the owned diffs, SceneState's 256-character ID limits, revision/evidence handling, temporal pool equality/revision behavior, camera receipt consumers/reset, shared display helper callers, policy consumers and the affected audit consumers. A textual search of the seven owned runtime files found no remaining hash/digest/fingerprint occurrence after the edits. This is source review, not proof of execution or a passing test result.

Tests, test edits, lint, formatting checks, type checks, syntax/compile checks, builds, probes, model/GPU calls and health checks were not run (not authorized).
