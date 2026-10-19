# Native evaluation orchestrator hash-removal report

Status: DONE_WITH_CONCERNS. The active orchestrator implementation is migrated. Frozen historical engines still contain their original hash-derived protocols; this report does not claim they are hash-free or compatible with a newly changed worker protocol.

## Baselines and method boundaries

- Checkout/worktree: `/Users/pengyuch/goodVision/worktrees/harness-eval-integration-20261015`.
- HEAD read for this bounded edit: `33a70faa52e9597b7915d907fe0af31c3d4010ad`.
- The original, pre-task Harness comparison baseline required by the project rules was not independently established in this bounded task. The HEAD above is the immediate implementation baseline only; it must not be substituted for that original Harness baseline. Cumulative differences between the original Harness and this whole branch remain outside this report and unverified here.
- `current-main` remains the existing profile name selecting `ROOT/src`. The name does not establish that the checkout is GitHub main, Ashley native, or an unchanged original Harness.
- `frozen51` still selects `experiments/ashley-detector182-20261012/frozen_runtime/src` and its event definitions. Its archive declares source commit `d167188a4dcca7b2e7fc8fe9c4de121ef7e87b25`. The publication manifest separately declares method `ashley_native_siyu_main_rules_detector_ablation`, native base `fca08e118c013d053ff880153dc6008584805e64`, previous native commit `cfd6f748023b50932017f5daa924fcb24a9a3372`, and detector reference `f9c82f57ad74513f79f44a329717ad9af9aaa497`. These are distinct provenance claims, not replacements for the original Harness baseline.

Only `benchmark/bench/native.py` and this report were edited. Frozen runtime source, archive/publication manifests, weights, questions, observations, historical results, caches and paid-request records were not rewritten.

## Changed contracts

1. The application `digest` helper and all checksum computation/comparison in the current native orchestrator were removed.
2. `frozen_source_inventory()` reads the existing archive/publication revisions and declared file paths. It checks that paths are relative, remain inside the appropriate root, are regular existing files, are not direct symlinks, and are not duplicated within each inventory. It records explicit manifest paths, publication method/branch/revisions, and inventories, with `content_equality_verified: false`. It does not compare or recalculate the historical checksum fields. The legacy public `verify_frozen_source()` name is retained for callers, but its docstring explicitly limits its meaning to inventory checks and returning the declared source commit.
3. The media plan retains the same question order, selected labels and file resolution rules, but no longer computes media content identities or gates on a historical `sha256` field. Each source has the existing `video_id`, resolved `media_file`, and `manifest_relative_path`. Relative-path/root containment/existence checks remain. Paths are provenance and do not establish content equality with a historical file.
4. New orchestrator provenance uses `schema: 2`, source paths, declared/current Git revision records, a runtime source-file inventory, and the frozen source record. `orchestrator_patch_sha256` is replaced by `orchestrator_has_uncommitted_changes` and `uncommitted_source_files`, obtained from Git's filename-only diff. Full patches or configuration files are not serialized as replacement identities, avoiding exposure of secrets previously hidden by checksums. Existing explicitly logged effective inference settings remain unchanged.
5. Window audit `source_frames` rows now contain `[frame_sequence, timestamp_seconds]`, in original order, without a third checksum element. `source_frame` audit events retain `frame` and `t`, without `jpeg_sha256`. The retained native frame objects and all image bytes supplied to tracking/inspection are unchanged.
6. Per-question capture provenance uses `media_source` instead of `media_sha256`. The new record is local provenance, never a model-visible identifier or a serialized request identity.

The current direct entrypoint is `benchmark/run_native.py` importing `bench.native.main`. `main` still calls `evaluate_frames`; the dynamic helpers are loaded from current `src/goodvision_harness/native_event_audit.py` and `offline_benchmark.py`, even when the engine profile is frozen. The current audit writer accepts the sequence/time audit fields without a hash requirement. This file has no direct `NativeCaptureRecorder.tracking_frame` call; the current recording API's optional legacy `jpeg_sha256=None` parameter remains compatible and is not populated by this orchestrator.

## Preserved evaluation behavior

No weights, event definitions, question strings/options, scoring/extraction rules, observation sampling parameters, source order, tracking startup/append sizes, EOF/drain ordering, model/provider/request parameters, model-call budget, error handling or batch-abort policy were changed. Neither profile sends question text to observation windows or adds a final answer-model call. The explicit `historical_publication_reproduction: false` boundary remains.

Fresh output-directory and exclusive audit/result creation remain. No cache/response/unknown record was deleted or migrated, no new retry path was introduced, and no known or unknown paid request was reclassified as unsent. This bounded edit does not add a new resume mechanism or claim that historical no-resend behavior outside this orchestrator has been audited.

## Frozen dependencies that remain unchanged

The following are exact historical runtime dependencies, not leftovers in the edited orchestrator. They prevent a claim of complete hash-free execution under `frozen51`:

- `experiments/ashley-detector182-20261012/frozen_runtime/src/goodvision_harness/rich_labeling.py:346` requires exact detector provenance keys including `weights_sha256`; lines 352–354 require a 64-character hexadecimal value. A worker response that only implements a new non-hash provenance field would not satisfy that frozen parser. No fake checksum or historical parser change was introduced.
- Frozen `detector_native_session.py:128` derives a prepared frame entry from JPEG content; lines 135 and 160 derive/check the prepared payload hash. That prepared-input contract remains historical. The orchestrator itself does not manufacture its values.
- Frozen `video_tracking_session.py:611` and `:693` derive/check the prepared SAM append body hash. Its original bytes/session/order contract is unchanged.
- Frozen `scene_runtime.py:1398` derives a whole-frame-negative group slot, and `:1437` derives an event evidence ID. Replacing these would change the frozen engine's internal historical identity contract and was outside scope.
- Frozen `scene_model.py:631` hashes provider generation IDs for historical timing metadata. It remains part of frozen event-model execution.
- Frozen `native_event_audit.py` also contains historical hash-based request-ledger logic. The edited orchestrator explicitly loads the current audit helper for its own audited requests; this report does not claim that every other frozen import/entrypoint avoids that historical module.

Git commit/object references and archived manifest checksum strings remain historical provenance. None was recalculated here. Frozen webhook signing and other security code were not changed. Historical compatibility should be handled through an explicit runtime/worker selection or a separately scoped new method; changing frozen files or synthesizing hash fields would obscure the method boundary.

## Consumers and migration limits

- The retained `verify_frozen_source()` interface is imported by `benchmark/tests/test_native.py`; its legacy test name refers to archive hash matching. Tests and fixtures were not edited or executed. Existing tests also refer to image-hash audit fields from the old helper contract; this is not evidence of execution success or failure.
- Static searches found no active non-test consumer of the removed `native_runner_sha256`, `orchestrator_patch_sha256` or `runtime_source_sha256` fields in the inspected benchmark/current source paths. External analysis notebooks/controllers and frozen output readers were not exhaustively audited.
- New file inventories and current/declared source revisions do not establish that a file matches archived publication bytes. A modified file at the same path must not be reported as the original solely because inventory checks accept it.
- No active or historical evaluation was launched, resumed or redeployed. Existing frozen request protocols may require compatibility work before a future authorized execution; no runtime outcome is inferred from source inspection.

## Inspection limits

Only source reads, Git state/revision/diff reads and bounded text searches were performed. Final static inspection found no `hashlib`, checksum/digest computation, SHA field generation or media/source hash gate in `benchmark/bench/native.py`. The only source changed in this task is that orchestrator; the archive/manifests remain untouched.

No tests were run (no user authorization). No tests, helpers, fixtures, synthetic assertions, lint, format checks, type checks, syntax checks, builds, runtime checks, model calls, processes, hardware probes, dependency installs, resource operations, commits or push were performed.
