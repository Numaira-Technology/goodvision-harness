# Evaluation tools hash-removal implementation report

Status: DONE_WITH_CONCERNS. This report covers only the bounded evaluation-tool assignment, not completion of branch-wide migration or deployment.

## Scope and baseline

- Worktree: `/Users/pengyuch/goodVision/worktrees/harness-eval-integration-20261015`.
- Branch observed: `codex/eval-integration-20261015`.
- Immediate tool-edit baseline: `33a70faa52e9597b7915d907fe0af31c3d4010ad`; not the pre-task Harness baseline.
- Method: existing evaluation tooling for the Harness integration branch; this is not Ashley native SAM.
- All changes below are cumulative relative to that baseline. No source under `src/`, workers, frozen experiments, historical results, paid-request ledgers, test files, or customer packaging/security tools was edited by this assignment.
- No commits, pushes, launches, model requests, dependency installs, GPU operations, or deployment were performed.

## Field contract and implementation

Hash fields removed here were application audit identities or content-comparison gates. Replacement fields are local audit provenance, not model instructions or proof of content equality. No large serialized request or policy is used as an identifier.

| Files | Contract and resulting behavior |
| --- | --- |
| `tools/vlm_benchmark/run_remote.py`, `cloud.py` | New bundle manifests use `schema_version: 2`, `files: [relative_path, ...]`, and the existing ordered `videos` list. Source/media bytes are no longer hashed. The reader accepts both new lists and historical dictionary keys, leaving historical checksum values uninterpreted. Inventory checks still reject invalid/duplicate names, absolute paths, traversal, symlinks, paths escaping the bundle, and missing files. Archive extraction protections remain. This is file inventory checking, not cryptographic or byte-content integrity verification. |
| `tools/vlm_benchmark/local_model.py` | New deployment records store `engine_requirements` as the exact requirements text. Loading compares those values directly and retains the existing profile, installation-state, and checkout identity checks. A legacy installation with no explicit requirements text stops with a migration-specific error; it is not silently declared compatible or reinstalled. Requirements text is not used as a log/request ID or sent to a model. |
| `tools/vlm_benchmark/behavior.py` | Packet policy values are compared directly against the current `annotation_policy` object, using its declared profile. A packet/reference pair requires matching explicit `source_video`, explicit reference `event_policy`, and the expected packet policy version. Generated source/reference/code checksums are replaced with source/reference paths and `source_files`. Frame content hashes are no longer computed or compared. Existing safe-path, JPEG format/dimensions, memory-size bounds, finite source-time, window ID/count/order, last-frame judgment-time, reference coverage, label/type, provenance, and request-budget checks remain. Repeated windows are identified by their same-source ordered timestamp tuple, rather than image content. |
| `tools/vlm_benchmark/modes.py`, `tools/vlm_multistream_benchmark.py` | Source audit manifests now contain `source_files` lists and video path records. No source/media hashing remains. The pre-existing Git revision field remains ordinary provenance supplied by Git; no Git objects are computed by this change. |
| `tools/vlm_benchmark/monitor.py` | Detector preview stage scopes use persisted local positive integer sequences in `detector-stage-scopes.json`. The map rejects invalid or duplicate sequence values. A new map entry is saved before being adopted in memory. Viewer/camera/epoch distinctions and frame sequence ordering remain. Stage names remain local audit keys; detector session IDs carry the short numeric stage scope. Authentication tokens remain intact. |
| `tools/vlm_jev_benchmark.py`, `tools/vlm_simple_jev_benchmark.py` | Audit manifests now record `source_video` and each selected frame's existing `frame_sequence` and `timestamp_seconds`. Frame images, requested window endpoints, model prompts, event labels, models, inference parameters, and scoring are unchanged. |
| `tools/vlm_parallel_benchmark.py`, `tools/vlm_simple_jev_dataset_benchmark.py` | Dataset/media/frame/request checksums and checksum manifest fields are removed. Dataset path, baseline/source file paths, existing sample/repeat/case/window/lane identities remain. Source existence and the original four-frame source-time order/endpoint contract remain explicit; endpoint comparison follows the existing decoder's three-decimal rounding. Existing dataset frozen/reference/sample ID checks are preserved. Imported historical sample records can still contain their old audit fields; those fields are not recalculated or used as content gates. |
| `tools/vlm_speed_matrix.py` | `image_metadata` retains dimensions and encoded byte count, and no longer produces a checksum. Byte count remains descriptive metadata, never a content-equality claim. |

## Synthetic preview behavior

`preview_demo_offset` previously derived phase from a source-name hash. Its new explicit rule uses the zero-based `cameraNN` number modulo 1000, scaled by the existing 14-second period; other source names start at zero. The recorded algorithm is `source-time-camera-index-v2`, and `phase_rule` replaces `phase_hash`.

This changes the scripted preview phase relative to historical hash-derived demonstrations. It affects only the opt-in synthetic preview mechanism, not model recognition, reference labels, scoring, or a running frozen experiment. The algorithm/version change is explicit; historical preview output must not be described as identical under the new rule. The existing period, positive-window duration, label order and real-execution disablement remain.

## Preservation of paid and unknown state

No request ledger, response cache, historical output, frozen input, or unknown receipt was rewritten or deleted. No missing metadata was regenerated through model calls. Fresh-output directory requirements, bounded/no-retry behavior, ordering, request budgets and existing success/unknown handling were not replaced. The removed `request_sha256` in the paired comparison was an audit field, not a cache/retry key; its existing sample/repeat/case/window/lane fields still locate a trial locally.

This implementation does not prove that other agents' ledger migrations are complete. The new model-ledger `operation` contract has no direct producer/consumer in these edited tool files; branch-wide compatibility remains an integration responsibility.

## Remaining integration concerns

1. Historical behavior packets carrying only `source_video_sha256`/`policy_sha256`, or old policy annotation instructions without the explicit values contract, cannot establish the new explicit source/policy provenance. They stop before paid dispatch and remain intact. An evidence-backed migration of their source/reference metadata is required before reuse; this tool neither fabricates provenance nor re-annotates them automatically. No active packet generator was found in the owned Python tools.
2. Legacy local-model deployment records carrying only `engine_requirements_sha256` require a separate review/migration of the actual original installation. Current requirements text must not simply be stamped onto an old deployment as proof of its historical installation. No deployment record was altered here.
3. New remote benchmark bundles intentionally no longer provide checksum content gates. Their safe inventory checks and SSH transport do not establish historical content equality. Customer update verification and webhook signing are outside this assignment and were not weakened.
4. The parent updated `tools/README.md` to describe source sequence/time provenance rather than checksum gates.
5. Read-only search found consumers in the existing tests expecting the old contracts: `tests/test_vlm_mode_benchmark.py` (`source_hashes`); `tests/test_vlm_behavior_compare.py` (hash-only packet/reference and source manifest); `tests/test_vlm_local_model.py` (`digest` and checksum-only install records); `tests/test_vlm_demo_events.py` (old synthetic phase algorithm). They were neither edited nor executed, as required. This is a known integration mismatch, not a claim of test success/failure.
6. The second bounded assignment migrated `tools/low_resolution_model_eval.py`, `tools/sam31_latency_run.py`, and `tools/replay_stress.py`; see its separate report. This report does not claim branch-wide hash removal.
7. No evidence from this assignment establishes that pending questions now fit an input budget or that runtime resume counts, latency, accuracy, or costs changed. Those require separately authorized actual preparation/evaluation.

## Inspection record

Parent review found and fixed two issues: remote reuse now matches explicit source provenance before copying; behavior packets must use the existing `research-v1` policy actually run. Scoped re-review found no remaining Important issue. Source provenance is not byte-content proof; legacy bundles without it must upload the selected video.

Only source reading, text searches, Git state/provenance reads and text-diff inspection were performed. A final bounded text search found no `hashlib`, SHA/MD5 checksum fields, fingerprint gates, or explicit `hash(...)` calls in the owned paths. That search is static inspection, not execution of these tools or a syntax/runtime check.

No tests were run (no user authorization). No tests, fixtures, lint, format checks, type checks, syntax checks, builds, health checks or model/resource probes were added, modified or executed.
