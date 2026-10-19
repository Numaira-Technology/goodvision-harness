# Remaining evaluation metadata migration report

Status: DONE_WITH_CONCERNS. This report covers the nine exclusive existing evaluation files named in the matching brief. It does not establish branch-wide completion, deployment, runtime compatibility or evaluation results.

## Baseline and scope

Worktree: `/Users/pengyuch/goodVision/worktrees/harness-eval-integration-20261015`.
Immediate edit baseline: `33a70faa52e9597b7915d907fe0af31c3d4010ad`, branch observed during the preceding bounded assignment: `codex/eval-integration-20261015`. This is not the pre-task Harness comparison baseline.

The changes below are cumulative relative to that original tool baseline, not relative to a modified copy. This is evaluation implementation associated with the Harness integration branch; it is not Ashley native SAM. Source/models/weights, inference inputs, semantic prompts, labels, thresholds, schedules, costs and scoring were not changed by this assignment. New audit fields do not enter model prompts.

Only the nine assigned implementation files and this report were edited. No tests, fixtures, synthetic/dummy assertions, frozen experiments, paid-request records, historical results, or customer/security protocols were edited. No commit/push, process launch, dependency installation, model request, GPU operation or deployment was performed.

## Changed field contracts

| File | Removed contract and replacement |
| --- | --- |
| `tools/low_resolution_model_eval.py` | Removes file/config/code/source/image checksum generation. New run metadata has `schema_version: 2`, explicit source/config/manifest file paths, the ordered sample inventory, existing clip/timestamp/size/actor identities, event sensitivities, model/provider, and explicit inference settings. Semantic prompt fields are retained as local audit settings, not identifiers or changed prompt text. The entire configuration is not serialized: camera URLs, transport credentials, extra headers, environment values, endpoint credentials/query/path and unrelated manifest fields are not expanded. Transport provenance records only scheme/host/port. Existing request parameters were already in the report and remain unchanged. |
| `tools/low_resolution_face_eval.py` | Replaces the source gallery checksum with its path. The existing direct comparison of the original gallery bytes after evaluation remains intact. No gallery contents, embeddings or enrollment names are added to the report. |
| `tools/local_gpu_capacity.py` | Counts distinct templates by comparing the already-retained request objects directly. The count and workload caveat remain; requests are not serialized into IDs or newly written to the report. |
| `tools/sam31_latency_run.py` | Replaces source JPEG checksums with `source_jpeg_file` paths naming retained original JPEG sidecars in `source-frames/`. `source_evidence_unchanged` and annotation source checks compare actual JPEG bytes against those retained values. Source-map and annotation-map rows retain frame sequence and source timestamps. Safe paths and a 64-MiB per-read cap protect sidecar loading; the existing source acquisition cap is 64 MiB in total and 512 frames. Byte counts remain descriptive bounds, not equality evidence. |
| `tools/sam31_postprocess_benchmark.py` | Replaces `mask_sha256` with lossless `mask_values` containing shape, dtype, encoding, and packed mask values. `compare` still compares frame sequence, IDs and every mask value. Source audit keeps filename, dimensions and encoded size without a checksum. Excluded output work is named `observation_encoding_seconds_excluded`. |
| `tools/sam31_stream_probe.py` | Uses the same lossless mask observation contract. `compare` still checks IDs and complete mask values. Existing finite frame/image/object constraints and the original probe logic remain. |
| `tools/replay_stress.py` | Replaces source/track checksum identity and mutation gates with retained exact local input values and direct streaming byte equality. File values are retained in temporary files with buffers capped at one MiB and an initial source-length bound; snapshots are closed after replay cleanup, including errors. Reports use source/track/manifest/config/tool paths plus `source_values_unchanged`, emitted only after the direct checks succeed. Common real-evaluation audit code was changed; synthetic track generation, oracle behavior and assertions were not changed. |
| `workers/contour/benchmark_gpu.py` | Replaces output mask checksums with shape/dtype and complete lossless mask values. Boolean masks use packed bits; other native mask dtypes retain exact raw bytes. Encoded values are captured before advancing shared native output buffers. Worker/source provenance uses paths and the existing upstream revision. Timings explicitly exclude observation encoding, with renamed encoding/wall-time fields. |
| `workers/contour/prepare_jetson.py` | Removes before/after checksums from patch audit records; records the path and a direct `content_changed` comparison. The approved upstream Git revision gate, exact source replacement anchors, patch transforms and original checkpoint loading/validation behavior remain intact. No package authentication or update protocol was weakened. |

## Actual equality measurements

`mask_values` is an observation, never a log identifier. For boolean masks the representation is `packbits-little-zlib-base64`, retaining the full shape and dtype so tail-bit padding is unambiguous. Other dtypes in the GPU capacity tool use `raw-zlib-base64` with full shape/dtype. Complete representations are compared, preserving mask contents rather than substituting area, dimensions, object count or byte length. Existing scores, areas and IDs remain. Output encodings copy native buffer contents before later buffer reuse can overwrite them.

SAM latency records retain the actual bounded sampled JPEG values in sidecars. Both final source-preservation reporting and rendered annotation checks use byte-for-byte comparison. Saved preview loading does not regenerate or invent historical original JPEGs from checksum metadata.

Replay retains original video and track file values locally before execution and compares one-MiB blocks after worker drain. A matching size alone never yields a successful content comparison. This adds temporary disk storage proportional to the input files while keeping comparison memory bounded. Snapshot preparation remains outside the replay timer; comparison remains after worker stop, where the prior checksum gate was located.

These are implementation contracts established by static reading, not observed execution outcomes. Encoding and sidecar storage can use more disk/CPU than fixed-size hashes; no new timing or resource claim is made.

## Historical state and resume compatibility

The low-resolution model tool compares new explicit run metadata directly during resume. Checksum-only legacy metadata, missing resume metadata, differing explicit settings, and an existing run without its journal stop before dispatch and before overwriting metadata. They require an evidence-backed migration of the original run. A fresh invocation cannot overwrite an existing report, journal or run-metadata file. Existing journal keys, including records representing errors or unknown results, remain completed exclusions; metadata cleanup does not turn them into new payable jobs.

The code does not change successful/unknown paid state or reconstruct missing model receipts. It does not automatically migrate a historical run by copying current settings into its old metadata, create a new output path, or initiate new paid work.

Old SAM latency outputs without retained `source_jpeg_file` values cannot satisfy new exact-source rendering. They remain unchanged and produce a migration-specific error instead of claiming byte equality based on their old checksums. Historical mask reports containing only checksums remain archived; exact-value comparisons now require actual mask observations and do not re-run a model to create them.

## Remaining integration concerns and retained boundaries

1. Existing tests contain old schema assumptions and were left intact: `tests/test_sam31_stream_probe.py` constructs `mask_sha256` rows; `tests/test_contour_gpu_benchmark.py` indexes `mask_sha256`; `tests/test_sam31_latency_run.py` mutates `source_sha256`. They were not edited or executed. Other tests may also depend on changed metadata/timing fields; no test compatibility result is claimed.
2. Consumers of renamed output timing fields must recognize `observation_encoding_seconds_excluded` and `probe_wall_seconds_including_io_encoding_and_eof`. No non-test Python consumer of these old fields was found in the bounded text search; external notebooks and frozen controllers were not exhaustively audited.
3. Source and code paths are provenance, not proof that current files equal historical content. Explicit settings replace the whole-config hash only for the declared non-secret evaluation fields. Secret transport changes and unrelated config fields are intentionally not persisted or used as content identities; they need independent provenance if historical equivalence matters.
4. Lossless mask reports and exact-source JPEG sidecars increase local evidence storage. The finite frame/mask/image constraints remain, but this assignment did not measure the new report size, CPU cost or available storage.
5. Replay contains both real and synthetic execution paths. Only common input preservation/audit metadata was changed. No test-only or synthetic assertion code needed modification; no hash exception was retained in the nine owned files. Security signing/customer-update handling remains outside scope.
6. Missing legacy explicit metadata is a migration requirement, not a claim that the underlying historical model outcome is invalid. Existing observations, answers, fees and unknown receipts remain authoritative historical data.

## Static inspection and execution limits

Source reads, bounded text searches and Git text diffs were the only inspection operations. A bounded final search found no `hashlib`, SHA/MD5/checksum/fingerprint computation or explicit `hash(...)` call in the nine assigned files. Static inspection does not establish syntax/runtime correctness or timing equivalence.

No tests were run (no user authorization). No tests, fixtures, lint, formatting checks, type checks, syntax checks, builds, health probes, model calls, hardware probes or resource operations were added, modified or executed.
