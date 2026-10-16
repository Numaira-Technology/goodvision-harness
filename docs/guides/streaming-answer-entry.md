# Streaming answer entry

`goodvision-answer` is the single entry for source capture, answering and local
scoring. Its frozen configuration has one `vlm_enabled` boolean:

- `false`: native person detection/tracking, saved observations, deterministic
  answers and PR63 scoring; zero observation or final-answer model requests.
- `true`: the same person path plus independent Qwen3-VL-8B observations,
  association, deterministic answers, final-model answers and both PR63 scores.

Face processing stays disabled. The observation workers receive source images
and the fixed event definitions; question text and private truth remain outside
capture. The final model receives the original public question and saved
observations. Ground truth is read only by local scoring.

## Runtime

The entry uses the production `StreamingSession` timed file source,
`RichLabelCoordinator` or causal SAM coordinator, `IndependentVLMBranch`,
`PersonRegistry` and `VLMFusion`. The video source continues while inference
runs. Native queues decide admission and replacement. EOF is followed by actual
tracking, independent-observation and association drain.

This is a headless invocation of production components with a timed local file,
not a claim that the production HTTP service, RTSP or WebRTC transport was
measured. Existing fixed-JPEG offline results retain their original method and
timing identity. They cannot be relabeled as streaming captures.

The outer controller owns GPU allocation, exact deployment identity and release.
It supplies the existing detector endpoint; this entry neither rents resources
nor opens the production HTTP port. SAM uses the explicit `sam_live` continuous
route. Other detectors use the production rich-label HTTP contract. The old
count-worker wire protocol is not a native streaming endpoint.

## Invocation and lifecycle

```sh
goodvision-answer --config /run/answer.json --config-sha256 CONFIG_SHA256
```

The first invocation captures all scheduled sources and returns
`capture_finished_awaiting_resource_release`. The outer owner then releases the
exact GPU and confirms its absence with the provider and local cleanup. Resume
the same entry with that receipt:

```sh
goodvision-answer --config /run/answer.json --config-sha256 CONFIG_SHA256 --release /run/resources-after.json --release-sha256 RELEASE_SHA256
```

The second invocation reads saved observations, answers and scores. It makes no
capture requests. Original questions, responses and historical results remain
unchanged. `python -m goodvision_harness.evaluation.answer` is the equivalent
checkout invocation.

## Frozen workflow contract

Use schema `native-answer-workflow-v1`. Required fields:

| Field | Meaning |
| --- | --- |
| `vlm_enabled`, `face_enabled` | Explicit boolean choice; face must be false. |
| `experiment_label`, `frozen_runtime_sha256` | Exact outer deployment identity. |
| `public_questions`, `public_questions_sha256` | Unchanged public JSONL, without truth. |
| `output_directory`, `owner_lock_path` | Immutable artifacts and shared task ownership lock. |
| `resources` | Existing `identity_path`, `identity_sha256`, `instance_id`, `reset_endpoint`. |
| `arms` | Ordered model names and source descriptors (`config`, `config_sha256`). |
| `scoring` | Local private `truth`, `truth_sha256` and existing PR63 `scorer` path. |

When VLM is enabled, also supply `model_ledger` with `output_directory`,
`seed_manifest`, `seed_manifest_sha256`, `owner_lock`; and `final_model` with
`settings`, `model`, `parameters`, `max_input_bytes`. All paths resolve against
the file containing them. Every source in a workflow uses the same cumulative
ledger paths and seed hash. Its owner lock must differ from the workflow owner
lock because the spawned observation process owns the ledger. Model settings
require HTTPS and `max_retries=0`.
Keep credentials in the selected environment variable. A frozen
`defer_postprocessing=true` stops final answering and scoring.

Each source configuration uses schema `native-stream-source-v1` with:

- `video_id`, fresh `stream_session_id`, `source_id`, and original public source
  metadata in `source`;
- `media_path`, `media_sha256`, `harness_config`, `harness_config_sha256`, and
  `effective_harness_config_sha256` from `effective_config_sha256(load_config(path))`
  in the intended execution environment during formal preparation;
- fixed `event_definitions`, `tracking.model`, `tracking.backend`,
  `tracking.endpoint`, and `provenance` containing the
  frozen runtime and original model/weight identity;
- `vlm_enabled`, `face_enabled=false`, `question_conditioned=false`,
  `sam_internal_patch=false`;
- positive `max_source_wall_seconds` and `drain_timeout_seconds`;
- when enabled, the shared `model_ledger` and `vlm.inspector_settings` plus
  `vlm.route_id`.

The Harness YAML selects the native tracker endpoint and source cadence. Timed
playback must have `realtime=true`, `playback_speed=1`, no frame-count truncation
and no reference memory. The VLM model is `qwen/qwen3-vl-8b-instruct`.
The source runtime imports no face recognizer. Existing service defaults and
historical frozen configurations are preserved.
The resolved configuration hash binds environment-derived settings as well as
the YAML. A changed endpoint, provider, cadence, queue or transport setting stops
capture. The digest is saved without exporting credentials. `rich_labeling.model`
must match the selected detector. `sam3.1` uses `sam31_stream`; `yolo26-s`, `yolo26-l`
and `rfdetr-l` use the generic `paddle` HTTP client contract, with the actual
detector/weights supplied and recorded by the outer deployment owner.
The detector model is also sent in the existing `x-goodvision-detector-model`
request header, so a worker's default model cannot silently replace the selection.

## Coverage and recovery

Raw callbacks are append-only, hash-chained records. They include every delivered
source frame hash/time, tracking completion, independently published report,
scene update, admission decision and final runtime statistics. Native VLM
windows are selected during streaming; they are recorded as runtime-admitted
windows, separately from offline preregistered windows.
Both independent lanes retain late reports. Sealing requires submitted,
finished, published and archived report counts to agree for each lane. The
answer parser checks each scheduled grounding window independently of the
information windows; historical paired-window semantics stay unchanged.

Runtime drain and complete observations are separate facts. A source can reach
EOF and drain all accepted requests while native queues have replaced person
frames. Those frames stay in the source plan, and the observation remains
incomplete. Failed or incomplete captures receive no paid final-model answer.
VLM admission/drop statistics accompany every capture; source cadence and
observation gaps remain relevant to interpreting a negative event answer.

Completed source directories are reused only under the exact source configuration
hash and confirmed worker cleanup. A recovery workflow may reference a completed
source using `reuse.directory` and `reuse.completion_sha256`; it keeps the
original method/session/GPU identity. An interrupted source requires an independent
frozen recovery configuration. Original records remain in place.

The ledger caches successful HTTP responses including malformed replies. Unknown
outcomes and missing receipts stay blocked. Each complete request input is
limited to three actual sends. The spawned native process owns the ledger writer
during observation; final answering obtains that writer after worker cleanup and
GPU release. Shared failures stop the workflow instead of failing later questions.

The existing PR63 formula, failure denominators and semantic empty-answer policy
apply. Completion means all scheduled work is terminal, not that every answer is
correct. A paired streaming trial must be reported separately from full-cohort
offline scores.

## Validation status

Implementation is awaiting the specifically authorized short-video path run.
Independent tests, lint, compilation and build checks have not been run.
The currently frozen RF evaluation continues with its original implementation.
