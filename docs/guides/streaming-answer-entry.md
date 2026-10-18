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

For newly frozen sources, set
`coverage_policy="native_processed_and_documented_skips_v1"` to use native
frame skipping. EOF, drained queues, saved person results and documented native
skips permit final answering from the actual observations. The source plan keeps
every delivered frame. The artifact reports input, processed and skipped counts,
their frame IDs and the native skip-reason counters. `person_frames_complete`
still describes literal all-frame coverage; `person_frames_accounted` records the
separate native-streaming accounting. Skipped frames are neither zero-person
observations nor negative events. Rule answers also use the observed frames,
with their coverage preserved alongside the answers.

Request failures and unexplained missing records remain failures. Old source
configurations retain the all-frame policy, and historical artifacts and scores
remain unchanged. The final model receives the selected detector identity and
independent VLM observations, together with their actual coverage.
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

The evaluation inspector retries explicit HTTP 429 responses only, on the same
job and parameters. It waits at least 60 seconds and honors a longer numeric
`Retry-After` up to 300 seconds. A longer delay returns the failure for outer
recovery. Each actual send remains separately ledgered, with the cumulative
three-send limit applying across restarts. Other errors and unknown outcomes
retain their existing behavior. Backoff is recorded in
`rate-limit-backoff.jsonl` and in report timing, separately from `model_seconds`;
`observation_wall_seconds` includes it. HTTP timing describes the last attempt;
all attempts are available in the model ledger.
The native process stop event interrupts evaluation backoff and prevents the
retry loop from continuing after shutdown.

### Window and source failure boundaries

A model request returns one window, not one video. The native capture archives
that report before classifying its failure. `failed_requests` is a counter, not
a batch-stop signal. Missing/invalid fields, empty responses, request timeouts
and a single request's connection error remain failed windows while subsequent
windows continue. HTTP 4xx/5xx responses are also window-local except for
credential/payment failures (401/402/403); 429 retains the limits above. This
change adds no retries and does not resend an unknown request.
Native admission timeouts and superseded windows also stay window-local.

After EOF and drain, a source containing failed windows is saved as incomplete.
Successful windows, failed-window identities, frame ranges and original receipts
remain available. The next video can be captured on the same GPU. Existing final
answer eligibility and scoring are unchanged: an incomplete capture is not
silently promoted to successful evidence or a paid final answer.

For YOLO/RF native sources, explicit source-wall or drain timeouts can terminate
only that source. Before continuing, capture must confirm source, tracking, VLM
and fusion shutdown, no shared callback failure, and an accepted completion for
every started detector request. Shutdown-time stale results do not satisfy this
condition. The detector already stores tracking state by `stream_session_id`;
the next source uses a distinct SID. This path does not call an unimplemented
reset endpoint, restart the detector, or release/reallocate the GPU. It retains
the partial archive and a source-terminal record. Questions without saved
evidence remain in the scoring denominator without a final-model call.

Observed shared errors take precedence after shutdown, including errors arriving while
the main thread waits for drain. Unknown worker exits, authentication/payment
failures, recording/ledger errors and generic non-EOF source endings are not
classified as known source-local failures. Unconfirmed
cleanup stops continuation rather than failing untouched questions. Stateful
SAM recovery is not covered by the YOLO/RF timeout path.

When freezing a recovery configuration, allow the drain window to accommodate
the retry schedule as well as pending native work. The per-request HTTP timeout
is unchanged. An already frozen run keeps its original drain limit and code.

The existing PR63 formula, failure denominators and semantic empty-answer policy
apply. Completion means all scheduled work is terminal, not that every answer is
correct. A paired streaming trial must be reported separately from full-cohort
offline scores.

## Validation status

The specifically authorized short-video native-skip path completed capture,
resource release, final answering and local scoring on October 16, 2026.
The explicit-429 recovery change is awaiting observation in the authorized full
evaluation; that short-video result does not validate this later change.
Independent tests, lint, compilation and build checks have not been run.
The window/source isolation change has only been reviewed by reading code; its
runtime behavior has not yet been observed in the authorized evaluation.
The completed RF comparison retains its original implementation and results.
