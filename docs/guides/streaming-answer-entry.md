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
goodvision-answer --config /run/answer.json
```

The first invocation captures all scheduled sources and returns
`capture_finished_awaiting_resource_release`. The outer owner then releases the
exact GPU and confirms its absence with the provider and local cleanup. Resume
the same entry with that receipt:

```sh
goodvision-answer --config /run/answer.json --release /run/resources-after.json
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
| `experiment_label` | Exact outer deployment identity. |
| `public_questions` | Unchanged public JSONL, without truth. |
| `output_directory`, `owner_lock_path` | Immutable artifacts and shared task ownership lock. |
| `resources` | Existing `identity_path`, `instance_id`, `reset_endpoint`. |
| `arms` | Ordered model names and source descriptors (`config`). |
| `scoring` | Local private `truth` and existing PR63 `scorer` path. |

When VLM is enabled, also supply `model_ledger` with `output_directory`,
`seed_manifest`, `owner_lock`; and `final_model` with
`settings`, `model`, `parameters`, `max_input_bytes`. All paths resolve against
the file containing them. Every source in a workflow uses the same cumulative
ledger paths. New seed manifests use `unified-model-ledger-seed-v2` with explicit
file paths; legacy archives remain readable without recomputing their hashes.
Its owner lock must differ from the workflow owner
lock because the spawned observation process owns the ledger. Model settings
require HTTPS and `max_retries=0`.
Keep credentials in the selected environment variable. The legacy
`defer_postprocessing` field is no longer an answering or scoring gate.
Duration discrepancies are advisory: retain the original question, source
metadata, saved observations, and scoring policy without replacing durations.
Batch wrappers must not filter sources using this legacy flag. Resource release,
evidence eligibility, context capacity, and paid-request deduplication still apply.

Do not hot-edit a running frozen batch to remove a historical wrapper filter.
After its controller finishes and the exact GPU is released, use a separate
saved-observation answer run for the affected, never-sent questions. Keep the
original capture and workflow files immutable, seed the cumulative ledger from
the finished run, and do not replay successful or unknown requests. No new
capture or GPU allocation is needed for this answer-only correction.

Each source configuration uses schema `native-stream-source-v1` with:

- `video_id`, fresh `stream_session_id`, `source_id`, and original public source
  metadata in `source`;
- `media_path`, `harness_config`, and an `effective_harness_config` snapshot from
  `effective_config_snapshot(load_config(path))` in the intended execution
  environment during formal preparation;
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
The resolved configuration snapshot records environment-derived settings as well
as the YAML, with credential fields redacted. When supplied, it must match the
runtime snapshot. A changed endpoint, provider, cadence, queue or transport
setting therefore stops capture. New launchers omit legacy checksum arguments;
their compatibility parameters do not perform integrity checks. `rich_labeling.model`
must match the selected detector. `sam3.1` uses `sam31_stream`; `yolo26-s`, `yolo26-l`
and `rfdetr-l` use the generic `paddle` HTTP client contract, with the actual
detector/weights supplied and recorded by the outer deployment owner.
The detector model is also sent in the existing `x-goodvision-detector-model`
request header, so a worker's default model cannot silently replace the selection.

## Coverage and recovery

Raw callbacks are append-only, sequential records. They include every delivered
source frame number/time, tracking completion, independently published report,
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

Completed source directories are reused only under matching declared source
settings, resolved Harness settings, event profile and confirmed worker cleanup.
The saved record sequence and source/session/video identity must also agree.
A recovery workflow may reference a completed source using `reuse.directory`; it keeps the
original method/session/GPU identity. An interrupted source requires an independent
frozen recovery configuration. Original records remain in place.

The ledger caches successful HTTP responses including malformed replies. Unknown
outcomes and missing receipts stay blocked. Each complete request input is
limited to three actual sends. The spawned native process owns the ledger writer
during observation; final answering obtains that writer after worker cleanup and
GPU release. Shared failures stop the workflow instead of failing later questions.

The evaluation inspector and final-answer entry retry explicit HTTP 429 responses
only, on the same input and parameters. Each waits at least 60 seconds and honors
a longer numeric `Retry-After`. Both final-answer HTTP transports preserve this
numeric header in the error and ledger receipt. A delay above 300 seconds ends
this input's retry sequence after that cooldown; an exhausted input also receives
its last cooldown before the worker moves on. Neither case requires batch recovery.
Each send remains separately ledgered, with the cumulative three-send limit
applying across restarts. These are per-worker waits, not an account-wide rate
limiter. Backoff is recorded in
`rate-limit-backoff.jsonl` and in report timing, separately from `model_seconds`;
`observation_wall_seconds` includes it. Final-answer outcomes retain wall time,
backoff time and model time. HTTP timing describes the last attempt; all attempts
are available in the model ledger.
The native process stop event interrupts evaluation backoff and prevents the
retry loop from continuing after shutdown.

### Window and source failure boundaries

A model request returns one window, not one video. The native capture archives
that report before classifying its failure. `failed_requests` is a counter, not
a batch-stop signal. Missing/invalid fields, empty responses, request timeouts
and a single request's connection error remain failed windows while subsequent
windows continue. HTTP 4xx/5xx responses are also window-local except for
credential/payment failures (401/402/403); an explicit 429 remains local even
if a worker normalized its error code. Other failures receive no new retries.
Native admission timeouts and superseded windows also stay window-local.
The native worker preserves the ledger's two known per-input refusals, unresolved
outcomes and exhausted attempts, without another send or a batch stop. The next
distinct input can continue. Other ledger errors remain shared interruptions.

After EOF and drain, a source containing failed windows is saved as incomplete.
Successful windows, failed-window identities, frame ranges and original receipts
remain available. The next video can be captured on the same GPU. Final answering
accepts native evidence with classified window-local gaps after EOF, queue drain,
person-frame accounting and preservation of usable observations. It passes the
saved gaps to the model, which can answer or report insufficient evidence. The
archive retains its failed windows and `complete=false`; rule answers and scoring
remain unchanged. Empty evidence, unfinished source lifecycles, shared failures
and unclassified archive failures still prevent a final-model call. Existing
terminal outputs remain unchanged, and unknown requests are not resent.

Expired or capacity-dropped association reports are skipped work, not a reason
to retry fusion or reject the entire question. New native captures archive the
report identity and its source-time range when it is dropped; the queue's limits,
expiry, scheduling and successful associations are unchanged. Final answering
adds a `skipped_observations` note identifying the affected lane and times, while
keeping the original evidence and its `complete=false`. A skip in person
association does not erase independent VLM observations or detector results.
Historical aggregate records without time ranges are explicitly time-unknown;
their aggregate record count must not be presented as a window count. Older
HTTP-error projections can be classified from their identity-matched saved raw
report when it supplies the missing status; no provider request is made.
Association exceptions and archival/callback failures are not covered by this
scheduling-skip allowance.

The note is added only for a new answer representation with no paid original or
losslessly compressed predecessor. Prior paid requests retain their identity,
successful cache and unknown-outcome block. Existing answer artifacts, terminal
outputs, rule answers and scores are not rewritten. A separate recovery may
answer never-sent questions from already-saved, safely finished captures.
Release the old deployment through its owner before starting a new frozen round;
both capture and answer must load the new source version, not just a pushed
branch. No tests were run for this follow-up change: the user excluded new tests.

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

## Final-model input and output

The answer projection contains the original question/options, timestamped
observations, short subject aliases and explicit gaps. Bookkeeping UUIDs, hashes,
paths and audit metadata are not model evidence. Full archives and receipt
records remain local. Reducing serialization overhead does not authorize
dropping observations or replacing them with an unapproved multi-stage summary.

The final response contract is one `prediction` string containing the requested
option, count, time point, interval or short answer. It requires no copied
observations, citation list or explanation. Input capacity and output truncation
are separate conditions; raising the output ceiling is not the remedy for an
unnecessary output contract. Prior successful, invalid and unknown requests are
not resent merely because the prompt representation changes.

## Validation status

The specifically authorized short-video native-skip path completed capture,
resource release, final answering and local scoring on October 16, 2026.
On October 18, the user authorized targeted 429/window-recovery tests; 38 cases
passed. The local tests cover cooldown and cumulative attempts, persisted-ledger reload, continued
windows in a real spawned worker, continued final answering, both HTTP clients'
`Retry-After` propagation, and shared-failure counterexamples. Model transports
and backoff clocks are controlled; the tests make no provider or GPU requests.
The test report and source review must not be presented as a measured reduction
in production resume counts. Frozen resume39 uses the preceding capture-isolation
commit, not these additions. No full suite, lint, compilation or build was run.
The completed RF comparison retains its original implementation and results.

On October 19, the separately authorized saved-answer run loaded published
commit `bedcb7807692bdf1479cbce2aa9e0cda08fa87d4`. All 70 never-dispatched
candidate questions fit the prepared input budget. It ended with 69 valid-format
answers and one 75-second final-answer timeout, retained as unknown without
replay. The 69 successful responses used 6–16 output tokens and ended with
`finish_reason=stop`; this run had no input-capacity or output-truncation outcome.
Six explicit 429 responses recovered under the existing cooldown policy. The
CPU controller exited normally and its exact container was removed; no GPU was
created. Valid-format answers do not imply correct answers, and these 70 items
are not the complete 182-question evaluation. The separate three-video wall-time
trial has not run. No additional unit tests, lint, compilation or build were run.
