# Evaluation-only reaction timing

See the [physical-conflict Harness adaptation guide](physical-conflict-harness-adaptation.md)
for the runtime changes, frame-attribution limits, and single-fight/four-event
measurement contract. The separate [physical-conflict reaction task](https://github.com/Numaira-Technology/goodvision/tree/codex/physical-conflict-on-off-eval/evaluation/lmms_eval/tasks/physical_conflict_reaction)
owns the derived benchmark protocol, offline matching, and saved-run comparison.

We completed two distinct 80-second evaluations of UBI F_8_1_0_0_0 and the
negative clips N_251_0_0_1_0 and N_478_0_0_1_0. The original `091f41d` run used
52 labels and parallel events/compact calls. The later `268f51b` run deployed
the fight-only profile and completed the same clips. Saved completion and
cleanup records confirm that both runs finished and released their resources.
Each selection contains one positive source case, so neither supports a
population p90. Keep their source revisions and results separate.

We changed the benchmark application from final QA answers to a derived event
reaction task: native window-level `true`/`false`/`null` reports, source-interval
matching, actual-input-to-readable-report timing, and separate event, video,
window, and request outcomes. The native window output already existed in
Harness. Original questions, annotations, and QA scoring remain unchanged.
The reaction task has its own protocol and denominators; this Harness guide
does not replace its benchmark implementation.

## Release and runtime correspondence

Git source, a prepared runtime, and an observed result are separate records.
Documentation-only commits after the application revisions below do not create
another runtime or latency result.

| Application source | Profile and role | Execution evidence reviewed for this guide |
| --- | --- | --- |
| `091f41d` | Historical 52-label benchmark profile, parallel events/compact calls, final QA disabled | Completed original 80-second run; saved input/publication records and confirmed release |
| `268f51b` | Fight-only profile introduced by `a07b27f` and refined by `8c3e5fb`; `capture_mode: physical_conflict`, no compact or final QA | Deployed and completed a separate 80-second run; saved event reports, input/publication clocks, request receipts, and confirmed release |
| `f457ec2` | Fixed 50-video extension of the same fight-only observation implementation | Deployed; the saved records reviewed for this update show capture in progress and completed sources, without a final cohort outcome |
| `e88e536` | Four-event comparison implementation; `capture_mode: native_four_events` | Published source; saved local state contains a 50-member staging selection only, with no preparation or launch. Launch follows complete single-fight termination and resource release |

The historical `091f41d` events lane used 1600 output tokens and a 75-second
deadline; its compact lane used 8192 tokens and 180 seconds. The fight-only and
four-event profiles use 160 tokens and five seconds for events. Both configure
Qwen3-VL-8B-Instruct through the same OpenRouter route. Keep each run's actual
provider, request durations, and runtime evidence separate.

The selected person detector is YOLO26-S with explicit size 1280 and confidence
0.10. Provenance declares `max_det: 300`, while the worker leaves that argument
to the installed SDK default. The saved evidence does not establish that
default. Native capture uses the internal `/v1/tracks` protocol.

The native tracking client in [native_stream.py](../src/goodvision_harness/evaluation/native_stream.py)
selects `RichLabelHTTPClient(mode="tracks")`. The client rewrites the configured
labels endpoint to `/v1/tracks`; the tracked
[detector worker](../workers/detector_rich_label/app.py) implements both routes.
The earlier review of the `091f41d` run established textual correspondence
between its frozen detector `app.py` and `detectors.py` and the tracked worker
files. This two-file correspondence
does not establish that the full deployment bundle is interchangeable. The
separate historical `/v1/count` service is not a dependency of native event
tracking; its integration gap must not be used to block this mode.
The saved `091f41d` worker log contains completed `/v1/tracks` requests. Those
records describe that historical deployment. The later `268f51b` execution has
its own completion, request, and release records. The worker's `model.predict`
call does not pass `max_det`; the SDK-default limitation applies to both.

### Historical deployment dependencies (`091f41d`)

The original preparation copied existing workspace artifacts. The published
[timing controller](../tools/streaming_walltime_trial.py) retains external
deployment dependencies, so an application checkout alone is insufficient for
standalone reconstruction. The inventory below describes the original run's
dependency evidence.

| Runtime component | Controller source or completed-run evidence | Boundary |
| --- | --- | --- |
| Application freeze | `freeze_source` in [saved_streaming_recovery.py](../tools/saved_streaming_recovery.py) archives committed `src` and two saved-streaming helpers; the controller copies its committed entry separately | The helper does not archive the detector deployment tree, images, or every run input |
| GPU service and configuration | `prepare` copies `source_round/gpu/source`, `harness.yaml`, and `cohort/seed-weights.json` from a prior saved capture | These are separate frozen workspace artifacts, not implied by the application Git revision |
| Resource control and release | `prepare` copies existing `control.py` and `driver.py` entries into `resource-control.py` and `resource-release.py` | The adapter retains local Docker image references, a dedicated network, and host tooling; no portable image build is established by this document |
| Detector environment and weights | Frozen `gpu/source/workers/detector/requirements.txt`, deployment bootstrap, `cohort/seed-weights.json`, and `gpu/output/weights/weights.json` | Requirements include Ultralytics 8.4.0 and Torch 2.7.1; `yolo26s.pt` declares the Ultralytics assets v8.4.0 source. The installed environment and staged weight receipt are runtime evidence, not a new build result |
| Source configuration and media | Three `sources` files, original media paths, staging selections, and model-ledger seed in the private experiment | Media, credentials, ledgers, and experiment data are not repository release contents; GT remains offline |

The historical `091f41d` run's `plan.json` names these origins. `ready.json` embeds that
plan, and launch compares the plan snapshot; it is not a full dependency-content
freeze. There is no separate whole-runtime source manifest in that run.
Application publication, successful historical execution, and standalone
reconstruction must not be reported as the same achievement.

The local launch wrapper uses `sys.executable`, and the two local containers
use their image's `python`. The saved launch record does not identify the
wrapper interpreter's actual path/version. The remote bootstrap creates a
Python 3.12 environment and saves installed Python packages, but uses dynamic
OS package installation and does not preserve a complete OS/Python-build
lock. These are additional reconstruction limits, not evidence that the
completed run failed.

Standalone reproduction would require versioning those deployment dependencies
and recording their preparation and execution. The completed fight-only run
establishes later execution evidence, but does not establish a portable build.
No software tests were run for this documentation update (no user authorization).

## Physical-conflict event tracking

The timing controller's `capture_mode: physical_conflict` selects the existing native `events`
lane with only `fight` and its standard `main-baseline` sensitivity. The 51
additional benchmark definitions and the compact grounding lane are absent.
The response contract is exactly `{"fight":true}`, `{"fight":false}`, or
`{"fight":null}`. A valid fight result does not require unrelated label keys.
The native information limit of 160 output tokens and five-second HTTP
deadline applies. Five seconds is a request deadline, not a guaranteed correct
reaction time; expired requests remain failures. The model, original image
preprocessing, source sampling, native temporal window scheduler and in-flight
limit retain their current settings. Person tracking remains an internal
Harness component; no separate counting, location, duration or final QA is
dispatched.

The timing controller calls `capture_native_stream` directly for this mode.
It creates no public-question file, answer workflow, observation index,
`capture.json` QA evidence, or final answer/score. It records only raw input
timestamps and dispositions, event reports, readable publication boundaries,
source end/runtime summaries, and failures/rejected reports. Per-frame person
results, admissions, and scene snapshots are not exported. Native tracking
and scene updates still execute internally. Completion records distinguish
actual EOF/drain from event-report quality, including all-failed or uncertain
windows; local output errors are not zero-frame-completeness gates.

File-iterator `OSError`, `ValueError`, and OpenCV decode exceptions retain the
source-local reason `native_file_decode_interrupted`. The controller may
continue only after the existing cleanup record confirms stopped workers,
settled tracking requests, and no shared failure. Recording callbacks retain
their fatal cause, and unknown session errors remain shared failures. Initial
media probing and a decoder returning `read_ok=false` keep their existing
behavior; this change adds neither a new completeness gate nor a claim that
all truncated files can be detected.

The complete benchmark profile and its existing evidence recorder retain
their behavior outside this explicit capture mode. Historical checkouts and
artifacts retain their original content. The `268f51b` run supplies execution
evidence for fight-only capture; publishing source or documentation remains
separate from starting an evaluation.

The saved 15.58968-second F_8 case contains 1.70146 seconds from source onset
to the last window image, 13.30816 seconds in the HTTP exchange, and a residual
0.58007 seconds for all other pre/post-request work. The HTTP exchange returned
419 tokens for 52 fields. Its 1.88005-second header boundary and 11.42800-second
body wait do not isolate server inference from queuing and transfer. A later
window returned only `fight=true` in nine tokens with a 1.74333-second HTTP
exchange; the old 52-field contract rejected it. Its historical 7.45987-second
arrival remains a diagnostic, not a replacement for the original result. The
two requests used different windows, so their difference is not a controlled
measurement of the new mode. Compact waits of 165.93/180 seconds prolonged
drain, not the measured independent first-report endpoint. Final QA was already
disabled in that historical run.

## Method and publication scope

The timing field contract below originated with `091f41d` and also applies to
the later profiles. Selecting physical-conflict mode replaces the historical
benchmark labels, paired grounding, and QA exports with fight-only recording.

This is the Harness evaluation method, not historical native SAM-Ashley. The
pre-task Harness baseline remains `6c28ddd`; cumulative changes are recorded in
[the component guide](guides/unified-eval-components.md) and
[the integration record](design/2026-10-19-hash-migration-integration-status.md).
The `091f41d` timing increment used `7b7c5d3` as its immediate implementation
base; the pre-task comparison baseline remains `6c28ddd`.

The `091f41d` timing increment changes source replay pacing under the evaluation
switch and adds local input/publication export. Detector
and VLM source/weights, image preprocessing, inference parameters, loading and
calls, busy admission/replacement and report ordering, final-answer contracts,
and PR63 scoring retain their existing versions. Earlier adapters, request
accounting, evidence projection and answer/score changes remain part of the
cumulative method; they are not described as an unchanged original Harness.
The source review for that increment did not repeat a whole-branch baseline audit.

## Switch and boundaries

Set `evaluation_timing: true` in a frozen native-stream source configuration.
The default is false. This is an evaluation replay instrument, not a model,
prompt, scoring, or production delivery change. No annotations enter capture.
The controller retains the original three-clip selection as its default and
also supports the fixed 50-video cohort described below. Preparation requires
`--prior-timing-round` to preserve the previous run's release and GPU accounting;
launch checks that exact historical GPU and containers again. It keeps the
latest saved-answer ledger and never runs a final-answer phase. The earlier
three-video run and its independent cost ledger remain historical artifacts.

With the switch on, the file source paces every decoded raw frame at source
speed before display/model sampling. The input clock is read after that wait,
before sampling, encoding, or downstream callbacks. Early decoding is not
counted as arrival. Existing cadence and late-frame-drop rules still apply.
Raw indices are zero-based decoded source indices; delivered frame sequences
are a separate coordinate. The media schedule uses index / declared source
fps, not decoder PTS. Use media whose time base has been independently checked;
variable-rate or unmatched media need a separate alignment assessment.

The output boundary is the independent information result bus becoming
readable in the capture parent process. A monotonic reading just before its
publication lock is released and another immediately after release bracket
that boundary. Their difference is retained, not rounded away. Grounding
publications are recorded separately and are not information event responses.
This endpoint excludes later fusion, archive persistence, webhooks, network
delivery, and browser rendering; it does not claim end-user display latency.

## Local field contract

These fields exist only in recording artifacts, never in prompts or answers.

| Record | Fields | Purpose |
| --- | --- | --- |
| `frame-input` | `raw_frame_index`, `timestamp_seconds` | Link source annotation coordinates to the actual decoded frame. |
| `frame-input` | `ingress_monotonic`, `scheduled_monotonic`, `lag_seconds` | Separate actual arrival from the ideal playback schedule. |
| `frame-input` | `disposition`, `frame_sequence` | Distinguish cadence exclusion, late drop, and selection for downstream delivery; selection does not prove later processing. |
| `publication` | `report_reference`, `lane`, `cursor`, `state` | Link one persisted original report to its independent bus publication, including local-window failures. |
| `publication` | `readable_monotonic_lower`, `readable_monotonic_upper`, `boundary_width_seconds` | Preserve the measured publication interval. |
| both | enclosing `clock_domain` | Require both endpoints from the same capture process/session. |
| final source statistics | decoded/selected/excluded/dropped counts and input observer wall time | Expose sampling coverage and synchronous instrumentation cost. |

The existing atomic recorder persists each input row. Its synchronous cost is
included in subsequent processing and reported separately; this is an
instrumented replay, not a claim of zero-overhead production latency. A crash
can leave an arrival or a published report without a durable timing row. Such
missing records are unavailable timing evidence, not inferred timestamps.
Recording failures propagate through the existing recording-failure boundary.
A shared-service failure report is archived and immediately raises the existing
stop signal before the timing observer runs. That saved failed report therefore
has no `publication` row. Report it as a shared failure with missing publication
timing; never infer its boundary from the archive clock or treat it as a miss
with zero latency. This instrument does not delay shared-failure shutdown.

## Offline interpretation

Join a source event's onset frame to `frame-input`, then select the first
correct, attributable information report using its original saved content and
evidence frames. Retain the matching rule and source annotation separately.
The reaction interval is `[readable_lower - ingress, readable_upper - ingress]`.
Do not use API duration, archive time, or final-answer time as a substitute.
Do not clamp an earlier report to zero latency: classify it separately.

Keep missing onset inputs, sampled-out inputs, late drops, missing timing,
request failures, invalid report content, misses, and false reports distinct.
An onset sampled out after actual ingress still has an input clock, but no
claim of onset-frame model observation. A later report may use other frames.
Existing admissions and window evidence establish what the model actually saw.
The timing switch preserves model/weight, sampling rate, and image
preprocessing. Original answers and QA scores remain intact. The derived
reaction protocol adds a different task interface, source-event matching, and
outcome denominators; preserving the original QA data does not make those
evaluation definitions unchanged.

No standalone tests, test edits, lint, compilation, builds, or health probes
are authorized for this change. Source/diff review is not a passing test.

## Authorized complete conflict cohort

The human subsequently authorized the same fight-only observation version on
all 50 members of the current fixed conflict benchmark. The preparation entry
accepts `cohort=physical_conflict50` only with actual input timing and the exact
ordered members 0001 through 0050. The historical three-video selection remains
the default. This extends the cohort configuration, not the observation code,
model, prompt, native scheduling, sampling, or events-only 160-token/5-second
request settings. Each source retains its media duration and the existing
duration-plus-300-second source limit rather than inheriting the short clip's
360-second value. Ground truth stays offline and is not mounted as model input.

The fixed 50 videos are a selected benchmark cohort, not the total upstream
physical-conflict inventory. Report source-time reaction delays, left-censored
events, missing or conflicting time references, unmatched events, video-level
negative false reports, unknown observations and request failures separately.
Do not count multiple overlapping positive windows as separate source events,
or treat an unknown observation as a correct negative. The released 80-second
fight-only run remains a separate historical result, not the complete cohort.

## Paired four-event control on the same 50 videos

The human approved a sequential comparison: finish the current single-fight
50-video round, confirm its actual resource release, and then run the identical
50 members with `fight`, `smoking`, `fall`, and `weapon` enabled together. Only
fight outcomes are scored. The other three labels are additional requested
outputs, not new claims of annotated accuracy on these videos.

`capture_mode: native_four_events` selects a separate native event-only profile,
with those four labels in that order and no benchmark definitions. It does not
select the historical 52-label `benchmark` profile. Both arms use the existing
`events` inspector, strict JSON parser, 160-token output limit, five-second HTTP
deadline, main-baseline policy, source sampling, temporal window scheduler and
native admission. Fight remains standard sensitivity. The other native defaults
remain smoking high, fall standard and weapon standard. The response still maps
each requested label to true, false or null. Missing or invalid keys reject the
whole response under the unchanged parser; a readable raw fight value is not
retroactively accepted. One shared failed call is not four fight failures.

The four-event preparation requires `comparison_baseline_round` and
`--prior-timing-round` to identify the same completed single-fight 50. It retains
the baseline commit, complete source list and resource-release references. The
baseline must have terminal outcomes for all 50, not zero failed windows. Each
staging selection references its corresponding frozen baseline source config,
and preserves media path, order, metadata, duration and tracker settings.
Preparation copies the actual baseline Harness YAML, detector deployment source
and seed weights; it compares the declared unchanged source fields directly,
without new hash computations or media checks. New session identities, output
paths and the new application commit remain separate run metadata.

No model, detector, preprocessing, frame cadence, concurrency, source/drain limit,
face setting, compact lane, QA phase, GT or scoring rule is changed for this
control. The actual source manifest must confirm the four-label profile and the
same effective Harness configuration, while request receipts confirm the actual
lane, model, timeout and token cap. Source configuration claims alone do not
establish the runtime model parameters or detector SDK defaults.

The same OpenRouter route is retained; it does not pin the actual remote
supplier, queue or hardware. A replacement standard 24GB RTX 4090 is also not
necessarily the same physical machine. Record observed providers and HTTP
distributions rather than claiming that all service conditions were isolated.
Native windows may differ because request latency changes busy admission; do
not replay frozen windows to force an artificial match.

Analyze each arm independently using the same offline GT and timing protocol.
Keep all 50 videos and 41 source intervals with their missing-time, startup,
conversion-conflict, continuous-split, ambiguity, input and clock statuses.
Pair events by member and source event ordinal, not request or window IDs.
Report full-denominator outcome transitions alongside the subset with reliable
latency in both arms. For four-event minus single-fight delay, subtract each
arm's already-computed bounds as `[four_lower - single_upper,
four_upper - single_lower]`; never subtract clocks from different runs.
Unknown observations, request failures, invalid outputs, truncation and unknown
receipts remain separate. A no-positive video with some valid negatives can
also contain unknowns and failures, so its coarse category is not a visual-miss
diagnosis. Preserve original artifacts and all uncertain costs.

The current single-fight round is immutable. Adding this entry and publishing
its source does not start the four-event run or prove its results. The paired
launch follows actual single-fight completion and release, once per prepared
directory, with one owned GPU at a time. Independent software tests, lint,
syntax checks, builds and health probes are not authorized or run.

Preparation inherits the latest timing round's cumulative prior-resource
inventory before appending its direct GPU. This preserves earlier timing GPU
identities and recorded charges, including unknown amounts, instead of falling
back to the older capture-only inventory. No historical record is rewritten.
