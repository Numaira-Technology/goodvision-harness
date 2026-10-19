# Evaluation-only reaction timing

The approved run of UBI F_8_1_0_0_0 and the negative clips N_251_0_0_1_0
and N_478_0_0_1_0 (80 seconds in total) finished using application revision
`091f41d`. Its resources were released. It used the historical 52-label,
parallel events/compact profile, not the later fight-only mode. One positive
clip supports a case latency, not a population p90. The later mode is published
source without a new prepared, deployed, or measured run. Do not replay the
completed run to fill that distinction.

## Release and runtime correspondence

Git source, a prepared runtime, and an observed result are separate records.
Documentation-only commits after the application revisions below do not create
another runtime or latency result.

| Component | Completed 80-second run | Later fight-only source |
| --- | --- | --- |
| Application and controller | `091f41d`; frozen under the workspace experiment `detector-reaction-ubi3-20261019/checkout` | Introduced by `a07b27f`, source-error classification refined by `8c3e5fb`; no new frozen experiment |
| Native capture mode | Historical benchmark profile; 52 event labels with compact grounding; final QA disabled | Explicit `capture_mode: physical_conflict`; fight events only; no compact or QA export |
| VLM route and budgets | Qwen3-VL-8B-Instruct through OpenRouter; events 1600 tokens/75 seconds, compact 8192 tokens/180 seconds | Same model route; events 160 tokens/5 seconds; actual future provider and latency remain unobserved |
| Person detection | YOLO26-S, explicit size 1280 and confidence 0.10; source provenance declares max_det 300, while the worker leaves that argument to the SDK default; internal `/v1/tracks` protocol | Same native internal tracking path, not the separate `/v1/count` evaluation protocol |
| Input and output evidence | Original source configurations, `plan.json`, `ready.json`, `launch.json`, per-frame input and publication records, `round-outcome.json`, `resources-after.json` | Source and documented contract only; no corresponding new runtime evidence |

The native tracking client in [native_stream.py](../src/goodvision_harness/evaluation/native_stream.py)
selects `RichLabelHTTPClient(mode="tracks")`. The client rewrites the configured
labels endpoint to `/v1/tracks`; the tracked
[detector worker](../workers/detector_rich_label/app.py) implements both routes.
The completed run's frozen detector `app.py` and `detectors.py` have the same
text as those two current tracked worker files. This two-file correspondence
does not establish that the full deployment bundle is interchangeable. The
separate historical `/v1/count` service is not a dependency of native event
tracking; its integration gap must not be used to block this mode.
The frozen worker log contains completed `/v1/tracks` requests. That evidence
is stronger than its Uvicorn-only startup receipt, but is still historical
execution, not a new deployment of the fight-only source. The worker's
`model.predict` call does not explicitly pass `max_det`; this review did not
recover the installed SDK default from the remote environment or change it.

### Dependencies outside the application revision

The published [timing controller](../tools/streaming_walltime_trial.py) still
prepares a runtime by copying several existing workspace artifacts. A checkout
of the application commit alone is therefore not a self-contained deployment.

| Runtime component | Controller source or completed-run evidence | Boundary |
| --- | --- | --- |
| Application freeze | `freeze_source` in [saved_streaming_recovery.py](../tools/saved_streaming_recovery.py) archives committed `src` and two saved-streaming helpers; the controller copies its committed entry separately | The helper does not archive the detector deployment tree, images, or every run input |
| GPU service and configuration | `prepare` copies `source_round/gpu/source`, `harness.yaml`, and `cohort/seed-weights.json`; the completed run names `detector-streaming182-yolo-s-resume41-20261019` as that source | These are separate frozen workspace artifacts, not implied by the application Git revision |
| Resource control and release | `prepare` copies `control.py` and `driver.py` from `detector-rf-count36-1288-resume5-20261016` into `resource-control.py` and `resource-release.py` | The adapter retains local Docker image references, a dedicated network, and host tooling; no portable image build is established by this document |
| Detector environment and weights | Frozen `gpu/source/workers/detector/requirements.txt`, deployment bootstrap, `cohort/seed-weights.json`, and `gpu/output/weights/weights.json` | Requirements include Ultralytics 8.4.0 and Torch 2.7.1; `yolo26s.pt` declares the Ultralytics assets v8.4.0 source. The installed environment and staged weight receipt are runtime evidence, not a new build result |
| Source configuration and media | Three `sources` files, original media paths, staging selections, and model-ledger seed in the private experiment | Media, credentials, ledgers, and experiment data are not repository release contents; GT remains offline |

The completed run's `plan.json` names these origins. `ready.json` embeds that
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

This inventory documents the existing dependency boundary without migrating
the resource adapter, changing the detector, rebuilding an image, or starting
another run. A later portable release would need to include or explicitly
version those deployment dependencies and record its actual preparation and
execution separately. It is not claimed here. No tests were run (no user
authorization).

## Physical-conflict event tracking

New preparations from `tools/streaming_walltime_trial.py` use
`capture_mode: physical_conflict`. This selects the existing native `events`
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

Without this explicit capture mode, the complete benchmark profile and its
existing evidence recorder retain their behavior. Frozen historical checkouts
and artifacts are not changed. This source change does not launch another
trial or claim an observed latency reduction. No tests, model calls, GPU
creation, preparation or launch were run for this change.

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

The remaining sections describe the original `091f41d` timing instrument.
The physical-conflict mode above supersedes its benchmark labels, paired
grounding and QA export only when explicitly selected.

This is the Harness evaluation method, not historical native SAM-Ashley. The
pre-task Harness baseline remains `6c28ddd`; cumulative changes are recorded in
[the component guide](guides/unified-eval-components.md) and
[the integration record](design/2026-10-19-hash-migration-integration-status.md).
The immediate implementation base is `7b7c5d3`, not a replacement baseline.

Relative to those cumulative categories, this increment changes source replay
pacing only under the evaluation switch, adds local input/publication export,
and extends the bounded timing controller to the approved new clips. Detector
and VLM source/weights, image preprocessing, inference parameters, loading and
calls, busy admission/replacement and report ordering, final-answer contracts,
and PR63 scoring retain their existing versions. Earlier adapters, request
accounting, evidence projection and answer/score changes remain part of the
cumulative method; they are not described as an unchanged original Harness.
A fresh whole-branch baseline line audit was not performed for this increment.

## Switch and boundaries

Set `evaluation_timing: true` in a frozen native-stream source configuration.
The default is false. This is an evaluation replay instrument, not a model,
prompt, scoring, or production delivery change. No annotations enter capture.
The existing `tools/streaming_walltime_trial.py` owner accepts this switch in
staging only for clips 0040, 0003 and 0030, in that order. Preparation requires
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
No model/weight, sampling rate, image preprocessing, task definition, original
answer, or original score is changed by this instrument.

No standalone tests, test edits, lint, compilation, builds, or health probes
are authorized for this change. Source/diff review is not a passing test.
