# Evaluation-only reaction timing

The approved first run uses UBI F_8_1_0_0_0 and the negative clips
N_251_0_0_1_0 and N_478_0_0_1_0 (80 seconds in total). Launch follows the
prerequisite source/annotation deliverables and publication of this change.
One positive clip supports a case latency, not a population p90.

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
