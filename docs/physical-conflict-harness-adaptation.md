# Physical conflict reaction evaluation with Harness

We adapted the Harness native `events` information lane to evaluate scene-level fight responses. We retain its temporal windows and busy-time scheduling, request a smaller event set, and record the elapsed time from a source event frame entering the system to an independent report becoming readable. The current observer judges a window of up to four images. It does not identify a positive frame within that window.

This repository owns the Harness runtime, event profiles, capture adapter, timing instrumentation, and their adaptation guide. Benchmark tasks, annotation contracts, and benchmark-side scoring changes belong in the separate [GoodVision evaluation project](https://github.com/Numaira-Technology/goodvision/tree/codex/physical-conflict-on-off-eval/evaluation). This guide describes the Harness implementation and its measurement contract; it does not change a benchmark task or a running configuration.

## Source baseline and profiles

Use the pre-task Harness source at [6c28ddd](https://github.com/Numaira-Technology/goodvision-harness/tree/6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3) as the comparison baseline. A recent recovery or the latest team branch is not a replacement baseline.

| Profile | Source reference | Observation contract |
| --- | --- | --- |
| Pre-task Harness | `6c28ddd` | Configurable event labels; an existing fast events information lane and a separate grounding and person-fusion path |
| Single fight adaptation | [f457ec2](https://github.com/Numaira-Technology/goodvision-harness/tree/f457ec2686190538929f05cecf4cc4e5a1d76eb9) | `capture_mode: physical_conflict`; request only `fight`; use events output without compact grounding or final QA |
| Four-event comparison implementation | [e88e536](https://github.com/Numaira-Technology/goodvision-harness/tree/e88e5365b7b6e62273c8cf8f25e82bc385e9c073) | `capture_mode: native_four_events`; request `fight`, `smoking`, `fall`, and `weapon` together; score fight using the same evaluation rules |

The baseline supports a configurable label set, rather than a fixed four-event system. The separate historical benchmark profile requested 51 additional definitions plus fight and used parallel events and compact calls. That 52-label profile is not the single-fight or four-event configuration.

## Changes relative to Harness

| Component | Retained behavior and adaptation |
| --- | --- |
| Source and weights | We added evaluation profiles, a streaming capture adapter, a recorder, and a controller around the existing runtime. The selected evaluation stack uses YOLO26-S and Qwen3-VL-8B-Instruct without retraining. These selected weights do not establish equivalence to an unspecified historical production deployment. |
| Input | We replay the video at source speed and record each decoded original frame's input clock before sampling. The evaluation selects source images at 2 fps. VLM request cadence remains a separate native scheduling decision. GT and questions do not enter the observer. |
| Parameters | The selected detector settings are size 1280 and confidence 0.10. Provenance records max_det 300, but the worker does not pass that argument to `predict`; the installed SDK default has not been established from the remote environment. The events lane uses 160 output tokens and a five-second HTTP deadline, as the baseline native fast path did. Five seconds is a deadline, not a guaranteed response time. Fight retains the standard rule. |
| Loading and calls | The evaluation adapter uses the native `/v1/tracks` person route and the existing event inspector. It does not invoke the separate counting service. Single fight disables compact grounding, faces, and separate count, location, duration, and final-answer calls. |
| Scheduling and state | We retain the native temporal window selection, admission, and busy-time frame replacement. The cumulative evaluation work also adds request accounting and separates window failures, source termination, and shared failures. This is not a claim that the entire Harness source stayed unchanged. |
| Evidence export | We add original-frame input clocks and independent-report publication clocks. The lightweight recorder retains input dispositions, report evidence, source completion, and failures. A separate request ledger retains requests, receipts, and costs. The recorder omits person-result, admission, and scene-snapshot exports used by the broader evaluation path; person tracking and scene updates still run internally. |
| Answers and scoring | This mode does not generate a final QA answer. We preserve the original questions, annotations, and QA scoring and evaluate saved event reports against GT offline. The prior Harness-plus-final-answer method and historical native SAM-Ashley remain separate methods. |

We do not claim that the source baseline and an earlier production deployment used identical models, weights, provider hardware, preprocessing, or runtime configuration. The paired evaluation preserves the frozen settings of its single-fight arm.

## Observation and alarm boundaries

The baseline already separates two paths:

```text
Video frames -> temporal window -> events JSON -> independent information report
             -> person tracking and compact grounding -> person association -> event state
```

The single-fight adaptation selects the first path for measurement. It keeps person tracking internally but disables the compact calls that would supply person grounding. An events report uses `confirmation_state: scene_observation`. A valid scene-level fight observation can have `accepted=false`, `grounded=false`, and no associated event ID. Those fields describe missing person grounding; they do not turn a completed JSON response into a request failure.

The original person event state machine has different boundaries. A first accepted positive can create a `suspected` episode and emit `start`. The default second positive confirms it. The default second valid negative ends it and emits `end`. A null observation does not count as a negative. The runtime can also end an episode after sufficient tracked-participant absence, with a distinct absence reason. Four images in one grounded window contribute one observation, not four votes.

This evaluation measures the independent information report becoming readable. It does not measure a person event becoming confirmed or ended, webhook delivery, or browser display. A latency measured at this endpoint must not be labeled complete Harness alarm latency.

## Frame attribution

Keep these coordinates separate:

| Coordinate | Meaning |
| --- | --- |
| GT frame | The source annotation boundary; the current source intervals use one-based inclusive endpoints |
| `raw_frame_index` | The zero-based decoded original frame; source frame `S` maps to raw index `S - 1` |
| `frame_sequence` | The sequence of an image selected for downstream delivery; this is not the original frame number |
| Report `source_frame_index` | The sampled sequence at the window endpoint, despite the field name; this is not a model-selected positive frame |
| Report evidence | The images supplied in the request, joined back to original frame indices and video timestamps |
| Readable clock | The system time at which a report becomes available; this is not a video timestamp |

The baseline events inspector and the current adaptation both return one `true`, `false`, or `null` per requested event for the whole window. A positive window cannot establish which individual image the model judged positive. An offline GT match can identify an input frame inside an annotated positive interval, but that remains a GT-derived fact, not a model prediction for that frame.

Report the original frame indices and video times of the complete first positive window, its endpoint, and its publication time. Mark the model-selected positive frame as unavailable under this contract. Per-image predictions would require a new output protocol in both comparison arms; they have not been implemented here.

The replay uses original index divided by the source frame rate for its schedule, rather than decoder PTS. Variable-rate media or conflicting time bases require a separate alignment assessment. The source interval boundary is not an independently accepted natural-action boundary.

## ON and OFF reaction timing

The timing switch is off by default. With it enabled, we read a monotonic input clock after source-speed pacing and before sampling, encoding, or downstream callbacks. We bracket independent publication with readings before and after the parent releases the report-store lock. We retain both bounds from the same clock domain:

```text
ON response delay = [report readable lower - ON input,
                     report readable upper - ON input]
```

The ON analysis joins the annotated onset to its actual original-frame input record, then finds the earliest completed fight-positive information report whose supplied evidence intersects that source interval. It preserves cross-interval ambiguity and missing clocks rather than choosing a later convenient report. A sampled-out onset still has an input clock, but does not establish that the model saw that exact frame. Synchronous recording overhead remains part of the instrumented runtime.

Current analysis implements this ON candidate measure. OFF analysis remains unimplemented. An OFF extension must distinguish the first valid negative observation after a source boundary from a confirmed native `ended` state. It must also distinguish an ON-to-OFF transition from a model that reported false throughout the event.

For a one-based inclusive last positive frame `E`, the next frame has raw index `E` and is the candidate first negative input. An evaluator must establish that the next frame exists and belongs to an annotated negative interval before using its clock. Adjacent positive segments with no intervening negative frame do not create an OFF-to-ON transition. Events that continue to EOF have no observed OFF boundary. Mixed positive/negative windows, early negative reports, nulls, and request failures retain separate states. A null or a failed request does not close an event.

## Cohort and comparison

The current fixed physical-conflict selection contains 50 unique videos and 67 associated questions: 33 videos have one question and 17 have two. The questions comprise 50 event-recognition items and 17 temporal items. The video labels contain 25 positive and 25 negative examples. Thus 67 questions do not represent 67 independent videos.

The 23 positive videos with source time annotations contain 41 intervals; two further positive videos lack time boundaries. Preserve all 50 videos and 41 intervals in reporting. Separate missing time, clip-start events, source conversion conflicts, continuous segments, ambiguous attribution, missing event input, and missing clocks. Do not manufacture a reaction delay for these cases or remove them from the outcome inventory.

The paired comparison runs single fight first, releases its resources, then runs the same 50 videos with four events. Preserve media and order, detector and VLM route, preprocessing, window and scheduling policies, fight sensitivity, output budget, timeout, parser, and GT matching. The other native sensitivities remain smoking high and fall/weapon standard. Keep faces, compact, and final QA disabled. No accuracy claim for the other three events follows from fight annotations.

Keep the existing strict parser: an invalid requested key can invalidate the shared four-event response. Count a shared call once rather than four failures. Report actual providers and request durations; the same API route does not guarantee the same supplier, queue, or hardware. Native busy-time admission can produce different windows as latency changes, so preserve those differences rather than replaying a fixed window plan.

For each arm, report temporal hits, source events without positive reports, false reports in negative videos, null observations, request failures, format failures, truncation, and unknown receipts. A missing positive report can result from sampling, uncertain observations, or request failure; it does not by itself establish a visual classification error. Keep failures in the relevant denominators. Unknown is not a correct negative or a known zero cost.

Compare latency only for events with usable timing in both arms, alongside the full outcome transitions. For four-event minus single-fight latency, the difference bounds are `[four_lower - single_upper, four_upper - single_lower]`. The common successful subset cannot replace the full-cohort quality results.

## Implementation references

- [Original Harness event inspector](https://github.com/Numaira-Technology/goodvision-harness/blob/6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3/src/goodvision_harness/vlm_compact_model.py) and [original independent window and publication path](https://github.com/Numaira-Technology/goodvision-harness/blob/6c28dddd26bed7a7be50762a4c7dd6b796b2dbe3/src/goodvision_harness/vlm_independent.py).
- [Single-fight profile](https://github.com/Numaira-Technology/goodvision-harness/blob/f457ec2686190538929f05cecf4cc4e5a1d76eb9/src/goodvision_harness/evaluation/profile.py), [streaming adapter](https://github.com/Numaira-Technology/goodvision-harness/blob/f457ec2686190538929f05cecf4cc4e5a1d76eb9/src/goodvision_harness/evaluation/native_stream.py), and [lightweight recorder](https://github.com/Numaira-Technology/goodvision-harness/blob/f457ec2686190538929f05cecf4cc4e5a1d76eb9/src/goodvision_harness/evaluation/stream_recording.py).
- [Person event state machine](https://github.com/Numaira-Technology/goodvision-harness/blob/f457ec2686190538929f05cecf4cc4e5a1d76eb9/src/goodvision_harness/events/state_machine.py), which has separate suspected, confirmed, and ended semantics.
- [Four-event profiles](https://github.com/Numaira-Technology/goodvision-harness/blob/e88e5365b7b6e62273c8cf8f25e82bc385e9c073/src/goodvision_harness/evaluation/profile.py) and [paired timing controller](https://github.com/Numaira-Technology/goodvision-harness/blob/e88e5365b7b6e62273c8cf8f25e82bc385e9c073/tools/streaming_walltime_trial.py).

Model request caches, media, observations, private annotations, and cost ledgers remain outside Git. Benchmark changes belong in the evaluation repository. This guide does not change the original benchmark or supply GT to an observation model.

