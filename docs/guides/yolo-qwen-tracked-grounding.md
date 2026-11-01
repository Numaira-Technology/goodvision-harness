# YOLO26-S tracking + Qwen3.5 35B

> 2026-10-20：本文实验工具使用独立的 `tools/benchmark_runtime/`；产品 grounding 已恢复 main。历史结果不等于恢复后的产品性能，见[代码与测试边界](pr48-49-person-grounding-integration.md)。

The current version shares its runtime implementation with the PR48/49 main
integration. See [integration and rerun contract](pr48-49-person-grounding-integration.md).

This is an opt-in benchmark path. Importing it or running its default command
does not rent a machine, start a detector, read an API key, or call a model.

## What changes

The detector worker from PR #42, also used by PR #48, owns person detection and
the existing IoU/motion tracker. The new backend broker submits each source
frame once to `/v1/tracks`, in source order. It keeps a separate worker session
per camera and resets sessions between trials. The preview is only a reader;
closing or opening a browser cannot affect inference scheduling.

The Qwen grounding request still contains the same 2–4 original full-scene JPEGs
at 720 maximum side and quality 90. Frame-specific normalized locations and
track IDs are added as metadata. Qwen returns those IDs and event judgments,
never new boxes. Internal boxes remain necessary to locate each person.

The six-event policy and sensitivities are unchanged and hash-checked:
`d9c487c37404df2f6d1f4d1f3da39559abffe28f59c76e7fef85f4a9f9e48a7b`.
Every last-frame person must have a judgment for each person event. Ground
litter remains scene-level. An unassigned event row is allowed for an event
whose actor is outside the available tracks; detector omissions never imply
that the event is absent. Predicted-only tracks are not sent as observed people.
Missing, duplicated or invented IDs, missing labels and unsolicited coordinates
fail validation. Uncertainty remains `null`, never an automatic negative.

Events keeps its existing scene prompt. Grounding uses the new person inputs.
Parallel runs both, sharing the same detector snapshots across overlapping
windows. It does not run YOLO once for each VLM branch.

Track IDs are scoped to a camera session, not real-world identity. The existing
tracker can still switch or lose IDs during crossings, occlusion or re-entry.
This change does not claim to solve re-identification.

## Frozen workload and accounting

- Qwen `qwen/qwen3.5-35b-a3b` through OpenRouter, Alibaba preferred, same-model
  provider fallback allowed. This is not a fixed-Alibaba claim.
- Twelve streams; 55.798021 seconds; 1.25 windows/s per camera; two rounds of
  events, grounding and parallel. Each branch has 828 planned windows per round.
- Same staggered source release; 2.5 source frames/s; at most four frames per
  window; global concurrency 48, per-camera shared concurrency 4, paired parallel
  scheduling. VLM request-rate cap 40/s, admission wait 1 second.
- Existing HTTP limits: events 5 seconds and grounding 20 seconds; no automatic
  retries. Three seconds is the scoring deadline, not the transport timeout.
- Detector queue capacity 48, one GPU call at a time, two-second queue-age and
  detector-call bounds. The broker has a two-second snapshot wait. These costs
  are measured, not subtracted from end-to-end latency. This first version
  requires the detector to sustain roughly 30 frames/s across twelve streams;
  actual capacity must be measured before claiming the target is reachable.
- The source is offered to the VLM branch before its detector task is enqueued.
  Source-to-publication timing therefore includes waiting for the detector,
  admission, Qwen and publication. Model/HTTP timing is reported separately;
  neither is claimed as pure GPU inference time.
- Detector failures, queue expiry and unavailable snapshots prevent the
  corresponding grounding HTTP call and remain incomplete planned windows.
  They are separately counted from VLM 429, timeout and schema failures.
- Historical results remain in raw records and totals, but cannot overwrite
  newer live state. Parallel's joint metric requires both branches of the same
  window to complete within the deadline. Acceptance requires **each** camera
  to achieve at least 66/69 timely complete windows (95%), not a pooled average.

## Display and recordings

By default, each camera shows one original-speed 15 FPS preview, no right-hand
evidence pane and no bounding rectangles. `--person-display boxes` adds boxes;
`--person-display off` hides annotations. These switches change only the monitor,
not detection, person judgments or load. The backend's observed person IDs appear as
text at their source positions. A position older than 0.5 seconds relative to
the playing video is hidden rather than attached to a different current person.
This can mean fewer visible IDs when detection is slow; it must not be presented
as proof that nobody is present. Person judgments below the video include their
evidence time. No event color is assigned to a person from a scene-wide alarm.

## Commands

From the repository root, inspect the plan without any network/model calls:

```sh
PYTHONPATH=src python -m tools.tracked_grounding
PYTHONPATH=src python -m tools.tracked_grounding --baseline
```

Offline tests, including real spawned workers talking to local fake detector
and Qwen HTTP servers:

```sh
PYTHONPATH=src python -m pytest tests/test_tracked_grounding.py
PYTHONPATH=src VLM_MONITOR_BROWSER_TESTS=1 python -m pytest tests/test_tracked_grounding_browser.py
```

Browser regression requires installed Chromium/Playwright, PyAV and FFmpeg.
It checks all three modes with twelve video tiles, ID drawing without boxes,
stale-ID hiding, historical-result guarding and final-summary visibility.

Only after explicit authorization, on the Linux test host with the existing
YOLO26-S worker running and `OPENROUTER_API_KEY` configured:

The existing worker image in `workers/detector_rich_label/Dockerfile` listens on
port **8092**. Start it with `DETECTOR_MODEL=yolo26-s`, `DETECTOR_DEVICE=cuda`
and its weights directory mounted at `/models`. `/health/tracking` must report
`ready: true`, `gpu_ready: true`, and `model: yolo26-s`; the suite checks this
before reading credentials or starting paid calls.

```sh
PYTHONPATH=src python -m tools.tracked_grounding --real \
  --video /path/to/output.mp4 \
  --detector-url http://127.0.0.1:8092 \
  --packet /path/to/astra-unified-v2-20261009/packet \
  --output outputs/yolo26s-qwen35-new-run
```

The runner verifies the original video SHA and exact Astra evidence before
paid calls, locks against concurrent suites, runs a short one-camera preflight,
then six twelve-camera trials with warmups and cooldowns. It uses the existing
memory-backed admission database. A fresh output directory is required.
Use `--baseline` without `--detector-url` for the original Qwen-owned grounding
under the same current load, timeout and scheduling settings. This is a new
matched baseline; historical October 10 results had other scheduling details.

Outputs include `report.html`, `tracked-comparison.json`, per-trial
`detector.jsonl`, raw `reports.jsonl`, telemetry and MP4 recordings.
The report includes sent/complete fractions, per-camera three-second coverage,
P50/P95, detector losses, HTTP status counts and Astra agreement.
The existing Astra labels only score aggregated scene judgments. They cannot
prove person-ID association accuracy; that requires person-level annotations.

No formal model run or GPU capacity claim is part of this implementation review.

## Four-second cloud comparison (2026-10-19)

`--load-profile four-second` keeps twelve cameras, the three modes and two rounds,
but offers one window every four seconds per camera (`observation_fps=0.25`)
with **four shared VLM slots across all cameras and both parallel branches**.
The historical profile remains the default. Per-camera limit 4, the 40/s rate
ceiling, one-second admission wait, timeouts, prompt, provider routing and six-event
policy are unchanged. This is the existing benchmark at reduced load, not PR48's
extra single-image preview selection call or a promise of a response every four seconds.

Source/YOLO sampling stays **2.5 FPS**, with the same 720px JPEG evidence and
2–4 frame windows; reducing VLM cadence must not silently reduce source sampling.
The original clip has **14 planned windows per camera**, **168 per branch/round**,
and **1,344 formal branch requests** over the six trials, plus separate preflight
and warmup calls. Unsent/failed/missing windows still count against this plan.
At this small sample size, a per-camera 95% target requires all 14 windows.

Offline plan (no credentials or network):

```sh
PYTHONPATH=src python -m tools.tracked_grounding --load-profile four-second
```

On the Linux host, after checking the existing YOLO26-S service and credentials:

```sh
PYTHONPATH=src python -m tools.tracked_grounding --real \
  --load-profile four-second --person-display ids \
  --video /path/to/output.mp4 --packet /path/to/reference \
  --detector-url http://127.0.0.1:8092 \
  --output outputs/yolo-qwen-four-second-new-run
```

Cloud runs additionally produce `api-comparison.json` and `api-comparison.html`.
These report completed nonstream API send-to-full-body latency, excluding request
construction, detector work, local admission, parsing and publication. Missing
HTTP timing stays unknown. Parallel pairs use exact camera/window IDs and require
both APIs individually within three seconds; no combined wall-clock API latency
is invented. The existing `report.html` retains explicitly labeled end-to-end
diagnostics and Astra scene agreement. Warmup/preflight never enter either comparison.

To add the historical cloud run, without issuing any model requests:

```sh
PYTHONPATH=src python -m tools.tracked_grounding.api_report \
  outputs/yolo-qwen-four-second-new-run --baseline /path/to/historical/results
```

This is a historical comparison: the new run uses the shared main-based runtime,
so changes cannot all be attributed to frequency/concurrency. Provider capacity,
fallback routing and cache state can also differ. Keep source manifests and do
not overwrite the October 17 results.

The previously tested Qwen3.8 27B (`cerebras/fp16`, fixed routing) and GLM 5.3 Flash
(`coreweave/nvfp4`, low reasoning) are additional candidates, not implemented by
this fixed-35B profile. They require their historical route adapters and a tracked
person preflight: 27B previously needed two-image packaging; GLM needed reasoning
enabled and an 8,192-token output allowance. Do not substitute their names into
the 35B command and assume compatible prompts, HTTP timing or output limits.


## Cloud candidates: GLM and 27B with YOLO IDs

The benchmark-only `tools.tracked_grounding.cloud_candidates` entry point reuses
strict person-ID grounding and the original six-class policy. It preserves the
previously tested route-specific streaming transport: Qwen3.8 27B is fixed to
`cerebras/fp16` with paced sends; GLM5.3 Flash is fixed to `coreweave/nvfp4` with
reasoning `low` and an 8,192-token allowance. Both retain the original source
frames through lossless context-strip packaging plus the untouched latest frame.
The earlier frames' detector coordinates remain relative to each source panel.

**Current requested scope (October 19):** the four-second experiment and the
Qwen35B formal run were cancelled. The 35B attempt made one single-camera
preflight request and no measurement trials. GLM and 27B are the remaining
candidates. The user then authorized running both on the original load;
the active run uses 1.25 windows/second/camera and 48 shared slots.

Planning performs no model calls:

```sh
PYTHONPATH=src python -m tools.tracked_grounding.cloud_candidates --arm qwen27_paced
PYTHONPATH=src python -m tools.tracked_grounding.cloud_candidates --arm glm_low
```

The candidate default is the **original** profile: twelve cameras, 1.25 VLM
windows/second/camera, 48 global slots, three modes, two rounds, 828 planned
windows per branch/round. The optional `--load-profile four-second` remains
available in code, but is not the active experiment. Execution additionally
requires `--real`, input/reference paths, the detector endpoint and a new output
directory. No command above starts a paid test.

Candidate API latency uses SSE send-to-full-body timing. The SSE header timer
already excludes request construction and local pacing, so request-build time
must not be subtracted again. Failed or unsent windows remain in completion
rate denominators. Candidate model/provider attribution is verified after each
stage. Historical 27B/GLM tests used VLM-produced boxes, so comparisons must also
identify the change to YOLO person IDs; they are not frequency-only comparisons.

### GLM provider recovery diagnostics

`tools.tracked_grounding.glm_health_probe` runs exactly five Events and five
Grounding requests sequentially, with two seconds between completed calls and
no request retries. It reuses the source clip, YOLO IDs, six-class parsers,
two-image packing, `low` reasoning and 8,192-token budgets. Its health gate
requires all ten valid replies with verified model/provider attribution. These
are service diagnostics, not twelve-stream performance results.

`tools.tracked_grounding.glm_recovery_warmup` requires a passed health result for
the same provider. It runs twelve streams for ten seconds in each mode, keeping
the original 1.25 Hz / 48-slot load. It stops on input/result corruption, detector
failure, no valid replies in a branch, or any 401/402/403/429. Passing indicates
readiness to measure; it does not certify 95% completion within three seconds.

The default GLM route remains `coreweave/nvfp4`. The explicitly selected
`--glm-provider deepinfra/fp4` on `cloud_candidates` permits a separate fixed
DeepInfra run without fallback. The health/warmup commands use `--provider`
for the same choice. Provider changes must use new output directories and be
reported separately: provider, quantization implementation and available
capacity may differ even when the model and inference settings are unchanged.
Every executable diagnostic requires `--real`; planning performs no API calls.
