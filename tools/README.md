# Tools

Run command-line tools from the repository root in an activated development
environment. Most support `python tools/<name>.py --help`; runtime helpers are
used through their documented callers. See [CONTRIBUTING.md](../CONTRIBUTING.md)
for installation and offline checks. Generated outputs belong in `output/`,
`outputs/` or `dist/`.

## Startup and delivery

| Tool | Purpose |
| --- | --- |
| `start_delivery.py` | Start the recorded `2026.09.10-full.4` customer package. |
| `start_windows_hybrid_demo.ps1` | Start the native Windows / Docker worker demo. |
| `start_local_gpu.py` | Detect NVIDIA GPUs and start SAM on this host using the Contour runtime; `--streams N` provisions camera lanes, `--with-local-inference` also starts the local VLM and CPU processing companion. |
| `start_modular.py` | Deploy application, SAM or VLM separately; `main --inference-server ... --inference-protocol goodvision` delegates VLM image preparation to the model host. See [modular services](../docs/guides/modular-services.md). |
| `export_full_delivery.py` | Export the recorded full.4 image pair, metadata and checksums. |
| `run_full_delivery_smoke.ps1` | Run isolated image protocol checks; accepts candidate image tags. |
| `delivery_smoke.py` | Exercise a disposable main container using synthetic frames. |
| `paddle_delivery_smoke.py` | Check worker HTTP behavior with synthetic models. |
| `face_gpu_smoke.py` | Check runtime packages or run real accelerated face warmup. |
| `customer_check.py` | Installed check dispatcher; mode comes from its installed filename. |
| `runtime_gateway.py` | Runtime process supervisor and customer diagnostic filtering. |
| `update_docker.py` | Update an installed Compose harness or Contour deployment from source or loaded images, with readiness checks and image rollback. [Usage](../docs/guides/docker-updates.md). |
| `customer_update.py` | Verify a customer package, stop local analysis, and restore startup settings for the update window. |
| `customer_update_gui.py` | Windows customer update window; distributed as `Update.exe` with prepared images so customers can update by double-clicking. [Customer steps](../docs/guides/docker-updates.md#customer-updates-on-windows). |
| `package_customer_docker_update.py` | Package validated images and a standalone Windows updater in a customer ZIP. [Maintainer steps](../docs/guides/docker-updates.md#package-a-windows-customer-update). |

See [release development](../docs/guides/release.md) and the
[customer deployment guide](../DEPLOY.md). Recorded image tags and release
checks describe that delivery, not subsequent source changes.

## Evaluation

| Tool | Purpose |
| --- | --- |
| `face_provider_benchmark.py` | Compare ONNX Runtime face inference providers. |
| `face_llm_interop_eval.py` | Check real CPU face inference alongside synthetic remote blocking. |
| `low_resolution_face_eval.py` | Evaluate local media/models with a disposable face gallery. |
| `low_resolution_model_eval.py` | Replay listed samples through a real model service. |
| `replay_stress.py` | Paced active-scene replay with recorded tracks, temporal HTTP fault injection, compressed timelines and a shared request budget. [Scope and usage](../docs/guides/diagnostics-replay.md). |
| `low_resolution_tracking_eval.py` | Replay saved observations through person tracking. |
| `summarize_sam_timing.py` | Join saved client, SAM worker and cloud pool timing logs offline, keeping stage durations and missing measurements separate. |
| `low_resolution_evidence_eval.py` | Exercise synthetic event continuity and H.264 evidence offline. |
| `benchmark_display_cadence.py` | Compare local synthetic display cadence, lag and frame drops. |
| `evaluate_pipeline_latency.cjs` | Play a complete original video through an isolated real pipeline and production Chrome renderer; record actual presented-frame contour coverage, SSE arrivals and delay. Requires Playwright, Chrome, and an authorized real-model service with a same-origin byte-range video URL; see the script header and [measurement record](../docs/validation/network-and-display.md). |
| `rich_label_benchmark.py` | Benchmark the JPEG `/v1/labels` worker on local video. |
| `vlm_speed_matrix.py` | Measure the native rich client with an explicit local `--cases` JSON file. |
| `vlm_benchmark/modes.py` | Compare events, compact grounding, and parallel with per-lane results and optional telemetry. `--monitor` shows live inputs/results; `--record` saves the browser viewport as MP4. [Guide](../docs/guides/vlm-mode-benchmark.md). |
| `vlm_benchmark/local_model.sh` | Package/install/start/check a pinned local Qwen3.8 27B W4A16 service on a 4090, then run the mode benchmark. [Guide](../docs/guides/vlm-local-model.md). |
| `vlm_simple_jev_benchmark.py` | Compare Simple Jev direct vision with Qwen event-only and full JSON; see below. |
| `vlm_delivery_probe.py` | Independently receive real VLM/general/event SSE, poll SAM/face progress, and pull live/annotated MJPEG; preserve arrival times and pending positive observations. |

Model/GPU/service runs need the corresponding environment and authorized inputs.
Protocol success is not model accuracy. See the
[validation reports](../docs/README.md#validation-records) for reviewed measurements.

## Simple Jev direct vision comparison

Run only with authorized media and API usage. The default batch sends 20 public
Simple Jev requests without authentication and 76 OpenRouter requests using
`OPENROUTER_API_KEY` from the given environment file (or environment). The four
original images go directly to each service. The public demo accepted images in
the recorded experiment even though its documentation describes text-only input;
future deployment behavior may differ.

```powershell
python tools/vlm_simple_jev_benchmark.py --env-file .env --video path/to/authorized-sample.mp4 --output artifacts/simplejev-comparison --repeats 3
```

The output directory must not already exist. Defaults use six four-frame windows,
720-pixel images and three repetitions, plus one warmup per route and one blank
control. Each request has a 45-second timeout and is not retried. Results include
failed attempts; summaries exclude controls and warmups. `--cases`, `--endpoints`
and `--side` can restrict the batch. This is an offline experiment, not a
production integration. Event-only cases do not supply person boxes.

See [recognition findings](../docs/validation/recognition-quality.md) for the
recorded configuration, latency scope, observed misses and comparison limits.

### Frozen multi-video datasets

`vlm_simple_jev_dataset_benchmark.py` reuses exactly the same model request and
decode functions for multiple videos. A dataset JSON contains `samples` with
neutral `sample_id` values, local `path`, `endpoint`, video `sha256`, four
`frame_sha256` values, `group`, and separate `expected` labels. References must be
visually reviewed and marked `annotation_status: verified_before_inference`
before the runner will accept them. Filenames, groups and reference annotations
are never inserted into model messages.

```powershell
python tools/vlm_simple_jev_dataset_benchmark.py --dataset path/to/dataset.json --env-file .env --output artifacts/dataset-preflight --prepare-only
python tools/vlm_simple_jev_dataset_benchmark.py --dataset path/to/dataset.json --env-file .env --output artifacts/dataset-run --repeats 3 --max-requests 456
```

With 30 videos, this runs 450 measured requests, five warmups and one blank
control. It interleaves fall, daily-activity and smoking groups, rotates route
order, keeps inference serial and never retries. The preflight only verifies
media/frame fingerprints and writes a manifest. Output directories must be new.
Unknown predictions and request errors remain separate from negative answers;
repetitions of a clip are not independent samples.

## Repository hygiene

`check_repo_hygiene.py` checks tracked paths against repository ignore rules;
run it without arguments locally or in CI.

Superseded deployment recipes, the old fixed-bundle auditor, and redundant VLM
probes/fixed comparison cases have been deleted. Their source is available in the
pre-cleanup main revision `a7e5ec8`; there is no second archive
copy to maintain in the working tree. Maintained topic summaries and necessary reviewed evidence remain in
`docs/validation/`; raw runs stay in ignored output directories.

`audit_customer_bundle.py` checks customer bundles or the repository (`--repository`)
for the anonymization patterns recorded in [交付与升级检查](../docs/validation/delivery-and-updates.md).


## Parallel event JSON experiment

`vlm_parallel_benchmark.py` compares one 8B model/provider with full output,
conditional compact-negative output, event-only JSON, and concurrent event/compact
requests on identical frozen windows. Thirty clips and three repetitions use 450
measured requests plus five warmups. Each pair publishes its two replies separately;
different trials do not overlap. It uses the existing frozen dataset format above.

```powershell
python tools/vlm_parallel_benchmark.py --dataset path/to/dataset.json --env-file .env --output artifacts/parallel-run --repeats 3 --max-requests 455
python tools/vlm_parallel_analyze.py artifacts/parallel-run
```

Real API usage and media transmission require authorized inputs. Output directories
must be new; there are no retries. Latency excludes initial video decode and SAM.
The experiment retains failed attempts, missing judgments and unknowns. A short
negative reply is not useful acceleration if it misses positive events.

The isolated runtime defaults to `SCENE_VLM_OUTPUT_MODE=parallel`; `full` and
`compact` require an explicit override. Runtime information publishes through existing VLM
interfaces. The grounding lane alone supplies display/person-event evidence and
is retained separately on `branch.grounding_results`. When both lanes admit a window they share `request_id` and
`window_id`. Each lane admits independently; a full grounding lane does not block
new information windows. Information completes out of order and discards superseded
late results, with an HTTP timeout capped at five seconds; grounding remains ordered.
Each lane allows `SCENE_MAX_IN_FLIGHT` requests (up to eight total in parallel mode).
Before replay, verify `/v1/vlm/status` reports `output_mode=parallel`,
`state=running`, and both `events` and `compact` lanes. Stop a mismatched run
before playback and save the effective status with its evidence.
When display has pending positives or active person events, subsequent windows
request full grounded negative/uncertain follow-up; `no_events` cannot clear a person.

## Text observation comparison

`vlm_jev_benchmark.py` compares direct JSON with Qwen descriptions followed by
TypeSafe Jev text classification. It is an offline experiment; the intermediate
description lost fall evidence in the recorded comparison and is not a production
replacement. It reuses `jev_observation.py` and the current strict result parser.

```powershell
python tools/vlm_jev_benchmark.py --env-file .env --video path/to/authorized-sample.mp4 --output artifacts/jev-new-run --repeats 3 --smoking high --description-style brief
```

Use a new output directory. `--description-style temporal` tests per-frame prose;
`--smoking standard` changes that sensitivity. Qwen receives the authorized images,
TypeSafe receives descriptions only; both services require configured credentials.
See [recognition findings](../docs/validation/recognition-quality.md).
