# Runtime and optional recognition

[Documentation index](../README.md) · [Repository overview](../../README.md)

Commands and unlinked file paths in this guide are relative to the repository root.

The service publishes source-correlated safety/face observations and annotated
video. This guide covers runtime behavior and optional recognition backends.

## Harness behavior

The service performs one external model request for whichever visual safety
events are selected, while the local known-person matcher checks the same frame
only when `stranger` is selected. In mixed streaming sessions, safety and local
faces have independent bounded request slots and publish independently on completion.
Each message retains its own source frame; capability cards use each capability's
latest observation. A slow remote request cannot consume the local face slot,
and a slow face result cannot delay a completed safety judgment.
It does not run one process or one LLM call per feature. Omitting `features`
selects all five capabilities. Before a real multi-event
or face-only session accepts its first frame, the service loads the local engine and
runs one blank inference to remove its first-frame cold start; dry-run skips
that warmup. The default runtime settings are:

- source/face sampling: 2.5 FPS, 1280 px maximum image side, JPEG quality 90;
- safety input: 320 px maximum image side, JPEG quality 75, for the single
  remote safety request;
- display output: up to 30 FPS using original source frames after camera-view selection,
  JPEG quality 90, delayed 2.5 seconds, with intervening frames inheriting
  the preceding annotations;
- one-frame windows with stride 1 and six concurrent model requests;
- `max_tokens=1200`, no API retry, and one explicitly selected model route
  with automatic route fallback disabled;
- one multi-label safety prompt that independently evaluates `fight`,
  `smoking`, `fall`, and `weapon` and permits simultaneous positives.
  The console offers standard/high sensitivity independently for each safety
  event before starting. Smoking defaults to high: plausible smoking gestures
  and context can trigger an alert without identifying a smoking item. Standard
  smoking requires an identifiable item. Other events default to standard.
  Safety requests retain full-scene, one-frame, 320 px inputs; sensitivity does
  not change Paddle or face recognition. Validate both missed events and false
  alarms on fixed test recordings when choosing a sensitivity.

The original headless CLI keeps its prior event and result protocol. Browser
testing now uses one unified page, based on the familiar original Demo UI.
Both browser-camera and network-stream modes use the same grounded delivery
pipeline and expose customer JSON plus analyzed and raw video outputs. The
legacy single-event and face-only modes remain available as diagnostic API
modes; the page lets the operator select any non-empty feature combination.

When every selected safety event is negative, the model returns only
`{"no_events":0.95}` (using its confidence in that verdict). The service expands
this into the existing per-event negative judgments, with bilingual summary
text and no boxes. It omits auxiliary actor detection on these frames. Positive
responses retain the full evidence and box contract and the 1200-token budget;
face recognition remains independent, and safety images remain 320×180 for
16:9 input. Legacy full negative responses are still accepted.

The console's route check selects the lowest median latency among routes with
three successful responses; model size no longer gets priority. Each route row
also has a manual selection button, available before starting detection and
outside a route check. Both automatic and manual selections are saved for the
next session and survive restart. `POST /v1/deployment/routes/select` accepts
only `{"route_id":"A"}` through `{"route_id":"H"}`, using the displayed IDs.
Running another automatic check replaces the saved selection with its winner.

## Display and evidence

For every successful JSON result, the harness still annotates the exact newest
frame that the result references. `GET /v1/video/latest.jpg` and every
multipart JPEG in `GET /v1/video/stream.mjpeg` carry
`X-Analysis-Message-Id`; its value is the same as the JSON `message_id`. This
exact evidence feed remains at model-result cadence.

For the display path, the container buffers source frames for the delay selected
in the console. The annotated output uses each original source frame at up to
30 FPS, holding the preceding annotations until a newer result updates them.
Display FPS is independent of LLM and InsightFace sampling. Face and safety-event
boxes are drawn independently, so event targets do not require a face or subject
association. Between analysis completions, original video pixels continue moving
while the preceding boxes and labels are held. A late event can also
update the final paused frame. Result-triggered repaints are coalesced and paced
at the display rate after source frames pause; resumed source frames cancel a
pending repaint. Exact analysis JPEGs remain on the evidence feed.

Display event state is keyed by session and event type. Ten consecutive valid
negative source frames clear both the behavior alert and its retained boxes;
each positive updates the boxes and resets the count. Duplicate frame observations
count once. Missing or failed branches and other event types do not advance the
clear counter. Older positives cannot override newer confirmed clears. Late results still enter the
immediate result stream, and late positives produce a separate event notice.
Current image and latest-result endpoints remain in source-frame order.
Archive identity associations remain independent of these display rules.
Faces missing for over three seconds stop being displayed without clearing safety
events or their boxes. Event-only sessions also hold the preceding annotations without
interpolating toward a future result. Multipart frames report `X-Annotation-Mode`
(`exact`, `held`, or `none`), the annotation's analysis message ID, the original
display frame's sequence and timestamp, and `X-Stream-Delay`. Rendering runs in
its own thread; if it falls behind by over one output frame interval, it skips
to the newest due source frame instead of accumulating extra playback delay.
The configured delay stays unchanged. Status reports `publish_lag_seconds`
for time spent beyond that configured delay, including rendering. The
`/v1/video/live/*` endpoints remain the unannotated, near-live diagnostic feed.

See the [display validation record](../validation/display-cadence-20260910.md)
for measured cadence and regression coverage, and the
[API contract](../../API_DELIVERY.md#视频与精确证据帧) for receiver integration.

## Optional rich video labels

Rich person/object labels require a usable GPU on their worker. Without one,
rich annotation is disabled; event analysis and face recognition continue.
The harness samples source JPEGs to an isolated PaddleX/PaddleDetection
worker, keeps only the newest pending request, and uses its tracked boxes as
keyframes for the delayed display compositor. The normal safety VLM still sees
the original frame and its event boxes are drawn as the alert layer. A missing,
slow, or failed rich-label worker therefore drops only the decorative labels;
it does not block VLM analysis or the raw video path.

`RICH_LABEL_ENABLED=true` requests rich annotation. The shared tracking client
requires the worker's `/health/tracking` to report both `ready=true` and
`gpu_ready=true`; `/health/ready` retains the full rich-label readiness contract.
The worker checks GPU execution before loading its models;
CPU mode stays disabled and cannot accept inference requests. Missing capability
information or an unreachable worker also leaves rich annotation disabled.
The Harness retries readiness while the requested worker warms up and enables
tracking automatically when the worker becomes ready. `/v1/status` reports
`disabled_reason=gpu_required` while GPU readiness is unconfirmed. No CPU
fallback is used. Changes to the configured worker endpoint still require a
Harness restart.

For the customer's RTX 5090, use the separate CUDA 12.9 overlay. It selects
`PP-YOLOE_plus-L` as the provisional
higher-accuracy default. Build it, run the real CUDA/model smoke check, and only
then start the services:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml --profile rich build rich-label-worker
docker compose -f compose.yaml -f compose.gpu.yaml --profile rich run --rm \
  rich-label-worker python gpu_smoke.py
docker compose -f compose.yaml -f compose.gpu.yaml --profile rich up -d
```

The principal settings are:

| Variable | Meaning |
| --- | --- |
| `RICH_LABEL_ENABLED` | Requests the optional worker; usable GPU readiness is also required. Default: `false`. |
| `RICH_LABEL_ENDPOINT` | Worker inference URL. Compose default: `http://rich-label-worker:8091/v1/labels`. |
| `RICH_DEVICE` | GPU device requested by the worker. The GPU overlay reserves `gpu:0`; `cpu` leaves rich annotation disabled. |
| `RICH_DETECTOR_MODEL` | PaddleX detector name used by the CPU image. Default: `PicoDet-S`. |
| `RICH_GPU_DETECTOR_MODEL` | Detector used by the GPU overlays. Provisional RTX 5090 default: `PP-YOLOE_plus-L`. |
| `RICH_ATTRIBUTE_MODEL` | PaddleX pedestrian-attribute model name. Default: `PP-LCNet_x1_0_pedestrian_attribute`. |
| `RICH_DETECTION_THRESHOLD` | Minimum object detector score. Default: `0.45`. |
| `RICH_TRACK_VISIBLE_IDLE_FRAMES` | Hold a track through this many detector misses to prevent one-frame bbox flicker. Default: `1`; use `0` to disable. |
| `RICH_ATTRIBUTE_INTERVAL_FRAMES` | Re-run pedestrian attributes every N tracked frames. Default: `10`. |
| `RICH_ATTRIBUTE_RETRY_SECONDS` | Backoff after optional pedestrian attributes fail to load. Default: `60`. |
| `RICH_ATTRIBUTE_SHUTDOWN_TIMEOUT_SECONDS` | Bounded drain time for an in-flight attribute task. Default: `5`. |
| `RICH_DETECTOR_MODEL_DIR` / `RICH_ATTRIBUTE_MODEL_DIR` | Optional paths to pre-provisioned, authorized model directories. |

`rich_labeling.max_fps`, timeout, history size, interpolation gap, and base
tail hold remain in `config.yaml`. Display holding also accounts for recent
successful inference latency and completion intervals, with an adaptive ceiling
of 15 seconds (or the configured base hold if it is larger). This keeps
labels visible between successful results. Failed attempts do not extend the
window; an empty detector snapshot clears labels immediately. Raw snapshots
used for identity matching retain their original timestamps and age limits.
The effective window is exposed as `rich_labeling.display_hold_seconds` in
`/v1/status`. Set the inference request timeout long enough for the actual
worker cadence; a timeout shorter than inference leaves abandoned work and
causes subsequent busy responses. The worker's first model-backed start can be
slow while PaddleX
downloads model files into the persistent `rich-models` volume; allow outbound
HTTPS, disk space, and readiness headroom, or pre-provision both model
directories. `Dockerfile.gpu` is an
independent Paddle 3.3.0 CUDA 12.9 image: that is Paddle's current supported
binary route for the RTX 5090's Blackwell `sm_120` architecture (see Paddle's
[GPU architecture table](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html)). It deliberately
uses ordinary Paddle GPU FP32 first, without TensorRT/high-performance mode, to
keep the initial customer acceptance path small. The Linux host needs NVIDIA
driver 570.26 or newer for this CUDA 12.9/cuDNN 9.9 Blackwell stack, plus the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
use the current production driver when possible.

The GPU image can be built on a host without an NVIDIA GPU; its build checks the
installed versions and verifies that the wheel metadata contains architecture
`120`. The NVIDIA runtime supplies `libcuda.so.1` only when the container is
started with GPU access, so `gpu_smoke.py` intentionally performs the real
Paddle import there. It fails unless a GPU is visible and then runs a CUDA
tensor, Paddle's own check, the detector, and the attribute model. Actual
5090 latency, VRAM, cold starts, and restart stability must pass on the customer
host before enabling `RICH_LABEL_ENABLED`.

### Optional GPU face recognition

The base Harness image intentionally keeps InsightFace on CPU. Merely deploying
that image to an RTX 5090 does not move face detection to the GPU. Use the
separate face overlay to install the CUDA ONNX Runtime build, expose one NVIDIA
device to the Harness container, and select `CUDAExecutionProvider`:

```bash
docker compose -f compose.yaml -f compose.face-gpu.yaml build harness
docker compose -f compose.yaml -f compose.face-gpu.yaml run --rm \
  -e FACE_ALLOW_MODEL_DOWNLOAD=true \
  harness python tools/face_gpu_smoke.py
docker compose -f compose.yaml -f compose.face-gpu.yaml up -d
```

Use `FACE_ALLOW_MODEL_DOWNLOAD=true` only for the licensed non-commercial demo,
or pre-provision authorized weights in the `face-models` volume and omit it.
The smoke command loads the real detection and recognition sessions, executes a
warm inference, and fails if either session silently falls back to CPU.

`Dockerfile.face-gpu` pins ONNX Runtime GPU 1.26, whose official wheel uses
CUDA 12.8 and cuDNN 9. Starting with ONNX Runtime 1.27, the default PyPI GPU
wheel uses CUDA 13, so the pin is deliberate; see the official
[CUDA execution-provider compatibility table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
The image installs the CUDA/cuDNN user-space libraries through the wheel extras;
the host still needs a current Blackwell-capable NVIDIA driver and NVIDIA
Container Toolkit.

To run both accelerated paths on one 5090, combine the independent overlays:

```bash
docker compose -f compose.yaml -f compose.face-gpu.yaml -f compose.gpu.yaml \
  --profile rich up -d
```

The two containers share the device rather than receiving exclusive GPUs.
Benchmark face-only and face-plus-rich loads on the customer host before
acceptance. `FACE_LIVE_SCHEDULER=auto` keeps CPU InsightFace single-flight, but
uses full sampling for DirectML or CUDA: every admitted mixed-detector request
runs its own face branch while the remote safety branches retain their
configured concurrency. Operators can force `full` for an experiment or
`single_flight` as a bounded-backlog fallback.

To benchmark the real HTTP worker with a local video, expose a separate test
instance on loopback in one terminal:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml --profile rich run --rm \
  -p 127.0.0.1:8091:8091 rich-label-worker
```

Then run the reproducible benchmark from another terminal:

```bash
python tools/rich_label_benchmark.py input.mp4 \
  --output-dir tmp/rich-benchmark \
  --sample-fps 7.5 --timeout-seconds 0.8
```

It writes aggregate latency/success/track metrics to `metrics.json`, one record
per source frame to `frames.jsonl`, and the full annotated result to
`annotated.mp4`. The first request includes model loading (and may download
weights); run a short warm-up first when measuring steady-state throughput.

Treat rich metadata as display hints. `P...` and `O...` values are tracking IDs
within one stream session, can change after an occlusion or restart, and are
not a person's identity. Age is an estimated range, and gender is an appearance
model estimate; neither should be presented as a verified personal fact. Real
known-person matching remains the separate, consent-controlled face gallery.

The bundled pedestrian-attribute weights expose only three age bands: `<18`,
`18-60`, and `>60`. The worker requests the complete 26-score output and applies
the PP-Human grouped decoder (including the negative `Female` branch for
male-presenting results) instead of treating PaddleX's positive-label list as a
complete classification. The HUD shows the age estimate and a compact selection
of direction, clothing, bag, and accessory hints. Gender is retained in the
worker's raw attribute data but is not displayed in the overlay.

### Experimental VLM rich annotation

`config.vlm-rich.yaml` enables the `vlm` rich-label backend on this experimental
branch. The current experiment uses Qwen 3.6 Flash through OpenRouter/Alibaba
for annotations, independently of the event model and its DeepInfra route.
The VLM returns person/object boxes and
the same appearance-attribute vocabulary as Paddle through the existing
rich-label snapshot interface, coordinator, box fusion, and renderer. The VLM
assigns local spatial track IDs. The safety-event request is separate, but both
results use the same source frame timeline and final box compositor. Matched
alerts attach to the rich box; a fusion error falls back to events on the
original frame instead of drawing two independent box layers.

The default experiment uses one request per sampled rich-label frame and a
strict compact response schema with all 13 person attributes (unknown values
are null). Both single and parallel modes reuse their HTTP connection pool,
report first-content-token timing and token/cache usage to response observers,
and apply a deadline to frame preparation and inference without retries.
`rich_labeling.vlm_wire_format: compact` requests shorter JSON rows and expands
them locally into the existing annotation protocol. With that format,
`rich_labeling.vlm_execution_mode: parallel` enables three simultaneous requests:
two overlapping person crops and one full-frame object request. Crop coordinates
are mapped back locally. Opposite-crop duplicates are matched one-to-one by
IoU, while unmatched people from either crop are retained. All three responses
must validate before a rich snapshot is published. This mode reuses its
connection pool and applies one timeout to
the entire frame; it makes three paid API requests per frame, with no retry.
`vlm_parallel_strategy: full_frame` instead shares one full image among all
three calls and divides person output by horizontal regions. It keeps full-image
coordinates, but still requires visual recall and duplicate checks; it is not the
default. The September 8 tests found fast but incomplete results in both modes.
`rich_labeling.vlm_structured_output: true` additionally requests a strict JSON
schema for compact responses. Each person must have one complete box paired
with one attribute object. The selected provider must support this schema;
local coordinate and attribute validation still applies to every response.
`vlm_require_complete: true` also checks all 13 fields, counts, types, and scopes
locally before publishing. It is enabled in the experiment, independently of
upstream schema support. Unknown fields stay null until public conversion.

TLS trust loading uses the certifi bundle while preserving CA and hostname
validation. `vlm_keepalive_expiry_seconds` defaults to 60 and `vlm_trust_env` to
true. `vlm_http2: true` is available with the `vlm` extra and is off by default.
Observers distinguish TCP/TLS, content generation, DONE and EOF; transport
tracing is opt-in through the benchmark runner.

`rich_labeling.vlm_prompt_cache: true` puts fixed text in a separate message
before the changing frame. Caching is off by default because it changed recall
in testing. Inspect actual `cached_tokens`; support and minimum prefix size
depend on the upstream provider.

The experimental configuration retains an 8-second request deadline and a
10-second display buffer. These bound waiting and align video, not improve
inference. The [VLM evaluation report](../validation/vlm-evaluation.md) records
the September 6–10 latency, recall, caching and model comparisons. Those tests
did not establish accurate, reliable full-frame annotation within the target;
provider availability and later source changes need their own verification.

Use `tools/vlm_speed_matrix.py` for VLM measurements with an explicit local
`--cases` JSON file and authorized input images. It compares model/provider
parameters, HTTP versions, streaming versus JSON responses, TLS loading, image
size, and parallel strategies. The tool also offers a diagnostic named-field
wire format. `protocol_valid` is not visual accuracy; failures stay in the
denominator and missing usage is not reported as zero cost.

The older native/latency/packed probes and fixed September comparison cases
have been removed. The dated reports retain their measured outcomes; use the
pre-cleanup main revision `a7e5ec8` to reproduce those exact experiments.

Coordinator status reports `last_queue_seconds`,
`last_latency_seconds` (inference), and `last_total_latency_seconds` (submission
through inference completion); none includes the configured display buffer.
`last_attempt_frame_sequence` and `last_attempt_succeeded` identify the attempt
these timings describe, while `last_frame_sequence` remains the last success.

PaddleX and PaddleDetection source code is Apache-2.0, but that does not by
itself grant rights to every pretrained weight or its training data. Before
customer or commercial use, confirm the license and redistribution terms for
the exact detector and attribute weights being shipped.
