# Goodvision Harness

An independent demo service that pulls the robot's live WHEP/WebRTC stream and
detects any selected subset of `fight`, `smoking`, `fall`, `weapon`, and
`stranger` (all five remain the default). It exposes one structured result and one exactly correlated,
annotated frame for every analyzed video frame, plus a continuous delayed
annotation stream for display.

The harness has no runtime dependency on the original `goodvision`
repository.

## Harness behavior

The service performs one external model request for whichever visual safety
events are selected, while the local known-person matcher checks the same frame
only when `stranger` is selected. Mixed results are merged in canonical order.
It does not run one process or one LLM call per feature. Omitting `features`
keeps the original fixed-five result. Before a real multi-event
or face-only session accepts its first frame, the service loads the local engine and
runs one blank inference to remove its first-frame cold start; dry-run skips
that warmup. The default runtime settings are:

- source/face sampling: 2.5 FPS, 1280 px maximum image side, JPEG quality 90;
- the same frame is resized to 320 px, JPEG quality 75, for the single remote
  safety request;
- display output: 15 FPS, 720 px maximum image side, JPEG quality 70, delayed
  2.5 seconds so most completed 2.5 FPS analysis keyframes can be interpolated;
- one-frame windows with stride 1 and six concurrent model requests;
- `max_tokens=1200`, no API retry, and one explicitly selected model route
  with automatic route fallback disabled;
- one multi-label safety prompt that independently evaluates `fight`,
  `smoking`, `fall`, and `weapon` and permits simultaneous positives.
  Minimum visual-evidence rules for `fall`/`smoking` override the general
  high-recall preference: stable supported crouching and empty-hand-to-mouth
  gestures are not sufficient. These prompt-only changes keep the original
  one-frame, 320 px safety input and do not change Paddle or face recognition.
  They may also miss subtle falls or small/occluded smoking items; validate
  both false positives and missed positives on fixed test recordings.

The original headless CLI keeps its prior event and result protocol. Browser
testing now uses one unified page, based on the familiar original Demo UI.
Both browser-camera and network-stream modes use the same grounded delivery
pipeline and expose customer JSON plus analyzed and raw video outputs. The
legacy single-event and face-only modes remain available as diagnostic API
modes; the page lets the operator select any non-empty feature combination.

## Optional rich video labels

Rich person/object labels require a usable GPU on their worker. Without one,
rich annotation is disabled; event analysis and face recognition continue.
The harness samples source JPEGs to an isolated PaddleX/PaddleDetection
worker, keeps only the newest pending request, and uses its tracked boxes as
keyframes for the delayed display compositor. The normal safety VLM still sees
the original frame and its event boxes are drawn as the alert layer. A missing,
slow, or failed rich-label worker therefore drops only the decorative labels;
it does not block VLM analysis or the raw video path.

`RICH_LABEL_ENABLED=true` requests rich annotation, but the Harness enables it
only after the worker's `/health/ready` reports both `ready=true` and
`gpu_ready=true`. The worker checks GPU execution before loading its models;
CPU mode stays disabled and cannot accept inference requests. Missing capability
information or an unreachable worker also leaves rich annotation disabled.
Start the GPU worker before the Harness; restart the Harness after changing the
worker's GPU availability. `/v1/status` reports `disabled_reason=gpu_required`
when GPU readiness was not confirmed. No CPU fallback is used.

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
docker compose --profile rich run --rm \
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
complete classification. The HUD shows age and gender estimates first, followed
by a compact selection of direction, clothing, bag, and accessory hints.

PaddleX and PaddleDetection source code is Apache-2.0, but that does not by
itself grant rights to every pretrained weight or its training data. Before
customer or commercial use, confirm the license and redistribution terms for
the exact detector and attribute weights being shipped.

## LAN deployment topology

```text
robot WHEP server on 192.168.20.199
      |
      | TCP 1985 signaling + UDP 8000 media
      v
goodvision-harness container on 192.168.20.168
      |                         |
      | HTTPS 443               | localhost:8080
      v                         v
configured model API      JSON + delayed/exact/raw video
                                |
                                v
                       LAN receiver
```

The receiver does **not** need a webhook for this demo. It can read the latest
JSON, subscribe to the JSON SSE stream, and consume the MJPEG stream directly.
Webhook delivery is an optional adapter when the receiver specifically wants
the harness to push each result.

The default host bind (`127.0.0.1:8080`) is for a receiver or reverse proxy
running on `192.168.20.168`. Compose also creates the stable `goodvision-demo`
Docker network. A receiver container can join that network and call
`http://goodvision-harness:8080`; `127.0.0.1` inside the receiver container
would refer to the receiver itself.

## Server prerequisites

The default deployment needs:

- a 64-bit Linux host with Docker Engine 20.10+ and Docker Compose v2;
- 4 CPU cores and 8 GB RAM recommended for a comfortable demo
  (2 cores and 4 GB is the practical minimum for one stream);
- about 10 GB free disk for the image, build cache, and operating headroom;
- outbound HTTPS/DNS access to the configured LLM endpoint on TCP 443;
- authorized detection and recognition weights under
  `FACE_MODEL_ROOT/models/<configured-profile>/` for stranger detection;
- routed network access from the container to `192.168.20.199`, including TCP
  1985 for WHEP signaling and UDP 8000 for WebRTC media;
- local TCP 8080 for the receiver or reverse proxy;
- correct system time and CA certificates for HTTPS.

A GPU is not required: video decoding, overlays, and the default local face
matcher can run on CPU, while the four safety-event judgments use the public
LLM API. On an NVIDIA host, `compose.face-gpu.yaml` can move the local face
matcher to CUDA. The host does not need inbound Internet access to port 8080
when the receiver is also on the host.

## Deploy

For a prebuilt image-tar delivery with no source tree on the customer server,
follow [DEPLOY.md](DEPLOY.md) and run the supplied `start_delivery.py` for the
complete package. Both current images include their GPU components; the launcher
enables the rich worker only when a usable GPU is available. The source-tree
equivalent is `python tools/start_delivery.py`.
[API_DELIVERY.md](API_DELIVERY.md) is the public interface contract.
The regular source-tree `compose.yaml` remains the developer build path;
the older `compose.delivery.yaml` and delivery overlays remain available for
their historical CPU/separate-image releases and must not be mixed into the
current full package.

On the GoodVision server (`192.168.20.168`):

```bash
cp .env.example .env
# The robot WHEP URL is configured; set the selected inference route's API key.
docker compose up --build -d
docker compose ps
```

For normal fixed-five mode, the demo configuration may initialize authorized
local face files in the Compose `face-models` volume on first use. Alternatively,
pre-populate `/models/models/<configured-profile>/`. `HARNESS_DRY_RUN=true` does
not need them.

The development default model route uses AIClub with `qwen3.5-plus`. Set
`INFERENCE_PROVIDER` to switch to OpenRouter or DashScope; model and endpoint
overrides use the same streaming pipeline. See the configuration examples below.
The full customer GPU package instead retains its fixed service preset.

The default source is the robot's LAN WHEP endpoint:

```text
webrtc+http://192.168.20.199:1985/rtc/v1/whep/?app=live&stream=livestream
```

`webrtc+http://` is the harness source marker; the underlying WHEP signaling
request is normal HTTP. Override the source with `HARNESS_STREAM_URL`. The old
`RTSP_URL` variable remains a lower-priority compatibility fallback. On the
robot, SRS must advertise `192.168.20.199` as its ICE candidate and allow TCP
1985 plus UDP 8000 from the GoodVision host/container.

The HTTP service binds to `127.0.0.1:8080` on the host by default:

```bash
curl http://127.0.0.1:8080/health/live
curl http://127.0.0.1:8080/health/ready
curl http://127.0.0.1:8080/v1/status
```

`/health/live` only proves the HTTP process is alive. `/health/ready` returns
200 only after both a fresh source frame and a fresh successful analysis have
arrived; invalid API credentials or repeated model failures therefore keep the
application unready. The Docker healthcheck intentionally uses `/health/live`
so a robot/network outage is visible without putting the container into a
restart loop; monitor `/health/ready` separately for end-to-end readiness.

Open `http://127.0.0.1:8080/` for the single internal test page. It has Camera /
Network stream controls, five separate event status cards, the continuous
delayed annotation stream, and the complete JSON result together. A link under
the player opens the exact 2.5 FPS evidence stream for debugging and audit.
Leave the network URL blank to use `HARNESS_STREAM_URL`, or enter a different
RTSP/RTMP/WHEP URL for that run.

Event and stranger analysis run concurrently on the same admitted sample when
the face worker is available. In CPU single-flight mode, samples arriving while
face inference is busy still receive event analysis, with no stranger judgment
for that sample. The live classification cards retain each feature's latest
valid result; the per-frame log marks incomplete coverage as `PARTIAL` and lists
the features not analyzed. Only complete negative coverage is logged as `CLEAR`.

Runtime errors track event and face branches independently. A newer successful
result clears only that branch's error; an older response or a skipped branch
cannot claim recovery. The console distinguishes timeout, connection, service,
empty-response and invalid-response errors. Server warnings retain the branch,
window index, fixed failure code and HTTP status without storing provider text,
credentials or camera images.

A browser can use its local camera from `localhost`. When the page is opened
remotely from `192.168.1.50`, browsers normally require HTTPS/WSS before they
allow camera access. Use an HTTPS reverse proxy or an SSH tunnel to
`localhost:8080`. Network-stream mode does not depend on browser camera
permission. Restrict this unauthenticated internal page with the server
firewall or reverse proxy.

### Customer-server operations

The image default is
`webrtc+http://192.168.20.199:1985/rtc/v1/whep/?app=live&stream=livestream`.
The **Persistent deployment defaults** panel can override that value and save
the formal/dry-run mode,
autostart choice, and a write-only API key in the persistent `/data` volume.
Stop the active session before saving. Saved values become effective for the
next session and remain authoritative after container replacement or restart;
the API key and complete stream URL are never returned by the settings API.

Open the UI through `http://127.0.0.1:8080/` on the server, or through the SSH
tunnel / HTTPS reverse proxy described above. To change a running source, press
**Stop**, select **Network stream**, and either leave the URL blank to use the
deployed `HARNESS_STREAM_URL` or enter an RTSP/RTMP/WHEP URL for a temporary
test. The server does not write a URL entered in the UI to disk or use it for the next
container start; reloading the page also clears the field. Put the real
persistent source in `.env` as `HARNESS_STREAM_URL`, then run
`docker compose up -d` to recreate the service with that value. A value already
saved in `/data/deployment-settings.json` remains authoritative; update it from
the console after stopping the active session if the old deployment saved an
RTSP source.

Use these checks after the UI reports that the session is running:

```bash
# Display-oriented delayed stream with smooth, full-cadence boxes.
ffplay http://127.0.0.1:8080/v1/video/annotated/stream.mjpeg

# Exact analyzed evidence frames; X-Analysis-Message-Id matches JSON message_id.
ffplay http://127.0.0.1:8080/v1/video/stream.mjpeg

# Latest JSON and the live Server-Sent Events feed.
curl http://127.0.0.1:8080/v1/results/latest
curl -N http://127.0.0.1:8080/v1/results/events
```

These are HTTP MJPEG/JSON/SSE outputs, not a new RTSP publication. If an
upstream proxy is used, disable response buffering for MJPEG and SSE, allow
long-lived responses, and pass WebSocket upgrades for `/ws`.

Stop the active detection first, then manage the known-person gallery from
**Known people gallery** in the same UI: enroll one clear, single-face image
with a stable person ID and name, inspect the sample count, and use **Delete**
to remove all samples for a person. For API or headless troubleshooting:

```bash
curl http://127.0.0.1:8080/v1/faces/gallery
curl --fail-with-body -X POST --data-binary @person.jpg \
  -H 'Content-Type: image/jpeg' \
  'http://127.0.0.1:8080/v1/faces/gallery/person_01?name=Person%20One'
curl --fail-with-body -X DELETE \
  http://127.0.0.1:8080/v1/faces/gallery/person_01
# List continuous events, then use a returned evidence.available.url.
curl http://127.0.0.1:8080/v1/events
curl --output event.mp4 http://127.0.0.1:8080/v1/events/EVENT_ID/video
docker compose logs --tail=200 harness
docker volume inspect goodvision-harness_face-models goodvision-harness_face-data
docker compose exec harness sh -lc \
  'id; ls -la /models/models /data; test -w /models; test -w /data'
```

In Compose, authorized face weights belong under
`/models/models/<configured-profile>/` in the persistent `face-models`
volume, while the gallery is `/data/face-gallery.json` in the persistent
`face-data` volume. Continuous event metadata and short MP4 evidence live in
`/data/event-evidence/`, with 7-day/10-GiB retention. Stranger crop and persistent
stranger-embedding capture have been removed. See [event evidence](EVENT_EVIDENCE.md)
for APIs, identity continuity, revisions and legacy-file cleanup. `docker compose down` preserves both volumes;
`docker compose down -v` deletes them. Back up the gallery volume before
destructive maintenance. Optional local-file initialization is for the declared
demo only. A customer deployment
must use face-model weights whose license permits that production use.

### Zero-cost smoke mode

To test WHEP ingestion, JSON delivery, and all video outputs without calling the LLM:

```dotenv
HARNESS_DRY_RUN=true
```

Restart with `docker compose up -d`. Dry-run bypasses both the external model
and local face weights, and emits deterministic selected-feature results with
synthetic normalized boxes.

## Selecting detection features

Use one session with a `features` array rather than separate processes or one
endpoint per combination. The allowed values are `fight`, `smoking`, `fall`,
`weapon`, and `stranger`. The array must be non-empty; duplicates and unknown
values are rejected. Results contain only enabled judgments, always in the
canonical order above. Omit `features` to retain the original all-five behavior.

For a configured network stream:

```bash
# Stranger + smoking
curl --fail-with-body -X POST http://127.0.0.1:8080/v1/sessions/start \
  -H 'Content-Type: application/json' \
  -d '{"mode":"stream","features":["stranger","smoking"]}'

curl --fail-with-body -X POST http://127.0.0.1:8080/v1/sessions/stop
```

Another valid request is
`{"mode":"stream","features":["fight","fall","smoking"]}`. Browser-camera
clients use the same field in the existing `/ws` start message:
`{"type":"start","mode":"camera","analysis_mode":"multi","features":["stranger"]}`.
Only one input session runs at a time; selected safety labels share one model
request, and `stranger` uses the local face matcher on the same frame.

The combined output remains available at `/v1/results/latest` and
`/v1/results/events`. Each feature also has an independent pull URL that returns
only that judgment and the objects it references:

| Feature | Latest result | SSE result stream |
| --- | --- | --- |
| Fight | `GET /v1/results/fight/latest` | `GET /v1/results/fight/events` |
| Smoking | `GET /v1/results/smoking/latest` | `GET /v1/results/smoking/events` |
| Fall | `GET /v1/results/fall/latest` | `GET /v1/results/fall/events` |
| Weapon | `GET /v1/results/weapon/latest` | `GET /v1/results/weapon/events` |
| Stranger | `GET /v1/results/stranger/latest` | `GET /v1/results/stranger/events` |

For example, `curl http://127.0.0.1:8080/v1/results/smoking/latest`
never includes fight, fall, weapon, or stranger judgments. It returns 404 when
the latest analysis has no smoking judgment (for example, when smoking was not
enabled for that session).

## Output endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health/live` | HTTP-process liveness. |
| `GET /health/ready` | Ready only with fresh source video and a fresh successful analysis. Used by the Docker healthcheck. |
| `GET /v1/status` | Sanitized source, pipeline, counters, and delivery state. |
| `POST /v1/sessions/start` | Start one network-stream session; JSON accepts the optional `features` array. Browser cameras start over `/ws`. |
| `POST /v1/sessions/stop` | Stop the active session. |
| `GET /v1/events` | List persistent continuous events and available video evidence. |
| `GET /v1/events/{event_id}/video` | Play available MP4 evidence with HTTP Range support. |
| `GET /v1/results/latest` | Latest customer-facing analysis JSON. Returns 404 before the first result. |
| `GET /v1/results/events` | Server-Sent Events stream of analysis JSON with bounded in-memory replay. |
| `GET /v1/results/<feature>/latest` | Independent latest result for one of the five feature names. |
| `GET /v1/results/<feature>/events` | Independent SSE result stream for one feature. |
| `GET /v1/video/annotated/latest.jpg` | Latest full-cadence delayed frame with exact, interpolated, or briefly held bbox. |
| `GET /v1/video/annotated/stream.mjpeg` | Continuous delayed annotated MJPEG, 15 FPS by default. This is the display output. |
| `GET /v1/video/latest.jpg` | Latest exact analyzed frame with burned-in verdict/bbox. |
| `GET /v1/video/stream.mjpeg` | Annotated MJPEG at result cadence: one exact analyzed frame per JSON result. |
| `GET /v1/video/live/latest.jpg` | Latest unannotated source-preview frame. |
| `GET /v1/video/live/stream.mjpeg` | Continuous unannotated source preview. |
| `GET /` | Single Camera / Network stream internal test page. |

Examples:

```bash
curl http://127.0.0.1:8080/v1/results/latest
curl -N http://127.0.0.1:8080/v1/results/events
ffplay http://127.0.0.1:8080/v1/video/annotated/stream.mjpeg
ffplay http://127.0.0.1:8080/v1/video/stream.mjpeg
ffplay http://127.0.0.1:8080/v1/video/live/stream.mjpeg
```

## Analysis result contract

The canonical schema is
[`schemas/analysis-result-v2.schema.json`](schemas/analysis-result-v2.schema.json).
A result has the following shape:

```json
{
  "type": "analysis_result",
  "schema_version": "2",
  "analysis_kind": "fixed_five_events",
  "message_id": "cb44b47c-332f-4947-971b-8297fc8c64db",
  "observation_id": "cb44b47c-332f-4947-971b-8297fc8c64db",
  "source_id": "external-camera",
  "stream_session_id": "7da2e5db-6fcc-45fa-ad04-99ec62f50e87",
  "sequence": 18,
  "emitted_at": "2026-08-25T11:42:08.321Z",
  "frame": {
    "source_frame_index": 2417,
    "timestamp_seconds": 80.567,
    "observed_at": "2026-08-25T11:42:06.978Z",
    "width": 1920,
    "height": 1080,
    "clock_source": "ingest_estimated"
  },
  "description": "smoking: 一人将电子烟举到嘴边。 / A person holds an e-cigarette at the mouth.; stranger: 一张人脸与已知人员库不匹配。 / One face does not match the known-person gallery.",
  "analysis_complete": true,
  "missing_event_labels": [],
  "judgments": [
    {
      "event_label": "fight",
      "detected": false,
      "confidence": 0.91,
      "description": "未发现打架迹象。 / No fighting cues are visible.",
      "object_ids": []
    },
    {
      "event_label": "smoking",
      "detected": true,
      "confidence": 0.94,
      "description": "一人将电子烟举到嘴边。 / A person holds an e-cigarette at the mouth.",
      "object_ids": ["object-1", "object-2"]
    },
    {
      "event_label": "fall",
      "detected": false,
      "confidence": 0.88,
      "description": "未发现跌倒。 / No fall is visible.",
      "object_ids": []
    },
    {
      "event_label": "weapon",
      "detected": false,
      "confidence": 0.86,
      "description": "未发现武器迹象。 / No weapon cue is visible.",
      "object_ids": []
    },
    {
      "event_label": "stranger",
      "detected": true,
      "confidence": 0.93,
      "description": "一张人脸与已知人员库不匹配。 / One face does not match the known-person gallery.",
      "object_ids": ["object-3"]
    }
  ],
  "objects": [
    {
      "object_id": "object-1",
      "label": "person",
      "confidence": 0.91,
      "bbox": {
        "coordinate_system": "source_normalized_xyxy",
        "x1": 0.183,
        "y1": 0.112,
        "x2": 0.704,
        "y2": 0.902
      },
      "attributes": {}
    },
    {
      "object_id": "object-2",
      "label": "e-cigarette",
      "confidence": 0.86,
      "bbox": {
        "coordinate_system": "source_normalized_xyxy",
        "x1": 0.481,
        "y1": 0.294,
        "x2": 0.516,
        "y2": 0.354
      },
      "attributes": {}
    },
    {
      "object_id": "object-3",
      "label": "unknown_person",
      "confidence": 0.93,
      "bbox": {
        "coordinate_system": "source_normalized_xyxy",
        "x1": 0.361,
        "y1": 0.108,
        "x2": 0.491,
        "y2": 0.301
      },
      "attributes": {
        "status": "unknown",
        "threshold": 0.4
      }
    }
  ],
  "model": {
    "name": "numaira-beta-fast",
    "latency_seconds": 1.247
  }
}
```

Every emitted analysis result includes a required, non-empty top-level
`description` for the frame identified by `frame`. It joins detected events as
`event_label: description` in judgment order, separated by `; `. When no judgment
has `detected=true` and every enabled capability completed, it is
`"无异常 / No anomalies detected"`. This is a summary of this result's analyzed
events, generated without an extra model request. Event evidence is requested
as a Simplified Chinese sentence followed by its English equivalent, separated
by ` / `, in the same inference call. Local face and dry-run descriptions use
the same bilingual format. The per-event `judgments[].description` field remains
present, including negative evidence; JSON keys and event labels are unchanged.
Feature-specific JSON and SSE endpoints preserve the full-frame `description`
even when they filter `judgments` and `objects` to one capability.

Results use schema version `2`; the original v1 schema remains unchanged for
archived records. Strict consumers must adopt the v2 schema before upgrading
the producer. The HTTP endpoint paths remain `/v1/...`.
`analysis_complete` is false and `missing_event_labels` lists skipped or failed
capabilities when a frame is only partially analyzed. Such descriptions state
`Analysis incomplete` and retain any positive evidence, never reporting all-clear.
Live model responses must pass Chinese / English format validation; a malformed
or monolingual description fails that inference branch. This validates language
structure, not semantic translation equivalence.

The console shows this text in **Overall description · 全画面描述** below the video,
with the latest analyzed frame's index and timestamp. The card is also visible
in fullscreen, ignores older results, and resets when the session changes or
stops. Its text follows the latest analysis result; the annotated video retains
its configured display delay.

In `fixed_five_events` mode, `judgments` always has exactly these labels in
this order: `fight`, `smoking`, `fall`, `weapon`, `stranger`. Each status is
independent, so any number may be true in the same result. Negative judgments
use an empty `object_ids` array; every positive judgment references the boxes
that support that event. The `stranger` status comes from a mismatch against
the local known-person gallery, not merely from seeing a person.

`bbox` is top-left-origin `xyxy`, normalized to the source frame. Convert
it to pixels with `x * frame.width` and `y * frame.height`.
`timestamp_seconds` is monotonic within one stream session;
`observed_at` is the harness ingest-time estimate because ordinary RTSP
decoding does not guarantee access to the camera's absolute capture clock.

For every successful JSON result, the harness still annotates the exact newest
frame that the result references. `GET /v1/video/latest.jpg` and every
multipart JPEG in `GET /v1/video/stream.mjpeg` carry
`X-Analysis-Message-Id`; its value is the same as the JSON `message_id`. This
exact evidence feed remains at model-result cadence.

For the display path, the container separately buffers the continuous source
frames for 2.5 seconds. Results at 2.5 FPS act as bbox keyframes on the same
media timeline. Between two compatible keyframes, the service linearly moves
the box. If the next result is late or a sampled request is dropped, the last
analyzed state is latched onto every display frame; only a later keyframe may
update or explicitly clear it. This prevents a continuously detected bbox from
blinking between model samples. The original video frames are not synthesized
or skipped, so `/v1/video/annotated/stream.mjpeg` remains smooth while staying
intentionally behind the camera. Multipart frames report `X-Annotation-Mode`
(`exact`, `interpolated`, `held`, or `none`), the left and optional next
analysis IDs, and `X-Stream-Delay`. The
`/v1/video/live/*` endpoints remain the unannotated, near-live diagnostic feed.

## Optional webhook

Set a same-host receiver through the Docker host alias:

```dotenv
RESULT_WEBHOOK_URL=http://host.docker.internal:9000/events
HARNESS_WEBHOOK_SECRET=replace-with-a-shared-secret
```

Each POST uses `message_id` as `Idempotency-Key`. When a secret is set,
`X-Harness-Signature-256` contains an HMAC-SHA256 signature over the exact
request body. Delivery has a bounded in-memory queue and finite retries; it is
not durable across container restarts. Use SSE plus receiver-side persistence
when replay guarantees matter.

## Configuration

`config.yaml` controls sampling, model selection, the legacy single-event
prompt, the fixed-four `multi_event_prompt`, concurrency, readiness windows,
the `display.delay_seconds` / `display.bbox_max_gap_seconds` compositor timing,
overlay JPEG quality, and optional webhook behavior. Inference routes are
configured under `inference_service`. The API retry count remains zero so a
failed live request is not replayed as a stale result.
Persisted console settings override environment values; otherwise the generic
stream variable takes precedence over the legacy RTSP variable:

| Variable | Meaning |
| --- | --- |
| `HARNESS_STREAM_URL` | Primary RTSP/RTMP/WHEP input URL. |
| `RTSP_URL` | Deprecated compatibility fallback when `HARNESS_STREAM_URL` is empty. |
| `INFERENCE_PROVIDER` | Route: `aiclub` (default), `openrouter`, or `dashscope`. |
| `INFERENCE_MODEL` | Optional model override for the selected route. Empty uses its configured default. |
| `INFERENCE_ENDPOINT` | Optional complete chat-completions URL override for the selected route. |
| `INFERENCE_API_KEY` | Optional shared credential override, ahead of provider-specific environment keys. |
| `AICLUB_API_KEY` | AIClub credential, used only when that route is selected. |
| `OPENROUTER_API_KEY` | OpenRouter credential, used only when that route is selected. |
| `DASHSCOPE_API_KEY` | DashScope credential, used only when that route is selected. |
| `HARNESS_DRY_RUN` | Disable external inference and emit deterministic results. |
| `HARNESS_AUTOSTART` | Start the configured network stream when the service starts; set false for local camera testing. |
| `HARNESS_SOURCE_ID` | Stable customer-facing source identifier. |
| `RESULT_WEBHOOK_URL` | Optional result-push destination. |
| `HARNESS_WEBHOOK_SECRET` | Optional webhook HMAC secret. |
| `HARNESS_BIND_ADDRESS` | Host bind address used by Compose. |
| `HARNESS_PORT` | Host HTTP port used by Compose. |
| `FACE_MODEL_ROOT` | Root containing operator-supplied face-analysis files. |
| `FACE_GALLERY_PATH` | Persistent known-person gallery JSON path. |
| `FACE_PROVIDER` | InsightFace ONNX provider: CPU, DirectML, or CUDA. |
| `FACE_LIVE_SCHEDULER` | `auto` selects CPU single-flight and accelerated full sampling; `full`/`single_flight` override it. |
| `RICH_LABEL_ENABLED` | Enable the optional rich-label worker path. |
| `RICH_LABEL_ENDPOINT` | Rich-label worker inference URL. |

Do not commit `.env`, RTSP credentials, or LLM API keys.

### 切换 AI 渠道和模型

以下路由选择说明适用于当前源码开发运行及 `development` 镜像。
完整交付包按 [DEPLOY.md](DEPLOY.md) 部署，控制台可选择并保存当前 A–H 八个模型。
本节的任意渠道、模型及地址环境覆盖不适用于该交付包。
已导出的 `.2` 及更早历史镜像不会因源码更新而增加此能力；使用这些历史版本时，
应遵循其随包原说明和配置，不能只通过添加环境变量切换为新实现。

把各渠道的密钥保存在本机未纳入 Git 的 `.env` 中。切换时只需修改
`INFERENCE_PROVIDER`，再重启服务：

```dotenv
INFERENCE_PROVIDER=openrouter
AICLUB_API_KEY=
OPENROUTER_API_KEY=填入该渠道的密钥
DASHSCOPE_API_KEY=
INFERENCE_MODEL=
INFERENCE_ENDPOINT=
INFERENCE_API_KEY=
```

| `INFERENCE_PROVIDER` | 默认模型 | 使用的渠道密钥 |
| --- | --- | --- |
| `aiclub` | `qwen3.5-plus` | `AICLUB_API_KEY` |
| `openrouter` | `qwen/qwen3.5-35b-a3b` | `OPENROUTER_API_KEY` |
| `dashscope` | `qwen3.5-plus` | `DASHSCOPE_API_KEY` |

例如，将 `INFERENCE_PROVIDER=dashscope` 即可改走 DashScope。测试其他模型时，
设置 `INFERENCE_MODEL` 为该渠道支持的模型 ID；留空则恢复该渠道的默认模型。
三个渠道的默认模型并不完全相同。比较线路速度时，应选择各渠道都支持的同一模型，
并保持输入、提示词和请求参数一致。可用模型见
[OpenRouter 模型页](https://openrouter.ai/qwen/qwen3.5-35b-a3b)。
若要使用自定义网关、区域或工作空间地址，可设置完整 URL；DashScope 地址格式见
[官方兼容接口文档](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)：

```dotenv
INFERENCE_ENDPOINT=https://gateway.example/v1/chat/completions
```

切换渠道时，清空不再需要的 `INFERENCE_MODEL`、`INFERENCE_ENDPOINT` 和
`INFERENCE_API_KEY` 覆盖值，避免沿用上一次测试的设置。部署控制台保存的密钥优先于
环境变量，且只用于保存时对应的渠道；使用独立渠道环境密钥时，保持
`INFERENCE_API_KEY` 为空。

源码 Compose 部署修改 `.env` 后执行 `docker compose up -d --force-recreate harness`。
使用已包含该功能的新交付镜像时，选择对应的交付 Compose 文件再重建主服务容器；
不要将仍固定历史镜像标签的启动文件误当成新版本升级入口。本地运行则重启 Harness 进程。

也可以在 `config.yaml` 的 `inference_service.provider` 中选择默认渠道。
各渠道的模型、完整 `endpoint`、`api_key_env` 和专用请求参数放在
`inference_service.providers.<渠道>` 下，例如：

```yaml
inference_service:
  provider: openrouter
  providers:
    openrouter:
      model: qwen/qwen3.5-35b-a3b
      request_parameters:
        reasoning:
          enabled: false
    dashscope:
      model: qwen3.5-plus
      request_parameters:
        enable_thinking: false
```

`INFERENCE_PROVIDER`、`INFERENCE_MODEL` 和 `INFERENCE_ENDPOINT` 环境变量优先于
配置文件。`inference.request_parameters` 保存各渠道通用参数（如温度和输出长度）；
选中渠道的 `request_parameters` 覆盖同名通用参数，其余渠道的专用参数不会发送。

## Local development

```powershell
python -m venv .venv
python -m pip install -e '.[dev,face,webrtc]'
$env:HARNESS_AUTOSTART = 'false'
goodvision-harness-demo --host 127.0.0.1 --port 7900
```

Open `http://127.0.0.1:7900`, enroll the known people, choose Camera, enable it,
then press Start.
For a network test, choose Network stream and either enter a URL or leave it
blank to use `HARNESS_STREAM_URL`. Set `HARNESS_DRY_RUN=true` for a no-cost pipeline
check. `goodvision-harness-demo` is now a compatibility launcher for the same
unified grounded page; it is not a second legacy Demo. `goodvision-harness`
serves that same page on port 8080 by default. `goodvision-harness-cli`
retains the original window lifecycle, `vision_result`, and final-state JSONL
behavior.

For the Windows non-NVIDIA GPU experiment, follow
[`WINDOWS_HYBRID.md`](WINDOWS_HYBRID.md). It runs InsightFace through DirectML
in a native Harness while retaining the complete Paddle worker on Docker CPU.

### Known-person gallery

Install the face extra (combine it with the test tools when developing):

```powershell
python -m pip install -e '.[face,webrtc]'
# or
python -m pip install -e '.[dev,face,webrtc]'
$env:HARNESS_AUTOSTART = 'false'
goodvision-harness-demo --host 127.0.0.1 --port 7900
```

The unified Five Events page keeps this gallery next to the detector controls
and uses it for the `stranger` judgment. Enroll each known person with a stable
person ID, name, and one clear single-face image. The stranger card includes a
face-diagnostics summary for every analyzed frame. The face-only backend mode
remains available for API compatibility, but it is no longer a separate mode
in the test UI. Install compatible detection and recognition weights under
`FACE_MODEL_ROOT/models/<configured-profile>/`, or allow the configured first-use
download. The harness does not bundle models; the known-person gallery persists
at `FACE_GALLERY_PATH`.

`face.min_face_size` is the live-recognition floor in analysis-frame pixels.
Small faces can positively match an enrolled person, but an unmatched face
under 60 pixels, a low-confidence detection, or an extreme face angle is
`unusable`, not evidence of a stranger. New unknown identities still require
three reliable observations. A known identity remains attached to a reliable
person trajectory through side views and ordinary mismatches; one contradictory
frame cannot replace it. Source-time motion and one-to-one associations bridge
brief gaps up to three seconds. Predicted boxes are not new evidence. Ambiguous
crossings, prolonged absence and sustained identity conflicts require reconfirmation.
No stranger photos or persistent stranger embeddings are created.

Each live session keeps its own identity history and stable `subject_id` values.
`person_id` remains the enrolled identity. GPU detector tracking operates independently
of optional attributes; non-GPU event requests may supply compact person locations.
The immediate analysis result never waits for either sidecar or archive work.
See [event evidence and continuity](EVENT_EVIDENCE.md) for exact behavior and APIs.
The [low-resolution validation report](LOW_RESOLUTION_VALIDATION.md) records real
CPU face and public-media model results, actor-output latency, observed misses,
crossing tradeoffs, and separately tested video-evidence behavior.

The harness does not grant rights to third-party face-analysis files. For
customer or commercial use, supply files whose license covers the intended use.

The result envelope is ready for the separate visual Memory layer:
`analysis_kind` distinguishes event and `face_recognition` observations,
`observation_id` is the stable alias of `message_id`, and
`objects[].attributes` carries mode-specific metadata. A Memory implementation
only needs to implement the non-blocking `VisualMemorySink.submit(...)`
interface, which receives a `VisualMemoryRecord` containing the observation
and its exactly correlated evidence JPEG.

### How the Memory pieces fit together

There are three small, deliberately separate responsibilities:

- **FaceGallery** is the primary identity memory for local face analysis
  mode. It persists embeddings and answers known/unknown locally. In this mode,
  `detected=true` means an **unknown** person is present.
- **Reference memory** is an optional remote-model input prefix migrated from
  Goodvision PR #53. It loads up to eight private reference images once,
  sends them before the live frame, and rejects model-reported memory IDs that
  were not supplied. It uses the same stranger-detection contract as
  the local matcher: `detected=true` means an **unknown** live face is present.
  Known identities are reported independently in `matched_memory_ids`.
- **VisualMemorySink** is the existing output seam. It receives any completed
  event or face observation plus its exactly correlated JPEG; it does not
  participate in recognition.

Remote face-reference results use the following matrix. Its `detected` column
matches the local matcher; `matched_memory_ids` is additional reference metadata:

| Live frame | `detected` | `matched_memory_ids` |
| --- | --- | --- |
| No face | `false` | `[]` |
| Known face only | `false` | Known IDs |
| Unknown face only | `true` | `[]` |
| Known and unknown faces | `true` | Known IDs |

The fixed-five mode and the face diagnostics mode both use local face analysis by
default. Enable remote reference memory only as an explicit legacy
fallback/experiment by setting
`memory.enabled: true`, adding reference entries under `memory.references`,
and using `inference.event_label: unknown_person`. The harness also enforces
that label whenever the built-in face-reference purpose is active. Relative
paths resolve from the selected config file. Docker exposes the ignored local
`memory/` directory at `/app/memory`; for local development, pass
`--config config.yaml` to use that same top-level directory. A memory-aware
result reports validated IDs in `judgments[].matched_memory_ids`, and uses
`analysis_kind=face_reference_match` for the built-in face purpose.
Enabling it sends the resized reference images to the configured multimodal
model provider on every inference request; keep it disabled unless that data
flow is allowed.

The old PR's process-local upload store is intentionally not duplicated: the
main face path already has persistent enroll/list/delete APIs and UI through
FaceGallery, while remote fallback references remain deployment configuration.

## Customer release builds

The two GPU Dockerfiles each define `runtime`, `customer`, and `development`
targets. The final `development` target is still the default, retaining the
existing developer entrypoints; CPU Dockerfiles and the local CPU/hybrid demo
path are unchanged. `compose.gpu.yaml` now builds its worker from the repository
root so the explicit customer target can use the shared process gateway.

Customer targets build the current source and dependencies from the public base
image. They do not require a product image from an earlier local delivery. From
the repository root, use distinct candidate tags while validating changes:

```bash
docker build --platform linux/amd64 -f Dockerfile.face-gpu --target customer -t goodvision-harness:consolidation-test .
docker build --platform linux/amd64 -f workers/paddle_rich_label/Dockerfile.gpu --target customer -t goodvision-rich-label-worker:consolidation-test .
```

The customer main image fixes `HARNESS_INFERENCE_PROFILE=customer-standard`.
That internal preset retains OpenRouter, the DeepInfra backend only, and
`qwen/qwen3.5-35b-a3b`, with fallback disabled. Delivery Compose does not forward
provider, model or endpoint overrides; the development target keeps configurable
routing. Customer configuration, API profiles and UI continue to use neutral
feature names. Saved credentials are scoped to their service: do not silently
reuse a previous service's key. Supply the preset's credential through the
existing console or neutral `INFERENCE_API_KEY` when upgrading.

The corresponding developer checks use `--target development` with different
test tags. A clean dependency build needs network access and substantial free
disk space. Recipe/static checks and tests using previously installed dependency
layers do not prove that a fresh full GPU build has completed; record which
validation was actually performed. Dependency version ranges also mean a clean
build is not a promise of byte-for-byte reproducibility.

The current release uses `2026.09.09-full.1` for both product images and
`dist/goodvision-full-2026.09.09.1` as its staging directory. Its archive is
`goodvision-full-2026.09.09.1.tar.gz`. `Dockerfile.release` can reuse the audited
GPU dependencies while reinstalling every main application file and replacing
the worker and runtime entrypoints from the current checkout. Source hashes
verify the resulting application files; dependency checks verify the inherited
runtime. The full GPU Dockerfiles remain the path for rebuilding dependencies.
Use new product tags and archive names for each later release, and update the
launcher, Compose files, deployment guide, exporter, smoke defaults and tests.

### Customer-facing boundary

Neutral names apply to the supported UI, API, event/webhook output, deployment
materials, default service logs and supplied check entrypoints. Known engine,
model and provider markers are handled by the shared `privacy.py` helpers; errors
must not expose upstream exception text or echo private input. The customer
process gateway also handles native stdout/stderr while preserving exit status
and shutdown behavior. A single `tools/customer_check.py` selects its check mode
from the installed filename and enters that gateway.

Business identifiers (for example person, source and message IDs) remain exact,
even when customer-supplied text resembles a brand. They are correlation/CRUD
keys, not implementation metadata; redacting them would break lookups or merge
distinct records.

This is not removal of working dependencies or a promise against image
inspection. Preserve upstream licenses, copyright notices and metadata. Runtime
internals, required dependency files and historical image layers may retain
implementation names; directly invoking internal tools bypasses the supported
customer-output boundary. Model authorization remains separate from code
packaging, and no real keys, model weights, photographs or galleries belong in a
release staging directory.

### Release staging and verification

Keep one deployment source, `DEPLOY.md`, in the repository. Use a new staging
directory and copy these reviewed sources:

| Repository source | Customer bundle destination |
| --- | --- |
| `DEPLOY.md` | Same name |
| `API_DELIVERY.md` | Same name |
| `compose.full-gpu.delivery.yaml` | `compose.yaml` |
| `compose.full-gpu.accelerated.yaml` | `compose.gpu.yaml` |
| `.env.full-gpu.delivery.example` | `.env.example` |
| `config.delivery.yaml` | Same name |
| `tools/start_delivery.py` | `start_delivery.py` |
| `schemas/analysis-result-v2.schema.json` | Same path |
| `schemas/analysis-result-v1.schema.json` (historical contract) | Same path |

Add a reviewed `RELEASE.md` recording the exact candidate versions, completed
checks and unverified boundaries, plus `SOURCE_HASHES.json` for the application
files verified inside both images. Export with
`python tools/export_full_delivery.py <new-bundle-directory>`.
The exporter streams exactly the two current image tags through a binary-safe
gzip pipe, verifies the complete stream and manifest, and writes `IMAGES.json`
and `SHA256SUMS.txt`. `docker load -i` accepts the compressed archive directly.
The fixed eleven-file audit in `tools/audit_customer_bundle.py` describes the
historical 2026.09.03 bundle; it is not the current release inventory.

Before publishing, run the full unit suite and isolated image protocol check
with the candidate image tags:

```powershell
pwsh -NoProfile -File tools/run_full_delivery_smoke.ps1 -HarnessImage goodvision-harness:2026.09.09-full.1 -WorkerImage goodvision-rich-label-worker:2026.09.09-full.1
```

The smoke uses disposable resources, not production sessions or data volumes.
Reload the new archive and repeat the checks against its selected tags. Synthetic
protocol tests and `--runtime` dependency checks do not establish actual GPU or
model correctness. Real face/detection/attribute inference, authorized-video
results, evidence photos and performance still require the target-host checks
in the delivery guide. Do not restart the local demo merely to validate a release.

## Verification

```bash
python -m pytest
docker compose config
docker build -t goodvision-harness:demo .
```

The test suite is offline by default: it validates positive-result bbox
requirements, normalized pixel placement, exact frame/result correlation,
out-of-order timeline recovery, delayed interpolation and session resets,
callback completion, analysis readiness, result/frame fan-out, HTTP endpoints,
and webhook delivery/shutdown without calling RTSP or an LLM.
