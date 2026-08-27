# Goodvision Harness

An independent demo service that pulls a live RTSP stream, sends sampled
frames to a public vision-LLM API, and exposes both structured analysis JSON
and an annotated video stream.

The harness has no runtime dependency on the original `goodvision`
repository.

## Preserved Harness behavior

The delivery service extends the original Harness; it does not retune its
recognition loop. These original defaults remain fixed:

- model sampling: 2.5 FPS, 320 px maximum image side, JPEG quality 75;
- display sampling: 15 FPS, 720 px maximum image side, JPEG quality 70;
- one-frame windows with stride 1 and six concurrent model requests;
- `max_tokens=96`, no API retry, and OpenRouter routed only to CoreWeave with
  provider fallback disabled;
- the original system and user prompts, with one bbox delivery paragraph
  appended to the end of the user prompt.

The original headless CLI keeps its prior event and result protocol. Browser
testing now uses one unified page, based on the familiar original Demo UI.
Both browser-camera and network-stream modes use the same grounded delivery
pipeline and expose customer JSON plus analyzed and raw video outputs.

## Demo topology on `14.198.55.6`

```text
external camera / RTSP publisher
      |
      | configured RTSP_URL (pulled by the server)
      v
goodvision-harness container on 14.198.55.6
      |                         |
      | HTTPS 443               | localhost:8080
      v                         v
OpenRouter / Qwen          JSON + analyzed/raw video
                                |
                                v
                    receiver on 14.198.55.6
```

The receiver does **not** need a webhook for this demo. It can read the latest
JSON, subscribe to the JSON SSE stream, and consume the MJPEG stream directly.
Webhook delivery is an optional adapter when the receiver specifically wants
the harness to push each result.

The default host bind (`127.0.0.1:8080`) is for a receiver running as a process
on `14.198.55.6`. Compose also creates the stable `goodvision-demo` Docker
network. A receiver container can join that network and call
`http://goodvision-harness:8080`; `127.0.0.1` inside the receiver container
would refer to the receiver itself.

## Server prerequisites

The default deployment needs:

- a 64-bit Linux host with Docker Engine 20.10+ and Docker Compose v2;
- 4 CPU cores and 8 GB RAM recommended for a comfortable demo
  (2 cores and 4 GB is the practical minimum for one stream);
- about 10 GB free disk for the image, build cache, and operating headroom;
- outbound HTTPS/DNS access to the configured LLM endpoint on TCP 443;
- network access from the container to the configured RTSP endpoint and port;
- local TCP 8080 for the receiver or reverse proxy;
- correct system time and CA certificates for HTTPS.

A GPU is not required: video decoding and overlays run on CPU, while model
inference runs through the public LLM API. The host does not need inbound
Internet access to port 8080 when the receiver is also on the host.

## Deploy

On `14.198.55.6`:

```bash
cp .env.example .env
# Put the agreed camera RTSP_URL and the real OPENROUTER_API_KEY in .env.
docker compose up --build -d
docker compose ps
```

`RTSP_URL` has no automatic default. Use the camera's agreed public/private
RTSP endpoint directly. If an RTSP relay actually runs on `14.198.55.6`, the
Compose host-gateway mapping also permits:

```text
rtsp://host.docker.internal:8554/live/livestream
```

Leaving `RTSP_URL` empty keeps the service unready instead of silently
connecting to an assumed source. OpenCV/FFmpeg transport options are likewise
not forced; the original RTSP behavior is preserved. Add
`OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|timeout;5000000` to the
container environment only when troubleshooting proves that source needs it.

The HTTP service binds to `127.0.0.1:8080` on the host by default:

```bash
curl http://127.0.0.1:8080/health/live
curl http://127.0.0.1:8080/health/ready
curl http://127.0.0.1:8080/v1/status
```

`/health/live` only proves the HTTP process is alive. `/health/ready` returns
200 only after both a fresh source frame and a fresh successful analysis have
arrived; invalid API credentials or repeated model failures therefore keep the
container unready. The Docker image uses this stricter endpoint for its health
check, with a startup allowance for the first model request.

Open `http://127.0.0.1:8080/` for the single internal test page. It has the
familiar Camera / Network stream controls and shows the raw preview, the exact
analyzed frame with bbox, and the complete JSON result together. Leave the
network URL blank to use `RTSP_URL`, or enter a different RTSP/RTMP/WHEP URL
for that run.

A browser can use its local camera from `localhost`. When the page is opened
remotely from `14.198.55.6`, browsers normally require HTTPS/WSS before they
allow camera access. Use an HTTPS reverse proxy or an SSH tunnel to
`localhost:8080`. Network-stream mode does not depend on browser camera
permission. Restrict this unauthenticated internal page with the server
firewall or reverse proxy.

### Zero-cost smoke mode

To test RTSP, JSON delivery, and both video outputs without calling the LLM:

```dotenv
HARNESS_DRY_RUN=true
```

Restart with `docker compose up -d`. Dry-run results are deterministic and
include a synthetic normalized bbox.

## Output endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health/live` | HTTP-process liveness. |
| `GET /health/ready` | Ready only with fresh source video and a fresh successful analysis. Used by the Docker healthcheck. |
| `GET /v1/status` | Sanitized source, pipeline, counters, and delivery state. |
| `GET /v1/results/latest` | Latest customer-facing analysis JSON. Returns 404 before the first result. |
| `GET /v1/results/events` | Server-Sent Events stream of analysis JSON with bounded in-memory replay. |
| `GET /v1/video/latest.jpg` | Latest exact analyzed frame with burned-in verdict/bbox. |
| `GET /v1/video/stream.mjpeg` | Annotated MJPEG at result cadence: one exact analyzed frame per JSON result. |
| `GET /v1/video/live/latest.jpg` | Latest unannotated source-preview frame. |
| `GET /v1/video/live/stream.mjpeg` | Continuous unannotated source preview. |
| `GET /` | Single Camera / Network stream internal test page. |

Examples:

```bash
curl http://127.0.0.1:8080/v1/results/latest
curl -N http://127.0.0.1:8080/v1/results/events
ffplay http://127.0.0.1:8080/v1/video/stream.mjpeg
ffplay http://127.0.0.1:8080/v1/video/live/stream.mjpeg
```

## Analysis result contract

The canonical schema is
[`schemas/analysis-result-v1.schema.json`](schemas/analysis-result-v1.schema.json).
A result has the following shape:

```json
{
  "type": "analysis_result",
  "schema_version": "1",
  "message_id": "cb44b47c-332f-4947-971b-8297fc8c64db",
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
  "judgments": [
    {
      "event_label": "smoking",
      "detected": true,
      "confidence": 0.94,
      "description": "A person is visibly smoking at the mouth.",
      "object_ids": ["object-1"]
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
      }
    }
  ],
  "model": {
    "name": "qwen/qwen3.5-35b-a3b",
    "latency_seconds": 1.247
  }
}
```

`bbox` is top-left-origin `xyxy`, normalized to the source frame. Convert
it to pixels with `x * frame.width` and `y * frame.height`.
`timestamp_seconds` is monotonic within one stream session;
`observed_at` is the harness ingest-time estimate because ordinary RTSP
decoding does not guarantee access to the camera's absolute capture clock.

For every successful JSON result, the harness annotates the exact newest frame
that the result references. `GET /v1/video/latest.jpg` and every multipart JPEG
in `GET /v1/video/stream.mjpeg` carry `X-Analysis-Message-Id`; its value is the
same as the JSON `message_id`. The annotated feed runs at model-result cadence
and can look stepwise when inference is slow. The `/v1/video/live/*` endpoints
provide a smoother raw preview and intentionally do not paint delayed bbox data
onto later frames.

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

`config.yaml` controls sampling, model selection, prompts, concurrency,
readiness windows, overlay JPEG quality, and optional webhook behavior. The
original CoreWeave-only OpenRouter route remains explicit:
`provider.only=[coreweave]` and `allow_fallbacks=false`. The API retry count
also remains zero so a failed live request is not replayed as a stale result.
Environment variables override deployment-specific values:

| Variable | Meaning |
| --- | --- |
| `RTSP_URL` | RTSP/RTMP/WHEP input URL. |
| `OPENROUTER_API_KEY` | Public LLM API credential. |
| `HARNESS_DRY_RUN` | Disable external inference and emit deterministic results. |
| `HARNESS_AUTOSTART` | Start the configured network stream when the service starts; set false for local camera testing. |
| `HARNESS_SOURCE_ID` | Stable customer-facing source identifier. |
| `RESULT_WEBHOOK_URL` | Optional result-push destination. |
| `HARNESS_WEBHOOK_SECRET` | Optional webhook HMAC secret. |
| `HARNESS_BIND_ADDRESS` | Host bind address used by Compose. |
| `HARNESS_PORT` | Host HTTP port used by Compose. |
| `FACE_MODEL_ROOT` | InsightFace model cache; downloaded on first use. |
| `FACE_GALLERY_PATH` | Persistent known-person gallery JSON path. |

Do not commit `.env`, RTSP credentials, or LLM API keys.

## Local development

```powershell
python -m venv .venv
python -m pip install -e '.[dev]'
$env:HARNESS_AUTOSTART = 'false'
goodvision-harness-demo --host 127.0.0.1 --port 7900
```

Open `http://127.0.0.1:7900`, choose Camera, enable it, then press Start.
For a network test, choose Network stream and either enter a URL or leave it
blank to use `RTSP_URL`. Set `HARNESS_DRY_RUN=true` for a no-cost pipeline
check. `goodvision-harness-demo` is now a compatibility launcher for the same
unified grounded page; it is not a second legacy Demo. `goodvision-harness`
serves that same page on port 8080 by default. `goodvision-harness-cli`
retains the original window lifecycle, `vision_result`, and final-state JSONL
behavior.

### Local InsightFace face demo

Install the face extra (combine it with the test tools when developing):

```powershell
python -m pip install -e '.[face]'
# or
python -m pip install -e '.[dev,face]'
$env:HARNESS_AUTOSTART = 'false'
goodvision-harness-demo --host 127.0.0.1 --port 7900
```

Open `http://127.0.0.1:7900`, select **Face recognition**, enroll each known
person with a stable person ID, name, and one clear single-face image, then
choose Camera or Network stream and press Start. The first enrollment or face
run downloads the configured InsightFace model into `FACE_MODEL_ROOT`; the
known-person gallery persists at `FACE_GALLERY_PATH`.

The result envelope is ready for the separate visual Memory layer:
`analysis_kind` distinguishes event and `face_recognition` observations,
`observation_id` is the stable alias of `message_id`, and
`objects[].attributes` carries mode-specific metadata. A Memory implementation
only needs to implement the non-blocking `VisualMemorySink.submit(...)`
interface, which receives a `VisualMemoryRecord` containing the observation
and its exactly correlated evidence JPEG.

### How the Memory pieces fit together

There are three small, deliberately separate responsibilities:

- **FaceGallery** is the primary identity memory for the local InsightFace
  mode. It persists embeddings and answers known/unknown locally. In this mode,
  `detected=true` means an **unknown** person is present.
- **Reference memory** is an optional Qwen input prefix migrated from
  Goodvision PR #53. It loads up to eight private reference images once,
  sends them before the live frame, and rejects model-reported memory IDs that
  were not supplied. It uses the same stranger-detection contract as
  InsightFace: `detected=true` means an **unknown** live face is present.
  Known identities are reported independently in `matched_memory_ids`.
- **VisualMemorySink** is the existing output seam. It receives any completed
  event or face observation plus its exactly correlated JPEG; it does not
  participate in recognition.

Qwen face-reference results use the following matrix. Its `detected` column
matches InsightFace; `matched_memory_ids` is additional Qwen reference metadata:

| Live frame | `detected` | `matched_memory_ids` |
| --- | --- | --- |
| No face | `false` | `[]` |
| Known face only | `false` | Known IDs |
| Unknown face only | `true` | `[]` |
| Known and unknown faces | `true` | Known IDs |

The default remains local InsightFace for face recognition. Enable Qwen
reference memory only as an explicit fallback/experiment by setting
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
FaceGallery, while Qwen fallback references remain deployment configuration.

## Verification

```bash
python -m pytest
docker compose config
docker build -t goodvision-harness:demo .
```

The test suite is offline by default: it validates positive-result bbox
requirements, normalized pixel placement, exact frame/result correlation,
callback completion, analysis readiness, result/frame fan-out, HTTP endpoints,
and webhook delivery/shutdown without calling RTSP or an LLM.
