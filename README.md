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
that warmup. These runtime defaults remain fixed:

- source/face sampling: 2.5 FPS, 1280 px maximum image side, JPEG quality 90;
- the same frame is resized to 320 px, JPEG quality 75, for the single remote
  safety request;
- display output: 15 FPS, 720 px maximum image side, JPEG quality 70, delayed
  3.0 seconds so most completed 2.5 FPS analysis keyframes can be interpolated;
- one-frame windows with stride 1 and six concurrent model requests;
- `max_tokens=1200`, no API retry, and a fixed validated model route with
  route fallback disabled;
- one multi-label, high-recall safety prompt that independently evaluates
  `fight`, `smoking`, `fall`, and `weapon` and permits simultaneous positives.

The original headless CLI keeps its prior event and result protocol. Browser
testing now uses one unified page, based on the familiar original Demo UI.
Both browser-camera and network-stream modes use the same grounded delivery
pipeline and expose customer JSON plus analyzed and raw video outputs. The
legacy single-event and face-only modes remain available as diagnostic API
modes; the page lets the operator select any non-empty feature combination.

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
LLM API. The host does not need inbound Internet access to port 8080 when the
receiver is also on the host.

## Deploy

For a prebuilt image-tar delivery with no source tree on the customer server,
use `compose.delivery.yaml` and follow [DEPLOY.md](DEPLOY.md). The regular
`compose.yaml` remains the developer build path.

On the GoodVision server (`192.168.20.168`):

```bash
cp .env.example .env
# The robot WHEP URL is already configured; add the real MODEL_API_KEY.
docker compose up --build -d
docker compose ps
```

For normal fixed-five mode, the demo configuration may initialize authorized
local face files in the Compose `face-models` volume on first use. Alternatively,
pre-populate `/models/models/<configured-profile>/`. `HARNESS_DRY_RUN=true` does
not need them.

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
# List captured strangers, then retrieve a returned stranger_id.
curl http://127.0.0.1:8080/v1/faces/strangers
curl --output stranger.jpg \
  http://127.0.0.1:8080/v1/faces/strangers/stranger-ID/photo
docker compose logs --tail=200 harness
docker volume inspect goodvision-harness_face-models goodvision-harness_face-data
docker compose exec harness sh -lc \
  'id; ls -la /models/models /data; test -w /models; test -w /data'
```

In Compose, authorized face weights belong under
`/models/models/<configured-profile>/` in the persistent `face-models`
volume, while the gallery is `/data/face-gallery.json` in the persistent
`face-data` volume. Stranger embeddings and one representative face crop per
ID are retained in `/data/face-gallery-strangers.json` and
`/data/stranger-captures/` on that same volume. Repeated unknown faces whose
embeddings meet the configured threshold reuse the same `stranger_id`; live
result objects include that ID and its `photo_url`. `docker compose down` preserves both volumes;
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
| `GET /v1/faces/strangers` | List retained stranger IDs and their photo URLs. |
| `GET /v1/faces/strangers/{stranger_id}/photo` | Return one retained face crop as `image/jpeg`. |
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
[`schemas/analysis-result-v1.schema.json`](schemas/analysis-result-v1.schema.json).
A result has the following shape:

```json
{
  "type": "analysis_result",
  "schema_version": "1",
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
  "judgments": [
    {
      "event_label": "fight",
      "detected": false,
      "confidence": 0.91,
      "description": "No fighting cues are visible.",
      "object_ids": []
    },
    {
      "event_label": "smoking",
      "detected": true,
      "confidence": 0.94,
      "description": "A person holds an e-cigarette at the mouth.",
      "object_ids": ["object-1", "object-2"]
    },
    {
      "event_label": "fall",
      "detected": false,
      "confidence": 0.88,
      "description": "No fall is visible.",
      "object_ids": []
    },
    {
      "event_label": "weapon",
      "detected": false,
      "confidence": 0.86,
      "description": "No weapon cue is visible.",
      "object_ids": []
    },
    {
      "event_label": "stranger",
      "detected": true,
      "confidence": 0.93,
      "description": "One face does not match the known-person gallery.",
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
frames for 3.0 seconds. Results at 2.5 FPS act as bbox keyframes on the same
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
overlay JPEG quality, and optional webhook behavior. The
validated API route remains fixed and route fallback is disabled. The API retry
count remains zero so a failed live request is not replayed as a stale result.
Persisted console settings override environment values; otherwise the generic
stream variable takes precedence over the legacy RTSP variable:

| Variable | Meaning |
| --- | --- |
| `HARNESS_STREAM_URL` | Primary RTSP/RTMP/WHEP input URL. |
| `RTSP_URL` | Deprecated compatibility fallback when `HARNESS_STREAM_URL` is empty. |
| `MODEL_API_KEY` | Model API credential. |
| `HARNESS_DRY_RUN` | Disable external inference and emit deterministic results. |
| `HARNESS_AUTOSTART` | Start the configured network stream when the service starts; set false for local camera testing. |
| `HARNESS_SOURCE_ID` | Stable customer-facing source identifier. |
| `RESULT_WEBHOOK_URL` | Optional result-push destination. |
| `HARNESS_WEBHOOK_SECRET` | Optional webhook HMAC secret. |
| `HARNESS_BIND_ADDRESS` | Host bind address used by Compose. |
| `HARNESS_PORT` | Host HTTP port used by Compose. |
| `FACE_MODEL_ROOT` | Root containing operator-supplied face-analysis files. |
| `FACE_GALLERY_PATH` | Persistent known-person gallery JSON path. |

Do not commit `.env`, RTSP credentials, or LLM API keys.

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

`face.min_face_size` is the live-recognition floor in source-frame pixels. It is
kept low enough for distant safety detections, while
`face.enrollment_min_face_size` remains stricter so low-detail registration
photos cannot degrade the known-person gallery.

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
