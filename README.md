# Goodvision Harness

An independent demo service that pulls a live RTSP stream, sends sampled
frames to a public vision-LLM API, and exposes both structured analysis JSON
and an annotated video stream.

The harness has no runtime dependency on the original `goodvision`
repository.

## Demo topology on `14.198.55.6`

```text
external camera
      |
      | RTSP published to 14.198.55.6:8554
      v
Linux host 14.198.55.6
      |
      | rtsp://host.docker.internal:8554/live/livestream
      v
goodvision-harness container
      |                         |
      | HTTPS 443               | localhost:8080
      v                         v
OpenRouter / Qwen          JSON + annotated MJPEG
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
- access from the container to the host RTSP publisher on TCP 8554;
- local TCP 8080 for the receiver or reverse proxy;
- correct system time and CA certificates for HTTPS.

A GPU is not required: video decoding and overlays run on CPU, while model
inference runs through the public LLM API. The host does not need inbound
Internet access to port 8080 when the receiver is also on the host.

## Deploy

On `14.198.55.6`:

```bash
cp .env.example .env
# Put the real OPENROUTER_API_KEY in .env.
docker compose up --build -d
docker compose ps
```

The Compose file already maps `host.docker.internal` to the Linux Docker
host. Its default input is therefore:

```text
rtsp://host.docker.internal:8554/live/livestream
```

The hardware-provided public endpoint remains:

```text
rtsp://14.198.55.6:8554/live/livestream
```

Using the host alias from inside Docker avoids leaving the host only to
hairpin back through its public address. Override `RTSP_URL` in `.env` if
the publisher is elsewhere or requires credentials.

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

Open `http://127.0.0.1:8080/` on the server for the minimal dashboard. If a
remote browser must access it, use an authenticated reverse proxy or set
`HARNESS_BIND_ADDRESS=0.0.0.0` and restrict port 8080 with the server
firewall. Do not expose an unauthenticated demo port broadly. If a separate
browser frontend consumes the APIs, put it behind the same-origin reverse
proxy; permissive CORS is intentionally not enabled. The built-in dashboard is
same-origin and needs no CORS configuration.

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
| `GET /` | Minimal same-service demo dashboard. |

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
readiness windows, overlay JPEG quality, and optional webhook behavior.
The demo default leaves provider selection to OpenRouter so it can route around
a provider outage; restrict `inference.request_parameters.provider` only when a
customer data policy requires it. See OpenRouter's
[provider-routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection).
Environment variables override deployment-specific values:

| Variable | Meaning |
| --- | --- |
| `RTSP_URL` | RTSP/RTMP/WHEP input URL. |
| `OPENROUTER_API_KEY` | Public LLM API credential. |
| `HARNESS_DRY_RUN` | Disable external inference and emit deterministic results. |
| `HARNESS_SOURCE_ID` | Stable customer-facing source identifier. |
| `RESULT_WEBHOOK_URL` | Optional result-push destination. |
| `HARNESS_WEBHOOK_SECRET` | Optional webhook HMAC secret. |
| `HARNESS_BIND_ADDRESS` | Host bind address used by Compose. |
| `HARNESS_PORT` | Host HTTP port used by Compose. |

Do not commit `.env`, RTSP credentials, or LLM API keys.

## Local development

```powershell
python -m venv .venv
python -m pip install -e '.[dev]'
$env:RTSP_URL = 'rtsp://14.198.55.6:8554/live/livestream'
$env:HARNESS_DRY_RUN = 'true'
goodvision-harness
```

Use `export RTSP_URL=...` and `export HARNESS_DRY_RUN=true` on Linux/macOS.
The retained headless CLI emits the canonical grounded `analysis_result` JSONL;
the browser-input development demo uses the same grounded client. They are
available as `goodvision-harness-cli` and `goodvision-harness-demo`. The Docker
HTTP service is the supported server delivery path.

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
