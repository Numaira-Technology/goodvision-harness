# Source deployment and operations

[Documentation index](../README.md) · [Repository overview](../../README.md)

Commands and unlinked file paths in this guide are relative to the repository root.

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
follow [DEPLOY.md](../../DEPLOY.md) and run the supplied `start_delivery.py` for the
complete package. Both current images include their GPU components; the launcher
enables the rich worker only when a usable GPU is available. The source-tree
equivalent is `python tools/start_delivery.py`.
[API_DELIVERY.md](../../API_DELIVERY.md) is the public interface contract.
The regular source-tree `compose.yaml` remains the developer build path.
Superseded CPU/separate-image delivery recipes have been removed; use the
full-package files listed in the deployment guide.

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

The development default model route uses OpenRouter with `qwen/qwen3.5-35b-a3b:nitro`. Set
`INFERENCE_PROVIDER=dashscope` to switch to DashScope; model and endpoint
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

Event and stranger analysis run independently when the face worker is available.
The current local face pool retains admitted tasks and commits identity results
in source order; capacity failure is explicit. Legacy integrations using CPU
single-flight may still omit the face branch on a busy sample; this is not the
new continuous route's scheduling policy. The live classification cards retain each feature's latest
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
automatic video connection choice, and a write-only API key in the persistent
`/data` volume. Automatic connection never starts analysis; use the global
analysis control after the selected video connections are ready.
Stop the active session before saving. Saved values become effective for the
next session and remain authoritative after container replacement or restart;
the API key and complete stream URL are never returned by the settings API.

Under **Output debug & delivery checks**, **视频缓存（秒）** adjusts the display
buffer from 0 to 30 seconds (decimals supported; default 2.5). Click
**应用并保存缓存时间** to apply it during an active session and retain it across
restarts. Increasing the delay briefly pauses output while the buffer fills;
decreasing it lets queued frames catch up. A writable deployment data volume is
required. The equivalent API request is `PUT /v1/deployment/settings` with
`{"buffer_delay_seconds": 3.5}`; only a buffer-only update is allowed while running.

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
# Delayed original-frame display; holds annotations between inference results.
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
`/data/event-evidence/`, with 7-day/10-GiB retention. The removed public stranger
gallery is separate from the current private ordered face store: source images,
raw embeddings and observation facts are retained under the gallery directory's
`private-face` folder for identity commit and review. These records stay private;
see [event evidence](../architecture/event-evidence.md)
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
