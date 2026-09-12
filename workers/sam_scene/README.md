# SAM scene perception worker

This is a real optional SAM 3 image detector behind the Harness rich-label
contract. It discovers people and configured object categories at keyframes,
maintains short-lived online tracks, and distinguishes real observations from
motion predictions. It contains no simulated production detector and never
downloads model weights at startup.

## Exact model and scope

- Official source: `facebookresearch/sam3`, commit
  `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` (2026-08-26).
- Checkpoint: **`facebook/sam3` → `sam3.pt`**. SAM 3.1 multiplex checkpoints
  belong to a different video model and are not supported by this adapter.
- Uses `build_sam3_image_model`, `Sam3Processor.set_image`,
  `reset_all_prompts`, and `set_text_prompt`. Pixel xyxy boxes are converted to
  the source-normalized xyxy contract. Detector weights are loaded strictly
  from the local checkpoint with `weights_only=True`; no random-weight fallback.
  The official checkpoint's unused `backbone.visual.sam2_convs` branch is
  accepted only when each extra key and tensor shape matches its image `convs`
  counterpart. All required image weights and shapes remain mandatory; other
  unexpected detector keys fail readiness.
- CUDA is mandatory in deployment. Readiness requires model loading, text-prompt
  inference for all configured labels, and CUDA synchronization to succeed.
  Python tests inject API doubles; their health responses always say
  `gpu_ready: false` even if a double claims otherwise.
- **This is not native SAM 3.1 video tracking.** The inspected shared video
  predictor starts a session from a completed resource and propagates through
  that resource. Its public dispatcher has no live-frame append operation.
  `handle_stream_request` streams *results*, not incoming camera frames.
  We do not append files behind that API or run overlapping offline chunks while
  claiming uninterrupted SAM memory.
- Between SAM keyframes, a bounded constant-velocity association tracker
  predicts boxes. It is deliberately conservative around crowded scenes;
  ambiguous associations retire old IDs. Prediction collisions stop propagation
  and force a new detection. Long source gaps and resolution changes also create
  fresh IDs. There is no appearance ReID, face identity, pose, optical flow,
  cross-camera identity, or event classification in this worker.
- SAM computes masks as part of its supported image API. This version exports
  bounding boxes only. Person-specific high-resolution evidence stays in the
  Harness; SAM's own input is capped at a 1280-pixel longest side by default to
  bound the processor's mask upsampling memory.

Official references verified 2026-09-12:

- [Pinned model builder and checkpoint loading](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model_builder.py)
- [Pinned optional SAM2 neck construction](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/necks.py)
- [Pinned image processor and output coordinates](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_image_processor.py)
- [Shared SAM 3 / 3.1 video request dispatcher](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_base_predictor.py)
- [Official CUDA / PyTorch installation requirements](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/README.md)
- [Official PyTorch 2.10.0 / torchvision 0.25.0 CUDA wheels](https://pytorch.org/get-started/previous-versions/#v2100)
- [Model access request](https://huggingface.co/facebook/sam3)
- [SAM License](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/LICENSE)

## Build and run

Build context is this directory. The Linux image uses Python 3.12, PyTorch 2.10.0,
torchvision 0.25.0 and CUDA 12.8 wheels. It needs a compatible NVIDIA driver and
NVIDIA Container Toolkit on the host. The build imports the pinned SAM API and
runs `pip check` without downloading weights. Direct dependencies are pinned;
the requirements file is not a full transitive/hash lock.

```sh
docker build -t goodvision-sam-scene workers/sam_scene
```

After model access has been granted, an authorized operator can authenticate
Hugging Face on the GPU host and download **`facebook/sam3/sam3.pt`** into a local
model directory. No token is needed by the running worker. Do not commit the
checkpoint or credentials. The following run example assumes the approved file
already exists at `/srv/goodvision/models/sam3.pt`:

```sh
docker run --rm --name goodvision-sam-scene --gpus all \
  -p 127.0.0.1:8092:8092 \
  --mount type=bind,source=/srv/goodvision/models,target=/models,readonly \
  -e SAM_SCENE_LABELS=person \
  goodvision-sam-scene
```

Bind the worker to localhost or a private service network. It has no external
authentication endpoint. Run **one Uvicorn process** per instance; multiple
processes would have separate models and separate track state. For multiple
instances, route each camera session consistently to the same instance.

Missing checkpoint, incompatible state dictionaries, missing dependencies and
CUDA failures leave `/health/live` available while readiness returns 503. After
correcting startup conditions, restart the worker to load the model. A live
health response alone does not enable inference.

## HTTP contract

`POST /v1/tracks?include_attributes=false` receives raw JPEG bytes with:

```text
Content-Type: image/jpeg
X-GoodVision-Stream-Session-Id: camera-session-unique-id
X-GoodVision-Frame-Sequence: 123
X-GoodVision-Timestamp-Seconds: 4.1
```

The response has exactly the Harness `RichLabelSnapshot` top-level fields:

```json
{
  "stream_session_id": "camera-session-unique-id",
  "frame_sequence": 123,
  "timestamp_seconds": 4.1,
  "objects": [{
    "track_id": "sam-fresh-session-00001",
    "label": "person",
    "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.9},
    "confidence": 0.92,
    "attributes": {},
    "observation_state": "predicted",
    "last_observed_timestamp_seconds": 4.0,
    "last_observed_frame_sequence": 120
  }]
}
```

`observed` objects reference the current source timestamp and sequence;
`predicted` objects retain the last SAM observation timestamp and sequence.
Prediction confidence is the last detector confidence, **not** evidence of a new
observation. Entity/event consumers must not count predicted boxes as fresh
positive or negative evidence. IDs are opaque, unique across worker sessions,
and are never real-world identity claims.

`POST /v1/labels` is a compatibility alias returning the same provenance fields.
`include_attributes` is accepted for client compatibility; attributes remain
empty because this worker does not run an attribute classifier.

`GET /health/ready` and `GET /health/tracking` return the same readiness, with
`ready`, `gpu_ready`, `backend: "sam3_image_motion"`, `sam3_commit`,
`native_sam_video_tracking: false`, and `error_code`. Only a production response
with both booleans true establishes GPU readiness. `GET /health/live` checks the
HTTP process only.

Error responses:

| Status | Meaning |
| --- | --- |
| 400 | Invalid metadata or JPEG |
| 409 | Duplicate/backward sequence or backward source timestamp |
| 413 | JPEG byte or pixel limit exceeded |
| 415 | Content type must be `image/jpeg` |
| 429 | Inference busy or all session slots active; `Retry-After: 1` |
| 503 | Real model unavailable or inference failed |
| 504 | Request upload/decode/inference deadline exceeded |

Frame sequence must increase within each session. Timestamps are source seconds
and may be equal on successive frames, but cannot decrease. On camera/decoder
restart send a new session ID. Stale requests do not reset accepted state. Session
idle cleanup uses worker monotonic time; source gaps and track expiry use source
timestamps.

One global admission slot bounds outstanding upload/decode/GPU work. A timed-out
CUDA call cannot safely be killed from Python, so the slot remains busy until
the call actually returns; late results never commit track state. A permanently
hung GPU requires a worker restart. The client timeout should be a little longer
than the worker deadline for diagnostic calls. Live Harness sampling can use a
shorter client timeout, but a dropped result still consumes one bounded call.

## Configuration

All values use the `SAM_SCENE_` prefix; all controls are validated at startup.

| Variable suffix | Default | Meaning |
| --- | --- | --- |
| `CHECKPOINT_PATH` | `/models/sam3.pt` | Approved local SAM 3 checkpoint |
| `LABELS` | `person` | Comma-separated prompts, 1–16 unique values |
| `CONFIDENCE` | `0.5` | Minimum detector confidence |
| `KEYFRAME_SECONDS` | `0.4` | Maximum source interval between SAM detections |
| `KEYFRAME_REQUESTS` | `5` | Maximum accepted request interval between detections |
| `TRACK_TTL_SECONDS` | `1.2` | Maximum age of the last real observation |
| `SOURCE_GAP_SECONDS` | `2.0` | Gap that resets track identities |
| `MATCH_IOU` | `0.25` | Association candidate overlap threshold |
| `AMBIGUITY_MARGIN` | `0.15` | Minimum margin between competing association scores |
| `CROSSING_IOU` | `0.5` | Predicted overlap which retires both old identities |
| `MAX_TRACKS` | `64` | Total tracked objects per camera session |
| `MAX_SESSIONS` | `8` | Maximum active camera sessions; active sessions are not evicted |
| `SESSION_IDLE_SECONDS` | `60.0` | Worker-time idle expiry |
| `MAX_JPEG_BYTES` | `8388608` | Compressed request size cap |
| `MAX_IMAGE_PIXELS` | `8294400` | Decode cap before RGB allocation (3840×2160) |
| `INFERENCE_MAX_SIDE` | `1280` | Longest side supplied to SAM's processor |
| `REQUEST_TIMEOUT_SECONDS` | `15.0` | Upload + decode + inference deadline |

These defaults are engineering starting points, not measured capacity. Crowding,
camera rotation, blur, occlusion, and rapid nonlinear motion can break the simple
association tracker. Ambiguity intentionally fragments tracks to prevent carrying
someone else's identity or event evidence. Native online SAM tracking or an
appearance/flow tracker can replace this adapter after separate evaluation.

## Verification and remaining hardware validation

Automated tests cover protocol parsing by the actual Harness parser, temporal
provenance, motion prediction, ambiguity, reset/order handling, resource bounds,
timeout isolation, fake-readiness rejection and the official model API call
contract. The API-contract test uses doubles; it is not a model result.

From the repository root in a development environment:

```sh
python -m pytest tests/test_sam_scene_worker.py -q
python -m ruff check workers/sam_scene tests/test_sam_scene_worker.py
```

Once access and a GPU are available, run the real smoke tool against chronologically
ordered JPEG frames from the same camera. The first image must visibly contain a
person. It refuses a fake/CPU-unready service, validates source provenance, and
prints per-frame latency and IDs:

```sh
python workers/sam_scene/smoke.py \
  --endpoint http://127.0.0.1:8092 --frame-interval 0.2 \
  /data/frame000.jpg /data/frame001.jpg /data/frame002.jpg
```

Pending: gated checkpoint loading on real CUDA hardware, building/running this
Linux image in the target deployment environment, real multi-person crossings
and occlusion replay, latency/VRAM measurements, and coexistence with face/VLM
services. No GPU throughput or detection-accuracy result is claimed by these unit
tests. In particular, SAM 3.1's published H100 video speed does not describe this
SAM 3 image-keyframe implementation.
