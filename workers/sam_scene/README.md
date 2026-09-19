# SAM scene perception worker

This worker provides three separately enabled perception paths:

- Existing `/v1/tracks`: SAM 3 image keyframes and conservative motion association.
- Optional [`/v1/video/batches`](#native-sam-31-video-batches): native SAM 3.1
  detection/tracking memory over a finite batch of already received frames.
- Optional `/v1/video/sessions`: persistent native SAM 3.1 memory with bounded
  history, appending only newly arrived frames within one explicit session.

It contains no simulated production detector and never downloads model weights
at startup. Native SAM 3.1 strict checkpoint loading, CUDA warmup and real video
and continuous-session inference have passed on the rented RTX 4090. See the
[GPU validation record](../../docs/architecture/sam31-gpu-validation-20260912.md)
for measured scope and remaining quality/long-run checks. The image adapter's
unit tests do not constitute its own real-checkpoint GPU acceptance.

## Existing SAM 3 image endpoints: exact model and scope

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
- **The image endpoints are not native SAM 3.1 video tracking.** The inspected shared video
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
- SAM exports simplified external mask contours on observed source frames,
  alongside bounding boxes. Predicted frames retain box-only geometry.
  Person-specific high-resolution evidence stays in the Harness; SAM's own input
  is capped at a 1280-pixel longest side by default to bound mask memory.

Official references verified 2026-09-12:

- [Pinned model builder and checkpoint loading](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model_builder.py)
- [Pinned optional SAM2 neck construction](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/necks.py)
- [Pinned image processor and output coordinates](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_image_processor.py)
- [Shared SAM 3 / 3.1 video request dispatcher](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_base_predictor.py)
- [Official CUDA / PyTorch installation requirements](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/README.md)
- [Official PyTorch 2.10.0 / torchvision 0.25.0 CUDA wheels](https://pytorch.org/get-started/previous-versions/#v2100)
- [Model access request](https://huggingface.co/facebook/sam3)
- [SAM License](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/LICENSE)

## Remote GPU startup and recovery

From the repository root, `docker compose -f compose.yaml -f
compose.sam31-remote.yaml up --build` runs `tools/start_sam.py`. The controller
owns the remote GPU lifecycle and uploads the current worker source. Startup
has three stages:

1. Before renting, copy the host-mounted SSH private key into a private temporary
   file with mode `0600` and validate it. The Windows source key is unchanged.
2. When creating the instance, pass a Vast `onstart_cmd` that restores root
   ownership and `700/600` permissions on `/root/.ssh` and `authorized_keys`.
   Existing public keys are preserved. This runs before the controller needs
   SSH; the later remote installation script cannot repair a blocked login.
   Try direct SSH first, then the ordinary Vast gateway with the same key.
3. Upload the worker, install dependencies, download the SAM checkpoint, and
   enable `sam_live`. Report ready only after its health response confirms
   `enabled`, `ready`, and `gpu_ready`.

The cloud startup hook uses Vast's documented
[SSH onstart mechanism](https://docs.vast.ai/api-reference/creating-instances-with-api).
It does not add keys, loosen SSH checks, or automatically rebuild an existing
instance when a connection fails.

For an existing running instance labelled `goodvision-sam-*`, run the controller
with `--resume-instance-id <id>` instead of its default command. This skips offer
selection and rental. A complete environment with the expected Python, PyTorch,
runtime dependency versions, SAM commit, and successful imports skips installation.
An incomplete environment reuses the existing Conda/Python executables while
installing missing or mismatched dependencies. Cached SAM weights are reused.
The retired hybrid detector and its weights are no longer uploaded or downloaded.
Failure and normal shutdown both release the selected instance, preserving the
normal replacement flow: the next default launch selects and rents a new GPU.
During creation, an explicit refusal that an offer is missing, already rented,
or unavailable advances to the next candidate, up to three different machines.
Authentication, credit, and parameter errors stop immediately with the provider's
redacted diagnostic. Timeouts and uncertain responses stop selection and reconcile
the unique instance label before cleanup, avoiding a second ambiguous rental.
An existing instance with broken SSH permissions still needs its startup hook
repaired separately; recovery does not silently alter its cloud configuration.

## Build and run

Build context is this directory. The Linux image uses Python 3.12, PyTorch 2.10.0,
torchvision 0.25.0 and CUDA 12.8 wheels by default. The build argument
`TORCH_CUDA` accepts `cu128` or `cu126`; other values fail before installation.
It needs a compatible NVIDIA driver and
NVIDIA Container Toolkit on the host. The build imports the pinned SAM API and
runs `pip check` without downloading weights. Direct dependencies are pinned;
the requirements file is not a full transitive/hash lock.

```sh
docker build -t goodvision-sam-scene workers/sam_scene
```

For a host whose driver reports CUDA 12.6, explicitly select the matching wheels:

```sh
docker build --build-arg TORCH_CUDA=cu126 \
  -t goodvision-sam-scene workers/sam_scene
```

On 2026-09-12, SSH inspection of the rented RTX 4090 24 GB machine confirmed
driver **560.35.03**, driver-reported **CUDA 12.6**, **Ubuntu 24.04** and
**Python 3.12.3**. Use `cu126` for this machine. Its container template includes
CUDA 12.8.1; that does not change the host driver or require using cu128 wheels.
That machine has since passed native SAM 3.1 strict checkpoint loading, GPU
warmup and real video/continuous inference. These results apply to the measured
configuration; other GPUs and the separate SAM 3 image checkpoint need their own
model validation.

Both wheel variants are published for the exact pinned torch/torchvision pair.
The pinned SAM README requires CUDA 12.6 or later. CUDA 12.x minor-version
compatibility can allow newer runtimes on older drivers, but has feature and
driver PTX compilation limitations; it is not a substitute for testing the
complete model. Keep torch's matching Triton dependency: Triton 3.6 uses its
bundled ptxas to produce target-GPU cubins, and `compile=False` does not disable
all custom GPU kernels. Do not downgrade Triton or override its compiler merely
because the bundled compiler reports a newer CUDA version.
[NVIDIA compatibility limits](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
[Triton 3.6 compiler](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/backend/compiler.py#L427).

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

## Native SAM 3.1 video batches

This optional backend runs the official **SAM 3.1 multiplex video model**, with
native detection, mask propagation, object discovery and tracker memory inside
each submitted batch. It does not use the image worker's motion tracker.
Every batch creates and finally clears its own native inference state. The GPU
model remains loaded, but **no native memory or native ID continuity crosses a
batch boundary**. Repeated `batch_id` values are not an idempotency mechanism;
every completed request gets a new `predictor_epoch`.

All batch frames have already arrived before inference starts. This is an
explicit finite-video path, not an implementation of camera-frame append.
The host must reconcile entities conservatively across epochs and preserve
event history separately. Do not concatenate raw SAM object IDs from different
epochs. `out_probs` originates from native detector confidence; it is not an
independent, calibrated probability of identity or event correctness.

### Exact runtime and model

- Uses the same source commit `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`,
  which already includes SAM 3.1. Runtime verifies the installed SAM package's
  VCS `direct_url.json` commit. Install with the pinned Git URL in the Dockerfile;
  an unverified wheel or editable checkout will not pass this source check.
- Requires **`facebook/sam3.1` → `sam3.1_multiplex.pt`**, provided locally after
  obtaining the appropriate repository access. `facebook/sam3/sam3.pt` is a
  different model and cannot substitute. No HF token is needed during serving.
- Calls `build_sam3_multiplex_video_predictor` with `use_fa3=False`,
  `use_rope_real=False`, `compile=False`, `warm_up=False`,
  `async_loading_frames=False`, and 16 objects per multiplex bucket.
  Uses Python 3.12 / torch 2.10.0 / torchvision 0.25.0, with cu128 by default or
  the explicit cu126 build selection above. FlashAttention 3 is neither installed
  nor invoked.
- The official builder initially loads a merged checkpoint non-strictly. Before
  readiness, the worker separately validates the complete resulting model's
  required keys, unexpected keys, tensor shapes and finite floating weights,
  and performs `load_state_dict(strict=True)`. Only the official legacy prefix
  remapping (`sam3_model.` → `detector.`, `sam2_predictor.` → `tracker.`) is
  accepted, with collisions rejected. No random or partial-weight fallback.
  The native complex RoPE path matches the official checkpoint's `freqs_cis`
  buffers; enabling real RoPE would demand extra persistent real/imaginary
  buffers absent from the checkpoint. Both builder-entered autocast contexts
  are unwound before each inference call uses its own thread-local context.
- The pinned public `start_session` wrapper passes `offload_state_to_cpu` to
  a multiplex `init_state` that does not accept it. This adapter instead calls
  the model's exact `init_state(resource_path=PIL_images,
  offload_video_to_cpu=True, async_loading_frames=False)`, then
  `add_prompt(..., frame_idx=0, text_str="person")` and forward-only
  `propagate_in_video`. No growing directories, private session injection,
  reverse-time propagation or repeated prompts between frames are used.
- Grounding batching is disabled and postprocessing batch size is one to bound
  extra image/mask buffers. Native hotstart and masklet confirmation rules remain
  intact; short batches may contain no confirmed visible objects. Empty output
  does not establish an empty or safe scene.
- Readiness runs three real blank frames through these native operations,
  validates every frame's output, and synchronizes CUDA. It does not measure
  person-detection accuracy. API doubles always report both readiness flags false.

Official implementation references:

- [SAM 3.1 release and benchmark scope](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/RELEASE_SAM3p1.md)
- [Multiplex builder and default settings](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model_builder.py#L1006)
- [Native initialization, prompting and propagation](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_multiplex_tracking.py)
- [Public wrapper initialization signature](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_base_predictor.py#L112)

### Native HTTP contract

`GET /health/video` returns `ready`, `gpu_ready`, `enabled`,
`backend="sam3.1_video"`, `native_sam_video_tracking=true`,
`memory_scope="batch"`, `sam3_commit`, `limits`, and `error_code`.
It returns 503 until real GPU loading and native warmup succeed. `/health/live`
only reports process liveness. The existing `/health/tracking` and image routes
retain their image-model contract.

`POST /v1/video/batches` takes `Content-Type: application/json`:

```json
{
  "stream_session_id": "camera-session-a",
  "batch_id": "batch-001",
  "frames": [
    {"frame_sequence": 100, "timestamp_seconds": 25.0, "jpeg_base64": "..."},
    {"frame_sequence": 104, "timestamp_seconds": 25.25, "jpeg_base64": "..."}
  ]
}
```

Identifiers contain 1–160 ASCII letters, digits, underscore, dot, colon or
hyphen. Frame sequences are nonnegative 63-bit integers; source timestamps are
nonnegative finite numbers. Both strictly increase within a batch. Sequence
gaps are supported. All frames must have the same original dimensions.
Only the documented fields and valid base64 JPEGs are accepted; duplicate
JSON fields, coercion from strings/booleans, and nonfinite numbers are rejected.

The response is:

```json
{
  "stream_session_id": "camera-session-a",
  "batch_id": "batch-001",
  "predictor_epoch": "fresh-random-epoch",
  "backend": "sam3.1_video",
  "memory_scope": "batch",
  "frames": ["one RichLabelSnapshot per submitted source frame"],
  "completed_frame_sequence": 104,
  "completed_timestamp_seconds": 25.25
}
```

Each entry in `frames` is the existing `RichLabelSnapshot` schema, in exact source
order, including an empty `objects` list when no visible native mask exists.
Every visible person has normalized xyxy `bbox`, `observation_state="observed"`,
matching `last_observed_frame_sequence` / `last_observed_timestamp_seconds`, and:

```json
{
  "track_id": "sam31-fresh-random-epoch-7",
  "attributes": {
    "native_track": {
      "backend": "sam3.1_video",
      "epoch": "fresh-random-epoch",
      "object_id": "7"
    }
  }
}
```

Bounding boxes and `attributes.mask_contours` are derived from the actual nonempty
native masks; disconnected components of at most four pixels are ignored for
display geometry. Each contour payload contains `coordinate_system="normalized"`,
`source_frame_sequence`, `source_timestamp_seconds`, and `polygons` as arrays of
`[x, y]` points. Coordinates use six decimal places and are bounded to `[0, 1]`;
geometry is limited to 16 exterior polygons, 128 points per polygon and 1024
points total. Contours are drawn only on their own source frame. Held/interpolated
geometry and Hybrid's detection-only observations fall back to bounding boxes.
The Harness bounds complete video responses to 16 MiB to accommodate masks.

Duplicate IDs,
malformed masks/scores, missing source outputs or an incorrect native frame order
fail the entire response and quarantine the model until restart. No partial batch
is published as successful. A deadline discards all late results while retaining
the shared admission slot until the real GPU task and cleanup finish.

### Resource limits and configuration

All variables below use the prefix **`SAM31_`**. Limits can be lowered but not
raised beyond the table's hard ceilings (timeout can be raised to 120 seconds).

| Suffix | Default / maximum | Purpose |
| --- | --- | --- |
| `ENABLED` | `false` | Explicitly enable the native model |
| `CHECKPOINT_PATH` | `/models/sam3.1_multiplex.pt` | Approved local merged checkpoint |
| `COMPILE_MODE` | `off` | `off` or experimental `encoder`; benchmark CUDA before retaining it |
| `MAX_FRAMES` | 16 | Already received frames per batch; one-frame flush accepted |
| `MAX_REQUEST_BYTES` | 33554432 | Streamed JSON request cap, including base64 overhead |
| `MAX_JPEG_BYTES` | 8388608 | One decoded base64 JPEG payload |
| `MAX_TOTAL_JPEG_BYTES` | 25165824 | Total JPEG payload per request |
| `MAX_IMAGE_PIXELS` | 8294400 | Original pixels per image |
| `MAX_TOTAL_IMAGE_PIXELS` | 67108864 | Total original pixels per batch |
| `INFERENCE_MAX_SIDE` | 1280 | Maximum retained PIL image side before native loading |
| `MAX_OBJECTS` | 16 | Native object limit |
| `REQUEST_TIMEOUT_SECONDS` | 60 / 120 | Upload, decode, inference and cleanup deadline |

Each JPEG is decoded, immediately shrunk, and its full-resolution pixels released
before decoding the next frame; at most 16 bounded PIL images are retained.
The original-pixel batch cap still applies, so 16 full 4K images are not accepted.
The native encoder itself uses 1008-pixel inputs. Native state, mask history and
output buffers remain bounded by this finite batch and object cap; their actual
VRAM footprint depends on frame size, object count and the execution path. See
the isolated GPU continuation measurements below; they are not a capacity guarantee.

`SAM31_COMPILE_MODE=encoder` is an independent, opt-in optimization for both native
batch and continuous paths. After strict loading it applies `torch.compile` only
to the detector ViT trunk's bound `forward` (`inductor`, `mode=default`,
`fullgraph=True`, `dynamic=False`). Module identity, weight names, tensor storage
and complex-valued RoPE remain unchanged; it neither requests FA3 nor enables the
upstream all-component compiler's global function replacements. Existing BF16
autocast is retained. The compiled function executes during real CUDA warmup;
compilation or execution errors fail readiness instead of silently falling back.
Compilation may substantially increase cold startup time and compiler cache use.
The RTX 4090 comparison on the same 512-pixel input and profile 5 measured roughly
2.4 seconds per 16-frame native operation in both modes, with no demonstrated
throughput improvement. Use a writable
compiler cache (the container already provides `/tmp`) and allow startup to finish
before recording steady-state timings. `off` remains the default and comparison.

Native session logs identify the configured `route` and validated `request_id`,
with input/output frame counts and source-sequence ranges, accepted/emitted
watermarks, pending count, EOF/replay flags and outcome/status. Invalid identifiers
are never logged; rejected requests that have not parsed a valid identifier use
`request_id=None`. Source/session identifiers, identity/object IDs, JPEGs and model
output text are not logged. Successful requests use INFO; failures use WARNING.

The append record separates disjoint host wall-clock stages: `validation_seconds`,
`decode_seconds` (including resize), `native_advance_seconds` (iterator advancement
and its existing waits), `snapshot_seconds` (mask/contour conversion, including any
device-to-host wait), `retention_seconds`, `response_build_seconds`,
`response_serialize_seconds`, and `cleanup_seconds`. No CUDA synchronization is
added for measurement. These are **not CUDA kernel timings**. The old coarse
`validation_decode_seconds`, `native_output_seconds`, and
`retention_response_seconds` names remain as aggregates.

The HTTP record uses the same request ID and adds `body_read_seconds`,
`worker_dispatch_seconds`, `processing_seconds`, `http_response_encode_seconds`
and total time through response preparation. `processing_seconds` includes
dispatch/event-loop waiting; it overlaps the append record, so do not sum both.
Client round-trip time additionally includes network work outside this interval.
Replays count zero newly emitted outputs. `output_frames / total_seconds` measures
worker service speed only; sustained analysis FPS must use distinct source frames
and elapsed client completion time, including upload and between-request gaps.

There is one shared admission/GPU slot across **both image and video endpoints**.
Busy requests get 429, oversized requests 413, invalid input 400/415, out-of-order
source frames 409, deadlines 504 and unavailable models 503. There is no remote
session map or automatic queue. A hung CUDA call cannot be safely interrupted;
the slot stays occupied and the process needs restart. Stop the host before
restarting the worker so the host can create fresh state.

### Run the native model

Use the same build command as above. Provide the approved weight at
`/srv/goodvision/models/sam3.1_multiplex.pt`:

```sh
docker run --rm --name goodvision-sam31 --gpus all \
  -p 127.0.0.1:8092:8092 \
  --mount type=bind,source=/srv/goodvision/models,target=/models,readonly \
  -e SAM31_ENABLED=true \
  -e SAM_SCENE_IMAGE_ENABLED=false \
  -e SAM31_CHECKPOINT_PATH=/models/sam3.1_multiplex.pt \
  goodvision-sam-scene
```

`SAM_SCENE_IMAGE_ENABLED=false` avoids loading a second GPU model. The old image
routes remain present but are unavailable in that deployment. Keep one Uvicorn
process. Docker health automatically checks `/health/video` when native mode is
enabled. Missing weights or GPU resources do not trigger downloads or CPU fallback.

For a direct installation inside the verified Vast container, create an isolated
environment from the repository root. This avoids running Docker inside the
rented container and selects the exact cu126 pair:

```sh
python3.12 -m venv .venv-sam31
. .venv-sam31/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r workers/sam_scene/requirements.txt
python -m pip install \
  git+https://github.com/facebookresearch/sam3.git@660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
python -m pip check
```

The system still needs a C/C++ compiler, Git and the shared libraries listed in
the Dockerfile. Then set the native model variables above plus `HF_HUB_OFFLINE=1`,
and run `python -m uvicorn app:app --host 127.0.0.1 --port 8092 --workers 1`
from `workers/sam_scene`. Runtime credentials do not belong in these variables.

After genuine GPU readiness, exercise a chronological sample:

```sh
python workers/sam_scene/video_smoke.py --endpoint http://127.0.0.1:8092 \
  --frame-interval 0.25 --timeout 75 \
  /data/frame000.jpg /data/frame001.jpg /data/frame002.jpg
```

Supply up to 16 same-size JPEGs. The report records GPU readiness, native epoch,
source count, visible observations and latency, always `accuracy_measured=false`.
A zero-observation report is possible for blank or insufficient native evidence;
use labeled person recordings to evaluate usefulness. The smoke check does not
measure identity switches, recall or event accuracy.

Software verification:

```sh
python -m pytest tests/test_sam31_worker.py tests/test_sam_scene_worker.py -q
python -m ruff check workers/sam_scene tests/test_sam31_worker.py tests/test_sam_scene_worker.py
```

These tests use dependency-injected detections and official-API doubles. The
approved merged checkpoint has also passed strict loading and real CUDA inference
on the rental GPU. Labeled multi-person occlusion/identity accuracy and sustained
deployment reliability still need evaluation. Published H100 speedups are not
measurements of this uncompiled worker on a rental GPU.

### 可选连续会话（实验功能）

设置 `SAM31_ENABLED=true`、`SAM31_SESSION_ENABLED=true`，以及
`SAM_SCENE_IMAGE_ENABLED=false`。仍使用同一个本地 `sam3.1_multiplex.pt` 和已验证的
CUDA 模型；不会再加载第二份权重。建议先使用 `SAM31_INFERENCE_MAX_SIDE=640`。
Docker 的健康检查会自动切换到 `/health/video/session`；该接口只有真实 CUDA 模型加载、
连续处理与无新帧 EOF 预热成功后，才会返回 `ready=true`、`gpu_ready=true`。

此路径使用固定源码的 `Sam3MultiplexTrackingProd`，在同一会话中保留原生人物 ID 和
记忆。只允许按时间向前加入已经收到的帧。SAM 的前向处理范围与检测器对 count 的定义
不同，因此适配层分别传入正确范围，避免边界帧错误使用上一帧的检测特征。
输入固定为 16 个槽位；保留最近 32 帧跟踪历史和仍会被注意力使用的 conditioning
锚点，同时删除过期的 mask、对象视图及逐帧元数据。已经清理的帧不再支持反向追踪或
历史交互修改。

2026-09-12 的真实 GPU 对照中，48 帧的一次处理、分段处理、仅新帧到来后追加，
三者逐帧原生 ID 与 mask hash 全等。128 帧的未清理追加与上述清理版本也逐帧全等。
该片段中，清理版本输入 CPU tensor 固定为 97,542,144 字节；第 48 帧以后保留 32 个
non-conditioning 输出、1 个 conditioning 输出和 14 帧待发结果，CUDA allocated
约稳定在 5.92–5.93 GB。未清理版本到 128 帧时 CPU tensor 约 780 MB、CUDA
allocated 约 7.30 GB。这是特定录像和配置的执行/一致性证据，**不表示测过人物分割、
身份切换或事件识别的准确率，也不等于长期生产验收**。

随后 256 帧真实多人画面（源片 6.0–69.8 秒）加入最新元数据清理和空 EOF，
与未清理参考逐帧 ID／mask hash 仍全等；[详细 GPU 记录](../../docs/architecture/sam31-gpu-validation-20260912.md)
保留对应输入、计时和质量问题。

接口顺序：

1. `POST /v1/video/sessions`，JSON 为 `{"stream_session_id":"camera-1"}`。
   保存回复的 `video_session_id` 和 `predictor_epoch`。同时只允许一个活动会话；另一个
   create 返回 409。活动会话期间旧 `/v1/video/batches` 返回 429。
2. `POST /v1/video/sessions/{video_session_id}/append`，JSON 顶层严格为
   `stream_session_id`、`request_id`、`frames`、`is_last_batch`。每个 frame 与 batch
   接口相同：`frame_sequence`、`timestamp_seconds`、`jpeg_base64`。首个非 final 请求
   必须有 16 帧；后续每次 1–16 帧。序号和时间在请求内、跨请求都必须严格递增。
3. 最后一个有帧请求可设置 `is_last_batch=true`。已经初始化的会话也支持
   `frames=[]`、`is_last_batch=true`，直接发出保存的末尾结果，不制造新帧、不重复推理。
   首个请求若已经 final，则接受 1–16 帧，使用有限视频路径完成并关闭。
4. `DELETE /v1/video/sessions/{video_session_id}` 用于取消或断流，回复
   `discarded_pending_frames`。它释放状态并明确丢弃尚未发出的帧，不把它们当作成功证据。

可选 H264 传输只在 `SAM31_SESSION_H264_ENABLED=true` 且 CPU 解码器初始化成功时
启用；health 和创建响应的 `limits.h264_transport` 用整数 `1` 表示支持、`0` 表示关闭。
它复用相同的 append 地址，在原来的四个顶层字段之外增加第五个字段：

```json
{
  "stream_session_id": "camera-1",
  "request_id": "request-2",
  "frames": [{"frame_sequence": 104, "timestamp_seconds": 25.25}],
  "is_last_batch": false,
  "video": {"format": "h264_annexb", "data_base64": "..."}
}
```

此模式的每个 frame 只包含原始序号和时间；不得混入 `jpeg_base64`。每个请求都是独立、
以 IDR 开始、无 B 帧的 Annex-B elementary stream，包含与元数据数量相同且顺序一致的
1–16 帧，支持 baseline/main、8-bit 4:2:0。解码前先检查 SPS、参考帧和尺寸；最长边
不超过 1280，编码字节不超过 8 MiB，且同时受请求总字节和原有总像素预算约束。
只接收严格的原始 H264，不接收容器文件、外部引用或分辨率变更。解码和推理共用原来的
单请求槽与总截止时间；失败不会悄悄切回 JPEG。空 EOF 继续使用四字段 JSON，不带 video。
请求重放 hash 覆盖编码格式、完整 bitstream、所有源帧元数据及 final 标志。
worker 仅暂存本次解码图像，持续状态仍只保留待发源帧元数据；主程序的证据仍对应原始采样帧。
压缩会改变模型看到的像素，因此传输性能和人物分割质量都需要用同一录像对照。

append 回复的 `backend` 是 `sam3.1_video`、`memory_scope` 是 `session`；整个会话的
`predictor_epoch` 保持不变。`frames` 是此前已接收但尚未发出的源帧前缀，可能属于之前
的 append。SAM 默认保留 14 帧确认缓冲，所以第一批 16 帧通常只发出前 2 帧。
`accepted_frame_sequence` / `accepted_timestamp_seconds` 表示已经接收的最后源帧；
`processed_*` 在成功回复中与 accepted 相同；`emitted_*` 表示已经发出的最后源帧，
未发出任何帧时两项一起为 null。`pending_frame_count` 等于已接收但未发出的帧数。
正常 final 回复为 `state=closed`、pending 为 0，emitted 与 accepted 一致。

只缓存最后一次成功请求的 `request_id`、规范化内容 hash 和回复（上限 2 MiB）。相同
ID/内容重试返回缓存；相同 ID 内容改变、旧源帧和错误源会话返回 409。推理异常或超时
使当前 epoch 作废；不得在未知是否已经推进的旧状态上续传。超时后 GPU 槽仍占用到真实
推理与清理结束。不可预料的原生错误会同时取消共享 batch 模型的 readiness，需要重启。

会话额外配置如下，全部有硬上限：

| 环境变量 | 默认值 | 范围/含义 |
| --- | --- | --- |
| `SAM31_SESSION_ENABLED` | `false` | 显式启用实验连续路径 |
| `SAM31_SESSION_H264_ENABLED` | `false` | 显式接受有界 H264 append；要求 PyAV 17.1.0，默认 JPEG 协议仍可用 |
| `SAM31_SESSION_HOTSTART_DELAY` | `15` | 固定 profile：3、5、15；较短 profile 会改变早期去重／淘汰判断，需录像对照 |
| `SAM31_SESSION_POSTPROCESS_BATCH_SIZE` | `1` | 1 或实验 4；四帧一起后处理，不改变检测和跟踪记忆 |
| `SAM31_SESSION_HISTORY_FRAMES` | `32` | 32–64；须覆盖已核验的记忆与确认范围 |
| `SAM31_SESSION_MAX_SESSION_FRAMES` | `65536` | 16–65536；在官方 100000 帧 sentinel 前结束 |
| `SAM31_SESSION_MAX_CUMULATIVE_NATIVE_IDS` | `2048` | 16–2048；限制整个会话的累计原生 ID |
| `SAM31_SESSION_IDLE_SECONDS` | `120` | 5–600；空闲后自动释放会话 |

超过会话帧数或累计 ID 上限、源尺寸改变时，返回 409 `reset_required` 并释放状态。
客户端应建立新会话并明确换 epoch；不会静默复用旧人物编号。单请求的上传、像素、对象
数量和总超时仍服从 `SAM31_` 限额。服务只保留尚未发出的源帧元数据（最多 32 条），
不会保留它们的上传 JPEG。录制、证据保存和事件语义仍由主程序负责。

可单独使用 `SAM31_SESSION_HOTSTART_DELAY=5` 做低延迟对照，默认仍为 15。
三个 profile 的待发上限分别为 2、4、14 帧；health 和创建响应的 `limits` 同时报告
`hotstart_delay` 与 `max_pending_frames`。官方构造器要求早期未匹配／重复淘汰阈值不大于
hotstart，因此这两个阈值设为 `min(8, hotstart_delay)`；连续三次检测的 masklet 确认保持不变。
较短窗口可能改变误检和轨迹淘汰，不能沿用默认 profile 的质量结论。
首个非 final 请求仍须 16 帧，以保留已验证的输入环和注意力启动语义；之后可追加 1–16 帧。
这降低的是持续输出等待，不会把首批所需帧伪造成已经到达。第一批直接 final 的 1–16 帧
同样使用所选 profile，且无需填充未来帧；空 EOF 保持发完实际待发源帧的行为。

`SAM31_SESSION_POSTPROCESS_BATCH_SIZE=4` 只批量处理模型输出，默认仍为 1。
Prod 在每次 append 结束时会发完不足四帧的后处理结果，不把这部分等待带到下一请求；
空 EOF 也按最多四帧一组处理，14 帧尾部拆为 4／4／4／2，避免超过官方
`postprocess_batch_size × max_objects` 的输出缓冲容量。它不启用 batched grounding，
不改变确认窗口、源水位和 HTTP 协议。RTX 4090 上三轮相同 16 帧加空 EOF 的隔离对照，
逐帧原生 ID／mask hash 全等；模式 1 和 4 的平均 native session 耗时分别为
2.22553 秒和 2.22935 秒，没有观察到净提速，因此默认继续为 1。这仅是该录像和配置的结果。

隔离于正在服务的 GPU 推理，可以用以下工具排除网络影响：

```sh
python tools/sam31_postprocess_benchmark.py \
  --checkpoint /models/sam3.1_multiplex.pt \
  --frame-dir /srv/goodvision/sam31/test-media/long-frames \
  --max-side 512 --hotstart-delay 5 --repeats 3 \
  --output /tmp/sam31-postprocess-comparison.json
```

如果远端是平铺的 worker 文件目录，增加 `--worker-dir /path/to/worker`。
工具严格加载同一模型，分别预热两种后处理模式，再交替执行三轮 1／4 对照；每次都以相同
16 张已存在的 JPEG 建立新 native session，再发送空 EOF。报告保存源文件 hash、各阶段
耗时、CUDA allocated/peak 及逐帧 ID／mask hash。JPEG 加载和 hash 计算不计入模型计时，
不调用 HTTP/VLM。`--start-index` 可换用目录内另一组16张，`--budget-seconds` 最多600秒，
这只是协作式截止，不能中断挂起的 CUDA。输出文件须不存在；未提供人工真值，因此仍不报告准确率。

隔离 GPU 对照工具：

```sh
python tools/sam31_stream_probe.py --checkpoint /models/sam3.1_multiplex.pt \
  --chunk-frames 16 --output /tmp/stream-48.json /data/frame{000..047}.jpg
python tools/sam31_stream_prune_probe.py --checkpoint /models/sam3.1_multiplex.pt \
  --history-frames 32 --empty-final --output /tmp/stream-128.json \
  /data/frame{000..127}.jpg
```

工具最多接收 256 帧、64 MiB JPEG；为避免同时驻留两份模型，测试时暂停 serving 进程。
`--empty-final` 专门对照有帧 final 与无新帧 EOF 的输出。后续验收还需要真实长录像中的
进入/离开、遮挡、交叉、断网、取消和跨 epoch 恢复，以及人工标注的准确率评估。

## SAM 连续分析

`perception_routes.py` 是插件目录；每项仅提供后端工厂和不可变 `SessionProfile`。
`continuous.py` 统一负责请求限额、JPEG/H264 解码、源帧映射、截止时间、幂等重试、
取消、会话关闭及状态清理。生产目录仅挂载 `sam_live`。

| 路线 | 独立实现 | 启用变量 | 输出后端 |
| --- | --- | --- | --- |
| `sam_live` | `sam_live.py` | `SAM31_LIVE_ENABLED=true` | `sam3.1_video` |

启用变量默认 `false`。本地 Compose 和远程启动器显式启用 `sam_live`，只需要
经过校验的本地 SAM 权重。历史 hybrid 实验实现不在生产镜像、上传列表或接口目录中。

接口目录为 `GET /v1/perception/routes`，返回协议版本、路由名称、各自 readiness、
后端、端点和限额。单条路线接口如下：

```text
GET    /health/perception/{route}
POST   /v1/perception/{route}/sessions
POST   /v1/perception/{route}/sessions/{video_session_id}/append
DELETE /v1/perception/{route}/sessions/{video_session_id}
```

创建／append／删除请求沿用现有有界会话协议。新路线的 `startup_frames=1`、
`max_append_frames=4`、`hotstart_delay=1`、`max_pending_frames=0`，其中
`hotstart_delay` 表示对外交付等待，不能据此认为一个人物已通过内部三次检测确认。
客户端空闲时可立即发出一帧；已撤回“推理期间只保留最新一个待处理帧”的修改，
恢复此前四帧待发队列，等待帧随后可合批发送。worker 仍允许最多四帧追加。
原有队列溢出丢弃策略仍存在，恢复容量不代表零丢帧，也不是最终吞吐方案；
需要结合队列、源帧覆盖率和分段耗时日志解决过载，不能靠丢关键帧降低延迟。
HTTP 仍按请求返回完整 JSON，尚未实现逐帧 HTTP 分块输出。
旧 `/v1/video/sessions` 保留原有 16 帧启动和确认缓冲协议。

新接口额外返回 `route_metadata`，与严格源帧对象协议分开：

```json
{
  "route": "sam_live",
  "confirmation_policy": "causal_native_three_detections",
  "observations": [
    {"frame_sequence": 12, "provisional_object_ids": ["4"]}
  ]
}
```

`sam_live` 保留原生三次检测确认及五帧早期淘汰规则，但只用当前及历史信息立即交付
真实观测；暂未确认的人物明确标为 provisional。它不读取未来确认状态，也不回填或
伪造源帧。首帧初始化仅安装文本提示，避免官方 `add_prompt` 先推理第零帧、后续又
推理同一帧造成重复确认计数。输入环提前建立空的地址槽，只有真实到达的图像才能
填入；未到达或过期的绝对源索引读取失败。路线启用后，CUDA warmup 必须实际走完
首帧、四帧追加、输入环回绕、历史裁剪及空 EOF，才能报告 GPU ready。

整个进程只允许一个活动会话，覆盖旧协议和 SAM 连续分析。更换会话前须停止并关闭旧会话；
新路线得到新的会话 ID 和 predictor epoch，不能延续其他算法的内部编号。活动会话
期间，独立图片／批量视频入口也被阻止，避免并发使用共享 SAM。
新路线 health 的 `active_session` 报告全局占用，包含其他路线仍在关闭的状态；旧
`/health/video/session` 保持原有语义。即使模型已失去 readiness，DELETE
仍可在原有 admission、身份及截止时间保护下清理会话，避免失效状态无法释放。

本地协议、状态及隔离测试使用替身；SAM 连续分析的真实 GPU 延迟、识别质量、
长时运行和交叉遮挡效果需要独立实测，不能沿用旧协议的实测数据。
