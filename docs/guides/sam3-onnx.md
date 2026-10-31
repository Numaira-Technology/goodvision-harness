# Optional SAM3 ONNX person segmentation

This component adds single-image, text-prompted person detection and segmentation
through OSAM 0.9.4. It feeds the existing detector worker's per-session IoU tracker
and `/v1/labels` / `/v1/tracks` contracts. The prompt is fixed to `person`, matching
the existing person-tracking pipeline. The console option is `单帧分割(可选)`
(`det-segment`; worker model `sam3-onnx`). Existing configuration defaults,
YOLO/RF-DETR paths, `sam_live`, and event/VLM inference are unchanged.

This is supplemental image segmentation. It has no SAM video memory, native SAM
IDs, pedestrian attributes, or event reasoning. IoU IDs can change after occlusion,
model switches, resolution changes, session eviction, or a worker restart.
The existing tracker also expires tracks after a three-second source-time gap;
slow inference can therefore lose ID continuity. It is
not an equivalent replacement for continuous `sam_live` tracking.

## Install separately

Run from a source checkout with Python 3.11+ (required for CUDA):

```bash
python3.12 -m venv .venv-sam3
source .venv-sam3/bin/activate
python -m pip install -e '.[sam3-onnx]'
```

The `sam3-onnx` extra installs OSAM's CPU build of ONNX Runtime, which also
contains CoreML on macOS. On an existing NVIDIA host, use a separate environment
and `.[sam3-onnx-cuda]` instead. Do not install both extras: `onnxruntime` and
`onnxruntime-gpu` overwrite the same files. This component has no dependency on
PyTorch, Ultralytics, or RF-DETR. Its optional packages are not added to the default
worker image or paid GPU launcher; the commands here launch an already provisioned
host directly. No GPU rental or deployment is performed by these commands.

## Provision authorized weights

The integration code and upstream OSAM / sam3-onnx code use MIT licenses. **Model
weights have the separate Meta SAM License.** Review and obtain any required
approval before acquiring them. Installing this extra does not download weights
or accept model terms; the worker never calls OSAM's model downloader, including
at startup, during health checks, or on an inference request. Do not put weights
in Git or a redistributable image.

OSAM 0.9.4 pins [this SAM3 export](https://github.com/wkentaro/sam3-onnx/tree/f921702fb17fe6f0c6dd054519df71a1c407b142)
and [model revision 40ffee63](https://huggingface.co/wkentaro/sam3-onnx-models-v0.6.0/tree/40ffee63eb38bc3779318319bf81e9db06687d91).
The [model license](https://huggingface.co/wkentaro/sam3-onnx-models-v0.6.0/blob/40ffee63eb38bc3779318319bf81e9db06687d91/LICENSE)
is independent of the source-code license. Stage the following already authorized
files in one local directory, preserving filenames:

| File | SHA-256 |
| --- | --- |
| `sam3_image_encoder.onnx` | `c6b984d95ca7ca0564356253b92ee30287188cfffab5d4f87709b79d5d9aa6c8` |
| `sam3_language_encoder.onnx` | `7868036ee14bb4f09672849c86d6fd5df43f06f089089a99fc2665cb0e84fc26` |
| `sam3_decoder.onnx` | `b4f3588d299611546d9aeb1abf2cf4651a78c7cc021668f379d211b80db1cba3` |

The adapter verifies all three hashes before opening any model session. The point
decoder is unused and need not be staged. Older exports are intentionally rejected.
OSAM is version-pinned because the adapter overrides its private blob-loading
boundary; upgrades require contract tests and a review of files, hashes and API.

## Launch and connect

For a Mac with staged weights, from the checkout root:

```bash
DETECTOR_MODEL=sam3-onnx \
SAM3_ONNX_DEVICE=coreml \
SAM3_ONNX_MODEL_DIR="$PWD/.models/sam3-onnx" \
SAM3_ONNX_CACHE_DIR="$PWD/.cache/sam3-onnx" \
python -m uvicorn workers.detector_rich_label.app:app \
  --host 127.0.0.1 --port 8092 --workers 1
```

Use `SAM3_ONNX_DEVICE=cuda` on a provisioned NVIDIA host or `cpu` for diagnostics.
Only these explicit devices are accepted; there is no `auto` or silent CPU
fallback. CoreML may rewrite graphs and compile caches into the writable cache
directory. Weights can be mounted read-only. Keep one worker process to avoid
loading multiple copies and splitting session tracking state.

Check readiness after startup warmup:

```bash
curl -s http://127.0.0.1:8092/health/ready
```

A successful response includes `ready: true`, `provider_ready: true`, the configured
`device`, and `execution_providers` for `image_encoder`, `language_encoder`, and
`decoder`. The selected provider must be first in every session. `gpu_ready`
retains its CUDA-only meaning; CoreML and explicit CPU readiness use the new
provider contract. Provider presence does not prove that every operator ran on
that accelerator: ORT can partition unsupported operators onto CPU. Runtime
provider-failure fallback is disabled. `last_inference_ms` measures the last
successful detector call (initial warmup also includes loading), including image
encoding and mask processing, and
excludes HTTP transport, queueing, rendering, and other Harness branches.

Point a separate Harness instance at this worker using the existing backend:

```bash
RICH_LABEL_BACKEND=paddle \
RICH_LABEL_ENDPOINT=http://127.0.0.1:8092/v1/labels \
RICH_LABEL_ENABLED=true PERSON_TRACKING_ENABLED=true \
RICH_LABEL_TIMEOUT_SECONDS=30 RICH_LABEL_MAX_FPS=1 \
HARNESS_AUTOSTART=false goodvision-harness
```

These are initial diagnostic settings, not a latency recommendation. Existing
saved deployment settings can override the tracking location: in the console,
select the optional single-frame segmentation model and use this worker's address
as the tracking server. The selection persists as `rich_label_model=det-segment`
and sends `X-GoodVision-Detector-Model: sam3-onnx` on both health and inference
requests. When no selection header is supplied, the worker uses `DETECTOR_MODEL`.
A selected SAM3 model cannot borrow readiness from the worker's default detector.
First selection can trigger local loading/warmup; an initial Harness health probe
may time out and retry while warmup finishes.

| Worker setting | Default | Meaning |
| --- | --- | --- |
| `SAM3_ONNX_MODEL_DIR` | required | Directory containing the verified local graphs |
| `SAM3_ONNX_DEVICE` | `cuda` | `cuda`, `coreml`, or `cpu`; CUDA requires Python 3.11+ |
| `SAM3_ONNX_CACHE_DIR` | `<model dir>/.coreml` | Writable CoreML rewrite/compilation cache |
| `SAM3_ONNX_SCORE_THRESHOLD` | `0.5` | Finite confidence threshold in `[0,1]` |
| `SAM3_ONNX_MAX_OBJECTS` | `64` | Integer from 1 to 256 |
| `SAM3_ONNX_MASKS` | `true` | `false` emits boxes only; it does not skip model mask decoding |

Worker busy responses are HTTP 429, invalid request/source metadata is 400, and
inference failures are 503. Health returns `ready: false` with an `error` for
missing dependencies, missing/corrupt files, unavailable providers, or failed
warmup. After fixing dependencies/configuration, restart the worker. Failed frames
do not advance the session's source sequence. No unbounded inference queue is
created: one inference runs at a time, concurrent requests are rejected, and the
Harness coordinator retains only the newest pending frame.

## Geometry and freshness

OSAM's inclusive pixel boxes become Harness half-open boxes, normalized against
the original frame dimensions. Cropped masks become external contour polygons in
the same original coordinate system. The payload is bounded to 16 polygons,
128 points per polygon and 1024 total points per object. Holes are not represented;
small/degenerate outlines may be omitted while retaining the box.

Only observed tracks carry `attributes.mask_contours`, with the exact source
sequence and timestamp. Predicted/held tracker boxes do not reuse a previous
frame's mask, and existing Harness interpolation removes masks. Original frame
metadata is echoed and validated by the HTTP client. Measure source-to-display
age and dropped frames with representative video before using this path for fresh
live overlays; increasing the timeout does not make inference faster.

The [upstream benchmark snapshot](https://github.com/wkentaro/osam/blob/bd0c9fdf43344df9d43d298006ed290d49237a08/docs/benchmark.md)
reports image encoding separately from prompt execution. In that snapshot, serial
new-image encoding plus one text prompt is about **0.43 s on RTX 3080 Ti** and
**2.80 s on Apple M5 CoreML**, excluding model loading and Harness overhead. The
approximately 0.08 s CUDA prompt number reuses an embedding; it is not new-frame
or end-to-end video performance. These are upstream reference measurements, not
results for this component or its newer pinned export.

## Validation and rollback

```bash
python -m pip install -e '.[dev,sam3-onnx]'
python -m pytest tests/test_sam3_onnx.py tests/test_sam3_onnx_package.py \
  workers/detector_rich_label/test_contract.py
```

Tests cover RGB conversion, inclusive boxes, source-aligned bounded masks,
observed/predicted separation, independent sessions, resolution changes, stale
metadata, backpressure, provider mismatch, recovery, offline file verification,
and the real pinned OSAM preprocessing/decoding API with synthetic session outputs.
The optional package tests skip when OSAM is absent; CI installs it explicitly in
a dedicated contract job. Synthetic tests do not establish model accuracy,
accelerator execution or throughput. Actual SAM3 inference and representative
video latency/quality have not yet been validated for this component.

To roll back an opt-in deployment, stop this worker, restore the prior tracking
server/model selection and remove these environment overrides. Existing defaults
need no rollback. The feature can also be removed by reverting its Git commit;
no database migration, deployment, license acceptance or weight redistribution is
part of this change.
