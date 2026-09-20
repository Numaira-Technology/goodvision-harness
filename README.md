# Goodvision Harness

An independent service that pulls a robot's live WHEP/WebRTC stream and detects
any selected subset of `fight`, `smoking`, `fall`, `weapon`, and `stranger`.
It publishes structured observations, source-correlated images, continuous
annotated video, and event video evidence. It has no runtime dependency on the
original `goodvision` repository.

## Start here

| Task | Read |
| --- | --- |
| Install a customer delivery | [Deployment guide](DEPLOY.md) and [release record](RELEASE.md) |
| Integrate a receiver | [Public API contract](API_DELIVERY.md) |
| Develop or review changes | [Contributor guide](CONTRIBUTING.md) |
| Find detailed guides and past results | [Documentation index](docs/README.md) |
| Run operational or evaluation tools | [Tool inventory](tools/README.md) |

## Quick start

From the repository root, start a local dry-run console without model calls or
face weights:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,webrtc]'
HARNESS_AUTOSTART=false HARNESS_DRY_RUN=true \
  .venv/bin/goodvision-harness-demo --host 127.0.0.1 --port 7900
```

Open `http://127.0.0.1:7900`, choose Camera, enable it, and press Start. Dry-run
results are synthetic. For real recognition, follow the
[local development](CONTRIBUTING.md#local-development) to install face
dependencies and provision authorized weights, and configure the selected
[inference route](docs/guides/configuration.md). Windows setup is covered in the
[contributor guide](CONTRIBUTING.md) and [hybrid deployment guide](WINDOWS_HYBRID.md).

The display runs at up to 30 FPS on original source frames while face and event
inference keep their independent sampling. Face boxes and event boxes are drawn
independently; intervening frames hold the preceding annotations. See the
[runtime guide](docs/guides/runtime.md#display-and-evidence) for display, clearing
and exact-evidence behavior.

## Continuous scene analysis and multi-camera deployment

`SCENE_ENGINE_MODE=active` enables persistent people, source-aligned SAM contours,
temporal event evidence and ordered face identity updates. The default `disabled`
mode keeps the existing per-frame workflow. In fleet deployments, connect each
preview independently, then start or stop analysis for all connected cameras.

- [Continuous scene operation and validation limits](docs/guides/continuous-scene.md)
- [Local NVIDIA GPU deployment](docs/local-gpu.md)
- [Separate application, tracking and model services](docs/modular-services.md)
- [Multi-camera API](docs/multi-camera-api.md)

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/goodvision_harness/` | Application, API, video pipeline, event archive, and console |
| `workers/` | SAM scene, standalone Contour and Paddle rich-label workers |
| `tests/` | Offline unit and integration checks |
| `schemas/` | Versioned analysis and event contracts |
| `tools/` | Deployment, diagnostics, benchmarking, and evaluation entrypoints |
| `docs/guides/` | Runtime, configuration, deployment, and release guides |
| `docs/architecture/` | Design and implementation notes |
| `docs/validation/` | Dated verification reports and curated aggregate results |
| `docs/internal/` | Internal reference material |
| Root `compose*.yaml`, `Dockerfile*`, and config examples | Stable build and deployment entrypoints |

Keep credentials, model weights, face galleries, recordings, build outputs, and
local worktrees outside Git. See [repository hygiene](CONTRIBUTING.md).

<!-- Preserve existing README heading links; detailed content is indexed in docs/README.md. -->
<a id="harness-behavior"></a>
<a id="optional-rich-video-labels"></a>
<a id="optional-gpu-face-recognition"></a>
<a id="experimental-vlm-rich-annotation"></a>
<a id="lan-deployment-topology"></a>
<a id="server-prerequisites"></a>
<a id="deploy"></a>
<a id="customer-server-operations"></a>
<a id="zero-cost-smoke-mode"></a>
<a id="selecting-detection-features"></a>
<a id="output-endpoints"></a>
<a id="analysis-result-contract"></a>
<a id="optional-webhook"></a>
<a id="configuration"></a>
<a id="切换-ai-渠道和模型"></a>
<a id="local-development"></a>
<a id="known-person-gallery"></a>
<a id="how-the-memory-pieces-fit-together"></a>
<a id="customer-release-builds"></a>
<a id="customer-facing-boundary"></a>
<a id="release-staging-and-verification"></a>
<a id="verification"></a>

Detailed runtime, configuration, build and validation references are maintained
in the [documentation index](docs/README.md).
