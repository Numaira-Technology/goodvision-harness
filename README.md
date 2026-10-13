# Goodvision Harness

An independent service that pulls a robot's live WHEP/WebRTC stream and detects
any selected subset of `fight`, `smoking`, `fall`, `weapon`, and `stranger`.
It publishes structured observations, source-correlated images, continuous
annotated video, and event video evidence. It has no runtime dependency on the
original `goodvision` repository.

`python tools/start_local_gpu.py` 自动检测本机 NVIDIA GPU，使用与 Contour 相同的 SAM 引擎运行人物追踪，大模型继续使用云端 API。
本地 NVIDIA 部署**默认 1 路，使用 `--streams N` 按实际需要启动正整数路数**，
12 路是容量参考值，不是软件上限。3、6、12、16 路示例见 [本地 GPU 部署说明](docs/guides/local-gpu.md)。
同一控制台可以分别连接和并排显示已部署的 N 路视频；连接只预览，分析通过“开始全部分析”
和“停止全部分析”统一控制所有已连接的视频。停止分析后预览继续，各路都可展开详细控制，
独立保存检测项目、敏感度并查看结果。分析运行和收尾期间锁定视频连接、各路设置与共享模型配置。
人物追踪使用本机 GPU 或
自备 GPU 服务器均可多路，大模型仍可用云端。受管云 GPU 已支持 N 路视频和 UI 修改租用型号、数量，
通过 `python tools/start_cloud_gpu.py --streams N` 启动，详见[云 GPU 多路追踪](docs/guides/cloud-gpu.md)。
普通旧版外部单会话追踪服务保留兼容入口。
云 GPU 和独立 GPU 部署将人脸识别、事件处理与证据生成放到 GPU 主机，使用该主机的 CPU。
摄像头默认仍由本机拉流并进行必要的接入解码，再转发到 GPU 主机分析；控制台的
“摄像头拉流位置”可选择 GPU 主机直接拉流。远端不可用时不会回退到本机分析。
独立服务器先在 GPU 主机启动 `start_modular.py analysis`，再在本机启动 `main`，
见 [模块化部署说明](docs/guides/modular-services.md)。
主程序、人物追踪、大模型可以分别部署在不同机器，主程序无需显卡；控制台分别设置运行位置、
服务器地址和追踪容量，见 [模块化部署说明](docs/guides/modular-services.md)。
增加 `--with-local-inference` 可同时部署本地大模型和 GPU 人脸，支持单卡共用、双卡分配。
本地大模型部署同时启动 CPU 处理服务，把图片解码、缩放、提示词构建和回复校验放到模型机器。
已有 OpenAI 兼容端点继续支持；自备 GoodVision 模型服务器可在控制台选择对应协议，
或使用 `--inference-protocol goodvision` 启用 CPU 委派，见[分机部署说明](docs/guides/modular-services.md)。
每路大模型分析目标沿用 1.25 次/秒，N 路共 1.25 × N
请求/秒，默认追踪共 2 × N 帧/秒；12 路参考目标为 15 请求/秒。
容量验收需要真实 L40S 上的持续压测，部署成功不代表已达到该吞吐。

## Start here

控制台按「实时工作台、能力与规则、事件与回放、输出与集成、部署与设置」组织，
现有功能位置与操作流程见 [控制台使用说明](docs/guides/console.md)。

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
- [Local NVIDIA GPU deployment](docs/guides/local-gpu.md)
- [Separate application, tracking and model services](docs/guides/modular-services.md)
- [Multi-camera API](docs/reference/multi-camera-api.md)

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/goodvision_harness/` | Application, API, video pipeline, event archive, and console |
| `workers/` | SAM scene, standalone Contour and Paddle rich-label workers |
| `tests/` | Offline unit and integration checks |
| `schemas/` | Versioned analysis and event contracts |
| `tools/` | Deployment, diagnostics, benchmarking, and evaluation entrypoints |
| `docs/guides/` | Runtime, configuration, deployment, and release guides |
| `docs/reference/` | Specialized API reference documentation |
| `docs/architecture/` | Design and implementation notes |
| `docs/validation/` | Maintained validation conclusions, open acceptance gaps and essential evidence |
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
