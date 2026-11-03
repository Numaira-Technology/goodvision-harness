# 本地运行与客户交付清单

> 本地源码运行与客户交付共用正式控制台，差异在部署文件、依赖和配置。
> 原模拟分析模式及独立演示入口已移除。本页保留原链接以兼容已有文档。
>
> 权威依据：[`README.md`](../../README.md)（"Start here" 表 + Quick start + Repository layout）
> 与 [`CONTRIBUTING.md`](../../CONTRIBUTING.md)（"What belongs in Git" / "Organization and stable paths"）。

## 先厘清一个关键认知

**本地运行和客户交付使用同一套代码。** README / CONTRIBUTING 明确：

- `goodvision-harness` 是统一的正式控制台入口，默认端口 `8080`。
  启动控制台不会自动开始分析，缺少服务或凭据时会报错，不生成模拟识别结果。
- 根目录的 Compose / Dockerfile / env 示例 / 客户交付文档是 **"脚本、测试、运维共用的稳定入口"**，
  移动前须审查其消费者（CONTRIBUTING "Organization and stable paths"）。
- `config.yaml` 与 `src/goodvision_harness/config.yaml` **必须逐字节一致**，有回归测试强制。

> 选择部署方式使用相应的启动器、Compose 文件和配置。
> 按环境长期维护分支会增加共用代码的同步负担。分支本身不会破坏配置一致性；
> 无论在哪个分支，目录整理都应保留稳定入口，并同步修改两份 `config.yaml`。

## 启动方式

```bash
# 本地正式控制台（先配置服务，再手动开始分析）
HARNESS_AUTOSTART=false HARNESS_HOST=127.0.0.1 goodvision-harness

# production / 客户交付
python tools/start_delivery.py            # 见 DEPLOY.md / RELEASE.md
```

---

## A. Production / 客户交付（生产环境的东西）

README "Install a customer delivery" 路径所指的一套：

| 文件 / 入口 | 角色 |
|---|---|
| `DEPLOY.md` | 客户交付部署指南（"Install a customer delivery"） |
| `RELEASE.md` | 发布记录 / 交付内容清单 |
| `API_DELIVERY.md` | 对接方公开 API 契约（"Integrate a receiver"） |
| `compose.full-gpu.delivery.yaml` | 交付部署 compose |
| `config.delivery.yaml` | 交付配置 |
| `.env.full-gpu.delivery.example` | 交付环境变量示例 |
| `Dockerfile.release` | 发布/交付镜像 |
| `tools/start_delivery.py` · `tools/delivery_smoke.py` · `tools/run_full_delivery_smoke.ps1` | 交付启动与冒烟 |
| `tools/export_full_delivery.py` · `tools/package_customer_delivery.py` · `tools/package_customer_update.py` · `tools/package_customer_docker_update.py` | 交付/客户更新打包 |
| `tools/audit_customer_bundle.py` · `tools/customer_check.py` · `tools/customer_discovery.py` · `tools/customer_runtime.py` · `tools/customer_update.py` · `tools/customer_update_gui.py` | 客户侧审计/运行/更新 |

## B. 本地源码运行

| 文件 / 入口 | 角色 |
|---|---|
| `goodvision-harness`（控制台入口，默认端口 8080） | 正式控制台；默认在 `.goodvision/` 持久保存设置 |
| `compose.yaml` | 基础栈，默认网络名 `goodvision-demo` |
| `config.yaml` · `config.vlm-rich.yaml` | 开发/富标注配置（`config.yaml` 与包内副本须一致） |
| `.env.example` | 开发环境变量示例 |
| `Dockerfile` · `Dockerfile.face-gpu` | 开发 / 人脸 GPU 镜像 |
| `WINDOWS_HYBRID.md` · `tools/start_windows_hybrid_demo.ps1` | Windows 非 NVIDIA 实验与演示 |

## C. Shared / 两个环境共用核心（不归任一环境）

README "Repository layout" 列的主体，本地运行与客户交付共用：

| 路径 | 角色 |
|---|---|
| `src/goodvision_harness/` | 应用、API、视频流水线、事件存档、控制台 |
| `workers/` | SAM scene / Contour / Paddle rich-label worker |
| `tests/` | 离线单测与集成检查 |
| `schemas/` | 版本化分析与事件契约 |
| `tools/`（除 A 列的交付工具外） | 部署/诊断/基准/评测通用入口（见 `tools/README.md`） |
| `docs/` | 指南 / 参考 / 架构 / 验证 |
| `tools/event_benchmark/` | 当前 main 的冻结事件评测工具，与部署环境正交；后续目录迁移以合并后的实际路径为准 |
| `pyproject.toml` · `.github/` | 构建与 CI |

### B/C 之间的模块化 overlay（按需叠加，非环境专属）

这些 Compose 文件按部署拓扑选择，也并非全部都要与
`compose.yaml` 叠加。`compose.gpu.yaml`、`compose.face-gpu.yaml` 等提供 overlay；
`compose.main-server.yaml` 通过 `extends` 引用基础服务；`compose.vlm-server.yaml`、
`compose.analysis-gateway.yaml` 和 `compose.full-gpu.delivery.yaml` 是各自的部署入口。
使用对应启动器或指南给出的 Compose 文件组合，见[源码部署](deployment.md)与[分机部署](modular-services.md)。

## Compose 文件分组索引

根目录的 `compose*.yaml` 是**扁平的稳定入口**，被脚本、测试和运维按原路径使用，
见 [CONTRIBUTING "Organization and stable paths"](../../CONTRIBUTING.md#organization-and-stable-paths)。
这里只按用途分组，保留原路径。下表区分独立入口与 overlay；“叠加”表示在所列文件之后再加本行文件，
按顺序传入 `-f`。独立入口无需额外传入 `compose.yaml`，但仍可能通过 `extends` 引用其他文件。
具体环境变量、权重与启动参数以对应指南、启动器和文件内说明为准。

| 组 | 文件 | 用途 | 组合 / 启动入口 |
|---|---|---|---|
| 基础 | [`compose.yaml`](../../compose.yaml) | 源码基础栈，默认网络 `goodvision-demo` | 独立入口；见[源码部署](deployment.md) |
| 应用主机 (CPU) | [`compose.main-server.yaml`](../../compose.main-server.yaml) | CPU 应用主机，SAM / VLM 可在独立服务器 | 独立入口，通过 `extends` 继承基础服务；`python tools/start_modular.py main`，见[分机部署](modular-services.md) |
| | [`compose.analysis-gateway.yaml`](../../compose.analysis-gateway.yaml) | 仅接入 / 转发控制台、视频和输出流量 | 独立网关入口，需配置远端分析地址；见[分机部署](modular-services.md) |
| 本地 GPU | [`compose.local-tracking.yaml`](../../compose.local-tracking.yaml) | 主程序 + 本机 Contour/SAM 追踪；人脸仍用 CPU，大模型使用已配置的外部服务 | 独立入口；`python tools/start_local_gpu.py`，见[本机追踪](local-gpu.md) |
| | [`compose.local-gpu.yaml`](../../compose.local-gpu.yaml) | 本地 VLM、CPU 处理服务、SAM 与 GPU 人脸；Paddle 可选 | 叠加在 `compose.yaml` 上，不与其他 GPU / remote overlay 混用；`python tools/start_local_gpu.py --with-local-inference`，见[完整本地部署](local-gpu-full.md) |
| 源码 GPU 加速 | [`compose.gpu.yaml`](../../compose.gpu.yaml) | Paddle rich-label worker GPU overlay | 叠加在 `compose.yaml` 上，启用 `--profile rich`；见[运行指南](runtime.md) |
| | [`compose.face-gpu.yaml`](../../compose.face-gpu.yaml) | 使用 `Dockerfile.face-gpu` 启用 GPU 人脸识别 | 叠加在 `compose.yaml` 上；见[运行指南](runtime.md#optional-gpu-face-recognition) |
| 云 / 远端 GPU | [`compose.cloud-gpu.yaml`](../../compose.cloud-gpu.yaml) | 本地网关 + UI 控制的云 GPU 租用，分析在租用主机运行 | 独立入口；`python tools/start_cloud_gpu.py`，见[云 GPU 部署](cloud-gpu.md) |
| | [`compose.sam31-remote.yaml`](../../compose.sam31-remote.yaml) | 网关 + 云 GPU 租用控制器，UI 明确启动租用后运行远端分析 | 叠加在 `compose.yaml` 上，替代本地 SAM overlay；见[云 GPU 部署](cloud-gpu.md) |
| | [`compose.detector-remote.yaml`](../../compose.detector-remote.yaml) | 租 GPU 跑 YOLO / RF-DETR 检测器 | 按文件内示例与 `compose.yaml` 组合，只启动 `detector-remote`；启动该服务即租用付费实例 |
| SAM 服务 / 验证 | [`compose.sam-server.yaml`](../../compose.sam-server.yaml) | 独立 SAM 服务，权重位于该 NVIDIA GPU 主机 | 独立入口；`python tools/start_modular.py sam`，见[分机部署](modular-services.md) |
| | [`compose.sam31.yaml`](../../compose.sam31.yaml) | 原生 SAM 3.1 视频验证，每个有界批次独立模型记忆 | 叠加在 `compose.yaml` 上；需本地 SAM3.1 权重 |
| | [`compose.sam31-stream.yaml`](../../compose.sam31-stream.yaml) | 实验性持久原生记忆 | 叠加在 `compose.yaml` + `compose.sam31.yaml` 后 |
| 场景 / 感知 | [`compose.scene.yaml`](../../compose.scene.yaml) | 连续人物 / 事件引擎 | 叠加在 `compose.yaml` 上；需本地 SAM3 权重 |
| | [`compose.perception.yaml`](../../compose.perception.yaml) | SAM 连续分析 | 叠加在 `compose.yaml` + `compose.sam31.yaml` 后 |
| 独立工作流模块 | [`compose.modules.yaml`](../../compose.modules.yaml) | Bbox / System 1 独立进程、模型卷与健康检查，共用版本化 GPU 底座 | 独立入口；`--profile bbox` / `--profile system1`，见[模块部署](workflow-module-deployment.md) |
| 大模型服务 | [`compose.vlm-server.yaml`](../../compose.vlm-server.yaml) | NVIDIA GPU 模型服务 + CPU 处理服务 | 独立入口；`python tools/start_modular.py vlm`，见[分机部署](modular-services.md) |
| **交付 / production** | [`compose.full-gpu.delivery.yaml`](../../compose.full-gpu.delivery.yaml) | 客户交付镜像入口，默认 CPU 人脸、关闭 rich-label | 独立入口；`python tools/start_delivery.py`，见 [DEPLOY.md](../../DEPLOY.md) / [RELEASE.md](../../RELEASE.md) |
| | [`compose.full-gpu.accelerated.yaml`](../../compose.full-gpu.accelerated.yaml) | 交付镜像的 GPU 人脸 + rich-label 加速 | 叠加在 `compose.full-gpu.delivery.yaml` 上，启用 `--profile gpu`；由交付启动器探测 GPU 后选用 |
| 诊断（在 `diag/`） | [`diag/compose.sam-tunnel.yaml`](../../diag/compose.sam-tunnel.yaml) | 向本机开放远端 SAM 隧道端口 | 叠加在 `compose.yaml` + `compose.sam31-remote.yaml` 后 |
| | [`diag/compose.sam-multistream.yaml`](../../diag/compose.sam-multistream.yaml) | 多路 GPU 竞争诊断，设置共享模型的摄像头路数 | 再叠加在上述诊断隧道组合之后 |

> 配套配置：[`config.yaml`](../../config.yaml)（源码默认，须与
> [`src/goodvision_harness/config.yaml`](../../src/goodvision_harness/config.yaml) 逐字节一致）、
> [`config.vlm-rich.yaml`](../../config.vlm-rich.yaml)（富标注）、[`config.delivery.yaml`](../../config.delivery.yaml)（交付）。

## D. 本地运行数据与已审阅夹具

凭据、模型权重、人脸库、录像和生成输出不应提交；以 `CONTRIBUTING.md` 的目录约定与
`.gitignore` 为准。不要把整个 `video_samples/` 视为未跟踪目录：当前仓库保留了
`sample_video.annotation.json` 和 `sample_video.annotation.md` 两个已跟踪标注文件。
检查某个本地文件时使用 `git ls-files` 与 `git check-ignore -v`，不能从文件名推断跟踪状态。

---

## 结论与建议

1. **本地运行与交付共用代码。** 按实际硬件选择启动器、依赖和配置。
2. 本地控制台与交付控制台均使用真实分析服务，缺少配置时明确报错。
3. **根部署文件（Compose / Dockerfile / env / 交付文档）不要擅自移动**；若日后仍想做物理归组，
   须先审查脚本/测试/运维消费者，并保留 `config.yaml` 一致性不变量。
