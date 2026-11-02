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

根目录的 `compose*.yaml` 是**扁平的稳定入口**（被 150+ 处测试/脚本/文档/Dockerfile 按原路径引用，
见 CONTRIBUTING "Organization and stable paths"），**不移动**；这里只按用途分组，便于查找。用法是
`docker compose -f compose.yaml -f compose.<overlay>.yaml ...`，或用对应启动器（`tools/start_*.py`）。

| 组 | 文件 | 用途 |
|---|---|---|
| 基础 | `compose.yaml` | 基础栈（网络 `goodvision-demo`），其余按需叠加其上 |
| 应用主机 (CPU) | `compose.main-server.yaml` | CPU 应用主机（`start_modular main`，无需显卡） |
| | `compose.analysis-gateway.yaml` | 仅转发 console / camera / output 流量 |
| 本地 GPU 追踪 | `compose.local-gpu.yaml` | 本地 NVIDIA GPU overlay（`start_local_gpu` 填设备；只叠在 `compose.yaml` 上） |
| | `compose.local-tracking.yaml` | 应用主机本地 GPU 上的 SAM 运行时 |
| | `compose.gpu.yaml` | Paddle rich-label worker 的 GPU overlay |
| 人脸 / 全功能加速 | `compose.face-gpu.yaml` | GPU 人脸识别（`Dockerfile.face-gpu`） |
| | `compose.full-gpu.accelerated.yaml` | 全功能 GPU 加速（`FACE_ACCELERATOR=gpu` + rich-label） |
| 云 / 远端 GPU | `compose.cloud-gpu.yaml` | 本地网关 + UI 控制的云 GPU 租用 |
| | `compose.detector-remote.yaml` | 租 GPU 跑 YOLO / RF-DETR 检测器 |
| SAM 服务 / 验证 | `compose.sam-server.yaml` | 独立 SAM 服务（权重属该 GPU 主机） |
| | `compose.sam31.yaml` | 原生 SAM 3.1 视频验证（有界批次各自模型记忆） |
| | `compose.sam31-remote.yaml` | 连远端 SAM（替代本地 SAM overlay） |
| | `compose.sam31-stream.yaml` | 实验性持久原生记忆（叠在 `sam31` 上） |
| 场景 / 感知 | `compose.scene.yaml` | 连续人物 / 事件引擎（需 `compose.yaml` + 本地 SAM3 权重） |
| | `compose.perception.yaml` | SAM 连续分析（叠在 `compose.yaml` + `sam31` 后） |
| 大模型服务 | `compose.vlm-server.yaml` | 独立模型 + CPU 处理服务 |
| **交付 / production** | `compose.full-gpu.delivery.yaml` | 客户交付部署入口（见 [DEPLOY.md](../../DEPLOY.md) / [RELEASE.md](../../RELEASE.md)） |
| 诊断（在 `diag/`） | `diag/compose.sam-multistream.yaml` · `diag/compose.sam-tunnel.yaml` | 诊断专用，已单独放在 `diag/` 子目录 |

> 配套的 `config.*.yaml`：`config.yaml`（开发默认，须与 `src/goodvision_harness/config.yaml` 逐字节一致）、
> `config.vlm-rich.yaml`（富标注）、`config.delivery.yaml`（交付）。

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
