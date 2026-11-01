# Repo 整理 · production（交付）vs demo（运行模式）清单

> 目的：厘清"生产环境"和"demo 环境"在本仓库里**到底是什么、由哪些文件/入口承载**，
> 为日后的部署文档与运维脚本提供一张对照表。**本次不移动任何代码、不改目录结构。**
>
> 权威依据：[`README.md`](../../README.md)（"Start here" 表 + Quick start + Repository layout）
> 与 [`CONTRIBUTING.md`](../../CONTRIBUTING.md)（"What belongs in Git" / "Organization and stable paths"）。

## 先厘清一个关键认知

**demo 和 production 不是两套平行文件，也不该用两条分支区分。** README / CONTRIBUTING 明确：

- `goodvision-harness-demo` 与 `goodvision-harness` **是同一套代码、同一个控制台**，默认端口不同（启动参数也各自独立）
  （demo `7900` / 生产 `8080`）。demo 是一个**运行模式/入口 + dry-run 标志**，不是独立代码。
- 根目录的 Compose / Dockerfile / env 示例 / 客户交付文档是 **"脚本、测试、运维共用的稳定入口"**，
  移动前须审查其消费者（CONTRIBUTING "Organization and stable paths"）。
- `config.yaml` 与 `src/goodvision_harness/config.yaml` **必须逐字节一致**，有回归测试强制。

> 结论：**切换环境靠"入口 + 运行标志 + delivery overlay/config"，不是靠切分支。**
> 按环境长期维护分支会增加共用代码的同步负担。分支本身不会破坏配置一致性；
> 无论在哪个分支，目录整理都应保留稳定入口，并同步修改两份 `config.yaml`。

## 怎么"切换"这两个环境（现成做法）

```bash
# demo / dry-run（合成结果、不调模型、不要人脸权重）
HARNESS_AUTOSTART=false HARNESS_DRY_RUN=true \
  goodvision-harness-demo --host 127.0.0.1 --port 7900

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

## B. Demo / dev（同一 app 的运行模式 + 开发用）

| 文件 / 入口 | 角色 |
|---|---|
| `goodvision-harness-demo`（控制台入口，端口 7900）+ `HARNESS_DRY_RUN=true` | **demo 运行模式**（非独立文件/分支） |
| `compose.yaml` | 基础栈，默认网络名 `goodvision-demo` |
| `config.yaml` · `config.vlm-rich.yaml` | 开发/富标注配置（`config.yaml` 与包内副本须一致） |
| `.env.example` | 开发环境变量示例 |
| `Dockerfile` · `Dockerfile.face-gpu` | 开发 / 人脸 GPU 镜像 |
| `WINDOWS_HYBRID.md` · `tools/start_windows_hybrid_demo.ps1` | Windows 非 NVIDIA 实验与演示 |

## C. Shared / 两个环境共用核心（不归任一环境）

README "Repository layout" 列的主体，demo 与 production 跑的都是它：

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

这些 Compose 文件按部署拓扑选择，不等于 demo 或 production 的归属，也并非全部都要与
`compose.yaml` 叠加。`compose.gpu.yaml`、`compose.face-gpu.yaml` 等提供 overlay；
`compose.main-server.yaml` 通过 `extends` 引用基础服务；`compose.vlm-server.yaml`、
`compose.analysis-gateway.yaml` 和 `compose.full-gpu.delivery.yaml` 是各自的部署入口。
使用对应启动器或指南给出的 Compose 文件组合，见[源码部署](deployment.md)与[分机部署](modular-services.md)。

## D. 本地运行数据与已审阅夹具

凭据、模型权重、人脸库、录像和生成输出不应提交；以 `CONTRIBUTING.md` 的目录约定与
`.gitignore` 为准。不要把整个 `video_samples/` 视为未跟踪目录：当前仓库保留了
`sample_video.annotation.json` 和 `sample_video.annotation.md` 两个已跟踪标注文件。
检查某个本地文件时使用 `git ls-files` 与 `git check-ignore -v`，不能从文件名推断跟踪状态。

---

## 结论与建议

1. **不要用分支分 demo/product。** 该项目本就以"同一代码 + 不同入口/标志/overlay"区分二者；
   分支切换会与设计相悖并带来维护负担。
2. 真正有价值的"整理"是**把上面这张对照表沉淀进部署文档**（或 `README` 的一个小节），
   让人一眼知道生产环境=交付那套、demo=运行模式。
3. **根部署文件（Compose / Dockerfile / env / 交付文档）不要擅自移动**；若日后仍想做物理归组，
   须先审查脚本/测试/运维消费者，并保留 `config.yaml` 一致性不变量。
