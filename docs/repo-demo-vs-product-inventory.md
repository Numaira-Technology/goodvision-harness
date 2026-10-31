# Repo 整理 · demo vs product 清单（梳理，不重构）

> 目的：在**不移动任何代码**的前提下，把当前仓库里的 compose / config / Dockerfile / 部署文档
> 按"**product（交付/delivery）**"与"**demo/dev（演示与开发）**"分类，便于后续决定是否真正拆分。
>
> **本清单是基于命名与文件头注释的推断**，不是权威划分。标 ⚠️ 的条目需要对应 owner 确认后才算定。
> 生成于仓库整理任务；代码此刻未改动。

## 判定信号（从现状观察到的）

- **product / 交付**：文件名含 `delivery` / `release` / `customer`，或文档标题写"交付"。这一套是给客户部署的完整功能包。
- **demo / dev**：`compose.yaml` 的默认网络名是 `goodvision-demo`；众多 `compose.*.yaml` overlay 是模块化开发/实验栈（SAM、VLM、scene、perception、本地/云 GPU 等）。
- **shared / 共用核心**：`src/` `tests/` `tools/` `schemas/` `workers/` 等两边都依赖，不归任何一边。

---

## A. Product / 交付（customer delivery）

| 文件 | 说明 |
|---|---|
| `compose.full-gpu.delivery.yaml` | 完整 GPU 交付栈（`name: goodvision-harness`） |
| `config.delivery.yaml` | 交付用配置 |
| `.env.full-gpu.delivery.example` | 交付环境变量示例 |
| `Dockerfile.release` | 发布/交付镜像 |
| `DEPLOY.md` | 《GoodVision 完整功能部署 · …full1》 |
| `API_DELIVERY.md` | 《GoodVision 接口交付说明》 |
| `RELEASE.md` | 《GoodVision 完整交付 · …full1》 |
| `tools/start_delivery.py` · `tools/delivery_smoke.py` · `tools/run_full_delivery_smoke.ps1` | 交付启动与冒烟 |
| `tools/export_full_delivery.py` | 导出完整交付包 |
| `tools/package_customer_delivery.py` · `tools/package_customer_update.py` · `tools/package_customer_docker_update.py` | 打客户交付/更新包 |
| `tools/audit_customer_bundle.py` · `tools/customer_check.py` · `tools/customer_discovery.py` · `tools/customer_runtime.py` · `tools/customer_update.py` · `tools/customer_update_gui.py` | 客户侧审计/运行/更新工具 |

## B. Demo / dev（演示与开发）

| 文件 | 说明 |
|---|---|
| `compose.yaml` | 基础栈，默认网络 `goodvision-demo` |
| `config.yaml` · `config.vlm-rich.yaml` | 开发用配置 |
| `.env.example` | 开发环境变量示例 |
| `Dockerfile` · `Dockerfile.face-gpu` | 开发/人脸 GPU 镜像 |
| `WINDOWS_HYBRID.md` | Windows 非 NVIDIA 实验 |
| `tools/start_windows_hybrid_demo.ps1` | Windows 混合演示启动 |

### B1. 模块化 overlay（demo/dev 为主，⚠️ 需 owner 确认是否也进交付）

这些是"叠在 `compose.yaml` 之上"的开发/实验栈。部分可能被交付流程间接复用，需确认：

| 文件 | 文件头说明 |
|---|---|
| `compose.main-server.yaml` | CPU 应用主机（`goodvision-main`） |
| `compose.analysis-gateway.yaml` | 仅转发 console/camera/output（`goodvision-main`） |
| `compose.cloud-gpu.yaml` | 本地网关 + UI 控制的云 GPU 租用（`goodvision-cloud`） ⚠️ |
| `compose.local-gpu.yaml` | 本地 NVIDIA GPU overlay |
| `compose.detector-remote.yaml` | 租 GPU 跑 YOLO/RF-DETR 检测器 |
| `compose.sam-server.yaml` · `compose.sam31.yaml` · `compose.sam31-remote.yaml` · `compose.sam31-stream.yaml` | SAM 系列服务/验证/实验 |
| `compose.perception.yaml` · `compose.scene.yaml` | SAM 连续分析 / 人物·事件引擎 |
| `compose.vlm-server.yaml` | 独立模型 + CPU 处理服务 |
| `compose.local-tracking.yaml` | 应用主机本地 GPU 上的 SAM 运行时 |
| `compose.face-gpu.yaml` · `compose.gpu.yaml` · `compose.full-gpu.accelerated.yaml` | 人脸/通用/加速 GPU 栈 ⚠️ `accelerated` 命名与交付接近，待确认归属 |

## C. Shared / 共用核心（两边都依赖，不归任一环境）

| 路径 | 说明 |
|---|---|
| `src/` | 主代码（`goodvision_harness` 包） |
| `tests/` | 测试 |
| `tools/`（除上面列入 A 的交付工具外） | 通用脚本与基准/诊断工具 |
| `schemas/` · `workers/` · `diag/` | schema、worker、诊断 |
| `bench/` · `benchmark/` · `bench_v1_sample/` | 评测体系（偏 dev，⚠️ 若要可单列一类） |
| `README.md` · `CONTRIBUTING.md` | 项目总说明 |
| `pyproject.toml` · `.github/` | 构建与 CI |

## D. 本机杂物（未被 git 跟踪，不在 repo 内，仅本地）

`yolo*.pt` / `yolov8*.pt` 权重、`testABvideos.zip` / `testABvideos/`、`auto.crt` / `auto.key`、
`video_samples/`、`outputs/` 等是本机工作目录文件，**git 未跟踪**，所以没有撑大仓库。
如需减少本机混乱，可自行清理，不影响 repo。

---

## ⚠️ 需 owner 确认的点

1. **overlay 归属**：B1 里的 overlay 哪些实际进客户交付？特别是 `compose.full-gpu.accelerated.yaml`
   （命名像交付）和 `compose.cloud-gpu.yaml`（云租用）。
2. **benchmark 体系**（`bench/` `benchmark/`）是否算 dev-only、要不要单独一类。
3. **是否真的要物理拆分**。本清单只做梳理；若日后要落地，见下方建议。

## 如果日后决定落地拆分（备选，暂不执行）

- **最小改动**：在 repo 内建 `deploy/product/` 与 `deploy/demo/` 两个目录，把对应 compose/config/env
  软链接或移动过去，顶层留 `README` 指路。共用代码 `src/`/`tools/` 不动。
- **彻底隔离**：product 留本 repo，demo 另开 repo —— 迁移成本与历史处理较大，不推荐作为第一步。

> 下一步：请 owner 核对 A/B 分类与 ⚠️ 项；确认后再谈是否物理拆分。
