# 物体存在节点 · 真实 VideoJev 实机记录（2026-11-19）

PR #121 的验收证据。这次只证明服务链路跑通，**不证明**当前的刀具、酒瓶预设可用，也不用来选新门限。

## 版本与环境
- **代码：** `4802836`（物体节点、请求体和问题模板都用这个提交）。
- **模型：** `ZhuYuhan/VideoJev-4B` @ `1c1a253bf90dfa66176124d25a48ffb791d517d5`，权重从 R2 拉取。
- **服务：** 生产版 `goodvision_harness.videojev_service`，不是 benchmark 打分服务。
- **硬件：** Vast RTX 4090 一台，镜像 `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`，环境按 `tools/videojev_probe/requirements.txt` 安装。运行约 14 分钟，之后已销毁。
- **视频：** 翻包 demo 的 10 段视频（manifest_120 #101–120）。真值与 `tools/bag_demo/server.py` 中的 `TRUTH` 一致：刀具 IMG_2731、IMG_2732；酒瓶 IMG_2729（这个标签存疑）。视频和抽出的图片都不在仓库里。

## 方法
- **离线（`run_eval.py`）：** 逐帧判断，帧宽 720，分别按 1 fps 和 2.5 fps 抽帧。每段视频新建一个 `ObjectPresenceManager`，用刀具、酒瓶两个预设。
- **实时（`run_live.py`）：** 按真实时间回放 2.5 fps 帧，物体节点设 `sample_fps 1`、`clear_after 3`，客户端超时 10 s。
  - `--parallel` 会再开一个线程，**仿照** System 1 事件分支的请求节奏：每 0.8 s 发一个 2 帧窗口，帧间隔 0.5 s，带 6 个内置问题，用独立 session，超时 20 s。这个线程没有走真实的 `IndependentVLMBranch` 代码。
- **启动（`start_objeval.py`）：** 由 `tools/start_bag_demo.py` 改来，改成启动 `videojev_service`，端口 8097。
  - 用法：设置环境变量 `OBJEVAL_REPO`，可选 `OBJEVAL_ENV_FILE`。
  - 服务启动参数是 `--request-timeout 120 --queue-age 119`，**和生产默认的 20 / 19.5 不同**。实测所有请求都在 1 s 内完成。
- **汇总（`analyze.py`）：** 先 gunzip 出各 `*.jsonl`。百分位用 `tools/unified_benchmark/latency.py` 的 nearest-rank。

## 结果（完整输出见 `summary.txt`）
| | 仅物体节点 | 事件 + 物体并行 |
|---|---|---|
| 物体实际间隔 | P50 1.00 s / P95 1.01 s | P50 1.00 s / P95 1.09 s |
| 物体请求耗时 | P50 0.39 s / P95 0.41 s | P50 0.41 s / P95 0.57 s |
| 错误 | 物体 0/68 | 物体 0/68，事件 0/84 |
| 门限 0.95 下的刀具 | FN 2，FP 0 | FN 2，FP 0 |
| 门限 0.95 下的酒瓶 | FN 1，FP 0 | FN 1，FP 0 |

- **刀具：** 两段真刀的最高概率是 0.94 和 0.74。负样本最高是 IMG_1187，0.78。
- **事件分支：** 内置 `weapon` 问题在两段真刀上最高只有 0.40 和 0.19。

## 未验证
- 真实事件节点和生产队列配置下的并行表现。
- 多路吞吐。
- 新的问法或 4 帧采样。
- 留出集上的门限。正样本只有 3 个，`summary.txt` 里的门限对比只是描述，**不能**用来选门限。

## 记录字段
每行 JSON 是一次物体记录（`kind` 为 object，或离线运行时不带 `kind`）或一次事件请求（`kind` 为 event）。

字段包括：概率、作答结果、门限判断、保持状态、`transport_seconds`、墙钟时间、错误码。不含图片、视频或凭据。
