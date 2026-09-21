# Contour 实时队列：配置与现场验收

本轮基于 `codex/entity-event-engine` 的 `bcceda6`，包含 PR #21 和后续 HDR、
采样时钟、实时队列、独立 VLM、浏览器工作量限制等改动。先补配置透传、
显式实验入口和运行状态字段；后续增加多路 worker 的有界 CPU 解码准备阶段。
不替换 Contour，不改默认队列策略、检测阈值、播放缓冲或分辨率。

## 后续调整：解码不再占用共享 GPU 槽

多路 `sam_live` 请求改为：接收请求 → 等待准备配额 → CPU 解析、JPEG/H.264
解码及原尺寸规则下的缩放 → 等待 GPU → 原有串行模型推理、状态更新与清理。
仅移动 worker 收到的图片/码流解码；不是重新实现上游的 HDR 转码修复。

- 每个模型进程最多 **2 个**准备中的请求，计数覆盖解码、等待 GPU、推理直至释放图片。
  等待准备配额时不解码，因此不会为每路都积攒一份解码后的图像。
- 每路始终只有一个在途请求。超时/取消不能终止已经执行的底层解码或模型调用，
  它们退出前仍占用配额和该路入口，重试返回 429；不会无限追加后台线程。
- 未进入模型的解码/排队失败不修改跟踪状态；已进入模型的超时沿用原有失效规则。
  获取 GPU 后重新验证会话，重复请求直接返回缓存，不重复解码或推进模型。
- GPU 执行槽仍为 1；模型、跟踪历史、输入尺寸、采样率及现有双卡分流均不改变。
  旧单会话/批处理入口保留原执行顺序。本改动不新增双卡路由，也没有跨 GPU 的新锁。
- Review 后修复 JPEG 解码的告警配置竞争：不再在请求中修改全局 `warnings` 过滤器，
  改为在 RGB 内存分配前显式检查应用像素上限和 Pillow 的像素保护阈值。
  保持两路并发解码，不新增全局解码锁；坏图片和超限输入仍返回受控错误。

worker 的内部 `GET /health/cameras` 新增 `max_prepared_requests`（2）、
`prepared_requests` 和 `preparation_waiting_requests`。这些是每个模型进程的数据，
不是全站统计；Contour 公共健康接口未承诺透传这些新字段。HTTP 日志新增：

| 字段 | 口径 |
| --- | --- |
| `preparation_wait_seconds` | 等待准备配额的时间 |
| `preparation_seconds` | 准备阶段墙钟耗时，包含线程派发、解析、预检查及解码 |
| `preparation_decode_seconds` | 准备阶段中的解码、缩放子耗时；不另加到上一项 |
| `queue_wait_seconds` | 准备完成后等待 GPU 的时间，不再包含准备配额等待 |

多路 append 的 `decode_seconds` 现在为 0，因为解码已在进入 append 前完成；
单会话旧入口仍记录它。不要只看 GPU 等待下降就认定端到端提速，需要同时看准备等待、
HTTP 总耗时、吞吐、源时间落后和逐帧输出。超时日志表示截至 HTTP 返回的已观察耗时，
不是尚未退出的后台解码最终用时。

部署需要在维护窗口重建并更新 **SAM/Contour engine 镜像**，只更新主程序不生效。
沿用现场已有两卡各六路的节点及绑定，不要用本文的单节点示例替换现场部署。
先在相同视频、FPS、队列策略下对比旧/新 engine；该阶段通过后，再单独试实时队列。
解码改动的回退是恢复上一版 engine 镜像，`durable_ordered` 仅回退主程序队列策略。

本地模拟测试覆盖：慢解码时另一通道可推理、准备配额与已解码缓存有界、超时取消、
重复请求/EOF、输入预算、分辨率变化和输出协议一致性。它们不是实际双卡吞吐、
真实模型 mask 等价或 12 路长时稳定性证明；CPU 解码不是主要瓶颈时收益可能有限。

## 先核对双卡

仓库当前标准 Contour GPU 部署只有一个 `engine`，绑定 `CONTOUR_GPU_ID` 指定的
**一张卡**。`CONTOUR_CAPACITY=12` 只是开 12 个会话通道，不是两卡各六路。
现场可能有额外 Compose 覆盖或多个节点，需查实际部署，不能只凭源码判断。

在 GPU 主机实际 Contour 安装目录 `workers/contour` 执行只读检查。
若现场另有 Compose 文件、项目名或容器名，沿用现场值：

```sh
docker compose -f compose.yaml -f compose.gpu.yaml ps
docker inspect --format '{{json .HostConfig.DeviceRequests}}' goodvision-contour-engine-1
docker compose -f compose.yaml -f compose.gpu.yaml exec -T engine python -c \
  "import torch; print({'visible_gpu_count': torch.cuda.device_count()})"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
```

核对预留 GPU、可见卡数、两卡上的实际进程和各进程收到的摄像头请求。
另一张卡有其他任务不代表 Contour 在用它。不要公开完整容器配置或节点密钥。
如果只有一张卡在工作，下一步应在 Contour 现有认证网关与部署入口补固定分流；
不能只增加 GPU 可见数或有状态服务的 Uvicorn worker 数。本轮未实现双卡分流。

## 基线与实验

保留现有地址、节点密钥和 VLM 路由。在主程序 `.env` 设置
`TRACKING_SERVER_PROTOCOL=contour`。先停止全部分析、等待收尾，再从主程序仓库根目录运行。
下面 IP 是示例；**仅在节点已具备对应容量时用 12 路**，否则从已部署容量开始。

```sh
python tools/start_modular.py main --streams 12 \
  --tracking-server http://192.168.10.20:8093 --tracking-capacity 12 \
  --tracking-fps 1 --sam-queue-policy durable_ordered
```

记录基线后停止分析、等待收尾，用相同命令只将最后一项换成
`--sam-queue-policy realtime_latest`。两组都是每路 1 FPS 分析采样，不是 1 FPS 播放。
开关不自动改 FPS，也不启停 GPU 节点。命令默认重建主程序镜像；旧镜像不能仅靠变量获得新功能。
可先加 `--dry-run` 查看不含密钥的计划。

控制台已保存的服务器、容量和 FPS 优先于启动默认值，需同步核对。
密钥仍在控制台配置，不放命令行。修改 `.env` 后需要重新创建主程序容器；
不要在同一组对照中同时改视频、队列、分辨率和采样率。

## 参数与生效值

这些参数写在**主程序**而非 GPU 节点的 `.env`。CLI 策略覆盖进程环境和 `.env`；
进程环境覆盖 `.env`；未设或为空时保留 YAML。未显式覆盖的窗口参数保持原配置。

| 变量 | 仓库默认 | 含义 |
| --- | --- | --- |
| `SAM31_QUEUE_POLICY` | `durable_ordered` | 显式 `realtime_latest` 才启用实验，仅支持 `sam_live` |
| `SAM31_REALTIME_APPEND_FRAMES` | 1 | 追加上限：1、2、4 |
| `SAM31_REALTIME_PENDING_FRAMES` | 2 | 未准备候选数：1–8；不是全链路总帧数 |
| `SAM31_REALTIME_MAX_AGE_SECONDS` | 2 | 未发送候选年龄上限：0.01–30 秒 |
| `SAM31_RESULT_DEADLINE_SECONDS` | 4 | 及时率统计阈值：0.01–30 秒；不强制取消请求 |

启动器选实时策略时同时设置 `PERCEPTION_METHOD=sam_live`；直接运行 Compose 时需显式设置。
窗口默认值不是已证明的最优值。[已有实验](../validation/sam-realtime-queue-experiment-20260920.md)
表明过早丢帧会损失及时覆盖率。已发送请求不被年龄阈值取消，因此没有硬性延迟上限。

分析启动后查看每路状态的 `processing` 对象：

- 生效配置：`queue_policy`、`effective_max_append_frames`、`pending_capacity`、`realtime_max_age_seconds`。
- 积压：`queue_frames`、`oldest_pending_seconds`、`latest_result_age_seconds`。
- 处理与跳过：`processed_frames`、`skipped_by_reason`、`session_admitted_frames`。
- 及时率：`result_deadline_seconds`、`results_within_deadline`、`results_after_deadline`、`admitted_deadline_coverage`。

主路接口是 `GET /v1/status`；多路是 `GET /v1/fleet/cameras/camera01/status` 到
`camera12/status`，也可打开每路诊断页。未创建协调器时配置为 `null`，不会伪装成已生效。
停止后可能保留上轮统计，采集前确认当前分析与会话。

及时率分母包括已接纳但跳过的帧，不含采样前忽略的帧；它不是播放轮廓覆盖率、事件召回率
或摄像头采集到结果的完整端到端延迟。还需记录源时间落后、播放、描边率、事件检出、CPU、
显存、GPU 等待、推理和 VLM 延迟。PR #21 的诊断保留，见
[Contour 性能文档](../../workers/contour/PERFORMANCE.md)。

建议相同样本各跑三次短对照，通过后再做至少 60 分钟的 12 路测试。各路需持续产出，
队列、显存和延迟不随运行时间不断上涨，正样本与困难负样本表现达到事先约定的标准。
不能靠大量跳帧或只看平均值认定稳定。

回退时停止分析并等待收尾，显式用 `--sam-queue-policy durable_ordered` 重启主程序。
只删除 CLI 参数不会清除 `.env` 或 YAML 中的实验配置。不删除数据卷、原视频或跳帧审计。

本地自动化只验证配置、协议、队列边界和状态；真实双卡、GPU 性能、长时播放与检测质量待现场验收。

## 本轮本地验证记录

2026-09-20，macOS / Python 3.13，未使用 GPU 或真实模型服务：

- 初轮全量：4216 通过、25 失败、46 跳过。未修改的 `bcceda6` 独立快照在同一
  Python 环境、同一工作目录下复跑：4193 通过、同样 25 项失败、46 跳过；失败项一致。
- 补充极端整数输入校验后，部署、实时队列和 Contour 集成定向回归：62 通过、1 跳过。
  跳过的是缺少 PyAV 的 H.264 集成用例。
- 全量中的 5 项回环网络失败是沙箱禁止监听端口。允许本地回环监听后，相关测试组
  22 项全部通过。其余失败涉及既有测试/默认值差异、环境配置及缺失编码器或交付产物，
  不在本次补丁中顺带修复。
- 改动 Python 文件 Ruff、`git diff --check`、仓库 hygiene 检查通过。

定向复跑（使用已安装开发依赖的环境）：

```sh
PYTHONPATH=src:tests:. python -m pytest -o addopts='' -q \
  tests/test_realtime_deployment.py tests/test_modular_deploy.py \
  tests/test_sam_realtime_queue.py tests/test_contour_integration.py \
  tests/test_contour_multistream.py
```

测试使用模拟模型/HTTP 和真实 Compose 配置渲染，没有启动 12 路实际视频。

后续有界预解码补丁（同日、同一基线）：

- 新增 20 项准备阶段测试；连同旧会话、H.264、多路、云端分流、Contour 集成与
  前述配置入口，定向回归 **197 通过、1 跳过**（缺少 PyAV）。
- 最终全量 **4237 通过、25 失败、46 跳过**；逐项对比原始 `bcceda6` 的测试报告，
  25 个失败用例集合完全一致，没有新增失败。Ruff、diff 与仓库 hygiene 检查通过。
- 准备阶段及多路并发的 61 项测试连续三轮通过。
- 允许本地回环监听后，传输复用和超时相关的三个测试文件 **44 项通过**。
- 模拟编解码/模型输出验证通过，不代表真实 H.264 编解码或 CUDA 数值等价验证。

解码调整的定向复跑：

```sh
PYTHONPATH=src:tests:. python -m pytest -o addopts='' -q \
  tests/test_sam_preparation.py tests/test_sam31_continuous_worker.py \
  tests/test_sam31_h264_sessions.py tests/test_sam_multistream.py \
  tests/test_contour_multistream.py tests/test_cloud_gpu.py
```

以上六个文件共 139 项通过。首次使用 H.264 的 GPU 主机仍需验证其实际 PyAV/FFmpeg
依赖，以及相同输入帧的 mask、ID、置信度和事件检测结果。

JPEG 告警竞争修复追加验证：新增 `tests/test_sam_jpeg_decode.py` 的 21 项测试，
覆盖正常/失败解码的交错退出、全局过滤器不变、Pillow 阈值及告警策略变化。
其中 3 项并发回归在修复前均能复现失败，修复后全部通过；连同图片、批处理、
会话及多路 worker 相关回归，共 275 项通过。
修复后全量为 **4258 通过、25 失败、46 跳过**；对比原始基线报告，失败用例集合
仍完全一致，没有新增失败。21 项新增测试连续三轮通过，Ruff、diff 与 hygiene 检查通过。
