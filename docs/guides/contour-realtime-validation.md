# Contour 实时队列：配置与现场验收

本轮基于 `codex/entity-event-engine` 的 `bcceda6`，包含 PR #21 和后续 HDR、
采样时钟、实时队列、独立 VLM、浏览器工作量限制等改动。先补配置透传、
显式实验入口和运行状态字段；后续增加多路 worker 的有界 CPU 解码准备阶段。
不替换 Contour，不改检测阈值、播放缓冲或分辨率。2026-09-22 起，主程序的
`sam_live` 默认使用下述 `latest_only`；显式配置的旧策略仍保留。

## 默认策略：两个近期候选与提前准备

`latest_only` 固定保留最多两个尚未开始处理的内存候选。新采样到来时，容量已满则
淘汰最旧候选，剩余候选按源序号处理。提交通道有容量时，准备线程可在上一个 HTTP
请求运行期间取走一帧、落盘并编码。另有且仅有一个正在准备或准备完成的槽位；
它不计入两个可替换候选。HTTP 仍只有一个在途请求，不并发修改同一跟踪会话。

未发送候选和准备完成的帧复用 `SAM31_REALTIME_MAX_AGE_SECONDS`，默认 2 秒，
从本机收到该采样时计时。没有新帧到来也会清理过期等待；正在编码的工作不强制
中断，编码结束后、发送前再次检查年龄。过期准备批次保留审计记录并释放源图/
请求体；已发送请求和已返回结果不会因该期限被取消。不会为凑批等待，也不预测
请求完成时间；本次仍固定单帧发送，未改变采样率。

候选阶段没有持久化保证，覆盖记录写入诊断计数和日志；选中的源图在发送前可靠落盘。
已发送请求、结果持久化、身份/事件提交和失败恢复沿用原有顺序。源序号与时间戳不
重编号，同一跟踪会话继续推进。这个保证不等于跳帧后真实身份/事件召回率不变。

该模式固定单帧追加、双候选，忽略旧实验的 `SAM31_REALTIME_APPEND_FRAMES` 和
`SAM31_REALTIME_PENDING_FRAMES`，但使用 `SAM31_REALTIME_MAX_AGE_SECONDS`。
`SAM31_RESULT_DEADLINE_SECONDS` 仍只用于统计。需要处理全部采样时显式选择
`durable_ordered`；旧 `realtime_latest` 的数量/年龄实验仍可使用。

本次同时将 SAM 源图接纳落盘移出采集回调，并省去选中帧落盘后立即回读的步骤。
未将事件归档改成丢弃模式，未改变其他分支采样频率、显示缓冲或传播有效期。
[实现验证和取舍](../validation/tracking-performance.md)。

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
| `SAM31_QUEUE_POLICY` | `latest_only` | `sam_live` 双候选、单槽提前准备；可选 `durable_ordered` 或旧实验 `realtime_latest` |
| `SAM31_REALTIME_APPEND_FRAMES` | 1 | 仅旧实验：追加上限 1、2、4 |
| `SAM31_REALTIME_PENDING_FRAMES` | 2 | 仅旧实验：未准备候选数 1–8 |
| `SAM31_REALTIME_MAX_AGE_SECONDS` | 2 | 两种实时策略的未发送帧年龄上限 0.01–30 秒，含准备完成的帧 |
| `SAM31_RESULT_DEADLINE_SECONDS` | 4 | 及时率统计阈值：0.01–30 秒；不强制取消请求 |

启动器选实时策略时同时设置 `PERCEPTION_METHOD=sam_live`；直接运行 Compose 时需显式设置。
窗口默认值不是已证明的最优值。[追踪性能与验收边界](../validation/tracking-performance.md)
表明过早丢帧会损失及时覆盖率。已发送请求不被年龄阈值取消，因此没有硬性延迟上限。

分析启动后查看每路状态的 `processing` 对象：

- 生效配置：`queue_policy`、`effective_max_append_frames`、`pending_capacity`、`realtime_max_age_seconds`。
- 积压：`candidate_frames`、`queue_frames`、`oldest_pending_seconds`、`latest_result_age_seconds`。
  `latest_only` 的候选数最多 2；`queue_frames` 还含一个正在准备/准备完成的帧，最多 3，
  不包括在途 HTTP 请求和等待提交的结果。
- 处理与跳过：`processed_frames`、`skipped_by_reason`、`session_admitted_frames`。
- 及时率：`result_deadline_seconds`、`results_within_deadline`、`results_after_deadline`、`admitted_deadline_coverage`。

主路接口是 `GET /v1/status`；多路是 `GET /v1/fleet/cameras/camera01/status` 到
`camera12/status`，也可打开每路诊断页。未创建协调器时配置为 `null`，不会伪装成已生效。
停止后可能保留上轮统计，采集前确认当前分析与会话。

`latest_only` 的 `session_candidate_frames` 包含被覆盖候选，`session_admitted_frames`
只统计已选中并落盘的帧，含之后因准备过期未发送的帧；其及时率分母不含在内存中
被覆盖或过期的候选，必须同时查看处理与跳过数量。
旧 `realtime_latest` 的及时率分母包括已接纳但跳过的帧。它们都不是播放轮廓覆盖率、事件召回率
或摄像头采集到结果的完整端到端延迟。还需记录源时间落后、播放、描边率、事件检出、CPU、
显存、GPU 等待、推理和 VLM 延迟。PR #21 的诊断保留，见
[Contour 性能文档](../../workers/contour/PERFORMANCE.md)。

建议相同样本各跑三次短对照，通过后再做至少 60 分钟的 12 路测试。各路需持续产出，
队列、显存和延迟不随运行时间不断上涨，正样本与困难负样本表现达到事先约定的标准。
不能靠大量跳帧或只看平均值认定稳定。

回退时停止分析并等待收尾，显式用 `--sam-queue-policy durable_ordered` 重启主程序。
只删除 CLI 参数不会清除 `.env` 或 YAML 中的实验配置。不删除数据卷、原视频或跳帧审计。

本地自动化只验证配置、协议、队列边界和状态；真实双卡、GPU 性能、长时播放与检测质量待现场验收。

## 本地回归入口

运行 `python -m pytest`；队列、生命周期和部署配置的定向检查见
`tests/test_sam_latest_only.py`、`tests/test_sam_realtime_queue.py`、
`tests/test_sam_tracking_pipeline.py` 与 `tests/test_realtime_deployment.py`。
预解码及真实编码依赖检查见 `tests/test_sam_preparation.py`、
`tests/test_sam31_h264_sessions.py` 和 `tests/test_sam_jpeg_decode.py`。
本地替身测试不证明 CUDA 性能或检测质量；历史测量边界统一见[追踪性能](../validation/tracking-performance.md)。
