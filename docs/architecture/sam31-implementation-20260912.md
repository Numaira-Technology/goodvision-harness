# SAM 3.1 原生连续人物跟踪：实现与验收边界

更新：2026-09-13。分支：`codex/entity-event-engine`。
研究依据见 [SAM3.1 架构评估](sam31-evaluation-20260912.md)，实机记录见
[真实 GPU 验证](sam31-gpu-validation-20260912.md)。

## 已完成的路径

可选 `sam31_stream` 后端已经接入主程序。它在同一摄像头会话中保持 SAM 3.1 原生
人物编号和跟踪记忆，只加入已经收到的新帧，并回收过期输入与模型历史。
既有 `sam31` 保留为每批新 epoch 的原生视频对照；`sam` 保留图像关键帧检测加保守运动跟踪。
默认后端与 `SCENE_ENGINE_MODE=disabled` 没有因实验而自动切换。

三条路径都交给同一个人物／事件引擎。原生编号不是现实姓名；人物登记、人脸确认、
事件开始／持续／结束／修订继续由上层完成，无需把整个程序再次改成另一套架构。

| 层 | 连续路径的行为 |
| --- | --- |
| 模型服务 | `/health/video/session` 和 `/v1/video/sessions`；共享已严格验证的模型张量，不加载第二份权重 |
| 输入 | 首个非 final 请求 16 帧，后续 1–16 帧；只能按源帧号和源时间向前追加 |
| 原生记忆 | 固定 16 个输入槽；最近 32 帧历史及仍参与注意力的 conditioning 锚点；不支持已清理帧的反向或历史交互修改 |
| 输出 | 同一会话使用稳定 epoch；区分 accepted、processed、emitted，允许返回以前请求的源帧 |
| 人物关联 | `(source/session, backend, predictor epoch, native ID)` 映射人物；同 epoch 不再按位置二次改号 |
| 调度与背压 | 独立后台协调器，默认 4 FPS、16 帧、4 秒批次等待；执行和等待窗口有界，不无限排队 |
| 结束与故障 | 有帧 final、无新帧 EOF、显式 DELETE、TTL 清理；未知推进状态禁止续传旧 epoch |
| 事件与展示 | 保留原始源证据、独立时序判断、人物/事件面板、SSE 和事件元数据记录 |

官方 Prod 实现默认留出 14 帧确认缓冲：第一批 16 帧通常只输出前 2 帧。
成功回复的 processed 与 accepted 相同；emitted 可能落后。人物为空时仍输出对应源帧的
空快照，不能把缺失输出当作“没有人”。最后一个有帧请求或空 EOF 会发完待发结果；
显式取消使用 DELETE 时明确报告丢弃的待发帧数，不制造证据。主程序自然 EOF 或无错误断流
会先进入收尾状态，给原生输出和最后产生的时序检查合计最多 180 秒，再关闭场景；超时
或失败使旧回调失效并报告收尾结果。短录像的真实 GPU 收尾已经验证，较长实时回放的结果
继续记录在 GPU 验证文档中。

主程序只消费已输出的严格源帧前缀。未知接受状态、协议异常或推理超时使会话失效，
不会在旧编号上盲目重试。服务只缓存最后一次成功请求的 ID、内容 hash 和回复，支持
相同请求幂等重放。一个 worker 同时仅有一个活动会话；其间 batch 接口返回 429。

## 有界不等于无限身份记忆

会话硬上限为 65,536 个输入帧、2,048 个累计原生 ID、默认 120 秒空闲时间。
越界或源尺寸变化会返回 409 `reset_required`，客户端需建立新 epoch，不能静默复用人物编号。
帧上限也避开了固定官方代码中 100,000 的遮挡 sentinel 假设。

单请求上传最多 32 MiB，JPEG 合计最多 24 MiB，原始像素合计最多 64 Mi 像素；
先逐张解码并缩小再保留。上传、解码与推理共用单 GPU admission 槽，默认 60 秒总截止；
HTTP 超时不能强行中断 CUDA，槽保持到真实任务和清理结束。不可预料的原生错误会取消
共享模型的 readiness，需重启。完整限制与接口见 [worker 文档](../../workers/sam_scene/README.md)。

主程序连续路径的源证据缓冲上限为 512 帧、64 MiB，保留时间为
`max(32, 32 / max_fps + 24)` 秒；默认 4 FPS 时是 32 秒，最低 0.5 FPS 时是 88 秒。
这里只保存被连续跟踪协调器接纳的采样原图，避免高帧率预览挤掉尚待返回结果的证据；
原有完整视频／人脸归档继续接收自己的源画面。人物历史使用同一时间预算。
先前 16 秒预算在真实远程回放中不足：两次网络请求使部分源证据到达 17.25 秒才被消费，
触发 `evidence_expired`。现有预算覆盖模型确认延迟及两次远程请求的余量，仍需在目标网络下
验证。超过保留期的结果会被拒绝，60 秒故障截止并不是允许实际跟踪滞后 60 秒。

## 已完成的真实验证

租用 RTX 4090 24 GB 上已经完成获批权重下载、固定源码校验、全量键和形状校验、
严格重载、原生视频和连续会话 CUDA 预热。环境为 Ubuntu 24.04、Python 3.12.3、
torch 2.10.0+cu126、torchvision 0.25.0+cu126、驱动 560.35.03。
worker 通过 supervisor 运行，仅绑定服务器回环地址，主程序通过 SSH 转发访问。

- 48 帧一次处理、同状态分段、只在新帧到来后追加：全部原生 ID 和 mask hash 全等。
- 128 帧未清理追加与有界清理：全部 ID 和 mask hash 全等，清理后的输入及跟踪历史数量稳定。
- 256 帧真实多人画面（源片 6.0–69.8 秒），加入最新元数据清理和空 EOF：
  未清理参考与清理版本全部 ID 和 mask hash 全等。
- 已完成真实 48 帧 HTTP 会话到 `HarnessRuntime` 的人物跟踪接入；详细请求、源映射和计时
  以 [GPU 验证记录](sam31-gpu-validation-20260912.md) 为准。

实机还发现并修正了 RoPE 派生 buffer 与权重不符、线程间 autocast 上下文，及官方
processing-order 的 inclusive offset 与 detector 的 exclusive count 冲突。
没有放宽已学习权重的完整性要求；FA3 和编译仍关闭。

这些结果验证实现与资源回收的一致性，不测分割准确率或事件准确率。多人样例中曾出现
漏检和异常外接框，必须保留负面证据；不能把 HTTP 200、高置信度或编号连续当作质量保证。
当前尚无人工标注的充分遮挡／交叉／离开重入验收，也没有数小时持续运行结论。

## 运行连续路径

权重固定来自 `facebook/sam3.1/sam3.1_multiplex.pt`，来源 revision 与文件 hash
记录在 [GPU 验证记录](sam31-gpu-validation-20260912.md)。运行不需要 HF token，
也不自动下载门控权重。安装步骤和宿主驱动要求见 [worker 文档](../../workers/sam_scene/README.md)。

已安装环境的 GPU worker，从 `workers/sam_scene` 启动：

```sh
SAM31_ENABLED=true SAM31_SESSION_ENABLED=true SAM_SCENE_IMAGE_ENABLED=false \
SAM31_CHECKPOINT_PATH=/srv/goodvision/models/sam3.1_multiplex.pt \
SAM31_INFERENCE_MAX_SIDE=640 HF_HUB_OFFLINE=1 \
python -m uvicorn app:app --host 127.0.0.1 --port 8092 --workers 1
```

本地主程序也需要完整依赖。默认页面选中陌生人识别，因此仅安装 SAM 客户端或使用轻量测试环境
会在正式启动时因缺少 InsightFace／ONNX Runtime 失败。使用独立虚拟环境安装项目 extras：

```powershell
python -m venv .venv-demo
.venv-demo/Scripts/python.exe -m pip install -e ".[face,webrtc,vlm]" "opencv-python>=4.10,<5"
```

人脸模型仍从已有的 `FACE_MODEL_ROOT`／配置目录加载。确认本地人脸预热通过，以及远端
`/health/video/session` 的 `ready` 与 `gpu_ready` 都为 true 后，在完整环境启动本地主程序：

```powershell
$env:SCENE_ENGINE_MODE = 'active'
$env:PERSON_TRACKING_ENABLED = 'true'
$env:RICH_LABEL_ENABLED = 'true'
$env:RICH_LABEL_BACKEND = 'sam31_stream'
$env:RICH_LABEL_ENDPOINT = 'http://127.0.0.1:8092/v1/video/sessions'
$env:RICH_LABEL_MAX_FPS = '4'
$env:SAM31_BATCH_FRAMES = '16'
$env:SAM31_BATCH_SECONDS = '4'
$env:SAM31_CLIENT_TIMEOUT_SECONDS = '75'
$env:HARNESS_AUTOSTART = 'false'
.venv-demo/Scripts/python.exe -m goodvision_harness.service
```

沿用本地已有的模型与人脸配置，不在实验报告或 Git 中保存密钥。活动安全识别使用
持续事件链路；选择陌生人能力时，独立人脸识别仍可运行。默认时序请求单次包含 2–4 张
原始画面，15 秒总超时，最多一个请求在途；运行服务没有整场固定请求次数上限。

原生视频结果的源时间可能已经落后于实时画面。事件任务准备后最多排队 20 秒，首次领取后
另有 20 秒执行预算，按单调时钟计算；此前的原生追踪／网络耗时不会挤占模型执行时间。
队列仍限 4 个任务、16 MiB；超时结果不会重新激活，证据源时间与画面滞后保持原值。
状态会分别报告事件任务过期数和处理滞后；普通即时后端保留原有源时间过期规则。

连续容器部署叠加专用 overlay：

```sh
export SAM31_CHECKPOINT_FILE=/path/to/sam3.1_multiplex.pt
docker compose -f compose.yaml -f compose.sam31.yaml -f compose.sam31-stream.yaml \
  build --build-arg TORCH_CUDA=cu126 sam31-worker
docker compose -f compose.yaml -f compose.sam31.yaml -f compose.sam31-stream.yaml \
  build harness
docker compose -f compose.yaml -f compose.sam31.yaml -f compose.sam31-stream.yaml \
  up --no-build -d
```

[compose.sam31-stream.yaml](../../compose.sam31-stream.yaml) 设置主程序的连续 backend/endpoint，
启用 worker session 路径，并以 `/health/video/session` 检查健康状态。
默认 checkpoint 挂载为 `.models/sam3.1/sam3.1_multiplex.pt`，可用 `SAM31_CHECKPOINT_FILE`
指定；文件必须存在。镜像构建默认 cu128，上例 cu126 对应已验证实例的驱动。

## 保留批次对照与有预算实验

`compose.sam31.yaml` 默认仍启动 `sam31` 批次对照，不等同于启用 `sam31_stream`：

```sh
docker compose -f compose.yaml -f compose.sam31.yaml build \
  --build-arg TORCH_CUDA=cu126 sam31-worker
docker compose -f compose.yaml -f compose.sam31.yaml build harness
docker compose -f compose.yaml -f compose.sam31.yaml up --no-build -d
```

有界的单会话录像工具会按时间顺序处理本地视频，并在原始对应画面上标出延迟返回的人物框：

```sh
python tools/sam31_session_run.py recording.mp4 \
  --sam-endpoint http://127.0.0.1:8092/v1/video/sessions \
  --output runs/sam31-session-01 --fps 4 --max-seconds 60 --no-inference
```

[`tools/sam31_session_run.py`](../../tools/sam31_session_run.py) 输出 `annotated.mp4`、
`annotations.jsonl`（每张输出与原始源帧的映射）、`tracks.jsonl`（原生及人物轨迹）、
`appends.jsonl`（请求、水位与耗时）、`scene.jsonl` 和 `report.json`。全片保持一个 epoch，
EOF 发完尾部结果；报告标明 `memory_scope=session`、`offline_ordered_validation=true`、
`live_latency_measured=false`。它验证顺序执行，不按实时到达节奏评价网络延迟。

默认仅调用真实 SAM。加入 `--with-inference --max-requests 12 --events smoking,fall`
可检查选定事件并限制整个工具运行的 VLM 请求次数；事件还可选 `fight`、`weapon`，
请求上限允许 0–100，默认 12。`--no-inference` 只关闭 VLM。采样率范围 0.5–15 FPS，
`--batch-frames` 必须为 16，片长最多 600 秒，输出目录必须不存在。
可用 `--transport-max-side 640 --transport-jpeg-quality 75` 缩小给远程 SAM 的 JPEG；
本地时序证据和叠加底图仍保留原来的采样原图，报告分别记录两种尺寸及字节数。
准确率字段仍为 false，人物编号数量不等于人工身份切换指标。

批次录像工具仍保留：

```sh
python tools/sam31_video_run.py recording.mp4 \
  --sam-endpoint http://127.0.0.1:8092/v1/video/batches \
  --output runs/sam31-perception-01 --max-seconds 60 --no-inference
```

使用 `--with-inference --max-requests 12` 可做限定语义模型请求量的实验。
输出目录必须是新目录；原生轨迹、人物轨迹、场景状态和报告仍标记 `accuracy_measured=false`。
该批次工具的记忆不跨批次；连续 native 等价与回收使用 `tools/sam31_stream_probe.py` 和
`tools/sam31_stream_prune_probe.py --empty-final`，最多 256 帧、64 MiB JPEG。

## 后续质量验收

使用固定且有人工真值的录像，对照 `sam`、`sam31` 和 `sam31_stream` 的人物召回、
身份切换、遮挡重现、误检、事件归属及确认／结束延迟。再验证真实到达节奏下的网络、背压、
断线和源重连，以及长时间显存与 CPU 内存曲线。上限、取消、重启和新 epoch 恢复也需要覆盖。
通过这些验收后再决定是否切换默认后端；当前实验链路不作生产告警质量保证。
