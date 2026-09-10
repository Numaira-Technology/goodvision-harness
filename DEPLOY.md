# GoodVision 完整功能部署 · 2026.09.10

本版提供完整双镜像，由随包启动脚本检查容器运行环境并自动选择 GPU 或 CPU 模式。
所有应用代码、控制台和校验入口均来自本版源码；GPU 运行依赖复用经过版本与兼容性
核对的依赖层。两个镜像始终使用本版程序。

容器平台为 Linux/amd64，使用 Docker Compose v2；自动启动脚本需要 Python 3.10 或以上。
Windows Docker Desktop 使用 Linux 容器模式，启动命令可写为 `py -3 start_delivery.py`。

| 服务 | 镜像 | 功能 |
| --- | --- | --- |
| 主服务 | `goodvision-harness:2026.09.10-full.3` | 五能力判断、人脸识别、连续事件视频证据、JSON/SSE、视频与叠框 |
| 感知服务 | `goodvision-rich-label-worker:2026.09.10-full.3` | 人物/物品检测、会话内跟踪、异步行人属性 |

控制台保留当前 8 个模型 **A–H** 的选择与保存。吸烟判断要求可辨认烟具和可见互动；
陌生人标注在连续视频帧中保持跟踪。行人属性标注不显示性别。

## 1. 交付文件与模型

完整包使用 `goodvision-full-2026.09.10.3.tar`，包含上述两个镜像。

本版视频叠字内置中文字体，离线运行无需安装系统字体。状态栏仅显示 `CLEAR`
或已检测事件；分析尚未完成、结果过期或人脸不可用时隐藏状态，不将缺失判断当成无异常。
事件描述、已知人员姓名及标注卡均支持中文。完整包包含视频编解码、GPU 运行库和
HTTP/2 通信依赖；模型权重仍使用下述持久卷。接口版本保持不变。
同目录还提供 `start_delivery.py`、`compose.yaml`、`compose.gpu.yaml`、
`.env.example`、`config.delivery.yaml`、接口说明和自检入口。

**镜像包含程序与运行环境，不包含模型权重。** 人脸识别需要人脸标准模型包；
GPU 感知功能还需要目标检测高精度模型包和属性标准模型包。
已有模型沿用持久卷 `face-models`、`rich-models`。模型的原始许可与归属声明保持不变。

感知服务默认使用 `RICH_DETECTOR_PROFILE=accurate` 和
`RICH_ATTRIBUTE_PROFILE=standard`。可通过 `.env` 中的
`RICH_DETECTOR_MODEL_DIR`、`RICH_ATTRIBUTE_MODEL_DIR` 指定容器内模型目录；
留空时使用选定档位的缓存。人脸程序是否允许下载由 `config.delivery.yaml` 中的
`face.allow_model_download` 决定。无 GPU 模式不加载感知模型。

### 网络摄像头手动选择单目 / 双目

`2026.09.10-full.3` 已包含手动单目 / 双目选择、控制台缓存设置和 API key 配置。
网络摄像头不再自动识别或等待探测帧。

在控制台网络视频源旁选择“单目 · 完整画面”或“双目 · 使用左半幅”。双目指
左右并排输入。需要重启后沿用时，在 Deployment defaults 中保存摄像头类型；
视频源旁的选择只作用于本次会话。运行期间类型锁定，先停止再切换并启动。
从第一帧开始使用所选画面，预览、分析、叠框及事件视频始终使用相同坐标。
文件和浏览器摄像头输入保持全画面。

新版 `config.delivery.yaml` 默认 `stream.frame_crop: none`。旧配置中的 `auto`
也按完整画面处理，不会再触发识别；旧 `left_half` / `right_half` 显式配置仍兼容。
保存的 `camera_layout` 优先于配置文件，单次启动的同名字段优先于保存值。
持久化设置使用现有数据卷，无需新增依赖、环境变量或挂载。
`/v1/status` 的 `stream.camera_layout` 和 `stream.crop_selection` 可查看本次选择。

## 2. 加载并启动

在完整交付包目录执行：

```bash
docker load -i goodvision-full-2026.09.10.3.tar
python3 start_delivery.py
```

无需创建 `.env` 即可启动控制台。需要自定义监听地址、端口或其他环境配置时，可复制
`.env.example` 为 `.env` 后修改；已有 `.env` 和部署设置应保留。
启动脚本自动查找随包配置文件，检查 GPU 能否供容器
使用，并选择对应的启动配置：

- 有可用 GPU：启动本版主服务与感知服务，启用 GPU 加速。
- 无可用 GPU：使用同一个本版主服务镜像运行，保留人脸识别、五能力判断与视频接口；
  感知服务不启动，人物/物品及属性标注不启用。
- GPU 模式启动或运行出错：报告错误，不改用旧版镜像或旧功能。

GPU 镜像和组件始终保留在完整交付包中。感知服务运行在另一台机器时，是否启用标注
以该服务实际确认的 GPU 能力为准。

默认主服务端口为 `8080`，无 `.env` 时仅监听本机，可打开 `http://localhost:8080`。
需要局域网访问时，在 `.env` 中设置 `HARNESS_BIND_ADDRESS=0.0.0.0`。
首次安装保持待机和测试模式，不自动连接视频源，先完成以下配置：

1. 在控制台填写 API key，点击“保存 API key”。无需修改容器或创建 `.env`，保存后立即可用。
2. 设置视频源、单目 / 双目及模型 A–H；正式检测需取消 Test mode，再启动所选检测。
3. 需要重启后自动检测时，在默认设置中取消 Test mode、勾选自动启动并保存。

密钥写入现有 `face-data` 数据卷，不会通过设置接口回显；容器重建后仍可读取。
尚未配置密钥时，正式检测会明确提示先在控制台保存 API key，控制台始终可用。
仅本地人脸检测和测试模式无需模型 API key。已有持久设置继续使用原模式与自动启动选择。
`INFERENCE_API_KEY` 环境变量仍可用于预配置，但不再是使用控制台的前置条件。

视频缓存支持在控制台设置 0–30 秒（可含小数），点击应用后立即生效并保存到同一数据卷，
无需停止会话。新版本页面与后端必须同时升级，以支持 `buffer_delay_seconds` 设置。

项目名保持 `goodvision-harness`，继续使用 `face-models`、`face-data`、`rich-models`
三个持久卷。升级时保留模型、已知人脸、连续事件视频证据与部署设置，不执行 `down -v`。

## 3. GPU 模式自检

在具备可用 GPU 且模型已准备好的服务器上，使用随包配置运行：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml --profile gpu run --rm --no-deps harness python tools/check_runtime.py
docker compose -f compose.yaml -f compose.gpu.yaml --profile gpu run --rm --no-deps harness python tools/check_face.py
docker compose -f compose.yaml -f compose.gpu.yaml --profile gpu run --rm --no-deps rich-label-worker python tools/check_perception.py
```

运行环境自检核对依赖组合。人脸自检验证检测与识别实际使用 GPU；感知自检验证
GPU 运算、目标检测和属性分析，并要求实际 GPU 就绪。运行环境自检通过不等于
真实模型效果或现场性能已经验收。

## 4. 视频与标注验收

GPU 感知服务首次加载模型期间，主服务继续运行。感知服务确认 GPU 和模型就绪后，
主服务会自动启用标注，无需因冷启动等待而重启主服务。
感知服务的 `/health/ready` 返回 `ready: true` 和 `gpu_ready: true`，才允许启用标注。

输入含人的测试视频后，在 GPU 模式确认：

- `/v1/status` 中 `rich_labeling.succeeded > 0`，且没有持续错误。
- `/v1/rich-labels/latest` 返回跟踪对象和后续帧的行人属性。
- `/v1/video/annotated/stream.mjpeg` 中人物/物品跟踪框与属性正常，性别不显示。
- 模型 A–H 可切换，陌生人标注连续，手指靠近嘴部不会单独成为吸烟证据。
- 五能力 JSON/SSE 和连续事件视频证据接口按 [接口说明](API_DELIVERY.md) 工作。

无 GPU 模式下，感知标注不启用是预期状态；应验收主服务、人脸识别、五能力接口与
连续视频。构建与合成测试不能替代目标服务器的真实 GPU 推理、视频效果和性能验收。
本版已完成的验证记录以 `RELEASE.md` 为准。


## 连续事件与证据存储

即时识别保持原采样与发布节奏，后台事件归并和视频编码不阻塞当前结果。
事件列表与视频接口见 [事件证据说明](EVENT_EVIDENCE.md)。默认每个连续事件
保留首次阳性前1秒后2秒的5 FPS MP4，保留7天且总量不超过10 GiB。

容器使用现有 `/data/event-evidence` 持久目录；Windows 默认在已知人脸库旁的
`event-evidence` 子目录，可用 `EVENT_EVIDENCE_DIRECTORY` 覆盖。需要可执行的
H.264 编码器；交付自检应实际生成、下载并解码视频。

GPU detector-only `/health/tracking` 就绪即可开启 `/v1/tracks`，与属性及界面
标注开关独立。无GPU时保持现有模型人物框/本地人脸位置追踪，不增加必需服务。

旧陌生人照片接口返回410。旧 `face-gallery-strangers.json` 和 `stranger-captures/`
不再加载或写入；确认备份后可由运维清理。保留 `face-gallery.json`、模型和部署
设置，不能通过删除持久卷来升级。归档状态和录制失败计数见 `/v1/status`。

混合会话的人脸与事件模型独立完成、独立发布；消费通用 SSE 时按能力判断源帧先后，
不能用较新的吸烟帧丢弃尚未收到的人脸结果。各能力的 `latest` 接口可分别读取当前
观测，缺失不表示阴性。诊断时查看 `/v1/status` 的 `pipeline.branches`：两支各有
完成/跳过计数，人脸的 `last_completion.face` 区分尺寸/质量不足与身份尚待确认。
部署验收应在占满远程请求名额时检查人脸仍能进展，并在人脸慢响应时检查吸烟阳转阴
能及时发布；说明见 [接口文档](API_DELIVERY.md) 与 [联合验证](STRANGER_LLM_INTEROP_VALIDATION.md)。
