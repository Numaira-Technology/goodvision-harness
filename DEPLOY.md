# GoodVision 完整功能部署 · 2026.09.10-full.4

本版提供完整双镜像，由随包启动脚本检查容器运行环境并自动选择 GPU 或 CPU 模式。
所有应用代码、控制台和校验入口均来自本版源码；GPU 运行依赖复用经过版本与兼容性
核对的依赖层。两个镜像始终使用本版程序。

容器平台为 Linux/amd64，使用 Docker Compose v2；自动启动脚本需要 Python 3.10 或以上。
Windows Docker Desktop 使用 Linux 容器模式，启动命令可写为 `py -3 start_delivery.py`。

| 服务 | 镜像 | 功能 |
| --- | --- | --- |
| 主服务 | `goodvision-harness:2026.09.10-full.4` | 五能力判断、人脸识别、连续事件视频证据、JSON/SSE、视频与叠框 |
| 感知服务 | `goodvision-rich-label-worker:2026.09.10-full.4` | 人物/物品检测、会话内跟踪、异步行人属性 |

控制台支持 **A–H** 八条线路的手动选择与保存；自动测速按成功请求的中位耗时选最快线路，
候选线路需所有测试请求成功。四类行为事件可以在运行前分别选择敏感度；吸烟默认高敏感度，
无需强制认清烟具。陌生人标注在连续视频帧中保持跟踪。行人属性标注不显示性别。

## 1. 交付文件与模型

完整包使用 `goodvision-full-2026.09.10.4.tar`，包含上述两个镜像。
`IMAGES.json` 记录平台、镜像 ID 和归档校验值；`SHA256SUMS.txt` 用于校验镜像包。
`SOURCE_HASHES.json` 记录与镜像内核对过的程序文件，`RELEASE.md` 记录本版验证范围。

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

本版继续支持手动单目 / 双目选择、控制台缓存设置和 API key 配置。
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
sha256sum -c SHA256SUMS.txt
docker load -i goodvision-full-2026.09.10.4.tar
python3 start_delivery.py
```

macOS 校验可用 `shasum -a 256 -c SHA256SUMS.txt`；Windows PowerShell 可用
`Get-FileHash .\goodvision-full-2026.09.10.4.tar -Algorithm SHA256`，与
`SHA256SUMS.txt` 比对后执行 `docker load` 和 `py -3 start_delivery.py`。

本次 macOS Docker Desktop 环境的自动 GPU 探测返回无法确认设备，因此在本机使用
`python3 start_delivery.py --mode cpu`。确认不使用 GPU 的其他环境也可显式选择 CPU；
这会保留完整主服务，并停止可选感知服务。自动探测遇到未知 GPU 运行时错误会明确停止，
不会将“探测失败”当作“确定没有 GPU”。可加 `--dry-run` 先查看启动命令。

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
2. 设置视频源、单目 / 双目、线路 A–H 和各事件敏感度；正式检测需取消 Test mode，再启动所选检测。
3. 需要重启后自动检测时，在默认设置中取消 Test mode、勾选自动启动并保存。

密钥写入现有 `face-data` 数据卷，不会通过设置接口回显；容器重建后仍可读取。
尚未配置密钥时，正式检测会明确提示先在控制台保存 API key，控制台始终可用。
仅本地人脸检测和测试模式无需模型 API key。已有持久设置继续使用原模式与自动启动选择。
`INFERENCE_API_KEY` 环境变量仍可用于预配置，但不再是使用控制台的前置条件。

视频缓存支持在控制台设置 0–30 秒（可含小数），点击应用后立即生效并保存到同一数据卷，
无需停止会话。新版本页面与后端必须同时升级，以支持 `buffer_delay_seconds` 设置。

项目名保持 `goodvision-harness`，继续使用 `face-models`、`face-data`、`rich-models`
三个持久卷。升级时保留模型、已知人脸、连续事件视频证据与部署设置，不执行 `down -v`。

### 从旧版升级

先在控制台停止检测，将新包解压到独立目录并校验。沿用原部署的 `.env`（如有），
核对其中路径后，使用新包的 `compose.yaml`、`compose.gpu.yaml`、`start_delivery.py`
和 `config.delivery.yaml`。加载新镜像，再运行新包启动脚本。项目名与数据卷名保持一致，
容器会使用新镜像重建，数据卷保持原内容。不要把旧配置直接覆盖回新包，以免沿用旧的
事件分辨率；自定义参数应逐项合并。升级后刷新浏览器，以加载新版控制台脚本。

需要回退时停止检测，进入保留的旧交付目录，加载对应旧镜像并运行旧启动脚本；
同样保留数据卷。本版未迁移人脸库或部署设置格式。

### 事件敏感度、分辨率与解除警报

在 Run now 的“事件敏感度 · 启动前调整”中分别设置打架、吸烟、跌倒和持械。
吸烟“标准敏感度”要求明确识别烟具；“高敏感度”允许结合动作与上下文判断，
不要求认清小型或遮挡烟具，并可能增加误报。吸烟默认高档，其余默认标准。
选择仅应用于本次运行，运行后锁定；同一页面停止后可重新调整，刷新页面恢复默认值。
自动启动或 API 未指定时使用同样默认值。API 字段和枚举见 [接口说明](API_DELIVERY.md)。

事件送模采用完整画面，最长边 320 px、JPEG 质量 75；16:9 输入为 320×180，
其他比例等比缩放，不为多人场景逐人裁剪。本地人脸识别独立使用最长边 1280 px、
JPEG 质量 90。默认 2.5 FPS 单帧事件分析、六个并发请求；无事件时使用极简模型输出，
对外 JSON 仍是原有完整契约。

行为告警在连续 10 个有效阴性源帧后解除，阳性重新计数。同一源帧的重复结果只计一次，
失败或缺失结果不计阴性。2.5 FPS 下通常需要约四秒有效阴性；真实响应和采样可能使时间
更长。人脸模式按人、按事件保持红框，其他熟人仍为绿框；无人脸模式在待解除期间保留
告警标题，但不会将旧物体框贴到新的阴性画面。原始 JSON 和精确证据仍表示单帧模型判断。

迟到阳性仍显示独立事件通知并进入结果流，最新画面和最新结果按源帧时间前进。
接入方需区分事件通知、当前画面状态和逐帧原始判断；十帧解除规则不修改归档片段的独立
合并/结束策略。远景、遮挡与多人场景仍应分别验证检出率和误报率。

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
- 线路 A–H 可手动切换或测速自动选择，陌生人标注连续。
- 运行前可分别调整敏感度；吸烟标准档要求烟具，高档允许根据可见动作判断疑似吸烟。
- 行为阳性后第 1–9 次有效阴性仍保持告警，第 10 次解除；迟到阳性有通知但不使画面倒退。
- 五能力 JSON/SSE 和连续事件视频证据接口按 [接口说明](API_DELIVERY.md) 工作。

无 GPU 模式下，感知标注不启用是预期状态；应验收主服务、人脸识别、五能力接口与
连续视频。构建与合成测试不能替代目标服务器的真实 GPU 推理、视频效果和性能验收。
本版已完成的验证记录以 `RELEASE.md` 为准。


## 连续事件与证据存储

容器将事件证据保存在现有 `/data/event-evidence` 持久目录，需要可执行的
H.264 编码器。验收应实际生成、下载并解码视频，并检查 `/v1/status` 的归档
状态与录制失败计数；接口、保留期限和修订规则见
[事件证据说明](docs/architecture/event-evidence.md)。

升级时保留 `face-gallery.json`、模型和部署设置，不要删除持久卷。旧陌生人照片
接口返回 410；旧 `face-gallery-strangers.json` 和 `stranger-captures/` 已不再使用，
只在确认备份后清理。

混合会话的人脸和事件分支独立完成、独立发布。验收应在远程请求占满时检查人脸
仍能进展，并在人脸响应缓慢时检查吸烟阳转阴能及时发布。诊断计数位于
`/v1/status` 的 `pipeline.branches`；消费规则见 [接口文档](API_DELIVERY.md)，
已完成的目标环境检查见 [本版验证](RELEASE.md)。
