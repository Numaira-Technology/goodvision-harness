# GoodVision 完整功能部署 · 2026.09.09

本版提供完整双镜像，由随包启动脚本检查容器运行环境并自动选择 GPU 或 CPU 模式。
所有应用代码、控制台和校验入口均来自本版源码；GPU 运行依赖复用经过版本与兼容性
核对的依赖层。两个镜像始终使用本版程序。

容器平台为 Linux/amd64，使用 Docker Compose v2；自动启动脚本需要 Python 3.10 或以上。
Windows Docker Desktop 使用 Linux 容器模式，启动命令可写为 `py -3 start_delivery.py`。

| 服务 | 镜像 | 功能 |
| --- | --- | --- |
| 主服务 | `goodvision-harness:2026.09.09-full.1` | 五能力判断、人脸识别、陌生人照片、JSON/SSE、视频与叠框 |
| 感知服务 | `goodvision-rich-label-worker:2026.09.09-full.1` | 人物/物品检测、会话内跟踪、异步行人属性 |

控制台保留当前 8 个模型 **A–H** 的选择与保存。吸烟判断要求可辨认烟具和可见互动；
陌生人标注在连续视频帧中保持跟踪。行人属性标注不显示性别。

## 1. 交付文件与模型

完整包使用 `goodvision-full-2026.09.09.1.tar.gz`，包含上述两个镜像。
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

## 2. 加载并启动

在完整交付包目录执行：

```bash
docker load -i goodvision-full-2026.09.09.1.tar.gz
cp -n .env.example .env
chmod 600 .env
python3 start_delivery.py
```

已有 `.env` 和部署设置应保留。启动脚本自动查找随包配置文件，检查 GPU 能否供容器
使用，并选择对应的启动配置：

- 有可用 GPU：启动本版主服务与感知服务，启用 GPU 加速。
- 无可用 GPU：使用同一个本版主服务镜像运行，保留人脸识别、五能力判断与视频接口；
  感知服务不启动，人物/物品及属性标注不启用。
- GPU 模式启动或运行出错：报告错误，不改用旧版镜像或旧功能。

GPU 镜像和组件始终保留在完整交付包中。感知服务运行在另一台机器时，是否启用标注
以该服务实际确认的 GPU 能力为准。

默认主服务端口为 `8080`。在浏览器打开 `http://服务器地址:8080`，设置视频源、
推理服务凭据及模型 A–H。凭据可通过控制台保存，也可填入 `.env` 的
`INFERENCE_API_KEY`。首次安装默认测试模式；已有持久设置继续使用原模式。

项目名保持 `goodvision-harness`，继续使用 `face-models`、`face-data`、`rich-models`
三个持久卷。升级时保留模型、已知人脸、陌生人照片与部署设置，不执行 `down -v`。

## 3. GPU 模式自检

在具备可用 GPU 且模型已准备好的服务器上，使用随包配置运行：

```bash
docker compose --env-file .env -f compose.yaml -f compose.gpu.yaml --profile gpu run --rm --no-deps harness python tools/check_runtime.py
docker compose --env-file .env -f compose.yaml -f compose.gpu.yaml --profile gpu run --rm --no-deps harness python tools/check_face.py
docker compose --env-file .env -f compose.yaml -f compose.gpu.yaml --profile gpu run --rm --no-deps rich-label-worker python tools/check_perception.py
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
- 五能力 JSON/SSE 和陌生人照片接口按 [接口说明](API_DELIVERY.md) 工作。

无 GPU 模式下，感知标注不启用是预期状态；应验收主服务、人脸识别、五能力接口与
连续视频。构建与合成测试不能替代目标服务器的真实 GPU 推理、视频效果和性能验收。
本版已完成的验证记录以 `RELEASE.md` 为准。
