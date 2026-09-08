# GoodVision 完整 GPU 功能部署

本入口面向已验证兼容的 Linux/amd64 GPU 服务器，同时部署主服务与感知服务。
完整功能保留，客户配置使用功能名称和模型档位，无需指定底层实现。

| 服务 | 镜像 | 功能 |
| --- | --- | --- |
| 主服务 | `goodvision-harness:2026.09.03-full-gpu.1` | 五能力判断、人脸识别、陌生人照片、JSON/SSE、视频与叠框 |
| 感知服务 | `goodvision-rich-label-worker:2026.09.03-full-gpu.1` | 人物/物品检测、会话内跟踪、异步行人属性 |

本版包含最新程序改进，并固定使用预设推理服务和标准视觉模型档位。
控制台操作、五能力接口和视频接入方式保持兼容；无需选择服务渠道或模型。
旧版交付包继续保留，使用旧镜像时仍以对应包内说明和配置为准。

## 1. 前提与模型

宿主机须具有兼容的 GPU 驱动和容器 GPU 运行环境，并允许两个服务访问 GPU。
两个服务使用同一张可用 GPU；资源声明不意味着各占用一整张卡。
具体硬件兼容性须通过下述目标服务器自检确认。

**本包包含程序与运行环境，不包含模型权重，不是断网开箱即用包。**
真实推理需要以下已授权模型，模型文件应按相应档位准备：

- 人脸标准模型包：人脸检测和识别。
- 目标检测高精度模型包：人物与物品检测。
- 属性标准模型包：行人属性分析。

模型使用权限须单独确认；标准人脸模型包未经另行授权时仅限非商业研究，
不能把镜像可运行视为已取得商业模型使用许可。

已有模型优先沿用持久卷 `face-models`、`rich-models`，不要覆盖现场数据。
感知服务默认使用 `RICH_DETECTOR_PROFILE=accurate` 和
`RICH_ATTRIBUTE_PROFILE=standard`。可通过 `.env` 中的
`RICH_DETECTOR_MODEL_DIR`、`RICH_ATTRIBUTE_MODEL_DIR` 指向容器内已预置模型目录。
目录留空时使用选定档位的缓存，缺少模型时可能尝试下载；目录必须是容器内路径。

人脸主程序是否允许下载由 `config.delivery.yaml` 的
`face.allow_model_download` 决定。`FACE_ALLOW_MODEL_DOWNLOAD` 仅用于人脸自检，
不能替代主程序配置。严格离线部署须预置模型、关闭模型下载并断网自检；
若安全判断使用网络推理服务，仍须具备该服务的连通性，或另行部署已验证的内网服务。

## 2. 加载与保留现场设置

在完整交付包目录执行：

```bash
docker load -i goodvision-full-2026.09.03-gpu.1.tar.gz
cp -n .env.full-gpu.delivery.example .env
chmod 600 .env
docker compose --env-file .env -f compose.full-gpu.delivery.yaml config --quiet
```

已有 `.env` 必须保留。此入口固定启用人脸 GPU 加速与完整感知链路，不需要
额外 profile 或叠加旧的 GPU 覆盖文件。`config.delivery.yaml` 挂载为主服务配置；
不要把开发配置或旧包配置覆盖到该文件。

推理凭据可在原控制台保存，或通过 `INFERENCE_API_KEY` 提供。该凭据须适用于
本版预设服务。其他服务的已保存凭据不会自动混用；升级后如提示缺少凭据，
在原控制台填写本预设服务的正确凭据即可。旧凭据仅存在环境变量时，也须确认
它适用于本预设后再填入上述中性变量。不要将实际凭据写入交付示例。

客户镜像固定使用预设服务，旧配置中的渠道、模型或服务地址覆盖不会改变该预设。
视频采集选项使用 `VIDEO_CAPTURE_OPTIONS`，示例保留原默认值。

保留项目名 `goodvision-harness` 及三个持久卷，不执行 `down -v`、volume prune，
不要清空模型、已知人脸、陌生人照片或部署设置。升级前应备份现场持久数据。

## 3. 在目标 GPU 服务器先做真实自检

先确认宿主机 GPU 可用，再运行随镜像提供的三个统一自检入口：

```bash
docker compose --env-file .env -f compose.full-gpu.delivery.yaml run --rm --no-deps harness python tools/check_runtime.py
docker compose --env-file .env -f compose.full-gpu.delivery.yaml run --rm --no-deps harness python tools/check_face.py
docker compose --env-file .env -f compose.full-gpu.delivery.yaml run --rm --no-deps rich-label-worker python tools/check_perception.py
```

运行环境检查用于确认依赖组合可用；它不等于模型推理验收。人脸自检要求
已授权权重存在，并验证人脸检测、识别实际使用 GPU；感知自检验证 GPU 运算、
目标检测和属性分析。任一失败都应先解决，不能只以健康检查或容器启动作为验收依据。

底层依赖的许可与原始归属声明保留在镜像内部；本包未修改或伪造这些声明。
运行环境自检包含已知兼容性检查，不能替代真实视频效果和性能验收。

## 4. 启动完整服务

```bash
docker compose --env-file .env -f compose.full-gpu.delivery.yaml up -d --force-recreate
docker compose --env-file .env -f compose.full-gpu.delivery.yaml ps
```

该命令同时启动主服务和感知服务。感知服务仅在内部网络的 8091 端口提供接口，
外部通过主服务访问，默认端口为 8080。控制台可配置视频源和推理服务凭据。
首次启动默认 `dry_run`；已有持久部署设置会保留原模式。真实五能力判断需
切换为正式模式，不能将演示结果当成真实识别结果。

感知服务不可用时主服务仍可能输出安全判断和原视频，所以“视频能看”不等于
完整链路已工作。输入一段获授权的含人视频后，应同时确认：

- `/v1/status` 中 `rich_labeling.succeeded > 0`，且没有持续错误。
- `/v1/rich-labels/latest` 返回 200，存在跟踪对象，后续帧出现非空行人属性。
- `/v1/video/annotated/stream.mjpeg` 中人物/物品跟踪框及属性正常。
- 五能力 JSON/SSE 和陌生人照片接口按 [接口说明](API_DELIVERY.md) 工作。

感知服务的 `/health/ready` 只代表检测器就绪，不代表行人属性已成功。
要求离线加载模型时，还须断开模型下载网络、重启两服务并重复上述验收。

## 5. 构建验证与现场验收的区别

构建机没有可用于本包的 GPU。本地可核对配置、镜像加载、运行环境及隔离的
HTTP/视频程序链路，但不能验收真实 GPU 推理、模型效果或现场性能。
本次已完成的验证以 `RELEASE.md` 为准；合成画面与测试模型只验证程序协议，
不能替代上述真实模型自检，也不能代替授权现场视频验收。

接口路径与之前保持兼容。旧的 `http://192.168.1.50` 无端口地址仍需要现场
80 端口映射；直接连接默认使用 `http://192.168.1.50:8080`。服务和控制台应仅
对受信任网络开放。
