# GoodVision WebRTC 局域网部署

交付包包含已构建镜像、`compose.delivery.yaml`、环境示例和校验清单；服务器
无需源码或现场构建。机器人 `192.168.20.199` 提供 WHEP/WebRTC，GoodVision
服务器 `192.168.20.168` 在受保护局域网的 TCP 8080 提供输出。

## 安装与升级

```bash
sha256sum -c SHA256SUMS
docker load -i goodvision-harness-2026.08.28-face-distance.1.tar
cp -n .env.delivery.example .env
chmod 600 .env
docker compose --env-file .env -f compose.delivery.yaml config --quiet
docker compose --env-file .env -f compose.delivery.yaml up -d --force-recreate
docker compose --env-file .env -f compose.delivery.yaml ps
```

已有 `.env` 必须保留。升级时不要执行 `down -v`、`docker volume prune` 或修改
Compose 顶层项目名，以保留 `goodvision-harness_face-data` 和
`goodvision-harness_face-models`。

## 可选 rich bbox worker

源码部署可用独立的 PaddleX/PaddleDetection sidecar。CPU 回退路径仍是：

```bash
docker compose --profile rich up --build
```

它只为延迟展示流提供人物/物品跟踪框和外观属性，不参与也不阻塞 VLM 安全
判断；worker 超时或不可用时 Harness 会继续输出 VLM 结果与原始视频。交付版
`compose.delivery.yaml` 不内置 worker，若启用须单独交付兼容镜像、加入同一
Docker 网络，并把 `RICH_LABEL_ENDPOINT` 指向其 `/v1/labels`。

随源码提供的基础镜像固定 Paddle 3.2.2，默认使用低延迟的 `PicoDet-S`，以避开
Paddle 3.3.x 已确认的 CPU oneDNN 回归。客户 RTX 5090 是 Blackwell `sm_120`，
必须使用独立的 `Dockerfile.gpu` / `compose.gpu.yaml`：其中安装官方 Paddle GPU
3.3.0 CUDA 12.9 包，默认用精度更高的 `PP-YOLOE_plus-L`，不启用 TensorRT/HPI。
Linux 宿主机须安装 NVIDIA 570.26 或更新驱动（优先当前 production driver）和
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

源码部署在 5090 上的启用顺序：

```bash
nvidia-smi
docker compose -f compose.yaml -f compose.gpu.yaml --profile rich build rich-label-worker
docker compose -f compose.yaml -f compose.gpu.yaml --profile rich run --rm \
  rich-label-worker python gpu_smoke.py
docker compose -f compose.yaml -f compose.gpu.yaml --profile rich up -d
```

离线镜像交付则先加载两个 tar，保持 `RICH_LABEL_ENABLED=false`，用交付覆盖文件
完成相同的实际 CUDA/模型自检：

```bash
docker compose -f compose.delivery.yaml -f compose.rich-gpu.delivery.yaml \
  --profile rich run --rm rich-label-worker python gpu_smoke.py
```

自检会验证容器内 GPU 可见、执行 CUDA tensor 和 Paddle 官方检查，并实际加载
检测与行人属性模型。首次启动可能下载模型，需预留外网、磁盘和启动时间，或把
已授权模型预置到持久 volume。自检通过后再把 `.env` 的
`RICH_LABEL_ENABLED=true` 并启动完整 Compose；`/health/live` 本身不能证明
`sm_120` kernel 或模型推理成功。

交付前可用 `tools/rich_label_benchmark.py` 对同一视频执行真实 HTTP worker 测速；
它会输出 `metrics.json`、逐帧 `frames.jsonl` 和 `annotated.mp4`，用于核对超时、
吞吐、框和 ID 连续性。冷启动会包含模型加载，热态指标应在短预热后另跑一次。
本地 CPU 结果作为功能和降级基线；5090 上还须检查首帧、热态 p50/p95、吞吐、
显存及至少 10 次冷启动。当前 7.5 FPS 目标的最低验收线为完整视频零失败、零
429、吞吐不低于 7.5 FPS、p95 小于 133 ms，并对 CPU/GPU 同帧输出做抽样复核。
`P...`/`O...` 仅为单次视频会话内的跟踪 ID，并非真实身份；年龄是区间估计，
gender 是外观模型估计。PaddleX/PaddleDetection 源码为 Apache-2.0，但商用或
随交付包分发前仍须逐一确认具体预训练权重及其训练数据的许可条件。

## 可选 InsightFace GPU 加速

基础 Harness 镜像安装 CPU 版 ONNX Runtime；服务器即使有 RTX 5090，也不会
自动加速人脸识别。GPU 交付须额外加载 `Dockerfile.face-gpu` 构建的 Harness
镜像，并叠加 `compose.face-gpu.delivery.yaml`。该镜像固定 ONNX Runtime GPU
1.26（CUDA 12.8、cuDNN 9），避免 1.27 及以后默认 CUDA 13 wheel 带来的运行时
漂移。

先在客户机执行真实模型自检，再启动服务：

```bash
docker compose -f compose.delivery.yaml -f compose.face-gpu.delivery.yaml \
  run --rm harness python tools/face_gpu_smoke.py
docker compose -f compose.delivery.yaml -f compose.face-gpu.delivery.yaml \
  up -d --force-recreate
```

自检要求已授权的 detection/recognition 权重存在于 `face-models` volume；仅在
许可覆盖的非商业测试中，才可临时增加
`-e FACE_ALLOW_MODEL_DOWNLOAD=true`。Harness 会核对两条实际 ONNX Session 的
首选 provider，CUDA 缺失或静默回落 CPU 时直接失败。若同时启用 Paddle rich
worker，可再叠加 `compose.rich-gpu.delivery.yaml --profile rich`；两容器共享同一
5090，应分别记录 face-only 与并跑时的热态 p50/p95、吞吐和显存。

GPU 能显著减少单次人脸推理及 CPU 争用，但 live scheduler 仍保持 single-flight：
混合检测继续保留 LLM 六并发，忙碌的人脸帧直接丢弃而不排队；仅选陌生人时，
Harness 自动将分析并发降为 1。这样 GPU 故障或回退也不会重新形成六深度队列。

## 输入与永久设置

默认输入：

```text
webrtc+http://192.168.20.199:1985/rtc/v1/whep/?app=live&stream=livestream
```

机器人视频是 2560×720 左右双目拼接流。交付配置中的
`stream.frame_crop: left_half` 会在任何缩放、模型推理和 bbox 坐标归一化之前
取左眼 1280×720 画面，因此控制台、分析结果和标注输出共享同一坐标系。

服务器和容器必须能到达机器人 TCP 1985，并能接收机器人配置的 WebRTC 媒体
端口及局域网 ICE candidate。控制台可以永久保存视频地址、测试/正式模式、
自动启动和模型 API Key。Key 不会由页面或 API 回显。

永久设置位于 `/data/deployment-settings.json`，Key 位于
`/data/model-api-key`，已知人员库同样位于 `face-data` volume；本地人脸分析
文件位于独立的 `face-models` volume。

## 局域网输出

交付环境监听 `0.0.0.0:8080`：

```text
http://192.168.20.168:8080/v1/results/events
http://192.168.20.168:8080/v1/results/latest
http://192.168.20.168:8080/v1/video/annotated/stream.mjpeg
```

诊断端点为 `/health/live`、`/health/ready` 和 `/v1/status`。输入协议变更不会
影响输出协议或路径。`/health/ready` 只有在收到新帧并产生分析结果后才返回
200。

控制台不得暴露到公网。请用服务器防火墙仅允许受信任局域网访问 TCP 8080，
或使用带认证的 HTTPS 反向代理。Compose 已启用只读根文件系统、非 root
用户、`no-new-privileges` 和日志轮转。
