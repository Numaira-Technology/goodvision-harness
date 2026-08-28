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

源码部署可用独立的 PaddleX/PaddleDetection sidecar；在 `.env` 设置
`RICH_LABEL_ENABLED=true` 后运行：

```bash
docker compose --profile rich up --build
```

它只为延迟展示流提供人物/物品跟踪框和外观属性，不参与也不阻塞 VLM 安全
判断；worker 超时或不可用时 Harness 会继续输出 VLM 结果与原始视频。交付版
`compose.delivery.yaml` 不内置 worker，若启用须单独交付兼容镜像、加入同一
Docker 网络，并把 `RICH_LABEL_ENDPOINT` 指向其 `/v1/labels`。

随源码提供的是 CPU 镜像，默认使用低延迟的 `PicoDet-S`；只有目标机器具备足够
推理余量时才建议设置 `RICH_DETECTOR_MODEL=PicoDet-M`、`PicoDet-L` 或
`PP-YOLOE_plus-S`。镜像固定 Paddle 3.2.2，以避开 Paddle 3.3.x 已确认的 CPU
oneDNN 回归。GPU 需要另行构建匹配的 Paddle GPU 镜像。首次启动可能下载模型，
需预留外网、磁盘和启动时间，或把已授权模型预置到持久 volume。

交付前可用 `tools/rich_label_benchmark.py` 对本地视频执行真实 HTTP worker 测速；
它会输出 `metrics.json`、逐帧 `frames.jsonl` 和 `annotated.mp4`，用于核对超时、
吞吐、框和 ID 连续性。冷启动会包含模型加载，热态指标应在短预热后另跑一次。
`P...`/`O...` 仅为单次视频会话内的跟踪 ID，并非真实身份；年龄是区间估计，
gender 是外观模型估计。PaddleX/PaddleDetection 源码为 Apache-2.0，但商用或
随交付包分发前仍须逐一确认具体预训练权重及其训练数据的许可条件。

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
