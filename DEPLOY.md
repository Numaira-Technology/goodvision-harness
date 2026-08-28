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
