# 现有视频转 WebRTC 实测

日期：2026-09-21。结论：本轮未复现输入丢包和雪花，但主进程计算压力下的输出降帧仍然存在。
仅验证当前工作区源码，没有部署或启动原 8080 服务。

## 实际测试路径

使用现有 `tmp/sample-video-setup/sample-sdr.mp4`，1808×1030、BT.709、约 55.78 秒。
原文件包含 B 帧且帧率不固定；为兼容 WebRTC，生成保留分辨率的独立 H.264 Baseline
副本：30 FPS、无 B 帧、关键帧间隔 1 秒、目标码率 3 Mbps。原文件未修改。

测试路径为：视频副本循环播放 → FFmpeg RTSP/TCP 发布 → 临时 MediaMTX 1.21.0
网关 → **真实 WHEP/ICE/DTLS/SRTP UDP** → 产品 NetworkStreamAdapter 接收子进程 →
主进程 JPEG 编码 → SceneRuntime 与 ContinuousTrackedSceneDisplayWorker →
FrameBus → 产品 `_mjpeg_response` → 独立进程 HTTP 客户端。

网关日志确认 UDP ICE 候选和 H.264 视频轨道，未将 RTSP 直接接收冒充 WebRTC。
测试实例位于本机 Docker 网络；HTTP 客户端是独立进程，走真实 HTTP multipart MJPEG。
未经过局域网另一台物理设备，所以不能据此验收物理 LAN 或原摄像头网关。

本地 Contour 状态为 `model_staged`、`gpu_ready=false`、未配置推理引擎。
因此本轮**没有运行 SAM、人脸或 VLM，也没有生成假轮廓**；展示链路运行空标注的真实
SceneRuntime 与生产合成/HTTP 输出函数，使用四个 JSON 计算线程测试主进程竞争。
这不是完整 HarnessRuntime 的整套分析验收。

## 方法与结果

从首帧起共 105 秒：空载 20 秒 → 四线程计算压力 70 秒 → 撤掉压力恢复 15 秒。
覆盖源视频循环边界。独立客户端检查每个 JPEG 长度、结束标记和解码，记录实际帧到达时间，
并生成保留到达时间间隔的 904×514 H.264 回放。所有 HTTP JPEG 检查使用原生分辨率。

| 指标 | 空载 | 计算压力 | 撤掉压力 |
| --- | ---: | ---: | ---: |
| 主进程获得完整图像，FPS | 15.04 | 10.47 | 15.01 |
| HTTP 客户端收到的不同画面，FPS | **12.83** | **6.60** | **13.31** |
| HTTP 收到帧数 | 255 | 461 | 201 |
| HTTP 最大到帧间隔 | 188 ms | **538 ms** | 98 ms |
| HTTP 到帧间隔 P95 | 120 ms | 271 ms | 88 ms |
| 无效 JPEG | 0 | 0 | 0 |

- WebRTC 收到 **37,002 RTP 包，packets_lost=0**。
- 系统 **UDP RcvbufErrors 增量为 0**；按接收子进程 FD/inode 关联的具体 UDP socket，
  每秒样本中 drops 均为 0，接收队列峰值 13,824 字节。
- 解码 3,154 帧，稳定窗口约 30 FPS；损坏标志可用的 3,154 帧中没有 corrupt 标记。
  15 FPS 采样交付 1,577 帧，另 1,577 帧为正常采样跳过，不是网络丢包。
- 主进程复制繁忙时跳过 203 张完整图像；展示收到 1,259 帧、发布 934 帧，
  325 帧被更新的待展示完整帧替换。客户端最终收到 917 张不同画面，HTTP/JPEG 检查无错误。
- 事件循环接收与解码没有随压力出现持续延迟增长；结尾 PTS 延迟增长约 -24 ms。
- 每 5 秒采样的合成耗时：空载中位数约 81 ms、压力期约 **140 ms**、恢复后约 25 ms。
  这是稀疏日志样本，不是逐帧耗时分布，也未分解每个函数的 CPU 与等待时间。

结论指向两条独立路径：隔离后的压缩视频接收保持正常；主进程取完整帧、JPEG 编码、
合成和 HTTP 输出仍受计算竞争影响。剩余降帧不能解释为本轮网络丢包。下一步应针对
这段图像处理和输出路径测量/隔离，尤其避免分析计算阻塞原始画面交付与合成。

## 画面校验

每约 10 秒保留一张从 HTTP 收到的原生 JPEG，共 11 张。对每张自动定位到兼容视频副本
中最相近的原始帧，再比较完整分辨率像素：平均绝对误差 0.636～0.696（0～255 标度），
PSNR 47.42～48.11 dB。人工查看压力期 30 秒、60 秒和恢复期 101 秒截图，未见雪花。
这支持本轮没有出现历史上的大面积压缩损坏；它不等于逐帧人工检查了全部录像。

## 产物与结束状态

私有产物位于 `.goodvision/diagnostics/sample-webrtc-20260921/`：

- `received-playback.mp4`：实际 HTTP 拉流录像，约 104.95 秒、917 帧，保留到达间隔。
- `http-*.jpg`：原生分辨率抽检图像。
- `service.log`、`gateway.log`、`publisher.log`：接收、合成与转流日志。
- `metrics.jsonl`：系统 UDP、接收 socket、网卡流量及负载采样。
- `capture-frames.jsonl`、`http-frames.jsonl`：主进程和实际 HTTP 客户端逐帧时间记录。
- `validation-summary.json`、`stage-summary.json`：接收、输出及画面对照汇总。
- `run_video_probe.py`、`analyze_video_probe.py`：本轮可复核测试方法。

结束时测试 HTTP 服务使用 1 秒有界关闭，主动取消仍等待下一帧的长连接，日志中出现
Uvicorn shutdown cancellation；客户端完整退出、录像成功封装，无运行期 HTTP 解码失败。
末尾展示 stats 的 5.307 秒表示停帧后等待关闭期间的画面年龄，不是运行期固定输出延迟。

本轮创建的发布器、WebRTC 网关和接收测试容器均已关闭。原有容器保持原状态；
8080 没有启动，没有新建云 GPU，也没有自动开始真实摄像头监控。
