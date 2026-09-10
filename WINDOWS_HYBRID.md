# Windows non-NVIDIA GPU experiment

This experiment runs the Harness natively on Windows so InsightFace can use
the laptop GPU through ONNX Runtime DirectML. Rich annotation is disabled
unless an external GPU worker is explicitly supplied and verified. Browser
camera frames enter the native Harness at `http://127.0.0.1:8082`.

This path is only for non-NVIDIA Windows GPUs. An NVIDIA Linux/Windows host
should use the existing CUDA overlays so InsightFace and Paddle can both use
CUDA. DirectML availability for InsightFace does not enable Paddle rich
annotation. No CPU rich worker is started by this launcher.

## 1. Create an isolated DirectML environment

Use a separate virtual environment. `onnxruntime`, `onnxruntime-gpu`, and
`onnxruntime-directml` install the same Python package and must not be mixed in
one environment.

```powershell
py -3 -m venv .venv-directml
.\.venv-directml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-directml\Scripts\python.exe -m pip install -e ".[dev,face-directml,webrtc]"
```

Confirm that ONNX Runtime can see DirectML before starting either model:

```powershell
.\.venv-directml\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The output must contain `DmlExecutionProvider`. A CPU-only result is a failed
precondition, not a valid GPU benchmark.

## 2. Start the hybrid demo

The script starts the native Harness directly. Docker Desktop is not required.
The default experiment settings are:

- `FACE_PROVIDER=DmlExecutionProvider`;
- `FACE_LIVE_SCHEDULER=full`, so every admitted analysis window runs current
  InsightFace inference instead of the CPU single-flight sampling policy;
- `RICH_LABEL_ENABLED=false`; face recognition and selected event detection
  remain available;
- `HARNESS_DRY_RUN=false` and `HARNESS_AUTOSTART=false` for a live local camera
  run.

```powershell
.\tools\start_windows_hybrid_demo.ps1
```

Open `http://127.0.0.1:8082`, choose **Camera**, select the desired detections,
and start the session. The existing ignored `.env` may supply `MODEL_API_KEY`;
the script never copies or prints that key. Face models and the enrolled face
gallery remain in the ignored `.models/insightface` and `.goodvision`
directories. The launcher deliberately reuses the existing authorized test
pack at `.models/insightface/models/buffalo_l` instead of downloading a second
copy.

To use an already running NVIDIA GPU rich worker, pass its inference endpoint:

```powershell
.\tools\start_windows_hybrid_demo.ps1 -RichEndpoint "http://127.0.0.1:8091/v1/labels"
```

Rich annotation is enabled only when `/health/ready` returns a successful
response with the JSON boolean `gpu_ready: true`. The launcher checks once,
with a three-second timeout. An unavailable worker or a worker without GPU
readiness leaves rich annotation disabled and continues the face/event demo.
The launcher does not build or start that external worker.

If the Harness port is occupied, use `-HarnessPort`. The former `-RichPort`,
`-RichReadyTimeoutSeconds`, and `-NoRichBuild` options are no longer used;
`-RichEndpoint` replaces the local worker startup path.

Do not compare only the overall result latency. Record the same camera scene
and feature set for CPU and DirectML, then compare face inference p50/p90,
fresh stranger results per minute, VLM latency, failures, and dropped windows.
Keep rich annotation disabled in both runs, or use the same external GPU worker
for both. The DirectML result does not substitute for the RTX 5090 CUDA smoke
and performance test.


## 连续事件与证据存储

即时识别保持原采样与发布节奏，后台事件归并和视频编码不阻塞当前结果。
事件列表与视频接口见 [事件证据说明](EVENT_EVIDENCE.md)。默认每个连续事件
保留首次阳性前1秒后2秒的5 FPS MP4，保留7天且总量不超过10 GiB。

容器使用现有 `/data/event-evidence` 持久目录；Windows 默认在已知人脸库旁的
`event-evidence` 子目录，可用 `EVENT_EVIDENCE_DIRECTORY` 覆盖。需要可执行的
FFmpeg H.264 编码器；交付自检应实际生成、下载并解码视频。

GPU detector-only `/health/tracking` 就绪即可开启 `/v1/tracks`，与属性及界面
标注开关独立。无GPU时保持现有模型人物框/本地人脸位置追踪，不增加必需服务。

旧陌生人照片接口返回410。旧 `face-gallery-strangers.json` 和 `stranger-captures/`
不再加载或写入；确认备份后可由运维清理。保留 `face-gallery.json`、模型和部署
设置，不能通过删除持久卷来升级。归档状态和录制失败计数见 `/v1/status`。
