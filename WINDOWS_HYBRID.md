# Windows non-NVIDIA GPU experiment

This experiment runs the complete Harness natively on Windows so InsightFace
can use the laptop GPU through ONNX Runtime DirectML. The complete Paddle rich
label worker stays in its normal Linux Docker container on CPU. Browser camera
frames enter the native Harness at `http://127.0.0.1:8082`; no camera device is
passed into Docker.

This path is only for non-NVIDIA Windows GPUs. An NVIDIA Linux/Windows host
should use the existing CUDA overlays so InsightFace and Paddle can both use
CUDA. Paddle's supported GPU packages do not use DirectML, which is why it
remains on CPU here.

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

Docker Desktop must be running. The script builds and starts only the CPU
Paddle worker, waits for its real model warmup to finish, and then starts the
native Harness with these fixed experiment settings:

- `FACE_PROVIDER=DmlExecutionProvider`;
- `FACE_LIVE_SCHEDULER=full`, so every admitted analysis window runs current
  InsightFace inference instead of the CPU single-flight sampling policy;
- `RICH_LABEL_ENABLED=true`, with the complete CPU worker reached through a
  loopback-only port;
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

On later runs, skip the Docker image build while retaining the real worker
warmup/readiness check:

```powershell
.\tools\start_windows_hybrid_demo.ps1 -NoRichBuild
```

If ports are occupied, use `-HarnessPort` and `-RichPort`. To stop the retained
CPU worker after the experiment:

```powershell
$env:RICH_HOST_PORT = "8091"
docker compose -f compose.yaml -f compose.rich-cpu-host.yaml --profile rich stop rich-label-worker
```

Do not compare only the overall result latency. Record the same camera scene
and feature set for CPU and DirectML, then compare face inference p50/p90,
fresh stranger results per minute, VLM latency, Paddle latency, failures, and
dropped windows. The DirectML result answers whether moving InsightFace off the
CPU removes local contention; it does not substitute for the RTX 5090 CUDA
smoke and performance test.
