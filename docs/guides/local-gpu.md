# 本机 NVIDIA GPU 人物追踪

主程序所在机器有 NVIDIA GPU 时，运行下面的启动程序即可使用“人物追踪 → 本机显卡”。它启动主程序和一个本地 SAM 服务，复用 **Contour 的 Docker 镜像、固定版本模型代码、CUDA 依赖、预热检查及多路追踪引擎**。人物画面通过 Docker 内网交给这台机器上的 GPU。

大模型沿用云端 API 或已有独立模型服务器配置。默认不启动本地大模型，也不要求下载大模型权重；人脸识别继续使用主程序 CPU 环境。需要把这些服务也部署到本机 GPU 时，使用[完整本地模型部署](local-gpu-full.md)中的 `--with-local-inference`。

## 准备

- Python 3.10+、Docker Compose 2.24.4+，以及可运行 `nvidia-smi` 的 NVIDIA 驱动。
- Linux 配好 NVIDIA Container Toolkit；Windows 使用 Docker Desktop 的 WSL2 后端和 Linux 容器，与 Contour 一样。
- 已准备的 SAM3.1 权重。默认优先复用 `workers/contour/models/contour.pt`，其次读取 `.models/sam3.1/sam3.1_multiplex.pt`。也可用 `--checkpoint` 或 `SAM31_CHECKPOINT_FILE` 指定文件。已有 Contour 模型包可直接使用，不需要复制第二份权重。
- 尚未准备模型时，按 [Contour 模型说明](../../workers/contour/README.md#下载并接入模型)运行下载工具。启动程序本身不下载权重。首次构建镜像需要网络；已有镜像可加 `--no-build`。

## 启动

在主程序仓库根目录执行，Windows 和 Linux 使用同一入口：

```sh
# 默认单路：检测显卡、构建镜像、检查容器 CUDA、加载 SAM 并启动主程序。
python tools/start_local_gpu.py

# 预置 3 路；每路状态独立，共享一份 SAM 模型。
python tools/start_local_gpu.py --streams 3

# 指定显卡和已有模型，使用已导入的镜像。
python tools/start_local_gpu.py --gpu 1 --checkpoint /path/to/contour.pt --no-build

# 开发机查看计划，不调用 Docker、不写入文件、不要求存在权重。
python tools/start_local_gpu.py --dry-run --gpu-count 1 --streams 3
```

打开 **http://127.0.0.1:8080**。新部署默认人物追踪为“本机显卡”，大模型为“云端 API”。已有控制台保存的选择继续优先：停止全部分析后，将人物追踪改为“本机显卡”并保存。配置云端 API key、连接视频，然后点击“开始全部分析”。停止分析会关闭追踪会话并保留视频预览，再次开始会创建新会话。

`--base-port 9000` 修改唯一的控制台端口。N 路都在同一个页面操作，SAM 和额外视频实例不发布主机端口。增加路数前停止分析，再用更大的 `--streams N` 启动。路数是预置容量，实际吞吐和显存需求需用目标 GPU 与视频实测。

默认使用检测到的第一张 NVIDIA GPU，`--gpu` 可以选择其他实际设备编号。即使存在两张卡，也不会因为只有 SAM 就强制使用第二张。容器只看得到选定的那张卡，内部设备编号为 0。

可选 `.env` 配置，命令行指定值优先：

```dotenv
LOCAL_TRACKING_GPU_ID=0
LOCAL_TRACKING_FPS=2
LOCAL_TRACKING_TORCH_CUDA=cu128
SAM31_CHECKPOINT_FILE=./workers/contour/models/contour.pt
# 大模型仍使用现有 INFERENCE_PROVIDER / API key。
# 使用独立大模型服务器时：
# INFERENCE_EXECUTION_MODE=server
# INFERENCE_SERVER_URL=http://model-server:8000
# INFERENCE_SERVER_MODEL=local-vision
```

`LOCAL_TRACKING_TORCH_CUDA=cu126` 可为需要 CUDA 12.6 的驱动构建对应镜像；它不会修改已经导入的镜像。启动时用实际 worker 镜像在选定 GPU 上执行 CUDA 张量运算，失败则不启动新的主程序。之后还要等 SAM 权重加载和预热成功，HTTP 存活不足以判定服务可用。缺少驱动、权重或 CUDA 不兼容会返回错误，不会自动租用云 GPU。

本机追踪和本机大模型是独立能力。这个默认部署只在页面启用本机追踪选项；若之前保存过本机大模型选项，改回云端 API 或独立模型服务器后即可继续。

## 运维

启动计划包含本次的 GPU、模型路径和 Compose 命令；不包含 API key。多路配置写入 `.goodvision/local-gpu-compose.yaml`。单路沿用 `face-data`；多路沿用 `camera-01-data` 至 `camera-NN-data`。单路与多路的数据卷独立，切换后需在页面确认对应配置。

查询与停止服务时，带上启动时的环境和 Compose 文件；多路再增加 `-f .goodvision/local-gpu-compose.yaml`：

```sh
# 将示例值换成本次启动计划输出的值，或写入用于运维的本地 .env。
# LOCAL_GPU_COUNT=1
# LOCAL_TRACKING_GPU_ID=0
# SAM31_CHECKPOINT_FILE=./workers/contour/models/contour.pt
docker compose -f compose.local-tracking.yaml logs --tail=80 local-tracking
docker compose -f compose.local-tracking.yaml stop local-tracking
```

页面切换运行位置只改变请求路由。SAM 容器持续保留模型，释放显存需停止 `local-tracking`，恢复本机追踪前重新运行启动程序。启动程序不删除旧容器或数据；从旧完整 GPU 部署切换时，先停止不再使用的 `local-inference` 等服务，缩减路数时也应先停止多余视频实例。

完全离线运行需提前准备 `goodvision-harness:demo`、`goodvision-contour-runtime:0.1.0`、SAM 权重和人脸模型卷，再使用 `--no-build`。现有普通云端和独立 Contour 服务部署方式不变。
