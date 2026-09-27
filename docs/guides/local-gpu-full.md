# 本地大模型、SAM 和 GPU 人脸的完整部署

本文适用于 `python tools/start_local_gpu.py --with-local-inference`。仅需要本机 SAM 人物追踪时，使用[默认本机 GPU 部署](local-gpu.md)，无需下载或启动本地大模型。

两张 L40S 的分配固定为 GPU 0 跑视觉大模型，GPU 1 跑 SAM3.1 人物追踪、人脸识别，以及启用时的 Paddle 工作进程。只有一张 NVIDIA 卡时，全部共享第一张卡。超过两张时仍使用前两张。每个容器只暴露指定的一张卡，容器内部编号均为 0；大模型 tensor parallel 固定为 1。

所有摄像头在**同一个控制台页面**配置、预览、启动和停止。默认预置 1 路容量；`--streams N` 接受正整数，为主控制台预置 N 个独立视频运行实例，共享一个 vLLM 服务与一个 SAM 模型，各路保留独立状态和数据。**12 路是容量参考值，并非软件上限**；可预置少于或多于 12 路，实际可运行数量取决于硬件与实测。

顶部可分别选择大模型和人物追踪的运行位置。人物追踪选择“本机显卡”后，可在同一页运行不超过已预置容量的多路视频，**大模型使用云端时也支持多路**。人物追踪还可选择“独立 GPU 服务器”，主程序无需与显卡在同一台机器，具体见 [模块化部署说明](modular-services.md)。只有“原部署服务”保留单路限制；没有对应远程服务配置时不会提供替代追踪能力。改变运行位置前须停止全部分析并完成收尾。已保存选择优先于启动默认值，切换人物追踪位置不会改变此同机部署中人脸等其他工作的显卡分配。

各路沿用每路 1.25 次模型分析/秒，按实际启用路数验收，本文以 12 路作为容量测试参考。**尚未在两张 L40S 上证明 12 路能持续达到每路 1.25 次模型分析/秒**，也未证明更高路数的吞吐。健康检查通过表示服务可用，不等于吞吐和准确率达标。

## 准备

- Linux、NVIDIA 驱动、NVIDIA Container Toolkit、Docker Compose **2.24.4 或更新版本**、Python 3.10 或更新版本。Windows 需使用支持 NVIDIA GPU 的 WSL2 / Docker Desktop Linux 容器。Compose 的 `!override` 保证只有主控制台发布主机端口，并清除额外摄像头继承的端口。[Compose 合并规则](https://docs.docker.com/reference/compose-file/merge/#replace-value)。
- `nvidia-smi` 必须能读取 NVIDIA 显卡。AMD、Intel、Apple 或驱动不可用时视为没有可用本地显卡，脚本报错，继续使用普通云端部署。
- 权重和 Docker 镜像需提前准备。SAM/人脸沿用仓库现有 CUDA 镜像依赖；驱动需兼容这些依赖，不能只根据“存在 NVIDIA 卡”判断可运行。

本地 VLM 默认使用客户选择的 `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`，默认目录为 `.models/qwen3-vl-30b-a3b-instruct-fp8`。这是支持图像输入的 MoE Instruct 模型，官方仓库提供已完成 FP8 量化的权重，无需先下载 BF16 权重再自行量化。云端模型和线路配置继续沿用已有设置。[官方模型说明](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)。

在联网准备机器的项目根目录下载**完整仓库**，固定为已核实的提交 `d9748a51ae66354c4dad665aab2c71f26cf2c8cd`，避免 `main` 后续变化影响交付。以下命令使用 Hugging Face 官方 CLI；下载、校验都在联网准备阶段执行，运行客户程序时只读取本地目录。[固定版本文件清单](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8/tree/d9748a51ae66354c4dad665aab2c71f26cf2c8cd)、[官方 CLI 下载和校验说明](https://huggingface.co/docs/huggingface_hub/guides/cli)。

```sh
python -m pip install --upgrade huggingface_hub
hf download Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --revision d9748a51ae66354c4dad665aab2c71f26cf2c8cd \
  --local-dir ./.models/qwen3-vl-30b-a3b-instruct-fp8
hf cache verify Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --revision d9748a51ae66354c4dad665aab2c71f26cf2c8cd \
  --local-dir ./.models/qwen3-vl-30b-a3b-instruct-fp8 \
  --fail-on-missing-files
```

该版本仓库约 **32.3 GB**，须为下载临时文件、镜像及其他模型另留磁盘空间。交付时整体复制上述目录，保留 `config.json`、`generation_config.json`、`chat_template.json`、`preprocessor_config.json`、`video_preprocessor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`merges.txt`、`vocab.json`、`model.safetensors.index.json`，以及 `model-00001-of-00004.safetensors` 至 `model-00004-of-00004.safetensors` 四个完整分片；同时保留仓库说明与许可信息。只有 Git LFS 指针、单个分片或词表都不是完整模型包。项目代码不包含这些权重，准备命令完成且校验通过后才能离线交付。[官方版本文件](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8/tree/d9748a51ae66354c4dad665aab2c71f26cf2c8cd)。

SAM 权重沿用 `.models/sam3.1/sam3.1_multiplex.pt`。需要审批的权重下载在准备机器上完成。人脸和可选 Paddle 的已有模型卷也应提前准备；基础配置允许的首次人脸/Paddle 模型下载不会因为使用这份部署文件自动变成离线。

## 启动和摄像头配置

在项目根目录执行，启动脚本只依赖 Python 标准库：

```sh
# 仅查看本机分配和默认单路启动计划，无 Docker 调用、下载或文件写入。
python tools/start_local_gpu.py --with-local-inference --dry-run

# 没有 GPU 的开发机器上模拟两卡计划；只允许用于 dry-run。
python tools/start_local_gpu.py --with-local-inference --dry-run --gpu-count 2 --streams 12

# 默认 1 路：检测显卡、检查权重、构建并启动，沿用普通主程序的数据卷和视频配置。
python tools/start_local_gpu.py --with-local-inference

# 为同一控制台预置所需容量；以下为不同部署规模的示例，任选其一。
python tools/start_local_gpu.py --with-local-inference --streams 3
python tools/start_local_gpu.py --with-local-inference --streams 6
python tools/start_local_gpu.py --with-local-inference --streams 12
python tools/start_local_gpu.py --with-local-inference --streams 16

# 所有镜像已导入时跳过构建。
python tools/start_local_gpu.py --with-local-inference --no-build --streams 12

# 需要旧 Paddle 服务时额外启动，仍使用追踪所在卡。
python tools/start_local_gpu.py --with-local-inference --with-paddle --streams 12
```

`--streams` 表示预置容量；省略或设为 1 时使用单路部署。N ≥ 2 时生成 `.goodvision/local-gpu-compose.yaml`，组合 `compose.yaml` 与 `compose.local-gpu.yaml` 后启动所需实例。生成文件只有配置模板，不包含摄像头 URL 或云端密钥的实际值。启动命令会等待健康检查，并输出统一控制台地址 `console_url` 和预置路数 `provisioned_lanes`。

无论预置多少路，**只需打开 `http://127.0.0.1:8080` 这一个页面**；额外摄像头、VLM 和 SAM 不发布主机端口。主控制台通过同一地址转发各路画面和控制请求。`--base-port 9000` 将统一入口改为 9000，对单路和多路均有效；未指定时沿用 `HARNESS_PORT`，否则使用 8080。仅这一个端口需要有效且未被占用，路数不会再消耗连续的主机端口。

每路先分别“连接视频”查看预览，此时不会调用模型。所有路均有“详细控制”，可以独立保存检测项目、敏感度和查看该路结果。确认所需视频都已连接后，使用页面上的“开始全部分析”；未连接的预留通道不参与。本次分析期间不能单独启停某一路，也不能连接、断开视频或修改检测与服务配置。需要调整时先“停止全部分析”，等待收尾完成；视频预览会保留，再按需修改并统一重新开始。

进入主控制台后，保存人物追踪运行位置，按需要选择不超过预置容量的路数，再在各路卡片配置视频地址并连接预览，最后统一开始分析。选择原部署追踪服务时页面仅允许单路。页面不会创建 Docker 容器；需要超过当前容量时，先停止相关视频，再用更大的 `--streams N` 重新运行启动脚本。

新建多路实例默认不自动采集视频。各路地址可在统一页面保存，也可在现有 `.env` 中设置初始值；额外路由主控制台启动，不会单独自动采集而绕过多路开关：

```dotenv
LOCAL_CAMERA_01_STREAM_URL=rtsp://camera-01.example/live
LOCAL_CAMERA_01_AUTOSTART=true
LOCAL_CAMERA_02_STREAM_URL=rtsp://camera-02.example/live
# 按实际 N 路继续配置到 LOCAL_CAMERA_NN_...（编号至少两位，如 03、16、100）
```

统一页面中为各路保存的设置优先于环境初始值。多路部署的数据保存在 `camera-01-data` 至 `camera-NN-data` 命名卷中，人脸模型卷共享。例如预置 3 路建立 3 个独立数据卷，6 路建立 6 个。单路部署沿用原 `face-data` 卷，多路不会覆盖其中的配置；从单路迁移时在统一页面重新配置各路。每路默认一个人脸推理实例，因此预置 N 路仍需考虑 N 份人脸模型的显存；显式设置 `FACE_INFERENCE_INSTANCES` 可覆盖默认值。

不要叠加其他 GPU 分配或远程租卡 Compose 文件。切换已有部署或缩减路数前，应停止不再使用的旧 worker/摄像头实例，防止它们继续占用显存；启动脚本不删除旧容器或数据。

## 模型配置与 vLLM 参数

保留原有 `INFERENCE_*`、供应商密钥和远程 `RICH_LABEL_*` 配置，在 `.env` 中新增需要覆盖的本地项：

```dotenv
LOCAL_INFERENCE_ENABLED=true
LOCAL_TRACKING_ENABLED=true
LOCAL_INFERENCE_MODEL=local-vision
LOCAL_INFERENCE_MODEL_DIRECTORY=./.models/qwen3-vl-30b-a3b-instruct-fp8
LOCAL_INFERENCE_MAX_MODEL_LEN=8192
LOCAL_INFERENCE_MAX_TOKENS=1600
LOCAL_INFERENCE_MAX_BATCHED_TOKENS=4096
LOCAL_TRACKING_FPS=2
SAM31_CHECKPOINT_FILE=./.models/sam3.1/sam3.1_multiplex.pt
```

`LOCAL_INFERENCE_MODEL` 是请求中的服务别名；实际模型由目录内容决定。相对权重路径从项目根目录解析。启动脚本读取的本地变量支持单行字面值和引号路径，不展开 `${OTHER_VARIABLE}`；进程环境优先于 `.env`。`LOCAL_CAMERA_XX_*` 直接由 Compose 读取。恢复远程 SAM 时还需保留 `RICH_LABEL_BACKEND=sam31_stream`、`RICH_LABEL_ENABLED=true`、远程 `RICH_LABEL_ENDPOINT` 和 `SCENE_ENGINE_MODE=active`，或在原 YAML 中配置对应值；未设置环境覆盖时保留 YAML 中已有的后端和场景模式。

默认镜像固定为 `vllm/vllm-openai:v0.17.1`，其支持列表包含 `Qwen3VLMoeForConditionalGeneration`，FP8 支持范围包含 L40S 所用的 Ada 架构。vLLM 从模型配置读取已有 FP8 量化信息；这里使用 Instruct 模型，不启用 Thinking 模型专用解析。可通过 `LOCAL_INFERENCE_IMAGE` 换成经过实测的版本。[vLLM 支持模型](https://docs.vllm.ai/en/v0.17.1/models/supported_models/)、[FP8 支持说明](https://docs.vllm.ai/en/v0.17.1/features/quantization/fp8/)、[NVIDIA L40S 规格](https://www.nvidia.com/en-us/data-center/l40s/)。

启动脚本按显卡数量选择以下**未经实机压测的起始参数**。这是本项目的初始配置，不是官方对 48 GB L40S 的容量认证；显式设置对应环境变量会覆盖这些默认值。

| 参数 | 1 张卡：大模型与追踪共享 | 2 张或更多卡：大模型独占 GPU 0 |
| --- | --- | --- |
| `LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION` | `0.80` | `0.90` |
| `LOCAL_INFERENCE_MAX_NUM_SEQS` | `4` | `8` |
| `LOCAL_INFERENCE_EXECUTION_MODE` | `--enforce-eager` | `--no-enforce-eager` |
| `LOCAL_INFERENCE_MAX_BATCHED_TOKENS` | `4096` | `4096` |
| `LOCAL_INFERENCE_MAX_MODEL_LEN` | `8192` | `8192` |
| `LOCAL_INFERENCE_MAX_TOKENS` | `1600` | `1600` |

默认启用连续批处理和分块预填充。单卡先使用 eager，避免 CUDA graph 的额外显存开销；双卡大模型独占一张卡时默认启用 CUDA graph，显存紧张时也可显式设为 `--enforce-eager`。更改后重新测量速度。并发数和批量越大，可能需要更多缓存；若发生缓存抢占和重算，反而可能降低吞吐。仍使用普通 KV 缓存，不默认启用 FP8 KV 或推测解码。[vLLM 显存说明](https://docs.vllm.ai/en/v0.17.1/configuration/conserving_memory/)、[官方优化说明](https://docs.vllm.ai/en/v0.17.1/configuration/optimization/)。

默认大模型最多接受 16 张图、图像处理器像素上限 262144，不接受视频文件输入；程序仍按现有方式发送视频抽帧。模型输入分辨率不会改变本地视频或证据原图。

双卡大模型显存占用比例默认 `0.90`，单卡默认 `0.80`；若 `.env` 显式设置 `0.90`，迁移单卡时也会保留这个值，需要重新核对。**单卡共享显存必须先核算**：一张 48 GB L40S 还需同时容纳 SAM、各路人脸模型、视觉激活和缓存，给 VLM 分配 80% 并不保证剩余空间足够。“支持一张卡”表示支持这种设备分配，不保证该模型与全部功能能装入任意单卡。启动前按实际权重大小检查 VLM 显存预算，只能排除明显放不下的配置，不能证明运行峰值可容纳；单卡可能仍在加载 SAM 或增加摄像头后出现显存不足。需要时使用显存更大的 NVIDIA 卡，并重新验证全部功能和实际路数。显存不足会报告失败，不会自动更换模型、挪用追踪显卡或切换付费云服务。

`A3B` 表示每个 token 激活约 3B 参数，不能据此按 3B 模型估算显存。本部署固定 tensor parallel 为 1，未启用 CPU 权重卸载，因此全部 30B 级权重仍需驻留大模型所在显卡；此外还需要视觉编码、运行时工作区和 KV 缓存。32.3 GB 是仓库文件大小，不是运行时显存峰值。官方配置采用 `fp8`、`e4m3`、128 × 128 分块，并保留部分未量化层。[Qwen 参数命名说明](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md)、[本模型固定版本配置](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8/blob/d9748a51ae66354c4dad665aab2c71f26cf2c8cd/config.json)。

## 按实际路数计算容量和验收

以实际启用的 N 路计算，VLM 目标为 **1.25 × N 请求/秒**，默认 SAM 追踪输入为 **2 × N 帧/秒**。VLM 和追踪是两个不同预算：

| 实际启用路数 N | VLM 请求/秒 | SAM 输入帧/秒 | 统一控制台端口 |
| --- | --- | --- | --- |
| 1（默认） | 1.25 | 2 | 8080 |
| 3 | 3.75 | 6 | 8080 |
| 6 | 7.5 | 12 | 8080 |
| 12（容量参考） | 15 | 24 | 8080 |
| 16 | 20 | 32 | 8080 |

下表列出 12 路参考规模的容量预算；其他路数按 N 计算负载，更高路数能否达标需重新压测：

| 工作 | 12 路目标 / 初始配置 | 实机需要观测 |
| --- | --- | --- |
| VLM 分析 | 每路 1.25 次/秒，共 **15 请求/秒** | 成功吞吐、排队、P95 延迟、超时和显存 |
| VLM 输出 | `max_tokens=1600` 是单请求上限 | 若每次用满，上限负载达 24,000 输出 token/秒；不能当作已支持容量 |
| SAM 追踪 | `LOCAL_TRACKING_FPS=2`，共 **24 输入帧/秒** | 实际完成帧率、每路等待、公平性、追踪质量和显存 |
| 人脸和其他本地工作 | 与 SAM 同在 GPU 1 | 和追踪并行时的总显存及延迟 |

事件输出通常短于最大 token 上限，应使用客户实际图像、提示词和事件返回长度测量。**尚未验证单张 L40S 专用于 Qwen3-VL-30B-A3B-Instruct-FP8 时，能否持续完成 12 × 1.25 = 15 个真实多模态请求/秒**；另一张卡承担追踪也不会增加这张卡的模型吞吐。模型权重能放入显卡和激活参数较少都不能代替该项压测。模型默认并发为单卡共享时 4 个、双卡独占时 8 个请求，剩余请求排队；`LOCAL_INFERENCE_MAX_NUM_SEQS` 是请求并发设置，不是视频路数上限。主程序每路原有最多 4 个模型请求在途，N 路可能有 4 × N 个在途请求，12 路时为 48 个。排队超过既有超时会失败，需要按结果调整并发、频率或硬件，不能用无限增大队列掩盖过载。

SAM 按实际 N 路维护独立 session，底层模型共享，GPU 操作有界串行调度；这避免为每路复制 SAM 权重，但并不承诺 12 路时达到 24 帧/秒。人脸模型仍在各路主程序中加载，需要把这些实例一起计入 GPU 1 的显存。控制台保存的采样帧率优先于启动默认值，并同步到各路；采样负载不能与 VLM 的请求速率混算。

1. 先用默认单路或 `--streams 1` 验证驱动、权重、图像理解、人物追踪和事件分析，然后按实际需求增加路数；例如逐步升到 3、6、12 路，需要更多时继续测 16 路等实际规模。
2. 在统一页面测试所需的运行位置组合：人物追踪使用本机或独立 GPU 服务器时，云端或自备大模型均可多路；追踪选择原部署服务时只允许第 1 路。逐路连接应只有预览，统一开始后所有已连接路进入分析；停止全部分析后画面仍更新。确认分析期间不能改变连接集合、各路检测设置或全局服务配置，停止并完成收尾后才能修改。各类服务器须已部署并配置正确地址。
3. 使用 `nvidia-smi` 核对 VLM 在 GPU 0，SAM、人脸及可选 Paddle 在 GPU 1，且持续运行时无显存增长失控。
4. 使用实际 N 路视频、每路 1.25 次模型分析/秒的输入，持续记录至少成功请求率、P95 延迟、超时、排队、SAM 每路完成帧率及检测质量；12 路参考测试通过不能代替更高路数的实际规模验收。仅服务健康或一次成功不能作为验收结论。

可以先用 `tools/local_gpu_capacity.py` 对 VLM 做独立的定速容量测试。准备 `.goodvision/capacity-requests.json`，内容是一个或一组真实 OpenAI chat-completions 请求：`model` 使用 `local-vision`，保留生产提示词、图像大小和 `max_tokens`，图像通过 `image_url` 的 `data:image/...;base64,...` 内嵌。用一组不同画面轮换，避免重复同一输入导致缓存虚高。该工具只验证请求成功率、吞吐、延迟和队列，不验证事件语义是否正确，也不能代替 SAM 同时运行的整机验收。

VLM 没有主机端口时，在现有 Docker 网络运行测试即可，无需修改服务暴露范围。以下为 12 路参考规模的压测示例，适用于项目根目录下的 Linux / WSL shell，测试结果重定向回主机。测试实际 N 路时设置 `--cameras N` 和 `--max-in-flight 4*N` 的计算结果：3 路用 `--cameras 3 --max-in-flight 12`，6 路用 `--cameras 6 --max-in-flight 24`，12 路用下例，16 路用 `--cameras 16 --max-in-flight 64`；各档保持 `--fps 1.25`。

```sh
docker run --rm --network goodvision-demo --entrypoint python \
  --mount "type=bind,src=$(pwd)/tools/local_gpu_capacity.py,dst=/tmp/capacity.py,readonly" \
  --mount "type=bind,src=$(pwd)/.goodvision/capacity-requests.json,dst=/tmp/requests.json,readonly" \
  goodvision-harness:demo-face-gpu-cu128 /tmp/capacity.py \
  --request-file /tmp/requests.json \
  --endpoint http://local-inference:8000/v1/chat/completions --allow-remote \
  --cameras 12 --fps 1.25 --duration 120 --warmup 10 \
  --max-in-flight 48 --timeout 15 --deadline 15 \
  --min-throughput-ratio 0.98 --max-queue-growth 1 \
  > .goodvision/capacity-result.json
```

此处 `--allow-remote` 仅允许测试工具访问指定的 Docker 内网主机名。默认至少完成目标吞吐的 98%，且测量期间在途请求增长不超过 1；请求失败、超时或队列持续增长都会判失败。退出码为 0 表示该次测量通过，1 表示测量不达标，2 表示输入错误。验收应使用多个实际画面和更长持续时间复测。

## 离线与运维

VLM 和 SAM 都以只读方式挂载已有权重，并开启离线加载；脚本不会下载权重，缺少文件或分片会报错。完全离线交付还需导入 `goodvision-harness:demo-face-gpu-cu128`、`goodvision-contour-runtime:0.1.0`、`vllm/vllm-openai:v0.17.1`，以及需要时的 `goodvision-rich-label-worker:demo-gpu-cu129` 镜像和已准备好的模型卷。

SAM 与默认的本机追踪部署使用同一份 Contour 引擎配置，保留依赖修复、可写缓存和显存回收。启动时同样核对镜像源码、实际检查每路 `/data/sam-spool` 和 GPU 缓存目录可写性，并执行 CUDA 运算；`--no-build` 也不能绕过这些检查。

启动时自动设置 `LOCAL_GPU_AVAILABLE=true`、`LOCAL_GPU_COUNT`、`LOCAL_INFERENCE_GPU_ID` 和 `LOCAL_TRACKING_GPU_ID`。每个视频实例还获得 `HARNESS_FLEET_CAPACITY=N`、自身的 `HARNESS_FLEET_CAMERA_INDEX` 和主入口内网地址 `HARNESS_FLEET_PRIMARY_URL=http://harness:8080`。主控制台统一访问固定的 `harness-02:8080` 等内网实例；这些地址不是需要分别打开的用户入口。

VLM 地址为 `http://local-inference:8000/v1/chat/completions`；单路追踪配置地址为 `http://local-tracking:8092/v1/video/sessions`，由主程序选择现有 `sam_live` 接口。多路分别使用 `http://local-tracking:8092/v1/cameras/camera-01/perception/sam_live/sessions` 等独立路径，健康检查对应 `/health/cameras/camera-01/perception/sam_live`。

两个本地模型服务预先启动，控制台开关只切换请求路由，不需要主程序访问 Docker socket。因此关闭开关不会卸载模型或立即释放显存；释放显存需部署人员停止相应 worker，再次开启该功能前先启动 worker。权重目录、模型参数或路数变更后重新执行启动脚本；控制台运行位置开关不需要重建镜像。
