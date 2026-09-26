# 会议的两个 SAM 云端 CPU 测试

本包落实会议里的两项验收，与 RF-DETR / YOLO 模型选型是不同实验。
两项都使用真实 SAM3.1；可用 HF_TOKEN 下载权重，也可复用已有权重或服务。
不能用历史 SAM 成绩代替这次迁移和双路实测。不会自动租卡。

## 两个测试文件

| 文件（相对包根目录） | 对照条件 | 默认实验量 |
| --- | --- | --- |
| `tools/cloud_cpu_tests/test_1_recording.json` | 单路，不录屏 / 本机录屏 | 0.5、1、1.5、2 秒缓冲，各条件重复 2 次，共 16 轮 |
| `tools/cloud_cpu_tests/test_2_dual_stream.json` | 不录屏，单路 / 同时双路 | 相同缓冲档位和重复次数，共 16 轮 |

每轮播放完整原视频。默认两路使用同一原片、两个独立视频解码器和两个独立 SAM 会话，
用于控制输入差异。两路在本机统一时钟下同步开始；不是把一个模型结果复制显示两遍。
单双路比较复用同一个常驻 SAM 模型，GPU 上的两路请求通过 main 的队列调度。
每个重复周期交替条件顺序（A/B、B/A），减少先后顺序影响。

0.5、1、1.5、2 秒是**显示缓冲档位**，不是预先规定 SAM 推理要花多久。
若都不达标，就报告不达标。需要更多档位时修改对应 JSON 的 `delays_seconds`。
这不改变原来的模型选型包：那个实验仍固定 main 缓冲。

## 部署分工

- **同一云端 Linux 实例**：一个 RTX 4090 SAM worker，两个独立 Harness CPU 进程，
  CPU InsightFace、VLM 请求准备、SAM 输入包装、队列和结果处理。
- **Mac**：原视频解码、上传、浏览器轮廓传播与呈现、测试编排和录屏。
  本机仍有浏览器 CPU 工作；“CPU 上云”指 Harness 服务端计算迁到云端。
- VLM 使用 main 提供的 OpenRouter 配置：`qwen/qwen3.5-35b-a3b:nitro`，
  `OPENROUTER_API_KEY` 在云端设置。它与历史报告的旧 OpenRouter 模型不同；
  本次两组对照统一使用此配置，不将历史数值当成严格配对对照。
  不启用外部 webhook，使用独立空人脸图库和测试输出目录。
- Harness 和 SAM 均只监听云端 loopback，通过 SSH 转发两个 Harness 端口；
  不需要把模型服务或控制接口公开到互联网。

原片仍由 Mac 本地读取，浏览器拦截视频文件请求并支持 Range；不会先从云端重新下载
整段视频。真正的帧上传、模型调用和结果接收都经过云端服务，未替换成模拟结果。

## 1. 云端安装与启动

把 `goodvision-sam-cloud-tests-openrouter.tar.gz` 上传并解压，进入包根目录
`goodvision-4090-benchmark`。需要 Linux x86_64、Python 3.12、CUDA 12.6 兼容驱动，
`nvidia-smi` 能看到 RTX 4090。Ubuntu 24.04 缺少基础依赖时：

```bash
apt-get update
apt-get install -y python3.12 python3.12-venv git build-essential \
  ca-certificates libgomp1 libglib2.0-0 libgl1 ffmpeg
```

两个环境分别安装 GPU SAM 和 CPU Harness：

```bash
bash tools/vast_benchmark/setup.sh sam
bash tools/cloud_cpu_tests/setup_cloud.sh
```

main 不提交权重文件，原租赁脚本通过 Hugging Face 下载。已租好的实例可以用本包的
独立入口，下载同一仓库和文件，无需再次执行会租机的原控制脚本。
先确认 Hugging Face 账号能够访问 `facebook/sam3.1`；token 通过环境变量读取：

```bash
read -s -p 'Hugging Face token: ' HF_TOKEN
export HF_TOKEN
.venv-sam31-benchmark/bin/python -m tools.cloud_cpu_tests.download_sam
```

也可以直接把已有 `sam3.1_multiplex.pt` 放到 `.models/sam3.1/`，跳过下载。
本包不含权重和密钥。然后在 GPU 终端执行：

```bash
.venv-sam31-benchmark/bin/python tools/vast_benchmark/start_sam.py \
  --checkpoint .models/sam3.1/sam3.1_multiplex.pt \
  --worker-env tools/cloud_cpu_tests/worker-env.json
```

这里的 worker 配置取自 main 租赁脚本，额外设置 `SAM31_LIVE_CAMERAS=2`，
让同一个模型提供两个独立会话槽。保持该服务运行，直到两项测试全部结束。
已有 worker 也可以复用，但 `/health/cameras` 必须就绪并提供两个槽。

另开云端终端，在包根目录设置 OpenRouter 密钥（bash 安静读入）：

```bash
read -s -p 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
.venv-harness-benchmark/bin/python -m tools.cloud_cpu_tests.server --allow-face-download
```

`--allow-face-download` 允许首次下载 CPU InsightFace 权重；已有权重可指定
`--face-model-root /path/to/insightface`。云端两个 Harness 端口是 18094、18095。
程序先确认双路 SAM worker 就绪。CPU 服务启动后保持该终端运行。
不要在两个实验之间更换显卡、权重、运行配置或 power limit。

## 2. Mac 准备

Mac 使用同一份解压包或本次修改后的源码，在根目录执行：

```bash
bash tools/cloud_cpu_tests/setup_local.sh
```

使用已安装的 Google Chrome。录屏校验需要 `ffprobe`；没有时可通过
`brew install ffmpeg` 安装。无需在 Mac 安装或运行 SAM、InsightFace。

另开终端，填入 Vast 提供的 SSH 地址与端口，并保持隧道运行：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 18094:127.0.0.1:18094 -L 18095:127.0.0.1:18095 \
  -p YOUR_SSH_PORT root@YOUR_HOST
```

可先只检查试验表，不播放视频、不调用模型、不录屏：

```bash
.venv-cloud-tests/bin/python -m tools.cloud_cpu_tests.run --test 1 --plan-only
.venv-cloud-tests/bin/python -m tools.cloud_cpu_tests.run --test 2 --plan-only
```

## 3. 测试一：不录屏 / Mac 系统录屏

使用 macOS 自带 `screencapture -v` 真实录屏，默认不录音。
给执行命令的终端授予 macOS 屏幕录制权限，指定只包含 SAM 浏览器画面的屏幕区域。
`--record-region` 格式为 `x,y,宽,高`，使用你的屏幕坐标，不能直接把占位符当成参数。
脚本在浏览器就绪后启动录屏，再开始视频播放，最后校验录屏文件含视频轨且时长足够。
测试期间让浏览器保持可见，不移动窗口，不在录制区域打开其他应用。

```bash
.venv-cloud-tests/bin/python -m tools.cloud_cpu_tests.run \
  --test 1 --recording-method macos --record-region X,Y,WIDTH,HEIGHT
```

如果只需要录制浏览器里的描边画布：

```bash
.venv-cloud-tests/bin/python -m tools.cloud_cpu_tests.run \
  --test 1 --recording-method canvas
```

canvas 使用浏览器 MediaRecorder，VP8、目标 30 FPS、5 Mbps；不录整个桌面。
它与系统录屏负载不同，报告会明确标记，不能将 canvas 结果当成系统录屏结果。
两个条件均使用可见 Chrome，自动截图关闭，减少额外截图负载。

默认“未明显退化”的判据是：录屏后描边率下降不超过 3 个百分点、云端 SAM append
P95 增长不超过 10%、呈现 FPS 不低于不录屏的 90%；每个档位两轮都满足才重复通过。
这些是可修改的工程容差，不是统计显著性证明。**是否达到 70% 单独报告**；两次都是
5% 描边率也可能“没有退化”，但显然没有达标。

## 4. 测试二：单路 / 同时双路

```bash
.venv-cloud-tests/bin/python -m tools.cloud_cpu_tests.run --test 2
```

该测试关闭录屏，按相同档位比较单路和双路。双路必须分别达到 70%，不能取均值；
每路呈现 FPS 至少保留单路的 90%。同时要求开始时间差不超过 0.25 秒、播放区间
重叠至少 90%，防止先后跑两路却标成并发。两路使用独立会话并共用同一 GPU。
测试期间保持 Mac 唤醒，关闭其他无关模型任务。

## 5. 输出与结论

每个测试默认 16 轮原片（约 56 秒/轮，加准备和排空）。正式运行会调用真实 VLM API。
每次使用新输出目录，终端最后打印 `report.html`；同目录保留：

- 冻结的测试配置、代码哈希、每轮作业、浏览器版本和原视频哈希。
- 每路逐帧计数、严格/允许近似补全的描边率、呈现 FPS、采集到呈现 P50/P95。
- 匹配会话的云端 SAM append 与“云端收到帧到结果”P50/P95，排除 EOF 排空样本。
- 云端主机/PID、CPU 数量、RAM、GPU UUID/显存/驱动/功耗限制，以及两端负载与内存。
- 录屏文件与时长校验、双路重叠比例和启动偏差、失败原因。

SAM append 含云端排队、传输及 CPU/GPU 工作，不等同于纯 GPU 内核耗时。
延迟使用各自时钟内的差值，不直接相减云端和本机的系统时间。
描边率口径沿用 main 浏览器生产计数，包含启动无描边帧和近似补全，只针对模型已知
人物；漏检可能抬高数值，仍需查看实际画面。

报告只判断“本次录屏没有明显退化”或“本次双路满足判据”。云端进程、配置和负载
记录用于佐证执行位置；不会仅凭两次速度相同宣称所有 CPU 工作迁移完美成功。
双路仍卡也不能直接得出“不是 CPU 问题”。历史 5%～6% 只作为背景，不作严格配对对照。

2026-09-26 的租赁实例实测结论见[验证记录](../../docs/validation/cloud-cpu-tests-20260926.md)；
新实例仍须重新验证云端安装、GPU 性能与系统录屏权限。
Ctrl+C 停止本机测试后，如有未排空会话，查看云端终端并重启这两个测试 Harness 进程；
不要在服务仍忙时继续运行下一组。GPU 实例的停止或销毁由你管理，脚本不会代为租赁。
