# VLM 三模式：事件、定位与双分支耗时对照

固定 Qwen3.5-35B＋Alibaba、十二路三模式、15 FPS 单视频无框录屏，见
[本轮专用入口](vlm-qwen35b-alibaba.md)。以下通用入口继续保留原有默认配置。

十二路 grounding 的每路并发 4/6/8 对照见
[专用脚本与统计口径](vlm-grounding-concurrency.md)。

入口：`python -m tools.vlm_benchmark.modes`，在仓库根目录运行。
它调用已有模型服务，不安装推理引擎、不下载权重、不创建付费实例。
当前测试默认后端为 **OpenRouter**，模型为 **Qwen3-VL-30B-A3B-Instruct**，
ID 为 `qwen/qwen3-vl-30b-a3b-instruct`，见 [OpenRouter 模型页](https://openrouter.ai/qwen/qwen3-vl-30b-a3b-instruct/api)。
沿用非 Thinking 测试方式；该模型的端点未声明支持 `reasoning` 参数，因此不发送
这一参数，避免 `require_parameters=true` 筛掉可用路由。
默认模拟；只有显式加 `--real` 才请求真实模型并产生 API 费用。
运行脚本的机器负责解码、调度和统计；VLM 推理由 OpenRouter 服务商执行，
不需要在租用的 4090 上下载模型、安装 vLLM 或部署四位量化权重。

## 三组含义

| 模式 | 请求内容 |
| --- | --- |
| `events` | 六类事件 boolean/null，OpenRouter 使用 JSON Schema；百炼使用 JSON Object + 本地六类校验，最多 160 output tokens |
| `compact` | parallel 中同一个 grounding adapter；全阴性允许简短回复，其余返回定位结构，最多 1600 tokens |
| `parallel` | 独立 events + compact 请求，分别发布、分别统计 |

`events+parallel` 会重复事件工作，因此不提供这个选项。以上均不依赖 SAM。
事件范围来自代码中的 `SAFETY_EVENT_LABELS`，包括 fight、smoking、fall、weapon、
ppe_violation、ground_litter。不能将完整 JSON 比例当作事件识别准确率。

三种模式共用同一份事件判断规则；新结果记录规则指纹，测试目录的
`event-policy.json` 可直接用于独立参考标注。六类定义、敏感度、不确定的处理以及
旧标注的兼容范围见 [统一事件判断与评测口径](vlm-event-policy.md)。

## 开始前

在同一台测试机上运行各轮，停止其他压测任务。
真实运行会使用本机套件锁，阻止同一仓库的重叠运行；其他主机或仓库仍需自行隔离。
可以用 Mac 或 CPU 云实例；若继续使用 4090 实例，它在本轮只充当测试机。
`nvidia-smi` 采集的也是测试机，不能解释为 OpenRouter 模型的 GPU 使用率。
所需 Python 3.12 依赖与 `tools/vlm_benchmark/requirements.txt` 一致，视频解码需 PyAV。
在隔离环境安装该文件后，以 `pip install --no-deps -e .` 安装本仓库。
API key 只在运行机器本地填写，不写进命令参数、报告或代码。

仅查看默认请求预算（不发请求，不加载视频）：

```bash
python -m tools.vlm_benchmark.modes --plan-only
```

先用模拟验证流程和报告：

```bash
python -m tools.vlm_benchmark.modes \
  --streams 1 12 --duration 3 --repeats 1 \
  --output outputs/vlm-modes/simulation
```

从本地安全设置 `OPENROUTER_API_KEY` 后，先做三种模式的单路短测；脚本会先做
真实图片请求和输出解析预检，失败就停止后续阶段。无需启动本地模型服务。

```bash
python -m tools.vlm_benchmark.modes --real --backend openrouter \
  --model qwen/qwen3-vl-30b-a3b-instruct \
  --video '/workspace/output 3.mp4' \
  --streams 1 --duration 6 --repeats 1
```

短测后再做同配置的正式阶梯对照（视频路径按测试机实际位置填写）：

```bash
python -m tools.vlm_benchmark.modes --real --backend openrouter \
  --model qwen/qwen3-vl-30b-a3b-instruct \
  --video '/workspace/output 3.mp4' \
  --streams 1 4 8 12 --duration 60 --repeats 2
```

OpenRouter 测试不提供本地 endpoint/metrics，也不声明服务商权重精度、显卡或缓存。
只记录客户端可观察的阶段，不能把 API 总等待拆成上游排队、视觉编码和 GPU 计算。
本套件的 OpenRouter 模式只读取 `OPENROUTER_API_KEY`；不会使用
`INFERENCE_API_KEY` 或 `MODEL_API_KEY` 覆盖它。百炼模式只读取 `DASHSCOPE_API_KEY`。
若此前仅配置了通用变量，需将所选渠道的密钥配置到对应专用变量。
独立的旧 capacity 入口及生产 Harness 的通用变量规则保持原样。

自部署仅作为独立的可选方案保留，需显式 `--backend local`，另见
[本地模型部署指南](vlm-local-model.md)。它的量化模型结果不能混作本轮 OpenRouter 结果。

## 可选：直接调用阿里云百炼

保留默认 OpenRouter 流程，另可选择 `--backend dashscope`。
直连模型 ID 为 `qwen3-vl-30b-a3b-instruct`；仅将现有默认模型 ID 映射为这个同模型名称，
不替换为 Plus、Thinking 或其他尺寸。events、grounding（compact）、parallel、
实时监看、录屏及耗时统计使用相同的调度和六类事件规则。

百炼只读取 `DASHSCOPE_API_KEY`，不会使用 `OPENROUTER_API_KEY`、
`INFERENCE_API_KEY` 或 `MODEL_API_KEY` 作为替代。密钥在运行环境本地设置，
不能用 OpenRouter 的密钥调用百炼。云端启动脚本也可从本地 `.env` 读取所选渠道的
专用密钥，否则在终端隐藏输入；不会把密钥写入测试计划或报告。

`--endpoint` 接收完整 Chat Completions URL（不是 SDK 的 base URL）。
省略时使用现有百炼默认地址，即北京共享域名；必须与密钥所属地域匹配。
也支持百炼工作空间专属域名，按控制台地址补上 `/chat/completions`。
例如新加坡地址为 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions`。

先生成计划，不读取密钥、不发付费请求：

```bash
python -m tools.vlm_benchmark.modes --plan-only \
  --backend dashscope \
  --endpoint https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions \
  --streams 12 --global-limit 48 --per-stream-limit 4 --request-rate 40 \
  --modes events grounding parallel --grounding-format lean-v1 \
  --video '/workspace/output 3.mp4' --record
```

正式本机运行时，将 `--plan-only` 替换为 `--real`。云实例仍只运行 Harness 和录屏浏览器，
VLM 推理由百炼执行；云端入口如下，SSH 参数填写实际实例：

```bash
bash tools/vlm_benchmark/run_remote.sh --suite modes \
  --provider dashscope \
  --endpoint https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions \
  --streams 12 --global-limit 48 --per-stream-limit 4 --request-rate 40 \
  --modes events grounding parallel --grounding-format lean-v1 \
  --video '/path/to/output 3.mp4' --real --record \
  --host "$VLM_SSH_HOST" --port "$VLM_SSH_PORT" \
  --identity ~/.ssh/id_ed25519_vast
```

直连不发送 OpenRouter 的 `provider`、`reasoning` 等路由参数，也不接受
`--provider-only`。百炼 Qwen3-VL Instruct 使用 JSON Object，继续通过原有解析器
检查六类字段、判断类型和人物引用；缺字段或无效结果仍记为失败。报告保存实际渠道、
模型、端点及结构化输出方式，不能把两种 API 的成绩混成同一条件，也不预设直连更快。
真实模型表现、账号权限和额度需另做实测验证。十二路默认单分支约 900 请求/分钟，
parallel 约 1800 请求/分钟，正式压测前应核对 RPM/TPM 配额；HTTP 429 仍停止后续阶段。

接口依据：[百炼模型说明](https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-30b-a3b-instruct)、
[地域和端点](https://help.aliyun.com/en/model-studio/base-url)、
[结构化输出](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)。

## 调度与计时口径

- 每路观察频率默认 1.25 Hz，送帧频率 2.5 Hz，保持原时间运行。
  每请求同一路 2–4 帧，图片最长边 720、JPEG 90；当前模型使用 Instruct 非 Thinking 版本。
- 每档第一轮 events → compact → parallel，第二轮逆序。
  三模式先各单路预检；正式轮次前分别进行不计入成绩的单路预热。
  正式测量默认为每档 60 秒、两次，预热与正式报告分开。
- 总并发默认 16，每视频路共享并发额度 4，本地线程上限每分支 4。
  **parallel 的两个分支共享每视频路 admission 额度**，不是每分支都获得 4 个共享名额。
  线程数量、在途请求和 GPU 连续批处理的 batch 大小是不同概念。
- 发起上限默认 40 请求/秒。十二路单分支理论 15 次/秒，双分支 30 次/秒。
  若设置低于计划负载的上限，默认拒绝；研究限流影响时显式加 `--allow-rate-limited`。
- HTTP 无自动重试；events 保持生产的最多 5 秒 HTTP 超时，compact 默认 15 秒。
  生产分支的源帧失效期限仍是 3 秒，共享名额等待上限默认 1 秒。
  这些实际限制写入 manifest；展示 1/2/3/5 秒分布不表示修改了生产超时。
- 计划窗口按每路完整源时长计算；失败、跳过、未返回保持在分母。
  ground 与 events 分别有自己的分母，不能把两条分支的成功相加当成事件完整率。
  同时存在两条完整结果的窗口另计：严格按 camera + window_id 配对。
  不用“时间戳相近”强行配对；忙时两分支可能处理不同窗口。
- admission 按“视频路 + 分支”保留一个未发送请求；新 events 只替换同路旧 events，
  不再替换 grounding。已发送的请求不替换，两条分支仍共享每路和全局并发限制。
  `paired_admission_superseded` 单列发生待发送替换的配对窗口；它也包含同分支正常的
  新旧替换，不能仅据此认定发生跨分支替换。升级时先停止旧 workers，再统一启动新版；
  持久化队列表自动补充分支字段并保留在途额度，不支持新旧 workers 混跑。
- events 和 parallel 均保留被更新结果超过的历史 events 回复，包括成功、超时、
  HTTP 错误与不完整结果。`historical_only` 标记使其进入历史查询、配对和质量统计，
  不回退实时最新状态。`late_information_reports_preserved` 记录保留数；
  不再通过丢弃旧回复隐藏 HTTP 429 或超时。
- `delivery_accounting` 区分 HTTP 已尝试、已尝试但不完整、确认未尝试 HTTP、
  HTTP 状态未知、计划中无记录、历史完整结果和非历史完整结果。
  `http_request_attempted` 只在客户端执行 send 时置真，不能用请求体大小、获准名额
  或 `inspect` 耗时代替；它也不证明服务商已经收到。模拟请求明确记为没有 HTTP。
  旧记录没有标记则保留未知，计划中无记录也不能自动算作请求失败或全部未发出。
- 每路每分支独立记录 submitted / finished / published，并与收集到的唯一请求记录
  核对；排空后不一致则保留诊断、停止后续轮次。旧版本缺少计数时核对结果为未知。
  完整返回含历史结果，非历史结果不代表浏览器实际展示次数；录屏实际帧率单独统计。
- 延迟：最新证据进入分支到发布，排除视频解码、摄像头上传和浏览器显示。
  P50/P95 对该分支全部完整结果计算；不平均各路 P95。
  3 秒只作报告参考，不等于已经批准的业务 SLA。
  新增 `source_to_completion_seconds` 与 `completion_to_publish_seconds`，
  区分本地调用处理结束前的耗时与之后的排序/IPC/发布等待，两者逐请求相加为发布延迟。
- request phases 含失败请求已观察到的阶段，各字段有自己的样本数；缺失不是零。
  source_to_request、dispatch_wait 与 admission_wait 有包含关系，不能相加。
  model_seconds 包含准备、HTTP 和解析；after_inspect_to_publish 含排序、IPC 和发布。
- 可选 vLLM 队列/prefill/decode 指标按整轮开始到结束的 counter 差值计算平均值，
  不把整个服务的指标归给某个请求或分支，不将均值当 P95。
  counter 重置、series 变化、sum/count 的标签不匹配或任一次采样缺失会标为不可用。
  未拆分视觉编码；进一步归因需推理引擎 trace/profiler。
- GPU 使用与显存保留在 telemetry.jsonl；host load average 不是 CPU 利用率。
  不采到 GPU 不代表 GPU 利用率为零。重复视频可能命中缓存，需固定并记录缓存策略。

## 输出及持续测试

### 实时监看与录屏交付

`output 3.mp4` 是测试输入。`--monitor` 在运行期间提供监看页面，显示原速视频、
实际返回的六类事件结果、对应证据帧和耗时；`--record` 同时把该页面录成 MP4。
它接收 Harness VLM 分支发布的结果，不会事后把结果提前贴到视频上。
这是本测试套件的监看页面，不是完整 Harness 产品界面或桌面录屏。
`--monitor-layout evidence`（默认）保留左右画面。事件模式不输出框；模型原始定位框
只画在右侧证据帧上。预览跟踪默认关闭，自动录屏也保持关闭，避免 grounding/parallel
额外的浏览器截帧、HTTP 和 OpenCV 工作影响三模式的容量与耗时对照。
仅作可视化检查时，可在监看 URL 后加 `?tracking=on`，让 grounding/parallel 的左侧
用原始定位作为种子，显示蓝色虚线的帧间跟踪估计；它不是新的检测或人员身份识别，
不改变六类事件判断。开启跟踪会增加客户端和测试机负载，不应用于容量或耗时对照；
进行对照测量时，应关闭所有开启跟踪的监看页面。
跟踪最多保留 8 秒历史、两个同时浏览的客户端；遮挡、交叉、跳转、过期或计算超时会隐藏
不可靠框。移除 `?tracking=on` 或改为 `?tracking=off` 可关闭跟踪。

`--monitor-layout single-video` 只显示每路一幅连续视频，隐藏右侧证据帧和全部定位/跟踪框，
仍显示各分支的六类状态、证据时间、耗时和最终统计。单路视频按剩余屏幕高度布局，
为最终统计预留空间；1、4、12 路均覆盖浏览器回归。实时视频未加入播放缓冲，
所以当前画面时间仍可能晚于结果对应的证据时间；隐藏框也不会删掉原始结果中的定位信息。
没有框不表示画面无人，完整输出也不表示已验证的识别准确率。

仅监看不需要 Playwright；录屏额外安装（以下 `python` 应指向上述测试环境）：

```bash
python -m pip install -r tools/vlm_benchmark/monitor-requirements.txt
python -m playwright install chromium
```

Linux 首次安装浏览器可用 `python -m playwright install --with-deps chromium`；
监看含中文，测试机还需中文字体，如 Ubuntu 的 `fonts-noto-cjk`。
模型预检前先检查浏览器能启动；每轮正式录制送帧前，还会检查页面和视频能加载。
失败会停止后续阶段。

先做不调用模型的短演示（画面会持续标注“模拟”）：

```bash
python -m tools.vlm_benchmark.modes --record --headed \
  --video '/Users/bihaotian/Downloads/output 3.mp4' \
  --streams 12 --duration 6 --repeats 1
```

正式测试在原命令上加 `--record`，并显式加 `--real`：

```bash
python -m tools.vlm_benchmark.modes --real --backend openrouter --record \
  --model qwen/qwen3-vl-30b-a3b-instruct \
  --video '/Users/bihaotian/Downloads/output 3.mp4' \
  --modes events compact parallel --streams 1 4 8 12 --duration 60 --repeats 2
```

按测试机实际位置修改视频路径，key 仍按上文在本机设置。
`--headed` 可在 Mac 上显示正在录制的 Chromium 窗口；省略时仍录制真实浏览器页面。
视频预览使用预先生成的 H.264、最长边 720、约 15 FPS 副本，保持原时间轴，
不会改模型使用的 JPEG 样本。预览 FPS 和模型请求频率是两件事。
逐路实际播放帧率写入 `recording.json`，也显示在报告中；它是诊断数据，不能把编码器
标称帧率当成实际播放帧率。录屏完成检查仍要求视频确实前进、所有路及最终统计在画面内、
没有页面/媒体错误。测试送帧独立于监看和日志收集，避免慢浏览器或慢日志拖延输入时钟；
迟到未送达的帧照常计入统计，不补发伪装成实时输入。
每阶段运行 `--duration` 指定的时长；短演示只播放前几秒，运行超过素材时长则循环。
单路录制 60 秒可覆盖本次约 56 秒的完整输入视频。

运行时终端显示 `Live monitor: ...`，地址也保存在 `monitor.json`；仅监听本机回环地址，
退出后关闭。在云端运行时可设置 `--monitor-port 8787`，通过 SSH 的
`-L 8787:localhost:8787` 转发，然后在本机打开终端给出的含随机路径的地址。
`run_remote.py --suite modes` 可把同一个三模式入口部署到云端；不带 `--suite modes`
时仍是原有容量压测入口，不能混用两套默认参数。
本地和云端入口都接受 `--monitor-layout single-video`；`--modes grounding` 是原有
`compact` 模式的别名，不改变模型请求或输出规则。

浏览器回归（模拟输入与结果，不调用真实模型）：

```bash
VLM_MONITOR_BROWSER_TESTS=1 python -m pytest \
  tests/test_vlm_monitor.py tests/test_vlm_preview_tracking.py -q
```

CI 的 Preview layouts and live recordings 检查会安装 Chromium 并运行这些回归，
包含三种模式的单路实际 MP4 录制及结束检查；截图和录屏作为 CI 附件保留。

### 云实例运行 Harness，OpenRouter 运行模型

本机负责打包、上传、观察进度和取回交付文件；云实例负责视频解码、Harness VLM
分支、请求调度、统计以及浏览器录屏。模型由 OpenRouter 服务商执行，云实例的
4090 不执行本次 VLM 推理。Mac 不运行负载生成器；浏览监看页也不会控制送帧时钟。
准备 Linux x86_64、Python 3.12、可用 SSH 的实例；启动器创建隔离环境并安装
测试依赖、Chromium 系统依赖和中文字体。无需模型权重、SAM、vLLM 或 Hugging Face token。

在已安装测试依赖的本机环境，先准备完整包（不连实例、不读 key、不产生 API 请求）：

```bash
python -m tools.vlm_benchmark.run_remote --suite modes --real --prepare-only \
  --video '/Users/bihaotian/Downloads/output 3.mp4'
```

拿到实例后，先做有录屏的单路短测（HOST、PORT 换成实例当前 SSH 地址）：

```bash
python -m tools.vlm_benchmark.run_remote --suite modes --real \
  --host HOST --port PORT --identity ~/.ssh/id_ed25519_vast \
  --video '/Users/bihaotian/Downloads/output 3.mp4' \
  --streams 1 --duration 6 --repeats 1
```

短测确认输入、三模式预检和录像完整后，再在同一实例执行正式测试：

```bash
python -m tools.vlm_benchmark.run_remote --suite modes --real \
  --host HOST --port PORT --identity ~/.ssh/id_ed25519_vast \
  --video '/Users/bihaotian/Downloads/output 3.mp4' \
  --streams 1 4 8 12 --duration 60 --repeats 2 --global-limit 48
```

云端三模式入口默认总并发 48、每路共享并发 4、请求上限 40/s、错峰输入，
默认录屏并监听回环端口 8787。与 `modes` 的默认总并发 16 不同，因此要对比旧基线
时显式加 `--global-limit 16`；其余模型、时间戳和统计口径仍调用同一实现。
默认不追加长时间测试；省略 `--real` 则整套为模拟运行。
key 仅从本机 `OPENROUTER_API_KEY` 或忽略的 `.env` 读取，缺失时隐藏输入。
经 SSH 标准输入传输，实例上的临时凭据文件权限为 0600，worker 读取后立即删除。
实例继承的通用 key 不覆盖本次 OpenRouter key。实例指引在部署前输出，操作前应阅读。

监看可用 `ssh -N -L 8787:127.0.0.1:8787 ...` 建立转发，再打开进度中显示的
含随机路径地址；浏览器录屏本身在云端完成，无需 Mac 一直打开页面。
SSH 断开后云任务继续；使用相同 SSH 参数加 `--suite modes --resume RUN_ID`
重新等待并取回结果，**不会重新提交付费测试**。`--stop RUN_ID` 停止对应任务并
取回已有结果，不会停止其他进程。Ctrl+C 也会请求停止；普通网络断连不自动停止。
报告、逐请求记录、MP4、原始 WebM 和最终画面一起下载到
`outputs/vlm-cloud/<RUN_ID>/results/`。中断或安装失败也保留状态和日志。

启动前会保存任务控制信息。服务注册、启动或读取进程编号失败时，删除临时 key，
尝试停止本次任务，并保留失败报告与 `launch-failure.json`。若显示停止尚未确认，
使用相同连接参数和 `--stop RUN_ID` 重试；此时不能认为云端任务已经停止。
已清理的失败任务可用 `--resume RUN_ID` 取回诊断，不会重新启动测试。

正式测量和持续测试各自生成一段录屏；预检和预热仅实时展示，单独保存诊断记录。
录制在送入首帧前开始，覆盖播放、等待剩余请求、最终统计，再保留两秒尾画面。
页面显示输入时间与结果更新时间，两分支分别更新；旧结果不能覆盖较新的结果。
录屏增加浏览器与编码负载，输入送帧时序依然按原规则校验。界面轮询有额外显示耗时，
不计入报告的“证据进入 VLM 分支到结果发布”延迟，不能把该数字当作屏幕端到端延迟。

每轮文件位于 `<阶段目录>/recording/`，总报告顶部提供链接：

| 文件 | 内容 |
| --- | --- |
| `viewport.mp4` | 可交付的浏览器页面录屏，仅画面、不含音轨 |
| `viewport.webm` | 浏览器原始录制文件 |
| `final.png` | 最终统计画面 |
| `recording.json` | 时长、分辨率、视频停滞/页面错误和完整性检查 |
| `../monitor.jsonl` | 送帧、真实结果到达、排空和结束的监看记录 |

录屏必须完成封装、全片解码检查，且测试已运行结束、包含播放与最终统计、最后的
证据图已加载，才能标记 `complete`。停滞检查也覆盖视频缓冲不足、暂时无法解码的情况。
文件过短、页面报错、预览停滞等会停止后续轮次并标记录制失败；已有数据和部分录像保留。
中断轮次写入 `summary.json` 的 `interrupted_jobs`，报告只提供已生成文件的链接，
不为缺失的统计补零，也不把中断录屏标记完整。录屏完整性与模型结果完整率分别记录。

播放结束后的排空等待单独计算：至少 15 秒，或该模式的请求超时加 3 秒源帧有效期、
再加 2 秒结果发布余量，取较大值。默认定位/双分支因此最多等待 20 秒；这不延长
任何模型请求的超时，不改变延迟起点或计划分母。每轮记录实际排空耗时和等待上限；
仍未排空的轮次继续标记无效，不能用于性能结论。

### 报告和持续测试

每次写入独立目录：`report.html`、`summary.json`、`manifest.json`，以及每轮的
`reports.jsonl`、`status.jsonl`、`telemetry.jsonl`。不会覆盖历史实测结果。
预检失败、送帧无效、未排空或记录丢失会停止后续阶段并保留已有报告。
HTTP 429 在当前有界轮次结束后停止继续升档；不会自动重试。
请求预算按每轮、每路的整数计划窗口累计，包含预检、预热和双分支请求。
准备阶段失败也会保留失败状态，执行中的阶段会写入报告，便于定位中断位置。

持续测试默认关闭。显式 `--soak-seconds 900` 时，各模式十二路短测的每一路、
每条分支完整率均达到 **99%** 才继续其 15 分钟测试；这是节约测试资源的筛选条件，
不是业务准确率或延迟验收标准。未满足的模式写入 `skipped_soaks`。
调整并发时重复同一视频/精度/频率/输出设置，仅变更待研究参数，避免同时改变多个变量。

### 帧间人框跟踪（监看与录屏）

在 `evidence` 布局下，有原始视频的 `compact` / `parallel` 监看默认启用轻量 CPU 跟踪。
左侧连续播放预览视频，蓝色虚线框表示跟踪估计；右侧显示原始定位证据帧。
`events` 没有人框，仍显示原始事件证据帧。请求失败改用蓝灰色，检测为阳性仍为红色。
`single-video` 布局仅显示连续视频，关闭跟踪与右侧证据帧，三个模式均不画框。
跟踪只影响监看与录屏，不改动事件结果、人物关联、VLM 请求频率、延迟统计或 API 用量；
也不会把跟踪框作为新的模型观测。它不使用 SAM，不具备跨定位结果的人员重识别能力。

实现使用 OpenCV 稀疏光流，逐帧检查前后向一致性、运动一致性及图像外观。
有效定位先在原始证据图像上初始化，再追过已经看到的历史帧；不会把旧框直接画到新图像。
每路每个查看窗口缓存最多 8 秒 / 128 张灰度预览帧，宽 320 像素；
每次最多追踪 12 个目标，每路仅一个在途预览请求，历史追赶预算为 120 ms。
遮挡或交叉匹配模糊、侧边出画、帧间断档超过 300 ms、源时钟跳转、定位过期时隐藏框，
等待下一次有效定位。超过 250 ms 才返回的预览结果不画回屏幕；视频继续刷新。
这是避免错框的保守策略，会减少有框时间，尤其是定位响应迟到数秒的片段。
侧边被截断的人框暂不跟踪；俯仰、尺度和姿态变化较大也可能提前丢失。

页面加 `?tracking=off` 可关闭跟踪作同源播放对照。一个阶段最多允许两个同时跟踪的查看窗口，
30 秒无请求的窗口会释放；正式录屏自身占一个。窗口打开后才开始积累历史帧。
`recording/recording.json` 的 `browser.tracking` 记录每路计算耗时、预览请求往返耗时、
显示间隔、跟踪丢失和错误；诊断样本有上限。这些是预览开销，不能当作 VLM 模型延迟。
12 路完整录制仍需使用实际云实例测量，不能从本机短测推断长期容量。

验证方式：

- `tests/test_vlm_preview_tracking.py` 使用有已知身份和逐帧位置的合成目标，检查迟到定位、
  交叉、遮挡、侧边出画、旧结果、视频循环、缓存上限及监看端点；隐藏框不能代替覆盖率检查。
- 设置 `VLM_MONITOR_BROWSER_TESTS=1` 后增加 1 / 12 路真实浏览器开启/关闭跟踪对照，
  核对已显示的框与已知位置的 IoU，并检查最终录屏统计是否能完整显示。
- `python -m tools.vlm_preview_tracking_check --video <input.mp4> --reports <reports.jsonl>
  --output <new-folder> [--camera 1] [--sample-fps 2.5]` 用历史响应离线重放。
  输出明确标记 `OFFLINE replay` 的视频、逐帧估计、接触表和耗时，不调用模型。
  该结果不是新一轮实时模型测试；有框比例也不是身份正确率。
- 正式验收仍需独立的逐人轨迹标注，报告可见目标覆盖、框偏移、身份切换和遮挡后恢复。
  原 VLM 框不能同时充当初始化输入和“跟对人”的真值。
# 十二路实际画面录制与精简 grounding

`--modes events grounding parallel` 分别运行事件、单独定位、事件与定位同时运行；
`grounding` 是内部 `compact` 模式的易读别名，`parallel` 已包括两个分支。
`--grounding-format lean-v1` 将精简协议同时用于独立与并行模式中的定位分支，
六类事件规则不变。默认仍为 `legacy`。

云端 `--suite modes` 默认录制正式测试的浏览器视口，先执行同路数、同模式的
无模型调用录制检查。检查失败即停止，不启动收费推理。正式录像覆盖播放前的准备、
原速视频播放、实际返回的检测结果、排空以及最终统计；保存 MP4、原始 WebM、
最终截图和 `recording.json`。多路窗口全部可见、视频未冻结、末帧证据已绘制、
录像时长完整才标记录制成功。

左侧显示实时视频与可用的帧间跟踪估计，右侧始终显示模型结果对应的原始证据帧及框。
迟到框不会被伪装成当前时刻的定位。录制的是实时测试监看页，不是完整产品界面，
也不是根据日志事后合成的回放。预览与录制的资源开销包含在本轮机器负载中。

示例配置：`--suite modes --real --streams 12 --repeats 1 --modes events grounding parallel
--grounding-format lean-v1 --provider-only deepinfra --http-timeout 20 --record`。
单路并发额度由两个分支共享，联合模式总请求率上限应覆盖两分支负载；
报告保留失败、跳过与未返回的分母。`--provider-only` 限定服务商并禁止自动回退。


## Unified research branch baseline

The default policy remains `main-baseline`. Use `--event-policy research-v1` to
select the common event criteria described in [the policy guide](vlm-event-policy.md).
Paired scheduling and lean output remain explicit options. Each manifest records
policy, wire format, scheduling settings, recursive runtime source hashes, and
`retain_historical_results=true`. This retention setting preserves complete
benchmark accounting without changing service defaults.
