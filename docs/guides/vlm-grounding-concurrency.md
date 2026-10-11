# Grounding 十二路并发对照

目的：检查增加并发后，是否减少未发出窗口并提高完整结果数量，同时记录延迟和超时。
使用现有生产 VLM 分支执行；默认离线模拟，必须显式 `--real` 才调用模型。
不使用 SAM，不评估事件真实准确率，也不自动租用实例。

## 固定条件

- 模型：`qwen/qwen3-vl-30b-a3b-instruct`，默认经 OpenRouter。
- 十二路复制同一视频，每路 1.25 个观察窗口/秒、源时钟错峰。
- 六类统一规则，`lean-v1` 定位输出；图片长边 720、JPEG 90。
- API 超时 20 秒、输出上限 1600 tokens、名额等待上限 1 秒。
- 请求发起上限 40 次/秒；不是将请求合并成 batch。
- 每档播放 55.798021 秒，默认每档两轮，每轮 828 个正式窗口。
- 第一轮顺序 4→6→8，第二轮 8→6→4。
- 每档同时设置：每路执行线程=每路并发名额=4/6/8，总名额=48/72/96。
- 每次正式测量前单路预热 10 秒；额外一次单路预检 10 秒。
- 默认不录屏，避免录制负载影响本机输入时序。

每个阶段的 `limits` 记录实际并发，覆盖顶层默认值。
默认最多 5052 次模型调用：4968 个正式窗口 + 84 个预检/预热窗口。
未发出的窗口不会变成模型调用；连接预热不属于模型推理请求。
输入、源代码、规则和请求参数保存在 manifest。

默认自动路由。若需固定供应商，可用 `--provider-only deepinfra` 等支持的路由；
固定后禁止自动回退。自动路由和时段波动会影响结果，不能单独证明并发修改的因果效果。
保留响应中可见的供应商标签，缺失或脱敏标签不会猜测。

## 本机或已登录的服务器

在项目根目录使用已安装项目及 video 可选依赖的 Python。先只查看计划：

```bash
python tools/vlm_grounding_concurrency.py --plan-only
```

离线检查调度、报告及 4/6/8 的执行限制，不使用 API：

```bash
python tools/vlm_grounding_concurrency.py \
  --duration 8 --warmup-seconds 3 --repeats 1 \
  --simulated-grounding-latency 6.5 \
  --output outputs/grounding-concurrency-simulation
```

正式测试（会使用付费 API）：

```bash
python tools/vlm_grounding_concurrency.py \
  --real --video "/path/to/output 3.mp4" \
  --output outputs/grounding-concurrency-real
```

Key 优先读取环境中的 `OPENROUTER_API_KEY`，否则读取项目 `.env`；
可用 `--env-file` 指向单独文件。只在真正运行时读取，不写入报告或命令参数。
百炼可用 `--backend dashscope --endpoint <官方完整兼容接口>`，读取
`DASHSCOPE_API_KEY`。各档位必须使用同一渠道、模型和 endpoint。
已有输出目录不会覆盖。该入口不会自动创建云实例或建立 SSH。

## 通过现有 SSH 云端脚本

云入口支持同一对照计划、部署、断线后恢复和结果下载。先查看计划，无需 SSH：

```bash
bash tools/vlm_benchmark/run_remote.sh \
  --suite modes --modes grounding --grounding-format lean-v1 \
  --streams 12 --grounding-concurrency-sweep 4 6 8 \
  --duration 55.798021 --warmup-seconds 10 --repeats 2 \
  --http-timeout 20 --no-record --plan-only
```

实际运行时去掉 `--plan-only`，加 `--real --host <地址> --port <端口>`、
`--identity <SSH 私钥路径>` 和 `--video <本地视频路径>`。
模型仍在 OpenRouter/百炼，云实例只运行 Harness。云入口仍提供实时监看；
`--no-record` 关闭录屏。不要同时运行其他模型压测。

## 输出与判读

- `grounding-concurrency.html`：每档每轮对照及逐路、分段诊断。
- `grounding-concurrency.json` / `.csv`：可复核数据。
- `report.html` / `summary.json`：全部阶段及原始记录链接。
- `manifest.json` / `event-policy.json`：冻结配置、输入与规则。

完整率同时给出全部计划窗口、实际请求尝试两个分母。
只使用明确的 HTTP 调用标记计算发出数，不以准备请求的耗时推断。
仅在逐路提交、结束、发布与留存核对通过后，才计算未发出数量；
其中包括未形成请求记录、申请名额失败及在发送前失败的窗口。
无法确认是否进入 HTTP 调用的记录单列，统计判为无效，不擅自归为未发出。
模拟运行中的请求尝试表示模拟任务，API 时间显示不可用。

主要比较完整结果/计划窗口及每分钟完整结果，再看延迟、超时和校验失败。
每分钟完整结果按输入及排空总耗时计算，包含迟到结果，不代表实时达标率。
总延迟和 API 延迟的主表只统计完整结果；JSON 另列全部尝试的 API 分布。
请求前等待已包含名额等待，二者不能相加；未发出窗口没有 API 延迟。
API 时间包含网络及远端处理，不是纯 GPU 推理时间。
定位完整只表示结构/引用通过校验，不证明框正确。

输入时序失效、记录丢失、无法排空或残留名额会停止后续阶段并保留诊断；
HTTP 401/402/403/429 同样停止后续阶段。单个普通超时仍保留在统计分母。
停止条件在阶段结束时检查，已发出的本阶段请求不会撤销。

执行层支持显式设置到每路 8；现有生产默认值和观测频率没有改变。
