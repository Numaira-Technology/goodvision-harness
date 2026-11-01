# OpenAI Decisions 六类事件测试

> 2026-10-20：本文实验工具使用独立的 `tools/benchmark_runtime/`；产品 grounding 已恢复 main。历史结果不等于恢复后的产品性能，见[代码与测试边界](pr48-49-person-grounding-integration.md)。

状态：测试脚本和离线回归；未运行本版正式模型测试。

使用 OpenAI 官方 `POST https://api.openai.com/v1/decisions`，模型固定
`gpt-6-luna`。只读取 `OPENAI_API_KEY`，不经过 OpenRouter，不回退供应商，
不自动重试。默认命令仅打印计划，不读取密钥或发请求。

## 比较范围

这版测试 **events 场景级六类判断**。每个窗口一次请求，六个 choice 问题，
positive / negative / uncertain 对应 true / false / null。保留原六类政策
`independent-events-v1`，吸烟 HIGH、其余 STANDARD；政策 SHA256 必须相同。
置信度不额外设置判断门槛。不确定是有效判断；拒答、漏项、重复项、
错误模型、无效概率或非法 JSON 不是完整结果，不能补成 negative。

Decisions 只选择已定义的答案，不在本测试中生成框、人物身份或人物 label。
本独立入口只测 events；需要人物级 grounding / parallel 时，使用
[YOLO26-S + Decisions](yolo-decisions-benchmark.md) 的外部检测/跟踪流程。
该流程每窗口一次人物问题请求，不是把相同的 events 请求复制两次。

## 固定负载和统计

- 原 `output.mp4`，55.798021 秒，校验视频和原 Astra JPEG/时间戳。
- 原始独立帧，不拼图；2.5 FPS 取帧，720 最大边、JPEG 90。
  沿用滚动上下文，最多四帧，启动首窗口可能不足四帧。
- 十二路，每路 1.25 窗口/秒，每轮 69×12=828，events 两轮。
- 总并发 48，每路/分支并发 4，错峰，tmpfs admission，等待上限 1 秒。
- HTTP 总期限 20 秒；3 秒是评分目标，不是截断 HTTP 的超时。
- 每轮前单路预热 10 秒；轮间冷却至少 60 秒，不改变轮内负载。
- 正式 1656 次调用，含内置预检和预热最多 1692 次；独立 preflight 另最多 3 次。
- 两个 15 FPS 单画面浏览器录屏，无框、无右侧证据画面。
- 每路三秒内完整率至少 66/69 才达到 95%，不能用十二路均值代替。
- 完整历史回复计入统计，但不覆盖新状态。未发、失败、迟到留在计划分母。

## 命令

在 VLM 工作区，使用包含 Harness 依赖的 Python 环境：

```bash
PYTHONPATH=src python -m tools.decisions_benchmark plan
PYTHONPATH=src python -m tools.decisions_benchmark bundle --execute --output /tmp/decisions-source.tar.gz
```

源码包不带密钥、视频和 Astra 标注。将视频/参考包独立上传到测试机。
云端安装仓库的 `tools/vlm_benchmark/requirements.txt`
和 `tools/vlm_benchmark/monitor-requirements.txt`，以及项目本体。
需 FFmpeg、Chromium 与中文字体；无需安装 vLLM / CUDA 的 model-requirements。
正式测试要求 Linux `/dev/shm` 可写。无需 GPU，自身不租机或 SSH。

显式指定已有密钥文件，或者在环境配置 `OPENAI_API_KEY`；不要将密钥放命令行。
仅在决定运行付费测试后执行：

```bash
PYTHONPATH=src python -m tools.decisions_benchmark preflight --execute \
  --video /data/output.mp4 --env-file /secure/openai.env --output outputs/decisions-probe-01

PYTHONPATH=src python -m tools.decisions_benchmark run --execute \
  --video /data/output.mp4 --packet /data/astra-packet \
  --env-file /secure/openai.env --output outputs/decisions-events-01

PYTHONPATH=src python -m tools.decisions_benchmark analyze --execute \
  --video /data/output.mp4 --packet /data/astra-packet --results outputs/decisions-events-01
```

run 有内置预检，独立 preflight 可选。已有输出目录不能覆盖；鉴权、
预检失败时停止，限流也如实记录，不通过降负载或剔除失败粉饰成绩。
已收到的 HTTP 错误状态不会被错误正文读取失败或超时覆盖；例如收到
401/403 后仍按鉴权失败停止。独立预检初始化或清理失败也会释放运行锁，
并保存未通过的状态，避免阻塞下一次测试。

## 结果与费用

- `report.html`：原实时测试报告、分轮结果和录屏。
- `decisions-comparison.html/json`：发出率、整体完整率、已发完整率、
  逐路三秒内完整率、完整结果延迟 P50/P95、HTTP 延迟、admission 等待、429、
  Astra 一致率、误报/漏报、各类可比样本数。
- `*/api-audit/*.jsonl`：逐请求保留 OpenAI request ID、有限数值限额/重置头、
  HTTP 状态、输入用量与耗时。原始错误消息、图片、密钥不写入这些文件。
  request ID 仅供操作者排查，不在实时监看页面展示。
- 逐条核对结果和请求日志的流 ID、请求 ID、HTTP 状态与完整性。
  缺失、重复、不匹配或发出状态未知都会使该轮验收无效；中断的整批测试
  也不能标记为达到三秒目标。历史完整回复仍参与统计。
- 若 OpenAI 提供 `openai-processing-ms`，原样按服务端计时统计，
  不能将它解释为纯 GPU inference 时间；缺失就是未知。

Decisions 不输出普通生成文本，因此**输出 TPS 不适用**，不能把返回 JSON
字符数当成 token。记录输入 tokens/s、完整窗口/s 和每个请求的输入用量。
截至 2026-10-17 官方基价为每百万输入 tokens $0.10；费用只估算报告了
用量的请求，区域/长上下文调整另计，不声称它就是账单金额。失败请求缺少
用量或请求日志不齐时，整批费用保留未知，另列已知用量小计，不能把空日志
或部分小计当作零费用/全部费用。已有小预检约 3225 输入 tokens/四帧请求，仅作估算参考：
1692 次约 $0.55 基价，真实用量和配额应通过新 preflight 确认。

每秒约 15 次请求、900 RPM；按 3225 输入 tokens 粗算约 290 万 TPM。
历史小预检出现过 200 万 TPM 响应头，不能假定现在仍相同，也不能据此
保证十二路不受限流。独立预检保存当前额度头，正式跑前先检查。

Astra 只比较完整、原始证据准确对齐且双方明确的判断；不确定与未对齐
单列，不能算错也不能算对。它不是人工真值，也不评价定位框或人物 label。
35B 历史配置的 provider、超时等不同，只能作历史参考。

官方资料（核对于 2026-10-17）：
- https://developers.openai.com/api/docs/guides/decisions
- https://developers.openai.com/api/reference/typescript/resources/decisions/methods/create
