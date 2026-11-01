# YOLO26-S 人物 ID + OpenAI Decisions

> 2026-10-20：本文实验工具使用独立的 `tools/benchmark_runtime/`；产品 grounding 已恢复 main。历史结果不等于恢复后的产品性能，见[代码与测试边界](pr48-49-person-grounding-integration.md)。

状态：已接入 2026-10-18 的 main（`fe0e912`，包含 #48），工作分支 `codex/yolo-decisions-main`。本次只做离线/模拟回归，没有重新调用模型、租机器或执行正式测试。此前 10 月 18 日的正式结果仍属于旧测试版本，不能当作本次 main 集成后的性能验收。

## 固定口径

- OpenAI 原生 `POST https://api.openai.com/v1/decisions`，`gpt-6-luna`，仅 `OPENAI_API_KEY`；不走 OpenRouter，无自动回退或重试。
- 十二路，events / grounding（内部 compact）/ parallel 各两轮。原 output.mp4，55.798021 秒，每路 1.25 窗口/秒，2.5 FPS 取证，原始全景 2–4 帧，不拼图、不裁掉背景。
- 每分支每轮 828 窗口；正式 6,624 次分支调用，十二路暖机 1,152 次，单路预检最多 4 次，总上限 7,780 次。实际费用需根据真实 input_tokens 与账户账单计算。
- 48 个全局 admission 名额、每路 4 个、配对调度、错峰、内存 admission、60 秒轮间冷却。HTTP 上限 20 秒；3 秒是完整率目标，所有未发出/失败/晚返回仍在分母。
- 沿用 YOLO26-S、640 输入、0.35 检测阈值、源顺序 IoU/motion 跟踪。人物 ID 是每路会话内的标签；内部仍有框，录屏只显示 ID、不画框，也没有右侧证据帧。
- 六类规则与原 policy SHA256 `d9c487c37404df2f6d1f4d1f3da39559abffe28f59c76e7fef85f4a9f9e48a7b` 完全一致。采用 positive / negative / uncertain 三选一，不额外调整概率门槛。

## 模式

- events：一次请求回答整个场景的六类判断，YOLO 仅供 ID 展示。
- grounding：一次请求回答每个最后一帧人物的五类人物事件，另有五个“未能归属现有 ID 的事件”问题和一个场景地面垃圾问题。没有检测到人也会发送六个场景问题，不能把“无框”当成阴性。
- parallel：同一窗口并行发 events 与人物 grounding，两个分支分别统计，也统计两者都完整/三秒内完整的覆盖率。

每窗口不会按人数拆成多次 API 请求。官方上限为每次 200 个问题；本协议 `5×人数+6`，最多支持 38 个最后一帧人物。超过时明确记录本地失败，不漏掉人物、不偷偷增加请求次数。缺项、重复项、拒答、错误模型、非法概率等都不算完整；uncertain 保留为不确定，不当作 negative。

## 使用（本次不执行带 --execute 的命令）

```sh
# 离线计划，不读取密钥
PYTHONPATH=src python -m tools.tracked_decisions plan
# 源码打包：不含 API Key、视频、模型权重
PYTHONPATH=src python -m tools.tracked_decisions bundle --output /tmp/yolo-decisions-source.tar.gz --execute
# 复用已经启动且 GPU-ready 的 YOLO26-S /health/tracking 与 /v1/tracks 服务
# 少量单路预检，不进入六轮正式测试
PYTHONPATH=src python -m tools.tracked_decisions preflight --video /data/output.mp4 --packet /data/astra-packet --output /data/decisions-preflight-new --detector-url http://127.0.0.1:8092 --execute
# 后续另行授权正式测试时
PYTHONPATH=src python -m tools.tracked_decisions run --video /data/output.mp4 --packet /data/astra-packet --output /data/decisions-six-rounds-new --detector-url http://127.0.0.1:8092 --execute
# 两套已有结果离线对比，不调用 Astra 或 Decisions
PYTHONPATH=src python -m tools.tracked_decisions compare --results /data/decisions-six-rounds-new --baseline /data/yolo-qwen35-results --video /data/output.mp4 --packet /data/astra-packet --output /data/decisions-vs-qwen35-new --execute
```

运行环境沿用既有 harness benchmark + Chromium 和独立 YOLO worker；普通 GPU 只负责 YOLO，Decisions 在 OpenAI 服务端执行，不使用 48GB FP8 的 GPU 型号限制。环境变量配置 OPENAI_API_KEY，或明确用 `--env-file` 指定已有密钥文件。计划与离线回归都不需要密钥。

保存每轮原始结果、检测日志、录屏、请求 ID、token 用量、限流响应头与安全错误码；不保存密钥或带图请求体。发现 401/402/403、insufficient_quota、model_not_found 后停止后续轮次；429 原样统计，不重试、不降低负载。

## 与 YOLO + Qwen35B 对比

共同报告发出率、整体完整率、每路三秒内完整率、P50/P95、parallel 双分支共同完成率、Astra 场景一致率。相同视频/六类 policy/负载是硬条件，不能把不同设置混为同一对照。

历史 YOLO + Qwen35B 的 events HTTP 上限是 5 秒，本版 Decisions 为 20 秒；三秒内覆盖率使用相同分母，但更长的在途请求也可能影响排队，最终完整率也受等待上限影响，报告必须注明。模型回答协议也由自由 JSON 改为固定问题。Astra 一致率只比较完整、对齐、双方明确的六类场景判断，不评价人物关联是否正确。Decisions 的结构化答案不能用来假装计算与 Qwen 相同的生成 TPS；保存 input token 用量，完整窗口/秒可由完成数和轮次时长计算。

## 离线回归与复查

```sh
PYTHONPATH=src python -m pytest -o addopts='' -q tests/test_tracked_decisions.py tests/test_tracked_decisions_flow.py tests/test_tracked_decisions_comparison.py tests/test_decisions_benchmark.py tests/test_tracked_grounding.py tests/test_tracked_main_integration.py
PYTHONPATH=src VLM_MONITOR_BROWSER_TESTS=1 python -m pytest -o addopts='' -q tests/test_tracked_grounding_browser.py
```

接口回归使用模拟 HTTP；十二路多进程测试和浏览器检查也使用模拟响应，不消耗模型额度。覆盖原图与 policy 保留、人物对应、无人物、人数超限、缺项/拒答、权限/额度失败、预检边界、请求审计和 Astra 分母。

这次 main 接入重点：

- 保留 main 的 runtime 模块拆分、#47 的单路录屏布局，以及 #48 的普通 YOLO preview 路径；人物 ID 测试通过独立入口开启。
- 人物 ID 测试禁用继承的 `PREVIEW_DETECTOR_URL` / `PREVIEW_SELECT_EVENTS` 预览调用，避免重复检测、额外 VLM 调用或偷偷恢复框；正常 preview 不受影响。
- events 与 grounding 完成即发布，迟到结果仍保留统计，但不回填当前特征、人物状态或报警；parallel 保持配对调度与相同的并发配置。
- 原生 Decisions 的路由、20 秒 HTTP 期限、证据帧指纹和审计信息按真实协议记录，不能误标为 OpenRouter。
- 浏览器验证 events、grounding、parallel 的十二路单画面、只画人物 ID、横竖视频坐标对齐、过期 ID 隐藏和最终统计可见。模拟 MP4 只证明录屏与展示流程正常，不证明识别准确率或正式 15 FPS 容量。

已离线读取 2026-10-17 YOLO + Qwen35B 与 2026-10-18 YOLO + Decisions 的各六轮结果；新版读取器均通过输入、录屏、留存与适用的请求审计检查。Astra 对比不重新请求 Astra。本次没有补跑正式十二路测试，也没有由回归结果推断性能提高。

2026-10-18 本次验证记录（本机 Python 3.12）：

- 全仓离线初检：5,821 通过、70 跳过；8 处失败为新增诊断字段/送帧回调的测试适配，以及旧版丢弃迟到回复的断言。已修正，并在后续相关复测中全部通过；没有把跳过项计为通过。
- 最终主干兼容、检测器、十二路原生流程、六类 policy 与送帧回归：121 项通过；parallel 整个测试文件：42 项通过。Decisions 接口、审计、离线对比和源码包独立导入也已在相关回归中通过。
- Chromium：6 项通过，覆盖 Qwen/Decisions × 三模式；Decisions 三模式保存模拟 MP4。此项没有加载真实 YOLO 权重或调用真实 API。
- 修复检测线程对 Python 3.11 `asyncio.Runner` 的依赖；通过删除该接口的模拟兼容回归，但未声称已在 Python 3.10 解释器或 Windows/Linux GPU 上实跑。
- 新增/修改行的 Ruff 检查、JavaScript 语法检查、仓库卫生检查和差异空白检查通过。main 原有三处格式提示未改动六类规则来规避。

本机验证日志位于 `outputs/yolo-decisions-main-review/validation/`。没有推送或触发 GitHub CI；新分支内容尚未合入 main。

官方接口依据（2026-10-18 核对）：
https://developers.openai.com/api/reference/resources/decisions/methods/create
