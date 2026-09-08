# VLM rich annotation 延迟复测 · 2026-09-08

本轮没有找到在当前完整标注要求下，能稳定且准确地在 **3 秒以内** 返回的合规配置。
已落实的明确改善是本机 TLS 信任库初始化：约 **0.85–0.94 秒降到 0.07–0.10 秒**。
它改善首次调用，不足以消除模型持续生成所需的数秒。

最接近目标的可保留候选是 Seed 2.0 Mini 三路并行：原图一次 **3.240 秒、4 人/8 物**；
镜像 **3.423 秒、5 人/7 物**，其中同一人重复标注。Qwen3-VL-8B 虽有 **1.78–2.13 秒**
的结果，却只返回实际 4 人中的 1 人。不能用这些结果宣称达标，也不能只看 JSON 合法率。

## 同步与约束

- 目标 worktree：`.worktrees/vlm-rich-annotation`，分支 `codex/vlm-rich-annotation`。
- 先将 main 的 `1f83f23` 合入为 `c8f5fa3`；工作期间 main 新增 `494f58b`
  （连接池和路由探测），又合入为 **`f50611c`**。最终 main 的 `494f58b` 是目标 HEAD 的祖先。
- 合并保留 rich dry-run 和 main 新增的 route-probe 功能。main 本身未被本轮修改。
- 原有未提交修改已经恢复；本轮优化也留为可审阅的未提交修改。
  两个恢复备份 stash `c8558b3`、`271b201` 均保留，未删除。
- 最终配置和可复用测速案例排除 **Anthropic、OpenAI、Google、CoreWeave、Novita**。
  显式 provider allowlist 加 `allow_fallbacks: false` 防止自动回落到其他供应商。
  用户追加限制之前的原始历史测量仍保留作记录，不构成当前选型或后续调用配置。
- 办公室图及镜像的本轮 OpenRouter 测试已获用户明确授权。密钥不写入报告或测量结果。

## 测量范围与口径

当前执行主机网络出口实测为 **TW/TPE**，不是香港。因此这里是 HK 可用模型候选的本机测速，
不是香港网络出口验收。阿里云官方区域文档列出香港模型服务；BytePlus 官方可用区域包含香港，
但有受限模型例外。供应商区域支持不等于某条 OpenRouter 后端已从香港实测通过。
最终 Qwen/Alibaba 沿用既有区域选择，Seed 保留为待香港复验的实验候选。
来源：[阿里云模型与区域](https://www.alibabacloud.com/help/en/model-studio/model-pricing)、
[Qwen Flash 区域说明](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-8-flash)、
[BytePlus 服务区域](https://docs.byteplus.com/en/docs/ModelArk/availability)。

办公室输入为已有 `tmp/server-live-left.jpg`，720×405，人工检查可见 4 人；交替使用水平镜像。
另用官方 [Ultralytics bus 样图](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg)
作前置兼容性筛选。公共图表现不能替代办公室图表现；镜像也不构成独立场景数据集。

所有记录合计 **115 个整帧尝试、193 个子请求尝试、165 个已捕获完成响应**。
包括早期筛选、用户追加排除项之前的历史尝试、失败和取消；不等同于 193 次成功计费。
响应中可见费用合计 **US$0.083784**，缺失 usage 的失败/取消请求使它只是已知下界。
汇总为 `tmp/vlm-latency-20260908/summary.json`，不以成功样本筛掉失败分母。
早期 probe 的 `valid` 只代表当时的解析成功，汇总沿用了该布尔值；不能将历史汇总中同名
`protocol_valid` 字段解释成所有记录都经过了最终新增的严格校验。

测量从 native `infer()` 开始，到整个 snapshot 返回为止；并行模式等全部三个子请求完成、
校验并合并。首次请求计入懒初始化连接池，后续复用同一客户端/事件循环。
图像文件读取和输入 JPEG 准备在计时外；服务内部解码、缩放、重编码仍在计时内。
matrix 的诊断开销计入 wall，额外离线 JSON Schema 复核在 wall 外；named 格式转换在 wall 内。

必须区分以下指标：

| 指标 | 本轮含义 |
| --- | --- |
| 首 token | 用户还不能获得完整标注，不作为 3 秒达标指标 |
| wall_seconds | 整帧调用完整结束，包含本地处理及必要校验 |
| protocol_valid | JSON、schema、字段、坐标、scope 和完成条件通过；不是视觉正确 |
| cold_client | 此案例首次客户端调用；是否重新建 TCP 要另看 trace |
| 视觉检查 | 将实际对象叠回原图/镜像，检查漏人、重复框和明显属性错误；未构造 IoU 真值集 |

没有把 `finish_reason=length`、只收到部分内容、格式修复后可解析、缓存命中或漏人结果算作
完整准确成功；流式路径继续要求正常 stop、DONE/EOF 及无流内错误。
矩阵工具明确输出 `visual_quality_evaluated: false`，因为程序没有自动评估视觉真值。
本报告的视觉判断来自人工检查生成的叠框对照图 `tmp/vlm-latency-20260908/quality-review.jpg`。

## 延迟究竟在哪里

| 环节 | 证据 | 结论与处理 |
| --- | --- | --- |
| Windows 本地 TLS context | 四组成对测量：cafile 0.938/0.879/0.922/0.847 秒；cadata 0.078/0.091/0.074/0.100 秒 | 已修复最清晰的冷启动开销 |
| 网络连接 | Seed 示例 TCP 0.034 秒、TLS 握手 0.073 秒；独立暖连接探测约 0.036–0.039 秒 | 网络不是该样本数秒延迟的主要部分，但冷连接仍会抖动 |
| 上游等待 | 最终 Qwen 首内容 2.161 秒；Seed 左人分支首内容 1.505 秒 | trace 无法进一步区分 OpenRouter、供应商排队和图像 prefill；不能全部归因于网络 |
| 内容生成 | 最终 Qwen 381 输出 token，首末内容跨度 4.819 秒 | 自回归输出是该次主要耗时，连接优化无法消除 |
| 完成尾部 | 同次 Qwen 最后内容到完成约 0.107 秒，到 EOF 再约 0.001 秒 | 提前截断流仅省约十分之一秒，且破坏完整性判定，未采用 |
| 输出协议 | 部分上游强制结构化对象仍输出缩进，曾达 861–1020 token | 请求紧凑格式不保证上游少 token；改成无 schema 又出现缺字段/围栏，未推广 |
| 三路并行 | 人物左右分支 + 全图物体，按最慢分支完成时间返回 | 缩短串行输出，但每帧 3 次请求、更多输入、受慢分支拖累，且增加重复/漏检风险 |
| 显示 | 8 秒请求 deadline，10 秒显示缓冲 | deadline 是超时界限，显示缓冲用于对齐；两者都不是推理加速，也不是 3 秒画面端到端承诺 |

TLS 优化用 Python 读取 **同一 certifi PEM** 再交给 OpenSSL 的 `cadata`。
逐一核对全部 **151 个 CA 的 DER SHA-256 集合一致**，保留 `CERT_REQUIRED`、主机名检查与
TLS 1.2 最低版本。没有关闭证书校验。显式 `SSL_CERT_FILE`/`SSL_CERT_DIR` 仍按文件/目录加载；
错误 override 不回落绕过。`trust_env` 默认保留代理/环境语义，`SSLKEYLOGFILE` 行为也保留。
本机数据在 `tls-context-audit.json`；并不承诺 Linux/Docker 获得同等幅度的收益。

## 模型、连接与调用实验

下表是办公室样本中的关键结果，秒数为完整返回耗时。每格样本数很少，不能推导 p95/SLA。

| 模型 / 路由 / 调用 | 实测秒数 | 返回质量与决定 |
| --- | --- | --- |
| Qwen3.6 Flash / Alibaba / 原单路 schema | 6.084、5.108、4.714 | 人/物为 4/0、3/5、4/0，模型本身有召回不稳定 |
| Qwen3-VL-8B / Alibaba / 单路 | 4.060、2.883 | 第二次只有 3 人，速度改善不稳定 |
| Qwen3-VL-8B / Alibaba / 裁剪并行 | 2.167、1.820 | 仅 2 人，不能采用 |
| Qwen3-VL-8B / Alibaba / 全图并行、整数示例 | 2.127、1.781 | 仅 1 人，分支产生同框后合并；明确不达质量要求 |
| Qwen3.6 Flash / Alibaba / 裁剪并行 | 4.020、3.050 | 人数 5、2，既有重复也有漏人 |
| Qwen3-VL-30B / Alibaba / 单路 | 6.400、5.030 | 都有 4 人/8 物，仍超时延目标 |
| Qwen3.5 35B / Darkbloom / 单路 | 7.300、5.240 | 后一次协议不合格；Alibaba 路由调用也出现失败 |
| Qwen3.8 Flash / Alibaba | 两次 12 秒 deadline | 目录中的新型号不等于更快，未采用 |
| Seed 2.0 Mini / Seed / named 单路 | 6.890、5.430 | 后一次值校验失败 |
| Seed 2.0 Lite / Seed / named 单路 | 9.510、7.450 | 比 Mini 更慢 |
| GLM-5.3 Flash / Modal | 一次 3.950，其余有 429；并行也 429 | 有 reasoning token 且限流，不满足稳定性；未纳入可复用 HK 案例 |
| Seed Mini / Seed FP8 / 裁剪并行 720 | 3.708、3.215 | 4/8、5/7，镜像重复人 |
| Seed Mini / Seed FP8 / 裁剪并行 480、HTTP/1.1 | **3.240、3.423** | 4/8、5/7；最接近的候选，仍不达标 |
| Seed Mini / Seed FP8 / 裁剪并行 480、HTTP/2 | 4.626、3.184 | 4/8、5/7；不能证明 H2 更快 |
| Seed Mini / Seed FP8 / 全图并行 720、HTTP/2 | 3.954、4.170 | 两次均 5 人/7 物；full_frame 未解决重复 |
| **最终保存配置：Qwen3.6 Flash / Alibaba / 单路 H1** | **7.179、5.230、5.530** | 4/5、3/4、4/0；完整协议通过，速度及视觉质量仍未验收 |

其他被实际检验但未设为默认的策略：

- **HTTP/2、连接池、环境代理**：保留长生命周期客户端和事件循环，keepalive 60 秒。
  trace 记录真实 TCP 建连数量；H2 三个请求可以共享连接，但没有一致的整帧收益。
  H2 需要安装 `httpx[http2]`；HTTPX 也没有承诺它总比 H1 快。
  来源：[HTTPX HTTP/2 文档](https://www.python-httpx.org/http2/)。
- **非流式 JSON 调用**：未显示稳定优势，生产继续流式完整消费。更快首 token 不代表更快整帧。
- **OpenRouter 路由**：核对当前 model/endpoints 元数据，试过固定后端、throughput 排序及
  `preferred_min_throughput`。throughput 偏好不能保证低首 token，也不是 3 秒硬 deadline。
  `preferred_max_latency` 同样基于历史表现的路由偏好；`:nitro` 还可能选择 priority tier，
  不应与普通 throughput 排序混为一谈。最终显式固定 Alibaba、关闭 fallback。
  来源：[Provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)、
  [Service tiers](https://openrouter.ai/docs/guides/features/service-tiers)。
- **关闭 reasoning**：支持关闭的模型显式关闭，并核对 usage；不假设所有模型都能遵守。
  来源：[Reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)。
- **缓存**：测速发送 `X-OpenRouter-Cache: false` 关闭响应缓存，并交替原图/镜像。
  这不会关闭供应商自动前缀缓存；Seed H1 两次仍报告 2104/4208 cached input tokens，
  因而不能把这些数字包装成完全无缓存结果。显式 prompt-cache 布局也影响召回，未默认开启。
  来源：[Response caching](https://openrouter.ai/docs/guides/features/response-caching)。
- **缩图、裁剪、全图分区**：480/720 对照未实现稳定正确的 3 秒。全图分区保留上下文但依赖
  模型遵守左右区域；重叠裁剪则需要可靠去重。当前不能仅靠人数看起来接近就通过验收。
- **缩短提示、去示例、schema 与输出 token**：去示例后 Qwen8B 全图并行甚至输出 0 人；
  取消 schema 有 Markdown 围栏、字段缺失、非法值。降低 max_tokens 会造成截断，不能作为完成加速。
  named 字段协议绕开部分供应商不支持 tuple schema 的问题，但仅保留实验适配器，未改变服务协议。

## 已落地的代码与最终配置

- `vlm_transport.py`：安全快速的 CA 加载、连接复用参数、可选 H2。
- `streaming_model_api.py`：首/末 token、生成、DONE、EOF、服务层级和可选 TCP/TLS trace，
  只记录安全聚合信息；保留错误、取消和完成检查。
- `vlm_validation.py`：本地核验原始 compact JSON，包括全部 13 个人物属性键、整数有效框、
  对象数量、scope、重复键、非有限数、未知键。null 表达未知，不能补字段伪装模型输出。
- `vlm_rich_labeling.py`：接入完整性校验和连接设置；增加 `full_frame` 并行实验；
  修复提示中的整数坐标被生成为浮点示例的问题。三个分支仍全部等待/校验后再合并。
- 配置与 service：新增 `vlm_require_complete`、`vlm_http2`、`vlm_trust_env`、
  `vlm_keepalive_expiry_seconds`、`vlm_parallel_strategy`；修复 customer-standard 中性配置
  导致 rich 独立 provider 被误清除的问题，保持事件模型的独立路由。
- `vlm_object_wire.py` 及 `tools/vlm_speed_matrix.py`：named 实验协议、模型/传输矩阵、
  成败完整记录、费用缺失说明和原始响应校验。named 格式不作为生产配置可选项。
- `config.vlm-rich.yaml`：继续使用 **Qwen3.6 Flash / Alibaba、single、compact、HTTP/1.1**，
  上游 schema 和本地完整性校验均启用；reasoning 关闭；fallback 关闭。
  原有 8 秒超时、10 秒显示缓冲保留并明确含义。未将快速但漏人的候选切入默认。
- `pyproject.toml`：显式 certifi 依赖，可选 `vlm` extra 提供 H2。

## 验证与复现

合入最新 main 后，全量离线回归 **1392 passed, 7 skipped**（38.15 秒），记录在
`tmp/vlm-latency-20260908/final-tests.txt`。随后增加独立 rich 路由回归并执行配置测试
**40 passed**；最后对配置、named 格式和完整性校验再做定向回归，**224 passed**。
保留的三个模型案例已离线加载验证。新模块与对应测试通过定向 Ruff，`git diff --check` 通过。
全量测试中修正了两个旧的交付 tag 断言，使其与 main 已有 Compose 标签
`2026.09.03-full-gpu.1` 一致，未改产品镜像版本。
离线测试不等于新建 Docker/GPU 环境或香港实流验收；本轮没有部署、重启或发布。

以下命令在目标 worktree 中执行。图像和 env 路径按本机授权文件填写；输出在忽略目录中。
测速会发出付费 API 请求。

```powershell
python tools/vlm_native_probe.py --config config.vlm-rich.yaml --image '../../tmp/server-live-left.jpg' --env-file '../../.env' --output tmp/native-recheck.jsonl --repeats 3
```

保留的合规矩阵案例为 `tools/vlm_speed_cases.hk.json`，只含 Qwen/Alibaba 与 Seed/Seed FP8。
Seed 案例供实验比较，并未承诺质量或香港出口实测。先安装项目及测试依赖、H2 extra；
每轮三案例共 7 个子请求，`--repeats 2` 会交替原图和镜像。

```powershell
python -m pip install -e '.[dev,vlm]'
python tools/vlm_speed_matrix.py --env-file '../../.env' --image '../../tmp/server-live-left.jpg' --cases tools/vlm_speed_cases.hk.json --output tmp/hk-recheck.jsonl --repeats 2 --timeout 8 --trace-transport
```

原始矩阵和 TLS 记录全部在 `tmp/vlm-latency-20260908/`。不要直接重跑其中限制追加前的历史
cases 文件；使用上面保留的 allowlist 案例。上一轮背景见
[2026-09-06 报告](VLM_LATENCY_REPORT_20260906.md)。

若完整标注的 3 秒是硬约束，现有证据指向需要换推理架构或获得更低延迟且召回可靠的上游：
例如本地检测/跟踪先确定框，再对人物区域异步补属性，或将合适 VLM 部署到邻近推理节点。
前者会改变“整帧完整属性同时到达”的语义，后者需要实际算力与部署测量；本轮未把它们当成
已经完成的优化。继续只调整同一公网 API 的 HTTP 参数，没有数据支持能补足当前差距。
