# VLM 完整标注：历史延迟与质量评估

本页合并 2026-09-06、09-08、09-10 三轮测量，保留各轮配置、数据与失败结论。
它不是当前源码或供应商的重新验收，也不是现行模型推荐。三轮都没有证明完整人物/物品标注
能稳定、准确地在目标时限内返回；JSON 合法或首 token 快不能代替完整视觉质量。

## 共同测量口径与复现

- 办公室输入为 `server-live-left.jpg`，720×405，人工可见 4 人；交替原图和水平镜像。
  镜像属于同一场景受控重复，不能代表多场景准确率、真实视频连续追踪或 P95/SLA。
- native `infer()` 计时到整个 snapshot 返回，包含内部解码、缩放、重编码、解析；
  外层文件读取/JPEG 准备不计时。并行模式等待所有分支校验、合并；首调用包含客户端初始化。
  09-08/10 的 matrix 诊断和 named 转换计入 wall，额外离线 Schema 复核在 wall 外。
- `wall_seconds` 是整帧完成时间；`cold_client` 仅表示客户端首次调用，实际 TCP 建连另看 trace。
  `protocol_valid` 不等于视觉正确；未建立逐框 IoU 或属性真值集，视觉判断来自人工叠框检查。
  矩阵输出 `visual_quality_evaluated: false`。最终严格校验不接受截断、部分内容或格式修补后伪装成功，
  流式路径要求正常 stop、DONE/EOF 且无流内错误；09-08 早期历史记录的例外见该节。
- 测试经授权使用既有图片；密钥不进入报告。`X-OpenRouter-Cache: false` 关闭响应缓存，
  不保证关闭供应商前缀缓存。各轮分别记录出口，不能混为香港验收。
- 旧 native/latency/packed 探针及固定模型案例已删除；历史复现需取清理前主线版本 `a7e5ec8`，
  自备授权图片、凭据与本地输出目录。`tools/vlm_speed_matrix.py` 仍接受显式本地案例开展新测量；
  实际复跑会产生付费请求。下文 `tmp/` 是当时忽略目录的证据位置，不是随仓库交付的文件。

<a id="2026-09-06"></a>
## 2026-09-06：缓存与连接复用，未达到 2.5 秒

来源分支 `codex/vlm-rich-annotation`。OpenRouter trace 为 **HK/HKG**：46 次尝试，43 份完整响应，
另有超时/429；完整响应包含合同不合格结果。早期过期密钥 8 次 401、隔离环境 8 次连接失败
不计入推理测速；阿里云直连 12 份响应单列，后续直连审批未获准，未继续。

当轮修复了单路每帧新建请求、rich 独立路由被 event profile 覆盖及缓存断点问题。
单路与并行共用异步 HTTP 客户端和事件循环，整帧时限覆盖准备/请求/解析，无自动重试；
首帧本地准备约 0.9–1.2 秒，后续请求前后开销约 3–4 毫秒，不能视为旧网络延迟的严格 A/B。
原紧凑提示曾以 1.87/2.22 秒返回 4 人，却漏掉物品和属性；改为严格 schema 要求全部 13 项人物字段，
不可见值填 null、公共属性字典仍按合同省略 null，不再人为追求 300–400 tokens。

### 缓存实测

Qwen 3.6 Flash / Alibaba 固定上游，禁用 fallback 和 OpenRouter 响应缓存：

| 请求结构 | 服务端报告 | 解释 |
| --- | --- | --- |
| text+image 同一 user 消息，text 加 cache_control | 换图 0 hit；再次原图命中 2,509 tokens | 断点包含图片，重复同图会乐观 |
| 固定 text 放 system 消息 | 0 cache read/write | 未报告有效显式缓存 |
| 独立 user 指令 + 固定 assistant `Ready.` + 新 user 图片 | 镜像命中 2,261 tokens | 诊断重复前缀越过门槛，证明变化帧可复用文本前缀 |
| 最终客户端，独立消息、正常长度指令+schema | 不同帧命中 1,320 / 1,630 tokens（约 81%） | 无填充，思考 tokens 为 0 |

最后一组仍需 **6.86 / 4.88 / 5.73 秒**，三次均仅 3 人。消息布局同时影响召回和输出长度，
不能把速度差全归因于缓存；显式缓存保留为实验开关，默认关闭。
缓存节省重复输入计算，不能消除逐 token 输出。[阿里云缓存说明](https://www.alibabacloud.com/help/en/model-studio/context-cache)、
[OpenRouter 缓存说明](https://openrouter.ai/docs/guides/best-practices/prompt-caching)。

### 最终客户端结果

每组原图、镜像、原图，复用客户端，首帧含初始化：

| 模型 / 配置 | 完整调用秒数 | 人 / 物品数 | 质量 |
| --- | --- | --- | --- |
| Qwen 3.6 Flash，缓存关闭 | 7.75 / 4.90 / 5.68 | 4/6、3/4、4/6 | 镜像漏边缘人物；方向和衣着错误 |
| Qwen 3.6 Flash，缓存开启 | 6.86 / 4.88 / 5.73 | 3/5、3/5、3/5 | 约 81% 命中，仍漏人 |
| Qwen3 VL 8B Instruct，缓存关闭 | 5.52 / 3.16 / 4.45 | 4/5、3/9、4/8 | 镜像漏人且超过 8 物品上限；属性错误更明显 |

8B 最后一次首字约 **0.35 秒**，请求仍约 **4.44 秒**，输出 448 tokens；生成/流式返回约 4.09 秒，
隐藏思考 tokens 均为 0。10 字符属性压缩曾达 **2.08–2.48 秒**，但 8B 漏物品/变行结构，
Flash 属性码错位，6 次全不合格，未接入产品。35B / DeepInfra 两次为 25 秒超时与 429，
不能用过去速度作精确的同日提速基线。

当时保存的 `config.vlm-rich.yaml` 为 Qwen 3.6 Flash / Alibaba、single、compact、严格 schema、
1,536 输出 token、720 px、8 秒截止，缓存关闭。显示缓冲由 25 秒调到 10 秒，仅改变显示延迟；
event 路由与公共 RichLabelSnapshot/overlay 合同不变，模型不生成本地 track ID 或帧时间。

原始目录 `tmp/vlm-latency-20260906/`：结果表来自 `native-no-cache.jsonl`、`native-verified.jsonl`、
`native-8b.jsonl`；缓存对照为 `cache-short.jsonl`、`cache-long.jsonl`、`cache-system.jsonl`、
`cache-message.jsonl`。`native-final.jsonl` 有 3 次模型成功但脚本误用 `to_dict`，改为 `as_dict`
后重测，未纳入最终表。`native-no-cache-overlay.jpg`、`native-8b-overlay.jpg` 经人工检查仍有框/属性误差。
早期全量回归曾有 11 个 GPU smoke 失败，归因于 pytest 参数污染，清空参数对照全部通过；
不能把那次全量运行描述为全绿。

<a id="2026-09-08"></a>
## 2026-09-08：TLS 冷启动与模型矩阵，未达到 3 秒

来源分支同上，合入 main `1f83f23` 和 `494f58b` 后基线为 **`f50611c`**。
出口 **TW/TPE**，属于 HK 可用候选的本机测速，Seed 仍待香港复验。
区域来源：[阿里云](https://www.alibabacloud.com/help/en/model-studio/model-pricing)、
[Qwen Flash](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-8-flash)、
[BytePlus](https://docs.byteplus.com/en/docs/ModelArk/availability)；供应商支持区域不等于该后端已在香港实测。
前置兼容性筛选另用 [Ultralytics bus 样图](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg)，
不能替代办公室质量检查。

最终案例排除 Anthropic、OpenAI、Google、CoreWeave、Novita，显式 allowlist 且禁止 fallback；
追加限制前的历史测量保留作记录，不构成后续调用配置。合计 **115 个整帧尝试、193 个子请求尝试、
165 个捕获的完成响应**，包括失败/取消，不能视为 193 次成功计费。
可见费用 **US$0.083784** 为下界，部分失败/取消缺 usage。
早期 probe 的 `valid` 仅指解析成功，汇总沿用该值；历史 `protocol_valid` 不能全部解释为最终严格校验。

### 延迟分解

| 环节 | 测量证据 | 结论 |
| --- | --- | --- |
| Windows TLS context | cafile 0.938/0.879/0.922/0.847 秒；cadata 0.078/0.091/0.074/0.100 秒 | 明确的冷启动改善 |
| 网络连接 | Seed TCP 0.034 秒、TLS 0.073 秒；暖连接探测 0.036–0.039 秒 | 非该样本数秒延迟主要来源，冷连接仍抖动 |
| 上游等待 | 最终 Qwen 首内容 2.161 秒；Seed 左人分支 1.505 秒 | 无法分离路由、排队、图像 prefill |
| 内容生成 | Qwen 381 输出 token，首末跨度 4.819 秒 | 连接优化无法消除 |
| 完成尾部 | 最后内容至完成 0.107 秒，至 EOF 再 0.001 秒 | 截断省时小且破坏完整性，未采用 |
| 输出协议 | 上游结构化对象仍有缩进，曾达 861–1020 tokens | 紧凑请求不保证短输出；取消 schema 又缺字段/有围栏 |
| 三路并行 | 左右人物 + 全图物体，等最慢分支 | 每帧 3 请求，输入更多，增加重复/漏检 |
| 显示 | 8 秒 deadline、10 秒显示缓冲 | 均不构成 3 秒端到端承诺 |

TLS 优化读取同一 certifi PEM 再用 `cadata`；**151 个 CA 的 DER SHA-256 集合完全一致**，
保留证书/主机名检查及 TLS 1.2 下限。显式 `SSL_CERT_FILE`/`SSL_CERT_DIR` 仍按文件/目录加载，
错误 override 不绕过；保留 `trust_env` 代理语义与 `SSLKEYLOGFILE`。收益不保证适用于 Linux/Docker。

### 办公室完整返回结果

| 模型 / 路由 / 调用 | 秒数 | 质量与决定 |
| --- | --- | --- |
| Qwen3.6 Flash / Alibaba / 原单路 schema | 6.084、5.108、4.714 | 人/物 4/0、3/5、4/0 |
| Qwen3-VL-8B / Alibaba / 单路 | 4.060、2.883 | 第二次只有 3 人 |
| Qwen3-VL-8B / Alibaba / 裁剪并行 | 2.167、1.820 | 仅 2 人 |
| Qwen3-VL-8B / Alibaba / 全图并行、整数示例 | 2.127、1.781 | 仅 1 人，分支同框合并 |
| Qwen3.6 Flash / Alibaba / 裁剪并行 | 4.020、3.050 | 5、2 人，重复及漏人 |
| Qwen3-VL-30B / Alibaba / 单路 | 6.400、5.030 | 均 4 人/8 物，超时延目标 |
| Qwen3.5 35B / Darkbloom / 单路 | 7.300、5.240 | 后次协议不合格；Alibaba 路由也有失败 |
| Qwen3.8 Flash / Alibaba | 两次 12 秒 deadline | 未采用 |
| Seed 2.0 Mini / Seed / named 单路 | 6.890、5.430 | 后次值校验失败 |
| Seed 2.0 Lite / Seed / named 单路 | 9.510、7.450 | 比 Mini 慢 |
| GLM-5.3 Flash / Modal | 一次 3.950，其余及并行有 429 | reasoning 与限流，不纳入可复用 HK 案例 |
| Seed Mini / Seed FP8 / 裁剪并行 720 | 3.708、3.215 | 4/8、5/7，镜像重复人 |
| Seed Mini / Seed FP8 / 裁剪并行 480、H1 | **3.240、3.423** | 4/8、5/7，最接近但不达标 |
| Seed Mini / Seed FP8 / 裁剪并行 480、H2 | 4.626、3.184 | 4/8、5/7，无法证明 H2 更快 |
| Seed Mini / Seed FP8 / 全图并行 720、H2 | 3.954、4.170 | 均 5/7，仍重复 |
| 最终 Qwen3.6 Flash / Alibaba / 单路 H1 | **7.179、5.230、5.530** | 4/5、3/4、4/0，协议通过但速度/质量未验收 |

其他反例与当时决策：

- H2 可共享连接，但未带来一致整帧收益；keepalive 为 60 秒，需 `httpx[http2]`。
  非流式也无稳定优势，继续完整消费流。[HTTPX H2 文档](https://www.python-httpx.org/http2/)。
- throughput 排序、`preferred_min_throughput`、`preferred_max_latency` 只是路由偏好，非时限保证；
  `:nitro` 还可能选 priority tier。最终固定 Alibaba、禁用 fallback。
  [Provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)、
  [Service tiers](https://openrouter.ai/docs/guides/features/service-tiers)。
- 支持时关闭 reasoning 并核对 usage，未假设所有模型遵守。
  [Reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)。
- Seed H1 仍报告 2104/4208 cached input tokens，不能宣称完全无缓存；显式 prompt-cache 布局影响召回，未默认启用。
  [Response caching](https://openrouter.ai/docs/guides/features/response-caching)。
- 480/720、裁剪/全图分区都未实现稳定正确的 3 秒；去示例后 Qwen8B 全图并行甚至 0 人。
  无 schema 出现围栏、缺字段、非法值；降低 token 上限造成截断；named 仅为实验适配协议。

当时落地安全 CA 加载、连接复用/H2、首末 token/DONE/EOF/传输 trace，以及 compact 原始 JSON
严格校验（13 项属性、整数框、数量/scope、重复/未知键、非有限数）；null 表示未知，不补字段伪装输出。
修复 rich 独立路由被 customer-standard 清除及整数坐标示例错误，保留三路完整等待。
新增配置 `vlm_require_complete`、`vlm_http2`、`vlm_trust_env`、`vlm_keepalive_expiry_seconds`、
`vlm_parallel_strategy`；named 未进入生产选项。保存配置仍为 Qwen3.6 Flash / Alibaba、single、compact、H1，
启用上下游完整校验，关闭 reasoning/fallback，保留 8 秒截止与 10 秒缓冲，未部署或发布。

证据目录 `tmp/vlm-latency-20260908/` 包含 `summary.json`、`tls-context-audit.json`、
`quality-review.jpg` 和原始矩阵。旧案例仅 Qwen/Alibaba 与 Seed/Seed FP8；每轮 3 案例、7 子请求，
两轮交替原图/镜像。历史离线测试不能替代 Docker/GPU 或香港实流验收。
若 3 秒为硬约束，异步补属性或邻近部署是尚未验证的架构方向；没有数据支持只调 HTTP 参数可补足差距。

<a id="2026-09-10"></a>
## 2026-09-10：Reka Edge 与 Qwen3-VL-8B 对照

来源 `codex/speed-reka-edge-vs-qwen3-vl-8b`，起点 **`575926f`**，Python 3.12.14。
未改生产客户端、默认配置或部署。输入 SHA-256 为
`85fc60427d7015d68d4ab4e4c7531de84917349462c7e50b3ab7a7926cd1985f`，出口 **KR/ICN**。
OpenRouter 固定 Reka `reka/bf16` 与 Qwen `alibaba`、禁用 fallback，响应 provider 已核对。
当时来源：[Reka 端点](https://openrouter.ai/api/v1/models/rekaai/reka-edge/endpoints)、
[Qwen 端点](https://openrouter.ai/api/v1/models/qwen/qwen3-vl-8b-instruct/endpoints)；
[平台模型页](https://openrouter.ai/rekaai/reka-edge) 汇总延迟不能替代本项目测量。

基准均 single、720 px、compact、上下游严格 Schema、`temperature=0`、`max_tokens=1536`、H1、
安全优化后的证书加载和复用连接池、25 秒截止、零重试。每模型 6 次：Reka→Qwen 先 2 轮，
反序再 4 轮；每轮输入一致，不并发测两模型；各有 2 次客户端首次调用、4 次复用调用。

| 指标 | Reka Edge | Qwen3-VL-8B Instruct |
| --- | ---: | ---: |
| 完整返回中位耗时 | **2.994 秒** | **3.049 秒** |
| 最快–最慢 | 2.749–3.328 秒 | 2.627–4.493 秒 |
| 首内容 token 中位耗时 | 1.265 秒 | 0.547 秒 |
| 严格协议通过 | 6/6 | 6/6 |
| 协议通过且小于 3 秒 | 3/6 | 3/6 |
| 原图人数 / 物品数（3 次一致） | 1 / 1 | 4 / 8 |
| 镜像人数 / 物品数（3 次一致） | 1 / 1 | 3 / 5 |
| 每次输出 tokens：原图 / 镜像 | 116 / 113 | 446 / 318 |
| 六次报告费用 | US$0.00052710 | US$0.001668342 |

Reka 中位低 0.055 秒（1.8%），但输出更少且六次人物框 `[100,100,500,900]`、椅子框
`[600,500,900,950]` 都照抄提示示例，镜像也不变、框落空地；返回 1 人不等于检出 1 人。
Qwen 原图有 4 人但框粗/属性错，镜像仅 3 人、漏右边缘人物。两者均未通过完整准确的 3 秒目标。

### 兼容性与提示反例

每设置/模型各 2 次，本地严格校验始终启用，不修补返回后计成功：

| 设置 | Reka 中位秒数 / 协议通过 | Qwen 中位秒数 / 协议通过 | 观察 |
| --- | --- | --- | --- |
| compact 去示例、保留 Schema | 6.595 / 0/2 | 2.580 / 0/2 | Reka 零面积/无效框；Qwen 属性字典 null |
| named 保留 Schema | 9.419 / 2/2 | 5.004 / 2/2 | Reka 框坐标仅 0–3；Qwen 原图 4 人、镜像 0 人 |
| compact 无上游 Schema | 2.765 / 0/2 | 3.454 / 0/2 | Reka 围栏/错行；Qwen 括号非法 |
| named 无上游 Schema | 4.968 / 0/2 | 2.473 / 0/2 | Reka 缺人物框/枚举错；Qwen 超 8 物且无人物 |

另 4 次短描述（两模型×原图/镜像、128 token 上限）均能描述办公室，不能归因于完全没读到图片。
Reka 回答 4 人但原图虚构站立/走动，Qwen 回答 5 人；该组共用客户端且首个 Reka 含建连，
不用于速度排名、不并入完整标注。

未开显式 prompt-cache，但 Reka 六次累计 3,568 cached input tokens，Qwen 为 0。
合计 **28 次完整标注（16 协议通过、12 拒绝）+4 短描述=32 请求**，均取得完整响应与费用，
usage 总计 **US$0.00546945**，未与账单另行核对。10 个案例逐对确认仅模型/供应商不同。
结论是保留 Qwen 基准、Reka Edge 暂不接入该完整标注流程；不能外推到 Reka 所有视觉任务。

证据目录 `tmp/reka-edge-20260910/`：`compact-screening.jsonl`（每模型 2 次）、
`compact-recheck.jsonl`（反序、每模型 4 次）、`diagnostics.jsonl`（去示例/named，8 次）、
`no-schema.jsonl`（8 次）、`grounding-sanity.jsonl`/`grounding_sanity.py`（4 次短描述）、
`metadata.json`、保留失败分母的 `summary.json`，以及人工复核的 `compact-quality-review.jpg`、
`named-quality-review.jpg`。历史完整复跑为基准 6 轮、诊断 2 轮、25 秒截止并启用传输记录。
