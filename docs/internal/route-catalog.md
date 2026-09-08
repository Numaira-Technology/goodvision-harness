# 内部线路对应表与实测记录

内部维护资料，不复制到客户界面、公开 API、事件 JSON 或交付说明中。公开部分仅展示线路代号、状态和耗时。已有代号不得重用来指向其他 provider/model 组合。

## 当前保留的 8 条线路

2026-09-08 从 44 条候选中按完整响应成功率和中位耗时筛选，排除复测失败的线路；按用户最新要求移除 Novita 和 CoreWeave，使用后续排名中的 B、F 补位。当前 8 条覆盖 3 个 provider、6 个 Qwen 版本，均为 Alibaba、DeepInfra 或 Parasail。以下耗时来自全量初测；最后一节记录移除 provider 后的八条复测。

| 代号 | Provider / endpoint slug | 模型 | 35B 优先组 | 中位耗时（秒） | 三次计分耗时（秒） |
|---|---|---|---|---:|---|
| C | Alibaba / `alibaba` | `qwen/qwen3-vl-8b-instruct` | 否 | 1.188 | 1.219 / 1.187 / 1.188 |
| D | Alibaba / `alibaba` | `qwen/qwen3.5-35b-a3b` | 是 | 1.579 | 1.579 / 1.563 / 1.625 |
| AA | Alibaba / `alibaba` | `qwen/qwen3-vl-30b-a3b-instruct` | 否 | 1.687 | 1.734 / 1.579 / 1.687 |
| AE | DeepInfra / `deepinfra/fp4` | `qwen/qwen3.5-122b-a10b` | 否 | 1.718 | 1.766 / 1.718 / 1.391 |
| G | Parasail / `parasail/fp8` | `qwen/qwen3.6-35b-a3b` | 是 | 2.141 | 3.219 / 2.141 / 1.390 |
| K | Alibaba / `alibaba` | `qwen/qwen3.5-27b` | 否 | 2.219 | 2.156 / 2.219 / 3.469 |
| B | Parasail / `parasail/fp8` | `qwen/qwen3.5-35b-a3b` | 是 | 2.672 | 2.469 / 2.703 / 2.672 |
| F | DeepInfra / `deepinfra/fp8` | `qwen/qwen3.6-35b-a3b` | 是 | 2.859 | 2.359 / 3.016 / 2.859 |

历史全量测试曾选 H（1.516 秒），现已按用户要求禁用。I 初测最快（1.078 秒），但首次八条复测出现请求失败，继续排除。H、AB 被 B、F 替代；被移除 provider 的全部历史代号 H、R、L、S、AB、AG、AK 均不能选中、保存、恢复或通过线路对象发起请求。原始复测保存在 tmp/qwen-route-expansion-20260908/results-shortlist-first.json。

## 测速条件与范围

测试时间：2026-09-08T19:30:08+08:00 至 2026-09-08T19:33:36+08:00。当前机器网络出口为 TW / TPE；香港出口验收尚未进行。用户已确认先按官方地区说明筛选，在当前出口测速，香港出口稍后复测。

共 44 条线路、7 个 provider、22 个图像输入/文本输出 Qwen 模型条目。39 条通过全部三次计分检查；V、W、Y、AD、AR 有请求失败，未进入候选。每条 1 次预热 + 3 次计分，共 176 次请求尝试。目录快照只选择 status=0 的可用端点，并对图像输入能力逐条校验。

输入为程序生成的 320×180、JPEG Q75 匿名示意图，沿用本地真实四事件 prompt、完整 JSON 合约和至少 1200 输出 token 上限。计时覆盖完整模型请求及响应解析，预热、生成测试图和任务等待不计分。没有使用或发送摄像头帧、stranger 照片或其他私人影像来比较新增线路。

最多 4 条线路并发，每条内部顺序取样；共享 6 连接的证书验证 HTTPS 连接池，排队发生在请求计时前。测试阶段不重试，连接/read 等 HTTP 超时设为不超过 20 秒。所有请求固定 provider.only，关闭 fallback，使用现有 OpenRouter 凭据。控制台失败信息保持通用，不回显上游正文或密钥。

测量的是热连接完整响应延迟，不是首 token、模型识别准确率、摄像头到画面的端到端延迟或香港网络表现。三次样本反映当前负载，接近的排名可能随线路负载改变。保留 8 条后，日常手动检查只发起 32 次请求尝试。

## 选择与兼容规则

1. 只有三次完整响应全部成功的线路可参选。
2. 如果存在中位耗时严格小于 2.5 秒的 35B 线路，选其中最快者。2.500 秒不满足优先条件。
3. 否则选择所有合格线路中最快者。全部失败或无法保存时保留原选择。
4. 只对表中 8 条线路进行日常测速；允许的历史代号可恢复已有配置，被明确移除 provider 的历史代号禁止恢复。测速与检测运行互斥。

选择以代号保存在数据目录 selected-route.json；重启后保留线路并清空测速进度。没有持久化目录时仅在当前进程有效。加载时由内部白名单解析，文件不能注入任意模型、provider 或 endpoint。

GET /v1/deployment/routes 返回 8 条不含实现名称的候选；POST /v1/deployment/routes/check 返回 202 后轮询 GET。接口不接收任意 provider/model 或图像，响应仅含代号、状态、耗时和通用错误码。

## 香港地区筛选依据（2026-09-08）

这是一轮政策初筛，不代表供应商承诺香港账户均可使用，也不是香港出口实测通过。以下资料没有发现明确排除香港的表述；新增 provider 只按这一有限结论进入本轮比较，仍需香港网络复测。

| Provider | 官方来源及筛选依据 | 本轮状态 |
|---|---|---|
| Alibaba / Qwen | [Model Studio 地区说明](https://www.alibabacloud.com/help/en/model-studio/regions) 明列 China (Hong Kong) 的实时推理支持；用户也已确认。该直连地区说明不等同于每个 OpenRouter 端点的香港 SLA。 | 纳入；香港出口待复测 |
| DeepInfra | 用户此前确认香港可调用；[服务条款](https://deepinfra.com/terms) 作为官方参考。 | 纳入；香港出口待复测 |
| Parasail | 用户明确同意纳入；[服务条款](https://parasail.io/legal/terms-of-service) 作为官方参考。 | 纳入；香港出口待复测 |
| CoreWeave | [官方服务条款](https://docs.coreweave.com/policies/terms-of-service) 为历史筛选依据；现不再参与调用。 | 已按用户要求移除 |
| Novita | [官方服务条款](https://novita.ai/legal/terms-of-service) 与 [使用政策](https://novita.ai/legal/acceptable-use-policy) 为历史筛选依据；现不再参与调用。 | 已按用户要求移除 |
| Chutes | [官方服务条款](https://chutes.ai/terms) 未发现明确香港地区禁用条款。 | 全量比较过，性能未进前八 |
| Reka | [官方服务条款](https://reka.ai/legal/terms-of-use) 的 International Users 段落讨论跨国访问，同时不保证各地可用；未明示排除香港。 | 全量比较过，性能未进前八 |

SiliconFlow 的 [国际站条款](https://docs.siliconflow.com/en/legals/terms-of-service) 包含地区范围措辞，香港适用性未在本轮明确，暂未纳入；其他未完成官方地区核对的 provider 同样不纳入。批处理专用条目和独立 thinking 变体不进入此次实时非思考响应比较。

模型和端点基于 [OpenRouter 图像 Qwen 目录](https://openrouter.ai/models?input_modalities=image&q=qwen)、[官方模型 API](https://openrouter.ai/api/v1/models) 及对应 /api/v1/models/{model_id}/endpoints 实时响应核对。原始快照保存在本地 tmp/qwen-route-expansion-20260908/vision-models.json 和 endpoints.jsonl，不含凭据。

## 全部 44 条历史映射与结果

下表保留全量证据，只有标为“保留”的 8 条出现在日常测速列表。历史 A–E 的对应关系未变。

| 代号 | endpoint slug | 模型 | 35B | 成功次数 | 中位秒 | 本轮处置 |
|---|---|---|---|---:|---:|---|
| A | `deepinfra` | `qwen/qwen3.5-35b-a3b` | 是 | 3/3 | 6.969 | 性能未进前八 |
| B | `parasail/fp8` | `qwen/qwen3.5-35b-a3b` | 是 | 3/3 | 2.672 | 保留 |
| C | `alibaba` | `qwen/qwen3-vl-8b-instruct` | 否 | 3/3 | 1.188 | 保留 |
| D | `alibaba` | `qwen/qwen3.5-35b-a3b` | 是 | 3/3 | 1.579 | 保留 |
| E | `alibaba` | `qwen/qwen3.5-flash-02-23` | 否 | 3/3 | 3.593 | 性能未进前八 |
| F | `deepinfra/fp8` | `qwen/qwen3.6-35b-a3b` | 是 | 3/3 | 2.859 | 保留 |
| G | `parasail/fp8` | `qwen/qwen3.6-35b-a3b` | 是 | 3/3 | 2.141 | 保留 |
| H | `coreweave/fp8` | `qwen/qwen3.6-35b-a3b` | 是 | 3/3 | 1.516 | 按用户要求移除 provider |
| I | `parasail/bf16` | `qwen/qwen3.5-9b` | 否 | 3/3 | 1.078 | 复测请求失败排除 |
| J | `deepinfra/fp8` | `qwen/qwen3.5-27b` | 否 | 3/3 | 4.266 | 性能未进前八 |
| K | `alibaba` | `qwen/qwen3.5-27b` | 否 | 3/3 | 2.219 | 保留 |
| L | `novita/bf16` | `qwen/qwen3.5-27b` | 否 | 3/3 | 5.360 | 按用户要求移除 provider |
| M | `deepinfra/fp8` | `qwen/qwen3.6-27b` | 否 | 3/3 | 4.297 | 性能未进前八 |
| N | `alibaba` | `qwen/qwen3.6-27b` | 否 | 3/3 | 10.563 | 性能未进前八 |
| O | `chutes/fp8` | `qwen/qwen3.6-27b` | 否 | 3/3 | 7.516 | 性能未进前八 |
| P | `alibaba` | `qwen/qwen3.8-27b` | 否 | 3/3 | 4.703 | 性能未进前八 |
| Q | `parasail/fp8` | `qwen/qwen3.8-27b` | 否 | 3/3 | 3.469 | 性能未进前八 |
| R | `coreweave/fp8` | `qwen/qwen3.8-27b` | 否 | 3/3 | 3.625 | 按用户要求移除 provider |
| S | `novita` | `qwen/qwen3.8-27b` | 否 | 3/3 | 5.453 | 按用户要求移除 provider |
| T | `chutes/fp8` | `qwen/qwen3.8-27b` | 否 | 3/3 | 4.437 | 性能未进前八 |
| U | `reka/fp8` | `qwen/qwen3.8-27b` | 否 | 3/3 | 3.718 | 性能未进前八 |
| V | `alibaba` | `qwen/qwen3.8-flash` | 否 | 1/3 | — | 请求失败排除 |
| W | `alibaba` | `qwen/qwen3.7-flash` | 否 | 1/3 | — | 请求失败排除 |
| X | `alibaba` | `qwen/qwen3.6-flash` | 否 | 3/3 | 6.172 | 性能未进前八 |
| Y | `parasail/bf16` | `qwen/qwen3-vl-8b-instruct` | 否 | 0/3 | — | 请求失败排除 |
| Z | `deepinfra/fp8` | `qwen/qwen3-vl-30b-a3b-instruct` | 否 | 3/3 | 4.859 | 性能未进前八 |
| AA | `alibaba` | `qwen/qwen3-vl-30b-a3b-instruct` | 否 | 3/3 | 1.687 | 保留 |
| AB | `novita/bf16` | `qwen/qwen3-vl-30b-a3b-instruct` | 否 | 3/3 | 2.156 | 按用户要求移除 provider |
| AC | `alibaba` | `qwen/qwen3-vl-32b-instruct` | 否 | 3/3 | 3.062 | 性能未进前八 |
| AD | `parasail/fp8` | `qwen/qwen2.5-vl-72b-instruct` | 否 | 1/3 | — | 请求失败排除 |
| AE | `deepinfra/fp4` | `qwen/qwen3.5-122b-a10b` | 否 | 3/3 | 1.718 | 保留 |
| AF | `alibaba` | `qwen/qwen3.5-122b-a10b` | 否 | 3/3 | 9.360 | 性能未进前八 |
| AG | `novita/bf16` | `qwen/qwen3.5-122b-a10b` | 否 | 3/3 | 2.922 | 按用户要求移除 provider |
| AH | `deepinfra/fp8` | `qwen/qwen3.5-397b-a17b` | 否 | 3/3 | 5.938 | 性能未进前八 |
| AI | `alibaba` | `qwen/qwen3.5-397b-a17b` | 否 | 3/3 | 5.594 | 性能未进前八 |
| AJ | `parasail/fp8` | `qwen/qwen3.5-397b-a17b` | 否 | 3/3 | 8.734 | 性能未进前八 |
| AK | `novita` | `qwen/qwen3.5-397b-a17b` | 否 | 3/3 | 6.110 | 按用户要求移除 provider |
| AL | `alibaba` | `qwen/qwen3-vl-235b-a22b-instruct` | 否 | 3/3 | 3.437 | 性能未进前八 |
| AM | `parasail/fp8` | `qwen/qwen3-vl-235b-a22b-instruct` | 否 | 3/3 | 2.969 | 性能未进前八 |
| AN | `alibaba` | `qwen/qwen3.5-plus-02-15` | 否 | 3/3 | 12.750 | 性能未进前八 |
| AO | `alibaba` | `qwen/qwen3.5-plus-20260420` | 否 | 3/3 | 12.062 | 性能未进前八 |
| AP | `alibaba` | `qwen/qwen3.6-plus` | 否 | 3/3 | 5.688 | 性能未进前八 |
| AQ | `alibaba` | `qwen/qwen3.7-plus` | 否 | 3/3 | 5.218 | 性能未进前八 |
| AR | `alibaba` | `qwen/qwen3.8-max-0902` | 否 | 0/3 | — | 请求失败排除 |

完整逐次耗时原始记录：tmp/qwen-route-expansion-20260908/results-all-44.json。

## 本地交付状态

本地源码控制台：http://127.0.0.1:8081/ 。只更新源码并重启该本地服务；没有构建或替换 Docker 交付镜像。原 8080 Docker demo 保持原版本。

之前的完整回归：642 passed, 4 skipped（Windows Python 3.12，2026-09-08）。移除两个 provider 后，线路、持久化、默认模型路由及 HTTP 客户端相关测试共 105 项通过，覆盖旧选路禁止恢复及发起请求。

### 移除 provider 后的八条复测

复测时间：2026-09-08T19:49:34+08:00 至 2026-09-08T19:49:58+08:00；选择 D，原因 preferred_below_threshold。

| 线路 | 成功次数 | 中位秒 | 三次计分耗时（秒） |
|---|---:|---:|---|
| C | 3/3 | 1.313 | 1.296 / 1.313 / 1.328 |
| D | 3/3 | 1.844 | 2.313 / 1.844 / 1.609 |
| AA | 3/3 | 1.563 | 1.703 / 1.563 / 1.547 |
| AE | 3/3 | 2.719 | 2.719 / 3.062 / 2.391 |
| G | 3/3 | 2.781 | 2.781 / 2.922 / 2.703 |
| K | 3/3 | 3.046 | 3.046 / 4.797 / 2.172 |
| B | 3/3 | 2.078 | 1.954 / 2.218 / 2.078 |
| F | 3/3 | 2.954 | 2.890 / 2.954 / 4.750 |

本轮复测原始记录：tmp/qwen-route-expansion-20260908/results-provider-filtered-eight.json；之前的八条结果保留在 results-final-eight.json，仅为历史记录。
