# Reka Edge 与 Qwen3-VL-8B 提速测试 · 2026-09-10

**本轮没有找到可替换 Qwen3-VL-8B 的 Reka Edge 提速配置。**
在现有完整人物/物品标注流程中，两者中位耗时接近，但 Reka 连续六次复用了提示词示例中的错误坐标。
Qwen 的标注更贴近画面，但镜像图稳定漏掉边缘人物，也没有通过完整准确的 3 秒目标。

## 独立工作区

- Worktree：`D:/Documents/Project Bird/code/goodvision-harness/.worktrees/speed-reka-edge-vs-qwen3-vl-8b`
- 分支：`codex/speed-reka-edge-vs-qwen3-vl-8b`
- 起点：当前工作区已提交版本 `575926f`。
- 新增两个测速案例文件及本报告；生产客户端、默认模型、路由和部署配置没有改动。
- 复用现有 Python 3.12.14 环境及测速工具；媒体、原始响应和诊断脚本保存在本 worktree 的忽略目录 `tmp/reka-edge-20260910/`。

## 测试条件

沿用先前测速办公室图片 `../../tmp/server-live-left.jpg`，720×405，人工检查可见 4 人，包括左侧被画面裁切的人。
输入文件 SHA-256：`85fc60427d7015d68d4ab4e4c7531de84917349462c7e50b3ab7a7926cd1985f`。
交替原图和水平镜像；这是同一场景的受控重复，不是多场景准确率评测。

两者均通过 OpenRouter。Reka 固定 `reka/bf16`，Qwen 固定 `alibaba`，关闭 fallback；响应中的 provider 分别核对为 Reka 和 Alibaba。
模型和后端信息来自实测时的 [Reka 端点 API](https://openrouter.ai/api/v1/models/rekaai/reka-edge/endpoints) 与
[Qwen 端点 API](https://openrouter.ai/api/v1/models/qwen/qwen3-vl-8b-instruct/endpoints)，快照保存在 `metadata.json`。
[Reka 模型页](https://openrouter.ai/rekaai/reka-edge) 的平台汇总延迟不能代替本项目完整标注耗时。

网络出口当时返回 **KR / ICN**，因此本轮不是香港出口验收。

基准条件一致：单路整帧、720 长边、compact 格式、上游 JSON Schema、本地严格完整性校验、
`temperature=0`、`max_tokens=1536`、HTTP/1.1、优化后的证书加载、保持证书校验、复用连接池、25 秒期限、零自动重试。
模型调用由项目原生 `VLMRichLabelClient.infer()` 执行，测到完整响应与解析结束；包含服务内部图像处理，
不包含最外层文件读取/JPEG 准备和离线 Schema 复核。首内容 token 单列，不作为整帧完成时间。

每个基准模型 6 次：先按 Reka→Qwen 跑 2 轮，再反转顺序跑 4 轮；每轮两者输入像素一致。
每个模型有 2 次客户端首次调用与 4 次复用调用。未在同一时间并发运行两个模型测速。

## 当前完整标注流程：每模型 6 次

| 指标 | Reka Edge | Qwen3-VL-8B Instruct |
|---|---:|---:|
| 完整返回中位耗时 | **2.994 秒** | **3.049 秒** |
| 最快–最慢 | 2.749–3.328 秒 | 2.627–4.493 秒 |
| 首内容 token 中位耗时 | 1.265 秒 | 0.547 秒 |
| 严格协议通过 | 6/6 | 6/6 |
| 协议通过且小于 3 秒 | 3/6 | 3/6 |
| 原图返回人数 / 物品数（3 次均相同） | 1 / 1 | 4 / 8 |
| 镜像返回人数 / 物品数（3 次均相同） | 1 / 1 | 3 / 5 |
| 每次输出 tokens：原图 / 镜像 | 116 / 113 | 446 / 318 |
| 六次响应报告费用合计 | US$0.00052710 | US$0.001668342 |

Reka 的中位数低约 0.055 秒（1.8%）；少量同场景重复不足以证明有稳定速度优势。
它输出的内容也明显更少，无法据此推断同等质量的吞吐或成本优势。

Reka 六次人物框均为 `[100,100,500,900]`，椅子框均为 `[600,500,900,950]`，
与生产提示中的结构示例一致。人物框大面积落在空地，镜像后仍不变化；返回的 1 人不代表正确检出了 1 人。
Qwen 原图返回 4 人，但框较粗、属性存在明显不符；镜像只返回 3 人，漏掉右边缘人物。
因此两者的协议合格率均不能解释为视觉准确率，也不能将 3/6 的小于 3 秒结果作为达标。

叠框人工复核文件：`tmp/reka-edge-20260910/compact-quality-review.jpg`。
图中保留模型原始标注供检查，不代表认可属性推断；未建立逐框 IoU 或属性真值集。

## 兼容性与提示实验：每种设置、每模型 2 次

每对模型使用相同设置，本地严格校验始终开启；没有修补返回内容后再算成功。

| 设置 | Reka 中位耗时 / 协议通过 | Qwen 中位耗时 / 协议通过 | 观察 |
|---|---:|---:|---|
| compact 去掉结构示例，保留 Schema | 6.595 秒 / 0/2 | 2.580 秒 / 0/2 | Reka 输出零面积/无效框；Qwen 人物属性字典为 null |
| named 具名字段，保留 Schema | 9.419 秒 / 2/2 | 5.004 秒 / 2/2 | Reka 人物框只在 0–3 的坐标范围，挤在左上角；Qwen 原图 4 人、镜像 0 人 |
| compact 关闭上游 Schema | 2.765 秒 / 0/2 | 3.454 秒 / 0/2 | Reka 有 Markdown 围栏、错误行结构；Qwen JSON 括号不合法 |
| named 关闭上游 Schema | 4.968 秒 / 0/2 | 2.473 秒 / 0/2 | Reka 缺人物框且枚举值不合格；Qwen 超过 8 个物品上限且未返回人物 |

关闭上游格式约束没有恢复可用输出。named 的合法 JSON 也没有改善 Reka 的坐标质量。
叠框对照：`tmp/reka-edge-20260910/named-quality-review.jpg`。
这些结果只覆盖当前 OpenRouter 后端、图片与项目提示，不能据此判定 Reka 所有视觉任务的能力。

另做了 4 次短提示图像描述检查（两模型×原图/镜像，128 token 上限、不要求完整标注）。
两者都能描述办公室场景，不能简单把完整标注失败归因于完全没有读到图片。
Reka 回答 4 人，但原图描述虚构了站立和走动；Qwen 回答 5 人，计数偏多。
该检查共用一个 HTTP 客户端，首个 Reka 请求含建连，因此**不用于速度排名**，也不与上表合并。
证据和复现脚本为 `grounding-sanity.jsonl`、`grounding_sanity.py`。

## 缓存、费用与验证

- 所有请求发送 `X-OpenRouter-Cache: false`；没有开启项目显式 prompt-cache 布局。
- 这不等于关闭供应商前缀缓存。Reka 基准六次报告累计 3,568 cached input tokens，Qwen 为 0；全部原样保留，不能称为完全无缓存测速。
- 共 **28 次完整标注尝试 + 4 次短描述检查 = 32 次请求**；完整标注中 16 次协议通过，12 次被严格校验拒绝。
- 全部 32 次均取得完整响应及费用记录，报告费用合计 **US$0.00546945**。这是 API usage 报告值，未另做账单核对。
- 现有测速工具与设置测试：**77 passed**。
- 两份文件合计 10 个案例均成功加载；逐对核对除模型/供应商外设置一致、固定后端、禁止 fallback。
- 没有修改产品默认配置或发布部署。本轮结论是保留 Qwen 基准，Reka Edge 暂不接入当前完整标注链路。

## 复现与记录

在本 worktree 执行以下命令会产生付费请求。原始基准分两次运行并反转模型顺序；下例提供同配置的六轮复跑。
文件名应使用新的路径，测速工具会追加写入已有输出。

```powershell
$benchmarkPython = '../person-event-evidence/.venv/Scripts/python.exe'
& $benchmarkPython -B tools/vlm_speed_matrix.py --env-file '../../.env' --image '../../tmp/server-live-left.jpg' --cases tools/vlm_speed_cases.reka-vs-qwen8b.json --output tmp/reka-edge-rerun/baseline.jsonl --repeats 6 --timeout 25 --trace-transport
& $benchmarkPython -B tools/vlm_speed_matrix.py --env-file '../../.env' --image '../../tmp/server-live-left.jpg' --cases tools/vlm_speed_cases.reka-vs-qwen8b-diagnostics.json --output tmp/reka-edge-rerun/diagnostics.jsonl --repeats 2 --timeout 25 --trace-transport
```

若复用环境路径不可用，可使用装好本项目依赖和 `dev` 依赖的 Python；工具主动导入本 worktree 的源码。
密钥仅从项目 `.env` 中读取所需变量，没有复制进测试工作区或写入报告。

本地原始记录位于 `tmp/reka-edge-20260910/`：

- `compact-screening.jsonl`：基准首轮，每模型 2 次。
- `compact-recheck.jsonl`：反转模型顺序复跑，每模型 4 次。
- `diagnostics.jsonl`：去示例、named Schema 实验，共 8 次。
- `no-schema.jsonl`：关闭上游 Schema 实验，共 8 次。
- `grounding-sanity.jsonl`：短描述检查，共 4 次。
- `metadata.json`：模型后端和网络出口快照。
- `summary.json`：只汇总 trial、保留失败分母的统计结果。
- `compact-quality-review.jpg`、`named-quality-review.jpg`：已人工检查的叠框对照图。
