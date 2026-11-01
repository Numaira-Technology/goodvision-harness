# 测评结果 — spec v1（50 片）

以下为保留的历史 v1 结果，**不是修正后 v2 runner 的验证结果**。旧 runner 会把部分非法回答当作阴性或跳过丢失帧，且这些输出没有媒体/答案指纹；目前无法从存档排除这些问题。v2 比较需重新跑全部候选，本次代码审查没有调用付费 API。

原始规则见 [`SPEC_V1.md`](SPEC_V1.md):自带 `question`、gate=0.5、720px、`--workers 12`。
**唯一变量是模型**;下面第二张表额外展示"同一套考卷换采样密度"的敏感度(采样本身也是冻结项,仅作对照)。

## 历史口径 — 1 fps（spec v1）

`results/{luna,qwen,gemini}_1fps.json` 为原始逐片输出。

| 模型(类) | fight | smoking | fall | weapon | 总体 |
|---|---|---|---|---|---|
| qwen (language) | 12/12 | 11/12 | 17/21 | 4/5 | **44/50 = 88%** |
| gemini (language) | 11/12 | 11/12 | 17/21 | 5/5 | **44/50 = 88%** |
| luna (decision) | 11/12 | 10/12 | 16/21 | 5/5 | **42/50 = 84%** |

混淆明细(TP/FN/TN/FP):

| 模型 | fight | smoking | fall | weapon |
|---|---|---|---|---|
| qwen | 6/0/6/0 | 11/1/—/— | 8/4/9/0 | 4/1/—/— |
| gemini | 6/0/5/**1** | 11/1/—/— | 8/4/9/0 | 5/0/—/— |
| luna | 5/1/6/0 | 10/2/—/— | 11/1/5/**4** | 5/0/—/— |

- smoking、weapon 本集**全阳性**,无负例 → 不计误报率/特异度(标 `—`)。
- 1 fps 下 qwen ≈ gemini(88%)略高于 luna(84%);gemini 有 1 个 fight 误报,luna fall 误报偏多(4)。

## 采样密度敏感度(同一套考卷,仅换节奏)

| 模型 | dense 能力上限(≤14 帧) | **1 fps（spec v1）** | 0.25 fps（历史 4s/帧) |
|---|---|---|---|
| gemini | 96% | **88%** | 80% |
| qwen | 92% | **88%** | 70% |
| luna | 88% | **84%** | 74% |

该附表是历史记录；本 PR 未包含 dense / 0.25 fps 的逐片记录，不能从此仓库复核。dense 上限为 14 帧，长片中可能比 1 fps 更稀疏，不能由此推导采样密度与准确率的普遍关系。

## v2 重跑（不覆盖历史结果）

```sh
python tools/event_benchmark/run_eval.py --candidate luna --samples /path/to/samples.json --media /path/to/media --out outputs/event-benchmark/luna-v2.json
python tools/event_benchmark/run_eval.py --candidate qwen --samples /path/to/samples.json --media /path/to/media --out outputs/event-benchmark/qwen-v2.json
python tools/event_benchmark/run_eval.py --candidate gemini --samples /path/to/samples.json --media /path/to/media --out outputs/event-benchmark/gemini-v2.json
# 对照节奏: 加 --interval 4 (产品 0.25fps) 或 --dense (能力上限)
```
