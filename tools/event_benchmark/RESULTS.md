# 测评结果 — spec v1（50 片）

所有数字用的是 [`SPEC.md`](SPEC.md) 冻结的同一套考卷:自带 `question`、gate=0.5、720px、`--workers 12`。
**唯一变量是模型**;下面第二张表额外展示"同一套考卷换采样密度"的敏感度(采样本身也是冻结项,仅作对照)。

## 官方口径 — 1 fps（spec v1 冻结节奏）

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

| 模型 | dense 能力上限(≤14 帧) | **1 fps（spec v1）** | 0.25 fps（产品现状 4s/帧） |
|---|---|---|---|
| gemini | 96% | **88%** | 80% |
| qwen | 92% | **88%** | 70% |
| luna | 88% | **84%** | 74% |

单调关系很干净:**采样越密、准确率越高**。1 fps 居中,是冻结口径。
⏳ 待定:生产当前 0.25 fps,是否统一到 1 fps 待 Frank 确认(见 SPEC §4)。

## 复现

```sh
python tools/event_benchmark/run_eval.py --candidate luna   --out results/luna_1fps.json
python tools/event_benchmark/run_eval.py --candidate qwen   --out results/qwen_1fps.json
python tools/event_benchmark/run_eval.py --candidate gemini --out results/gemini_1fps.json
# 对照节奏: 加 --interval 4 (产品 0.25fps) 或 --dense (能力上限)
```
