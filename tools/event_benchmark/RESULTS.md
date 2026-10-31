# 测评结果 — spec v1

📊 **实时汇总表（running Google Sheet）**：<https://docs.google.com/spreadsheets/d/1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA/edit> — 所有模型最新结果，数字带红→绿色阶可排序；`results` tab 为准确率主表，`threshold-tuning` tab 为 luna 阈值调参。由 `bench-sheet-push` skill 推送（预览 → 确认 → 写入，用会话内 Google OAuth，不存密钥）。

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

## 100 题 base-4 扩展集（provisional，GT 部分待审）

`results/{gemini,qwen,luna}_100base.json`。取 `media_extra_50/samples.json` 里 base-4 的 100 题（fight 32 · smoking 22 · fall 26 · weapon 20；bag_knife/bag_alcohol 共 20 题属 v1.1，不计入 base-4，`run_eval` 自动过滤）。**66/120 GT 仍 `pending_user_review`，故本集标 provisional。**

| 模型(类) | fight | smoking | fall | weapon | 总体 |
|---|---|---|---|---|---|
| gemini (language) | 30/32 | 21/22 | 22/26 | 17/20 | **90/100 = 90%** |
| qwen (language) | 31/32 | 21/22 | 22/26 | 15/20 | **89/100 = 89%** |
| luna (decision) | 30/32 | 20/22 | 18/26 | 17/20 | **85/100 = 85%** |

混淆明细(TP/FN/TN/FP):

| 模型 | fight | smoking | fall | weapon |
|---|---|---|---|---|
| gemini | 16/0/14/2 | 16/1/5/0 | 9/4/13/0 | 8/1/9/2 |
| qwen | 15/1/16/0 | 16/1/5/0 | 9/4/13/0 | 4/5/11/0 |
| luna | 14/2/16/0 | 15/2/5/0 | 12/1/6/**7** | 6/3/11/0 |

- 本集 smoking/weapon 有负例（不同于 50 片全阳性），可算误报；luna fall 误报偏多（FP7）。

## luna 阈值调参（held-out 24 验证集，SPEC §5/§8 脚注）

**只调 decision 类（luna）**；qwen/gemini 输出硬 0/1，阈值无意义。用独立 24 片验证集（4×6，3 正 3 负）每类选最优阈值（验证集 max-margin 中点），固定后在 test 上评。主表仍冻结在 gate=0.5，本节为脚注、不替换主表。

每类阈值：**fight 0.29 · smoking 0.53 · fall 0.58 · weapon 0.46**

| test | @0.5 基线 | @调参后 |
|---|---|---|
| 50 片 | 84% | **94%** |
| 100 片 | 85% | **92%** |

提升几乎全来自 fall：luna 在 0.5 时 fall 误报多（50 片 FP4、100 片 FP7），fall 阈值抬到 0.58 后这些误报消失。细节见 `results/luna_threshold_tuning.json` 与表格 `threshold-tuning` tab。
（注：adl-24、adl-28 两片验证视频暂未下载，smoking/fall 验证各用 5 片；补齐后阈值预计不变，因该两片为负例、分数远低于阈值。）

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

# 100 题 base-4 扩展集 (每个候选):
python tools/event_benchmark/run_eval.py --candidate <cand> \
  --samples <…/media_extra_50/samples.json> \
  --media <…/media50/media/media> <…/media_extra_50/新增基础视频_50段> \
  --out results/<cand>_100base.json

# luna 阈值调参 (先把 24 验证集放到 media_val_24/):
python tools/event_benchmark/run_eval.py --candidate luna \
  --samples <…/media_val_24/samples.json> --media <…/media_val_24/videos> \
  --out results/luna_val24.json
python tools/event_benchmark/tune_threshold.py \
  --val results/luna_val24.json \
  --test results/luna_1fps.json --test results/luna_100base.json \
  --out results/luna_threshold_tuning.json
```
