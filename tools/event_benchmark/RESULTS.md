# 历史测评结果 — spec v1

📊 **实时汇总表（running Google Sheet）**：<https://docs.google.com/spreadsheets/d/1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA/edit> — 所有模型最新结果，数字带红→绿色阶可排序；`results` tab 为准确率主表，`threshold-tuning` tab 为 luna 阈值调参。由 `bench-sheet-push` skill 推送（预览 → 确认 → 写入，用会话内 Google OAuth，不存密钥）。

本页的 **50 片、100 题 base-4 和 luna 阈值调参均为历史 v1 输出**，不是当前 v2 runner 的验收结果。新增历史输出中，Gemini/Luna/Qwen 分别有 1/6/50 行含失败帧；验证集 24 行只有 22 个分数，且均无 v2 provenance。冻结考卷规则见 [`SPEC.md`](SPEC.md)；历史 50 片沿用的原始规则见 [`SPEC_V1.md`](SPEC_V1.md)。**唯一变量是模型**。

---

## 历史 v1 — 50 片（存档，非 v2 验证结果）

> 以下为保留的历史 v1 结果，**不是修正后 v2 runner 的验证结果**。旧 runner 会把部分非法回答当作阴性或跳过丢失帧，且这些输出没有媒体/答案指纹；目前无法从存档排除这些问题。v2 比较需重新跑全部候选，本次代码审查没有调用付费 API。

### 1 fps（spec v1 冻结节奏）

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

### 采样密度敏感度(同一套考卷,仅换节奏)

| 模型 | dense 能力上限(≤14 帧) | **1 fps（spec v1）** | 0.25 fps（历史 4s/帧) |
|---|---|---|---|
| gemini | 96% | **88%** | 80% |
| qwen | 92% | **88%** | 70% |
| luna | 88% | **84%** | 74% |

该附表是历史记录；本仓库未包含 dense / 0.25 fps 的逐片记录，不能从此复核。dense 上限为 14 帧，长片中可能比 1 fps 更稀疏，**不能**由此推导"采样越密准确率越高"的普遍规律。1 fps 是冻结口径；生产历史上的 0.25 fps 只是 VLM 吞吐限制下的妥协，现统一到 1 fps（见 SPEC §4）。

---

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

**只调 decision 类（luna）**；qwen/gemini 输出硬 0/1，阈值无意义。历史拟定验证集有 24 片，但实际只对 22 个可用分数选阈（并列最优网格点的中位数），随后对历史 test 分数做算术比较。该流程未完成 v2 冻结与评分资格校验。主表仍冻结在 gate=0.5，本节为脚注、不替换主表。

每类阈值：**fight 0.29 · smoking 0.53 · fall 0.58 · weapon 0.46**

| test | @0.5 基线 | @调参后 |
|---|---|---|
| 50 片 | 84% | **94%** |
| 100 片 | 85% | **92%** |

提升几乎全来自 fall：luna 在 0.5 时 fall 误报多（50 片 FP4、100 片 FP7），fall 阈值抬到 0.58 后这些误报消失。细节见 `results/luna_threshold_tuning.json` 与表格 `threshold-tuning` tab。
（注：adl-24、adl-28 在存档中仅有 missing file 错误，没有分数。不能推断它们低于阈值，也不能预言补齐后的阈值不变。）

---

## 当前工具与后续 v2 验证

`tune_threshold.py` 现在只接受带媒体/答案/runner/spec 指纹的 v2 decision 输出，拒绝
validation/test 交集（文件名与媒体 SHA-256）、模型或采样不一致、无效概率和部分帧分数。
验证集必须完整且每类有正负例；测试集错误单列 attempted/scored/errors，不进入分数分母。
历史 v1 文件保留原样，不能输入当前工具来生成声称有效的 v2 校准报告。

本仓库尚无留出集的已审阅 v2 manifest；`validation_split.json` 只是历史候选清单。
120 题 manifest 仍为 draft。必须先完成答案审阅和媒体/答案哈希清单，再对所有候选重跑；
不得把历史 JSON 改写为 v2，也不得据此发布 v2 比较。

下面命令中的路径须替换成已审阅的实际文件。输出写入忽略目录 `outputs/`，不覆盖历史存档：

```sh
python tools/event_benchmark/run_eval.py --candidate luna \
  --samples /path/to/validation/samples.json \
  --manifest /path/to/validation/manifest-reviewed.json \
  --media /path/to/validation/videos --out outputs/luna_validation_v2.json
python tools/event_benchmark/run_eval.py --candidate luna \
  --samples /path/to/test/samples.json \
  --manifest /path/to/test/manifest-reviewed.json \
  --media /path/to/test/videos --out outputs/luna_test_v2.json
python tools/event_benchmark/tune_threshold.py \
  --val outputs/luna_validation_v2.json --test outputs/luna_test_v2.json \
  --out outputs/luna_thresholds_v2.json
```
