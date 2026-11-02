# 历史测评结果 — spec v1

📊 **团队汇总表**：<https://docs.google.com/spreadsheets/d/1D3-fwFW5Fv1ASDlbt3VUIFJJfAKKsrH2LXVNMkoPLjA/edit>。以各行的来源、版本与审阅状态为准；需要发布时使用 `bench-sheet-contribute` skill 核对实时表头、目标行与授权，用会话内 Google OAuth，不存密钥。本页的历史结果不代表已审阅的 v2 数据。

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

## Bag 存档：未审阅的二值实验，不是 v2 或三值任务验收

`results/{gemini,luna,maverick,qwen-235b}_bag.json` 保留原始字节。虽然文件写有
`spec_version=v2`，它们同时记录 `frozen=false`、`manifest_sha256=null` 和 8 次尝试，
且没有 lossless 标记；因此不符合冻结 v2 的 3 次总尝试和已审阅输入要求。
每份包含 20 题 / 10 视频，使用二值预测，不能验证 bag 的 yes/no/unknown 契约。
不要据此合并成已验收的 120 题结果，也不要事后补写 provenance 或精度标记。

当前 runner 对 bag/unknown 评分关闭，等待单独审阅的三值回答、聚合与指标协议。
新的 `--provisional` 只支持 base-4 诊断，输出独立的
[`base4-provisional-v1`](PROVISIONAL.md) 版本和显式未冻结标记；仍不能解除已审阅
validation/media/source-group 清单与新鲜兼容 lossless v2 证据的验收阻塞。

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
新 runner 原样保存用于判定的完整精度分数，并记录 `score_precision=lossless`；
调阈工具拒绝缺少该标记的旧 v2 输出及与分数不一致的基线判定。不得给旧存档补标记。
当前选阈按精确正确题数排序，选择最长连续最优网格区间的中点（并列取离 0.5 最近、
再取较低中点）；中点若不再最优，则回退到最近的最优网格点（并列取较低点）。
报告保存全部候选正确题数、所选区间与回退情况；历史阈值不是这一新规则的输出。
验证集必须与审阅 manifest 完全一致，且每类至少六题、同时有正负例；不足时本 CLI
直接拒绝运行，不把缺失题目当成小样本回退。测试集错误单列 attempted/scored/errors，
不进入分数分母；全错误事件仍展示覆盖数，准确率为 null。
历史 v1 文件保留原样，不能输入当前工具来生成声称有效的 v2 校准报告。

本仓库尚无留出集的已审阅 v2 manifest；`validation_split.json` 只是历史候选清单。
120 题 manifest 仍为 draft。必须先完成答案审阅和媒体/答案哈希清单，再对所有候选重跑；
不得把历史 JSON 改写为 v2，也不得据此发布 v2 比较。

用于调阈的 manifest 还必须为每个 base-4 问题保存 `human_review=user_reviewed` 和
经审阅的 `source_video_group`（同一原始视频或重叠片段使用同一组）。两个阶段都核验
manifest/samples 哈希、题号、标签、媒体及题干；验证/测试不能共享来源组，题干也必须一致。
来源分组必须在看分数前审阅；工具不验证人类审阅的真实性，也不证明模型训练集或开发集无交集。
训练来源未知时须披露；完整来源审阅、开发集隔离和新的 v2 实测仍是验收前提。

`select` 只读验证数据，写入不可覆盖的阈值 artifact 并打印其 SHA-256；`evaluate`
只接受该已冻结文件及预先保存的哈希，不重新选阈。两个阶段输出都写入新文件。
看过测试结果后不得改阈重试并继续声称该测试集未被使用。

下面命令中的路径须替换成已审阅的实际文件。输出写入忽略目录 `outputs/`，不覆盖历史存档：

```sh
python tools/event_benchmark/run_eval.py --candidate luna \
  --samples /path/to/validation/samples.json \
  --manifest /path/to/validation/manifest-reviewed.json \
  --media /path/to/validation/videos --out outputs/luna_validation_v2.json
python tools/event_benchmark/tune_threshold.py select \
  --val outputs/luna_validation_v2.json \
  --val-manifest /path/to/validation/manifest-reviewed.json \
  --out outputs/luna_thresholds_v2.json
# 保存上一步打印的 SHA-256；冻结阈值后才打开测试预测。
python tools/event_benchmark/run_eval.py --candidate luna \
  --samples /path/to/test/samples.json \
  --manifest /path/to/test/manifest-reviewed.json \
  --media /path/to/test/videos --out outputs/luna_test_v2.json
python tools/event_benchmark/tune_threshold.py evaluate \
  --thresholds outputs/luna_thresholds_v2.json \
  --thresholds-sha256 'REPLACE_WITH_SAVED_SHA256' \
  --test outputs/luna_test_v2.json \
  --test-manifest /path/to/test/manifest-reviewed.json \
  --out outputs/luna_threshold_report_v2.json
```
