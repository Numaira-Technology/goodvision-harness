# 数据集冻结协议

冻结 = **不可变 + 可验证**。要锁的不是"Drive 里那份别动"(没法证明),而是每个视频的 SHA-256
指纹 + `samples.json` 的整体指纹,记进 git。任何人任何时候都能校验手上的拷贝和冻结版是否逐字节一致。

## 分工

- **Drive**:只存大文件(视频),按版本放在**只读、永不原地改**的文件夹里。
- **git(本目录的 manifest)**:钉死内容。这才是权威冻结物。
- **run_eval.py**:v2 跑分前强制校验 manifest、samples.json 与每片 SHA，并将指纹写入结果。`--checksums` 仅保留为兼容参数。待审 draft manifest 不能用于正式跑分。

`--provisional` 仅提供 base-4 二值诊断，使用独立的
[`base4-provisional-v2`](../PROVISIONAL.md) 协议；可用 `--events` 筛选 base-4 事件。
此模式只记录输入指纹，不验证已审阅 manifest，不能与 `--manifest` 同用，也不能用于 v2
调阈或正式比较。bag 的 yes/no/unknown 任务仍不支持；不得把 unknown 改成 no 来绕过。

## manifest 内容

`manifest_<name>.json` 每条:`filename` · `sha256` · `event` · `question` · `proposed_answer` ·
`human_review`(审阅状态),外加顶层 `samples_json_sha256`(整份答案表的指纹)。

- `manifest_50.json` — 旧 50 片(fight12 / smoking12 / fall21 / weapon5)。
- `manifest_120.json` — 扩充 120 题 / 110 视频。合并的 PR #73 将状态记录为
  `human_reviewed`，全部 120 条记录为 `user_reviewed`；题干、答案及媒体/samples 指纹未变。
  这些标记本身不独立证明 owner 审阅、来源组隔离或新的 v2 实测；当前仍无 `source_video_group`。
  含两类新题型 `bag_knife` / `bag_alcohol_bottle`(翻包,答案空间带 unknown),属 v1.1 单独任务,不进 base-4 二值主表。

## 冻结一个新版本怎么做

1. **Drive**:用共享文件夹新建版本化子文件夹,**不覆盖旧的**,建完设为仅查看:
   - `goodvision-bench-v1-50/`、`goodvision-bench-v1-120-draft/`(定稿后再 add `…-120-final/`)
   - 父文件夹:https://drive.google.com/drive/folders/1zt7_9s0zNNY4aFBrzBV5HevBvnboWenq
2. **生成 manifest**:对该版本所有被 `samples.json` 引用的视频算 SHA-256,写入本目录,git 提交。
   (生成脚本见本次提交的命令;核心就是逐文件 `sha256` + `samples.json` 整体 `sha256`。)
3. **校验**:任何人重算 SHA 和 manifest 比对;对不上 = 内容被动过,结果作废。

## GT 定稿时

审阅只改**答案字段**,视频 SHA 不变。定稿后出一份 `-final` manifest,和 draft 版 diff
一眼能看出改了哪几条答案 —— 这就是"冻结 100 / 冻结 50"的可审计记录。

## 留出集（尚缺已审阅 v2 manifest）

`validation_split.json` 记录历史 24 片候选（每类 6，12 正 12 负），**不是已冻结的 v2 清单**。
它没有 samples.json 或媒体 SHA-256，不能替代 runner 所要求的 human_reviewed manifest。
原始 `luna_val24.json` 中 2 片文件缺失，只有 22 个分数。新增
`luna_val24_refilled.json` 保留这 22 行，另记录两行补分，共 24 行；它仍为 v1，
没有 lossless/manifest/media/runner/spec provenance，不能用于 v2 调阈。
存档阈值和测试集改善仅为历史 v1 算术结果；补分细节见 [结果说明](../RESULTS.md)。
`disjoint_assert` 是当时记录的文件名检查声明，当前工具不会将它视为内容不重叠的证明。

后续需要：审阅完整答案与媒体、生成哈希 manifest、完整验证集重跑，再在未参与选阈的测试集评估。
`tune_threshold.py` 强制 v2/v2.1 decision 概率、完整验证集、每类正负例、相同版本/模型/采样，以及
validation/test 的文件名和媒体 SHA-256 均不重叠；测试错误单列，主表保留 gate=0.5。
language 和未定义分数契约的 threshold backend 不能输入此工具。

调阈 manifest 沿用 runner 的 `status=human_reviewed`、`samples_json_sha256` 与 `files`
结构。每个 base-4 问题还须有 `human_review=user_reviewed` 和非空的
`source_video_group`：同一原始视频、重叠片段或复用镜头必须使用同一组，不能直接用
新文件名假定来源独立。此字段必须由实际来源审阅提供；本 PR 没有替现有文件生成或猜测分组。
`select --val-manifest ...` 与 `evaluate --test-manifest ...` 都会校验 manifest 的精确哈希
与完整题目集合；来源组和媒体均不得跨 validation/test，题干必须一致。

本 CLI 要求每个事件至少六个完整验证问题且两类都有；不足时拒绝选阈，仍可另行报告固定
0.5 基线。`select` 只读验证文件，输出独立、不可覆盖的阈值 artifact；`evaluate` 要求其
预存 SHA-256，并只应用已冻结阈值。旧 v2 的四位小数分数也不具备精确回放资格；新的
lossless 输出须由修正后的 runner 生成，不得给旧结果补写标记。详见 [运行示例](../RESULTS.md)。

## 为什么不直接改 Drive

原地改无法回答"现在这份和上周跑分用的那份一样吗"。版本化文件夹 + git manifest 让
每次结果都能钉到一个不可变、可验证的数据快照;出了分歧,校验和说了算。
