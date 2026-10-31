# 数据集冻结协议

冻结 = **不可变 + 可验证**。要锁的不是"Drive 里那份别动"(没法证明),而是每个视频的 SHA-256
指纹 + `samples.json` 的整体指纹,记进 git。任何人任何时候都能校验手上的拷贝和冻结版是否逐字节一致。

## 分工

- **Drive**:只存大文件(视频),按版本放在**只读、永不原地改**的文件夹里。
- **git(本目录的 manifest)**:钉死内容。这才是权威冻结物。
- **run_eval.py**:v2 跑分前强制校验 manifest、samples.json 与每片 SHA，并将指纹写入结果。`--checksums` 仅保留为兼容参数。待审 draft manifest 不能用于正式跑分。

## manifest 内容

`manifest_<name>.json` 每条:`filename` · `sha256` · `event` · `question` · `proposed_answer` ·
`human_review`(审阅状态),外加顶层 `samples_json_sha256`(整份答案表的指纹)。

- `manifest_50.json` — 旧 50 片(fight12 / smoking12 / fall21 / weapon5)。
- `manifest_120.json` — 扩充 120 题 / 110 视频。注意 `status=draft_pending_user_review`:66 条 GT 还是草稿。
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

## 留出集(threshold 调阈专用,不进主评测)

`validation_split.json` 冻结了一份**与 v1-50 / v1-100 测试集零重叠**的留出集(当前 24 片,fight/smoking/fall/weapon 各 6,12 正 12 负,1fps),**专用于 decision / threshold 类模型的 per-class 阈值标定**(SPEC §5/§8)。

- 协议:在**本留出集**上按事件选阈 → 冻结 → 在测试集上评估。**绝不在测试集上选阈,主表永远保留 gate 0.5。**
- 校准结果是**单独标注的变体**(如 `luna (calibrated)`),与 0.5 基线并列,不替换主表。
- 已固化的 v1 阈值:fight 0.29 · smoking 0.53 · fall 0.58 · weapon 0.46(把 luna 从 85%→92% @100 题、84%→94% @50 片,提升几乎全来自 fall 误报被砍)。
- language 类(qwen / gemini)输出硬 0/1,阈值无意义,**永不调阈**。
- 工具:`tune_threshold.py`(纯离线,读 `run_eval.py` 的 `score`,不调 API)。详见 `bench-threshold-tune` skill。
- `disjoint_assert` 字段记录了与各测试集的重叠数(必须为 0);扩充留出集时重跑核对。

## 为什么不直接改 Drive

原地改无法回答"现在这份和上周跑分用的那份一样吗"。版本化文件夹 + git manifest 让
每次结果都能钉到一个不可变、可验证的数据快照;出了分歧,校验和说了算。
