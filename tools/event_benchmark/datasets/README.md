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

## 为什么不直接改 Drive

原地改无法回答"现在这份和上周跑分用的那份一样吗"。版本化文件夹 + git manifest 让
每次结果都能钉到一个不可变、可验证的数据快照;出了分歧,校验和说了算。
