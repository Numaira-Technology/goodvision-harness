# 测评方法学(冻结考卷 — 换考生不换考卷)

这是一张"控制变量"的考卷:**唯一允许变化的是被测模型(API)**。prompt、抽帧、阈值、判分对每个候选逐字节相同。
改动任何一项 = 改考卷,要提版本号并重跑全部候选。

**权威来源在仓库里**,不要在本 skill 里复制一份规则导致漂移:
- 规范:`goodvision-harness/bench/event_benchmark/SPEC.md`(考卷规范 v1)
- Runner:`goodvision-harness/bench/event_benchmark/run_eval.py`
- 历史结果:`goodvision-harness/bench/event_benchmark/RESULTS.md` + `results/`

下面只是要点速查,细节以 SPEC.md 为准。

## 冻结项速查

- **题目(prompt)**:用数据集 `samples.json` 每条自带的 `question` 字段,运行时去掉结尾 `Answer yes or no.`。语义题目对所有模型相同;只有"输出信封"按模型类固定。
- **计分事件**:当前 fight / smoking / fall / weapon。片级(整段有没有发生),不评 track_id / 选人。
- **模型三类**,输出格式类内固定、不跨类对齐(完整契约见 `model-classes.md`):
  - **language**(qwen=baseline / gemini,chat completions):prompt + 帧 → JSON 布尔 `{"answer": true/false}`,每帧 0/1。
  - **decision**:Luna decisions(题型 `noul`,0–1 概率,可留出集校准);VideoJev-4B(原生六题多标签 present/absent/insufficient,选项分数未校准、无阈值)。
  - **threshold**(CLIP / SigLIP2,同事 `tools/clip_benchmark`,未接入 run_eval):每帧 **margin = cos(正)−cos(反)**,`> 自选阈值` → yes;margin 不是概率,**不借用 0.5**,未调基线是 thr=0。
  - 真实 runner 位置、各自节奏(VLM 1fps · CLIP 每片 1 帧@0.8s · VideoJev 每窗 2 帧@0.8s)、数据同步与诚实标注:见 `model-classes.md`。
- **抽帧**:1 fps(`--interval 1.0`,冻结节奏),720px 宽,一帧一次独立调用。
  - 0.25 fps(`--interval 4`)只是生产历史上 VLM 吞吐跟不上的妥协,无硬性要求,测评与生产统一 1 fps。
  - `--dense` 是能力上限对照(≤14 帧),仅用于采样敏感度分析,不是官方口径。
- **判定**:每帧 0–1 分,片级 = 帧分数最大值("任一帧为正即为正"),`≥ gate 0.5` → yes。gate 0.5 统一、所有事件所有模型一致,绝不在测评集上调。
- **运行参数**:temperature 0,chat max_tokens 30,超时 90s,重试 3 次,`--workers 12`(对齐 12 路产品,只影响快慢不影响判分)。

## 数据集可以扩,规范不动

样本量可扩(50 → 120 → ~250),**扩样本不改考卷**。扩充注意:
- 冻结文件清单 + 每片 SHA-256(`--checksums`),看过分数后不回标。
- 给只有正例的事件补负例后,才能谈该类误报率 / 特异度。
- 调阈 / 调 prompt 只能用 held-out 子集,不碰主评测片。
- 新题型(如翻包 bag_knife / bag_alcohol_bottle,答案空间含 unknown)属于**改考卷**,要单独升版并先定 unknown 处理规则,不能塞进二值主表。

## 数据集文件在哪(随项目变,运行前确认)

当前合并集(**本机示例路径,媒体不在 repo 内,以实际 `samples.json` 为准**):`~/Desktop/benchmark_media_test/media_extra_50/samples.json`(120 题),视频分散在:
- `media50/media/media/`(原 50)
- `media_extra_50/新增基础视频_50段/`
- `media_extra_50/翻包视频_10段/`(翻包,非 base-4)

`run_eval.py --media` 接受多个目录,按文件名跨目录找。
