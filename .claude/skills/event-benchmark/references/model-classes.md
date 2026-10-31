# 三类模型 × 同一张 120 题考卷(跨贡献者数据同步)

这张考卷要同时容纳三类很不一样的模型。公平的定义**不是**"把它们塞进同一个 runner / 同一种输出",
而是:**同一批片子、同一份 GT、模型都看不到答案(answer-blind);每类按自己真实的生产工作点跑,
输出信封按类固定、不跨类对齐。** 下面是每类的真实实现、真实节奏、输出形状,以及怎么把别人的结果
同步进我们的报告 —— 以及**绝不能**谎称的东西。

> 现状(2026-10,务必照实说):PR#59 的 1fps `run_eval.py` **只接了 language / decision(luna)**,
> **没有接 CLIP / VideoJev 后端**。CLIP、VideoJev 跑在各自同事的 runner 里。所以"三类同时在一张表上"
> 目前是**把各自 runner 的输出汇总进同一报告格式**,不是"一个 runner 跑三类"。旧的 120 结果
> (含 10/29 VideoJev 那次)**不等于**新 runner 结果,别混报。同事这次**只发源码,没发数据/结果**。

---

## A. language 类 —— qwen / gemini(我们的 `run_eval.py`)

- **位置**:`tools/event_benchmark/run_eval.py`(PR#59 分支)。backend `language`。
- **模型**:`qwen/qwen3-vl-8b-instruct`(**baseline / 参照行**)、`google/gemini-2.5-flash`。走 OpenRouter chat。
- **输出信封**:每帧 → JSON 布尔 `{"answer": true/false}` → 1.0 / 0.0。硬 0/1,**没有可调阈值**。
- **节奏**:1 fps(`--interval 1.0`),720px 宽,一帧一次独立调用,`--workers 12`。
- **判定**:片级 = 帧分最大值 `≥ gate 0.5`。对 0/1 而言 gate 0.5 就是"任一帧为 true",无可调空间。
- **阈值**:**永不调**。见 `bench-threshold-tune`。

## B. decision 类 —— luna decisions / VideoJev(原生判定,但两者不同)

decision 类都是"直接给判定"的系统一模型,但**内部两种**,别混:

### B1. Luna decisions —— 连续概率,可留出集校准
- **位置**:`tools/event_benchmark/run_eval.py` backend `decision`;模型 `openai/gpt-6-luna-decisions`。
- **输出**:题型 `noul`,返回 **0–1 校准概率**。
- **节奏 / 判定**:同 language(1fps、片级取最大、gate 0.5 进主表)。
- **阈值**:0.5 只是借来的边界,对连续分数不公平 → 用 **留出集 per-class 校准**(`tune_threshold.py`),
  作为 `luna (calibrated)` **单独变体行**,主表仍 0.5。见 `bench-threshold-tune`。

### B2. VideoJev-4B —— 原生多标签判定,无标量阈值
- **位置(公共实现,已并 main)**:`tools/videojev_probe/`(入口 `python -m tools.videojev_probe`
  `plan|prepare|download|run`)、`tools/videojev_benchmark/`;`src/goodvision_harness/videojev_{service,runtime,questions,core,model,contract}.py`。
  指南:`docs/guides/videojev-candidate.md`。
- **注意边界**:10/29 实跑用的是 **PR56 的 dcea006 版本 + 本地"翻包"六题接线(私有播放器)**,
  **上面的公共链接 ≠ 那次完整冻结包**。私有翻包接线不在公共源码里,**别去碰同事的私有接线 / Vast 机器**。
- **模型**:`ZhuYuhan/VideoJev-4B`(revision 锁定),官方 Transformers/FlashAttention,**不是 vLLM**。BF16+FP32 头。
- **输出信封**:一窗一次 encode,**六题** present / absent / **insufficient-evidence** 分组问答,**多标签**
  (可多类同时为正)。**选项分数未校准,无额外置信阈值**,保留"不确定"选项;缺答/非法答案 → **整窗判失败**。
- **节奏**:**每窗 2 帧**(原始 JPEG,2–4 张,不拼图/不裁剪),**目标每 0.8 秒一次**;原生位置时间按名义 2.5 fps。
- **阈值**:**不调、保持 0 / 原生判定**。没有标量概率可切,也没有该类的验证样本 —— 别硬编一个阈值。
- **answer-blind**:参照用独立 Astra packet,**其判定绝不进模型输入**。
- **计时边界**:probe 的计时**排除**模型加载/packet 校验,**含**预处理/推理/答案校验(+ 开启时的 YOLO);
  **不是**源到成片(source-to-publication)时间,也**不是**12 路吞吐测试。报速度列时照此标注,别当产品链路。

## C. threshold 类 —— CLIP / SigLIP2(同事 `tools/clip_benchmark/`,PR#62 draft)

- **位置**:`tools/clip_benchmark/`(PR#62,draft,commit `47f96db`,仅 5 脚本,不改 main,未附数据/结果)。
  - `run.py`:验证 → 等固定阈值 → 测试,总入口。参数 `--config --output --thresholds --cache-dir`。
  - `runtime.py`:`ClipEncoder` 载模型 + 固定文本;`run_streams` 原速输入、12 路换片、单帧推理 + 计时。
  - `scoring.py`:`select_thresholds` 离线选阈、`evaluate` 算分类指标 + 延迟(**函数接口,无独立分析 CLI**)。
  - `base_timing.py`:按真实时钟送帧;`prepare.sh`:装依赖,**需先有 CUDA 版 Torch**。
- **模型**:`openai/clip-vit-large-patch14`、`google/siglip2-base-patch16-224`(revision 锁定)。
- **输出信封**:每类一对**固定正/反文本**;每帧 **margin = cos(图, 正文本) − cos(图, 反文本)**;
  片级 score = **该类有效 margin 的最大值**;`score > threshold` → yes,否则 no;**无有效 margin → unknown**
  (unknown 留在分母,**不填成 false**)。
- **关键**:margin 是 **cosine 差,不是概率** —— **绝不能套 gate 0.5**。它的"未调"基线是 **threshold=0**
  (`evaluate` 里的 `zero_threshold_baseline`),不是 0.5。
- **阈值**:`scoring.select_thresholds` **已经**是正确的留出集做法 —— **只在 validation 引用上**为
  base 四类(fight/smoking/fall/weapon)选 per-class 阈值;**bag 两类无验证数据 → 阈值保持 0,uncalibrated**。
  选阈规则 `max(accuracy, -FP, -FN, -abs(threshold), -threshold)`。**复用它,别用我们的 `tune_threshold.py` 重算**
  (那个读的是 0–1 概率的 `score`,和 margin 不是一个量纲)。见 `bench-threshold-tune` 的 threshold 类小节。
- **节奏**:**每片 1 帧**,**目标每 0.8 秒一次**,12 路 paced。延迟 = `last_frame_ingress → classification_ready`。

---

## 这张考卷"公平"在哪 —— 不变量 vs 允许按类不同

**三类共享、绝不能动的不变量:**
1. **同一批 120 片 + 同一份 GT。**
2. **answer-blind**:模型输入里永远没有标准答案(run_eval 只发 question;CLIP 参照只在离线打分用;
   VideoJev 的 Astra 判定绝不进模型输入)。
3. **计分事件**:二值主表只看 fight / smoking / fall / weapon(片级有没有发生)。
4. **扩样本不改考卷**;新题型(翻包 bag / 第 6 类、含 unknown)属于**改考卷**,单独升版,**不进二值主表**。

**允许按类不同、且这**不算**"改考卷"的三件事:**
- **输出信封**:JSON 0/1(language)/ noul 概率(luna)/ 原生多标签判定(VideoJev)/ cosine margin(CLIP)。
  类内固定、**不跨类对齐**。别把 decision 硬掰成 yes/no 去凑 chat 模型。
- **工作点节奏**:VLM 1fps · CLIP 每片 1 帧@0.8s · VideoJev 每窗 2 帧@0.8s。这是"每个模型在它真实生产工作点"
  的诚实轴 —— **在报告里写清各自节奏**,别假装一个统一 fps。
- **阈值处理**:language 固定 0.5(= 任一帧 true);luna 留出集校准(单独变体,主表仍 0.5);
  CLIP base 四类留出集选阈、bag 两类 uncalibrated(thr 0);VideoJev 原生判定、无阈值。

---

## 数据同步:把别人的结果汇进我们的报告(不重跑、不冒充)

同事只发了**源码**。要出一张三类同台的表时:

1. **各自 runner 产出各自的结果 JSON**(我们不替他们在 Vast 上重跑):
   - language / luna → 我们的 `run_eval.py`(+ 速度用 `latency_bench.py`)。
   - CLIP → 同事 `clip_benchmark` 的 `evaluate` 产出的 `summary.json` / `per-question.json`
     (含 per-event quality、`zero_threshold_baseline`、延迟分布)。
   - VideoJev → `videojev_probe`/`videojev_benchmark` 的输出(六题 present/absent/insufficient + 计时)。
2. **映射到报告五列**(见 `report-format.md`):子任务成功率取各自 per-event 的 x/N;总体准确率取各自 OVERALL;
   速度/延迟列务必标清是**哪种计时**(我们的墙钟 vs CLIP 的 last-frame→ready vs VideoJev 的 probe 计时),
   **不要把三种计时并成一列假装可比**,在列里注明口径或分行。
3. **诚实标注**(每一条都踩过坑):
   - 还没走过 PR#59 新 runner 的类,标 **"各自 runner,未经统一 1fps runner"**;别写成同一 runner 的结果。
   - 没有**本次新跑**的格子就留空 / 标 **"未跑"**,**绝不**拿旧 120(含 10/29 VideoJev)充当新结果,也不编数。
   - CLIP 的分是 margin,报阈值时写"留出集选阈(base 四类)/ bag 两类 uncalibrated thr=0",别提 0.5。
   - VideoJev / CLIP-bag 无验证样本的类:阈值标 uncalibrated,别报"校准后"数字。
   - 要把某类的分推进共享 Google Sheet,用 `bench-sheet-contribute`(自带、各贡献者通用),
     按 model+dataset+变体**单独一行**,别覆盖别人的行。
