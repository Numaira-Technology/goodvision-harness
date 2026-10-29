# 事件识别测评集 — 固定规则（考卷规范 v1）

这是一张"换考生、不换考卷"的测评规范。**唯一允许变化的是被测模型(`--candidate` / `--model` + `--backend`)**；
下列所有项对每个候选模型**逐字节相同、运行即冻结**。改动本规范需提版本号并重跑全部候选。

数据集本身(视频清单、参考答案、来源、许可)由 `media50/` 的 `samples.json` / `人工审阅.md` 维护，本规范只约定**怎么考、怎么判**。
样本量当前 50，后续可扩到 ~250；**扩充不改变本规范**。

---

## 1. 题目（prompt）

- **唯一题源 = `samples.json` 每条的 `question` 字段**（不使用 `monitor.py` 的 `_LUNA_DESC` 产品定义）。每类一句，现为：
  - fight: `Does any fighting occur in this video? Answer yes or no.`
  - smoking: `Does this video show a person smoking? Answer yes or no.`
  - fall: `Does a person fall in this video? Answer yes or no.`
  - weapon: `Does this video show a visible weapon, or a visible object being used to strike, stab, or explicitly threaten a person? Answer yes or no.`
- 运行时取 `question_core` = `question` 去掉结尾的 `Answer yes or no.`（大小写不敏感、去首尾空白）。
- **语义题目对所有模型相同**；因 API 形态不同，只有"输出信封"按家族固定（见 §3）。严禁按模型改写题意。

## 2. 计分范围

- 计分事件：**fight / smoking / fall / weapon**。PPE、乱扔垃圾等即使模型会答也**不计分**。
- **片级（clip-level）识别**：只判"整段里有没有发生"。**不评选人 / track_id**（`person_ground_truth` 全为空，无依据）。
- 负例覆盖：本 50 片仅 **fight、fall 有负例**；smoking、weapon 全阳性，**不计误报率/特异度**，报告须注明。

## 3. 模型分三类，输出格式"类内固定、不跨类对齐"

**语义题目（§1）对所有模型相同**；输出封装**只在同一类模型内固定一致**，不同类之间不强求一样。
三类：

- **language**（语言/VLM，qwen / gemini，`POST /api/v1/chat/completions`）
  - 文本：`{question_core} Answer only with compact JSON: {"answer": true} or {"answer": false}.`
  - 外加该帧图像。解析 `content` 里的 JSON 布尔 `answer`。每帧硬输出 0/1。
- **decision**（system-one 决策类，Luna decisions；videojev 归此类，接入时沿用本格式，`POST /api/alpha/decisions`）
  - 题型 **noul**（固定，不用 choice / score）。
  - `state = ["Surveillance video frame.", {"type":"image_url","image_url":{"url": <帧 data URI>}}]`
  - `questions = {"answer": {"type":"noul", "instructions": question_core,
        "criteria": {"true": "Yes — "+question_core, "false": "No — "+question_core}}}`
  - 读 `answers["answer"]["noul"]`，得 0–1 概率。criteria 为**机械重述题目**，不含人工细化。
- **threshold**（阈值类，CLIP / YOLO 等；尚未接入）
  - 输出为每帧一个 score/logit，按**该模型自己固定的阈值**切 0/1；阈值随模型、写进本规范后冻结，不跨类借用 0.5。

## 4. 抽帧（所有模型逐字节相同）

- 用 `ffprobe` 取时长 `dur`。
- **采样 = 1 fps**（`--interval 1.0`，默认）：帧数 `n = max(1, round(dur/1.0))`，一秒一帧。
  - 取帧时刻：均匀居中 `t_i = dur * (i+0.5)/n`，i=0..n-1。
  - 备选口径（不改默认、仅对比用）：`--interval 4`＝0.25 fps（产品当前节奏，短片≈1 帧）；`--dense`＝无封顶的能力上限（`max(3,min(14,int(dur)))`）。
  - ⏳ 待定：生产当前是 0.25 fps（4s/帧），是否统一到 1 fps 待 Frank 确认；确认后本行同步。
- 分辨率：**宽 720px，高自适应为偶数**（`ffmpeg -vf scale=720:-2`），JPEG。
- 送法：**一帧一次独立调用**，帧之间无状态。

## 5. 判定规则

- 每帧得一个 **0–1 分数**：luna=noul 概率；chat=true→1.0 / false→0.0。
- 片级分数 = **帧分数的最大值**（"任一帧为正即为正"）。
- 阈值 **gate = 0.5 统一，所有事件、所有模型一致，且绝不在测评集上调**。
  - chat 为硬 0/1，gate 0.5 等价于"任一帧 true"。
  - 任何按事件调阈（如 fight@0.3）只能作为 **post-hoc 附注**，须：①保留 0.5 为主基线；②标为"同集调参"；③用**未参与选阈的留出片**另行验证，不得计入主表。

## 6. 运行参数（全部写死）

- `temperature = 0`，chat `max_tokens = 30`，请求超时 `90s`，失败**重试 3 次**。
- 模型 ID 钉死、记录于每次输出：`qwen/qwen3-vl-8b-instruct`、`google/gemini-2.5-flash`、`openai/gpt-6-luna-decisions`。
- provider = OpenRouter 默认路由；输出记录运行日期与 gate/帧规则。
- 并发只影响跑得快慢、不影响判分；**准确率与并发度无关**（延迟另有专门脚本按 12 路压测）。
  默认 `--workers 12`，对齐 12 路产品，对齐延迟测试时无需换算。

## 7. 判读与评分口径

- 预测：片级分数 `≥ gate` → `yes`，否则 `no`。
- 报告：每类 **TP / FN / TN / FP**、准确率 `(TP+TN)/N`，以及总体准确率。
- 无负例的类（smoking / weapon）：误报率/特异度标 `—（无负例）`，不得外推。
- 某片 3 次重试仍失败 → 计为 **error**，**从该类分母剔除**，但在输出中单列 error 数(可见、不掩盖)。

## 8. 数据治理（为扩到 ~250）

- 冻结**文件清单 + 每片 SHA-256**；看过分数后**不得回标**。
- 任何调阈/调 prompt 只能用 **held-out 子集**，不碰主评测片。
- 扩充时：同一视频可补多事件答案；**给 smoking / weapon 补负例**后才可谈这两类误报。
- 片级识别不产出 track_id；"谁发生了什么"需另建带 person 标注的集子。

## 9. 运行

```sh
# 换考生只改这一个参数：
python tools/event_benchmark/run_eval.py --candidate luna
python tools/event_benchmark/run_eval.py --candidate qwen
python tools/event_benchmark/run_eval.py --candidate gemini
# 或自定义： --model <id> --backend <language|decision>
# 对比口径： 默认 1fps；--interval 4（产品0.25fps）；--dense（能力上限）
```

默认读 `/Users/ashley/Downloads/media50/samples.json` 与同目录 `media/media/`，
API key 取自仓库 `.env` 的 `OPENROUTER_API_KEY`。每次运行把配置 + 逐片结果 + 汇总写入 `--out` 的 JSON。
