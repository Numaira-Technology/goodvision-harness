# 物件配方卡：设计说明（草案）

**状态：Ashley 已于 2026-11-23 确认第 5 节四点；第一期已实现。** 本文只讲结构，不声明任何识别效果。基于 PR #129（`claude/object-bench-recipe`）。

## 目标

换一个要识别的物件（刀、酒瓶，以后的打火机、充电宝……）时，只写一张**配方卡**（数据），不改代码。离线评测验证过的配方，就是工作台实际在跑的配方。

## 1. 现状：一个物件的配方散在五处

| 内容 | 现在在哪 |
|---|---|
| 工作台问法、门限 | `object_presence.PRESETS` |
| 评测问法（VideoJev 六题模板） | `tools/unified_benchmark/profiles/videojev.json`，#129 在 `object_bench_recipe.py` 里放了一份副本，并有防漂移测试 |
| CLIP / SigLIP2 文字对 | `profiles/clip.json`、`profiles/siglip2.json`，以及 `tools/clip_benchmark/protocol.py` |
| CLIP / SigLIP2 阈值 | 评测输出里的 `thresholds.json`，在私有输出目录，不进 git |
| 验证状态 | `object_presence.PRESET_STATUS`，Canvas JS 里另有一份 |

## 2. 配方卡长什么样

每个物件一张 JSON，放在 `src/goodvision_harness/object_cards/`，随代码发布：

```json
{
  "id": "bag_knife",
  "version": 1,
  "names": {"en": "knife in bag", "zh": "翻包刀具"},
  "scope": "包里露出、或正从包里拿出；不推断看不见的内容",
  "engine": "videojev",
  "recipe": {
    "kind": "native_choice",
    "source": {"profile": "videojev", "profile_sha256": "…", "question_id": "scene/bag_knife"},
    "window": {"frames": 2, "step_seconds": 1.0}
  },
  "decision": {"rule": "argmax"},
  "hard_negatives": ["尺子、笔等细长物", "刀的图片或印刷品"],
  "datasets": {
    "validation": {"name": "validation44", "manifest_sha256": "…"},
    "test": {"name": "current202", "manifest_sha256": "…"}
  },
  "status": {
    "state": "experimental",
    "note": "离线评测分数不代表实时准确率",
    "evidence": []
  },
  "live": {"supported": true}
}
```

各字段的意思：

- **engine**：`videojev` / `decision3` / `clip` / `siglip2` / `hosted_vlm` / `luna`。
- **recipe.kind** 跟着引擎的输出类型走：
  - `native_choice`：VideoJev 或 Decision 3 的三选一题。
  - `text_pair`：CLIP 或 SigLIP2 的「正例描述 / 反例描述」。
  - `vlm_boolean`：托管 VLM 回答是或否。
  - `probability`：Luna 输出概率。
- **recipe.source**：卡里**不另抄一份问法**，只指向评测配置文件里的某一题，并记下那个文件的哈希。工作台用的是随代码发布的副本，测试保证这份副本和评测文件一致（沿用 #129 的防漂移测试）。没有对应评测题的自定义问法，写在 `recipe.text` 里。
- **decision**：判定规则。
  - `argmax`：看哪项概率最高。
  - `gate`：最高且超过门限。
  - `margin`：CLIP 或 SigLIP2 的相似度差值超过阈值。阈值只从验证集选出、选定后冻结；卡里记下阈值本身，以及它来自哪个 `thresholds.json`（哈希）。
- **datasets**：只记数据集的名字和哈希。素材、标签、私有清单都不进仓库。
- **status.state**：`experimental` / `validated` / `retired`。
  - 改成 `validated` 必须在 `evidence` 里写明实时验收记录：方案版本、样本数、通过线、复核人、记录哈希。
  - 离线评测分数可以写进 `evidence`，但只有离线分数时，状态仍然是 `experimental`。
- **live.supported**：工作台现在能不能跑这张卡。比如 CLIP 翻包卡可以先建出来，供离线评测使用；在物体节点接上 CLIP 引擎之前它是 `false`，Canvas 不会显示它。

实时的保持参数（连续几次「没有」才清除）、每秒判断几次，是节点级设置，不放在卡里。

## 3. 谁读这张卡

- **工作台物体节点**：用户选预设时，节点按卡里的配方组装请求，按卡里的判定规则判断。卡的 ID 和版本写进每次判断的日志。
- **Canvas**：预设列表和「实验性 · 未验证」徽标都从服务端的卡目录读，不再在 JS 里另写一份。离线测试用的内置列表保留，并有测试保证它和服务端一致。
- **离线评测**：**不改冻结的评测流程**，评测文件仍然是唯一来源。卡指向评测文件，校验工具 `python -m tools.object_cards verify` 检查四件事：
  1. 卡记录的评测文件哈希和题目 ID 是否对得上；
  2. CLIP 文字对是否与评测文件一致；
  3. 如果提供了 `thresholds.json`，阈值和它的哈希是否一致；
  4. 状态改成 `validated` 时，验收记录是否齐全。

   这样两边读的是同一份配方，评测协议的版本和哈希也不受影响。

## 4. 第一期做什么

1. 卡的格式定义和校验，加上加载卡的注册表模块。
2. 把现有四个预设迁移成卡：`knife`、`alcohol_bottle`、`bag_knife`、`bag_alcohol_bottle`。迁移前后，节点发出的请求逐字节相同（加测试）；已保存的工作流图照常能用。
3. 新增两张只供离线使用的卡：CLIP 翻包刀、CLIP 翻包酒瓶。文字对指向 `profiles/clip.json`；阈值先留空，等有冻结的 `thresholds.json` 再填。
4. `tools/object_cards.py verify` 校验工具和相应测试。
5. Canvas 的预设列表和徽标改为从服务端的卡目录读。
6. 新建一张卡的步骤，写进本文第 6 节。

**不在第一期：**
- 物体节点接 CLIP、托管 VLM、Luna 引擎（按第 2 节的 `live.supported` 留好位置，另做）；
- 任何实时验收，任何 GPU 运行。

## 5. 已确认的选择（2026-11-23）

1. **问法用引用、不复制**：卡里指向评测文件，不另写一份；工作台用随代码发布的副本，由测试保证一致。
2. **冻结的评测流程不改**：卡和评测的「同一份配方」靠校验工具保证，而不是让评测程序改成读卡。
3. **改成 `validated` 的门槛**：必须有实时验收记录，只有离线分数不行。
4. **只供离线的卡**（比如 CLIP 翻包）也放进注册表，但 Canvas 里不显示，等对应引擎接进物体节点再开放。

## 5.1 第一期实现说明

- 卡：`src/goodvision_harness/object_cards/*.json`（6 张：4 张工作台预设，2 张只供离线的 CLIP 翻包卡）；加载和校验在 `object_cards/__init__.py`。
- `object_presence.PRESETS`、`PRESET_STATUS`、`RECIPES` 改由卡生成；`tests/fixtures/object_presets_before_cards.json` 是迁移前的快照，测试保证逐项相同。
- 哈希口径：VideoJev 题目卡对整个 `question_template` 取哈希（请求会发送整个模板）；CLIP 文字对只对该事件的一对文字取哈希。都用排序键、紧凑分隔的 JSON。
- 数据集不记哈希副本，只记钉住它们的常量（如 `tools.unified_benchmark.dataset202:PINNED_202_INPUTS`），`verify` 检查常量存在。
- Canvas：`GET /v1/workflows` 新增 `object_cards`，编辑器优先读它；没有时用内置的离线列表。
- 校验：`python -m tools.object_cards verify`；有冻结阈值时加 `--thresholds clip_bag_knife=路径/thresholds.json`。
- `validated` 验收证据不能只填字段名：`protocol`、`pass_line` 是非空文字，`samples`、`reviewers` 是正整数（样本数、复核人数），`record_sha256` 是 64 位小写十六进制哈希。这里只检查记录格式，不代替人工核实验收结果。
- 当前实时节点只支持两种卡：自定义文字 + `gate`（窗口由节点设置，卡不能另设窗口），或带 `recipe_id` 和窗口的评测引用 + `argmax`。原生评测引用的窗口还必须是 2 帧、间隔 1 秒，与冻结的评测适配器一致。

## 6. 新建一张卡（流程草案）

1. 收集素材、人工复核，分成验证集和测试集，两组视频不重叠；需要按时间计分时，标出物件的可见区间。
2. 写卡：名字、适用范围、引擎、问法或文字对、难反例、取几帧。
3. 跑离线验证集；需要阈值的引擎（CLIP 等）在这一步选阈值并冻结，写进卡。
4. 跑离线测试集，把分数写进 `evidence`，状态仍是 `experimental`。
5. 在实时画面上按验收方案做验收，通过后改成 `validated` 并附上记录。
6. `verify` 校验通过后提 PR；合并后，这张卡出现在 Canvas 的预设列表里。
