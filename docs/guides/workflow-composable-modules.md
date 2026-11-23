# Canvas 可组合模块：设计说明

**状态：第三版（2026-11-23），待 Ashley 确认第 5 节。** 第三版按 Ashley 的两条更正修改：VideoJev 和 Decision 3 是同级、各自独立的模块；新增自建 Qwen3-VL-8B 模块，并按团队评分表给理解层模块排序。第一期的代码已按第三版改好，实现见第 6 节，尚未推送。本文不声明任何识别效果。

目标：新项目接进来时，可以在 Canvas 上直接把需要的模块拉过来（VideoJev、Decision 3、托管 VLM、自建 VLM、CLIP、物体存在、总结……），按需加或不加 YOLO，快速搭出一个底子；每个模块在侧栏都有统一的调试入口。

## 总体分层

模块按层摆放，按端口类型连线。**每个识别模型都是独立的同级模块**：各自一个方块、各自的服务、健康检查和调试入口。不同识别模块之间共用传输代码没问题（Decision 3 的客户端就是继承 VideoJev 的），但在画布和模块契约里它们是平级的。

| 层 | 模块（引擎） | 输入 → 输出 | 现在的情况 |
|---|---|---|---|
| 输入 | 视频源；以后加录像文件、图片批 | — → `frames` | 只有实时视频源 |
| 感知（可选） | YOLO / Bbox、Contour（SAM 3.1）、人脸 | `frames` → `tracks` / `faces` | 已有，各自是独立节点 |
| 理解：识别模块 | VideoJev（System 1）、Decision 3、托管 VLM、自建 VLM（Qwen3-VL-8B）、CLIP（以后加 SigLIP2）、Luna | `frames`（+ 可选 `tracks`）→ `events` | 已有方块：VLM、System 1、CLIP；Decision 3 第一期加上；自建 VLM、SigLIP2、Luna 待做（见「理解层模块清单」） |
| 理解：任务模块 | 物体存在、定时总结 | `frames`（+ 可选 `tracks`）→ `objects` / `summary` | 物体存在由 VideoJev 回答（Decision 3 待加适配）；总结用托管 VLM |
| 决策（新） | 门限、持续 / 清除、多条件规则（如「有人 + 有刀 = 告警」）、身份 / 陌生人告警策略 | `events` / `objects` / `tracks` / `faces` → `alerts` | 门限和清除写在物体节点里（`object_presence.py`）；身份告警策略在场景结果里（`actions/alerts.py`）；没有独立节点 |
| 输出 | 看板、Webhook / 投递、归档 | 各类结果 → — | 一个输出节点，投递和归档走现有通道 |

几点取舍：

- **识别模块同级独立，任务模块另算。** VideoJev、Decision 3、托管 VLM、自建 VLM、CLIP 各是一个模块，不合成一个「识别器」节点再在里面选引擎。物体存在和总结是建在识别模型之上的**任务模块**：它们问的问题固定，在节点里选由哪个识别模型来回答（例如物体存在现在由 VideoJev 回答，以后加 Decision 3、Luna / Gemini）。
- **托管 VLM 和自建 VLM 分成两个模块。** 两者的健康检查、费用和 GPU 需求都不同。
- **YOLO 是可选上游。** 理解层节点都只要求 `frames`；接上 Bbox 的 `tracks` 时才用人框（物体节点裁剪，事件节点整帧 + 坐标），不接时行为和现在一样。
- **决策从识别里拆出来。** 识别节点只回答「看到了什么、置信多少」，要不要告警、持续多久、几个条件同时满足，放到决策节点里。这样同一份识别结果可以配不同项目的告警规则。

## 理解层模块清单

排序依据是团队评分表（Ashley 2026-11-23 认可）：离线事件基准，数据集 all-202，速度是并发 12 下的 P50 / P95 秒数，费用是每次运行的成本。表头被截掉了，各列按「总分、打架、吸烟、摔倒、武器、翻包刀具、翻包酒瓶」推断；翻包两类各约 10 道题，一题 10 分。**这些是离线基准分，不是工作台上的准确率**，不能拿来说工作台效果。

| 分组 | 模块 | 总分 | P50 / P95 | 费用 | 状态 |
|---|---|---|---|---|---|
| 本地 GPU，默认可用 | VideoJev-4B（System 1） | 83.7 | 0.97 / 1.08 s | 约 $0.03 | 已有，本地最好 |
| | CLIP L/14（神经网络） | 79.2 | 0.27 / 0.32 s | 约 $0.01 | 已有，适合做快速初筛 |
| | 自建 Qwen3-VL-8B（vLLM） | 80.2（与 API 相同） | 0.72 / 1.18 s | 约 $0.02 | 待做；翻包酒瓶 50 |
| 本地，可用但不默认 | Decision 3 | 77.2 | 1.17 / 1.21 s | — | 第一期加上，作为独立模块；各项都不如 VideoJev |
| | SigLIP2 B/16 | 72.3 | 0.18 / 0.21 s | 最低 | 待做，复用 CLIP 服务；最弱（翻包刀具 40），优先级最低 |
| API（托管 VLM 模块里的模型选项） | Gemini 2.5 Flash（关思考） | 85.6（最高） | 0.81 / 1.06 s | $0.12 | 作为托管 VLM 的模型选择 |
| | Luna（decision） | 84.7 | 0.40 / 0.76 s | — | 待做；输出的是概率，需要单独的适配器 |
| | Qwen3-VL-8B API | 80.2 | — | — | 参考基线 |

**不加：** Astra（所有档位，慢，$14–23）、Hydrogen low（P95 7.2 s，$6.1）、Qwen235B（76.2）、Llama4 Maverick（75.2）、Gemini 2.5 / 3.8 Flash low（准确率差不多，慢 2–3 倍）、Gemini Decision 1（和 Luna 差不多但更贵）。

**按输出形式分组**，这样第二期的决策节点可以用同一种方式接收它们：

| 输出形式 | 模块 |
|---|---|
| 原生多帧选项概率 | VideoJev、Decision 3 |
| 图文向量相似度差，用验证集调的门限 | CLIP、SigLIP2 |
| 生成式 JSON 布尔值 | 托管 VLM、自建 Qwen 8B |
| 决策概率 | Luna |

模块契约里每个识别模块都标了自己的输出形式（`output_contracts`）。

**自建 Qwen3-VL-8B 现状：** 目前只有基准用的工具：`tools/qwen8b/serve.py`（transformers）、`tools/qwen8b/serve_vllm.py`（vLLM），权重和环境固定在 R2（`tools/models.yaml` 的 `qwen8b`、`qwen8b-vllm`，Qwen/Qwen3-VL-8B-Instruct 修订 0c351dd），启动脚本 `tools/start_qwen8b_r2.py`、`tools/start_qwen8b_vllm.py`。都没有接到工作台。画布上的「VLM 语义识别」可以把模型服务设成 `local`（OpenAI 兼容接口，默认 `http://127.0.0.1:8000/v1/chat/completions`），`compose.vlm-server.yaml` 也能起 `vllm/vllm-openai:v0.17.1`，但没有证据表明 Qwen3-VL-8B 在工作台上端到端跑通过。所以本设计不声称它可用；要在工作台上验证需要 GPU，并且需要 Ashley 同意。

**物体存在的识别模型：** 以后应能选 Luna / Gemini（评分表里翻包酒瓶 90）。但刀具 / 酒瓶预设仍是实验性的：事件基准里的翻包刀具分数用的是另一套协议，不能当作物体存在预设的证据。

**建议顺序：** Decision 3（第一期）→ SigLIP2（复用 CLIP 服务）→ 自建 Qwen 8B → Luna。

## 0. 现状

- 节点类型和端口写死在两处：`workflows.py` 的 `NODE_PORTS` 和 `assets/workflow_canvas.js` 的 `definitions`。除视频源外，所有分析节点（人脸、追踪、事件、总结、物体）只有一个输入 `frames`，只能各自从视频源拉一条并行支路，彼此不能喂数据。
- 连线规则是「输出端口名 = 输入端口名」。`compile_graph` 只收集**直接连到输出节点**的模块（`consumed`），中间模块的输出没有去处。
- 运行时不是通用数据流引擎：编译出的 plan 被翻译成原生流水线的配置（Bbox → `rich_labeling`，System 1 → 独立 VLM 子进程），总结和物体节点则是 Harness 内的定时器（`SummaryManager`、`ObjectPresenceManager`）。
- Bbox（YOLO `yolo26-s`）只检测人（`classes=[0]`），输出的是带 `track_id` 的归一化人框，存于 `RichLabelCoordinator`，可以按源帧时间 `select()` 取出对应帧的框。
- VideoJev 的事件提示词里已经有一段「如果提供了人员位置（0..1 归一化、逐帧、ID 只是本地追踪编号）」的说明（`videojev_contract._PERSON_GUIDANCE`），但目前没有任何请求真的带上人员位置。
- 调试面不统一：物体节点有逐 tick 日志（`ObjectPresenceManager.tick_log()`，含计划时刻、是否发送、跳过原因、帧时间、耗时、错误）和 `GET /v1/workflow/objects`；总结有历史；追踪和事件只有结果快照；测试模式（`dry_run`）是全局开关，返回模拟结果。

## 1. 模块契约

把「一个模块是什么」从两份手写表收成一份声明，放在 `workflow_modules.py`，后端校验和前端画布都从它读（前端通过 `/v1/workflows` 返回的目录拿到，不再自己写一份）。

每个模块（按 节点类型 + 引擎）声明：

| 字段 | 含义 |
|---|---|
| `inputs` | 输入端口列表，每个有 `type` 和 `required`。`frames` 仍是必需；新增的上游输入一律是**可选**的。 |
| `accepts` | 可选输入允许来自哪些上游引擎，例如物体节点的 `tracks` 只接 `tracking/bbox`。 |
| `outputs` | 输出端口及类型。 |
| `health` | 就绪检查：`ready / starting / unavailable / not_configured / simulated` + 原因文字。复用现有的本地模块状态、Bbox 探测、VideoJev `/health` 探测。 |
| `debug` | 是否有逐次调试记录（见第 3 节）。 |

端口类型沿用现有名字：`frames / tracks / faces / events / summary / objects`。Bbox 的 `tracks` 就是「人框 + track_id」，不另起 `boxes` 类型，免得旧图失效。Contour 的 `tracks` 是轮廓锚点，没有框，所以 `accepts` 不放行它。

**校验和编译的改动：**
- 连线仍要求类型一致；另外检查目标端口的 `accepts`。
- `compile_graph` 的「被消费的模块」改为「输出节点的所有上游」（传递闭包）。这样 Bbox → 物体 → 输出 时，Bbox 即使没直接接到输出也会启动。
- 每个可选输入接上后，在 plan 里写成显式字段（如 `objects.focus = "person_crop"`）。没接时 plan 和今天逐字节一样。

**向后兼容：** 只给节点加可选输入，不改已有端口名、不改文档 `schema_version`。已保存的图、已应用的组快照全部原样通过校验，编译结果不变；会加测试锁住这一点。

### 1.1 第一个实例：YOLO 人框 → VideoJev

VideoJev 在 Harness 里有两个入口，处理方式不同：

**(a) 物体存在节点（第一期做这个）。** 加一个可选输入 `tracks`，只接 Bbox。接上后：
- 每个 tick 取窗口内各帧对应的人框（`RichLabelCoordinator.select`，按源帧时间对齐），把这些框取并集，四周外扩约 25%，再限制最小边长和长宽比，得到**一个**裁剪框；
- 窗口内 2–4 帧**用同一个裁剪框**裁，保证 VideoJev 看到的仍是时间连续的同一视野；
- 窗口里没有人框时，退回整帧（刀放在桌上、人没被检出时仍能看到），并在 tick 日志里记 `focus: "full_frame_no_person"`；
- tick 日志新增：用到的框数、裁剪框、对齐方式（exact / held / interpolated / none）。

选**裁剪**而不是「整帧 + 框坐标文字」的理由：物体节点问的是小件手持物（刀），请求前整帧会被缩到 720 像素长边，人手部位的像素很少；裁到人附近能把同样的 720 像素花在手上。框坐标写进文字不会增加像素，也会让每次请求的问题模板都不同，失去服务端模板缓存。

**(b) 事件节点 System 1（第二期，本 PR 只写设计不实现）。** 事件（打架、吸烟、摔倒等）依赖多人之间和场景的关系，裁剪会切掉上下文；按人分别裁剪会让请求数随人数增长。所以事件走「**整帧 + 人框坐标作为证据元数据**」，正好对应已有的 `_PERSON_GUIDANCE` 格式。不放在第一期的原因：它会改动冻结基准用的提示词，而且事件请求在独立子进程里，需要把人框沿着任务管道传进去，改动面更大。

**准确率：** 以上两种方式都只是「能接、能跑、能看日志」。裁剪对刀具召回是否有帮助、元数据对事件是否有帮助，都必须在真实视频上对比「接 / 不接人框」后才能说。物体节点按 `docs/validation/object-presence-knife-protocol.md` 的口径另做试点；事件按 event-benchmark 冻结流程，并用同口径基线对比。两者都要 GPU，开机前需 Ashley 明确同意。在这之前文档和 UI 不写「提升效果」。

### 1.2 同级识别模块和任务模块

- **识别模块**（Contour、Bbox、VLM、System 1、Decision 3、CLIP，以后加自建 VLM、SigLIP2、Luna）在模块库里各有方块，各有服务地址、健康检查和调试入口。模块契约的 `peers` 列出它们：层、模型 ID、本地端口、输出形式。换识别模型就是换一个方块，不在同一个节点里切换。
- **Decision 3 作为独立模块：** 模型 `vllm-sr/d3-nano`，服务端是 `videojev_service --runtime decision3`，本地端口 8099（System 1 是 8097）。客户端复用 VideoJev 的传输代码（`Decision3Inspector` 继承 `VideoJevInspector`），这只是代码复用，在画布上两者是平级的方块。
- **一路摄像头同时只跑一个事件识别模块。** 画布上可以同时放 System 1 和 Decision 3 两个方块，但 Apply 只允许启用其中一个，因为目前每路摄像头只有一条原生事件通道。两个模块并行跑同一路视频（例如做对比或投票）留给决策节点那一期再考虑。
- **任务模块**（物体存在、总结）在节点里选回答它的识别模型；有多个可选时节点里出现下拉，切换时连线保留、服务地址清空。物体存在目前只有 System 1。Decision 3 要先有物体请求和响应的适配：它只接受 2 帧、2 FPS 的窗口，返回 `raw.answers` 映射，而现有物体解码器只认 VideoJev 的列表。这个适配放到后面，不能靠调门限绕过（评审 2026-11-23）。

## 2. 基础盘模板

**模板**就是一张不带部署信息的图：节点、连线、坐标、模块配置；去掉视频源的 `camera_id`、各服务地址（`service_url`、`cloud_service_url`），保留引擎和执行方式（本地 / 服务器 / 云）。

- **内置模板按项目类型提供**，写在代码里（`workflow_templates.py`），随版本发布：
  - **人员 + 事件**：视频源 → Bbox → 输出；视频源 → System 1 事件识别 → 输出。可以换成同级的 Decision 3、VLM 或 CLIP 方块；人脸可选。
  - **物品检查 / 翻包**：视频源 → Bbox → 物体存在（接人框裁剪）→ 输出。物体预设是实验性的，未经验证。
  - **纯总结巡检**：视频源 → 总结 → 输出。
  - 第二期有了决策节点后，前两个模板改成经过决策节点再到输出（例如「有人 + 有刀 = 告警」）。
- 第一版里已写好的「VideoJev + YOLO 基础盘」并入「物品检查 / 翻包」，不再单列。
- **自定义模板**：给客户配好的图可以在画布上「另存为模板」，存在数据目录的 `workflow-templates.json`，单独文件、带上限（64 个、单个 2 MiB），**不进**工作流文档，因此不改 `schema_version`，旧版本读文档不受影响。
- **视频源是可重新绑定的占位**：模板里的视频源不带通道号，多路模板保存时按占位校验。从模板新建时，画布给每个视频源分配一路已启用、已停止分析且未分配给其他工作流的通道。单路没有合适通道时可先留空；多路通道不够就提示，不创建。
- **从模板新建**：生成新的工作流，名字可改。新建后和手工搭的图一样，需要自己选摄像头、填服务地址、Apply。
- 模板保存和实例化都走现有的 `validate_graph`，不合法的模板存不进去。

接口：`GET /v1/workflow-templates`（内置 + 自定义）、`POST /v1/workflow-templates`（另存）、`DELETE /v1/workflow-templates/{id}`（只能删自定义的）。模板不在工作流文档里，所以不走 `/v1/workflows/command` 的修订号；「从模板新建」由画布用模板的图调用现有的 `create_workflow`，图内节点 ID 只需在本图内唯一，沿用模板里的即可。

## 3. 统一调试入口

每个分析节点的侧栏加一个「调试」分区，内容一致：

1. **状态**：契约里的 `health`（就绪 / 启动中 / 不可用 / 未配置 / 模拟）+ 原因；
2. **最近 N 次**（默认 20，最多 200）：时间、输入摘要（帧序号、帧时间、上游框数 / 裁剪框）、输出摘要、耗时、错误原因；
3. **计数**：成功、失败、跳过（按原因）；
4. **测试模式**：显示当前是否模拟，并提供现有的「改用真实分析」按钮（仍需先停止分析）。测试模式是全局的，不做按节点切换，避免同一路摄像头一半真一半假。

**后端：** 新增 `GET /v1/workflow/nodes/{node_id}/debug?limit=`，按摄像头 worker 返回，fleet 端按摄像头转发，和 `/v1/workflow/objects` 一样。记录格式统一为：

```json
{"at": 0.0, "source_frame_sequence": 0, "frame_times": [], "input": {}, "output": {},
 "latency_seconds": 0.0, "error": null, "reason": "valid", "synthetic": false}
```

各节点的数据来源：

| 节点 | 第一期 | 来源 |
|---|---|---|
| 物体存在 | 完整 | 现有 tick 日志，加上人框字段 |
| 总结 | 完整 | 现有历史记录 |
| Bbox | 状态 + 最近快照 | `RichLabelCoordinator.history()/stats()` |
| System 1 / VLM 事件 | 状态 + 最近结果 | 现有事件结果；逐请求日志在第二期随 (b) 一起补 |
| 人脸、Contour | 状态 | 后续再补 |

## 4. 分期

**第一期（本 PR）：**
1. 模块契约：端口、必需 / 可选、`accepts`、调试类型；识别模块的同级清单（`peers`）和输出形式；任务模块标记。后端校验、编译，前端改读目录。
2. YOLO 作为可选上游：物体存在的 `tracks` 输入和人框裁剪，测试模式下用模拟人框。
3. Decision 3 作为独立的事件识别模块：有自己的方块、本地服务（8099）、健康检查和调试入口。物体存在暂不接 Decision 3（需要适配，见 1.2）。
4. 统一调试入口：接口加侧栏分区。物体、总结有完整记录，其他节点显示状态。
5. 模板存取，三个按项目类型的内置模板。

以上第一期的代码都已写好（见第 6 节）。

**第二期：**
- 独立的决策节点，输出 `alerts`，输出节点加 `alerts` 输入。规则支持：单个结果过门限、持续 N 次才触发、连续 M 次没有才清除、多条件同时满足（「有人 + 有刀」）、身份 / 陌生人告警策略（把 `actions/alerts.py` 的逻辑搬成节点配置）。按上面四种输出形式接收识别结果。
- 兼容：物体存在没接决策节点时，保留它现在内部的门限和清除；接了以后，持续和清除交给决策节点。事件模块现有的确认流程不动，决策节点先只读它的结果。
- 事件模块接人框（整帧 + 坐标），以及事件的逐请求调试日志。

**之后按顺序加识别模块：** SigLIP2（复用 CLIP 服务）→ 自建 Qwen3-VL-8B（vLLM，独立于托管 VLM 的模块）→ Luna（概率输出的适配器）。物体存在增加 Luna / Gemini 作为可选的识别模型。每加一个，都要先在工作台上用真实视频验证（需要 GPU 和 Ashley 同意），才能说它可用。

**再往后：** 输出层拆成看板、Webhook、归档三个节点；输入层加录像文件和图片批。

**不在范围内：** 通用数据流引擎（模块间任意传数据）、按节点的测试模式、任何 GPU 验证。

## 5. 需要确认的事

第一版的三项（YOLO → 物体先做裁剪、本 PR 接在 #121 后面、测试模式保持全局）已确认，不变。第三版要确认：

1. **分层和取舍：** 五层；理解层的识别模块（VideoJev、Decision 3、托管 VLM、自建 VLM、CLIP……）各自独立、同级；物体存在和总结是任务模块，在节点里选由哪个识别模型来回答。可以吗？
2. **第一期范围：** 第 4 节列的五项，代码已写好，一起提一个 PR。可以吗？
3. **第二期决策节点：** 规则范围按第 4 节；按四种输出形式接收结果；物体存在接了决策节点后，持续和清除交给决策节点，不接时保持现状。可以吗？
4. **之后的识别模块顺序：** SigLIP2 → 自建 Qwen 8B → Luna；不加「不加」清单里的模型。可以吗？
5. **准确率：** 不变。任何新接法（人框裁剪、Decision 3 回答物体问题、事件接人框、新的识别模块）都要在真实视频上和不接的情况对比之后才能说效果；评分表是离线基准，不代表工作台结果；开 GPU 需要 Ashley 同意。

## 6. 第一期实现情况（本地分支，尚未推送）

| 部分 | 位置 |
|---|---|
| 模块契约（端口、必需 / 可选、`accepts`、调试类型）和 `module_catalog()` | `workflow_modules.py`；`/v1/workflows` 返回 `modules` |
| 校验：`accepts` 检查；Apply 要求每个启用的模块接上 `frames` | `workflows.validate_graph` |
| 编译：输出节点的所有上游都运行；接上人框时 plan 多出 `objects.focus` 和 `objects.tracks_node_id` | `workflows.compile_graph` |
| 人框裁剪：`person_crop_box()`（并集、外扩 25%、最小边 0.35、移回画面内）、窗口共用一个裁剪框、无人退回整帧、框查询出错不影响判断 | `object_presence.py` |
| 人框来源：真实运行按源帧时间从 Bbox 取框；测试模式用一个固定的模拟人框 | `workflow_runtime._start_workflow_objects` |
| 模板：内置「人员 + 事件」「物品检查 / 翻包」「纯总结巡检」、另存 / 删除、去掉摄像头和服务地址 | `workflow_templates.py`、`workflow_routes.py` |
| Decision 3 独立模块：模型 `vllm-sr/d3-nano`，本地端口 8099，复用 System 1 的 Python 环境；原生事件通道和单帧测试使用 `Decision3Inspector`；目录 `peers` 列出所有识别模块及其输出形式 | `workflow_modules.py`、`module_service.py`、`module_services.py`、`workflow_runtime.py` |
| 画布：模块库新增 Decision 3 方块（和 System 1 平级）；任务模块有多个识别模型可选时才出现「识别模型」下拉（物体存在目前只有 System 1）；从模板新建时只绑定可用的独立视频通道 | `assets/workflow_canvas.js` |
| 按摄像头编译：一个模块只有在必需的 `frames` 来自本摄像头时才运行，可选的人框输入不能代替它 | `workflows.compile_graph` |
| 调试接口：`GET /v1/workflow/nodes/{node_id}/debug?limit=`（1–200，默认 20），fleet 按摄像头转发 | `workflow_runtime.workflow_node_debug`、`service.py`、`fleet_outputs.py` |
| 画布：端口表改读服务端目录；物体节点的 `tracks` 输入只列 Bbox；「从模板新建…」「另存为模板」；模块侧栏「调试」分区 | `assets/workflow_canvas.js` |
| 测试 | `tests/test_workflow_composable.py`、`tests/workflow_canvas.cjs` |

调试记录格式各节点一致。物体节点每条对应一个计划 tick（含跳过的 tick 和原因），`input.focus` 记录用了几个框、裁剪框和对齐方式；总结节点每条对应一次请求；Bbox 每条对应一个检测快照（人数、track_id）；事件节点目前只有各事件的最新结果，没有耗时。人脸、Contour 只显示状态。

**没有做的：** 决策节点（第二期）；物体存在的 Decision 3 适配；自建 Qwen 8B、SigLIP2、Luna 模块；System 1 / Decision 3 事件接人框元数据；人脸、Contour 的逐次记录；`compose.modules.yaml` 里的 Decision 3 服务。人框裁剪还没在真实视频上跑过，没有任何准确率数据；Decision 3 本地服务也没有在本次改动后实际启动验证过（需要 GPU）。
