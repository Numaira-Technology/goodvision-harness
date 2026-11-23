# Canvas 可组合模块：设计说明

**状态：2026-11-23 Ashley 确认第 5 节三项，第一期已实现（见第 6 节）。** 本文不声明任何识别效果。

目标：新客户项目可以在 Canvas 上把现有模块（例如 VideoJev + YOLO）拖在一起、连好线，作为起步的「基础盘」；每个模块在侧栏都有统一的调试入口。

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

## 2. 基础盘模板

**模板**就是一张不带部署信息的图：节点、连线、坐标、模块配置；去掉视频源的 `camera_id`、各服务地址（`service_url`、`cloud_service_url`），保留引擎和执行方式（本地 / 服务器 / 云）。

- **内置模板**写在代码里（`workflow_templates.py`），随版本发布。第一期提供一个「VideoJev + YOLO 基础盘」：视频源 → Bbox → 物体存在（tracks 接 Bbox）→ 输出，同时 视频源 → System 1 事件 → 输出、Bbox → 输出。物体预设沿用现有的「实验性，未验证」标记。
- **自定义模板**：画布上「另存为模板」，存在数据目录的 `workflow-templates.json`，单独文件、带上限（64 个、单个 2 MiB），**不进**工作流文档，因此不改 `schema_version`，旧版本读文档不受影响。
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
1. 模块契约声明 + 后端校验 / 编译改动 + 前端改读目录；
2. 物体节点的 `tracks` 输入、人框裁剪、tick 日志字段；测试模式下给出模拟人框；
3. 内置「VideoJev + YOLO 基础盘」+ 自定义模板的存取；
4. 调试接口和侧栏调试分区（物体、总结完整，其他节点状态）；
5. 测试：旧图兼容（逐字节同 plan）、`accepts` 拒绝 Contour → 物体、传递消费、裁剪框计算（含无人退回整帧）、模板去敏和实例化、调试接口；前端 `tests/workflow_canvas.cjs` 覆盖新端口和模板入口。

**第二期：** System 1 事件接人框元数据、逐请求调试日志；之后视需要给人脸、Contour 补调试记录。

**不在范围内：** 通用数据流引擎（模块间任意传数据）、按节点的测试模式、任何 GPU 验证。

## 5. 需要确认的事

1. 第一个 YOLO → VideoJev 实例落在**物体节点 + 裁剪**，事件节点的「整帧 + 框元数据」放第二期，可以吗？
2. 本分支基于 PR #121（物体节点）；PR #121 合并前，本 PR 作为它的后续 PR 提交，合并后再改到 main。可以吗？
3. 测试模式保持全局，调试面板只显示和提供现有开关，不做按节点模拟，可以吗？

## 6. 第一期实现情况

| 部分 | 位置 |
|---|---|
| 模块契约（端口、必需 / 可选、`accepts`、调试类型）和 `module_catalog()` | `workflow_modules.py`；`/v1/workflows` 返回 `modules` |
| 校验：`accepts` 检查；Apply 要求每个启用的模块接上 `frames` | `workflows.validate_graph` |
| 编译：输出节点的所有上游都运行；接上人框时 plan 多出 `objects.focus` 和 `objects.tracks_node_id` | `workflows.compile_graph` |
| 人框裁剪：`person_crop_box()`（并集、外扩 25%、最小边 0.35、移回画面内）、窗口共用一个裁剪框、无人退回整帧、框查询出错不影响判断 | `object_presence.py` |
| 人框来源：真实运行按源帧时间从 Bbox 取框；测试模式用一个固定的模拟人框 | `workflow_runtime._start_workflow_objects` |
| 模板：内置「VideoJev + YOLO 基础盘」、另存 / 删除、去掉摄像头和服务地址 | `workflow_templates.py`、`workflow_routes.py` |
| 调试接口：`GET /v1/workflow/nodes/{node_id}/debug?limit=`（1–200，默认 20），fleet 按摄像头转发 | `workflow_runtime.workflow_node_debug`、`service.py`、`fleet_outputs.py` |
| 画布：端口表改读服务端目录；物体节点的 `tracks` 输入只列 Bbox；「从模板新建…」「另存为模板」；模块侧栏「调试」分区 | `assets/workflow_canvas.js` |
| 测试 | `tests/test_workflow_composable.py`、`tests/workflow_canvas.cjs` |

调试记录格式各节点一致。物体节点每条对应一个计划 tick（含跳过的 tick 和原因），`input.focus` 记录用了几个框、裁剪框和对齐方式；总结节点每条对应一次请求；Bbox 每条对应一个检测快照（人数、track_id）；事件节点目前只有各事件的最新结果，没有耗时。人脸、Contour 只显示状态。

**没有做的：** 人框裁剪还没在真实视频上跑过，没有任何准确率数据；System 1 事件接人框元数据（第二期）；人脸、Contour 的逐次记录。
