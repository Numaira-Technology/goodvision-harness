# Supervision 组件适配实验

2026-09-23，分支 `codex/supervision-event-runtime`，从 `24e8ca8` 开始。
依据 Chat 的“分析Supervision优势”：通用 CV 使用成熟组件，GoodVision 维护
Observation、源时序、事件生命周期、身份关联与事件级可靠性。

适配后的实时部署与硬件结论见[部署评估](supervision-deployment-assessment.md)，
其中区分原始信号和确认报警，并记录开启组件的 CPU、RSS 与共享锁等待测量。

## 实际分层

```text
goodvision_harness/
  perception/
    adapters/       Supervision Detections 与现有 HTTP 检测器适配
    components.py   filter / NMS / polygon / line / smoother / renderer
    tracking.py     固定版本 ByteTrack 离线对照
    routes.py       现有 SAM/Contour 客户端与协调器工厂
    routing.py      现有模型路由
  observations/
    schema.py       Observation / EventObservation
    normalization.py
  events/
    state_machine.py 原有 SceneState、证据账本、重放、修订、解除
  runtime/
    scene.py        原有现场运行时、异步完成、原生跟踪提交与预算
    scheduling.py   原有证据缓冲与检查规划
  evaluation/
    replay.py       原有 JSONL replay + EventObservation 输入
    event_metrics.py 事件匹配、误报/时、延迟、边界误差、重复与漏检时长
    detection.py   Supervision mAP 与 COCO/YOLO/VOC 转换
    cli.py         事件级评测命令
  actions/
    alerts.py      原有告警门槛与身份复核
    results.py     原有 AnalysisResult v2 发布
```

这些目录拥有实际实现。原 `scene_*`、`model_routing` 导入以模块别名兼容，
保持旧调用方和测试替换指向同一个实现；旧 `perception` 公共导入仍可用。
本次没有机械搬迁其余视频、模型协议、人脸与归档模块。

## 组件与接入位置

| 组件 | 上游实现 | 当前接入 |
| --- | --- | --- |
| 检测容器、像素/归一化框 | `Detections`、`denormalize_boxes`、`clip_boxes` | 所有适配器的唯一 CV 数据容器 |
| mask 外轮廓 | `mask_to_polygons`、`approximate_polygon` | 外部 Detections → harness；最多 16 段、每段 64 点，保留原源帧标记 |
| 置信度过滤、按类别 NMS | `Detections` 索引、`with_nms` | 非原生 rich-label 客户端；参数默认不启用 |
| 多边形区域、人数、越线 | `PolygonZone`、`LineZone` | 实时 scene 的 `supervision.observation` |
| 平滑 | `DetectionsSmoother` | 独立 `display_detections`；不覆盖证据框 |
| 绘制、热力图 | `BoxAnnotator`、`LabelAnnotator`、`HeatMapAnnotator` | 离线/调试 renderer，复制原图，按会话重置 |
| 普通跟踪 | 固定版 `ByteTrack` | 离线 baseline，不接管 SAM 原生 ID |
| 检测评测、数据集 | `metrics.MeanAveragePrecision`、`DetectionDataset` | 独立 detection evaluation 模块 |

`SupervisionLabelClient` 装饰现有非原生检测客户端，按照当前 JPEG 分辨率进行转换、
过滤/NMS，再把原对象交还协调器。每一行携带原对象索引，身份属性、未知置信度、
真实 bbox 和 mask 证据不会被一次 CV round trip 改写。

SAM/Contour 的原生批次仍完整提交。可选几何分析在原有源证据与身份提交之后运行，
不会删除低分原生人物、抑制重叠人物或重新分配 ID。实时区域使用 1000×1000 的
归一化工作平面，bottom-center 为触发点；边界精度是该平面的一个像素。
这是几何观察，不会自动生成 intrusion 或其他安全告警。

## 时序和证据边界

- 所有 CV 状态按 source/session 隔离，要求源时间不倒退且序号严格递增；拒绝重复、
  迟到的几何输入。VLM 完成仍可乱序，`EventObservation` 进入已有源账本重放。
- `SceneState` 继续决定 suspected/confirmed/ended/retracted，保留 N 帧确认、
  有效 clear、失踪门槛、修订和人物合并语义；本次不改变公开事件阶段名称。
- `detected=null`、模型失败、预测框、过期框不变成 clear。成功的空检测帧允许
  区域计数为零，但不会凭空确认安全事件。
- 平滑可能把框移到较早的位置，因此只进入调试显示。现有生产证据视频、中文 HUD、
  SAM 同源轮廓与异步显示融合继续使用原有语义。
- 原生 epoch 改变或超过 `max_gap_seconds` 时清空 CV 历史与越线计数，并在输出中
  增加 `generation`、`reset_reason`。这些计数只覆盖当前 generation。
- 最多 128 个输入对象/帧、256 个类别、默认 2048 个累计轨迹 ID、每类最多 16 个区域。
  超限或可选组件失败公开为 `state=unavailable` 并增加 failures；不回滚已提交的原生
  证据。长会话触及 ID 上限需新会话/epoch 或合理调整上限，未实现无限期区域统计。
- `EventObservation` 携带模型名称、可选版本与执行耗时；真实时序 inspection 当前
  标记来源 `temporal_inspector`，未知版本/耗时保留空值，不伪造模型版本。

## 开启实验

```bash
python -m pip install -e '.[dev,webrtc,vlm,supervision]'
```

源码默认 `supervision.enabled: false`，未启用时不导入 Supervision。
在实际使用的 harness YAML 中配置，例如：

```yaml
supervision:
  enabled: true
  confidence_threshold: null
  nms_threshold: null
  smoothing_length: 5
  max_gap_seconds: 3.0
  max_tracks: 2048
  zones:
    - name: restricted
      points: [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
  lines:
    - name: entrance
      points: [[0.0, 0.5], [1.0, 0.5]]
```

现有 active scene 会话的 `/v1/scene` 和 scene SSE 返回 `supervision` 字段。
非原生 rich-label 路径还可单独设置 `confidence_threshold: 0.4` 和
`nms_threshold: 0.5`。原生 SAM 不应用这两个筛选参数。修改配置后新建会话。
启动、模型就绪、GPU/远端凭证继续使用[现有配置方式](../guides/configuration.md)。

## 离线复现

```bash
python tools/benchmark_supervision.py --frames 100
python tools/scene_replay.py tests/fixtures/scene_smoking.jsonl
python -m goodvision_harness.evaluation.cli predictions.json targets.json --camera-hours 2
```

replay 保留旧 `session/person/event/tick` 格式，并支持
`EventObservation.as_dict()` 生成的 `event_observation` 行。
该格式会按原源时间重放，`revision` 增加可显式修订或撤回；旧格式行为保持兼容。

事件评测的两个输入文件均为以下 episode 对象数组，prediction 还必须包含
`alert_seconds`（**实际告警到达时刻，相对相同会话起点**）：

```json
[{
  "event_id": "smoking-1",
  "source_id": "camera-1",
  "stream_session_id": "session-1",
  "event_label": "smoking",
  "subject_ids": ["person-1"],
  "start_seconds": 10.0,
  "end_seconds": 20.0,
  "alert_seconds": 12.3
}]
```

默认要求 camera、session、事件类型、人物组一致及 temporal IoU ≥ 0.3，进行确定性的
最大匹配数量的一对一匹配，优先相交比例。跨 detector 的 actor ID 尚未对齐时可显式使用
`--ignore-actors`，报告会记录该选择。未匹配预测计误报，重叠重复另计 duplicate。
`camera_hours` 必须包括无事件时段；不得只用 positive clips 的时长计算误报率。
无预测/无标签的 precision/recall 保留 null；无命中时延迟也是 null。
漏检时长按每个目标区间扣除其唯一匹配预测的重叠长度，重复告警不额外抵扣。

这不是自动把 `confirmed_since_timestamp_seconds` 当作告警时延：前者是源证据
时刻，后者包含模型/调度/交付延迟，需实际记录。未知事件结束边界应先标注观察窗口，
不要将尚未结束的事件伪造为完整 episode。

## 版本、许可和待验收项

依赖固定 [Supervision 0.30.5](https://github.com/roboflow/supervision/releases/tag/0.30.5)，
遵循其 [MIT 许可](https://github.com/roboflow/supervision/blob/0.30.5/LICENSE.md)，
随源码保留[许可副本](../../src/goodvision_harness/assets/Supervision-LICENSE.txt)。
使用库调用适配，不复制上游算法代码。Supervision 的直接/间接依赖仍保留各自许可。

上游 `ByteTrack` 自 0.28 弃用并计划在 0.31 移除，所以本实验固定版本且只作离线对照。
独立 Roboflow Trackers 2.6.0 声明依赖 `opencv-python`，会与现有
`opencv-python-headless` 同时提供 cv2；当前 extras 不引入该冲突。未来迁移应先做
统一 OpenCV 环境与真实遮挡/移动摄像头 benchmark，再决定替换范围。

没有增加模型下载、Roboflow 服务账户、通知外发或新的推理路线。没有用通用 ByteTrack
取代 SAM3.1 的原生记忆，也没有以 polygon primitive 宣称复杂事件可靠性提升。
真实视频准确率、移动摄像头、长会话、十二路并发与端到端延迟仍需专门验收。
验证数字见[事件与生命周期验证](../validation/events-and-lifecycle.md#supervision-组件实验2026-09-23)。
