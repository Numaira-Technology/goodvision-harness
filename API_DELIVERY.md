# GoodVision 接口交付说明

人员追踪模块的公开处理方式标识为 `person_tracking`。设置接口接受
`{"processing_method":"person_tracking"}`，状态和会话响应使用同一标识；
已有部署设置会自动兼容。交付配置通过 `display.tracking_delay_seconds`
设置该模块的默认画面延迟，控制台保存的延迟设置优先。

## 模型服务运行位置

`GET /v1/deployment/settings` 和 `PUT /v1/deployment/settings` 分别配置两类服务：

| 字段 | 值和含义 |
| --- | --- |
| `inference_mode` | `cloud` 云端、`local` 本机 GPU、`server` 独立模型服务器 |
| `tracking_mode` | `legacy` 原部署服务（单路）、`local` 本机 GPU、`server` 自备追踪服务器 |
| `inference_server_url` | 独立模型服务器的 HTTP(S) 主机地址，例如 `http://192.168.1.60:8000` |
| `inference_server_model` | 独立服务器对外提供的模型名称，默认 `local-vision` |
| `tracking_server_url` | 追踪服务器的 HTTP(S) 主机地址，例如 `http://192.168.1.61:8092` |
| `tracking_server_capacity` | 追踪服务器实际配置的正整数路数，默认 1 |
| `tracking_sample_fps` | 每路送入人物追踪的采样帧率，有限数值 `0.5–30`，默认 `2`；不改变显示、人脸采样或轮廓补帧频率 |

控制台“采样帧率（FPS）”对应 `tracking_sample_fps`。停止全部分析后保存，对下一次分析生效；
保存值优先于 `LOCAL_TRACKING_FPS`，并随部署设置持久保留。各视频通道在统一开始分析时同步主控制台的采样率。

两种 `server` 模式均不要求主程序机器有显卡。地址只接受主机和端口，可带末尾 `/`；
不接受接口路径、查询参数或嵌入的账号密码。模型服务器须兼容聊天补全接口；追踪服务器
须使用本项目的独立追踪服务。配置容量不会自动扩容另一台机器，须与该服务启动路数一致。
例如云端大模型加独立 16 路追踪：

```json
{"inference_mode":"cloud","tracking_mode":"server","tracking_server_url":"http://192.168.1.61:8092","tracking_server_capacity":16}
```

原布尔字段 `local_inference_enabled` / `local_tracking_enabled` 继续兼容，只有 `local`
模式返回 `true`；只提交旧字段时，`false` 分别恢复 `cloud` / `legacy`。
同时提交新旧字段必须一致。保存后改变后续会话路由并在重启后保持；运行或清理中修改
返回 409，未配置本机 GPU 服务时选择 `local` 返回 400，不会自动回退到其他服务。

响应的 `local_gpu` 提供主机显卡数量、两类工作的显卡编号和分配方式；它表示部署
能力，模型实际就绪仍由各服务健康检查判断。`api_key_required=false` 表示当前自备
推理服务无需云端密钥。切换保留各线路原有密钥，不把云端密钥发送给自备服务；
自备服务可使用独立的 `LOCAL_INFERENCE_API_KEY` 或该线路保存的密钥。

默认启动 1 路；`--streams N` 接受正整数，按实际需求部署 N 路，12 路不是软件上限。
多路在同一控制台分别连接和并排显示，统一开始或停止所有已连接视频的分析；
各路使用独立持久数据卷，共享大模型与人员追踪模块权重。
仅主控制台对外开放端口（默认 8080），其他通道通过内部网络访问。
人物追踪选择本机 GPU 或自备服务器时允许多路，大模型可以使用云端；原部署模式仅允许第 1 路。
`GET /v1/fleet` 返回已部署容量 `capacity`、当前可用容量 `usable_capacity`、追踪位置
`tracking_mode`、全局 `analysis` 状态，以及每路连接、详细设置、结果和画面地址。自备追踪模式可用容量取主程序
已部署路数与 `tracking_server_capacity` 中的较小值，超出容量的通道不能启动。
各通道地址为 `/v1/fleet/cameras/camera01/...`、`camera02/...`，编号可超过 12；
使用返回的 `status_url`、`connect_url`、`disconnect_url`、`detail_url`、
`results_url`、`scene_url`、`settings_url`、`preview_url`；全局启停使用
`analysis_start_url` 和 `analysis_stop_url`。已保存的视频地址不回显。
改变运行位置、服务地址、模型名称、容量或其他全局推理设置前，必须停止全部分析并完成清理；状态未知时返回 409。
单路沿用原 `face-data` 卷，多路按 N 创建独立摄像头数据卷。
每路大模型分析目标为 1.25 次/秒，N 路共 1.25 × N 请求/秒；默认追踪为每路 2 帧/秒。
12 路仅是容量参考值，各部署规模的实际吞吐仍需实机验证。启动、参数与容量验证见
[本地 GPU 部署说明](docs/local-gpu.md) 和 [模块化部署说明](docs/modular-services.md)。

## 多路视频连接与统一分析

连接视频只拉流、解码和预览，不调用人员追踪、大模型或人脸分析，也不要求分析密钥已配置。
每路可以分别连接、断开；确认画面和各路详细设置后，只通过全局按钮开始所有已连接视频的分析。
尚未连接的预留通道不参与本次分析，不必接满部署容量。分析启动前统一同步服务配置并检查各路；
任一路准备失败不会开始其他路，启动阶段失败会尝试停止本批已启动的分析并返回错误。
停止全部分析保留视频连接与实时预览，待分析收尾完成后可修改设置或重新开始。
`autostart=true` 现在只在主入口启动时连接其已保存视频，不自动开始分析；单路部署也遵循此规则。
有通道仍在连接或断开时，须等待完成后再统一开始，避免遗漏尚未就绪的视频。

| 接口 | 用途与请求 |
| --- | --- |
| `POST /v1/fleet/cameras/{camera_id}/connect` | 单路连接预览；接受可选 `stream_url`、`camera_layout`，省略时使用已保存值 |
| `POST /v1/fleet/cameras/{camera_id}/disconnect` | 单路断开视频；提交 `{}`，须先停止全部分析 |
| `GET /v1/fleet/cameras/{camera_id}/details` | 读取该路检测项目、敏感度及可选值 |
| `PUT /v1/fleet/cameras/{camera_id}/details` | 保存该路 `features` 与 `event_sensitivities`，重启后保留 |
| `GET /v1/fleet/cameras/{camera_id}/results/latest` | 查看该路最新分析 JSON；尚无结果时返回 404 |
| `GET /v1/fleet/cameras/{camera_id}/scene` | 查看该路人物与事件状态 |
| `POST /v1/fleet/analysis/start` | 提交 `{}`，开始所有已连接视频的分析；不接受单路选择或临时检测设置 |
| `POST /v1/fleet/analysis/stop` | 提交 `{}`，停止全部分析并保留视频预览 |

例如给第二路保存 `{"features":["smoking","stranger"],"event_sensitivities":{"smoking":"high"}}`
到 `/v1/fleet/cameras/camera02/details`，只改变第二路的检测选择。所有路均提供相同的详细控制，
其中模型运行位置、服务地址和密钥仍由主控制台统一管理。

状态接口分别报告 `connection_state` / `connected` 和 `analysis_state`；连接画面就绪不代表
已开始分析。全局 `analysis` 包含 `state`、本次 `camera_ids` 与各路 `errors`。
分析启动、运行或收尾期间，连接/断开视频、修改视频地址、每路分析设置和共享模型配置返回 409；
查看画面、详情和结果不受影响。仅调整显示缓存秒数仍保留现有的运行中修改能力。

旧的 `/v1/fleet/cameras/{camera_id}/start`、`/stop` 已停用并返回 409；
描述中的 `start_url` / `stop_url` 仅保留兼容字段，不可再调用。
部署容量大于 1 时，旧 `/v1/sessions/start`、`/stop` 和 WebSocket 单路启停同样禁止绕过统一控制。
下文保留的这些旧接口示例仅适用于单路兼容接入；多路接入应使用本节流程。

## N 路完整输出接口

所有读取输出均可使用 `/v1/fleet/cameras/{camera_id}` 替换原路径中的 `/v1`。
覆盖三种视频/JPEG、通用及五能力结果与 SSE、整体描述、追踪标签、人脸结果、
场景与历史、证据图片、事件列表/详情/实时通知/MP4，以及显示元数据与帧数据包。
`GET /v1/fleet` 的每路 `outputs` 提供完整地址和模板，`mjpeg_url` 为标注视频流地址。

JSON/SSE 增加 `camera_id`，二进制响应带 `X-Camera-Id`，GVD1 数据包头也带 `camera_id`。
证据和录像链接指向所属通道，游标按通道及接口独立保存；原标识与旧单路接口保持兼容。
完整路径表、参数、断线续传、录像 Range 和接入示例见
[N 路视频输出接口](docs/multi-camera-api.md)。

## 访问地址与交付范围

以下接口均以现场实际部署地址 `BASE` 为前缀。默认直连端口为 `8080`，例如
`http://192.168.1.50:8080`。此前使用的 `http://192.168.1.50` 只有在现场已配置
80 端口转发到服务端口时才可继续使用；IP 地址本身不是镜像固定配置。

接口路径在基础版和完整 GPU 版之间保持一致。基础版只有 CPU 主服务；完整
GPU 版另包含人脸 GPU 支持和目标检测、跟踪、行人属性服务。具体镜像与
模型文件范围以随包 `RELEASE.md` 为准，完整 GPU 的安装与验收见随包部署说明。
接口存在不等于真实模型已经就绪。默认首次启动为 `dry_run` 演示模式，正式
识别需配置视频源、相应模型及推理服务凭据，并切换为 `dry_run=false`。

## 通用 JSON 与五能力独立 JSON

### 持续事件模式的兼容输出（2026-09-19）

`SCENE_ENGINE_MODE=active` 时，`/v1/results/latest`、五能力独立 `latest`、
对应 `events` SSE、整体描述和已配置的 Webhook 现在接收持续事件状态的
AnalysisResult v2 投影，`analysis_kind="scene_state"`。HTTP 路径、消息名、
字段结构和 `schema_version="2"` 保持不变；它不是恢复严格 v1 schema。

每类事件的 `judgments[].detected` 在该类至少有一个公开 `phase="confirmed"`
事件时为 `true`；仅有 `suspected`、`ended` 或 `retracted` 时为 `false`。
不同人物分别维护状态，一个人物结束不会清除其他人物仍确认的同类事件。
陌生人同样使用身份复核后的公开 phase；原始人脸候选不再直接向旧结果通道
发布阳性。需要即时人脸检测、身份及原始框时使用 `/v1/faces/fast`。

尚无事件记录的能力不伪造阴性，列入 `missing_event_labels`；全部无记录时
尚无分析结果。首次疑似、确认、证据更新、结束和撤回触发兼容消息，重复的
scene 心跳不重复发布。没有新消息时 `latest` 保留最近结果，不代表新一帧判断。
仅有未确认/结束/撤回的事件时顶层描述为“暂无已确认的事件 / No confirmed events”；
有缺失类别时仍按 v2 标记分析未完成。

兼容消息是当前场景状态，`frame` 来自当前会话最后收到的真实源帧，
`model.latency_seconds=0` 表示转换本身没有新推理。`objects` 是事件关联人物；
`attributes.subject_id`、`event_ids`、`scene_events` 保留人物、事件、phase 和证据时间。
框可能是人物最后一次可见位置，必须结合 `attributes.observation_state`、
`last_observed_timestamp_seconds` 和 `last_observed_frame_sequence` 使用，不能当成
每条消息均有同帧检测框。`attributes.scene_state` 区分 running/stopped 等运行状态；
停止检测不自动改写已经确认的历史事件。停机后身份复核的最终状态仅在未开始新会话时
更新旧结果缓存，旧会话不会覆盖新会话。

日志 `scene_result_published` 记录会话、消息、源帧和各类 detected；它证明服务端
发布到结果通道，不代表外部接收端已收到。多摄像头接口沿用相同转换。

**下面关于逐帧原始消息、分支独立发布及源帧严格对应的说明适用于 disabled 模式；
active 模式以本节的持续事件状态语义为准。**

当前分析消息使用 `schema_version: "2"`，契约文件为
`schemas/analysis-result-v2.schema.json`。新增的描述及完成状态属于 v2；
原 `schemas/analysis-result-v1.schema.json` 保持不变，供历史 v1 结果校验。
下述 HTTP 路径保持 `/v1/...`，路径版本不等于返回 JSON 的 schema 版本。
使用严格 v1 校验器的接入方需先支持 v2，再升级程序。

之前提供的通用接口继续保留：

- `GET /v1/results/latest`：源帧时间最新的一条原始分析消息，返回 `application/json`。
- `GET /v1/results/events`：持续输出分析结果，返回 SSE，而非单个 JSON 文档。

五个能力分别提供以下接口，所有方法均为 `GET`：

| 能力 | 最新 JSON | 实时 SSE |
| --- | --- | --- |
| 打架 | `/v1/results/fight/latest` | `/v1/results/fight/events` |
| 吸烟 | `/v1/results/smoking/latest` | `/v1/results/smoking/events` |
| 跌倒 | `/v1/results/fall/latest` | `/v1/results/fall/events` |
| 武器 | `/v1/results/weapon/latest` | `/v1/results/weapon/events` |
| 陌生人 | `/v1/results/stranger/latest` | `/v1/results/stranger/events` |

独立接口复用同一份分析结果，保留关联字段 `message_id`、`stream_session_id`
等，仅过滤 `judgments` 和 `objects` 为对应能力内容，不会启动新的推理任务。
各能力的 `latest` 分别返回该能力源帧时间最新的原始观测；全局最新消息不含
陌生人时，`/v1/results/stranger/latest` 仍可返回最近一次人脸分支结果。
仅当前会话尚无该能力结果时返回 `404`，不会为了保留告警而寻找更早的阳性。
正常返回且 `detected=false` 表示该次判断未检出，不是接口错误。

混合识别中的事件分析和人脸分析独立调度，任一分支完成即发布，无需等待另一支。
每条消息保留自己的 `message_id`、原始源帧和该分支的判断，不将不同帧拼成一条
完整结果。通用 SSE 和已配置的 Webhook 都接收这些独立消息。因此，较新事件帧
之后可能到达较旧的人脸帧；同一分支更旧的结果不会推翻已发布的新判断。
客户端应按能力比较 `frame.timestamp_seconds`，同时间再比较
`frame.source_frame_index`，不要用整个结果流的源时间过滤尚未收到的人脸分支。
全局 `latest` 仍只选源帧最新的一条消息，不是各能力的汇总；同源帧后到的消息可更新它。
会话切换会清空全局和各能力最新记录。

### 人脸快速结果（未融合）

`GET /v1/faces/fast` 返回最新源帧的人脸库匹配结果。本地人脸推理完成后即
更新独立内存缓存，早于原始结果落盘、有序身份提交、场景融合和原有结果回调。
读取接口不获取服务状态锁，不启动新推理；原有结果接口保持不变。

响应始终为 `200`，并带 `Cache-Control: no-store`。外层字段：

- `stream_session_id`、`source_id`、`generation`：对齐当前分析会话。
- `analysis_kind: "face_raw"`：仅当帧检测和人脸库匹配，不是最终事件判断。
- `active`：本会话人脸快速通道是否仍接收结果；停止后为 `false`。
- `available`、`result`：尚无结果时为 `false`、`null`；人脸库变更也会清空旧结果。
- `revision`：当前缓存版本；用 `(stream_session_id, generation, revision)` 去重。
- `age_seconds`：缓存发布距今秒数，不是视频源到结果的端到端延迟。

`result` 包含 `frame`（`source_frame_index`、`timestamp_seconds`、原始
`width`/`height`）、`emitted_at`、`gallery_version`、`error` 和 `faces`。
每个人脸包含归一化 `bbox: {x1,y1,x2,y2}`、`status`、检测分数，以及适用时的
`person_id`、`name`、`similarity`、`threshold`、`match_margin`、`reason`。
状态为 `known`、`unknown` 或 `unusable`。`unknown` 只是该帧未匹配人脸库，
不能直接当作融合后的陌生人告警。成功但无人脸时 `faces: []`、`error: null`；
推理/解码失败时 `faces: []`、`error` 为公开错误，不会伪装为无人脸。

缓存只保留一条源帧最新的结果：慢的旧帧不能覆盖新帧，轮询可能跳过中间帧；
完整有序结果仍从原接口取得。停止后保留最后结果并标记 `active: false`，新会话
建立新缓存，旧会话的迟到结果不会串入新会话。纯事件会话的快速通道不激活。
此接口不返回图片、特征向量、人物追踪 ID 或跨帧身份修正，也不绕过原有采样、
任务入队落盘和输入侧等待。多路部署应请求对应摄像头实例的接口。

`2026.09.10-full.4` 会发布所有有效的迟到观察，包括阳性和阴性。迟到阳性用于事件
通知，迟到阴性供消费者重建连续阴性计数；它们不会回退对应能力的 `latest` 或当前
精确画面。控制台按源帧顺序连续累计 2 次有效阴性才解除行为告警，阳性重置计数，重复
源帧仅计一次，失败/缺失不计阴性。`judgments[].detected` 始终表示原始单帧判断，
不是两帧确认解除后的告警状态；接入方需要相同保持行为时应自行按能力/人物重建状态。

每条分析结果新增顶层 `description`，与 `frame`、`judgments` 同级，描述
`frame` 对应画面在本次分析中检出的事件。存在 `detected=true` 的判断时，按
`judgments` 顺序拼接这些判断的 `event_label: description`，以 `; ` 分隔；
已启用能力全部完成且未检出时返回 `"无异常 / No anomalies detected"`。
原有 `judgments[].description` 继续保留，
表示各事件的判断依据。描述采用“简体中文 / English”双语格式，两种语言
表达相同的画面证据，由同一次识别请求生成，不增加翻译请求；本地人脸识别
和演示模式也采用双语描述。独立能力接口保留完整结果的全帧 `description`，即使
该接口过滤后的 `judgments` 只有一个未检出的事件，也可能在顶层描述中看到
其他能力检出的事件。完整帧描述可通过 `result.description` 读取。

`analysis_complete` 表示这一条消息是否包含全部已启用能力；`missing_event_labels`
列出该消息未包含的能力。混合识别正常发布的独立分支消息通常为 `false`，另一支
会用自己的消息发布；缺失不表示阴性，也不单独表示分支失败。此类原帧描述附带
“分析未完成 / Analysis incomplete”，不会报告全帧无异常。分支错误通过状态接口
另行报告。v2 schema 要求 `analysis_complete=true` 时缺失列表为空，反之非空。
控制台能力总览按各能力最新有效观测汇总；全部所选能力都有记录后可显示就绪，
不会因为每条消息都是 partial 而一直显示 `INCOMPLETE`。缺失分支不清除其他能力，
陌生人观测在后续源帧超过三秒后转为等待更新。原始 JSON、原帧日志和描述保持不变。
模型输出须通过中文、分隔符 ` / `、英文的格式校验；不符合格式时该推理
分支按失败处理，不将单语文本当作合规的双语结果发布。格式校验不验证翻译语义。

SSE 的 `Content-Type` 为 `text/event-stream`，每条消息以空行结束，结构如下：

```text
id: <结果流标识>:<游标>
event: analysis_result
data: <一条原始分析消息 JSON，可能仅包含一个分支>

```

客户端应监听 `analysis_result` 事件，并对 `data` 执行 JSON 解析。无新结果时
会收到 `: heartbeat` 注释，不能将其作为 JSON 解析。独立 SSE 会跳过不含对应
能力的结果。支持 `?after=<非负整数游标>`，也支持把上一条完整 `id` 放入
`Last-Event-ID` 请求头续传；显式非零 `after` 优先。续传仅针对服务保留的
内存历史，不能作为永久归档接口。`latest` 响应头提供 `X-Result-Cursor` 和
`X-Result-Stream`。游标按发布顺序单调增加，并非源帧时间；各能力的游标允许跳号。
全局 `latest` 的游标也可能小于另一能力 `latest` 的游标，因为两者按源帧时间选取。

## 视频状态显示

`2026.09.10-full.3` 改为手动选择摄像头类型，不再自动判断或等待探测帧。

网络视频源的 `camera_layout` 仅接受 `mono`（单目，完整画面）或 `stereo`
（左右并排双目，使用左半幅）。控制台可保存默认类型，也可在开始前选择本次类型。
`PUT /v1/deployment/settings` 可提交 `{"camera_layout":"stereo"}` 保存默认值；
`GET /v1/deployment/settings` 和 `GET /v1/ui/info` 返回当前默认 `camera_layout`。
`POST /v1/sessions/start` 和 WebSocket 的 `mode=stream` 启动消息接受同名可选字段，
仅覆盖本次会话；省略时使用保存值，否则使用配置文件。会话启动响应也返回该字段。
运行中修改已保存类型返回 `409`，切换须停止后重新开始。

配置文件默认 `stream.frame_crop: none`。旧 `auto` 值按 `none` 处理；显式
`left_half` / `right_half` 仍兼容，保存或提交 `camera_layout` 后以手动选择为准。
文件和浏览器摄像头输入保持全画面。网络流从第一帧开始固定使用所选画面，预览、
分析、叠框和事件视频共用同一坐标系；`frame.width` / `frame.height` 及归一化框
对应实际交付画面。

`GET /v1/status` 的 `stream.camera_layout` 和 `stream.frame_crop` 返回本次选择；
`stream.crop_selection` 包含 `requested`、`effective`、`reason=manual` 和
`probed_frames=0`。这些诊断字段不改变 v2 分析消息，也不显示在视频上。

视频状态栏仅显示确认无异常的 `CLEAR` 或当前检测到的事件（例如 `SMOKING`、
`FALL`、陌生人 `UNKNOWN`）。没有可用的完整判断时隐藏状态栏和描述；不显示
`INCOMPLETE`、`STALE`、`NO FACE` 或 `NO USABLE FACE`，也不误报 `CLEAR`。
部分分支已有事件时照常显示该事件及其描述。状态文案、对象标签和已知人员姓名
支持中文，字体已随程序内置。API 的 `analysis_complete`、`missing_event_labels`
及原始双语 `description` 保持原语义，接入方仍应据此判断结果是否完整。

## 控制台首次配置与缓存设置

`2026.09.10-full.3` 无需 `.env` 即可启动，首次安装保持待机和测试模式。
控制台“保存 API key”调用 `PUT /v1/deployment/settings`，请求仅包含
`{"api_key":"<密钥>"}`，不修改视频源、测试模式、自动启动或摄像头类型。
也可在同一请求提交 `dry_run`、`autostart`、`stream_url`、`camera_layout`。
密钥立即对下一次检测生效，并保存到现有部署数据卷，容器重建后仍有效。
设置接口仅返回 `api_key_present` / `api_key_persisted`，不回显密钥。
正式事件检测缺少密钥时，HTTP 启动接口返回 `400`，WebSocket 返回 `type=error`，
文案明确提示在控制台保存 API key，无需创建 `.env`；不会启动分析或预热人脸模型，已连接预览保持可用。
测试模式和仅本地人脸检测不要求此密钥。运行中的检测需停止后再保存密钥。

缓存时间通过同一 PUT 接口提交 `{"buffer_delay_seconds":3.5}`，接受 0–30 的有限数字，
不接受字符串或布尔值。仅修改缓存的请求允许在运行中提交，立即生效并保存；
若同时包含其他设置，则沿用停止会话后再保存的限制。GET 设置接口返回当前缓存秒数。
摄像头类型、缓存和密钥共享现有数据卷，无需新增环境变量。

## 运行前事件敏感度与线路选择

`GET /v1/ui/info` 返回 `event_sensitivity_options` 和 `event_sensitivity_defaults`。
打架、吸烟、跌倒、持械分别使用 `fight`、`smoking`、`fall`、`weapon`，允许值均为
`standard` / `high`。吸烟默认 `high`，其他行为默认 `standard`；本地陌生人识别
沿用独立人脸设置，不属于此映射。

HTTP 启动示例：

```json
{
  "mode": "stream",
  "features": ["smoking", "stranger"],
  "dry_run": false,
  "event_sensitivities": {"smoking": "high"}
}
```

单路兼容接入提交到 `POST /v1/sessions/start`。WebSocket `/ws` 的 `type=start` 消息接受相同
`event_sensitivities` 字段，浏览器摄像头使用 `mode=camera, analysis_mode=multi`。
省略的事件使用默认值；不在所选能力中的合法配置不生效。未知事件、未知档位、
非对象配置会被拒绝。设置仅在启动前解析，运行中不能通过重新启动请求更改；
启动响应和 `/v1/status` 的 `event_sensitivities` 返回本次实际配置。
多路控制台将 `features` 和 `event_sensitivities` 保存到各路 `/details`，再调用全局分析启动接口。

标准吸烟要求能辨认烟具。高敏感度可依据可见动作和上下文输出疑似吸烟；烟具不可辨认时，
正例的 `objects` 可以是 `smoking_action` 动作区域或 `person`，不能假设所有吸烟框
都是香烟物体。这些框仍是当前原始帧上的归一化坐标。两档均保留双语描述、严格 JSON
结构和正例必须有可见区域框的约束。

事件图像使用最长边 320 px 的完整画面，本地人脸使用独立高分辨率输入。模型无事件
时的极简输出由服务还原成原有完整 JSON 契约，接入方不应处理内部 `no_events` 格式。

`GET /v1/deployment/routes` 返回线路状态；`POST /v1/deployment/routes/select` 提交
`{"route_id":"A"}`（A–H）保存手动选择。自动测速仍调用
`POST /v1/deployment/routes/check`，从测试全部成功的线路中选择中位延迟最低者。
检测或测速进行中不能切换线路，手动线路选择保存在部署数据卷中。

## 人员身份 person_id 的返回位置

`person_id` 是录入人脸库时指定的身份 ID。实时识别成功后，读取分析结果中
`label="known_person"` 对象的 **`objects[].attributes.person_id`**；姓名位于
同一对象的 `attributes.name`。它不是分析结果的顶层字段，也不在 `judgments` 内。

| 接口 / 消息 | person_id 字段路径（相对于返回 JSON 或 SSE data） |
| --- | --- |
| `GET /v1/results/stranger/latest` | `objects[].attributes.person_id` |
| `GET /v1/results/stranger/events` 的 `analysis_result` | `objects[].attributes.person_id` |
| `GET /v1/results/latest`、`GET /v1/results/events` 的 `analysis_result` | 消息包含已知人脸时，位于 `objects[].attributes.person_id` |
| `GET /v1/events` | `events[].person_id` |
| `GET /v1/events/{event_id}` | `person_id`（事件对象顶层） |
| `GET /v1/events/stream` 的 `event_change` | `events[].person_id` |
| `POST /v1/faces/gallery/{person_id}` 录入成功响应 | `person.person_id` |
| `GET /v1/faces/gallery` | `people[].person_id` |

实时识别结果片段示例（省略其他字段）：

```json
{
  "objects": [
    {
      "label": "known_person",
      "attributes": {
        "status": "known",
        "person_id": "employee-007",
        "name": "Alex"
      }
    }
  ]
}
```

接入人员识别建议使用陌生人能力的独立 `latest` / SSE，并遍历 `objects`。
即使陌生人判断为 `detected=false`，仍可能返回已知人员；已知人脸不在陌生人判断的
`object_ids` 中。混合模式的通用 `latest` 可能是事件分支消息，因此不保证包含人脸。
`overall` 描述接口不返回 `objects`，也不返回 `person_id`。

未知或待确认人脸不提供 `attributes.person_id`，不要把对象编号当作录入身份。
当其他人物对象关联到已确认身份时，还可在
`objects[].attributes.subject_identity.person_id` 读取关联身份；该字段按关联情况出现。
连续事件未关联到录入身份时，`person_id` 为 `null`；群体事件也为 `null`。
`subject_id` 用于会话内人物轨迹关联，`object_id` 仅标识当前分析消息内的对象，
两者均不等于人脸库的 `person_id`。

## 全画面描述独立接口（Overall description）

以下路径属于程序通用接口，以实际部署地址 `BASE` 为前缀，与演示端口无关。
所有方法均为 `GET`：

| 数据 | 最新 JSON | 实时 SSE |
| --- | --- | --- |
| 全画面描述 | `/v1/results/overall/latest` | `/v1/results/overall/events` |

`latest` 返回 `application/json`，`events` 返回 `text/event-stream`。
两者使用相同的精简 JSON 结构，顶层 `description` 与对应完整分析结果中的
`description` 完全一致。接口复用已有结果，不启动额外推理，不要求启用全部
五种能力；每次分析产生结果时均可输出，未检出事件也会输出。
这里的“全画面描述”是本次已启用能力的检测摘要，并非额外生成的场景解说。

返回示例：

```json
{
  "type": "overall_description",
  "schema_version": "2",
  "message_id": "message-1",
  "observation_id": "message-1",
  "source_id": "external-camera",
  "stream_session_id": "session-1",
  "sequence": 1,
  "emitted_at": "2026-09-09T10:00:01.000Z",
  "frame": {
    "source_frame_index": 42,
    "timestamp_seconds": 1.25,
    "observed_at": "2026-09-09T10:00:00.000Z",
    "width": 1920,
    "height": 1080,
    "clock_source": "ingest_estimated"
  },
  "analysis_complete": true,
  "missing_event_labels": [],
  "description": "无异常 / No anomalies detected"
}
```

`type` 固定为 `overall_description`；其余字段沿用完整结果中的同名字段。
`message_id`、`observation_id`、`source_id`、`stream_session_id` 为字符串，
`sequence` 为结果序号整数，`emitted_at` 为 UTC 时间字符串，`frame` 为对应
分析帧的信息。接口省略 `judgments`、`objects`、`model` 和 `analysis_kind`。
客户端可用 `message_id` 与五能力结果、精确证据帧关联。

`description` 为“简体中文 / English”双语字符串；全部已启用能力完成且未检出时为
`"无异常 / No anomalies detected"`，有事件时例如
`"smoking: 有人吸烟。 / A person is smoking."`。
精简结果也保留 `analysis_complete` 和 `missing_event_labels`，可区分检测完成
与部分完成；后者描述为“分析未完成。部分能力暂无结果。 / Analysis incomplete;
some capabilities have no result.”，若已有阳性事件则同时保留其描述。
尚未产生分析结果时，`latest` 返回 `404`，不会把“尚无结果”当作“无异常”。
成功响应包含 `Cache-Control: no-store`、`X-Result-Cursor` 和 `X-Result-Stream`。

此接口的 SSE 事件名为 **`overall_description`**：

```text
id: <结果流标识>:<游标>
event: overall_description
data: <与 overall/latest 相同结构的 JSON>

```

客户端监听 `overall_description` 并解析 `event.data`，读取其中的 `description`。
SSE 保留通用结果流的游标和续传规则：支持 `?after=<非负整数游标>` 与
`Last-Event-ID` 请求头，显式非零 `after` 优先；仅可续传内存中仍保留的结果。
无新结果时发送 `: heartbeat` 注释。该流随分析结果更新，并非按原视频帧率推送。

## 连续事件与视频证据

即时识别的 JSON/SSE 与现有逐帧结果保持不变。后台按人物和事件类型整理连续事件；
停 1 秒后继续吸烟可沿用同一事件编号，但暂停帧的即时识别仍为阴性。

| GET 接口 | 用途 |
| --- | --- |
| `/v1/events` | 连续事件列表，按事件、人物、会话、源时间或 `message_id` 查询 |
| `/v1/events/stream` | 事件创建、修订、合并/拆分及视频状态 SSE |
| `/v1/events/{event_id}` | 当前事件详情，自动解析合并后的编号 |
| `/v1/events/{event_id}/video` | 当前可用视频，支持 Range；`?revision=N` 读取固定版本 |

每个事件默认保存首次有效阳性前 1 秒、后 2 秒的约 3 秒 MP4，5 FPS、最长边 720。
视频状态与事件状态分别返回：先有事件，视频随后完成；断流或源边界会标记实际截短。
迟到结果可能更正首次触发时间并生成新视频版本。读取 `evidence.available.url`
播放已经可用的具体版本；新版生成时不要把旧片段当成新版。

事件的 `subject_id` 表示会话内人物轨迹，`person_id` 表示录入身份；`object_id` 只在当前分析消息内有效。原 AnalysisResult v2 可在对象 `attributes`
中补充人物关联；事件编号尚未建立时，后续用 `/v1/events?message_id=...` 查询。

已知人脸以 `known_person` 保留在分析结果的 `objects` 中，包含
`attributes.person_id`，但不作为陌生人阳性的 `object_ids` 引用。
陌生人能力的独立结果接口同样保留这些身份对象。联合检测仅在所选安全检查完整、
且没有安全事件阳性时显示已知人脸的绿色 ID 框；安全事件阳性或结果缺失时隐藏绿框，
身份对象仍保留在 JSON 中。

事件流发送 `event_change`，一次携带受影响 `events` 和 `aliases`。客户端原子
更新这些记录。列表支持 `limit` / `before`，用响应中的 `epoch:cursor` 连接事件流；
收到 `reset` 后重新读取列表。已删除或不存在的事件返回 404。视频未就绪返回
202，生成失败且无可用版本返回 503。

默认保留 7 天且总量上限 10 GiB；清理范围包含事件、视频与关联索引。
完整字段、时间规则、修订机制、配置和迁移说明见 [事件证据说明](EVENT_EVIDENCE.md)。

旧 `/v1/faces/strangers` 和 `/v1/faces/strangers/{stranger_id}/photo`
统一返回 **410 Gone**，包含 `stranger_photos_removed` 与 `/v1/events` 迁移提示。
停止保存陌生人照片与陌生人特征库，已有旧文件不自动删除。

## 视频与精确证据帧

所有方法均为 `GET`：

| 输出 | 连续流 | 最新 JPEG |
| --- | --- | --- |
| 连续带框展示视频 | `/v1/video/annotated/stream.mjpeg` | `/v1/video/annotated/latest.jpg` |
| 精确分析证据帧 | `/v1/video/stream.mjpeg` | `/v1/video/latest.jpg` |
| 近实时无框原视频 | `/v1/video/live/stream.mjpeg` | `/v1/video/live/latest.jpg` |

此前的 `/v1/video/annotated/stream.mjpeg` 保持不变。带框展示流有配置决定的
延迟，并可能使用插值或保持的框；精确分析证据帧接口保留对应分析帧的内容，
应通过下述关联头确认它对应哪条 JSON。该流不保证原视频帧率，也不会因较旧的
人脸分支完成而倒退，因此不保证每条独立分支 JSON 都在 JPEG/MJPEG 输出中出现。

精确帧的 `X-Analysis-Message-Id` 对应 JSON 的 `message_id`；JPEG 接口在响应
头中提供它，MJPEG 在每个 JPEG 分段的头中提供它。还可读取
`X-Frame-Sequence`、`X-Media-Timestamp`、`X-Observed-At` 辅助定位。独立 JSON
与通用 JSON 共享该关联标识。最新截图只代表读取时的最新帧，不是按
`message_id` 查询历史图片的接口；需要归档时由客户端同步接收并保存。

## 启动、停止与能力选择

`POST /v1/sessions/start` 接收 `application/json`。以下为启用全部五个能力的
正式模式示例，视频地址应替换为现场可达地址：

```json
{
  "mode": "stream",
  "stream_url": "webrtc+http://192.168.20.199:1985/rtc/v1/whep/?app=live&stream=livestream",
  "analysis_mode": "multi",
  "features": ["fight", "smoking", "fall", "weapon", "stranger"],
  "dry_run": false
}
```

只开陌生人检测时，将 `features` 改成 `["stranger"]`。`features` 仅适用于
`analysis_mode="multi"`，必须为非空、无重复的受支持能力数组；省略时启用
全部五个能力。改为 `dry_run=true` 可验证演示链路，可生成合成输入的视频证据，不代表真实模型效果。

`POST /v1/sessions/stop` 无需请求体，返回 `{"stopped": true}` 或
`{"stopped": false}`。同一实例同时只允许一个运行中的 Session；如已有会话，
再次启动返回 `409`，需先停止。HTTP 启动接口用于网络流；浏览器摄像头会话
应走 `/ws`，不能在此传 `mode="camera"`。

停止请求先停止采集；已接收的人脸检测、人物复核合并、事件复核和视频证据归档
会继续在后台完成。`stopped=true` 表示已接受停止请求，不能用它判断归档完成。
收尾期间 `/v1/status` 的 `pipeline_state` 为 `draining`，`scene_drain.stage`
显示当前阶段；完成后 `scene_drain.state` 为 `completed`，失败则为 `failed`。
收尾结束前不能启动新会话。陌生人证据列表按当前会话中合并后的人物去重，
每人保留一段代表性证据，优先使用可播放的视频。

## 状态与接口文档

- `GET /v1/status`：运行状态与会话信息。
- `GET /health/live`、`GET /health/ready`：存活与就绪检查；不代替真实模型验收。
- `GET /openapi.json`、`GET /docs`：接口定义与接口参考文档。
- `GET /v1/rich-labels/latest`：最新目标跟踪与行人属性结果；完整 GPU 包
  默认启用该链路，基础包未启用独立 worker 时可能无结果。它不是五能力通用
  结果接口，也没有对应的 rich-label SSE。

## 目标跟踪与属性接口

外部集成优先使用主服务的 `GET /v1/rich-labels/latest`，读取最新快照中的
`objects`：包括 `track_id`、`label`、`confidence`、归一化 `bbox` 和
`attributes`。无结果时返回 `404`。这些跟踪 ID 仅在当前视频会话内有效，
不是人员真实身份或跨会话身份。行人属性异步补充，因此第一帧可能尚无属性；
年龄段、外观性别等字段是模型估计，不是身份事实。

完整包的感知服务在容器内部提供下列接口，不默认向局域网单独开放
8091 端口：

- `POST http://rich-label-worker:8091/v1/labels`：原始 JPEG 请求体，
  `Content-Type: image/jpeg`；必须提供 `X-Goodvision-Stream-Session-Id`、
  `X-Goodvision-Frame-Sequence`、`X-Goodvision-Timestamp-Seconds` 请求头。
  输入图片不超过 8 MiB；返回该帧的检测、跟踪和已就绪属性。忙时返回 `429`。
- `GET /health/live`、`GET /health/ready`：worker 存活和检测器就绪状态。
  `ready` 不代表异步属性模型已经成功推理，完整验收还须运行双模型自检并
  检查实际视频产生的非空行人属性。

感知服务用于人物/物品检测、跟踪、属性及带框展示；五个安全能力的判断和
陌生人识别仍由主服务的对应分析链路完成。

对外接入前请确认实际端口、现场网络访问控制，以及正式模式下的真实视频和模型
联调结果。以上路径兼容保留，但旧镜像不会因路径相同而自动获得新增接口。
