# GoodVision 接口交付说明

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

网络视频源默认采用 `stream.frame_crop: auto`：每次连接开始时检查连续画面，确认是
左右并排双目视图才取左半幅；单目或证据不足时保留全画面。判断结果在会话中固定，
所有预览、分析、叠框和事件视频都使用同一坐标系。`frame.width` / `frame.height`
及归一化框仍对应实际交付画面，而不是裁剪前的双目拼接画面。

`GET /v1/status` 的 `stream.frame_crop` 返回配置模式；新增
`stream.crop_selection` 返回 `requested`、`effective`、`reason` 和 `probed_frames`。
自动判断中 `effective=null`、`reason=probing`；确认双目后为
`effective=left_half`、`reason=stereo_confirmed`；未确认则为
`effective=none`、`reason=stereo_not_confirmed`。手动模式的 `reason=manual`。
这是会话诊断信息，不改变 v2 分析消息和判断完整性字段。该信息无需显示在视频上。

视频状态栏仅显示确认无异常的 `CLEAR` 或当前检测到的事件（例如 `SMOKING`、
`FALL`、陌生人 `UNKNOWN`）。没有可用的完整判断时隐藏状态栏和描述；不显示
`INCOMPLETE`、`STALE`、`NO FACE` 或 `NO USABLE FACE`，也不误报 `CLEAR`。
部分分支已有事件时照常显示该事件及其描述。状态文案、对象标签和已知人员姓名
支持中文，字体已随程序内置。API 的 `analysis_complete`、`missing_event_labels`
及原始双语 `description` 保持原语义，接入方仍应据此判断结果是否完整。

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
