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

之前提供的通用接口继续保留：

- `GET /v1/results/latest`：最新一条完整分析结果，返回 `application/json`。
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
`latest` 在尚无结果、或最新分析不含该能力时返回 `404`；它不会向前寻找历史
阳性结果。正常返回且 `detected=false` 表示该次判断未检出，不是接口错误。

每条分析结果新增顶层 `description`，与 `frame`、`judgments` 同级，描述
`frame` 对应画面在本次分析中检出的事件。存在 `detected=true` 的判断时，按
`judgments` 顺序拼接这些判断的 `event_label: description`，以 `; ` 分隔；
全部未检出时固定返回 `"无异常 / No anomalies detected"`。原有 `judgments[].description` 继续保留，
表示各事件的判断依据。描述采用“简体中文 / English”双语格式，两种语言
表达相同的画面证据，由同一次识别请求生成，不增加翻译请求；本地人脸识别
和演示模式也采用双语描述。独立能力接口保留完整结果的全帧 `description`，即使
该接口过滤后的 `judgments` 只有一个未检出的事件，也可能在顶层描述中看到
其他能力检出的事件。完整帧描述可通过 `result.description` 读取。

SSE 的 `Content-Type` 为 `text/event-stream`，每条消息以空行结束，结构如下：

```text
id: <结果流标识>:<游标>
event: analysis_result
data: <一条完整分析结果 JSON>

```

客户端应监听 `analysis_result` 事件，并对 `data` 执行 JSON 解析。无新结果时
会收到 `: heartbeat` 注释，不能将其作为 JSON 解析。独立 SSE 会跳过不含对应
能力的结果。支持 `?after=<非负整数游标>`，也支持把上一条完整 `id` 放入
`Last-Event-ID` 请求头续传；显式非零 `after` 优先。续传仅针对服务保留的
内存历史，不能作为永久归档接口。`latest` 响应头提供 `X-Result-Cursor` 和
`X-Result-Stream`。

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
  "schema_version": "1",
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
  "description": "无异常 / No anomalies detected"
}
```

`type` 固定为 `overall_description`；其余字段沿用完整结果中的同名字段。
`message_id`、`observation_id`、`source_id`、`stream_session_id` 为字符串，
`sequence` 为结果序号整数，`emitted_at` 为 UTC 时间字符串，`frame` 为对应
分析帧的信息。接口省略 `judgments`、`objects`、`model` 和 `analysis_kind`。
客户端可用 `message_id` 与五能力结果、精确证据帧关联。

`description` 为“简体中文 / English”双语字符串；没有检出事件时固定为
`"无异常 / No anomalies detected"`，有事件时例如
`"smoking: 有人吸烟。 / A person is smoking."`。
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

## 陌生人证据照片

先调用 `GET /v1/faces/strangers` 获取已保留的陌生人列表，返回示例：

```json
{
  "count": 1,
  "strangers": [
    {
      "stranger_id": "stranger-00000000000000000000000000000001",
      "photo_available": true,
      "photo_url": "/v1/faces/strangers/stranger-00000000000000000000000000000001/photo"
    }
  ]
}
```

在 `photo_available=true` 时，使用 `BASE + photo_url` 下载照片：

`GET /v1/faces/strangers/{stranger_id}/photo`

上面的 ID 仅为格式示例；实际请求应直接使用列表返回的 `photo_url`，不要自行
编造 ID。

成功返回 `image/jpeg`，响应头包含 `X-Stranger-Id`；照片不存在时返回 `404`。
陌生人结果对象的 `attributes` 中也可能直接包含 `stranger_id` 和 `photo_url`。

照片是保留的人脸裁剪图，不是每次事件的全帧截图，也不是逐事件照片归档；同一
陌生人可能复用已有照片。真实照片生成要求正式模式 `dry_run=false`、启用
`stranger` 能力、已授权的人脸检测与识别模型正常工作，并实际检出未知人员。
留存失败时仍可能有告警，所以不能假定每条告警都必有照片。

## 视频与精确证据帧

所有方法均为 `GET`：

| 输出 | 连续流 | 最新 JPEG |
| --- | --- | --- |
| 连续带框展示视频 | `/v1/video/annotated/stream.mjpeg` | `/v1/video/annotated/latest.jpg` |
| 精确分析证据帧 | `/v1/video/stream.mjpeg` | `/v1/video/latest.jpg` |
| 近实时无框原视频 | `/v1/video/live/stream.mjpeg` | `/v1/video/live/latest.jpg` |

此前的 `/v1/video/annotated/stream.mjpeg` 保持不变。带框展示流有配置决定的
延迟，并可能使用插值或保持的框；需要与某次 JSON 严格对应时，应使用精确
分析证据帧接口。该流按分析结果更新，不保证原视频帧率。

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
全部五个能力。改为 `dry_run=true` 可验证演示链路，但不会生成真实陌生人照片。

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
