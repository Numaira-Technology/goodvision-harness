# N 路视频输出接口

所有地址都请求**主程序**，默认端口 8080。无需直连各通道容器或 GPU 节点。
先请求 `GET /v1/fleet`，读取 `cameras[].id`、`enabled` 和 `outputs`。
`outputs` 列出每类输出的地址；带 `{event_id}` 等占位符的项是地址模板。
编号依次为 `camera01`、`camera02`、…、`camera100`，以实际部署容量为准。

以下表格中的 `C` 表示 `/v1/fleet/cameras/{camera_id}`。所有输出方法为 `GET`。
例如第二路标注视频地址是 `/v1/fleet/cameras/camera02/video/annotated/stream.mjpeg`。
第一路将其中 `camera02` 换为 `camera01`；每一路都具备表中的完整输出能力。

| 输出 | 地址 | 格式 |
| --- | --- | --- |
| 运行、连接与分析状态 | `C/status` | JSON |
| 最新分析结果 / 持续结果 | `C/results/latest` / `C/results/events` | JSON / SSE |
| 整体描述 / 持续描述 | `C/results/overall/latest` / `C/results/overall/events` | JSON / SSE |
| 五类检测结果 / 持续结果 | `C/results/{feature}/latest` / `C/results/{feature}/events` | JSON / SSE |
| 人物追踪与丰富标签 | `C/rich-labels/latest` | JSON |
| 人脸快速识别结果 | `C/faces/fast` | JSON |
| 已登记人脸库状态 | `C/faces/gallery` | JSON |
| 场景、人物、身份、事件关联状态 | `C/scene` / `C/scene/events` | JSON / SSE |
| 场景历史 | `C/scene/history?limit=100` | JSON |
| 场景证据清单 | `C/scene/evidence` | JSON |
| 场景证据图片 | `C/scene/evidence/{request_id}/{frame_sequence}.jpg` | JPEG |
| 事件列表及筛选 | `C/events` | JSON |
| 事件详情 | `C/events/{event_id}` | JSON |
| 事件新增、更新、重置通知 | `C/events/stream` | SSE |
| 事件录像 | `C/events/{event_id}/video?revision=N` | MP4，支持 Range |
| 分析帧快照 / 视频 | `C/video/latest.jpg` / `C/video/stream.mjpeg` | JPEG / MJPEG |
| 实时预览快照 / 视频 | `C/video/live/latest.jpg` / `C/video/live/stream.mjpeg` | JPEG / MJPEG |
| 标注快照 / 视频 | `C/video/annotated/latest.jpg` / `C/video/annotated/stream.mjpeg` | JPEG / MJPEG |
| 显示元数据 | `C/video/display/metadata` | JSON |
| 显示帧与元数据批次 | `C/video/display/frames` | GVD1 二进制 |

`feature` 为 `fight`、`smoking`、`fall`、`weapon`、`stranger`。
`C/experimental/scene` 保留为场景兼容别名。
退休的 `C/faces/strangers` 和 `C/faces/strangers/{stranger_id}/photo` 仍返回 410，
其中替代链接指向该通道的事件接口。

## 数据归属与关联

- JSON 对象和每个 SSE JSON 数据包在顶层增加 `camera_id`；原有字段结构不变。
  嵌套事件、人物、检测框、证据等记录归属于这一通道。
- 所有成功进入输出处理的 HTTP 响应带 `X-Camera-Id`，包括 JPEG、MJPEG、MP4 和 GVD1。
  GVD1 的 JSON 头也包含 `camera_id`；长度字段同步更新，图片字节和帧描述保留。
- 返回的证据图片、事件录像、迁移地址等内部输出链接已经带同一通道前缀，直接使用返回链接即可。
- `source_id`、`stream_session_id`、`message_id`、`event_id`、`subject_id`、`track_id`
  等原始标识保持不变。多路接收方以 `(camera_id, 原始标识)` 建索引，
  不能仅凭不同通道上恰好相同的人员或事件 ID 合并数据。
- 视频仍保留原接口的帧序号、媒体时间和分析消息关联头。
  MJPEG 每帧的关联头保持不变，整条连接的所属通道由 URL 和 `X-Camera-Id` 标识。
- 分析结果 v2 和事件 v1 的随附 JSON Schema 已增加可选的 `camera_id`。

## 参数、续传与错误

每个接口保留原接口的查询参数、校验和状态码，详见主程序 `/docs`、`/openapi.json`
或 [接口交付说明](../../API_DELIVERY.md)。例如：

```text
C/events?event_label=smoking&subject_id=person-1&limit=50&before=...
C/scene/history?limit=100
C/results/events?after=42
C/video/display/frames?after=42&connection_id=...&revision=7&stream_session_id=...
```

每条 SSE 连接仅输出该通道，分别保存该接口返回的 `id`。重连时发送 `Last-Event-ID`；
不要把第一路的游标用于第二路，也不要在不同种类的 SSE 之间共用游标。
结果 SSE 保留 `epoch:cursor` 语义，显式非零 `after` 优先；事件归档 SSE 的
`Last-Event-ID` 优先于 `after`，历史过期或服务换代会按原协议发送 `reset`。
场景 SSE 每次重连先发送当前快照。注释心跳不含 JSON，无需当作结果解析。

事件 MP4 保留 `Range`、`If-Range`、`Content-Range`、ETag 和 206/416 状态，
可继续拖动播放。录像尚在生成时返回 202 JSON 和 `Retry-After`，不要当作 MP4 播放。
指定尚不存在的录像修订返回 404。未产生结果或画面时也可能返回 404。

不存在的通道返回 404；通道超过当前可用容量时视频输出返回 409，状态及历史数据仍可读取。
上游失联、重定向、异常内容或超过响应上限，在尚未发送响应时返回 502；
持续流中途故障会结束连接，客户端可重连。读取错误不会退到其他通道。
JSON 和单个 SSE 事件上限 2 MiB，图片和 GVD1 批次上限 8 MiB；
MJPEG、SSE、MP4 按块传送，客户端断开即释放上游连接。输出连接池与控制连接池独立。

## 两路调用示例

以下示例主程序地址为 `http://192.168.20.105:8080`，接入其他现场时换为实际地址。

```powershell
$base = "http://192.168.20.105:8080"
Invoke-RestMethod "$base/v1/fleet"
Invoke-RestMethod "$base/v1/fleet/cameras/camera01/results/latest"
Invoke-RestMethod "$base/v1/fleet/cameras/camera02/results/latest"
curl.exe -N "$base/v1/fleet/cameras/camera02/results/events"
```

两个视频播放器分别使用：

```text
http://192.168.20.105:8080/v1/fleet/cameras/camera01/video/annotated/stream.mjpeg
http://192.168.20.105:8080/v1/fleet/cameras/camera02/video/annotated/stream.mjpeg
```

控制仍使用每路 `connect`、`disconnect`、`settings`、`details` 和全局
`/v1/fleet/analysis/start`、`/stop`；显示统计回传仍为 `POST C/video/display/metrics`。
浏览器摄像头上传及控制 WebSocket `/ws` 保持原协议，没有新增可绕过全局分析控制的转发入口。
消费多路输出使用上表的 HTTP/SSE 接口。

旧 `/v1/video/...`、`/v1/results/...`、`/v1/events/...` 等地址继续返回主运行时（第一路）的
原格式，未变为合流。接入多路应统一使用带 `camera_id` 的地址。
