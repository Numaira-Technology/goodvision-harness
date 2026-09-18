# GoodVision Contour API v1

公开产品标识为 `goodvision-contour-1`，结果 `backend` 为 `goodvision_contour`。
除控制台静态资源及 `/health/live` 外，所有端点都要求：

```http
Authorization: Bearer <node-key>
Content-Type: application/json
```

| 方法 / 路径 | 用途 |
| --- | --- |
| `GET /health/live` | HTTP 服务存活，无需密钥 |
| `GET /health/ready` | 模型可接收新会话时为 200，否则 503 |
| `GET /api/status` | 控制台状态，未就绪也返回 200；依赖状态看 `ready` |
| `POST /api/admission` | `{"accepting_sessions": false}` 暂停新会话，true 恢复 |
| `GET /v1/models` | 产品模型标识和能力 |
| `POST /v1/contour/sessions` | 创建会话 |
| `POST /v1/contour/sessions/{id}/append` | 追加帧 |
| `DELETE /v1/contour/sessions/{id}` | 结束会话并释放资源，无请求体 |

创建会话：

```json
{"stream_session_id": "camera-01-session-unique-id"}
```

成功返回 `video_session_id`、`stream_session_id`、`predictor_epoch`、`backend`、
`memory_scope: "session"`、`state: "active"` 以及引擎公布的 `limits`。
首次追加 1 帧，此后每次最多 `limits.max_append_frames` 帧；
Jetson 节点为 1，常规 GPU 节点为 4。按会话串行发送，遵守返回限额。
每个通道最多一个活动会话，通道之间隔离追踪状态并共享模型；节点通道数由部署时的 `CONTOUR_CAPACITY` 决定。

多路使用 `/v1/cameras/camera-01/contour/sessions`、`camera-02` 等路径创建会话，追加和删除路径同样加上会话 ID。
每路就绪检查为 `GET /api/cameras/camera-01/status`；一路繁忙不影响另一路建立会话。
这些接口均需同一个节点访问密钥。旧 `/v1/contour/sessions` 接口继续访问第一路。
`GET /api/status` 返回 `camera_capacity`、`active_sessions` 和 `cameras` 汇总；`limits` 描述单通道限额。
不同通道可以同时提交请求，推理引擎按先进先出顺序使用 GPU；同一路必须串行提交。

```json
{
  "stream_session_id": "camera-01-session-unique-id",
  "request_id": "unique-append-001",
  "is_last_batch": false,
  "frames": [
    {"frame_sequence": 0, "timestamp_seconds": 0.0, "jpeg_base64": "<JPEG base64>"}
  ]
}
```

`frame_sequence` 必须递增，时间戳是源视频时间；新视频/重连必须使用新会话。
JPEG 单张最多 8 MiB，整个 JSON 最多 32 MiB。图像及会话其他限额以 `limits` 为准。
`stream_session_id` 和 `request_id` 为 1–160 位 ASCII 字母、数字、`_`、`.`、`:`、`-`。
客户端不能将 HTTP 上传时间充当源视频时间。

追加响应保留会话身份、请求 ID、accepted/processed/emitted 的帧序号和源时间、
`pending_frame_count`，以及 `frames`。每帧包含：

- `stream_session_id`、`frame_sequence`、`timestamp_seconds`。
- `objects`：稳定且不透明的 `ct-…` 跟踪 ID、`label: "person"`、0–1 坐标框、置信度。
- `attributes.native_track`：品牌 backend、会话 epoch、原生对象 ID。
- `attributes.mask_contours`：当前源画面上归一化轮廓点和源时间；可为空缺。
- `observation_state` 和最近观测帧/时间，避免将过期信息视为新观察。

`route_metadata.route` 为 `contour`，`observations` 中的 `provisional_object_ids`
标记尚未确认的新目标。此字段引用 `native_track.object_id`，不要与 `track_id` 混用。
确认策略为 `causal_native_three_detections`。轮廓引擎不提供真实世界身份识别。

最后一次追加设 `is_last_batch: true`；也可在有已处理帧后以空 `frames` 结束。
再调用 DELETE 清理。暂停接收新会话后，已有会话的追加和 DELETE 仍然有效。

相同会话中最后一次追加的完全相同请求可幂等重放；不要给同一个请求 ID 发送不同内容。
网关不会自动重试请求。504 表示结果不确定，不能直接推进到下一帧；
客户端需按相同请求重试或关闭会话。创建会话超时可能暂时占用内部名额，等待空闲清理后重建。
网关重启不清除内部会话；保留会话 ID 可继续，丢失 ID 则等待引擎空闲超时。

失败格式始终是 `{"error":{"code":"…"}}`，不向客户返回上游异常、文件路径或实现版本。

| HTTP | 含义 |
| --- | --- |
| 400 / 415 | 请求内容或类型错误 |
| 401 | 密钥不正确 |
| 404 | 会话或端点不存在 |
| 409 | 已有活动会话、顺序或幂等冲突 |
| 413 | 请求超出限额 |
| 429 | 推理繁忙，附 `Retry-After: 1` |
| 502 | 内部响应不符合公开协议 |
| 503 | 模型尚未安装、引擎不可用或已暂停新会话 |
| 504 | 请求超时，操作结果可能不确定 |

框架模式下创建会话返回 `model_not_installed`，不会伪造空检测或推理成功。
