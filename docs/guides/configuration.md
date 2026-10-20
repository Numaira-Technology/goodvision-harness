# Configuration and inference routes

[Documentation index](../README.md) · [Repository overview](../../README.md)

Commands and unlinked file paths in this guide are relative to the repository root.

## Configuration

`display.max_image_side: null` preserves each decoded network frame's dimensions
after camera-view selection. A 2560×720 stereo frame is first cropped to one
1280×720 eye, then independently encoded for display and inference; it is never
downscaled before cropping. A 1920×1080 monocular source displays at 1920×1080.
Set a positive side limit (for example, `1280`) to reduce display bandwidth;
smaller sources are never enlarged. Browser-camera output preserves the uploaded
frame dimensions, which remain bounded by the browser capture settings.
Display JPEGs and annotation re-encoding use quality 90. Matching resolution
does not make JPEG output lossless or match the source frame rate: display
cadence targets 30 FPS (limited by available source frames), and higher
resolutions cost more bandwidth, encoding time, and buffer memory. Event archive clips retain their separate `events`
resolution limit.

`config.yaml` controls sampling, model selection, the legacy single-event
prompt, the fixed-four `multi_event_prompt`, concurrency, readiness windows,
the `display.delay_seconds` / `display.bbox_max_gap_seconds` compositor timing,
overlay JPEG quality, and optional webhook behavior. Inference routes are
configured under `inference_service`. The API retry count remains zero so a
failed live request is not replayed as a stale result.
Persisted console settings override environment values; otherwise the generic
stream variable takes precedence over the legacy RTSP variable:

| Variable | Meaning |
| --- | --- |
| `HARNESS_STREAM_URL` | Primary RTSP/RTMP/WHEP input URL. |
| `RTSP_URL` | Deprecated compatibility fallback when `HARNESS_STREAM_URL` is empty. |
| `INFERENCE_PROVIDER` | Route: `openrouter` (default) or `dashscope`. |
| `INFERENCE_MODEL` | Optional model override for the selected route. Empty uses its configured default. |
| `INFERENCE_ENDPOINT` | Optional complete chat-completions URL override for the selected route. |
| `INFERENCE_API_KEY` | Optional shared credential override, ahead of provider-specific environment keys. |
| `OPENROUTER_API_KEY` | OpenRouter credential, used only when that route is selected. |
| `DASHSCOPE_API_KEY` | DashScope credential, used only when that route is selected. |
| `HARNESS_DRY_RUN` | Disable external inference and emit deterministic results. |
| `HARNESS_AUTOSTART` | Start the configured network stream when the service starts; set false for local camera testing. |
| `HARNESS_SOURCE_ID` | Stable customer-facing source identifier. |
| `RESULT_WEBHOOK_URL` | Optional result-push destination. |
| `HARNESS_WEBHOOK_SECRET` | Signed webhooks are disabled in this branch. A nonempty secret with a webhook destination causes an explicit error before delivery. Disable that webhook configuration; do not clear its secret to bypass the refusal. |
| `HARNESS_BIND_ADDRESS` | Host bind address used by Compose. |
| `HARNESS_PORT` | Host HTTP port used by Compose. |
| `FACE_MODEL_ROOT` | Root containing operator-supplied face-analysis files. |
| `FACE_GALLERY_PATH` | Persistent known-person gallery JSON path. |
| `FACE_PROVIDER` | InsightFace ONNX provider: CPU, DirectML, or CUDA. |
| `FACE_LIVE_SCHEDULER` | Legacy scheduler for integrations without the ordered face pool; `auto` selects CPU single-flight and accelerated full sampling. |
| `FACE_INFERENCE_INSTANCES` | Independent local face models; default 2, allowed 1–4. Benchmark CPU/GPU contention before raising. |
| `FACE_INFERENCE_THREADS` | Per-instance inference threads; default 2, allowed 1–16. |
| `SCENE_MAX_IN_FLIGHT` | Continuous temporal VLM client slots; default 4, allowed 1–4. |
| `SCENE_OBSERVATION_FPS` | Continuous VLM distinct-source endpoint target; default 1.25, maximum 3. Requires sufficient real SAM observations. |
| `LOCAL_CAMERA_DISPLAY_ENABLED` | Opt-in local-camera FIFO and motion-contour experiment; default false. |
| `GOODVISION_SAM_SPOOL_DIRECTORY` | Private durable SAM task directory; default `.goodvision/sam-spool`. |
| `RICH_LABEL_ENABLED` | Enable the optional rich-label worker path. |
| `RICH_LABEL_ENDPOINT` | Rich-label worker inference URL. |

Do not commit `.env`, RTSP credentials, or LLM API keys.

### 切换 AI 渠道和模型

AIClub 路由已移除，源码默认使用 OpenRouter。旧的 `provider: aiclub`、
`INFERENCE_PROVIDER=aiclub` 或 AIClub 地址会被拒绝，须修改配置后重启。
旧 AIClub 密钥记录不会迁移到 OpenRouter；已有 OpenRouter 密钥存储保持不变。

以下路由选择说明适用于当前源码开发运行及 `development` 镜像。
完整交付包按 [DEPLOY.md](../../DEPLOY.md) 部署，控制台可选择并保存当前 A–H 八个模型。
本节的任意渠道、模型及地址环境覆盖不适用于该交付包。
已导出的 `.2` 及更早历史镜像不会因源码更新而增加此能力；使用这些历史版本时，
应遵循其随包原说明和配置，不能只通过添加环境变量切换为新实现。

把各渠道的密钥保存在本机未纳入 Git 的 `.env` 中。切换时只需修改
`INFERENCE_PROVIDER`，再重启服务：

```dotenv
INFERENCE_PROVIDER=openrouter
OPENROUTER_API_KEY=填入该渠道的密钥
DASHSCOPE_API_KEY=
INFERENCE_MODEL=
INFERENCE_ENDPOINT=
INFERENCE_API_KEY=
```

| `INFERENCE_PROVIDER` | 默认模型 | 使用的渠道密钥 |
| --- | --- | --- |
| `openrouter` | `qwen/qwen3.5-35b-a3b:nitro` | `OPENROUTER_API_KEY` |
| `dashscope` | `qwen3.5-plus` | `DASHSCOPE_API_KEY` |

例如，将 `INFERENCE_PROVIDER=dashscope` 即可改走 DashScope。测试其他模型时，
设置 `INFERENCE_MODEL` 为该渠道支持的模型 ID；留空则恢复该渠道的默认模型。
两个渠道的默认模型并不完全相同。比较线路速度时，应选择各渠道都支持的同一模型，
并保持输入、提示词和请求参数一致。可用模型见
[OpenRouter 模型页](https://openrouter.ai/qwen/qwen3.5-35b-a3b)。
若要使用自定义网关、区域或工作空间地址，可设置完整 URL；DashScope 地址格式见
[官方兼容接口文档](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)：

```dotenv
INFERENCE_ENDPOINT=https://gateway.example/v1/chat/completions
```

切换渠道时，清空不再需要的 `INFERENCE_MODEL`、`INFERENCE_ENDPOINT` 和
`INFERENCE_API_KEY` 覆盖值，避免沿用上一次测试的设置。部署控制台保存的密钥优先于
环境变量，且只用于保存时对应的渠道；使用独立渠道环境密钥时，保持
`INFERENCE_API_KEY` 为空。

源码 Compose 部署修改 `.env` 后执行 `docker compose up -d --force-recreate harness`。
使用已包含该功能的新交付镜像时，选择对应的交付 Compose 文件再重建主服务容器；
不要将仍固定历史镜像标签的启动文件误当成新版本升级入口。本地运行则重启 Harness 进程。

也可以在 `config.yaml` 的 `inference_service.provider` 中选择默认渠道。
各渠道的模型、完整 `endpoint`、`api_key_env` 和专用请求参数放在
`inference_service.providers.<渠道>` 下，例如：

```yaml
inference_service:
  provider: openrouter
  providers:
    openrouter:
      model: qwen/qwen3.5-35b-a3b:nitro
      request_parameters:
        reasoning:
          enabled: false
    dashscope:
      model: qwen3.5-plus
      request_parameters:
        enable_thinking: false
```

`INFERENCE_PROVIDER`、`INFERENCE_MODEL` 和 `INFERENCE_ENDPOINT` 环境变量优先于
配置文件。`inference.request_parameters` 保存各渠道通用参数（如温度和输出长度）；
选中渠道的 `request_parameters` 覆盖同名通用参数，其余渠道的专用参数不会发送。

## Face and reference memory

The ordered local face pool stores source images until successful identity commit
(failed tasks retain them), plus embeddings and observation facts for later review.
These files live under the gallery directory's private `private-face` folder and
are excluded from public results. Storage limits and process-restart recovery
boundaries are described in the [事件、归档与收尾验证](../validation/events-and-lifecycle.md).

The three memory components have separate roles:

| Component | Role |
| --- | --- |
| `FaceGallery` | Primary local identity memory; persists enrolled embeddings and matches known/unknown faces. |
| Reference memory | Optional remote-model input prefix; loads up to eight private images and rejects returned memory IDs that were not supplied. |
| `VisualMemorySink` | Non-blocking output integration; receives a `VisualMemoryRecord` containing a completed observation and its exactly correlated JPEG. It does not participate in recognition. |

The result envelope distinguishes event and face observations with
`analysis_kind`; `observation_id` aliases `message_id`, and
`objects[].attributes` carries mode-specific metadata. Gallery setup is in
[CONTRIBUTING.md](../../CONTRIBUTING.md#known-person-gallery); identity continuity
and display behavior are defined in [event evidence](../architecture/event-evidence.md#people-and-continuity).

Local face matching and the remote face-reference experiment use the same
stranger meaning: `detected=true` means an unknown live face is present.
`matched_memory_ids` is additional reference metadata:

| Live frame | `detected` | `matched_memory_ids` |
| --- | --- | --- |
| No face | `false` | `[]` |
| Known face only | `false` | Known IDs |
| Unknown face only | `true` | `[]` |
| Known and unknown faces | `true` | Known IDs |

Fixed-five and face-diagnostics modes use local face analysis by default.
Enable remote reference memory only as an explicit fallback/experiment with
`memory.enabled: true`, entries under `memory.references`, and
`inference.event_label: unknown_person`. The built-in face-reference purpose
also enforces that label. A memory-aware result includes validated IDs in
`judgments[].matched_memory_ids` and uses `analysis_kind=face_reference_match`
for the built-in face purpose.

Relative reference paths resolve from the selected config file. Docker mounts
the ignored local `memory/` directory at `/app/memory`; local runs can pass
`--config config.yaml` to use the same top-level directory. Enabling reference
memory sends resized private reference images to the configured provider on
every inference request, so keep it disabled unless that data flow is allowed.

The old process-local upload store is not duplicated: the main face path already
has persistent enroll/list/delete APIs and UI, while remote references remain
explicit deployment configuration.
