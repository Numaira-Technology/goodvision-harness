# SAM 连续分析：最近真实运行的延迟与吞吐审计

后续更新：本文保留 18:56 会话的历史审计。22:30 新计时版本的实测及明确瓶颈见 [新版本诊断](sam-live-instrumented-diagnosis-20260916.md)，不再以本文 1.08 FPS 代表最新运行。

日期：2026-09-16。只读取本机已经保存的日志，没有重新运行模型、启动服务或租用 GPU。**最近一次可确认的真实 SAM 会话平均完成约 1.08 帧／秒，仅略高于现阶段 1 FPS 底线；现有日志无法评估 VLM、InsightFace 各自的实际完成帧率，也无法测出浏览器实际显示帧率和摄像头到屏幕延迟。**

按用户最新要求，阶段底线同时包含：带完整人物描边的视频连续显示至少 15 个不同源帧／秒，**SAM、VLM、InsightFace 各自实际分析至少 1 FPS**，摄像头到屏幕延迟尽量低、争取两秒以内。三个分析分支均继续向各自 3 FPS 优化。允许参考已有补帧方案改善显示，显示层估计不能当成真实分析或事件证据。事件报警和描边变色可稍后更新。不能通过丢掉待分析关键帧或重复显示同一源帧来宣称达标。

## 撤回此前缺乏依据的调度修改

撤回“分析忙时仅保留最新一帧”的降延迟方案，也撤回“浏览器发送缓冲必须为空才采集”的新增限制。主机待发容量恢复此前的 4 帧，浏览器恢复此前 1 MB 发送缓冲准入规则。这只是撤回本轮收紧，**原来的有界队列溢出仍会丢弃等待帧，恢复后不能承诺零丢帧**。下一步必须同时记录到达、计划采样、入队、溢出、完成和失败数，改善吞吐与可靠接纳机制。

采样率是计划输入频率，完成率是实际产出；二者不能混用。已有的零默认播放等待只是去掉人为下限，也不证明低延迟或高吞吐。

## 使用哪一次运行

原始快照目录：`.goodvision/diagnostics/stranger-contour-20260916/`。

- 最近真实 SAM 会话：北京时间 **18:56:51.948–18:58:36.362**，对应 UTC **10:56:51.948–10:58:36.362**，时长 **104.413770 秒**。
- `sam-remote.log:1658` 为创建成功；`:1745` 为删除成功。共 29 次 append，全部返回 HTTP 200；29 条处理记录均有对应请求记录与访问日志。
- 第一批 1 帧，随后 28 批每批 4 帧，共 **113 个输入帧、113 个输出帧**；各批 `pending_frames=0`。首批处理记录在 `sam-remote.log:1654`，末批在 `:1741`。
- 18:49 的短会话属于探测，18:56:03–18:56:43 是前一次实际会话，均不混入上述统计。

日志 stdout/stderr 存在交错：例如第 1654 行的时间晚于第 1658 行。因此先按每行 Docker 时间戳排序，再截取会话区间，并按处理完成顺序对应 append 和 request 记录；不能直接按原始行号切片统计。29 对记录的最大时间差为 6.76 毫秒。

`.goodvision/diagnostics/latency-20260916/` 保存的是更早的 16:31:36–16:32:36 会话。已从其原始排序日志重算：24 批、92 帧，四帧批次 body 接收中位数 1.001276 秒、native 阶段 0.543247 秒、服务端总计 1.568586 秒，与该目录的 summary 一致。**它不是最近一次运行**，不能用这组较快数据替代下表。

## 最近运行的实际时间账单

以下只统计相同大小的 **28 个四帧批次**。P95 使用排序后线性插值；样本较少，不能代表长期尾延迟。各列独立计算，不能相加中位数或 P95 推导端到端统计。

| 已记录阶段 | 中位数 | P95 | 测量边界 |
| --- | ---: | ---: | --- |
| worker 接收请求体 | 1.508309 秒 | 2.341680 秒 | 进入请求计时后直到 body 读完，包含接收等待；不是纯网络传输时长 |
| 校验与解码 | 0.017230 秒 | 0.035042 秒 | 收到 payload 后的校验和解码 |
| SAM 原生处理与输出整理 | 0.838631 秒 | 0.939139 秒 | `native_output_seconds`，包含迭代模型结果及逐帧 mask／轮廓整理；不是 GPU kernel 时间 |
| 保留结果与构建返回对象 | 0.000370 秒 | 0.000584 秒 | `retention_response_seconds`，不含实际响应下载 |
| 请求体读完后的全部处理 | 0.865911 秒 | 0.957125 秒 | `processing_seconds`，还包含线程调度及未细分开销 |
| worker 请求计时总计 | **2.370958 秒** | **3.201179 秒** | `total_seconds`，不含主机排队、完整请求往返及浏览器显示 |
| 请求体大小 | 42,548 字节 | 57,301.75 字节 | 服务端实际收到的 body 字节数 |

接收请求体占四帧批次累计 worker 计时的 **64.22%**。这说明接收等待值得优先分段排查，但不能据此判定是地理距离、链路吞吐、丢包、SSH/TCP 行为、客户端发送节奏中的哪一项。

原始例子便于复核：

- `sam-remote.log:1661–1662`，UTC 10:56:59.373–10:56:59.376：4 帧，native 0.894802 秒，body 2.494745 秒，total 3.407609 秒。
- `sam-remote.log:1741–1742`，UTC 10:58:35.851–10:58:35.854：4 帧，decode 0.017122 秒，native 0.856166 秒，body 2.249491 秒，total 3.123595 秒。

连续 append 完成之间平均相隔 **3.601696 秒**。将每个完成间隔减去后一批 `total_seconds`，仍有平均 **1.198157 秒**没有被 worker 计时覆盖，范围 0.806757–2.143092 秒。它可能覆盖前一批响应回传、主机解析／提交／回调、下一批准备／发送、进入 worker 处理器之前的等待；日志输出本身也有微小缓冲。现有记录无法进一步归因，更不能把差额全叫“网络慢”。

## 吞吐：处理活跃时够快，不代表整条链路达标

| 口径 | 结果 | 解释 |
| --- | ---: | --- |
| 创建至删除整会话 | **113 / 104.413770 = 1.082233 帧／秒** | 含会话起止开销的实际输出吞吐 |
| 首次至末次 append 完成之间 | **112 / 100.845963 = 1.110605 帧／秒** | 去掉首批后，该时间内实际又完成 112 帧 |
| 仅累计 native 阶段活跃时间 | 4.767847 帧／秒 | 不含接收、请求间隙等，仅说明 native 阶段不能直接等同全链路瓶颈 |

第一条 native 完成时间为 UTC 10:56:55.005499（`sam-remote.log:1654`），最后为 10:58:35.851462（`:1741`）。上述按成功 append 的输入／输出帧数统计；源帧清单未写入旧日志，无法仅凭日志逐帧独立核验重复源帧、采样间隔及关键帧覆盖。新验收必须以 `(stream_session_id, source_frame_sequence)` 去重后的完成数为准。

1 FPS 的四帧批次需要长期平均至少每 **4 秒**完成一批；本次观测为约 **3.602 秒**，但最慢完成间隔为 **4.525 秒**，仍有明显停顿。全会话 1.08 FPS 只说明平均值略高于阶段底线，不证明各连续窗口均达标，更不证明实时延迟、分析质量或其他分支达标。后续 3 FPS 优化方向要求平均每 **1.333 秒**完成一批。需要同时定位请求接收和请求之间的开销，再决定传输、处理或协议的优化顺序。

`pending_frames=0` 只说明 worker 不等未来帧确认而滞留输出。它不代表主机无排队、不丢帧，也不代表完成了配置计划的采样量。旧日志没有记录本地到达数、计划采样数、准入数和丢弃数，无法从中恢复这些值。

## VLM 与 InsightFace：旧日志无法评估各自 1 FPS

同一会话时间段内，`harness.log` 共 992 行，其中 974 行为 HTTP 访问记录。其余 18 行是 9 条 InsightFace `face_align.py` 的 `FutureWarning` 与对应续行；例如 `:3333`、`:3348`、`:3426`、`:4251`。警告只说明相关代码曾被执行，不保证每次人脸分析都输出一条，不能把 9 条警告算成 9 个处理帧或用它推算帧率。

这份 harness 日志没有 VLM／`scene_model` 请求完成、模型输入源帧清单、InsightFace 完成计数或各自耗时的记录。保留的 `latest-run-events.json` 是 6 条陌生人事件状态快照，证据源时间跨度约 0.094–100.007 秒；`scene-events.sqlite3` 只有 `events` 表，共 37 条事件记录，没有逐推理调用流水。事件快照数量、累计阳性／阴性观察、HTTP `/scene` 访问次数都不能代替分支的实际分析帧数。陌生人状态存在也不能证明 VLM 在该会话按某个频率完成了分析。

因此 VLM ≥ 1 FPS、InsightFace ≥ 1 FPS 在历史记录上都属于**无法判定**。下一次应分别记录被准入、实际送模型和成功完成的源帧标识与时间。VLM 的 2–4 张时序上下文只形成一次观察，以不同的成功观察终点计完成率，输入张数与重复证据比例另列，不能把一次请求算成四个新分析观察。没有可见人脸的源帧也应记录实际执行检测的完成情况，不能只以识别到的人脸数作分母或分子。

代码还存在独立的调度边界：`src/goodvision_harness/scene_runtime.py:173` 创建 `InspectionPlanner(full_scene_interval_seconds=5.0, refresh_seconds=3.0)`，`scene_worker.py` 仅允许一个在途任务（`max_in_flight=1`）。VLM 当前按新人物、事件复核和全景巡检触发，本来不承诺每秒产生一个新观察；这些间隔也不是简单固定帧率，多个触发原因可交错。若阶段底线要求 VLM 持续完成至少一次新观察／秒，还需审查调度频率、请求预算和模型实际耗时。只调整摄像头采样或增加日志不能实现该目标；本轮仅记录现状，没有盲改事件调度策略。

## 显示至少 15 FPS：当前架构还有独立缺口

现有 SAM 主显示路径在已完成 tracking 的源帧中选择对应原图，使用同源 mask；事件迟到时还可能重绘同一张图。历史运行代码见 `runtime-scene_delivery.py:625–628`；当前对应实现是 `TrackedSceneDisplayWorker` 与 `SceneRuntime.tracked_display_frame()`。

因此，在人物采样上限 3 FPS 的配置下，当前主显示路径长期最多获得约 3 个不同的已分析源帧／秒，实效更低时显示也受限。**把线程上限或浏览器轮询改成 15 FPS，不能让这套仅显示 SAM 完成源帧的架构达到 15 个不同源帧／秒。** 批次完成时的一小段快速播放也不能代替持续帧率。

达到显示 15 FPS、三个分析分支各至少 1 FPS，需要额外完成架构设计与质量验证。可以参考已有补帧和几何插值流程，让高频真实显示源帧获得对应的轮廓估计，或进一步提高真实 SAM 完成频率。轮廓传播如果参与显示，必须明确是估计，保持人物编号、遮挡和出入画的正确性，不能写回成真实分析证据。不能把旧 mask 原样贴到新图，也不能仅靠无描边的原始预览宣称满足目标。

`harness.log` 只记录 `GET /v1/video/annotated/latest.jpg` 的状态，没有返回源序号、浏览器成功解码／绘制完成信息。一次 GET 可以取得重复图片，多页面还能重复请求，所以 HTTP 次数与成功状态都不能推导实际不同源帧率。`harness.log:3351` 在 UTC 10:56:55.258 返回标注图片，但无法据此恢复画面新鲜度或真实描边覆盖率。

## 运行健康及缺失证据

`sam-remote.log:1591` 记录 GPU readiness 检查通过；本次 29 个 append 全部成功，未发现会话期间的推理异常。日志未保留 GPU 利用率、显存压力、时钟、逐 kernel 时间，不能据此认定 GPU 饱和或需要换卡。

`harness.log:2241` 的 `connection_failed` 出现在 UTC 10:51:11，早于本次会话；`:4343` 的 graceful shutdown 超时发生在 UTC 10:59:01，晚于会话结束。二者都不能直接解释中间的稳定延迟。开始和停止附近的 latest.jpg 404 也不是一次完整的绘制延迟测量。

更早诊断保存的 `buffer_delay_seconds=8` 属于 16:32 附近的设置快照；最新会话前存在配置更新访问记录（`harness.log:2627`，UTC 10:55:50.669），但没有请求体。因此不能把 8 秒当作 18:56 会话已确认的播放等待配置。最新日志同样没有 `source_to_result_seconds`、本地队列等待、历史显示年龄、编码耗时或浏览器绘制耗时。

## 下一轮必须保留的测量与验收

先补观测，再依据结果优化。日志应只写时间、计数、大小、会话关联标识和源序号，避免图像、人物姓名、访问凭据及完整远端 URL。

1. 采集与采样：源帧到达时间、源时间／序号、计划采样数、实际准入数、拒绝／溢出原因、队列深度及等待；分别验证 SAM、VLM、InsightFace 的实际完成率，不得通过丢关键帧换低延迟。
2. 每批传输：关联同一批源帧，记录编码耗时、大小、发送开始、请求往返、响应大小／读取、解析、回调／提交。worker 补 body 分段接收、解码、模型迭代等待、mask／轮廓和结果构建的分项。
3. 显示：记录新源帧发布、同帧事件重绘、源到发布年龄、渲染耗时、配置中的播放等待，以及浏览器成功绘制的源序号与本地完成时间。分开统计发布帧率与实际显示帧率。
4. 联测报告：实时采集期间按各自成功完成的不同源帧统计 SAM、InsightFace 各 ≥ 1 FPS；VLM 按不同的成功时序观察终点统计 ≥ 1 次／秒，不能将上下文图片数当完成观察数。记录三个分析分支距各自 3 FPS 优化方向的差距；浏览器显示率 ≥ 15 FPS，另报摄像头到屏幕 P50/P95/峰值、冻结、丢失与关键事件覆盖。显示补帧／轮廓估计与真实分析帧分开计数，估计不得计入分析完成率。采集停止后的追赶产出单列，失败与冻结不能从样本中删掉。两秒是争取达到的端到端目标，目前没有达标证据。

## 新计时日志的保存和使用

以下是本轮新增的观测能力，不是新的真实 GPU 实测。客户端、调度、显示和分析分支的结构化计时通过现有 `uvicorn.error` 通道以 INFO 输出。**本地启动时必须保存 stderr，不能只保留 stdout；容器运行时同时保存带时间戳的 harness 与 worker 日志。** 可使用现有 `docker logs`／`docker compose logs` 或启动器的标准输出、标准错误文件，不需要额外启动诊断服务。保留原始顺序与时间戳，分析时再排序；访问日志、计时日志和失败记录都应保留。

| 记录 | 关键字段 | 用途与边界 |
| --- | --- | --- |
| `sam_client_http_timing` | `request_id`、`route`、首末源序号／源时间、输入帧数、请求／响应大小；`payload_prepare_seconds`、`http_until_headers_seconds`、`response_read_seconds`、`response_json_seconds`、`result_validation_seconds`、`total_seconds` | 区分准备编码、收到响应头之前、读响应、解析和验证；失败保留 `phase`、`outcome` 和错误类型 |
| `sam_coordinator_dispatch` | `input_sequences`、`queue_oldest_seconds`、`pending_frames`；`session_source_frames`、`session_admitted_frames`、`session_sampled_out`、`session_dropped_backpressure` | 记录派发时等待最长样本的队列年龄及入口／采样／溢出计数；派发不等于完成 |
| `sam_coordinator_timing` | `request_id`、输入／输出源序号；`session_setup_seconds`、`append_seconds`、`commit_seconds`、`completion_interval_seconds`、逐帧 `source_to_result`；`pending_frames_at_completion`、`during_capture`、`outcome` | 关联会话初始化、客户端请求、主机提交和真实完成节奏；源到结果从主机接纳样本起算，不含曝光到主机或浏览器显示 |
| `sam_queue_overflow` | `dropped_sequences`、`admitted_sequence` 及会话关联信息 | 记录原有过载策略实际丢掉的源帧，不能忽略或算作成功；日志没有消除丢弃行为 |
| `sam_display_timing` | `source_frame_sequence`、`source_timestamp_seconds`、`source_to_publish_seconds`、`snapshot_seconds`、`render_seconds`、`publish_seconds`、`late_event_repaint`、`distinct_source_frames`、`same_source_repaints`、`distinct_source_fps` | 区分服务端发布新源帧与同源事件重绘；`distinct_source_fps` 是发布率，**不是浏览器成功绘制率** |
| `analysis_branch_timing` | `branch`、会话／源帧序号与时间、`completed_requests`、`completed_observations`、`success`、`model_seconds`、`error_type` | `branch=insightface` 记录已完成的人脸分支调用；只将成功且不同的源帧计入完成率。错误或无有效结果不算成功，异常时耗时可能为 null，不能当作零耗时 |

worker 追加记录新增 `validation_seconds`、`decode_seconds`、`native_advance_seconds`、`snapshot_seconds`、`retention_seconds`、`response_build_seconds`、`response_serialize_seconds` 和 `cleanup_seconds`；HTTP 记录补充 `worker_dispatch_seconds`、`http_response_encode_seconds` 等，完整定义见 [worker 日志说明](../../workers/sam_scene/README.md)。原粗粒度字段继续保留，可与历史日志比较。`native_advance_seconds` 和 `snapshot_seconds` 是现有执行路径的主机墙钟计时，没有新增 CUDA 同步，不能解释为纯 GPU kernel 时间。

按以下顺序读日志：

1. 使用同一个有效 `request_id` 配对客户端、coordinator 完成记录和 worker 的 append／HTTP 记录，核对 `route` 与源序号范围。派发记录尚无响应 ID 时用会话、generation 和输入源序号关联。失败或尚未通过校验的请求可能没有 ID，应保留为未配对失败，不能强行按邻近行拼接。worker 的 replay 不能重复计入新产出。
2. 先看 `queue_oldest_seconds`、`session_setup_seconds` 与准备编码，再看 HTTP 和响应／提交。客户端 `http_until_headers_seconds` 已包含 httpx 序列化、上传、worker 执行及响应头返回；它与 worker 阶段重叠，**不是纯网络时间**。worker `processing_seconds` 与 append 分项也重叠，不能全部相加。两端时间戳不能直接相减当单向链路延迟；时钟未校准时更不能这样推导。
3. 对每个输出源帧统计 `source_to_result`，对完成间隔看批次节奏；结合 overflow 明细与累计源／采样计数核对覆盖率。coordinator 的 `completed_analysis_fps` 使用 `completed_during_capture / capture_elapsed_seconds`，采集结束后分母停止增长、后续结果进入 `completed_after_capture`，不会用 EOF 追赶美化实时吞吐。另报滚动窗口和最长无完成间隔，平均值不能隐藏冻结。
4. 将显示记录按会话和源序号去重；`same_source_repaints` 只反映同帧更新事件颜色，不能增加不同源 FPS。浏览器真正完成绘制的计数、解码延迟及摄像头曝光前段仍需联测，服务端发布日志本身不能证明 15 FPS 或两秒达标。

VLM 与 InsightFace 的计时沿用同一 INFO／stderr 保存方式，不能只保存 SAM。`vlm_inspection_timing` 记录时序上下文源帧、观察终点、模型调用次数以及准备、模型、判定校验、提交／回调耗时；一个成功时序请求只算一次观察，不能按 `context_frames` 张数放大。`analysis_branch_timing` 已记录 InsightFace 实际成功完成源帧、模型耗时与 success／error；其中 `branch=vlm_frame` 属于旧逐帧模型分支，应与时序 VLM 分开分析。这些完成记录**尚不覆盖完整的采集、准入、拒绝和忙碌流水**，不能独自证明零丢帧或全部输入覆盖。旧快照缺少这些记录，不能回填历史 VLM 或人脸帧率。

本报告提供可复核的旧运行基线。新增计时仍需下一次真实运行产生日志；已有日志不能补造未记录的端到端阶段。
