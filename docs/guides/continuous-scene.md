# Continuous scene analysis and deployment

[Documentation index](../README.md) · [Repository overview](../../README.md)

本地 NVIDIA 部署**默认 1 路，使用 `--streams N` 按实际需要启动正整数路数**，
12 路是容量参考值，不是软件上限。3、6、12、16 路示例见 [本地 GPU 部署说明](local-gpu.md)。
同一控制台可以分别连接和并排显示已部署的 N 路视频；连接只预览，分析通过“开始全部分析”
和“停止全部分析”统一控制所有已连接的视频。停止分析后预览继续，各路都可展开详细控制，
独立保存检测项目、敏感度并查看结果。分析运行和收尾期间锁定视频连接、各路设置与共享模型配置。
人物追踪使用本机 GPU 或
自备 GPU 服务器均可多路，大模型仍可用云端；原部署追踪模式保留单路限制。
主程序、人物追踪、大模型可以分别部署在不同机器，主程序无需显卡；控制台分别设置运行位置、
服务器地址和追踪容量，见 [模块化部署说明](modular-services.md)。
同机部署仍支持单卡共用、双卡将大模型与追踪及其他 GPU 工作分开。
每路大模型分析目标沿用 1.25 次/秒，N 路共 1.25 × N
请求/秒，默认追踪共 2 × N 帧/秒；12 路参考目标为 15 请求/秒。
容量验收需要真实 L40S 上的持续压测，部署成功不代表已达到该吞吐。

持续人物／事件链路通过 `SCENE_ENGINE_MODE=active` 启用。SAM 负责发现和关联人物，
默认独立 VLM 从原图构建时序窗口并发布事件判断，之后再做人物关联；选择陌生人时运行独立人脸识别。
各分支接入控制台、视频叠加、SSE、原始证据和事件元数据记录。
默认 `disabled` 保留原有流程；`active` 启用持续人物与时序事件链路。
`shadow` 旁路模式已移除；旧 `SCENE_ENGINE_ENABLED=true` 配置会报错，请改用 `SCENE_ENGINE_MODE=active`。

持续场景视频优先描出 SAM 在对应源帧生成的人物 mask 轮廓；没有同源 mask 的
跟踪观察使用人物框。事件首次阳性就将对应实体的描边／框标黄，并在左上角显示事件。
迟到判断更新当前事件提示，不把旧轮廓移到新画面。同一实体、同一事件需要累计
10 个不同源帧的有效 clear 判定才结束，新的阳性重置计数；缺失、遮挡和模型失败不算 clear。
一个时序请求只贡献一个实际观察终点的判定，不按其中输入图片的数量重复计数。

图像关键帧检测与运动跟踪使用 [compose.scene.yaml](../../compose.scene.yaml)；
[compose.sam31.yaml](../../compose.sam31.yaml) 提供每批独立记忆的 SAM 3.1 视频对照。
再叠加 [compose.sam31-stream.yaml](../../compose.sam31-stream.yaml) 可启用 `sam31_stream`：
在同一个有界会话中追加已经收到的新帧，保持原生人物编号并回收过期模型历史。
首批 16 帧、默认 14 帧确认缓冲；会话达到 65,536 帧或 2,048 个累计原生 ID 时需要重建。
租用 RTX 4090 已完成严格权重加载、CUDA 预热和真实录像运行；48 帧追加以及
128／256 帧历史回收的原生 ID 与 mask hash 对照全等，详见
[追踪性能与验收边界](../validation/tracking-performance.md)。
实测环境为驱动 560.35.03、CUDA 12.6、Ubuntu 24.04、Python 3.12.3，
torch 2.10.0／torchvision 0.25.0 使用 `cu126`；镜像可传入
`--build-arg TORCH_CUDA=cu126`，默认仍为 `cu128`。
[worker 部署说明](../../workers/sam_scene/README.md) 和
[SAM 3.1 实施记录](../architecture/sam31-implementation-20260912.md) 包含具体步骤。
这些验证证明链路可运行及历史回收一致；人物、事件准确率和数小时运行仍需专项验收。
持续视频还可选择 `SAM31_TRANSPORT_CODEC=h264`，将每次追加的最多 16 张源图
直接编码为独立 H.264 数据，保留原图与源帧清单。客户端安装 `.[video]`（PyAV
17.1.0），worker 开启 `SAM31_SESSION_H264_ENABLED=true`；双方确认能力后才发送。
此模式默认最长边 512、`SAM31_TRANSPORT_CRF=23`（允许 18–35），可通过
`SAM31_TRANSPORT_MAX_SIDE` 调整尺寸；JPEG 质量参数只作用于 JPEG 模式。
默认仍为 `jpeg`。H.264 每批独立、无 B 帧，接收端先检查码流的尺寸、图片数量和
解码缓冲限制再解码。该压缩模式需另行验证实际跟踪质量，不改变源时间或事件证据。
传输压缩与历史回收的实测边界见[追踪性能](../validation/tracking-performance.md)。
控制台统一使用连续人物追踪，已移除 Hybrid 产品入口；首帧可启动，原生历史与检测确认仍在同一会话中推进。

## 当前链路与验收入口

显示、事件判断、人物及证据整理按[三层职责](../architecture/parallel-latency-design-20260916.md)分开。
独立 VLM 默认直接采样原图并发布报告，后续人物关联不应阻塞事件输出。
播放缓冲和分析同时进行；配置的 FPS 与等待时间不是实际吞吐承诺。

SAM 默认使用持久有序队列保留已接纳样本，准备、顺序追加与提交可重叠。
可选实时候选策略仍需质量对照，配置和回退见[队列验收指南](contour-realtime-validation.md)。
显示层估计不得写回真实分析证据；停止时需完成在线任务收尾，完整后台人物整理的进程重启恢复仍有边界。

- [网络与显示](../validation/network-and-display.md)：接收隔离有效，主进程输出在计算压力下仍会降帧。
- [事件与收尾](../validation/events-and-lifecycle.md)：独立输出、身份纠正、归档延迟及生命周期回归。
- [追踪性能](../validation/tracking-performance.md)：真实 L40S 基线、队列取舍与容量缺口。
- [识别质量](../validation/recognition-quality.md)：真实模型回复与尚未证明的准确率范围。

[持续事件架构与运行指南](../architecture/entity-event-engine.md) 包含本地 active 配置、
容器启动、录制视频验证、请求预算和具体边界。没有权重时可明确使用 `--recorded-tracks`
验证上层链路，报告会标识这不验证 SAM。

## Harness behavior

`disabled` 使用原有逐帧流程。`active` 默认启用独立 VLM 时序链路，
`SCENE_INDEPENDENT_VLM=false` 可恢复旧场景内调度。已接纳请求按源顺序提交，
重叠上下文不重复投票。空事件列表、遮挡、缺失标签与模型失败均不能被视为全部安全。
事件观察和确认结果、定位与标签完整性分别报告，详见[事件与收尾](../validation/events-and-lifecycle.md)。
