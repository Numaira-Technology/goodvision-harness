# Continuous scene analysis and deployment

[Documentation index](../README.md) · [Repository overview](../../README.md)

本地 NVIDIA 部署**默认 1 路，使用 `--streams N` 按实际需要启动正整数路数**，
12 路是容量参考值，不是软件上限。3、6、12、16 路示例见 [本地 GPU 部署说明](../../docs/local-gpu.md)。
同一控制台可以分别连接和并排显示已部署的 N 路视频；连接只预览，分析通过“开始全部分析”
和“停止全部分析”统一控制所有已连接的视频。停止分析后预览继续，各路都可展开详细控制，
独立保存检测项目、敏感度并查看结果。分析运行和收尾期间锁定视频连接、各路设置与共享模型配置。
人物追踪使用本机 GPU 或
自备 GPU 服务器均可多路，大模型仍可用云端；原部署追踪模式保留单路限制。
主程序、人物追踪、大模型可以分别部署在不同机器，主程序无需显卡；控制台分别设置运行位置、
服务器地址和追踪容量，见 [模块化部署说明](../../docs/modular-services.md)。
同机部署仍支持单卡共用、双卡将大模型与追踪及其他 GPU 工作分开。
每路大模型分析目标沿用 1.25 次/秒，N 路共 1.25 × N
请求/秒，默认追踪共 2 × N 帧/秒；12 路参考目标为 15 请求/秒。
容量验收需要真实 L40S 上的持续压测，部署成功不代表已达到该吞吐。

新增持续人物／事件链路：`SCENE_ENGINE_MODE=active` 先通过 SAM 发现和关联人物，再以
多个时刻的画面检查事件，维护疑似、确认、遮挡、结束及修订。它已接入独立后台时序请求、
控制台、视频叠加、SSE、最近原始证据和事件元数据记录；选择陌生人时继续运行独立人脸识别。
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
[真实 GPU 验证记录](../../docs/architecture/sam31-gpu-validation-20260912.md)。
实测环境为驱动 560.35.03、CUDA 12.6、Ubuntu 24.04、Python 3.12.3，
torch 2.10.0／torchvision 0.25.0 使用 `cu126`；镜像可传入
`--build-arg TORCH_CUDA=cu126`，默认仍为 `cu128`。
[worker 部署说明](../../workers/sam_scene/README.md) 和
[SAM 3.1 实施记录](../../docs/architecture/sam31-implementation-20260912.md) 包含具体步骤。
这些验证证明链路可运行及历史回收一致；人物、事件准确率和数小时运行仍需专项验收。
持续视频还可选择 `SAM31_TRANSPORT_CODEC=h264`，将每次追加的最多 16 张源图
直接编码为独立 H.264 数据，保留原图与源帧清单。客户端安装 `.[video]`（PyAV
17.1.0），worker 开启 `SAM31_SESSION_H264_ENABLED=true`；双方确认能力后才发送。
此模式默认最长边 512、`SAM31_TRANSPORT_CRF=23`（允许 18–35），可通过
`SAM31_TRANSPORT_MAX_SIDE` 调整尺寸；JPEG 质量参数只作用于 JPEG 模式。
默认仍为 `jpeg`。H.264 每批独立、无 B 帧，接收端先检查码流的尺寸、图片数量和
解码缓冲限制再解码。该压缩模式需另行验证实际跟踪质量，不改变源时间或事件证据。
本次 4090 实测、人物合并、源画面对齐和 Demo 采用的 3 FPS 配置见
[SAM 3.1 优化与验证记录](../../docs/architecture/sam31-optimization-20260913.md)。
Deployment console 统一使用 `sam_live`（SAM 连续分析），已移除 Hybrid 产品入口。
这条路线首帧即可启动，先输出当前人物轮廓，再保留原生三次检测确认；
人物描边不等待事件模型，迟到事件可以更新同一人物的描边颜色与报警提示。
完整数据流、现有瓶颈与两秒以内的显示目标见
[SAM 连续分析流程与延迟方案](../../docs/architecture/sam-live-latency-20260916.md)。
后续优化按[三层架构与并行优化执行计划v4](../../docs/architecture/parallel-latency-design-20260916.md)推进：
显示优先流畅，在线事件识别争取两秒以内，人物与evidence整理可在run结束后后台完成。
首版已接入SAM三阶段流水、VLM跨批窗口和受控并发、Face独立推理与顺序提交、
私有源／事实记录及run后人物整理，详见[9月17日实施记录](../../docs/architecture/parallel-latency-implementation-20260917.md)。
9月17日联测失败：显示约12.9 FPS且零描边，VLM成功观察约0.64次／秒，并发生吸烟误报。
本地显示修复版已于9月17日部署并重新启用：修复旧状态响应误停新会话、请求阻塞与SAM采样时钟，采用较新的真实SAM轮廓配合短距离运动跟随；VLM时间窗修正尚未进入此镜像。
根因、证据范围与下一步见[本次故障诊断](../../docs/architecture/parallel-failure-diagnosis-20260917.md)。
原run近景回放已有锚点后15/16次完整描边；真实Chrome合成运动与实测SAM时间轨迹回放约19.98 FPS、回调到画面延迟P95约1.73秒。远景小人物仍有误跟随，真实摄像头并发联测尚待完成。详细证据及边界见
[轮廓跟随修复记录](../../docs/architecture/contour-following-repair-20260917.md)；吸烟误报按用户要求暂缓处理。
新路线尚无真实 GPU 的两秒达标结论；旧连续接口的历史实测不代表 `sam_live` 的速度。
已复核的[新计时版本实测](../../docs/architecture/sam-live-instrumented-diagnosis-20260916.md)显示
SAM 平均 1.70 FPS、人脸 1.30 FPS、VLM 0.33 次／秒；不同源描边画面仅 0.55 FPS。
当前阶段底线为 SAM、VLM、InsightFace 各自实际分析
至少 1 FPS，带完整描边的实际显示至少 15 个不同源帧／秒；三个分析分支各 3 FPS 仍是优化方向。
这些平均值不保证逐秒达标，不能靠丢关键帧或重复显示达标；本次修复也不代表VLM／Face与真实摄像头并发已经达标。
SAM连续路线现以私有spool保留已接纳样本，准备、顺序append和提交可以重叠；
“最多四帧”是单批上限，不再表示忙时覆盖／丢弃旧帧的待发队列。
容量不足明确暂停接纳或报告失败／待恢复，不能靠丢帧维持正常状态。
默认兼容显示仍受已完成SAM源帧限制；本地摄像头新增独立真实帧FIFO和轮廓运动传播实验，
该路径通过`LOCAL_CAMERA_DISPLAY_ENABLED=true`开启，通用默认仍关闭，当前本地部署已开启近景修复版；SAM使用已实测的1.1 FPS临时采样档。
GVC2上传等待匹配源ACK，
已选分析帧不被新帧覆盖；未确认任务、容量暂停和无轮廓帧分别记录。
显示层轮廓估计不写回真实分析与事件证据，真实15 FPS与轮廓质量仍待验收。
停止后先排空SAM、VLM、Face在线任务，再允许下一run；旧run的人物／证据后台整理可继续。
私有记录提供恢复材料，但完整后台整理在进程重启后自动续跑尚未贯通。
[持续事件架构与运行指南](../../docs/architecture/entity-event-engine.md) 包含本地 active 配置、
容器启动、录制视频验证、请求预算和具体边界。没有权重时可明确使用 `--recorded-tracks`
验证上层链路，报告会标识这不验证 SAM。

## Harness behavior

`disabled` 使用原有逐帧流程。`active` 停止旧逐帧安全请求，
使用最多四张、默认720最长边的时序输入，默认请求超时15秒。
`sam31_stream`首版按源时间跨批供给窗口，默认目标1.25个不同终点／秒、最多4个独立客户端在途，
乱序完成后有界顺序提交；其他active后端仍为单请求。重叠上下文不重复投票，保留原双证据确认；
事件状态由持续引擎驱动。空事件列表、遮挡与模型失败都不会显示全部安全。
