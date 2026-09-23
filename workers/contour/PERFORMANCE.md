# Contour 多路视频显存与延迟排查

> 2026-09-20 后续调整：多路 worker 已增加有界 CPU 预解码，不再让解码占用共享
> GPU 槽。每个模型进程最多持有两个准备/处理中的请求；GPU 推理仍串行。
> 新日志口径、部署和回退方法见
> [实时链路验收指南](../../docs/guides/contour-realtime-validation.md)。
> **下文的 L40S 实测均为此前版本，不包含此次预解码改动的性能收益。**

2026-09-18。审查基线：`codex/entity-event-engine` 的
`bf614ae14720923bcec68db4354a6d2398d33633`；修复分支：
`codex/contour-gpu-efficiency`。上游 SAM 固定版本仍为
`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`。

**已找到 NAS 上的正式权重，并在 105 的 L40S 上完成真实模型的 1／2／6 路对照。**
本次修改确实减少显存，但未显著提高原生吞吐。每卡六路在本片上的饱和上限约
1.22 FPS／路，因此若要求两张卡支撑 12 路、每路 3 FPS，当前逐帧全量 SAM 路径不足。
尚无 5060 Ti 现场配置和日志，不能把它的 4／8 秒直接归因为单次 GPU 推理耗时。

## 正式权重原生推理结果

原文件位于：
`/nas2/edwin/hf_cache/models--facebook--sam3.1/snapshots/daa63191845a41281374e725f4c9e51c7a824460/sam3.1_multiplex.pt`。
已实际读完整文件核对 SHA-256，与固定版本完全一致；已有完整缓存使此前的下载 403
不再阻塞测试，没有改动 NAS 缓存或使用其他账号的凭据。测试副本在
`105:/home/edwin/projects/goodvision-contour-20260918/models/contour.pt`。

使用公开视频 OpenCV `vtest.avi` 前 96 个顺序样本，3 FPS 采样，输入缩至 512×384；
每路独立状态、同一片段，最高每帧 9 个输出目标。逐轮交错执行，超过 32 帧历史窗口后
计算饱和吞吐。以下显存为本进程 CUDA allocator 的峰值活跃张量，不含其他任务，
也不是整个设备的 `nvidia-smi` 数值。

- **1 路**：原分支 7.19 FPS、5.62 GiB；最终补丁 7.21 FPS、4.26 GiB。
- **2 路**：原分支合计 7.35 FPS、6.20 GiB；最终补丁合计 7.33 FPS、4.87 GiB。
- **6 路**：原分支合计 7.35 FPS、8.64 GiB；最终补丁合计 7.35 FPS、7.32 GiB。

两路峰值活跃张量减少约 **1.34 GiB（21.5%）**；两路 allocator 峰值保留显存
从 6.80 降至 5.51 GiB。六路峰值保留显存从 9.48 降至 8.06 GiB。
运行库、CUDA context 和现场其他进程仍需额外显存；本次没有 5060 Ti 实机验证。

**最终补丁与未修改原分支，共 864 个输出帧的 ID、mask hash、概率逐项完全一致。**
这支持本片上的数值等价，不是有真值的召回率或数小时稳定性验收。
固定 person 特征缓存已在普通 GPU Contour 配置中默认开启，保留 false 回退开关。
没有改权重精度、视觉分辨率、历史长度、帧丢弃规则或原有检测阈值。

最终镜像 `goodvision-contour-bench:20260918-final`，ID
`sha256:c2e415521baa9cec5a6b7708f8b4ba8ec64f5e4e3590497dd9d953902fca4c2a`。
原分支通过只读挂载原始文件运行，同一个 probe 仅补充 allocator 读数，未修改模型路径。
早期探索版的缓存 A/B 也完全一致；改变单帧启动语义的实验发生输出差异，已从最终补丁移除。

每批四帧的探索结果为总吞吐 7.47–7.54 FPS，单批 P50 约 0.52 秒，仍无法满足 18 FPS。
但首批四帧与首批一帧会触发不同的上游检测模式，故不能把两组差值完全归因于批量效率。
没有据此修改客户端批次默认值。

## 完整 Contour 网关的实时到达回放

使用最终镜像，实际 HTTP client → Contour 网关 → 六路 `sam_live` → 原生模型 →
轮廓 JSON。六路同时按时钟提供同一段公开视频，各跑 48 秒；JPEG 缩至 512，质量 55，
编码在计时内。每路一个在途请求加四帧等待，队列满时明确丢弃新采样帧。
测量从预定采样时刻到轮廓响应，不含完整 Harness 调度器、RTSP 摄像头、浏览器、
人脸或事件模型；现场客户端采用不同丢帧策略时，延迟分布会不同。

- **六路各 1 FPS**：288 帧全部完成，**零丢帧**，每路采集期间都完成 1 FPS。
  全部响应延迟 P50 **0.586 秒**、P95 **1.000 秒**，最慢单路 P95 **1.113 秒**，
  全部最大值 **1.184 秒**，没有 EOF 之后的额外追赶。
- **六路各 3 FPS**：提交采样 864 帧，完成 **344 帧**，丢弃 **520 帧（60.2%）**。
  采集期间每路仅完成约 **1.08–1.10 FPS**。全部已完成响应的延迟 P50 **4.411 秒**、
  P95 **4.776 秒**、最大 **5.024 秒**；EOF 后另追赶 **4.290 秒**。

这说明仅有限队列、仅减少显存，不能把超载系统变为低延迟系统。被丢弃的帧没有推理结果，
不能把 344 个成功响应解释成 3 FPS 全帧达标；这些延迟也不是单次模型推理耗时。
此实验能解释吞吐不足如何形成数秒积压，但不证明现场 5060 Ti 的 4／8 秒具有完全相同来源。

实际建议：先按每卡六路、每路 **1 FPS 分析采样**验收，保留原视频显示帧率；
观察更长时间和业务人数峰值。该样本余量有限，尚未达到保留 30% 波动空间的容量设计。
若业务明确需要 2–3 FPS／路，应改变逐帧执行完整 SAM 的方案，或增加经过实测的算力，
不能把 `CONTOUR_CAPACITY=12` 当成扩容优化。双卡同时服务 12 路尚未实测。

## 测试环境与证据

以上正式模型实验使用共享 NVIDIA L40S，驱动 560.28.03、Python 3.12、PyTorch 2.10.0+cu128、CUDA runtime 12.8；容器 CPU 配额 8 核，PyTorch/OMP/MKL 线程数 4。未停止其他任务，因此不是独占 GPU 基准。

固定模型、输入、最终源码指纹、正式汇总及 HTTP 回放工具见[追踪验证证据](../../docs/validation/tracking-performance.md)。原始结果位于本仓库 `tmp/l40s-20260918/results/` 和远端 `/home/edwin/projects/goodvision-contour-20260918/results/`。

取得正式权重前的随机参数编码器实验与文本缓存夹具已退出验收依据，其逐次计时、无效初始化记录及临时探针已删除。缓存是否有效以正式模型的 mask/ID 对照和净显存测量为准。

## 已确认的部署路径

`compose.gpu.yaml` → 单个 Uvicorn worker → `sam_live` →
`SamLiveContinuation` → 原生 SAM 3.1 detector + multiplex tracker。
`CONTOUR_CAPACITY>1` 启用 `CameraFleet`：所有通道共享模型权重，但各自持有视频状态，
共用一个 FIFO GPU 执行槽。`CONTOUR_CAPACITY` 是准入数量，不是吞吐保证。

当前已经使用 `eval()`、`inference_mode()`、BF16 autocast、关闭批量 grounding、
逐帧后处理、16 帧输入环和默认 32 帧跟踪历史。不存在“每路视频各加载一份 SAM”的
代码默认行为。若现场启动了多个容器或多个 worker，则需另查；镜像标签 `0.1.0`
本身不能证明具体代码版本。

SAM 3.1 的 Object Multiplex 合并的是同一视频的多个对象，不能把 `multiplex_count=16`
解读成同时处理 16 个摄像头。上游发布说明中的 H100/128 对象加速数字也不是这里的
跨摄像头容量。[上游说明](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/RELEASE_SAM3p1.md)。

## 慢在哪里：确定问题与待测假设

### 全局 GPU 槽覆盖上传等待：本次已修复

原 `register_session_routes.handle()` 在读取请求体之前申请全局 GPU 槽。
因此一路上传、等待网络分块时，另一路即使准备好数据也无法开始推理。
这段等待增加串行服务时间，但不会提高 GPU 利用率。

多路接口现改为：每通道最多一个有界请求接收数据 → 完整请求进入 FIFO GPU 队列 →
串行推理。单通道重复请求返回 429，不能越过正在上传的请求。取消、超时和请求体超限
均释放自己的上传槽；未进入推理的失败不破坏原会话，也不释放其他任务的 GPU 槽。
已经开始的原生操作仍持有 GPU 槽直到真正退出。旧独占接口行为保留。

**收益取决于 engine 实际的请求体接收时间。** Contour 网关已在每通道内收完外部上传
再转发给 engine；两者同机时内部传输可能很快。此修复不能被当成已解释全部 8 秒。
CPU 解码、原生执行和轮廓转换目前仍在执行槽内；跨通道 CPU/GPU 流水线是后续工作。

### 显存：混合精度计算不等于权重已压成 BF16

普通 GPU 镜像使用上游 builder 的 FP32 模型构造和 `.cuda().eval()`，随后严格重载权重。
现有 autocast 选择计算精度，并不把整份模型的常驻参数存储改成 BF16。
不能仅凭 BF16 已开启就认定常驻显存已最小化。
[模型构造源码](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model_builder.py#L1070)，
[PyTorch AMP](https://docs.pytorch.org/docs/2.10/amp.html)。

Contour 只接受固定 `person` 提示词，但普通 GPU 版本一直保留完整文本编码器。
本次将已有 Jetson 的固定提示词缓存整理成共享实现，增加
`CONTOUR_FIXED_PERSON_TEXT=true` 开关：严格验证权重后，在真实 CUDA 预热中计算提示词
特征，保留特征并释放文本编码器权重。每次返回独立特征副本，拒绝其他提示词。
日志 `released_weight_bytes` 是释放参数的字节数，不等于进程整体显存降幅。

普通 GPU Contour 的该开关现在默认开启；通用 SAM worker 的默认值仍为关闭。
设为 false 可作同片 A/B 验证；它不量化权重、不降低分辨率、不修改跟踪历史。
它在模型完整加载之后生效，不能解决加载模型时就发生的 OOM。改变它需要重建容器进程。

### 每通道仍有独立状态，且 CPU offload 标志不能按字面理解

上游 `init_state(offload_video_to_cpu=True)` 先在 CPU 加载图像，但
`_construct_initial_input_batch()` 随后对整个 batch 执行 `recursive_to(..., cuda)`，
包括图像张量。本项目 `RingFrames` 再 `clone()`，保持原设备。
因此这条特定路径中的 16 帧输入环仍可驻留 GPU。
[上游实现](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_multiplex_tracking.py#L106)。

该固定版本的 PIL 加载器使用 FP16 保存 RGB 输入；1008×1008 的一条完整环张量大小为
`16 × 3 × 1008 × 1008 × 2 = 97,542,144 bytes`，约 93 MiB；
这是按[加载器源码](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/io_utils.py#L44)
计算的单项大小，不是实测总显存。特征缓存、对象 memory、mask、临时工作区
和模型常驻权重还要另算。GPU 队列会串行执行，但不会自动卸载其他摄像头的状态。

后续可评估真正将输入环放 CPU、减少不再使用的特征缓存或选择性卸载历史状态。
这需要验证传输成本、缓存引用和 mask/ID 一致性，本次未直接改变这些行为。
不能把通用 `offload_state_to_cpu` 参数直接塞进该版本的 multiplex `init_state`，
它并不接受该参数。

### 缩小传输图像，不会按比例降低视觉编码器计算量

原生 builder 固定 `image_size=1008`。512 的传输图仍被预处理成模型尺寸；
降低传输尺寸主要影响编解码、网络和输出 mask，不能把它当成原生 512 模型。
仓库历史对照还记录了 384 下遮挡和边缘人物的退化。
不能不看质量就继续降分辨率。

### 首批一帧被当单图：已确认，单独留待质量评估

上游 `is_image_type(list)` 用 `len(list)==1` 判断单图。本项目 `sam_live` 支持首批一帧，
初始化后没有覆盖该标志，后续持续视频仍会沿用单图相关检测/跟踪启发式。
实际对照将它强制设为 `False` 时，96 帧录像的第 64–95 帧发生输出差异，
其中新目标 ID 9 在视频模式下没有被加入。上游对新人物采用不同检测阈值，
因此不能把这个改动当成无损性能优化，也不能把少检测目标带来的加速记为收益。
**最终补丁保留普通 GPU 原有跟踪语义，Jetson 也保留原有独立补丁。**
该问题应使用有标注的出入画样本单独评估；本次没有宣称识别质量提高。
[上游判断](https://github.com/facebookresearch/sam3/blob/660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7/sam3/model/sam3_video_inference.py#L1713)。

## 先取得可解释的现场数据

更新源码并构建新 engine 镜像后，在 `workers/contour/.env` 中设置：

```dotenv
CONTOUR_GPU_DIAGNOSTICS=true
CONTOUR_FIXED_PERSON_TEXT=false
```

维护窗口内重建服务；只有修改 `.env` 不会改变已运行容器，导入旧镜像也不会带上修复：

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build --force-recreate
docker compose -f compose.yaml -f compose.gpu.yaml images
docker compose -f compose.yaml -f compose.gpu.yaml logs --since=10m engine > engine-timing.log
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

日志按 `api_prefix` 区分摄像头，按 `request_id` 配对 append 与 HTTP 记录。
对比 `queue_wait_seconds`、`body_read_seconds`、`decode_seconds`、
`native_advance_seconds`、`snapshot_seconds`、`retention_seconds`。
这些是主机墙钟时间，不是 CUDA kernel profile；不要把不同记录的总耗时重复相加。

显存分别看 `cuda_allocated_bytes`（活跃张量）、`cuda_reserved_bytes`（分配器保留），
以及 `cuda_device_free_bytes`（整张设备的剩余显存）。峰值计数在服务中从加载/最近一次
显式 reset 累积，不是每请求独立峰值。采样在 append 结束后执行，不添加 GPU 同步。
`nvidia-smi` 包含缓存和其他分配，不能单独证明“模型两路就需 8 GB”。
逐帧 `empty_cache()` 也不会释放仍被引用的张量。
[PyTorch 显存管理说明](https://docs.pytorch.org/docs/2.10/notes/cuda.html#memory-management)。

## 可直接重跑的 GPU 容量对照

新增 `benchmark_gpu.py`，已加入普通 GPU runtime 镜像与源码交付包。使用真实 JPEG 序列，
经过实际 `NativeSam31` 严格加载和 `SamLiveContinuation`；每通道独立状态，逐轮交错执行。
默认 96 帧，跨过 16 帧环回绕和 32 帧历史窗口。每条通道使用同一视频，是可复现的
压力对照，不代表真实不同摄像头的内容分布或长期运行质量。

例如从含人物、遮挡和进出画的录像导出 3 FPS 采样帧；帧目录需先创建且为空：

```sh
ffmpeg -i /data/representative.mp4 -vf fps=3 -frames:v 96 /data/contour-frames/%06d.jpg
```

在维护窗口停止本节点分析及 engine，避免压测另开一份模型挤占生产显存。
以下路径须换为本机绝对路径；`/data/contour-reports` 须允许容器 UID 10004 写入。

```sh
docker compose -f compose.yaml -f compose.gpu.yaml stop engine
docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps \
  -v /data/contour-frames:/frames:ro -v /data/contour-reports:/reports \
  --entrypoint python engine /engine/benchmark_gpu.py \
  --worker-dir /engine --checkpoint /models/contour.pt --frame-dir /frames \
  --frames 96 --lanes 1 2 --batch-frames 1 --output /reports/person-cache-off.json
docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps \
  -v /data/contour-frames:/frames:ro -v /data/contour-reports:/reports \
  --entrypoint python engine /engine/benchmark_gpu.py \
  --worker-dir /engine --checkpoint /models/contour.pt --frame-dir /frames \
  --frames 96 --lanes 1 2 --batch-frames 1 --fixed-person-text \
  --output /reports/person-cache-on.json
docker compose -f compose.yaml -f compose.gpu.yaml up -d engine
```

L40S 上再以 `--lanes 1 2 6` 重跑；另用 `--batch-frames 4` 检查吞吐与单次占槽时间。
每轮使用新报告文件名。建议重复三次、对调开关次序，并单独运行 `--lanes 1`
比较同片输出。脚本的开关以命令行参数为准，不继承 `.env` 的提示词缓存选项。

报告记录每批原生执行及保留状态耗时、前 32 帧之后的饱和 FPS、各阶段显存、输入 hash、
每帧对象 ID、mask hash 和分数。hash 在共享输出缓冲被下一帧覆盖之前计算；失败也写入
已经获得的数据并返回非零退出码。压测每个 case 重置峰值，服务端诊断不会重置它。

**`steady_saturated_native_fps` 是剔除读图、输出 hash 和网络后的原生能力估计，
不是端到端实时 FPS。** 它不包含 HTTP 网关、采集、轮廓 JSON、事件模型、人脸或浏览器。
先用它区分“模型本身不够快”和“外围队列导致延迟”；随后必须走真实 Contour 客户端，
按实时到达速度测试并记录源到结果 P50/P95、每路完成 FPS、积压、丢帧和 EOF 后追赶量。
容量与质量都要包含真实人数峰值，不能只跑无人画面。

## 两张 L40S / 12 路怎样做容量判断

建议按每卡一个 engine、每 engine 六条独立状态划分，两个节点分别绑定 GPU 0、GPU 1
并使用不同 Compose project/端口。单个现有 engine 仍按一张卡设计，不会因配置容量 12
而自动使用两张卡。独立节点部署可行；主程序如何把摄像头分到两个节点还需核对现场配置，
本次没有增加跨节点路由功能。

本片实测原生容量约 7.35 FPS／卡，因此六路每路 2 FPS 所需的 12 FPS，
以及每路 3 FPS 所需的 18 FPS，都超过当前路径的吞吐。仓库
`ServiceLocations.tracking_sample_fps` 默认 2.0，但现场实际值仍未知。
主程序／客户端可通过 `LOCAL_TRACKING_FPS` 或 `tracking_sample_fps` 调整分析采样率；
这与原视频显示帧率不同。单改 Contour 的容量数字不会减少 GPU 工作量。

若每路分析 3 FPS，则每卡需要持续完成 **18 个不同源帧/秒**；仅稳定不积压就要求
平均串行服务时间小于 `1/18 ≈ 55.6 ms/帧`。若计划保留 30% 波动余量，则预算约
`0.7/18 ≈ 38.9 ms/帧`。每路 1 FPS 时相应为 166.7 ms/帧与 116.7 ms/帧。
这些是容量预算的算术推导，不是 L40S 测试结果，也不能保证 P95 延迟。

大致显存预算为：共享模型 + 六路状态 + 最大一次操作的工作区 + 分配器/运行库开销。
GPU 串行执行时，工作区不能简单按六倍相乘；独立通道状态也不能只计一份。
同时观察长期内存平台是否稳定，以及每路真实人数增加后的斜率。

如果修复后原生吞吐仍不达预算，下一轮应实测跨摄像头视觉编码器小批次、CPU 后处理重叠，
或“轻量逐帧跟踪 + 较低频率 SAM 校正”的设计。后者改变检测频率和新人物发现延迟，
必须先确定产品验收条件。不能把排队帧直接丢掉再宣布延迟达标。
编码器编译、全面 BF16 常驻权重、TensorRT 或量化均需独立评估；仓库过去的 encoder compile
和后处理 batch 对照没有证明提速，不能默认开关一开就解决。

## 本地验证边界

本轮测试覆盖慢上传不阻塞其他通道、单通道不乱序、取消/超时/超限隔离、严格加载后才
安装提示词缓存、缓存副本隔离、显存计数语义、保留原有检测模式、Jetson 补丁兼容，
以及压测工具对复用输出缓冲和失败清理的处理。正式权重原生与 HTTP 测试及边界见本文开头；
长时间、多种真实现场内容和双卡同时运行的生产验收仍待完成。

本地为 macOS / Python 3.13，前期定向回归 **236 项通过**；GPU 阶段后续定向回归
**192 项通过**，最终保留检测模式后相关 **46 项通过**（这些集合有重叠，不相加）。
修改文件的 Ruff 和 `git diff --check` 通过。此前全套 4,119 项中
**4,096 通过、10 跳过、13 失败**；
将全部 13 个失败用例放到未修改的 `bf614ae` 独立 checkout、使用相同环境重跑，
13 项均以相同错误复现：12 项 WebSocket TestClient 退出时 `CancelledError`，
1 项缺少 `vastai`。因此不能宣称全套全绿，但没有发现此次修改引入的测试失败。
本机没有 Docker/NVIDIA runtime；最终 CUDA 镜像已经在远端 105 构建并完成正式权重验证。

模型原生耗时、HTTP 请求耗时与源到屏幕延迟应分别记录；不能用旧硬件或旧链路的分段计时承诺当前部署速度。
