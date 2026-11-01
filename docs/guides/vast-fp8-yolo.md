# 48GB RTX 4090：Qwen3.5 35B FP8 + YOLO26-S

> 2026-10-20：本文实验工具使用独立的 `tools/benchmark_runtime/`；产品 grounding 已恢复 main。历史结果不等于恢复后的产品性能，见[代码与测试边界](pr48-49-person-grounding-integration.md)。

状态：2026-10-18 已完成初次十二路、三个模式各两轮测试，容量与三秒目标未达标。当前新增 16/24/32 并发与 FP8 内核对照脚本，尚未部署或重新推理；下方部署记录是上一版的历史记录。
本次接入基于最新 main `9dbd510`（包含 PR #48、#49），位于 `codex/yolo-decisions-main` 工作分支。
入口：`PYTHONPATH=src python -m tools.vast_vlm.fp8 plan`。
所有动作默认只输出计划；需要显式 `--execute` 才安装、下载、启动或推理。
原 `tools.vast_vlm` BF16 入口保持独立。

## 本次部署记录（2026-10-18）

- 原实例 `56629097` 原地重建，独立卷 `56629096`（150GB）保留；用户确认容器盘没有需保留文件后才执行。新 SSH 端口为 `59259`。
- 启动脚本移除不适用于固定镜像的 `entrypoint.sh`，修复 SSH 目录权限并保留全部公钥；客户端补装与当前 main 一致的 `websockets==15.0.1`，实机 `pip check` 通过。
- 相关离线回归 51 项通过，Shell 语法、Ruff 与差异检查通过。
- 原 `output.mp4`、六类 policy 与 Astra 证据哈希校验通过；真实 YOLO 检测 12/12 帧请求成功，grounding 返回的人物 `P0001` 与检测结果关联。
- 单路预检：events 2/2 完整，端到端 0.432 / 0.585 秒；grounding 2/2 完整，端到端 4.815 / 4.025 秒。没有调用错误；这些是仅四个请求的原始值，不能外推十二路完整率或 P95。
- 浏览器和视频预览正常启动；预检阶段按现有脚本不导出 MP4 录屏。随后获得正式授权并完成十二路六轮，结果保存在本地 `outputs/vast-fp8-formal-20261018/results`。本次优化仅修改脚本，没有补跑。
- 原始结果在 `/models/results/preflight-20261018`，服务日志与环境记录在 `/models/runtime`；本地副本位于本工作树 `outputs/vast-fp8-main-review`。

## 固定条件

- 官方 `Qwen/Qwen3.5-35B-A3B-FP8`，固定 revision `9d1823d2dee688a6b25e77009dc727688c44936e`；启动校验权重哈希与 FP8 配置，不做临时量化、不回退 BF16、CPU 或云 API。
- vLLM 0.31.0，固定镜像如下；模型服务仅监听 127.0.0.1:8000。
- YOLO26-S + 原 IoU/motion 跟踪器，固定既有实测权重 SHA256，ultralytics 8.4.181，640 输入、confidence 0.35、2.5 FPS 检测采样。ID 属于单摄像头会话，不是身份识别，也不保证遮挡后保持编号。
- 十二路，1.25 窗口/秒/路，原 output.mp4 与六类 policy 哈希校验。events、grounding、parallel 各两轮，每分支每轮 828 窗口；正式分支请求 6,624 次，另有两项单路预检及每轮 10 秒暖机。
- 客户端仍为 48 个 admission 名额、每路 4 个；错峰、配对调度，内存 admission。HTTP 超时统一 20 秒，3 秒是统计目标而不是取消期限。旧云 YOLO events 的 5 秒调用超时不同，报告比较必须注明。
- vLLM 同批最大 16 序列、上下文 8192、显存利用率 0.85。它不改变计划窗口或请求频率；超出服务能力仍会排队。为 YOLO 留出显存只是初始配置，不能保证模型装得下或十二路达标。不得通过减少请求窗口掩盖容量不足。
- 15 FPS 单视频，只显示人物 ID，不显示框或右侧证据帧。内部保留框定位人物；六类规则不变，ground_litter 为场景级判断。

## 持久卷与环境

Vast 持久卷挂载 `/models`，建议 150GB（只保存本轮 FP8 时最低预留 100GB）；容器系统盘建议另留 80–100GB 给镜像解压、独立环境和临时安装。官方固定 revision 的权重文件共 37,463,724,680 字节，约 37.5GB / 34.9GiB；这不是 GPU 实际显存需求。

起步机器条件：一张实际 48GB 的 RTX 4090、至少 16 个可用 CPU 核心、128GB 可用主机内存、NVMe 存储。CPU 用于十二路解码、浏览器录屏和调度；这些是资源预留建议，不能替代实机吞吐验证。官方镜像构建使用 CUDA 13.0.3 / Python 3.12，宿主驱动须兼容，不能沿用旧的“CUDA 12.4 即可”筛选条件。

固定镜像（已对照 Docker Hub v0.31.0 tag 校验摘要）：

```text
vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
```

使用 Vast 支持 SSH 的 Docker 启动方式，覆盖原镜像自动启动模型的 entrypoint；在收到代码包并完成准备后按下文显式启动服务。不能把镜像粘贴到普通模板就认为已完成模型启动。也可用源码包的 Dockerfile 构建派生镜像，记录其新 digest。不要在容器中再启动 Docker。

### SSH 启动模板

在 **Interactive shell server, SSH** 模式中，将 `tools/vast_vlm/onstart-ssh.sh` 的完整内容粘贴到 **On-start Script**。不要沿用 Vast base-image 的 `entrypoint.sh`：固定 vLLM 镜像没有这个命令。脚本只修复 `/root`、`.ssh` 和现有 `authorized_keys` 的属主及权限，保留所有公钥内容，不关闭 SSH 的 StrictModes，不改动模型卷，也不自动下载或启动模型。异常符号链接、空或缺失的公钥文件会直接报错。

修改模板不等于已经修改运行中的实例。已有实例若能连接 SSH，可在保留旧启动脚本副本后应用修复；若 SSH 已因权限错误被拒绝，需要可用的管理终端或 Vast 维护入口。Vast 的 `update instance` / recreate 会重建容器，**不能当成保留容器文件的普通重启**。先备份容器中的所需文件并确认数据卷保留，再执行重建；仅查看模板或保存新模板不会自动修复现有实例。

创建本地持久卷时选定宿主机、设置容量，实际挂载位置以实例内的挂载信息为准。本次卷 `56629096` 实际挂载在 `/workspace`，核对独立卷来源后创建 `/models -> /workspace` 符号链接；没有格式化或复制卷。不要在未核对来源时把普通容器目录当成持久卷。卷容量不能事后调整；后续实例须在同一宿主机上挂回这块卷，换物理机器需要复制。卷必须是独立磁盘挂载，脚本解析符号链接后仍拒绝容器 overlay、根目录和 tmpfs。Vast 卷的保留、计费和宿主机绑定需在租机时确认；不能假定可随意挂到另一台物理机器。

卷保存模型 revision 目录、YOLO 权重、引擎编译缓存、服务日志与启动记录。建议一并保存：

- `/models/media/output.mp4`：原测试视频。
- `/models/reference/astra-packet`：既有标注包，只用于离线评分。
- `/models/results/<新运行名>`：报告、逐请求数据和录屏。
- `/models/source/`：本次源码包及其校验摘要。

服务器记录包含宿主启动 ID 和进程启动时间；换机或重启后旧记录不能当成存活服务。停止服务时清理其独立进程组，包括占用显存的子进程。复用时重新校验哈希，已验证权重不重新下载。客户端和 YOLO 环境独立于 vLLM 镜像环境；新容器需要重新运行环境安装，不能把旧虚拟环境跨镜像直接复用。

每次换机，先确认实际显存 >=46GiB、GPU 为 RTX 4090 Ada SM8.9、驱动兼容镜像。市场标注的 48GB 是非标准改装卡，不能用普通 24GB 4090 替代。本次实测为 49140MiB，驱动 595.84；权重加载占 34.25GiB，服务与 YOLO 启动后总显存约 40GiB。实际日志是线性层选择 Marlin，MoE 选择 Triton 且缺少该卡的专用调优配置；不能把所有 FP8 算子都说成 Marlin，也不能据 FP8 名称推断吞吐。

## 可复用操作顺序

先在本地已有 Harness 环境生成仅源码包（不会租机或下载模型）：

```sh
PYTHONPATH=src python -m tools.vast_vlm.fp8 bundle --output /tmp/vast-fp8-yolo-source.tar.gz --execute
```

上传解压到容器目录 `/opt/goodvision/harness` 后，在源码根目录（源码压缩包另存 `/models/source`；虚拟环境安装在容器盘）：

```sh
bash tools/vast_vlm/setup-client.sh --execute
bash tools/vast_vlm/setup-yolo.sh --execute
# 使用 vLLM 镜像的 Python 下载并验证两套权重到挂载卷
PYTHONPATH=src python -m tools.vast_vlm.fp8 prepare --volume /models --execute
# 终端 A：先启动 YOLO 并完成 GPU 暖机
PYTHONPATH=src .venv-vast-client/bin/python -m tools.vast_vlm.fp8 start-detector --volume /models --execute
# 终端 B：用镜像的 Python 启动 vLLM，保持前台存活
PYTHONPATH=src python -m tools.vast_vlm.fp8 start --volume /models --execute
# 终端 C：小规模预检，只跑单路 events 和人物 grounding，不进入正式六轮
PYTHONPATH=src .venv-vast-client/bin/python -m tools.vast_vlm.fp8 preflight --volume /models --video /models/media/output.mp4 --packet /models/reference/astra-packet --output /models/results/preflight-new --execute
# 仅在另行获得正式测试授权后执行完整六轮；本次不运行
PYTHONPATH=src .venv-vast-client/bin/python -m tools.vast_vlm.fp8 run --volume /models --video /models/media/output.mp4 --packet /models/reference/astra-packet --output /models/results/benchmark-new --execute
```

测试完成后可导出结果包，随后下载到本地；释放实例前确认持久卷仍保留：

```sh
PYTHONPATH=src .venv-vast-client/bin/python -m tools.vast_vlm.fp8 collect --results /models/results/benchmark-new --output /models/results/benchmark-new.tar.gz --execute
```

启动阶段没有推理成功或吞吐保证；OOM/不支持算子直接停止，保留日志，不自动降低画质、帧数、六类标准或换模型。前台服务支持 SIGTERM/Ctrl-C 清理自己的子进程，不杀其他任务。无需 OpenRouter API Key。

## 结果与判定

保存 manifest、实际 deployment、逐请求计时/token、YOLO 采样记录、主机/GPU 与 vLLM `/metrics` 时序、Astra 场景一致率、六段正式录屏。延迟包含源窗口进入到发布；记录完整率、各路 3 秒内完整率、P50/P95 和 parallel 双分支同时完成率。vLLM 聚合 queue/prefill/decode 指标用于定位瓶颈，不冒充逐请求纯 GPU 计算时间。

比较旧云 35B 时必须注明 FP8 精度、推理引擎、人物协议、调用超时和硬件差异；Astra 只评六类场景判断，不评 ID 关联正确性。没有实机结果前不能承诺 3 秒内 95%。

## 待执行的 16/24/32 对照

当前用户选择：本轮固定 `--kernel auto`，只安排 `matched` 下的 16/24/32 并发对照。Triton 调优和 Marlin 候选保留为后续备用，不纳入本轮执行计划，也不先运行内核调优。仍等待正式运行指令。

入口为 `python -m tools.vast_vlm.fp8_sweep`，与原 `fp8` 入口分开。默认只生成计划；没有 `--execute` 时，连 `start`、`preflight`、`run` 都不会连接 GPU 或请求模型。本次不切换运行中的服务。

| 组别 | 引擎最大同时处理数 | 默认客户端 admission 总名额 | 每路上限 |
| --- | ---: | ---: | ---: |
| seq16-matched | 16 | 16 | 4 |
| seq24-matched | 24 | 24 | 4 |
| seq32-matched | 32 | 32 | 4 |

`matched` 将客户端名额与引擎并发对应，减少请求在 vLLM 内部堆积的机会。它可能把损失转移到本地 admission 超时，不能只看已发请求的成功率。原始十二路仍按 1.25 窗口/秒/路产生输入，单模式 15 请求/秒、parallel 30 请求/秒；未发、失败、迟到都保留在每分支每轮 828 个窗口的分母中。

另备 `--admission fixed48`：客户端始终 48 名额，只改引擎 16/24/32，用于单独分析引擎并发的影响。不能把 `matched` 与 `fixed48` 混成同一组并发对照，也不能把结果提升全部归因于引擎。默认一组包含三个配置，每配置 events、grounding、parallel 各两轮，共 18 轮、19,872 个计划分支窗口，另有预检与暖机。本次仅准备，不执行。

所有候选固定 `max_num_batched_tokens=2048`、chunked prefill、显存利用率 0.85、8192 上下文；prefix 与多模态处理缓存保持关闭。原帧数、画质、输出上限、thinking=false、HTTP 20 秒超时、本地 admission 等待 1 秒、零自动重试、YOLO 权重与采样、六类 policy、仅显示人物 ID 均不变。并发升高可能增加单次计算耗时或导致 OOM，脚本不自动降低输入规格来继续。

### 内核候选与持久化

- `--kernel auto`：保留自动选择作为对照，保存实际选中的线性层与 MoE 后端日志。
- `--kernel triton-tuned --tuned-dir /models/kernels/qwen35-fp8-031`：显式使用 Triton MoE 和本机调优文件。此轮只优化 MoE 配置，不额外强制替换线性层后端。
- `--kernel marlin`：显式 Marlin MoE 备选，仍使用相同 FP8 checkpoint。不支持、启动失败或实际选中其他后端时停止，不静默回退。

同一组 16/24/32 保持一种内核设置；本轮已选 auto。后续若另行比较内核，需保持并发、admission 不变。所有候选均未宣称已提速。

调优入口 `python -m tools.vast_vlm.fp8_kernels plan` 仅输出配方。未来明确执行 `tune --execute` 才下载并校验固定 vLLM 0.31.0 的官方 `benchmark_moe.py`，在空闲 GPU 上寻找适合当前 4090 的 Triton 参数。它会检查模型哈希、GPU 架构、vLLM/Ray、MoE 形状；若 GPU 已被占用直接拒绝，不停止其他服务。调优本身占用 GPU、耗时并产生租赁费用，不能在 VLM/YOLO 运行中执行。

文件留在独立持久卷 `/models/kernels/<新目录>`：包含各 batch size 的内核参数、调优日志、模型 revision、GPU、vLLM/Triton 版本及 SHA256。固定官方脚本的 SHA256 为 `cc78ddf704150b8585d6ffb8fba8fd2d0518629b0c3e6c511a3f39d217b640dd`。重建容器后版本相符才复用；不修改模型权重，不覆盖已有调优目录。启动要求日志明确确认该配置文件加载，缺文件或版本不匹配不能当作“已调优”。调优完成也不证明十二路速度或识别质量达标。

### 后续操作示例（本次未执行）

本地查看三档计划，不接触服务器：

```sh
PYTHONPATH=src python -m tools.vast_vlm.fp8_sweep plan
PYTHONPATH=src python -m tools.vast_vlm.fp8_sweep plan --admission fixed48
PYTHONPATH=src python -m tools.vast_vlm.fp8_kernels plan
```

下列仅供下次获准验证后使用。每次选一个配置，服务前台运行；结束自己的旧 VLM 服务后才能切换，不能用 `pkill` 扫掉其他任务。调优需先显式结束本任务的 VLM 和 YOLO，完成后再按原入口启动 YOLO。服务启动使用固定镜像 Python，测试客户端使用独立虚拟环境。

```sh
# 后续备用，本轮不执行：空闲 GPU 上生成调优文件
PYTHONPATH=src python -m tools.vast_vlm.fp8_kernels tune --volume /models --output /models/kernels/qwen35-fp8-031 --execute

# 示例：先验证 24 并发 auto；16/32 只改 --sequences 和新的输出目录
PYTHONPATH=src python -m tools.vast_vlm.fp8_sweep start --sequences 24 --admission matched --kernel auto --execute
PYTHONPATH=src .venv-vast-client/bin/python -m tools.vast_vlm.fp8_sweep preflight --sequences 24 --admission matched --kernel auto --video /models/media/output.mp4 --packet /models/reference/astra-packet --output /models/results/seq24-auto-preflight --execute
PYTHONPATH=src .venv-vast-client/bin/python -m tools.vast_vlm.fp8_sweep run --sequences 24 --admission matched --kernel auto --video /models/media/output.mp4 --packet /models/reference/astra-packet --output /models/results/seq24-auto --execute
```

客户端会核对存活进程的真实启动参数、实验配置与内核证据；16 服务不能贴成 24 的报告。原入口也拒绝误用实验服务。若使用调优文件，启动和测试两端必须同时带相同 `--kernel triton-tuned --tuned-dir ...`。

三组结果齐全后，`compare` 只读已有结果重新评分，不调用模型：

```sh
PYTHONPATH=src .venv-vast-client/bin/python -m tools.vast_vlm.fp8_sweep compare --results /models/results/seq16-auto /models/results/seq24-auto /models/results/seq32-auto --video /models/media/output.mp4 --packet /models/reference/astra-packet --output /models/results/concurrency-comparison.json --execute
```

比较文件包含每模式每轮的发出率、完整率、每路三秒内完整率与最差一路、延迟 P50/P95、Astra 场景一致率、token/TPS 覆盖率、HTTP 状态及已有 vLLM queue/prefill/decode、GPU 遥测。三组须使用同一源码、内核、admission、视频与标注，缺轮、重复窗口、统计与原始结果不一致均拒绝比较。性能结论必须等后续实机结果；本地替身测试不能确认 FP8 内核实际兼容性或速度。


## 上游核对与验证边界

初期方案核对使用官方元数据和源码；2026-10-18 获得部署授权后，已使用固定镜像并下载、校验权重：

- [官方 FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)：固定 revision、FP8 配置及权重总量。
- [vLLM v0.31.0 Dockerfile](https://github.com/vllm-project/vllm/blob/v0.31.0/docker/Dockerfile)：Python/CUDA 环境。
- [Vast Volumes](https://docs.vast.ai/guides/instances/storage/volumes)：宿主绑定、挂载目录和容量限制。

离线回归包含固定负载校验、权重/挂载/进程记录检查、本地 SSE 的成功与失败解析、十二个真实子进程的三个模式，以及 Chromium 的 ID-only 三模式录屏。HTTP 和检测器响应为测试替身，不能用于宣称真实识别准确率、显存可用性或三秒指标达标。
