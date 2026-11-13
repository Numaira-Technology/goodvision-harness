# Docker 模块目录与版本 tracker（goodvision-xxx）

维护代码、部署说明和更新记录时，统一使用 `goodvision-xxx` 格式的完整模块名称。
名称使用小写英文字母、数字和连字符，固定以 `goodvision-` 开头；不随版本、CPU/GPU
或安装地点变化。显示名称、安装器短 ID、Compose service、镜像和容器实例名分别记录。
本目录随 Git 持久保存，新增模块或调整映射时在同一提交中更新本目录。

2026-11-12 更新独立 CLIP 容器与原五个模块的映射。历史安装定义来自
[独立模块 PR #75](https://github.com/Numaira-Technology/goodvision-harness/pull/75)。
这里记录源码定义与命名约定；已构建、已交付的镜像和各模块独立版本见
[下方版本记录](#最新交付与服务版本)，现场部署状态单独记录。
配置中的目标版本或本地镜像不等同于客户服务器已经部署。

## 独立安装单元

统一安装器原有五个完整模块名称同时固定为构建与交付使用的镜像仓库名称：
`goodvision-main`、`goodvision-bbox`、`goodvision-contour`、`goodvision-system1`、
`goodvision-gpu-runtime`。版本、设备和容器角色写在标签中，不另设镜像仓库。
安装器短 ID 对应
[安装器定义](../../tools/docker_modules.py) 的 `MODULES`、`deploy_modules.py --modules`
和源码包 `module-package.json.module`，继续按原值使用；完整名称不直接替代命令参数。
`goodvision-neural` 另由 CLIP 增补安装入口独立管理，不加入旧安装器的 `MODULES`，
也不改写原 `settings.json` 的模块字典。`<version>` 是各模块自己的发布版本，
`<namespace>` 是安装实例名；尖括号为占位符。

| 固定模块名称 | 安装器短 ID | 职责 | Compose service | 新安装镜像命名 | 源码 / 构建入口 |
| --- | --- | --- | --- | --- | --- |
| `goodvision-main` | `main` | 主程序：界面、视频接入、工作流、事件与结果 | 新安装为 `main`；旧安装为 `harness` / `harness-N` | `goodvision-main:<version>` | [Dockerfile](../../Dockerfile)、[应用源码](../../src/goodvision_harness) |
| `goodvision-bbox` | `bbox` | Bbox：YOLO26-S 检测和追踪 | `bbox` | `goodvision-bbox:<version>-cpu` / `goodvision-bbox:<version>-cuda` | [CPU Dockerfile](../../workers/bbox/Dockerfile.cpu)、[CUDA Dockerfile](../../workers/bbox/Dockerfile)、[服务源码](../../workers/detector_rich_label) |
| `goodvision-contour` | `contour` | Contour：对外接口与内部推理，作为一个安装单元管理 | `contour`（接口）+ `engine`（推理） | `goodvision-contour:<version>-api` + `goodvision-contour:<version>-engine` | [接口 Dockerfile](../../workers/contour/Dockerfile)、[推理 Dockerfile](../../workers/contour/Dockerfile.engine)、[接口源码](../../workers/contour)、[推理源码](../../workers/sam_scene) |
| `goodvision-system1` | `system1` | System 1：VideoJev 视频推理 | `system1` | `goodvision-system1:<version>` | [Dockerfile](../../workers/system1/Dockerfile)、[服务源码](../../src/goodvision_harness/videojev_service.py) |
| `goodvision-gpu-runtime` | `gpu-runtime` | 共享 GPU 依赖底座；只构建镜像，无常驻容器 | 无 | `goodvision-gpu-runtime:<version>-torch2.10-cu128` / `goodvision-gpu-runtime:<version>-torch2.8-cu128` | [Dockerfile](../../workers/gpu_runtime/Dockerfile)、[底座说明](../../workers/gpu_runtime/README.md) |
| `goodvision-neural` | 独立增补入口，不是旧 `--modules` 参数 | CLIP ViT-L/14 推理；生命周期独立于 VideoJev | `neural` | `goodvision-neural:<version>-<build>`；首次服务版本 V2.0 | [Dockerfile](../../workers/neural/Dockerfile)、[增补安装器](../../tools/neural_container.py)；V2.0 复用原 System1 镜像依赖层和 CLIP 应用 |

新交付的主程序镜像必须使用 `goodvision-main`，不得继续输出 `goodvision-harness`。
下文列出的旧镜像名称只用于识别历史部署，不是新构建或交付可使用的兼容别名。
Contour 的容器角色固定写作 `goodvision-contour/contour` 和
`goodvision-contour/engine`（完整模块名称 / service）。
`engine` 不是额外的安装单元。CPU/CUDA Bbox 是同一模块的两种构建类型。
Contour 与 CUDA Bbox 共用 PyTorch 2.10 底座，System 1 使用 2.8 底座；更新共享底座后
需要重新构建、验证依赖它的模块镜像，不能只替换底座标签就视为应用已升级。
首次 CLIP 镜像基于已交付的 V2.0 System1 镜像构建新入口和健康检查，默认只启动
`neural_service`；共享层不代表启动 VideoJev，也不代表本轮升级了 CLIP 算法。

## 安装实例与持久化位置

以下规则由 [统一安装器](../../tools/deploy_modules.py) 生成，适用于原五个模块的新安装。
独立 CLIP 的增补安装规则见表后说明。

| 对象 | 固定规则 | `office` 安装示例 |
| --- | --- | --- |
| Compose project | `goodvision-<namespace>-<module>`，`<module>` 使用安装器短 ID | `goodvision-office-bbox` |
| 容器实例 | 固定模块名后追加 `-<namespace>`；Contour 两角色分别追加 `-api` / `-engine` | `goodvision-main-office`、`goodvision-contour-office-api`、`goodvision-contour-office-engine` |
| 同机共享网络 | `goodvision-<namespace>` | `goodvision-office` |
| 网络别名 | `gv-<module>`，只在所在网络内使用 | `gv-bbox` |
| 默认状态根目录 | `~/.goodvision/docker/<namespace>`；可用 `--state-dir` 指定 | `~/.goodvision/docker/office` |
| 安装设置 | 状态目录下 `settings.json` | 版本、端口、地址、Bbox 设备、GPU 编号等 |
| 已应用的 Compose | 状态目录下 `compose.<module>.json` | `compose.bbox.json` |
| 模型目录 | 状态目录下 `models/<module>` | `models/bbox`、`models/contour`、`models/system1` |
| Contour 密钥 | 状态目录下 `secrets/contour-key` | 更新时保留，值不写入本目录 |
| 应用数据卷 | 项目内 `data`；主程序另有 `face-models` | 实际卷名以 Compose / inspect 为准 |
| 源码安装包的身份 | 包内 `module-package.json` 的 `module`、`version`、`bbox_device` | 源码文件另有 SHA-256 清单；生成规则见[打包器](../../tools/package_modules.py) |

| 模块 / 角色 | 默认网络地址 | 默认宿主机端口 |
| --- | --- | --- |
| `goodvision-main/main` | `http://gv-main:8080` | 8080 |
| `goodvision-bbox/bbox` | `http://gv-bbox:8096` | 8096 |
| `goodvision-contour/contour` | `http://gv-contour:8093` | 8093 |
| `goodvision-contour/engine` | `http://engine:8092`，仅 Contour 的内部网络 | 不发布 |
| `goodvision-system1/system1` | `http://gv-system1:8097` | 8097 |
| `goodvision-neural/neural` | 默认 `http://gv-neural:8098`，实际沿用 Main 的 neural URL | 增补安装器按现有配置/显式选项生成 |

同一安装各模块保持相同 namespace 与状态目录。升级保留这些标识、模型、密钥、数据卷
及保存的服务地址。`settings.json` 的版本是构建/配置目标，不能单独证明容器已运行该版本。
镜像标签也不能唯一证明内容；更新记录应同时记录实际 image ID / digest 与源码提交。

CLIP 增补安装器读取原交付目录的 `state/settings.json` 与 `state/compose.main.json`，
沿用 Main 的网络与 neural 服务地址、GPU 编号和原 `state/models/neural` 模型目录，
只生成独立的 `state/compose.neural.json` 和该模块部署记录。容器固定名为
`goodvision-neural`，默认网络别名 `gv-neural`，容器端口显式设为 8098，重启策略为
`unless-stopped`。原安装器的 Start-Server 不会因此自动管理 CLIP，使用新交付的
CLIP start/status 入口。原模型、其他模块的 Compose、设置、卷和密钥保留。

## 其他 Docker 角色与旧名称对应

下表统一内部记录用语，覆盖仓库其余 Compose service 及独立 Dockerfile。
完整模块名称不自动成为 `--modules` 参数；是否能独立安装由各自入口决定。
同一 service 名可能运行不同程序，判定时同时核对 Compose 入口、构建 target 和启动命令。

| 完整模块名称 / 角色 | 现有 service / 名称 | 镜像或源码入口 | 对应范围 |
| --- | --- | --- | --- |
| `goodvision-main` | `harness`、`harness-N` | [compose.yaml](../../compose.yaml)、[交付 Compose](../../compose.full-gpu.delivery.yaml)、[主服务器 Compose](../../compose.main-server.yaml) | `goodvision-harness:*` 的完整应用；`Dockerfile.face-gpu` 和 release harness target 均为该模块的构建类型 |
| `goodvision-analysis-gateway` | 仍叫 `harness`，但运行转发网关 | [网关 Compose](../../compose.analysis-gateway.yaml)、[远端 SAM Compose](../../compose.sam31-remote.yaml)；根 Dockerfile 的 `analysis-gateway` target | `goodvision-analysis-gateway:*` 或 `goodvision-harness:sam-remote`；只转发控制台、视频及结果，不凭 service 名认定为完整主程序 |
| `goodvision-rich-label` | `rich-label-worker` | [CPU Dockerfile](../../workers/paddle_rich_label/Dockerfile)、[GPU Dockerfile](../../workers/paddle_rich_label/Dockerfile.gpu) | `goodvision-rich-label-worker:*`；Paddle 属性服务，release worker target 仍属此角色 |
| `goodvision-contour/engine` | `engine`；本地复用时为 `local-tracking` | [Contour GPU Compose](../../workers/contour/compose.gpu.yaml)、[本地追踪 Compose](../../compose.local-tracking.yaml)、[本地 GPU Compose](../../compose.local-gpu.yaml) | `goodvision-contour-runtime:*`；`local-tracking` 直接复用推理引擎，不代表已安装对外 Contour 接口 |
| `goodvision-contour`（Jetson 类型） | `contour` + `engine` | [Jetson Compose](../../workers/contour/compose.jetson.yaml)、[Jetson Dockerfile](../../workers/contour/Dockerfile.jetson) | `goodvision-contour:*-jetson` / `goodvision-contour-runtime:*-jetson`；保留 ARM64/JetPack 构建类型 |
| `goodvision-sam31` | `sam31-worker` / `sam-server` | [SAM31 Compose](../../compose.sam31.yaml)、[独立 SAM Compose](../../compose.sam-server.yaml)、[worker Dockerfile](../../workers/sam_scene/Dockerfile) | `goodvision-sam31:experimental`；流式与 perception 覆盖文件复用原 service，不增加模块 ID |
| `goodvision-sam-scene` | `sam-scene-worker` | [scene Compose](../../compose.scene.yaml)、[worker Dockerfile](../../workers/sam_scene/Dockerfile) | `goodvision-sam-scene:experimental`；与 `sam31` 共用源码，但模型与运行模式不同 |
| `goodvision-sam-remote` | `sam-remote` | [远端 SAM Compose](../../compose.sam31-remote.yaml)、根 Dockerfile 的 `sam-remote` target | `goodvision-sam-remote:controller`；远端 GPU 生命周期控制器，与 GPU 上的推理进程分开记录 |
| `goodvision-detector-remote` | `detector-remote` | [远端 detector Compose](../../compose.detector-remote.yaml)、根 Dockerfile 的 `detector-remote` target | `goodvision-detector-remote:controller`；控制器入口为 `tools/start_detector.py`，不是独立 `bbox` 安装包 |
| `goodvision-detector-worker` | 无仓库 Compose service | [detector Dockerfile](../../workers/detector_rich_label/Dockerfile) | 旧的 YOLO / RF-DETR 通用 worker 构建入口；和 Bbox 共用应用源码，但入口、端口及交付方式不同 |
| `goodvision-vlm` | `vlm-server` / `local-inference` | [VLM Compose](../../compose.vlm-server.yaml)、[本地 GPU Compose](../../compose.local-gpu.yaml) | GPU 模型服务，镜像由 `LOCAL_INFERENCE_IMAGE` 配置；模型、量化与镜像版本单独记录 |
| `goodvision-vlm-processing` | `vlm-processing` / `local-vlm-processing` | 上述 VLM Compose；根 Dockerfile + `python -m goodvision_harness.vlm_server` | `goodvision-vlm-processing:local`；CPU 图像准备与 VLM 代理，与模型服务分开更新和验证 |
| `goodvision-vlm`（基准测试类型） | 无仓库 Compose service | [Vast VLM Dockerfile](../../tools/vast_vlm/Dockerfile)、[固定测试配置](../../tools/vast_vlm/profile.py) | 固定 vLLM digest 的实验环境；只用于对应测试协议，不当作生产镜像别名 |
| `goodvision-neural`（原标准 V2.0 状态） | 原交付没有 Docker service / 常驻容器 | [神经网络服务源码](../../src/goodvision_harness/neural_service.py) | 原模型已交付，GPU 验收曾临时使用 System1 镜像启动 CLIP 后删除；新的独立增补入口见上表，不能把旧验收容器当作客户常驻部署 |

`compose.modules.yaml` 是另一条手动部署入口，project 为 `goodvision-modules`，service
为 `bbox` / `system1`；其镜像仍为 `goodvision-bbox:0.1.0` / `goodvision-system1:0.1.0`，
不要套用统一安装器的 CPU/CUDA 标签或 project 规则。
手动底座构建器使用 `goodvision-gpu-runtime:torch<torch>-cu128`，没有发布版本前缀。
云端、perception、stream、GPU 加速与 `diag/` Compose 覆盖文件沿用被覆盖的角色名称；
实际运行组合仍需查看合并配置。

## 更新工具的名称兼容边界

目前 [源码更新器](../../tools/update_docker.py) 和
[Windows 客户更新包](../../tools/package_customer_docker_update.py) 的 component 仍只接受
`harness` / `contour`。完整模块名称 `goodvision-main` 对应旧更新器的 component `harness`，
但旧工具按 `harness` / `harness-N` service 发现主程序，不能据此声称已支持新 `main` service。
旧工具没有因此扩展到 `bbox` / `system1` / `gpu-runtime`。
新安装单元使用统一安装器的版本化 build / up 流程及独立镜像更新 ZIP；新更新包
提供 `update.cmd` / `rollback.cmd`，与旧客户更新包入口分开。具体步骤见
[Docker 安装指南](../guides/docker-modules.md)。

旧 Contour 引擎镜像曾叫 `goodvision-contour-engine:<release>`，也使用过
`goodvision-contour-runtime:<release>`；project 可能是 `contour` 或 `goodvision-contour`。
这些历史名称均指向 `goodvision-contour/engine` 角色，来源与兼容记录见
[Docker 更新指南](../guides/docker-updates.md#historical-compatibility-evidence-and-limits)。
更新 Contour 时核对接口与 engine 的配套版本；保留现有 project/service 标签、容器名、
网络别名、挂载、模型和密钥，不能只按镜像名推断安装身份。

## 后续维护约定

- 更新记录统一写 `goodvision-xxx` 完整模块名称，并补充 namespace / 主机、容器角色、源码提交、旧/新镜像
  ID 或 digest、版本、构建类型及验证结果。端口和容器名只作为该安装实例的定位信息。
- 每个模块独立维护版本；更新 Main 不连带其他模块涨版。构建/交付与现场部署分开登记，
  持久记录及后续发布规则统一维护在[下方版本记录](#最新交付与服务版本)。
- 新增模块先登记 `goodvision-xxx` 完整名称、安装器短 ID、职责、service、镜像与源码入口，
  再扩展安装器或更新器。显示名称可以调整；已持久化的模块名称和短 ID 不改义、不复用。
  改 service 或镜像仓库需保留旧名映射。
- 同一模块的 CPU/CUDA、x86/Jetson、模型版本与发布版本作为独立字段记录。
  需要不同接口或生命周期管理的服务，在本表中单独登记角色与兼容边界。
- 验证命名时用安装器 `plan` 及源码核对；真实镜像、GPU 推理和更新回滚需要各自验收。
  实测结论继续写入 [验证记录](../validation/README.md)，不在本目录重复保存运行日志。

## 独立版本与交付记录

维护日期：2026-11-12。每个模块独立发布、独立升级；首次交付均从 **V2.0** 开始。
Main 发布 V2.1 不会使 VideoJev、Bbox、Contour、CLIP 或 GPU 底座自动成为 V2.1。
模块名称、容器角色与配置映射见上文；本节记录同一目录的交付和验证状态。

本表区分四件事：源码具备功能、镜像已构建、包已交付、服务器已部署。
仓库与本地交付记录可以证明前三项；**本次没有操作或读取客户服务器，现场部署版本全部未知**。
镜像仓库名称是离线镜像的 repository，不表示已推送到任何远端 registry。

## 最新交付与服务版本

当前服务器操作安排：只运行原 V2.1 包的 `Update-Main.cmd` 更新 Main；System1 / VideoJev
保持原状。CLIP 等本轮独立容器交付完成后，再使用独立 CLIP 入口安装。
此安排不构成现场已执行或已更新的证据。

| 模块 | 最新服务交付版本 | 最新相关镜像 artifact | 功能 / 常驻容器 | 构建、交付与验证状态 | 客户服务器 |
| --- | --- | --- | --- | --- | --- |
| `goodvision-main` | V2.1 | `goodvision-main:v2.1-77362bd` | UI、视频源、工作流、结果、人脸库、总结；`goodvision-main` | 已构建；已放入原 `GoodVision-V2.1-upgrade.zip` 交付；Linux Docker 功能检查通过，详见下文 | 未核实 |
| `goodvision-system1` | **VideoJev V2.0** | 运行基线 `goodvision-system1:pr75-603f20d`；另已交付共享镜像 `goodvision-system1:v2.1-77362bd` | `videojev_service`、VideoJev-4B；`goodvision-system1`，8097 | V2.0 真实 GPU 推理与重启通过；V2.1 共享镜像已构建并装入原升级包，VideoJev 程序与已打包运行依赖未升级 | 未核实；不能凭 V2.1 artifact 推断服务已更新 |
| `goodvision-bbox` | V2.0 | `goodvision-bbox:pr75-9a7dd24-{cuda-amd64,cpu-amd64,cpu-arm64}` | YOLO26-S 检测与追踪；`goodvision-bbox` | 已构建、已交付；CUDA/amd64 的 V2.0 GPU 推理与重启通过；其余类型不沿用此 GPU 结论 | 未核实 |
| `goodvision-contour` | V2.0 | `goodvision-contour:pr75-4c10721-api` 与 `:pr75-4c10721-engine` | API 与推理引擎；`goodvision-contour-api`、`goodvision-contour-engine` | 两镜像已构建、已交付；V2.0 GPU 推理与重启通过 | 未核实 |
| `goodvision-gpu-runtime` | V2.0 | `goodvision-gpu-runtime:pr75-4c10721-torch2.10-cu128` 与 `:pr75-4c10721-torch2.8-cu128` | PyTorch/CUDA 依赖镜像；**无常驻容器** | 已构建、已交付；RTX 5090 CUDA 算子检查通过，底座本身不代表业务模型推理通过 | 未核实镜像是否已导入 |
| `goodvision-neural` | **V2.0**（安装器交付修订 r2） | `goodvision-neural:v2.0-603f20d-20261112` | `neural_service`、CLIP ViT-L/14；`goodvision-neural`，8098 | 已构建；r2 修复首次失败后的端口重试；144 项相关检查通过、2 项跳过；新容器真实 GPU 推理未验收 | 未安装验证；现场部署未知 |

`system1` 在旧安装器中是模块短 ID；服务是 VideoJev。旧共享镜像同时携带 CLIP 代码，
但不会同时运行两个服务。V2.1 包的 `system1/update.json` 确实标记 artifact 版本 `v2.1`；
该历史字段如实保留。它表示共享镜像发布，不代表 VideoJev 功能或独立 CLIP 安装已升版。

## 镜像与源码、构建对应

以下 ID 是 Docker image ID，不是远端 registry digest。历史原始清单保存在
[V2.0 组件快照](../validation/evidence/container-deliveries/v2.0-components.json)和
[V2.1 发布快照](../validation/evidence/container-deliveries/v2.1-release.json)。

| 镜像 tag | 完整 image ID | 应用源码 / 构建关系 |
| --- | --- | --- |
| `goodvision-main:pr75-603f20d` | `sha256:26b03fbe21d6be0f704554189abc5487e9997ca3129eead1ba093f67f548f307` | `603f20dc0089cc0b0f21429d8432f3c32aa4b2a6`；V2.0 Main |
| `goodvision-main:v2.1-77362bd` | `sha256:d2d7178bdd5c869ea431a80a39c801ee3f87af7db7f98400b29e6693fcea6179` | 应用 `77362bd30f5a31cbaaeb16798d8d0c483e2daa2c`；复用上行镜像的增量层；合并提交 `8f526925127f9336c46679713b602f7c7d815a5a` |
| `goodvision-system1:pr75-603f20d` | `sha256:b7e2098b9aa4e6dcf98810797d490df5993027db31e476ad22d326e73750f779` | `603f20dc0089cc0b0f21429d8432f3c32aa4b2a6`；V2.0 共享运行依赖和应用，默认运行 VideoJev |
| `goodvision-system1:v2.1-77362bd` | `sha256:5f9b0e677e2acf5a9a8e8e6175a068052f1afd4f4b84b7bf36166ac3a2cf09e4` | 应用 `77362bd30f5a31cbaaeb16798d8d0c483e2daa2c`；从上行镜像导出增量，保留其运行依赖；CLIP 代码有变化 |
| `goodvision-neural:v2.0-603f20d-20261112` | `sha256:de292f54234aadbd00cfbdb8b627cfe64676546e39b9b8967a88dc7395c3c42c` | 原 V2.0 应用 `603f20dc0089cc0b0f21429d8432f3c32aa4b2a6`；新增独立 CLIP 入口和健康检查，原 System1 层不改动；构建源码哈希见本轮 RELEASE |
| `goodvision-bbox:pr75-9a7dd24-cuda-amd64` | `sha256:bedcc097ae196b9d2271cafd1cac3a944c19e7cb26b8332a9646ba41660f7141` | `9a7dd242960d57aa6c0dd264638031823e3c7e81`；CUDA/amd64 |
| `goodvision-bbox:pr75-9a7dd24-cpu-amd64` | `sha256:7b183bd0ed28723739c9371d049f17308f5ab7f8b21551c01d137fa89ae55c13` | 同上；CPU/amd64 |
| `goodvision-bbox:pr75-9a7dd24-cpu-arm64` | `sha256:2db54689806f03528a02ba4f6335a8ca8feb9bf4625d5855cf0c5ffc1de1ad04` | 同上；CPU/arm64 |
| `goodvision-contour:pr75-4c10721-api` | `sha256:6c2d506fa31da8699d9ba62db97f9171e609f635f2bc477e5f9dad34844a546c` | `4c1072144a7dc1241bccf35cc2941efb23c48125`；Contour API |
| `goodvision-contour:pr75-4c10721-engine` | `sha256:a13ccf02181466e7cd4d3ec236b8ac336e816064be1917e2151f62a6ba537548` | 同上；Contour 推理引擎 |
| `goodvision-gpu-runtime:pr75-4c10721-torch2.10-cu128` | `sha256:b60f2440b80eb3483e7aaacdba3fb8e0a19b36e894c930ab904660e9ec081acf` | 同上；PyTorch 2.10.0、torchvision 0.25.0、CUDA 12.8；供 Contour/CUDA Bbox 使用 |
| `goodvision-gpu-runtime:pr75-4c10721-torch2.8-cu128` | `sha256:61d629ebeb68ac0700c396d325f04a650f1dde6253920acabdf88d231245ea98` | 同上；PyTorch 2.8.0、torchvision 0.23.0、CUDA 12.8；供 System1 使用 |

构建入口：Main [Dockerfile](../../Dockerfile)，VideoJev
[System1 Dockerfile](../../workers/system1/Dockerfile)，其余模块入口见[上方命名与职责](#独立安装单元)。
当前 checkout 的 Dockerfile/requirements 不一定与历史镜像内容相同，复现历史交付应以对应提交和镜像 ID 为准。
特别是 V2.1 增量交付复用了 V2.0 依赖层，不能把当前源码 requirements 的变化当作该镜像已安装的依赖变化。

## 各模块发布历史

| 模块 / 版本 | 变更与交付关系 | 验证边界 |
| --- | --- | --- |
| Main V2.0 | 标准 V2.0 初始交付，后续 Main 单独更新 | 旧 GPU 验收中的控制台与工作流联调 |
| Main V2.1 | 动态视频源和独立总结配置保留；共享人脸库同步；总结草稿放弃；结果 WebSocket 恢复与动态 worker 等修复。原升级包已交付，保留原文件及校验值 | Linux Docker、视频连接、确定性总结接口、配置保留、升级/回滚通过；未进行本轮真实 GPU 推理或付费 API 验收 |
| VideoJev V2.0 | V2.0 独立 `goodvision-system1` 容器，`ZhuYuhan/VideoJev-4B`；本轮不改服务、模型或依赖 | 2026-11-03 RTX 5090 真实推理、独立会话、重启通过 |
| System1 共享镜像 artifact V2.1 | 原 `GoodVision-V2.1-upgrade.zip` 中已包含；更新应用副本。CLIP 增加完整画面等比例补边、逐帧窗口、两帧采样、批处理和可选阈值等；VideoJev 服务代码未变 | 镜像导入与源码核对通过；V2.1 CLIP 未重新通过真实 GPU 推理验收；标准 V2.0 没有常驻 CLIP，原包不会创建它 |
| Bbox V2.0 | YOLO26-S 检测、追踪；CPU/CUDA 和架构是同一模块的构建类型 | 旧 GPU 验收限 CUDA/amd64；不据此声称全部类型实机验收 |
| Contour V2.0 | API 与 engine 作为同一模块的两个角色交付；两者配套维护 | 旧 RTX 5090 真实推理、独立会话、重启通过 |
| GPU runtime V2.0 | 两个依赖镜像独立于业务服务，未创建常驻容器 | 旧 RTX 5090 CUDA kernel 与 torchvision NMS 检查通过 |
| CLIP 独立容器 V2.0（本轮） | 首次制作 `goodvision-neural`。沿用原 V2.0 `goodvision-system1:pr75-603f20d` 应用和依赖层；显式 CLIP 入口、8098、GPU、离线模型挂载、健康检查与持久重启；本轮不引入 V2.1 CLIP 算法变化 | 新容器与安装工具的实际结果在本轮交付清单登记；原 GPU 结论只证明原模型/应用在临时串行环境可运行 |

## 原 V2.0 CLIP 验收与正式安装的区别

原 V2.0 `server_install.py` 的模块列表仅为 `contour,bbox,system1,main`，固定常驻容器
共五个，包含 Contour 的两个角色。[原交付清单](../validation/evidence/container-deliveries/v2.0-handoff.json)
中 CLIP 权重已随模型包交付，但没有 `goodvision-neural` Compose、常驻容器或启动入口。

[真实 GPU 验收脚本](../validation/evidence/container-deliveries/v2.0-gpu-acceptance-runner.py)
先停止 VideoJev，再用 System1 镜像启动临时 `goodvision-system1-clip-check`，
执行 CLIP 推理和 restart 检查，随后删除临时容器并恢复 VideoJev。
[GPU 验收记录](../validation/pr75-gpu-v2.0.json)中的 CLIP 成功结果对应这次临时运行。
这不能证明客户收到常驻 CLIP，不能证明 VideoJev 与 CLIP 并发显存足够，
也不能替代新容器或 V2.1 CLIP 代码的 GPU 验收。

独立 CLIP 安装应复用[原模型清单](../validation/evidence/container-deliveries/v2.0-model-manifest.json)中的
`state/models/neural/huggingface`：
`openai/clip-vit-large-patch14` snapshot `32bd64288804d66eefd0ccbe215aa642df71cc41`，
原 `model.safetensors` 为 1,710,540,580 bytes。保留模型、原 settings、Main、
VideoJev、Bbox、Contour 的配置、数据卷及密钥；不能因新增模块要求重新下载整包。

## 本轮独立 CLIP 构建与交付

当前交付修订为 **V2.0-r2**，仅修复安装器在首次失败后调整宿主端口重试的流程。
服务版本与镜像 ID 保持 V2.0；已成功、被自行修改或属于其他安装的配置仍会拒绝覆盖。
[PR #109](https://github.com/Numaira-Technology/goodvision-harness/pull/109) 审阅指出此问题，
修复提交为 `69e45012c2dbb94ef4f71489278cbd8d413ecca8`；新增五项回归测试，
相关检查共 144 项通过、2 项跳过。r2 使用独立文件名，原包及原校验记录保留。
[r2 发布清单](../validation/evidence/container-deliveries/neural-v2.0-r2-release.json)、
[r2 交付校验](../validation/evidence/container-deliveries/neural-v2.0-r2-handoff.json)、
[r2 验证范围](../validation/neural-container-v2.0-r2.json)。

当前包：`dist/GoodVision-Neural-V2.0-r2.zip`，105854 bytes。
SHA-256：`03a20c49ffdcbe1f4b59adbf804e46dede9a75ca2c0466b6136c304f3d17417d`。

以下为首次交付的历史记录：

交付文件为 `dist/GoodVision-Neural-V2.0.zip`，102,330 bytes（约 100 KB）。
SHA-256：`d34923e75a921c7bda54dfe864702e5edf815d0605d9043fa82cb0da59b44de3`。
本轮单独提供此新包，原 `GoodVision-V2.1-upgrade.zip` 未修改。
[发布清单](../validation/evidence/container-deliveries/neural-v2.0-release.json)、
[交付校验与导入记录](../validation/evidence/container-deliveries/neural-v2.0-handoff.json)、
[验证范围](../validation/neural-container-v2.0.json)在 repo 持久保存。

专用镜像构建命令（仓库根目录）：

```bash
docker build --platform linux/amd64 --network none --pull=false --provenance=false \
  -f workers/neural/Dockerfile -t goodvision-neural:v2.0-603f20d-20261112 .
```

新镜像基于精确 V2.0 System1 image ID（见上表），保留原应用及 CUDA 依赖；
两处应用副本各 134 个 Python 文件已与 `603f20d` 核对一致。新构建配置的 config ID
为 `sha256:74c97df9c11517d4a5633d97c7a2b5d3adcbfee1a088616a8b182c7d6276fd7f`。
本轮构建入口、安装器和 CMD 共九个源文件以 RELEASE 中的 SHA-256 定位；
`repository_revision` 为首次打包时的基准 checkout `8f52692`；当时新增文件由清单中的
SHA-256 定位，不能将这些文件声称为该基准提交已有内容。

包不含模型。8,409 bytes 的 OCI 增补归档省去已有的 10,123,533,969 bytes 镜像 blob，
依赖原 V2.0 镜像且仅适用于 containerd image store；最终归档已实际 `docker load` 核验。
通用安装推荐 `--offline-build`，从经过身份核验的原镜像与包内最小源码离线构建。
该方式已在本机 BuildKit 验证；classic store 的 ID/config ID 处理有自动测试，
没有另做 classic engine 集成验收。不同构建器产生的实际 ID 独立登记，不冒充预构建 ID。

解压后，Windows CMD 执行以下命令（将路径换成原安装目录）：

```bat
Install-Neural.cmd --goodvision-dir "D:\system files\GoodVision" --offline-build
Status-Neural.cmd --goodvision-dir "D:\system files\GoodVision"
```

Linux 命令及构建/启动说明见[CLIP 交付说明](../../workers/neural/README.md)。
首次安装不要求升级 System1，也不写原模块的 settings、Compose、数据卷、模型或密钥。
安装结果单独写入 `state/neural-deployment.json`，并核验实际容器镜像与健康。

本轮本机隔离验证覆盖入口、只读挂载、网络别名、健康拒绝未就绪/错误模型及重启；
HTTP 健康测试使用明确的模拟服务，不是真实推理。无 GPU 时默认入口按预期报错退出。
真实 NVIDIA 模型推理、VideoJev/CLIP 并发容量及 Windows CMD 实机仍待验收；
本次没有操作客户服务器，不能将交付完成写作现场已部署。

## 后续更新规则

1. 每个模块独立推进版本。首次 V2.0，后续按该模块实际交付变化选择版本；
   只改变 Main 时不改其他模块。服务版本、共享镜像 artifact 版本、模型版本分别记录。
2. 构建前确定固定模块名、service、入口、端口、模型和依赖基线。记录应用提交、
   Dockerfile/构建器提交或文件 SHA-256、基础镜像完整 ID/digest、平台和完整构建命令。
3. 构建后先登记实际 tag + image ID/digest、归档/安装包名称、大小和 SHA-256，
   再登记功能变更及兼容范围。远端 registry 地址和 digest 仅在真实推送后填写。
4. 验证逐项记录实际环境与结论，分别标记静态检查、CPU/模拟运行、真实 GPU 推理、
   健康、重启、更新/回滚及配置保留。旧模块、旧代码或串行验收结论不能迁移到新组合。
5. “已构建”“已打包”“已交付”和“现场已部署”分开更新。现场需要带主机/实例、时间、
   实际 image ID 和健康结果的部署证据；未知就保留“未核实”，不能从本地 tag 或配置目标推断。
6. 发布时在同一源码变更中更新本目录的命名、版本记录及对应发布清单。
   保留历史清单及其校验值。已交付 ZIP 不覆盖同名重发；修订另给唯一文件名和修订记录。

## 持久证据

- [证据来源和原文件 SHA-256](../validation/evidence/container-deliveries/sources.json)：
  记录原交付目录和本次只读快照，换机器后仍可从 repo 核对历史。
- [V2.0 组件](../validation/evidence/container-deliveries/v2.0-components.json)、
  [V2.0 handoff](../validation/evidence/container-deliveries/v2.0-handoff.json)、
  [V2.0 GPU](../validation/pr75-gpu-v2.0.json)、
  [临时 CLIP 运行脚本](../validation/evidence/container-deliveries/v2.0-gpu-acceptance-runner.py)。
- [V2.1 发布](../validation/evidence/container-deliveries/v2.1-release.json)与
  [V2.1 验证](../validation/evidence/container-deliveries/v2.1-validation.json)。
  验证快照的 CI 总结中保留了抓取时尚未结束的 Windows 作业，不能将其改写为全通过；
  Linux Docker 已接受检查和本地验证按快照字段读取。
