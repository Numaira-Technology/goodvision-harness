# Docker 模块命名目录（goodvision-xxx，内部）

维护代码、部署说明和更新记录时，统一使用 `goodvision-xxx` 格式的完整模块名称。
名称使用小写英文字母、数字和连字符，固定以 `goodvision-` 开头；不随版本、CPU/GPU
或安装地点变化。显示名称、安装器短 ID、Compose service、镜像和容器实例名分别记录。
本目录随 Git 持久保存，新增模块或调整映射时在同一提交中更新本目录。

2026-11-02 按五个固定模块名称核对安装器与交付配置，来自尚未合并的
[独立模块 PR #75](https://github.com/Numaira-Technology/goodvision-harness/pull/75)。
这里记录源码定义与命名约定；实际安装版本以该主机保存的配置及 Docker inspect 为准。
内部参考不代表任何镜像已构建、已部署或通过真实模型验收。

## 独立安装单元

以下五个完整模块名称同时固定为新安装、构建与交付使用的镜像仓库名称：
`goodvision-main`、`goodvision-bbox`、`goodvision-contour`、`goodvision-system1`、
`goodvision-gpu-runtime`。版本、设备和容器角色写在标签中，不另设镜像仓库。
安装器短 ID 对应
[安装器定义](../../tools/docker_modules.py) 的 `MODULES`、`deploy_modules.py --modules`
和源码包 `module-package.json.module`，继续按原值使用；完整名称不直接替代命令参数。
`<version>` 是发布版本，`<namespace>` 是安装实例名；尖括号为占位符。

| 固定模块名称 | 安装器短 ID | 职责 | Compose service | 新安装镜像命名 | 源码 / 构建入口 |
| --- | --- | --- | --- | --- | --- |
| `goodvision-main` | `main` | 主程序：界面、视频接入、工作流、事件与结果 | 新安装为 `main`；旧安装为 `harness` / `harness-N` | `goodvision-main:<version>` | [Dockerfile](../../Dockerfile)、[应用源码](../../src/goodvision_harness) |
| `goodvision-bbox` | `bbox` | Bbox：YOLO26-S 检测和追踪 | `bbox` | `goodvision-bbox:<version>-cpu` / `goodvision-bbox:<version>-cuda` | [CPU Dockerfile](../../workers/bbox/Dockerfile.cpu)、[CUDA Dockerfile](../../workers/bbox/Dockerfile)、[服务源码](../../workers/detector_rich_label) |
| `goodvision-contour` | `contour` | Contour：对外接口与内部推理，作为一个安装单元管理 | `contour`（接口）+ `engine`（推理） | `goodvision-contour:<version>` + `goodvision-contour:<version>-engine` | [接口 Dockerfile](../../workers/contour/Dockerfile)、[推理 Dockerfile](../../workers/contour/Dockerfile.engine)、[接口源码](../../workers/contour)、[推理源码](../../workers/sam_scene) |
| `goodvision-system1` | `system1` | System 1：VideoJev 视频推理 | `system1` | `goodvision-system1:<version>` | [Dockerfile](../../workers/system1/Dockerfile)、[服务源码](../../src/goodvision_harness/videojev_service.py) |
| `goodvision-gpu-runtime` | `gpu-runtime` | 共享 GPU 依赖底座；只构建镜像，无常驻容器 | 无 | `goodvision-gpu-runtime:<version>-torch2.10-cu128` / `goodvision-gpu-runtime:<version>-torch2.8-cu128` | [Dockerfile](../../workers/gpu_runtime/Dockerfile)、[底座说明](../../workers/gpu_runtime/README.md) |

新交付的主程序镜像必须使用 `goodvision-main`，不得继续输出 `goodvision-harness`。
下文列出的旧镜像名称只用于识别历史部署，不是新构建或交付可使用的兼容别名。
Contour 的容器角色固定写作 `goodvision-contour/contour` 和
`goodvision-contour/engine`（完整模块名称 / service）。
`engine` 不是第六个安装单元。CPU/CUDA Bbox 是同一模块的两种构建类型。
Contour 与 CUDA Bbox 共用 PyTorch 2.10 底座，System 1 使用 2.8 底座；更新共享底座后
需要重新构建、验证依赖它的模块镜像，不能只替换底座标签就视为应用已升级。

## 安装实例与持久化位置

以下规则由 [统一安装器](../../tools/deploy_modules.py) 生成，适用于上表的新安装。

| 对象 | 固定规则 | `office` 安装示例 |
| --- | --- | --- |
| Compose project | `goodvision-<namespace>-<module>`，`<module>` 使用安装器短 ID | `goodvision-office-bbox` |
| 容器实例 | 固定模块名后追加 `-<namespace>`；Contour 推理角色再追加 `-engine` | `goodvision-main-office`、`goodvision-contour-office-engine` |
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

同一安装各模块保持相同 namespace 与状态目录。升级保留这些标识、模型、密钥、数据卷
及保存的服务地址。`settings.json` 的版本是构建/配置目标，不能单独证明容器已运行该版本。
镜像标签也不能唯一证明内容；更新记录应同时记录实际 image ID / digest 与源码提交。

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
| `goodvision-neural`（预留） | 无 Docker service / 安装包 | [神经网络服务源码](../../src/goodvision_harness/neural_service.py) | 工作流中的 CLIP 服务入口；默认连接配置为 `http://gv-neural:8098`，安装器不会因此自动创建该服务或别名 |

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
新的 `bbox` / `system1` / `gpu-runtime` 也没有因此获得 Windows 双击更新包支持。
新安装单元目前使用统一安装器的版本化 build / up 流程，具体步骤见
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
- 新增模块先登记 `goodvision-xxx` 完整名称、安装器短 ID、职责、service、镜像与源码入口，
  再扩展安装器或更新器。显示名称可以调整；已持久化的模块名称和短 ID 不改义、不复用。
  改 service 或镜像仓库需保留旧名映射。
- 同一模块的 CPU/CUDA、x86/Jetson、模型版本与发布版本作为独立字段记录。
  需要不同接口或生命周期管理的服务，在本表中单独登记角色与兼容边界。
- 验证命名时用安装器 `plan` 及源码核对；真实镜像、GPU 推理和更新回滚需要各自验收。
  实测结论继续写入 [验证记录](../validation/README.md)，不在本目录重复保存运行日志。
