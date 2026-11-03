# 独立 Docker 安装单元

统一入口为 `tools/deploy_modules.py`。它支持主程序、Contour、Bbox、System 1 和共享
GPU 基础镜像；每个服务单元使用独立 Compose 项目，按部署名称共用一张网络。
与旧的源码 Compose、Contour 独立安装器并存，不自动接管旧容器或迁移旧数据。

新构建与交付固定使用五个镜像仓库：`goodvision-main`、`goodvision-bbox`、
`goodvision-contour`、`goodvision-system1`、`goodvision-gpu-runtime`。
主程序镜像为 `goodvision-main:<version>`；Contour 接口和推理分别使用
`goodvision-contour:<version>-api` 与 `goodvision-contour:<version>-engine`。
CPU/CUDA、PyTorch 版本和内部角色通过标签区分，不另设镜像仓库名称。

## 安装单元与依赖

| 单元 | 运行内容 | 安装依赖 |
| --- | --- | --- |
| `main` | 主程序、视频接入、调度、结果；默认保持空闲 | Docker，无需 NVIDIA |
| `bbox` | 独立 YOLO26-S 服务 | 自动选择原生 CPU 或 NVIDIA CUDA；CUDA 使用共享 PyTorch 2.10 |
| `contour` | 对外接口与内部推理两个容器，作为一套服务管理 | NVIDIA、共享 PyTorch 2.10、Contour 权重 |
| `system1` | VideoJev 独立推理服务 | NVIDIA、共享 PyTorch 2.8、固定版本权重和原生扩展 |
| `gpu-runtime` | CUDA/Python 和两个兼容的 PyTorch 基础镜像 | 只构建，不运行常驻容器 |

CLIP 和自备 VLM 继续使用现有独立服务入口，本轮没有为它们增加客户安装包。
主程序仍可通过保存的 URL 连接它们。

需要 Python 3.10+、Docker Compose v2，以及运行 Linux 容器的 Docker。
GPU 主机另需 NVIDIA 驱动和容器 GPU 支持。共享依赖减少相同镜像层的重复存储，
不共享模型内存或推理状态；各模块的 GPU 编号通过 `--gpu-id` 指定。

新装 Bbox 默认 `--bbox-device auto`，按 Docker **引擎**的 Linux CPU 架构选择，
不按运行安装器的 macOS/Windows 系统猜测。提供 `linux/arm64`、`linux/amd64`
CPU 版本及 `linux/amd64` CUDA 版本，全部位于 `goodvision-bbox` 仓库下；
例如 `<version>-cpu-arm64`、`<version>-cpu-amd64`、`<version>-cuda-amd64`。
amd64 自动模式在容器内检查 GPU 是否可用；失败则使用 CPU。已有显式 CPU/CUDA
选择保留，也可用 `--bbox-device auto` 启用自动选择。显式 CUDA 不静默回退。
`plan` 和 `configure` 保持离线；自动探测在 `build`、`up` 和更新时执行。

Bbox 源码包首次安装可使用 `python install.py up --build`；安装器构建匹配镜像、
复用持久权重并等待模型预热。缺失的默认 YOLO 权重仍按既有行为首次下载。
交叉构建可显式指定 `build --modules bbox --bbox-device cuda --platform linux/amd64`，
不要求构建电脑具有 NVIDIA GPU。Contour、System 1 和 GPU 底座仍要求其支持的
NVIDIA 运行环境，不会因 Bbox 的 CPU 回退而改变运行设备。

## 在源码目录查看计划

以下命令仅输出配置，不启动 Docker、不写文件、不租用机器：

```sh
python tools/deploy_modules.py plan --modules main bbox --bbox-device cpu --namespace office
python tools/deploy_modules.py plan --modules main contour bbox system1 --bbox-device cuda --namespace office
```

`--namespace` 标识一套安装。所有同机模块使用相同名称；另一套安装应使用不同名称
和宿主机端口。默认状态目录是用户目录下的 `.goodvision/docker/<namespace>`。
使用 `--state-dir` 时，同一套安装的各模块须指定相同的绝对目录。
目录内保存安装设置、各模块 Compose 文件、模型和 Contour 密钥；不要提交到 Git。
新建状态目录仅供当前用户访问；新建的 Contour 模型挂载目录允许容器 UID/GID 10004
读取和进入，密钥目录保持私有。已有模型目录和文件的权限不会自动修改；若容器不能
读取，安装器会报错，需由目录所有者明确配置所需访问权限后继续。

## 在有 Docker 的安装主机执行

CPU 主程序和 YOLO：

```sh
python tools/deploy_modules.py configure --modules main bbox --namespace office --bbox-device cpu
python tools/deploy_modules.py build --modules main bbox --namespace office
python tools/deploy_modules.py prepare-models --modules bbox --namespace office
python tools/deploy_modules.py up --modules main bbox --namespace office
```

构建与模型下载分开。只有显式 `prepare-models` 才执行预下载；YOLO 也保留首次启动
下载默认权重的既有行为。首次安装需要网络，之后使用持久模型目录。Contour 下载
可能需要获准的模型访问权限及本机 `HF_TOKEN`；令牌不会写入计划、命令或安装包。
System 1 按代码中固定的模型版本预下载，下载成功不代表 GPU 推理验收完成。

在 NVIDIA 主机上安装 GPU 模块：

```sh
python tools/deploy_modules.py configure --modules contour bbox system1 --namespace office --bbox-device cuda --gpu-id 0
python tools/deploy_modules.py build --modules contour bbox system1 --namespace office
python tools/deploy_modules.py prepare-models --modules contour bbox system1 --namespace office
python tools/deploy_modules.py up --modules contour bbox system1 --namespace office
```

`build` 复用已安装的共享底座：Contour 和 YOLO 使用同一 PyTorch 2.10 底座，System 1
使用 2.8。底座缺失时才补建；显式 `build --modules gpu-runtime --version <版本>`
用于升级公共依赖。每个模块单独保存版本，更新 main 的版本不会推进其余模块或
GPU 底座的版本。main / System 1 的代码层与依赖层分开，普通代码更新复用依赖层。

`up` 不隐式构建或拉取应用镜像；可显式加 `--build`。缺少 Contour/System 1 模型时，
在启动任何选中模块前报错。容器启动后等待健康检查，模型服务需要真正预热就绪；
主程序检查的是界面/API 存活，保持分析停止。开始分析时还会核对所选工作流服务。
发生启动失败时使用 `logs` 查看原因，已运行的其他独立模块不会被自动删除。

## 单模块更新与回滚

收到一个模块的预构建更新 ZIP 后，运行一条命令：

```sh
python tools/deploy_modules.py update --bundle goodvision-main-v2.zip --namespace office
```

安装器从 ZIP 识别模块，只载入该模块的变更镜像层并等待健康检查。已安装的底层镜像
通过 ID 核对并复用；端口、服务地址、密钥、模型和数据卷沿用当前配置。健康检查失败
会自动恢复此前的模块。重复执行同一个更新包保留当前容器和原回滚点。
Contour 的接口与引擎镜像可以分别复用未变化的角色；其他模块不会被重新构建或重启。
GPU 底座更新只载入镜像，不启动容器或重启使用它的现有模块。

同一状态目录的安装、配置、更新和回滚按顺序执行。安装器取得目录锁后才读取
共享设置，避免同时更新不同模块时丢失版本记录，失败恢复也不会覆盖另一模块的更新。
导出时拒绝重复镜像角色；导入前核对 OCI 归档的全部顶层镜像名称与身份，拒绝未声明
的别名，并在载入前记录原镜像身份供回滚使用。

Bbox 统一更新 ZIP 可同时包含三个平台/设备变体。使用同一条 `update --bundle`
命令，安装器只校验、解包、载入当前环境选中的变体，不启动其他架构或设备的容器。
回滚同时恢复该模块的镜像、运行设备和平台，其他模块版本不变。更新不隐式拉取
探测镜像；初次源码构建在需要时获取 Dockerfile 同用的轻量 Python 基础镜像。
CPU/CUDA 切换同时更新设备对应的健康检查；CPU 必须完成真实模型预热，CUDA
仍要求 GPU 就绪。

也可解压 ZIP，使用包内 `python install.py update --bundle update.json`；Windows 可直接
运行 `update.cmd`。这些入口沿用首次安装所需的 Python 3.10+ 与 Docker。
自定义安装使用原来的 `--namespace` 和 `--state-dir`。

```sh
python tools/deploy_modules.py rollback --modules main --namespace office
```

解压包内也提供 `python install.py rollback` 和 Windows `rollback.cmd`。
回滚保留数据和模型，恢复上一个成功版本的镜像引用。更新包中的新增模型只补充缺失
文件；遇到不同的已有权重会停止，不覆盖用户模型。

发布方在完成本模块构建后导出独立更新包：

```sh
python tools/package_image_update.py main --version v2 --image goodvision-main:v2 --baseline previous/update.json --output dist/goodvision-main-v2
```

基线可以是首次完整镜像包的 JSON 清单，也可以是该模块上一个更新包的 `update.json`。
基线中已有的 OCI blobs 不重复传输；不指定基线则导出该模块完整镜像包。
增量载入使用 Docker 的 containerd image store；缺少指定基线时需先安装对应版本，
或使用发布方提供的单模块完整包。更新动作不自动下载权重或构建镜像。

## 容器网络、地址与生命周期

默认项目名固定为 `goodvision-main`、`goodvision-bbox`、`goodvision-contour`、
`goodvision-system1`；主程序、Bbox、System1 容器与其项目同名。
Contour 项目内的两个容器为 `goodvision-contour-api`、`goodvision-contour-engine`。
更新和回滚保持容器名称不变，版本号只在镜像 tag 中改变。GPU runtime 是底座镜像。
默认网络名 `goodvision-<namespace>`。安装器创建并标识共享网络，遇到同名但不属于
本安装器的网络时拒绝接管。各项目单独拥有自己的数据卷。
仅需要同时运行多套部署时，显式传入 `--namespaced-names` 才将安装名称追加到
项目和容器名之后；该选项持久保存，不随模块版本变化。默认固定名称已被其他部署
占用时拒绝接管，不会覆盖另一套程序的数据。
从早期安装器生成的无 `-api` 名称升级时，先备份状态目录并记录当前镜像 ID，
再用新安装包执行 `build`、`up`；Compose 按原项目及 service 识别并重建接口容器。
保留 namespace、状态目录和数据卷，不要删除卷或重新生成模型、密钥及服务地址。
回滚时使用备份的 Compose 文件和原镜像；仅修改 `settings.json` 不会回滚运行容器。

旧版带环境或测试版本后缀的项目用以下命令迁移到固定名称后再更新。
迁移仅短暂重建所选模块的容器，不构建或下载镜像；按容器实际挂载绑定原数据卷，
保留镜像、端口、环境、共享/内部网络和更新回滚记录。健康检查失败会恢复旧项目。
已迁移的项目重复执行不会重启；后续 configure/up 继续使用原卷。
状态目录中的 `naming-backup.*.json` 保留迁移前的配置。

```sh
python tools/deploy_modules.py rename --modules main bbox --namespace office --state-dir ./state
# 独立交付包：python install.py rename --namespace office --state-dir ./state
```

| 服务 | 主程序容器使用的同机地址 | 默认宿主机端口 |
| --- | --- | --- |
| 主程序 | 用户通过浏览器访问 | 8080 |
| Bbox | `http://gv-bbox:8096` | 8096 |
| System 1 | `http://gv-system1:8097` | 8097 |
| Contour | `http://gv-contour:8093` | 8093 |

Contour 推理容器只连接自己的内部网络，不能被其他模块直接访问。其接口容器连接
共享网络并保留访问密钥验证。第一次使用时，将状态目录 `secrets/contour-key` 中的
密钥保存到主程序的追踪节点设置；不要将它填进大模型 API Key。

主程序设置 `GOODVISION_MODULE_MANAGER=compose`，只探测并调用模块，不在自身
容器中启动 Python 模型进程，也不挂载 Docker socket。工作流侧栏在检查后显示
“检查模块服务”，启动/停止由宿主机安装工具负责。

```sh
python tools/deploy_modules.py status --modules main bbox --namespace office
python tools/deploy_modules.py logs --modules bbox --namespace office
python tools/deploy_modules.py stop --modules bbox --namespace office
python tools/deploy_modules.py up --modules bbox --namespace office
```

`stop` 仅停止指定项目，不删除容器、模型、数据卷或共享网络，不影响其余模块。
`status`、`stop`、`logs` 使用最后生成的 Compose 文件，不隐式改写部署配置。

宿主机端口和监听地址可以保存：

```sh
python tools/deploy_modules.py configure --modules main bbox --namespace office --main-port 18080 --bbox-port 18096
python tools/deploy_modules.py up --modules main bbox --namespace office
```

端口映射变化不改变同机容器地址。默认只发布到 `127.0.0.1`；远端连接必须显式
使用 `--bind <私网IP>`，并在受控内网/VPN 中部署。跨主机没有共享 Docker 服务名，
应使用实际服务器地址：

```sh
python tools/deploy_modules.py configure --modules main --namespace office --bbox-url http://192.168.10.20:8096 --system1-url http://192.168.10.30:8097
python tools/deploy_modules.py up --modules main --namespace office
```

也可以在工作流选择独立服务器/云服务器并保存 URL。安装器设置的是初始部署地址，
已有控制台保存的远端部署选项仍优先。地址不接受用户名、密码、查询参数或无效端口。
已有安装迁移到此入口时，需要先备份并明确迁移设置、模型和密钥，避免旧新实例抢占端口；
安装器不会自动改动原来的运行目录或原数据卷。

## 生成独立源码安装包

发布工具 `tools/package_modules.py` 为五个单元分别导出 ZIP。仅打包 Git 已跟踪的
对应源码及公共安装工具，拒绝模型、私有视频、本机环境和密钥。每个包包含
`install.py`、`Install.ps1`、中文说明、文件校验清单以及本模块所需源码；无需 Git。

只检查输入，不生成实际安装包：

```sh
python tools/package_modules.py bbox --version release1 --bbox-device cpu --check
```

在后续发布机器上执行真正打包：

```sh
python tools/package_modules.py main --version release1 --output dist/goodvision-main-release1.zip
python tools/package_modules.py contour --version release1 --output dist/goodvision-contour-release1.zip
python tools/package_modules.py bbox --version release1 --bbox-device cpu --output dist/goodvision-bbox-cpu-release1.zip
python tools/package_modules.py system1 --version release1 --output dist/goodvision-system1-release1.zip
python tools/package_modules.py gpu-runtime --version release1 --output dist/goodvision-gpu-runtime-release1.zip
```

GPU YOLO 包使用 `--bbox-device cuda` 并使用不同文件名。包内设备选项和发布版本固定。
源码包不包含预构建镜像、权重或凭据；接收方按说明执行 build、prepare-models、up。
它们属于“可独立构建安装的源码包”，不能宣称为已发布、已验收的离线镜像包。

解压后调用 `python install.py <动作>`，Windows 也可调用
`powershell -File Install.ps1 <动作>`。单个包只操作自己的模块，多个包使用相同
namespace 和状态目录完成组合安装。安装时先检查清单，文件缺失或被修改时拒绝执行。

## 本轮验证边界

本机 Docker 已完成 Main/Bbox CPU 安装、真实 YOLO 推理、摄像头视频画框、
增量更新及回滚检查；隔离环境另验证并发模块更新、真实健康失败后的自动恢复，
以及已有运行容器保持不变。详细版本和分阶段证据见
[模块验证记录](../validation/workflow-modules.md)。
CPU/CUDA 切换的健康契约有回归测试；真实 NVIDIA GPU 推理尚未验收，
Contour、System 1 和 CLIP 的 GPU 功能门槛仍保留。
