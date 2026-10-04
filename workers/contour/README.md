# GoodVision Contour · 轮廓引擎

可部署在独立 GPU 服务器上的视觉服务，配有极简中文控制台。
主程序通过带访问密钥的 HTTP API 创建视频会话、追加画面、获取人物框和轮廓。
控制台、外部接口、模型标识和错误信息使用 GoodVision Contour 品牌。

当前提供完整框架、内部推理适配器、Docker 配置、模型下载工具和交付打包工具。
框架模式不生成假检测结果，不会把 HTTP 服务存活当成模型就绪。
模型来源和授权文件保留在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 一键部署到普通 NVIDIA GPU 服务器

**Windows 10/11 GPU 服务器**使用独立的 `deploy_contour.ps1`，无需安装 Python。
先启动 Docker Desktop，选择 WSL 2 后端及 Linux 容器，并安装支持 WSL 2 的 NVIDIA Windows 驱动。
将脚本、含模型 ZIP 和镜像压缩包放在同一目录，然后在 PowerShell 中执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\Contour\deploy_contour.ps1" -PackageDir "D:\Contour"
```

将目录替换为实际值。默认访问 `http://127.0.0.1:8093/`；跨机器访问需增加
`-BindAddress "服务器内网IPv4"` 并允许客户端访问该端口。
密钥文件在解压目录 `goodvision-contour\workers\contour\secrets\node-key`。
默认离线部署；无镜像压缩包时可加 `-Build` 联网构建。`-CheckOnly` 只准备文件、镜像并检查 GPU，
不启动应用服务。Docker Desktop 的 GPU 支持前提见
[官方文档](https://docs.docker.com/desktop/features/gpu/)。
Docker Desktop 不支持 Windows Server 系统，详见
[系统要求](https://docs.docker.com/desktop/setup/install/windows-install/)。

**Linux GPU 服务器**使用下面的 Python 脚本：

适用于 Linux Intel/AMD 服务器，要求已安装 Python 3、Docker Engine、Compose v2、
NVIDIA 驱动及 NVIDIA Container Toolkit。将 `deploy_contour.py`、
`goodvision-contour-with-model.zip`、`goodvision-contour-images.tar.gz`
放在同一目录；有 `.sha256` 文件时也一起复制。将下面的路径和私网 IP 换成实际值：

```sh
sudo python3 /data/contour/deploy_contour.py --package-dir /data/contour --bind 192.168.1.100
```

默认离线部署，解压到包目录的 `goodvision-contour/`，校验模型、导入镜像、
生成本节点访问密钥、检查容器 CUDA，然后等待模型加载和预热成功。
不会自动下载文件、安装系统组件、改动防火墙或运行持续视频测试。
失败会返回非零状态；启动后的就绪检查失败会停止本节点推理容器并输出日志。
成功后访问 `http://服务器私网IP:8093/`，读取脚本输出位置的 `secrets/node-key` 登录。
8093 端口需对访问方可达；脚本不更改服务器的安全组或防火墙。

也可让 `--package-dir` 指向已解压的根目录。
框架包需额外指定 `--models /绝对路径/模型目录`；该目录须包含模型、来源清单和许可证。
已导入镜像时无需再次提供镜像压缩包。没有离线镜像但允许下载依赖时，增加 `--build`；
该模式只构建镜像，不下载模型。`--gpu 1` 可选择其他 GPU，`--port 18093` 可修改端口。
只有联网构建时 `--cuda cu126` 才会改变 CUDA 变体，不能修改已导出的离线镜像。

重复运行保留现有访问密钥、沿用已有解压目录，配置变更前会备份 `.env`。
更新源码部署包时请使用新的 `--install-dir`，不会自动覆盖已有安装。
本脚本不适用于 Jetson；该设备必须使用下述单独的适配与内存保护流程。

## 更新已有 Docker 部署

Windows 客户收到节点更新 ZIP 后，保持 Docker Desktop 运行，完整解压并双击
**`Update.exe`**，按窗口中的进度等待更新结果。无需安装 Python、输入命令或构建镜像；
模型、密钥、设置和数据保留。存在多个安装时，在窗口中选择要更新的节点。

更新独立 GPU 主机前，先在主程序控制台停止使用该节点的分析并等待收尾，
再在 GPU 主机运行 Contour 更新包；节点有任务时窗口会提示先停止分析。
主程序在另一台主机时，在那台主机运行主程序更新包。
维护者打包与源码部署方法见 [Docker 更新指南](../../docs/guides/docker-updates.md)。
原部署脚本继续用于初次安装。

## 先运行控制台

下面的命令在本目录执行，需要 Python 3.12 和 Docker Compose。

```sh
python initialize.py
docker compose up -d --build
```

打开 `http://127.0.0.1:8093`，输入 `secrets/node-key` 文件中的访问密钥。
初始化不会覆盖已有密钥。不要提交或分享此文件。页面只在内存中保留密钥，刷新后需重新输入。
这一步仅构建轻量服务，不安装 GPU 依赖或自动下载权重。

如需直接运行 Python，在仓库根目录执行：

```sh
python -m pip install -r workers/contour/requirements.txt
python workers/contour/initialize.py
python -m uvicorn workers.contour.app:app --host 127.0.0.1 --port 8093 --workers 1 --no-server-header --no-access-log
```

建议独立虚拟环境，避免改变主程序的依赖。API 和控制台由同一服务提供，不需要 Node.js 构建、外部字体或 CDN。

## 下载并接入模型

下载必须由操作人员显式运行，服务启动不会触发下载。
下载工具使用本地 `HF_TOKEN` 或 Hugging Face 登录缓存，并要求已获模型访问权限。
模型仓库及固定版本见 `download_model.py` 和下载后的来源清单。

```sh
python -m pip install -r requirements.txt
python download_model.py
```

它会在 `models/` 中生成 `contour.pt`、原始 `MODEL_LICENSE.txt`、
包含官方校验值的 `model-manifest.json`。中断后重新执行可续传；校验成功后才改为正式文件名。
也可将这三个已经校验的文件一起复制到 GPU 服务器，不需要在运行服务时提供下载凭据。

GPU 服务器需要兼容的 NVIDIA 驱动和 NVIDIA Container Toolkit。
复制 `.env.example` 为 `.env`，设置私网监听地址、GPU 编号和模型目录；
`CONTOUR_MODELS_DIR` 推荐使用 GPU 服务器上的绝对路径。
默认 CUDA 轮子为 `cu128`；需要 12.6 运行时的服务器设 `CONTOUR_TORCH_CUDA=cu126`。

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

首次 GPU 镜像构建还需要下载运行时依赖和固定版本的上游代码。
权重通过只读目录挂载，既不嵌入镜像，也不会在镜像构建中下载。
推理组件仅在容器网络内提供服务，宿主机只发布 Contour 的 8093 端口。
默认仅监听本机。跨机器使用时设置 `CONTOUR_BIND_ADDRESS` 为服务器私网地址，
通过私网/VPN 或 HTTPS 反向代理访问，客户端仍需携带访问密钥。

查看和停止本套服务：

```sh
docker compose -f compose.yaml -f compose.gpu.yaml ps
docker compose -f compose.yaml -f compose.gpu.yaml logs --tail=80
docker compose -f compose.yaml -f compose.gpu.yaml down
```

控制台的“暂停新会话”只停止接收新任务，已有会话可以继续追加或关闭。
它不关闭容器、不卸载模型；重启节点后恢复接收。暂停时就绪检查返回 503。
更换权重或设备配置后用 Compose 重新创建服务；恢复接收按钮不负责重新加载模型。

## 主程序在另一台服务器

将节点 URL 和密钥配置到主程序的 Contour 客户端，媒体帧通过 HTTP 上传；不需要共享文件系统。
完整协议见 [API.md](API.md)，纯 Python 调用示例：

```sh
python example_client.py --url http://GPU_SERVER_PRIVATE_IP:8093 --key-file /private/node-key frame-001.jpg frame-002.jpg
```

主程序源码已提供 Contour 客户端，支持节点认证、视频会话、轮廓和人物追踪结果。
部署包含此修改的主程序版本后，在部署设置中选择人物追踪“独立 GPU 服务器”、
节点类型“GoodVision Contour”，填写节点地址、实际部署容量和节点访问密钥。
详见 [主程序接入说明](../../docs/guides/modular-services.md#连接-goodvision-contour-节点)。
节点控制台的“登录节点控制台”只用于当前浏览器访问节点管理功能；
主程序需要单独保存地址和密钥。已有内部推理协议由 `engine_adapter.py` 适配。
通过 `CONTOUR_CAPACITY=2` 可启用两个独立视频通道；控制台和引擎必须使用同一个值。
模型权重共用，每路分别保留视频和人物追踪状态，GPU 操作排队执行。
Windows 新部署可使用 `deploy_contour.ps1 -Capacity 2`，Linux 可使用 `--capacity 2`。
扩容前停止全部分析；启动会验证两路状态隔离，实际吞吐仍需按显存和视频采样频率测试。

多路显存与延迟排查、固定提示词缓存开关，以及 1/2/6 路真实 GPU 对照命令见
[性能排查与压测](PERFORMANCE.md)。`CONTOUR_CAPACITY` 只控制可用通道数，不保证实时吞吐。
普通 GPU 部署默认启用固定 `person` 特征缓存以释放文本编码器显存；
设置 `CONTOUR_FIXED_PERSON_TEXT=false` 可作回退和 A/B 对照。需要重建运行容器才会生效。

## 配置

| 配置 | 用途 |
| --- | --- |
| `CONTOUR_API_KEY_FILE` | 密钥文件；Compose 默认挂载为 secret |
| `CONTOUR_API_KEY` | 可选环境变量密钥，优先于文件；不写入页面 |
| `CONTOUR_ENGINE_URL` | 内部引擎地址；留空为框架模式 |
| `CONTOUR_CHECKPOINT_PATH` | 节点看到的模型文件，用于展示挂载状态 |
| `CONTOUR_REQUEST_TIMEOUT` | 端到端请求时限，默认 70 秒 |

文件存在只代表挂载成功；`gpu_ready` 必须由内部引擎加载、预热成功后报告。
`/health/live` 表示控制台存活；带认证的 `/health/ready` 才表示可接收新推理会话。
容器存活健康检查用于观察控制台，不能代替模型就绪检查。

## 准备交付

在仓库根目录执行：

```sh
python workers/contour/package_release.py --output dist/goodvision-contour-framework.zip
python workers/contour/package_release.py --include-model --output dist/goodvision-contour-with-model.zip
```

仅导出明确列出的服务文件和内部适配组件；不收集 `.env`、访问密钥、主程序私有媒体或本地缓存。
含模型包强制包含原始许可证和来源清单，并再次校验模型。每个 ZIP 另有 SHA-256 校验文件。
包内是源码和配置，不是离线容器镜像。完全离线部署还需构建并导出目标架构的镜像。
Google Drive 上传在确认目标文件夹后单独执行。

### 离线迁移镜像

两份镜像成功构建后，可在仓库根目录导出到一个独立文件：

```sh
python workers/contour/export_images.py --output dist/goodvision-contour-images.tar.gz
```

导出工具压缩镜像、生成 SHA-256 校验文件，不包含运行中的容器或节点密钥。
将这个镜像文件与含模型 ZIP 一起带到 GPU 服务器。先解压 ZIP，
导入镜像，再进入解压后的 `workers/contour` 目录：

```sh
docker load -i /path/to/goodvision-contour-images.tar.gz
python initialize.py
docker compose -f compose.yaml -f compose.gpu.yaml up -d --no-build --pull never
```

运行前仍需按前面的部署说明配置私网监听地址、模型目录和 GPU 编号。
离线镜像保留构建时选定的架构及 CUDA 变体；更改 `.env` 中的构建参数不会改变已导入的镜像。
默认构建是 Linux amd64 / CUDA 12.8；需要其他变体时先在有网络的构建机器上重新构建并导出。
新服务器应重新生成自己的节点密钥，交付包不包含原机器的密钥。

Jetson / ARM 设备不能使用上述 amd64 镜像；适配配置和限制见 [JETSON.md](JETSON.md)。

## 验证范围

自动化测试覆盖无模型状态、认证、限流、超时、错误隔离、会话收尾、
实际内部 HTTP 协议和轮廓序列化、幂等重试及不改写客户标识。
测试替身不构成真实模型精度或 GPU 性能验证。目标 GPU 上仍需完成真实权重加载、预热和视频验收。

```sh
python -m pytest tests/test_contour_node.py -q
```
