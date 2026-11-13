# GoodVision Neural V2.0 独立 CLIP 容器

首次补装 `goodvision-neural`，运行 `openai/clip-vit-large-patch14`，服务端口 8098。
沿用原 V2.0 CLIP 应用和模型，本次不升级 CLIP 算法。VideoJev 仍由
`goodvision-system1` 独立运行，端口 8097。
当前交付修订为 `V2.0-r2`：修复首次安装失败后的端口调整重试，服务和镜像仍为 V2.0。

## 安装

将本包解压到一个新目录，进入解压目录。保留原 GoodVision 目录及其 `state`。
需要 Python 3.10+、Docker Compose v2、Linux/amd64 Docker engine 和 NVIDIA GPU 容器支持。
原 V2.0 镜像 `goodvision-system1:pr75-603f20d` 和原模型必须仍在服务器上。

Windows CMD（把路径换成原 GoodVision 安装目录）：

```bat
Install-Neural.cmd --goodvision-dir "D:\system files\GoodVision" --offline-build
Status-Neural.cmd --goodvision-dir "D:\system files\GoodVision"
```

Linux shell：

```bash
python3 neural_container.py install --goodvision-dir "/srv/GoodVision" --offline-build
python3 neural_container.py status --goodvision-dir "/srv/GoodVision"
```

`--offline-build` 用已安装且身份核验通过的 V2.0 镜像层，在本机离线生成专用
CLIP 镜像；构建禁用网络，不安装依赖或下载模型。此方式同时支持 classic 和
containerd 镜像存储。不同构建器生成的 image ID 可能不同，安装记录会保存实际
ID 及本包预构建 ID，服务版本均为 V2.0。

包内也提供预构建镜像的 OCI 增补归档。仅在 Docker 使用 containerd 镜像存储时，
可省略 `--offline-build`，核验原基线后直接导入新镜像 metadata/新增层。
归档复用原镜像层，不是可用于空白机器的完整 CUDA 镜像。

安装会读取原 `state/settings.json` 和 `state/compose.main.json`，沿用原 GPU 编号、
共享网络、Main 的 neural 网络别名和绑定地址。默认增加宿主端口 28098，
如被其他程序占用，可在 install/start/status 命令均加 `--host-port 其他端口`。
首次安装因端口占用失败后，可用新的 `--host-port` 重新执行 install；仅允许
调整本安装器记录为未完成的配置中的宿主端口。已成功运行或自行修改的配置会保留。
可先执行 `python neural_container.py plan --goodvision-dir "原目录"` 查看配置。

安装器只生成 `state/compose.neural.json` 和 `state/neural-deployment.json`，
使用原安装锁协调写入。不会改写原 settings、Main/VideoJev/Bbox/Contour Compose、
数据卷、模型或密钥，也不会停止其他容器。模型目录
`state/models/neural` 只读挂载到 `/models`；包括原 CLIP snapshot 和 PPE 的
`yolo26s.pt`。容器显式启动 CLIP，健康检查要求 `ready=true` 且模型名称匹配。
启动成功前会等待 CUDA 模型预热和容器健康；失败会保留失败记录，不报告安装成功。

## 启动与检查

`goodvision-neural` 使用 `restart: unless-stopped`，随 Docker 重启恢复。
已安装后单独启动：

```bat
Start-Neural.cmd --goodvision-dir "D:\system files\GoodVision"
Status-Neural.cmd --goodvision-dir "D:\system files\GoodVision"
docker logs --tail 100 goodvision-neural
```

Linux 使用 `python3 neural_container.py start/status --goodvision-dir "/srv/GoodVision"`
（`start` 和 `status` 分别执行）。状态命令读取实际容器、健康和镜像身份。
原 `Start-Server.cmd` 继续负责原模块，新增 CLIP 使用本包入口。

本轮服务器原升级安排仅运行 `Update-Main.cmd`；System1 保持不动。
CLIP 在本包交付后另行安装，不需要执行原 `Update-System1.cmd`。

## 版本与验证

`RELEASE.json` 记录应用提交、原依赖镜像 ID、预构建镜像 ID、归档校验值及构建源码哈希。
仓库内 `docs/internal/docker-module-catalog.md` 是唯一的模块名称与独立版本目录。
本包内 `VALIDATION.json` 记录本次验证及局限；客户部署状态需要现场实际安装后记录。

本机没有 NVIDIA GPU；本次只验证 Linux 容器入口、原 V2.0 源码一致性、离线构建、
配置保留及隔离环境的挂载/健康/重启机制。原 RTX 5090 GPU 验收是在停止 VideoJev
后运行临时 CLIP，不能证明这次新增容器与 VideoJev 并发运行的显存足够。
本次未租用 GPU，未把模拟健康服务验证写作真实模型推理通过。

## 从仓库复现构建和打包

先核对基线 image ID 为
`sha256:b7e2098b9aa4e6dcf98810797d490df5993027db31e476ad22d326e73750f779`
（classic store 的 config ID 为
`sha256:e5685ef8ac313ef5455e9ac29c9ab1546d97b9b74cf552c45da26f0cd0d71ff0`）。
在仓库根目录执行：

```bash
docker build --platform linux/amd64 --network none --pull=false \
  -f workers/neural/Dockerfile -t goodvision-neural:v2.0-603f20d-20261112 .
python3 tools/package_neural_container.py \
  --image goodvision-neural:v2.0-603f20d-20261112 \
  --baseline "/path/to/original/GoodVision/goodvision-system1-v2.0.zip" \
  --output dist/GoodVision-Neural-V2.0-r2 \
  --validation docs/validation/neural-container-v2.0-r2.json
```

工具拒绝覆盖既有目录或 ZIP。后续版本独立发布，已交付包不覆盖同名重发。
本包的 `source/` 包含安装入口及最小镜像构建源，可直接用于服务器离线构建；
维护者打包工具须在完整仓库中运行。
