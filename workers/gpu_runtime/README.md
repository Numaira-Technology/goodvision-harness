# 共享 GPU 运行环境

统一 Docker 安装器及独立源码安装包入口见
[独立 Docker 安装单元](../../docs/guides/docker-modules.md)。该入口为底座和模块增加
发布版本标签，并统一管理网络与持久数据；下面的命令仍是原有手动构建入口。

NVIDIA 驱动由宿主机统一提供；Docker 的 NVIDIA Container Toolkit 也只配置一次。
`Dockerfile` 把 CUDA 12.8 开发运行库、Python 3.12、基础系统库做成公共镜像层，
再叠加固定版本的 PyTorch。模块镜像只安装自己的额外依赖和源码。
同一 Docker 主机上相同层只存一份，pip 构建缓存也共享。

| 模块 | PyTorch / torchvision | 共享范围 |
| --- | --- | --- |
| Contour | 2.10.0 / 0.25.0 | 与 Bbox 共用完整 GPU 底座 |
| Bbox / YOLO26-S | 2.10.0 / 0.25.0 | 与 Contour 共用完整 GPU 底座 |
| System 1 / VideoJev | 2.8.0 / 0.23.0 | 共用 CUDA/Python 层；保留已验证的独立 PyTorch ABI |

不能为了省空间把 VideoJev 的原生扩展放进 Contour 的不同版本 PyTorch 环境。
底座中的 pip constraints 防止模块安装时悄悄替换 GPU 二进制版本。
共享依赖不会共享运行内存或模型状态；每个模块仍是独立进程、模型缓存和日志。
这里使用 x86 Linux CUDA 镜像；Jetson 保留自己的 JetPack/ARM64 底座。

## 构建一次，模块分别部署

从仓库根目录执行，开发机先查看计划无需 Docker daemon：

```sh
python tools/build_gpu_runtime.py contour bbox system1 --plan
python tools/build_gpu_runtime.py contour bbox system1
docker compose -f compose.modules.yaml --profile bbox build bbox
docker compose -f compose.modules.yaml --profile bbox up -d bbox
```

Bbox 权重首次启动下载到独立 `bbox-models` 卷，随后复用。
System 1 需要先构建原生扩展和预下载固定版本模型：

```sh
docker compose -f compose.modules.yaml --profile system1 build system1
docker compose -f compose.modules.yaml --profile system1 run --rm --no-deps \
  -e HF_HUB_OFFLINE=0 --entrypoint python system1 -c \
  "from huggingface_hub import snapshot_download; from goodvision_harness.videojev_core import MODEL,REVISION; snapshot_download(MODEL, revision=REVISION, cache_dir='/models/huggingface')"
docker compose -f compose.modules.yaml --profile system1 up -d system1
```

System 1 正常服务阶段使用本地固定版本权重，启动包含真实模型预热。
构建和模型下载需要网络、足够磁盘及内存；运行需要支持 BF16 的 NVIDIA GPU。
不要把容器 HTTP 存活当成推理就绪，查看各服务 healthcheck。
默认端口仅绑定本机 8096/8097；远端部署通过内网代理或 SSH 隧道连接。

Contour 使用同一底座时追加覆盖文件：

```sh
docker compose -f workers/contour/compose.yaml \
  -f workers/contour/compose.gpu.yaml \
  -f workers/contour/compose.shared-runtime.yaml up -d --build
```

保留 Contour 原有模型、密钥及容量配置；不加覆盖文件时原有独立构建继续可用。
各模块停止/升级不要求重启主程序，主程序只连接相应 HTTP 服务。
Windows 无 NVIDIA GPU 时，Bbox 使用 `tools/setup_bbox.py --device cpu` 独立环境，
不安装这套 CUDA 镜像；System 1 不支持用这条 CPU 路径替代。

## 验证范围

2026-11-02：Windows CPU 的 YOLO 安装、真实视频请求和主程序集成在本机验证。
共享 GPU 镜像与 System 1 部署定义已提供，尚未在 NVIDIA 主机完成构建和推理验收。
用户明确要求暂不测试 System 1，因此本轮没有租用云 GPU。
