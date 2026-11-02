# Bbox、System 1 和神经网络模块部署

画布中三个模块都有独立的“运行位置”，设置随工作流应用并持久保存。

| 运行位置 | 调用方式 |
| --- | --- |
| 本机 | 启动仓库现有服务，无需在节点填写 URL |
| 独立服务器 | 调用该节点保存的独立服务器地址 |
| 云服务器 | 调用该节点单独保存的云服务器地址 |

本机指 **Harness 分析主机**。通过网关连接另一台分析主机时，模型在分析主机运行。
独立服务器与云服务器地址分别保留，切换位置不会覆盖另一项。已有填写
`service_url` 的工作流继续按独立服务器调用；空地址旧节点和新建节点默认本机。
云服务器连接的是已部署的 GPU 服务，选择该选项不会创建租用实例。

## 本机运行

| 模块 | 复用实现 | 默认本机地址 |
| --- | --- | --- |
| Bbox | `workers/detector_rich_label/app.py`，YOLO26-S + 会话独立追踪器 | `http://127.0.0.1:8096/v1/labels` |
| System 1 快速识别 | `goodvision_harness.videojev_service`，VideoJev-4B | `http://127.0.0.1:8097` |
| 神经网络识别 | `goodvision_harness.neural_service`，CLIP 固定视觉分类 | `http://127.0.0.1:8098` |

这些端口避开原有 SAM、Contour、detector 隧道和旧版分类服务默认端口。
Bbox 支持 CPU 和 NVIDIA CUDA；System 1 和当前神经网络服务仍需要支持的显卡。
主程序普通 CPU 镜像不包含这些 GPU 依赖。
VideoJev 沿用 Linux CUDA 环境和固定版本原生扩展，不能直接当作 Windows CPU 分类器。

本机 Bbox 使用独立环境和固定依赖，运行：

```powershell
# 使用已有 Python；安装到 .venv-module-bbox，权重放 .models/modules/yolo。
python tools/setup_bbox.py --device cpu
# NVIDIA 主机可改用 --device cu128。
```

Apple Silicon Mac 使用 Python 3.12 和同一 `--device cpu` 命令；安装器选择
macOS PyPI wheel 与兼容的 NumPy/OpenCV 版本，不接受 macOS CUDA 安装。

安装后主程序自动发现 `.venv-module-bbox`。它不会向主程序环境安装 PyTorch。
本机默认优先可用 CUDA，否则 CPU；显式设置 `DETECTOR_DEVICE` 后不会自动切换设备。
如需完全独立于主程序生命周期运行：

```powershell
$env:PYTHONPATH = 'src'
.\.venv-module-bbox\Scripts\python.exe -m goodvision_harness.module_service bbox --singleton
```

Linux 使用 `.venv-module-bbox/bin/python`。主程序会检查并复用已经运行的服务。
其他模块在独立 Python 环境安装项目对应的 `.[videojev]` 或 `.[neural]`。
CUDA PyTorch 要与目标主机兼容；VideoJev 还需按
[现有环境说明](videojev-candidate.md)和 `tools/videojev_probe/requirements.txt`
准备固定依赖、原生扩展及权重。Bbox 需保留仓库的 `workers/detector_rich_label` 源码；
默认 Docker 构建也会复制它。

启动 Harness 前，可指定各模块已配置好的 Python，例如：

```powershell
$env:GOODVISION_BBOX_PYTHON = 'D:\model-envs\bbox\Scripts\python.exe'
$env:GOODVISION_NEURAL_PYTHON = 'D:\model-envs\neural\Scripts\python.exe'
```

Linux 示例：

```sh
export GOODVISION_SYSTEM1_PYTHON=/data/videojev/env/bin/python
export GOODVISION_MODULE_MODEL_DIR=/data/models
export GOODVISION_MODULE_DATA_DIR=/data/module-services
```

未指定 Python 时优先使用仓库 `.venv-module-<模块名>`，不存在时使用 Harness 当前 Python。
模型目录默认 `.models/modules`；
服务日志与进程锁默认 `.goodvision/modules`。Bbox 可继续使用
`DETECTOR_WEIGHTS_DIR` / `DETECTOR_WEIGHTS` 指定已准备的 YOLO 权重。

停止分析后，在节点侧栏选择“本机”，点击“启动 / 检查本机服务”。页面会显示
启动中、已就绪或失败，启动中自动刷新。就绪后应用工作流并开始分析。
直接开始分析也会触发选中的本机模块启动；未就绪时返回提示，不会误报分析成功。
模拟分析不自动启动模型。

健康检查核对具体模型和 `ready`。Bbox 的 CPU 服务须明确返回
`device=cpu`、`runtime=ultralytics`、`provider_ready=true` 和 `gpu_ready=false`，
GPU 服务仍要求 `gpu_ready=true`。同主机摄像头进程
用操作系统锁避免重复加载模型。服务保持预热，直到创建它的主程序退出；关闭时
只结束自己创建的服务。失败日志为 `.goodvision/modules/bbox.log`、
`system1.log` 或 `neural.log`。

端口已有服务但健康响应失败、模型不匹配或响应超时时，不会另起模型进程争抢
该端口。先修复或停止原服务再重试。CLIP 启动入口完成一次真实推理预热后才
开放健康接口，权重加载成功本身不算推理就绪。

## 独立服务器或云服务器

在相应 GPU 主机的仓库目录和已配置依赖的环境中，分别运行所需服务：

```sh
python -m goodvision_harness.module_service bbox --host 0.0.0.0 --port 8096
python -m goodvision_harness.module_service system1 --host 0.0.0.0 --port 8097
python -m goodvision_harness.module_service neural --host 0.0.0.0 --port 8098
```

每条命令为独立常驻进程，分别使用对应 Python 环境。未安装项目时设置 `PYTHONPATH=src`。
默认仅监听 `127.0.0.1`，上例显式监听服务器网络接口。通过内网或已有访问代理连接。

节点选择相应位置后填写实际地址，如 `http://192.168.10.20:8096` 或
`https://models.example.com/system1`。Bbox 根地址自动补 `/v1/labels`，也兼容
完整检测接口。识别服务可填根地址或完整 `/predict` 地址。URL 不包含凭据、查询或片段。
分析仍通过原有视频采样、追踪 / 事件结果和输出画面链路执行，总结使用原有共享 VLM。

## Docker 与多摄像头

共享 GPU 依赖底座与独立模块镜像见
[共享 GPU 运行环境](../../workers/gpu_runtime/README.md)。
`compose.modules.yaml` 独立管理 Bbox 和 System 1，不包含主程序，也不替换主程序的数据卷。

Harness 在 CPU 容器中时，可将 GPU worker 部署在同宿主机的独立共享环境。
给每路分析容器传入对应地址：

```text
GOODVISION_BBOX_LOCAL_URL=http://host.docker.internal:8096
GOODVISION_SYSTEM1_LOCAL_URL=http://host.docker.internal:8097
GOODVISION_NEURAL_LOCAL_URL=http://host.docker.internal:8098
```

Linux Docker 需要 `host-gateway` 映射，也可使用共享 worker 容器服务名。
此时本机模式检查并复用预部署服务，不会在 CPU 容器内尝试创建远端地址对应的进程。
每路容器都必须收到相同的地址配置，以共享模型而非分别加载。

## 已验证范围

真实视频与运行中主程序的验收结果、共享 GPU 和 System 1 的待验收项见
[模块验证记录](../validation/workflow-modules.md)。
