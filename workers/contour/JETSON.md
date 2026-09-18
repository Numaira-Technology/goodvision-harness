# Jetson 部署

Jetson 使用 ARM64 和集成 GPU。普通 Linux amd64 / CUDA 镜像不能直接导入运行。
本配置面向 JetPack 6.2.x，采用 NVIDIA 的 `25.06-py3-igpu` 基础镜像。
构建时保留它提供的 PyTorch、TorchVision 和 Triton 版本，不安装 x86 CUDA 轮子。

在本目录配置 `.env` 中的私网监听地址、模型目录，以及设备组编号：

```sh
getent group video render
python3 initialize.py
sudo docker compose -f compose.yaml -f compose.jetson.yaml build
sudo docker compose -f compose.yaml -f compose.jetson.yaml up -d --no-deps --no-build contour
sudo systemd-run --unit=goodvision-contour-guard --collect \
  --property=OOMScoreAdjust=-900 \
  /usr/bin/python3 "$(pwd)/run_jetson.py"
```

`CONTOUR_VIDEO_GID`、`CONTOUR_RENDER_GID` 应与服务器的组编号一致。
需要事先安装 Docker、Compose 和 NVIDIA Container Toolkit，并启用 NVIDIA runtime。
账户未加入 docker 组时，通过 `sudo docker ...` 管理容器。

监听地址必须是 Jetson 网卡实际持有的地址，可用 `ip -brief address` 查看。
若 SSH 地址是路由器转发入口，不能将该入口地址直接用作容器的主机监听地址。
在未转发 8093 端口的网络中，可从客户端通过 SSH 连接控制台：

```sh
ssh -N -L 127.0.0.1:18094:<Jetson-private-IP>:8093 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 <user>@<SSH-address>
```

连接保持期间，在该客户端打开 `http://127.0.0.1:18094/`，使用远程部署目录的
`secrets/node-key` 登录。其他机器需要自己的通道或到 Jetson 的可达路由。

这个适配版本限制为一个摄像头、一个人物目标，每次追加一帧。
API 的 `limits.max_append_frames` 返回 1，客户端必须遵守此值。
多帧批次在实际人物视频测试中触发了整机内存保护，未开放给该设备。
容器内存限额为 4500 MiB，
不使用交换空间，并限制为 4 个 CPU。CUDA 分配器另外限制为设备总内存的 55%，
它与容器的内存限额不能简单相加作为实际占用保证。启动时仍需观察整机内存。
`run_jetson.py` 在宿主机持续监测可用内存，低于 1024 MiB 时只终止经核对的本节点
推理容器。依赖 cgroup v2 的 `cgroup.kill`，无需等待 Docker 服务响应；
它不停止设备原有程序。不要绕过保护工具直接启动推理容器。
此临时服务不设置开机启动，推理容器不自动重启，
避免内存不足导致反复重启。只有完整加载和真实 CUDA 预热通过后才会报告模型就绪。
8 GB 是整机共享内存，文件存在或控制台运行不能证明推理已通过。

Jetson 构建使用 meta 设备装配结构，再从本地权重文件内存映射并直接赋值，
避免随机初始化和随后被丢弃的骨干网络占用内存。位置缓存按实际请求创建，
非持久文本掩码按上游代码重建；随机深度配置标量在 CPU 上生成。
GPU 上的二维及以上浮点参数存为 BF16，解码器要求 FP32 的前馈层保留 FP32，
向量参数与缓冲区维持原精度（包括复数位置编码）。原始磁盘权重文件未改写；
完整键名、形状和有限值校验仍保留。BF16 运行结果不承诺与 FP32 逐位一致，
功能测试不能代替实际业务数据上的精度评估。
模型结构和原生输入分辨率不裁剪。关闭 autocast 权重副本缓存，
校验、预热、视频追加和会话清理后释放闲置内存，降低共享内存占用。
连续视频首帧显式使用视频模式，避免上游把单帧启动误判为图片而绕过目标数量限制。
本接口只识别人物。首次使用真实文本编码器计算固定提示特征后缓存结果，
释放编码器权重占用的 GPU 内存；后续会话各自复制这些特征。
不接受替换提示词，扩展到其他类别时必须调整这一适配并重新验收。
修改前后的代码校验值记录在镜像内 `/opt/notices/jetson-patches.json`。

在一台 8 GB Orin 上，初版加载期间整机可用内存曾降至约 67 MiB，随后 SSH 失去响应。
重启后 SSH 已恢复；不能单凭这一现象确定 SSH 故障的全部原因。
后续试验中的内存保护已能停止推理而保持 SSH 和控制台可用。
实际部署的推理验收结果以独立的部署清单和真实 API 测试记录为准。

公开服务只接受上传的 JPEG 或 H.264 帧，由 Pillow / PyAV 解码。
此版本不安装仅提供 x86 轮子的 decord，也不开放上游的视频文件路径加载功能。

若控制台显示未就绪，先检查该节点的引擎日志和内存状态：

```sh
docker compose -f compose.yaml -f compose.jetson.yaml logs --tail=100 engine
docker stats --no-stream
sudo journalctl -u goodvision-contour-guard -n 30 --no-pager
```

停止保护服务也会停止对应推理容器，控制台继续运行：

```sh
sudo systemctl stop goodvision-contour-guard
```

本地原有 `goodvision-contour-images.tar.gz` 是 amd64 交付物，不是 Jetson 镜像。
Jetson 镜像使用单独的 `0.1.0-jetson` 标签，需在兼容设备上构建和验收。
