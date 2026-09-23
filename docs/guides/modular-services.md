# 主程序、SAM 和大模型分机部署

控制台入口、GPU 主机分析、SAM 和大模型可以分别更新和重启。
GPU 模式的 CPU 分析也运行在 GPU 主机；本机负责控制台与视频接入，不执行人脸和事件分析。

| 模块 | 部署内容 | GPU 和模型文件 |
| --- | --- | --- |
| 本机入口 `main` | 控制台、本机拉流与必要的接入解码、视频和结果转发 | 不加载识别模型，不创建分析进程 |
| GPU 主机分析 `analysis` | 视频解码、人脸识别、事件处理、结果和证据存储 | 使用 GPU 主机的 CPU；人脸模型保存在该主机的 `/models` 卷 |
| SAM `sam` | 多路人物追踪，共用一份 SAM 模型 | NVIDIA GPU、SAM3.1 checkpoint |
| 大模型 `vlm`，可选 | vLLM + Qwen3-VL-30B-A3B-Instruct-FP8 | NVIDIA GPU、完整的离线 Qwen 模型目录 |

例如本机入口放在客户机器，`analysis` 和 SAM 放在同一台 L40S 服务器，大模型继续调用云端。以后把大模型迁到另一台 GPU 服务器，只需修改控制台的大模型部署位置、服务地址和模型名称。

本页使用独立的 `compose.main-server.yaml`、`compose.sam-server.yaml`、`compose.vlm-server.yaml`，不调用云端租机。现有 `tools/start_local_gpu.py` 仍用于所有模块在同一台 NVIDIA 机器上的部署；两个入口不要在相同端口同时启动。

独立模块的 `--gpu-id` 是各自宿主机上的编号。如果两个 worker 实际在同一台双卡机器，应明确选择不同的卡；单卡共用请使用现有同机入口，以采用为 SAM 留出显存的配置。

## 1. 在 SAM 显卡机器上启动追踪服务

这台机器准备 Docker、Docker Compose 2.24.4 或以上、NVIDIA 驱动与 NVIDIA Container Toolkit，并准备实际的模型文件：

```text
.models/sam3.1/sam3.1_multiplex.pt
```

在仓库根目录执行：

```sh
python tools/start_modular.py sam --streams 12 --gpu-id 0
```

这里的 `--streams 12` 是此服务预留的摄像头通道数，可以是任意正整数，如 1、4、8、16、24。每个通道保留自己的会话和追踪状态，共享 GPU 模型。服务默认监听宿主机的 `0.0.0.0:8092`，也可以显式指定机器的局域网 IP：

```sh
python tools/start_modular.py sam --streams 16 --gpu-id 1 --bind 192.168.10.20 --port 8092
```

该入口只检查 NVIDIA 和 SAM checkpoint，只启动 SAM 容器。checkpoint 可通过这台机器的 `.env` 中的 `SAM31_CHECKPOINT_FILE` 修改。模型文件挂载到容器，启动时不会下载。

## 2. 在 GPU 主机启动 CPU 分析，本机连接控制台

先在 GPU 主机部署完整分析进程，云端供应商、模型和 API Key 配置放在这台机器的 `.env`，也可稍后在控制台保存：

```sh
python tools/start_modular.py analysis --streams 12 --tracking-server http://host.docker.internal:8092 --tracking-capacity 12 --bind 192.168.10.20
```

再在本机启动控制台入口：

```sh
python tools/start_modular.py main --streams 12 --tracking-server http://192.168.10.20:8092 --tracking-capacity 12
```

`main` 在独立 GPU 模式默认连接追踪主机的 8080 分析端口。分析端口不同或使用 HTTPS 时指定
`--analysis-server https://gpu-host.example`，也可设置 `HARNESS_ANALYSIS_URL`。
这一步只连接远端，不创建或修改远端通道；路数、模型线路和采样参数在 GPU 主机的
`analysis` 启动命令或控制台配置。只有旧版无 GPU 追踪的 `main` 入口仍在本机运行分析。
`/gateway` 可查看连接状态，远端故障返回 503，不会自动切回本机 CPU。

然后打开 [控制台](http://127.0.0.1:8080)，在每一路填入视频地址并连接。也可以预填 `LOCAL_CAMERA_01_STREAM_URL`、`LOCAL_CAMERA_02_STREAM_URL` 等。第一路兼容已有 `HARNESS_STREAM_URL` / `RTSP_URL`；默认启动后保持空闲，等待控制台连接。不同路数使用独立的数据卷，其他摄像头通过主控制台访问，只有主控制台发布端口。

“连接视频”只建立该路预览，不调用 SAM、大模型或人脸分析。每一路均可展开“详细控制”，保存检测项目、敏感度并查看结果；服务位置和密钥统一在主控制台配置。视频可分别连接，分析必须点击“开始全部分析”，一起处理所有已连接视频。停止使用“停止全部分析”，视频连接和预览会保留；本次分析启动、运行或收尾期间不能增减连接或修改检测与部署设置。接口接入使用 `/v1/fleet/cameras/cameraNN/connect`、`/details` 和 `/v1/fleet/analysis/start`、`/stop`，详见 [接口说明](../../API_DELIVERY.md)。

本机入口不会探测 NVIDIA，不会启动模型容器，也不会检查权重。人脸处理使用 GPU 主机 CPU，需结合该主机 CPU、视频分辨率和路数测试。

浏览器摄像头的控制消息和 JPEG 数据通过网关原样转发。RTSP/RTMP/WHEP 默认在本机拉流，
经必要的解码后由本机主动建立的连接转发到 GPU 主机；保留源帧编号与时间戳，并限制待传输帧，
不会因远端变慢无限堆积。GPU 主机不需要访问内网摄像头或连接本机端口。
控制台“摄像头拉流位置”（`/gateway`）可选择“GPU 主机直接拉流”；保存后在下次连接生效。
只有选用此项时，才需要 GPU 主机能访问视频源。`HARNESS_VIDEO_PULL_LOCATION=local|gpu`
提供初始默认值，未配置时为 `local`，控制台保存的值优先。
升级已有独立服务器部署时，先停止旧分析，将原有 `camera-NN-data` / `face-data` 卷备份并
迁移到 GPU 主机，之后停止旧本机的多路分析容器。网关不自动复制跨主机 Docker 数据卷。

服务地址填写**主程序容器能访问的服务器地址**，只填写协议、主机和端口，不带 `/v1/...`、用户名、密码或查询参数。另一台机器不能写 `localhost`；如果模型运行在主程序宿主机，Docker 环境下可使用 `http://host.docker.internal:8092`。

`--tracking-capacity` 必须与 SAM 服务器启动时的 `--streams` 一致。省略它时使用 `.env` 的 `TRACKING_SERVER_CAPACITY`，再未设置则默认采用主程序的 `--streams`。即使当前只接入 4 路，若 SAM 已预留 16 路，主程序也应填 `--tracking-capacity 16`，这样会使用正确的多通道 API。

容量为 1 时，客户端将兼容入口 `/v1/video/sessions` 转到 `/v1/perception/sam_live/sessions`；容量大于 1 时，每路使用 `/v1/cameras/camera-01/perception/sam_live/sessions`、`camera-02` 等独立路径。健康检查对应 `/health/perception/sam_live` 或 `/health/cameras/camera-01/perception/sam_live`。控制台只填服务器 origin，路径由客户端自动选择。

控制台可以使用的路数为主程序预留路数与 SAM 配置容量的较小值；超过当前容量的通道不能启动。预留 12 路时不需要把 12 路都接满，增加到 16 或更多路也没有写死的产品上限。已有控制台保存的部署配置优先于环境变量；修改启动参数后，应在控制台同步修改已保存的服务位置和容量。

## 连接 GoodVision Contour 节点

已部署 GoodVision Contour 的 GPU 机器使用这一接入方式。主程序必须先更新到包含 Contour 客户端的版本；仅修改源码不会更新正在运行的 8080 容器。

停止全部分析并完成收尾后，在主程序控制台的部署设置中填写：

| 设置 | 值 |
| --- | --- |
| 人物追踪运行位置 | 独立 GPU 服务器 |
| 节点类型 | GoodVision Contour |
| 服务器地址 | `http://192.168.20.93:8093`（示例，替换为 GPU 机器的内网地址） |
| 服务容量 | 与 GPU 节点的 `CONTOUR_CAPACITY` 一致，例如 `2` |
| 节点访问密钥 | GPU 节点部署时生成的访问密钥 |

保存后，连接视频并开始分析。主程序部署也必须预留足够通道（例如 `--streams 2`）。大模型的运行位置和密钥仍独立配置。多路分别使用带认证的 `/api/cameras/camera-01/status` 和 `/v1/cameras/camera-01/contour/sessions`，其他通道依次编号；单路仍使用原接口。仅网络端口连通不能证明模型已就绪。

节点密钥独立保存于主程序配置目录的 `tracking-node-key` 文件，接口只返回是否已保存，不回显密钥。密钥输入框留空保留已有值；勾选清除后保存会删除它。不要把节点密钥填到云端大模型的 API Key 栏。

Contour 控制台的“登录节点控制台”只认证当前浏览器，不会替主程序配置连接。每个通道只允许一个视频会话；多路共享模型并独立保存追踪状态，GPU 请求排队处理。上文 8092 的原版多路服务与 Contour 的 8093 接口使用不同协议，选择节点类型后由客户端自动选择路径。

## 3. 可选：在另一台 GPU 机器上启动自备大模型

按 [本地 GPU 部署文档](local-gpu.md) 准备指定版本的完整 Qwen 模型目录：

```text
.models/qwen3-vl-30b-a3b-instruct-fp8/
```

模型必须含配置、tokenizer、processor、chat template 和全部 safetensors 分片。模型包通过只读卷提供，不塞入每个主程序镜像；离线交付时应同时交付相应 Docker 镜像与模型目录。

在大模型机器执行：

```sh
python tools/start_modular.py vlm --gpu-id 0 --bind 192.168.10.30 --port 8000
```

该入口只检查大模型权重和所选 NVIDIA 卡，不要求 SAM 文件。默认使用 `vllm/vllm-openai:v0.17.1`，服务模型别名为 `local-vision`。启动后，在主程序控制台将大模型部署位置改为“独立模型服务器”，填写 `http://192.168.10.30:8000` 和 `local-vision`。新部署也可直接执行：

```sh
python tools/start_modular.py analysis --streams 12 --tracking-server http://host.docker.internal:8092 --tracking-capacity 12 --inference-server http://192.168.10.30:8000 --inference-model local-vision --bind 192.168.10.20
```

SAM 与大模型两种选择互相独立；SAM 使用独立服务器、VLM 使用云端的组合完整支持多路。

主程序仅向自备 VLM 发送专用 `LOCAL_INFERENCE_API_KEY` 或控制台保存的自备模型密钥，不复用云端供应商密钥。若需要 VLM API Key，在两台机器各自的 `.env` 配置同一 `LOCAL_INFERENCE_API_KEY`；服务端将其传给 vLLM 的 `VLLM_API_KEY`。该变量是 vLLM 官方支持的认证入口，认证范围参见 [vLLM 0.17.1 文档](https://docs.vllm.ai/en/v0.17.1/usage/security/)。SAM 服务默认没有应用层认证，模型端口应通过客户内网或 VPN 互通。

## 配置和容量

控制台中的三个大模型位置对应 `cloud / local / server`；人物追踪位置对应 `legacy / local / server`。`local` 表示与主程序一起部署的本机 NVIDIA 服务，`server` 表示自备模型服务器。主程序 CPU 部署支持 `cloud / server` 大模型和 `legacy / server` 追踪，不能在没有 GPU 的主机切换成本机模式。

| 环境变量 | 含义 |
| --- | --- |
| `HARNESS_ANALYSIS_URL` | 本机网关连接的 GPU 主机 Harness HTTP(S) origin；不填时从追踪主机推导 8080 端口 |
| `HARNESS_VIDEO_PULL_LOCATION` | 摄像头拉流默认位置：`local`（本机，默认）或 `gpu`（GPU 主机）；控制台可覆盖 |
| `INFERENCE_EXECUTION_MODE` | `cloud` 或 `server`；同机 GPU 入口还支持 `local` |
| `TRACKING_EXECUTION_MODE` | `legacy` 或 `server`；同机 GPU 入口还支持 `local` |
| `INFERENCE_SERVER_URL` | 自备 VLM 服务的 HTTP(S) origin |
| `INFERENCE_SERVER_MODEL` | 服务端模型别名，默认 `local-vision` |
| `TRACKING_SERVER_URL` | 自备 SAM 服务的 HTTP(S) origin |
| `TRACKING_SERVER_CAPACITY` | SAM 服务预留的正整数通道数 |
| `SCENE_OBSERVATION_FPS` | 每路大模型目标分析频率，默认 `1.25` |
| `SCENE_MAX_IN_FLIGHT` | 每路待完成大模型请求上限，默认 `4` |
| `LOCAL_TRACKING_FPS` | SAM 输入采样频率，默认 `2`；控制台保存的“采样帧率（FPS）”优先 |

每路仍以约 1.25 次/秒为大模型分析目标，12 路对应约 15 请求/秒。独立 VLM 默认并发序列 8、上下文 8192、批处理 token 4096、GPU 显存预算 0.90，可通过已有 `LOCAL_INFERENCE_*` 环境变量调节。这些是起始配置，未通过真实 L40S + 客户视频压测，不能据此保证 12 路吞吐。评估同时看模型排队/延迟、SAM 跟踪吞吐、主程序 CPU/视频解码和机器间带宽；使用 [本地 GPU 文档](local-gpu.md) 的压测入口验证实际部署。

同一个 SAM 服务的通道编号供一个主程序集群使用，例如 `camera-01` 对应第一路；不要让两个独立主程序集群共用同一套通道。后续扩容应先停止全部分析并完成收尾，再修改 SAM 预留通道数并重启服务、按新路数重新启动主程序，最后在控制台更新 SAM 容量。已有同名数据卷保留各路设置；减少部署路数前先统一停止分析，再断开准备移除的通道，Docker Compose 的旧容器需要在运维时明确清理。

## 检查计划与日志

对现有 Contour 链路做实时队列 A/B 时，参见
[配置与现场验收](contour-realtime-validation.md)。主程序支持
`--sam-queue-policy durable_ordered|realtime_latest` 和独立的 `--tracking-fps`；
不指定时不改变原队列和采样默认值。实时策略是实验开关，不保证十二路容量。

任意入口加 `--dry-run` 可以查看不含密钥的计划；模拟显卡仅允许用于 worker 的 dry-run：

```sh
python tools/start_modular.py main --streams 16 --tracking-server http://192.168.10.20:8092 --dry-run
python tools/start_modular.py sam --streams 16 --gpu-count 1 --dry-run
python tools/start_modular.py vlm --gpu-count 1 --dry-run
```

三个 Compose 项目分别叫 `goodvision-main`、`goodvision-sam`、`goodvision-vlm`，日志和生命周期互相独立：

```sh
docker logs --tail 100 goodvision-main-harness-1
docker logs --tail 100 goodvision-sam-sam-server-1
docker logs --tail 100 goodvision-vlm-vlm-server-1
```

可用 `--env-file` 指向当前模块自己的配置文件，`--no-build` 使用已导入的应用/SAM 镜像。VLM 始终使用指定镜像。真正离线部署应提前准备这些镜像；启动脚本不会打包或下载模型文件。
