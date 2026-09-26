# 固定 main 配置的 4090 测试包

已接入 12 个分割模型的完整 Harness 显示评测。使用原视频、固定 main 的采样和显示配置，
逐模型测量描边率与延迟。默认两轮都达到 70% 才标记通过。
**不需要 HF Token、SAM3.1 权重或服务。** VLM 使用 main 的 OpenRouter preset。

完整流程见包根目录的 `docs/guides/segmentation-pipeline-benchmark.md`。

## 云端准备

单卡 RTX 4090、Linux x86_64、Python 3.12、32GB RAM，本次使用 50GB 磁盘。
逐模型运行并留意剩余空间；不要同时安装本次不使用的 SAM 环境。
主机驱动需要兼容安装脚本使用的 CUDA 12.6 PyTorch wheel，`nvidia-smi` 应能识别 GPU。
在 Ubuntu 24.04 root 容器中，缺少基础依赖时可执行：

```bash
apt-get update
apt-get install -y python3.12 python3.12-venv git build-essential \
  ca-certificates libgomp1 libglib2.0-0 libgl1 ffmpeg
```

把下面占位符替换成实际准备包、Vast SSH 端口和地址。在本机上传：

```bash
scp -P YOUR_SSH_PORT /path/to/goodvision-4090-main-fixed.tar.gz root@YOUR_HOST:/workspace/
```

在实例中解压、安装候选环境并启动第一个模型：

```bash
cd /workspace
tar -xzf goodvision-4090-main-fixed.tar.gz
cd goodvision-4090-benchmark
bash tools/vast_benchmark/setup.sh seg
.venv-seg-benchmark/bin/python -m tools.seg_pipeline.worker --model yolov8-n
```

安装脚本会注册 12 个默认停止的 `goodvision-seg` supervisor 服务，但不会启动模型、
租赁实例或启动 SAM。上面的最后一条命令用于手工运行单个 worker；首次模型启动会下载
公开权重，一次只跑一个模型。若使用下文的 `run_remote.sh`，不要手工启动 worker。

## 本机运行

本机同一份源码/解压包根目录：

```bash
bash tools/seg_pipeline/setup_client.sh
```

另开终端建立转发并保持运行：

```bash
ssh -N -L 8092:127.0.0.1:8092 -p YOUR_SSH_PORT root@YOUR_HOST
```

在测试终端设置 `OPENROUTER_API_KEY`（VLM 使用 main 的
`qwen/qwen3.5-35b-a3b:nitro` preset），然后执行：

```bash
.venv-harness-benchmark/bin/python -m tools.seg_pipeline \
  --model yolov8-n --repeats 2 --allow-face-download
```

结果路径会打印在终端。切换模型时同时修改云端和本机的 `--model`。
完整测试包含 VLM API 调用；CPU 人脸分支首次运行需要 InsightFace 权重，
`--allow-face-download` 用于允许首次下载。

需要自动切换 supervisor worker 并扫描所有模型时，在本机执行：

```bash
bash tools/vast_benchmark/run_remote.sh all \
  --host YOUR_HOST --port YOUR_SSH_PORT --identity /path/to/private-key \
  --delays 0.5,1,1.5,2
```

旧离线工具和样本保留，仅用于辅助选型，不替代本次完整链路验收。
历史 SAM 结果随包保留，不会重新运行；默认测试固定为 main 的 0 秒缓冲，
只有显式提供 `--delays` 时才扫描指定档位。

`BUNDLE_MANIFEST.json` 记录来源 commit 和每个文件的校验值，包括未提交改动。
包内不含密钥、`.env`、个人状态、虚拟环境或模型权重。
RTX 4090 实机安装和完整链路已经跑通；具体模型、轮次与覆盖率以保存的报告为准。
