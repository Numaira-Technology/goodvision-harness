# Qwen3.8 27B：单张 4090 部署与检查

**当前对照实验使用 OpenRouter，按[三模式测试指南](vlm-mode-benchmark.md)执行。**
本页是可选的自部署方案，不是 OpenRouter 测试的准备步骤；无需为当前实验执行本页命令。

入口：`bash tools/vlm_benchmark/local_model.sh ACTION`，在仓库根目录运行。
它安装本机模型服务并接入[三模式测试](vlm-mode-benchmark.md)，不创建或租用实例。
`plan`、`package` 在 Mac 上可用；其余命令在 Linux GPU 实例上执行。

## 固定部署方案

- 原始模型：`Qwen/Qwen3.8-27B`。
- 下载权重：`avyukth/Qwen3.8-27B-AWQ-INT4`，锁定
  `5a2ee524cbc9f3ac5df12729ecf0c4b57f311192`，不跟随 `main` 自动更新。
- 这是社区 **W4A16（AWQ + RTN）量化**，视觉编码器保留 BF16。
  作者披露部分层未应用 AWQ activation-aware scaling；不标成官方全精度模型。
- 推理环境固定 vLLM 0.31.0、Transformers 5.14.1；全部解析后的依赖写入
  `tools/vlm_benchmark/model-requirements.txt`。依赖解析目标为 Linux x86_64、
  Python 3.12、glibc 2.39，已在本地完成解析；实际 GPU 安装仍须实例验证。
- 测试环境单独安装到 `bench/`，不与 vLLM 共用 Python 包。
- 默认上下文 8192，引擎同时执行的序列上限 8，最多 4 张图片，关闭 thinking、
  prefix cache 和多模态 processor cache；eager 模式，CPU offload 为 0。
  引擎的 8 个序列名额与测试器的全局 16 个在途请求不同，额外请求会进入引擎队列。
  这是便于启动和诊断的初始配置，不声称最优或已支持十二路。

部署来源：[Qwen 原始模型](https://huggingface.co/Qwen/Qwen3.8-27B)、
[量化作者的模型卡](https://huggingface.co/avyukth/Qwen3.8-27B-AWQ-INT4)、
[vLLM 启动参数](https://docs.vllm.ai/en/latest/cli/serve/)、
[vLLM 0.31.0](https://pypi.org/project/vllm/0.31.0/)。
作者报告的其他 GPU 结果不能当作本项目 4090 实测。

脚本的安装门槛：Ubuntu 24.04 或同等 glibc 2.39 的 Linux x86_64、Python 3.12
（含 venv/pip）、NVIDIA RTX 4090、驱动 R580 以上、至少 32 GiB **可用**主机内存、
60 GiB **剩余**磁盘，以及至少 22500 MiB 空闲显存。
门槛不满足会在下载前失败。容器里的 CUDA 模板不能升级宿主机驱动；CUDA 13 的
驱动要求见 [NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 1. 打包和上传（Mac）

先查看固定方案，不联网、不下载模型：

```bash
bash tools/vlm_benchmark/local_model.sh plan
```

打包当前代码和已有测试视频；包内不会包含 `.env`、SSH key、历史输出或虚拟环境：

```bash
bash tools/vlm_benchmark/local_model.sh package \
  --video '/Users/bihaotian/Downloads/output 3.mp4' \
  --output dist/qwen38-local-model.tar.gz
```

通过实例提供的 SSH 主机/端口，将这个包上传到 `/workspace/`。
上传、租赁均不由本脚本自动执行。每次包使用新文件名，避免覆盖已有包。

## 2. 安装（GPU 实例）

将包解压到新的代码目录，再安装。以下在实例终端运行：

```bash
mkdir /workspace/goodvision-vlm-app
tar -xzf /workspace/qwen38-local-model.tar.gz -C /workspace/goodvision-vlm-app
cd /workspace/goodvision-vlm-app
bash tools/vlm_benchmark/local_model.sh install
```

默认部署目录 `/workspace/goodvision-vlm-qwen38`，与代码目录分开。
安装会检查环境、建立两套隔离环境、执行一次 CUDA 张量运算、下载指定权重快照、
核对视觉配置与四位量化格式，并注册一个专用 Supervisor 服务。此时尚未启动模型。
公开权重通常不需要 token；需要鉴权时只在实例本地设置 `HF_TOKEN`。
不需要 OpenRouter key。

同配置安装完成后可再次执行 install，直接复用。中途下载失败可以重试，保留下载缓存。
模型版本或配置变更需要新的 `--root`；所有后续命令也使用同一个 `--root`。
选择其他 GPU 编号或端口，可在首次 install 时加 `--gpu 1 --port 8001`。

## 3. 启动与健康检查

```bash
bash tools/vlm_benchmark/local_model.sh start
bash tools/vlm_benchmark/local_model.sh status
```

start 启动后台模型进程，默认最多等待 600 秒，再做真实图片检查：

1. 确认 `/health` 可用，`/v1/models` 返回预期模型名。
2. 两次四图检查，验证模型分别读出最后一张图的颜色，避免只测文本服务。
3. 用实际 events 提示和严格 schema，验证六类 boolean/null 输出。
4. 用实际 grounding 提示强制完整结构，验证六类判断及人物引用结构。

总计四次有界推理请求，不自动重试；检查使用合成图片，**不验证行为准确率**。
灰色图片没有人物，允许空人物列表，不能据此证明实际人框质量。
每次请求的 HTTP 超时设为 60 秒，整个探测进程上限 300 秒；这是启动检查期限，不修改正式测试的
events/compact HTTP 期限。`/metrics` 是否可用单独记录，缺失不伪造为零。

成功才写 `health.json: ready`。start 期间加载超时或图片/schema 检查失败，会停止本脚本管理的
模型进程并保留 `model.log`、`health.json`。不会反复重启 OOM 的模型，也不会自动
改成 CPU offload、小模型或外部 API。

独立重查：

```bash
bash tools/vlm_benchmark/local_model.sh health
```

默认服务只监听 `127.0.0.1:8000`，不公开到互联网。实例内测试无需 SSH 转发。
Supervisor 使用独立 UNIX socket，不修改实例已有的 Supervisor 服务。
SSH 断开后模型服务继续存在；租用实例继续计费。

## 4. 接入测试

先做单路短测；通过后再跑正式阶梯测试：

```bash
bash tools/vlm_benchmark/local_model.sh benchmark \
  --video media/video-01.mp4 --streams 1 --duration 6 --repeats 1 \
  --output outputs/local-qwen38-smoke

bash tools/vlm_benchmark/local_model.sh benchmark \
  --video media/video-01.mp4 --streams 1 4 8 12 --duration 60 --repeats 2 \
  --output outputs/local-qwen38-capacity
```

会先重新检查模型，然后运行 events / compact / parallel；模式含义及计时规则见
[三模式测试指南](vlm-mode-benchmark.md)。部署元数据会一同拷入报告目录，包括权重版本、
量化方式、引擎、显卡和启动参数。健康检查通过仅说明端点可用，短测是否足够及时、
十二路是否有积压由报告判断。当前量化结果要单独标记，不能直接归因于本地/云端差异。

测试命令是前台进程；长测请在实例的 tmux 会话内运行，避免 SSH 断开中断测试。
测试完成或失败均保留后台模型，便于看日志和重测。停止模型释放显存：

```bash
bash tools/vlm_benchmark/local_model.sh stop
```

stop 仅停模型，不关闭或销毁租用实例。测试结果位于代码目录的 `outputs/`；请取回后
再按需关闭实例。当前脚本经过本地控制流程、HTTP 模拟服务和打包回归验证，尚无这套
配置的 4090 真机启动或吞吐保证。
