# Vast 自部署 Qwen3.5 35B 基线

> 2026-10-20：本文实验工具使用独立的 `tools/benchmark_runtime/`；产品 grounding 已恢复 main。历史结果不等于恢复后的产品性能，见[代码与测试边界](pr48-49-person-grounding-integration.md)。

状态：部署和测试脚本已编写；尚未租 GPU、下载权重或进行真实 GPU 验证。
离线测试不能证明 GPU 能启动模型，也不能证明十二路达到三秒 95%。
本入口随最新 main 接入保留；48GB 4090 请使用独立的 [FP8 + YOLO 指南](vast-fp8-yolo.md)，本 BF16 入口不会自动换精度。

## 固定条件

- 官方 Qwen/Qwen3.5-35B-A3B，revision
  59d61f3ce65a6d9863b86d2e96597125219dc754。
- BF16、单 GPU、不做 CPU offload。启动检查要求至少 110 GiB 显存；
  H200 141GB 是候选，双卡/FP8 要建独立实验配置，不能悄悄替换。
- vLLM 0.31.0，Docker 镜像 digest 固定于 tools/vast_vlm/profile.py。
  模板必须选择该镜像，或用源代码包中的 Dockerfile 构建派生镜像。
  运行时验证 vLLM 版本，但容器内不能证明宿主实际使用的镜像 digest；
  租机记录中另存模板、机器及镜像信息。
- 保留原四帧序列，不使用 Cerebras 拼图；原 output.mp4 的 SHA256 被强校验。
- 六类标准的 SHA256 固定；吸烟 high，其余 standard，与历史标准相同。
- 十二路 × 每秒 1.25 个窗口；每轮 55.798021 秒、每路 69 个窗口。
- events / grounding（内部 compact）/ parallel，各两轮，六个录屏。
- 48 总并发、每路 4、配对调度、tmpfs admission、排队上限 1 秒；
  两分支 HTTP 超时均 20 秒、零重试。3 秒是独立评分期限。
- events 输出上限 160 tokens，grounding 1600；关闭 thinking。
- 15 FPS 单视频布局，没有右侧证据画面和框。
- 同视频十二份会产生重复输入。此基线关闭 prefix 与多模态处理缓存；
  模型预热 10 秒，预热不计入成绩。记录首次准备/再次启动的区别。

## 存储与环境

Vast 当前的卷只能挂到同一台物理机器上的实例，不能跨机器重新挂载。
换机器应从原卷/对象存储复制，或重新下载固定 revision。
GPU 释放后不要删除卷；仍需预算保留存储的费用。
建议预留 250–300GB，根据结果保留量确认。卷不是备份。

在 Vast 模板中将**真正的持久卷**挂到 /models。脚本检查 Linux mountinfo，
拒绝根目录、普通目录及 tmpfs；这只证明挂载，Vast 卷身份需由模板记录确认。
目录用途：

- /models/models/<revision>：模型、处理器、文件校验 manifest。
- /models/cache/<vllm>/<GPU>：兼容版本的启动缓存。
- /models/runtime/<timestamp>：启动参数和服务日志。
- /models/media：output.mp4；/models/reference：原 Astra packet。
- /models/results/<run>：测试报告、原始请求记录、浏览器录屏。

模型下载和启动分别加进程锁，未完整下载的目录不会被当成可用模型。
复用前重新校验权重；这会读磁盘，有成本，但不重新下载。
last-preparation.json 记录下载/复用校验耗时；每次 startup-timing.json
记录模型进程启动到健康检查通过的耗时，和包含权重校验的总准备耗时。
健康检查最长等待 15 分钟，失败会停止本次服务，不自动重启。
部署包不包含 .env、API Key、模型权重、视频或 Astra 标注。
标注独立保存，绝不发送给待测模型。

## 本地准备（不租机器）

从这个 checkout 根目录运行，先使用已有 Harness Python 环境：

```bash
PYTHONPATH=src python -m tools.vast_vlm plan
PYTHONPATH=src python -m tools.vast_vlm bundle --execute --output /tmp/vast-qwen35-source.tar.gz
```

所有子命令缺少 --execute 都只打印计划；不会自动租机、SSH、下载模型或调用接口。
bundle 只打包已有源码，输出 SHA256。独立的 output.mp4 和 Astra packet
需要在首次准备时上传；以后保留在卷中。

## 获得授权租机后的流程

使用支持 SSH 的 Vast Docker 模板（非 VM），基于固定镜像；不要在容器内再运行 Docker。
预留至少 128GB 主机内存、足够 CPU 和共享内存，确认无其他 GPU 工作负载。
这里的机器容量是起步配置，不是十二路性能保证。

上传源码包并解压到 /workspace/harness；进入该目录。
服务端使用镜像内 Python，测试客户端使用单独的 Python 3.12 环境：

```bash
bash tools/vast_vlm/setup-client.sh --execute
python -m tools.vast_vlm prepare --volume /models --execute
python -m tools.vast_vlm start --volume /models --execute
```

start 在前台运行，将 vLLM 日志写入 /models/runtime。另开 SSH 终端测试。
准备可用的 python3.12/venv、ffmpeg 和字体安装权限；setup-client 不安装 GPU 驱动、
不更改服务端 Torch/vLLM。可以在后续构建镜像时预装客户端以缩短冷启动。
Ctrl+C 只停止本次模型进程，不删除权重或卷；不要使用 killall。

服务只监听 127.0.0.1:8000，没有公开推理端口，也没有 OpenRouter 回退。
推理客户端忽略代理环境变量，不转发继承的认证头；无需提供 OpenRouter/OpenAI Key。
预检/正式测试要求由 start 启动的同卷服务仍在运行，防止把旧服务当成新模型。

```bash
.venv-vast-client/bin/python -m tools.vast_vlm preflight --execute --video /models/media/output.mp4
.venv-vast-client/bin/python -m tools.vast_vlm run --execute --video /models/media/output.mp4 --packet /models/reference/astra-packet --output /models/results/baseline-01
```

run 自带两次四帧预检，无需重复手工 preflight。随后原 runner 仍执行自己的
预检及预热。正式计划 6624 次分支调用；含全部预检/预热最多 6770 次，不自动重试。
已有结果目录拒绝重跑。服务不存在、模型名不符、视频/标准/参考不匹配时停止。
健康检查仅确认接口；四帧预检使用真实六类解析器，不能代表十二路容量。
模型 8K 上下文不足或 GPU 内存不够时应停止并更新独立实验配置，不能减帧凑数。

## 报告、导出与验收

- 原 report.html：每轮发出、完整、延迟、录屏及运行状态。
- self-hosted-comparison.json：每路三秒内完成率、parallel 同窗口双分支完成、
  Astra 一致率、输入/输出 tokens、TPS、服务端计时。
- 每阶段 telemetry.jsonl：GPU、CPU、vLLM 排队/prefill/decode/首 token 等。
  vLLM 指标是聚合计时，不是逐请求纯 GPU 时间；不存在的指标保持未知。
- self-hosted-plan.json：模型/运行镜像/六类标准/负载版本。

发出率和整体完整率均以 828 为分母；每路三秒 95% 要求至少 66/69。
parallel 要同一窗口两分支都完整且都不超过三秒。
失败、未发、丢记录、晚到不能从分母剔除；旧结果仍用于统计但不覆盖新状态。
运行完整性失败或录屏不完整不能宣称通过。
Astra 只比较完整、证据序列准确对应且双方明确的判断；同时报告可比数量、
不确定和无法对齐数量。它不是人工真实准确率，也不是定位框准确率。

```bash
.venv-vast-client/bin/python -m tools.vast_vlm analyze --execute --video /models/media/output.mp4 --packet /models/reference/astra-packet --results /models/results/baseline-01
.venv-vast-client/bin/python -m tools.vast_vlm collect --execute --results /models/results/baseline-01 --output /models/exports/baseline-01.tar.gz
```

collect 输出归档及文件/归档 SHA256 清单。用 scp 下载后校验，再结束实例；
脚本不自动销毁任何实例或卷。启动日志单独保存/收集，不混入可分享结果包。
历史 Alibaba 数据的路由、精度、超时和调度不同，报告只能作为历史参考，
不能把差异全部归因于 GPU 或 self-hosting。

## 离线回归与后续 GPU 验收

```bash
PYTHONPATH=src python -m pytest tests/test_vast_vlm.py
```

上述离线回归在完整源代码 checkout 执行；部署包不包含 tests。
覆盖默认不执行、负载漂移、模型校验失败、挂载验证、原四帧传输、
两分支 JSON/SSE、失败不重试、迟到数据、Astra 分母、打包及导出。
真实 GPU 验收另需检查：固定镜像参数可用、BF16 显存、四帧预检、六轮录屏、
服务计时存在、停机后同卷复用，以及不同机器的恢复流程。

参考：
- https://github.com/vast-ai/docs/blob/main/guides/instances/storage/volumes.mdx
- https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- https://docs.vllm.ai/en/latest/design/metrics/
