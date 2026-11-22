# VideoJev FP8 实验开关

VideoJev-4B 在 harness 里由 Transformers 直接加载（BF16 + FlashAttention-2），不走 vLLM，所以 Qwen8B 那种 `--quantization fp8_per_tensor` 参数用不了。这里在 Transformers 这一侧，用 torchao 的动态 FP8 自己加了一个开关。

**默认关闭。** 生产服务、跑分和到目前为止的所有结果都是 BF16。只有显式传 `--quantization fp8` 才会启用。

## 量化范围
按固定版本 `1c1a253b` 的权重索引，只量化语言模型里的这两类线性层：

| 模块 | 数量 | FP8 |
|---|---|---|
| 31 层 MLP 的 `gate_proj` / `up_proj` / `down_proj`（最后一层没有 MLP） | 93 | 是 |
| 8 个全注意力层的 `q_proj` / `k_proj` / `v_proj` / `o_proj` | 32 | 是 |
| 24 个线性注意力层（Gated DeltaNet，`linear_attn.*`） | — | 否，先不碰 |
| 视觉编码器（`prepare_for_inference` 让它保持 FP32） | — | 否 |
| 打分头 `head.q` / `head.k`（FP32） | — | 否 |
| 词嵌入 | — | 否 |

- **方案：** torchao 0.13.0（对应 torch 2.8.0）的 `Float8DynamicActivationFloat8WeightConfig`，按张量缩放（per-tensor），和 Qwen8B 的 `fp8_per_tensor` 一致。
- **GPU 要求：** sm_89 及以上，也就是 4090、L40S、H100。A100 是 sm_80，没有 FP8 张量核心，服务在加载权重之前就会报错退出。
- **记录：** 服务的 `runtime` 元数据里新增 `quantization`（模式和量化层数）、`gpu_allocated_bytes_after_load`；FP8 模式下 `versions` 里也会记录 torchao 版本。

## 启动
```bash
python -m goodvision_harness.videojev_service --cache-dir <weights> --result-log <log.jsonl> --quantization fp8
```
`--quantization fp8` 只支持 `--runtime videojev`，用在 `decision3` 上会直接报错。

## 环境
`tools/videojev_probe/requirements-fp8.txt` 就是 BF16 的依赖再加上 `torchao==0.13.0`。

- **只做测速时：** 在租来的机器上照常恢复现有的 VideoJev 环境包，再执行 `pip install torchao==0.13.0`。这个改动只留在那台临时机器上，不上传，也不改 R2 里的任何对象。
- **要发布成正式的新环境包时：** 等 #94 的构建配方修复（conda/pip 包归属冲突）合入后，再用 `tools/build_env.py --requirements tools/videojev_probe/requirements-fp8.txt` 构建，在 registry 里新开一个条目，用新的 `r2_key`。不覆盖 `envs/videojev/cu128-py312-torch280-ada.tar.gz`。

## GPU / 精度对照（必须有 baseline）
冻结的 native 跑分（`tools/unified_benchmark/native_run.py`）跑的是从历史提交导出的旧服务，并且写死只能用一张 RTX 4090。它是冻结协议，**不改**。所以上面的 FP8 开关和换卡的影响，都测不进那套跑分里。

`tools/unified_benchmark/gpu_sweep.py` 是给**当前** `videojev_service` 用的对照工具：
- **`run`：** 把冻结 bundle 里原样的请求体发给一个已经启动的服务，并逐条核对请求指纹。并发设置和冻结协议相同：12 个视频 worker，同一源最多 1 个在途请求。每个单元只发一次，不重试；失败的单元仍算在分母里。
  - **准确率：** 用冻结的计分器算。
  - **延迟：** 按 nearest-rank 计 P50/P95，并按服务返回的耗时分段统计：CPU 排队、预处理、GPU 排队、视觉编码、决策、PPE。
- **`compare`：** 把候选配置和 **baseline** 放在一起比，结果全部以"候选 / baseline"的形式给出：
  - 准确率变化，以及预测结果变了的题号；
  - 逐单元的答案翻转数，以及 baseline 里本来就很接近的题有多少（前两名概率差小于 0.1）；
  - 概率漂移；
  - 各段延迟的比值（小于 1 表示更快）；
  - 吞吐量和加载后的显存占用。

**baseline 规定为 RTX 4090 + BF16。** 没有 baseline 就不出比较结果：`compare` 拒绝其他配置当 baseline，除非显式加 `--allow-other-baseline`。bundle、划分（split）、冻结配置和服务设置也必须一致，包括 `gpu_batch_size`、`max_pixels`、上下文长度、prefix cache 和 CPU 线程数。

每台机器的服务参数与冻结的 native 设置保持一致，只改 `--quantization`：
```bash
python -m goodvision_harness.videojev_service --cache-dir <weights> --result-log <log.jsonl> \
  --ready-capacity 4 --gpu-batch-size 4 --max-pixels 409600 --prefix-cache \
  --cpu-threads 4 --request-timeout 20 --queue-age 19.5 --queue-capacity 48 --quantization bf16
```
```bash
python -m tools.unified_benchmark.gpu_sweep run --bundle <native-bundle> --label 4090-bf16 --output <runs>/4090-bf16
```
```bash
python -m tools.unified_benchmark.gpu_sweep compare --baseline <runs>/4090-bf16 --candidate <runs>/l40s-fp8 --output <runs>/l40s-fp8-vs-4090-bf16.json
```

- **第一轮：** 4090 BF16（baseline）、4090 FP8、L40S BF16、L40S FP8。L40S 也是 Ada 架构，可以直接用现有环境包。
- **H100 和 A100：** 各自需要另打环境包（flash-attn 要按 sm_90 / sm_80 重新编译）。A100 只能测 BF16。
- **数据保密：** 输出里含参考答案和逐题对错，和 bundle 一样只能留在授权的评测机上。

## 已在 CPU 上验证
- 用一个按真实层结构搭的小模型测试，筛选结果正好是 125 层；线性注意力、视觉编码器和打分头都没被选中。见 `tests/test_videojev_fp8.py`。
- 在临时环境里用真实的 torchao 0.13.0 跑了一遍（绕过它的 GPU 检查），被转换的层和筛选结果完全一致，一共 125 层。转换后的权重对外仍报告 BF16，所以模型里的 BF16 autocast 不会把它们转回去。
- 默认值在核心类、运行时和服务命令行三处都是 `bf16`。
- `gpu_sweep`：用合成 bundle 和假服务测试了计分、失败计入分母、请求指纹核对、和 baseline 的比较（答案翻转、延迟比值），以及拒绝不合格 baseline 和不同的服务设置。见 `tests/test_gpu_sweep.py`。

## 还没验证（要上 GPU）
- FP8 能不能在 4090 上正常加载和推理，包括和 flash-linear-attention、autocast 能否一起工作。
- 速度。没有 `torch.compile` 时，动态 FP8 每次调用都要额外量化一次激活值，速度可能不升反降。要在同一台机器上先跑 BF16 再跑 FP8，对比结果说了算。
- 概率漂移。三选一的结果要逐题比较 FP8 和 BF16 的概率。物体节点有 0.95 的门限，FP8 对它的影响也要单独比较。
