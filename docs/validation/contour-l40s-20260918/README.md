# L40S 原生推理、HTTP 回放与组件验证记录

2026-09-18，105 的 GPU 0。结论与限制见
[PERFORMANCE.md](../../../workers/contour/PERFORMANCE.md)。

已在 `/nas2/edwin/hf_cache/models--facebook--sam3.1/` 找到正式权重并核对完整 SHA-256。
`real-summary.json` 汇总原分支与最终补丁的真实推理、逐帧一致性和 HTTP 回放；
`model-manifest.json` 记录精确缓存位置和权重版本。原始完整输出保存在本仓库
`tmp/l40s-20260918/results/` 和远端同名测试目录。

以下组件记录来自取得权重前的探索，使用未训练参数；它们不替代上述正式模型结果：

- `encoder_probe.py`：三轮视觉组件计时；每组预热 15 次、计时 50 次。
- `encoder-random-architecture.json`：600 次原始计时及显存；三轮顺序有交替。
- `encoder-finite.json`：另行验证 batch 1／2／4 输出有限，计时结果不含此检查。
- `text_cache_probe.py`：显式初始化文本参数，验证实际 CUDA 缓存的数值、存储隔离及释放。
- `text-cache-initialized-architecture.json`：有效夹具的成功结果。
- `text-cache-random-architecture.json`：初版遗漏上游两个未初始化参数的失败记录。
  该夹具无效，不能解释成训练后模型的数值失败。成功版本加入有限值检查与未缓存重复对照。
- `gpu-samples.csv`／`gpu-process-samples.txt`：180 秒设备与进程遥测，非独占 GPU。
- `final-gpu-state.csv`：测试后 GPU 回到原有显存占用，测试容器均已退出。
- `pip-freeze.txt`／`source-sha256.txt`：构建环境依赖与远端 worker 源码指纹。

最终正式权重验证另存 `final-source-sha256.txt`、`real-final-gpu-state.csv`、
`input-manifest.json` 和 `real-summary.json`。`paced_http_probe.py` 是实际回放器，
`summarize_results.py --results /path/to/results` 可重新生成汇总；归档脚本只作格式整理。

远端工作目录：`/home/edwin/projects/goodvision-contour-20260918`。
镜像 `goodvision-contour-bench:20260918`，ID
`sha256:1f3b9d9db1a7c95dac7b4d393d5df0ddf6355c12dabce5cc8b9612bfecfda8c8`。
未停止其他人的任务，未修改驱动，也未向镜像复制凭据。

在已准备的 105 上重跑组件测试（输出名必须不存在）：

```sh
cd /home/edwin/projects/goodvision-contour-20260918
docker run --rm --gpus device=GPU-46b4ef0c-d6d5-9252-682e-5527891b51f1 \
  --network none --cpus 8 --memory 20g --shm-size 2g \
  --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e PYTHONPATH=/engine \
  -v "$PWD:/probe" goodvision-contour-bench:20260918 \
  python /probe/encoder_probe.py --output /probe/results/encoder-repeat.json
```

同一个容器启动方式，将 Python 命令换为
`python /probe/text_cache_probe.py /probe/results/text-cache-repeat.json`
可重跑文本缓存检查。先确认目标卡有空闲资源；不要与另一轮 GPU 计时并行执行。

下面重跑真实视频路径；它与上面的组件探针不同。
使用最终镜像，已验证原分支和最终缓存版的 1／2／6 路，每路 96 帧。
最终镜像 ID 为 `sha256:c2e415521baa9cec5a6b7708f8b4ba8ec64f5e4e3590497dd9d953902fca4c2a`。

```sh
docker run --rm --gpus device=GPU-46b4ef0c-d6d5-9252-682e-5527891b51f1 \
  --network none --cpus 8 --memory 24g --shm-size 2g --tmpfs /tmp:rw,size=2g,mode=1777 \
  --user "$(id -u):$(id -g)" -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v "$PWD/models:/models:ro" -v "$PWD/frames:/frames:ro" \
  -v "$PWD/results:/reports" goodvision-contour-bench:20260918-final \
  python /engine/benchmark_gpu.py --worker-dir /engine \
  --checkpoint /models/contour.pt --frame-dir /frames --frames 96 \
  --lanes 1 2 6 --batch-frames 1 --output /reports/person-cache-off-r1.json
```

以新输出名增加 `--fixed-person-text` 重跑，再比较 mask／ID 与显存、吞吐。
原分支基线使用 `/baseline` 只读挂载目录及相同 probe；最终版 `sam_live.py` 与原分支
相同，只有显存优化、准入修复和诊断变化。早期 `native-*-b1-r1.json` 使用实验视频标志，
其行为变化已撤回；验收应比较 `baseline-b1-r1.json` 与 `final-on-b1-r1.json`。
HTTP 回放通过单独 Contour 网关和 engine，使用隔离的内部 Docker 网络，不暴露宿主端口。
每路一个请求加四帧等待队列，队列满时明确丢弃新采样帧；编码纳入计时。
这不是摄像头 RTSP、浏览器、事件模型或完整 Harness 客户端的端到端测试。
