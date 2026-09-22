# 追踪性能与队列实验：可用结论

[验证索引](README.md) · [Contour 性能与复跑方法](../../workers/contour/PERFORMANCE.md) · [实时队列配置](../guides/contour-realtime-validation.md)

## L40S 正式权重基线

2026-09-18，基于 `bf614ae` 的 Contour 修复，固定上游 SAM 源码 `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`。模型 revision 为 `daa63191845a41281374e725f4c9e51c7a824460`，权重 SHA-256 为 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。

共享 L40S 上使用 OpenCV `vtest.avi` 的 96 帧顺序样本，3 FPS 采样、512×384 输入、每路独立状态。下表是原生饱和吞吐和本进程 CUDA allocator 峰值活跃张量；不含完整 Harness、浏览器、Face/VLM，也不是整卡显存占用。

| 路数 | 基线总 FPS / 峰值 GiB | 最终补丁总 FPS / 峰值 GiB |
| --- | --- | --- |
| 1 | 7.19 / 5.62 | 7.21 / 4.26 |
| 2 | 7.35 / 6.20 | 7.33 / 4.87 |
| 6 | 7.35 / 8.64 | 7.35 / 7.32 |

固定 person 文本缓存减少显存，未显著提升吞吐。最终补丁与基线共 864 个输出帧的 ID、mask hash、概率一致，支持本片数值等价；没有人工真值，不能解读为准确率达标。

真实 Contour HTTP 回放各持续 48 秒，每路一个在途请求和四帧等待队列：

| 输入 | 结果 |
| --- | --- |
| 六路各 1 FPS | 288/288 帧完成，零丢帧；响应 P50/P95 为 0.586/1.000 秒，无 EOF 额外追赶 |
| 六路各 3 FPS | 864 帧中完成 344、丢弃 520（60.2%）；采集期每路约 1.08～1.10 FPS；已完成响应 P95 4.776 秒，EOF 后追赶 4.290 秒 |

**该版本没有支撑六路各 3 FPS 的吞吐，也没有证明双卡 12 路长期容量。** 2026-09-20 后的 CPU 预解码改动不包含在这组性能数字内。正式验收应使用当前版本重跑，不能只改配置容量后沿用此基线。

保留的证据：

- [真实模型汇总](evidence/contour-l40s-20260918/real-summary.json)、[模型清单](evidence/contour-l40s-20260918/model-manifest.json)、[输入清单](evidence/contour-l40s-20260918/input-manifest.json)、[最终源码指纹](evidence/contour-l40s-20260918/final-source-sha256.txt)、[依赖版本](evidence/contour-l40s-20260918/pip-freeze.txt)。
- [HTTP 回放器](evidence/contour-l40s-20260918/paced_http_probe.py)与[汇总脚本](evidence/contour-l40s-20260918/summarize_results.py)。正式原生复跑使用 worker 自带的 [benchmark_gpu.py](../../workers/contour/benchmark_gpu.py)。
- 原始逐次结果位于 `tmp/l40s-20260918/results/` 和远端 `/home/edwin/projects/goodvision-contour-20260918/results/`；仓库汇总不能替代全部原始输出。

## 实时队列仍是可选实验

当前默认仍为 `durable_ordered`。`realtime_latest` 可限制未发送候选的数量与年龄，但不能撤销已执行的请求，也不提供硬延迟上限。

2026-09-20 基于 `bf614ae` 的双路模拟：每路 4 FPS，共 56 输入，模型替身在过载阶段每帧耗时 0.60 秒。每配置只跑一次，受线程调度影响。

| 策略 | 最坏源到结果 | 4 秒内完成 / 全部输入 |
| --- | ---: | ---: |
| FIFO | 5.02 秒 | 45/56（80.4%） |
| 4 候选、2 秒、单帧追加 | 2.24 秒 | 36/56（64.3%） |
| 8 候选、3 秒、单帧追加 | 3.19 秒 | 44/56（78.6%） |

更短等待可能牺牲及时覆盖率。该模拟不测真实 SAM 质量和浏览器轮廓覆盖，不能作为默认启用理由。复跑入口是 [benchmark_sam_realtime_queue.py](../../tests/benchmark_sam_realtime_queue.py)，临时输出写入 `tmp/`；回归见 [实时队列测试](../../tests/test_sam_realtime_queue.py)。正式启用前需同片对照 1/2/4 追加上限、多人遮挡与进出画面，再做双路长测。

## 仍有参考价值的早期质量边界

- 2026-09-12 的 RTX 4090 原生连续会话中，48 帧整体/分段/增量路径，以及 128、256 帧历史清理对照，ID 与 mask hash 分别全等。48 帧真实 HTTP 接入保留同一 epoch 和全部源图；15/32 帧短片能在 EOF 补齐。它们验证增量与回收一致性，不证明当前端到端吞吐、分割准确率或数小时稳定性。复跑工具为 [sam31_stream_probe.py](../../tools/sam31_stream_probe.py)。
- 2026-09-13 的 384 长边实验出现遮挡人物框退化与边缘漏人；H.264 512 与 JPEG512 的对照也出现遮挡/边缘差异。不能将缩小传输图视为无损提速，实际质量需同片检查。历史材料在 `tmp/sam31-opt-20260913/quality-review/`。
- 单帧追加可复用原生历史，不等于单帧一定最快；首批帧数还可能改变上游检测模式。改变该模式的早期实验已从最终 L40S 补丁撤回。

早期随机权重组件探针、无效初始化夹具及逐次测试输出已移除；它们不再承担当前配置的验收依据。
