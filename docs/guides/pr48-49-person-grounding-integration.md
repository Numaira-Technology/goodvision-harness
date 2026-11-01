# 恢复 main grounding，保留独立实验工具

2026-10-20 核对的远端 main 为 `9dbd5103a5cc1c4a7886d7c80492f70d8abe1e63`。
主程序 `src/goodvision_harness/`、工作流画布和两份默认配置恢复到该基线，
不再包含本地添加的“YOLO ID 替代 VLM grounding bbox”接入和工作流选项。
main 自带的检测预览、事件关联、grounding/parallel 行为保持原样。

## 保留的实验与部署

| 部分 | 处理 |
| --- | --- |
| Decisions API | 保留请求适配、预算/限流、测试配置与结果解析 |
| Qwen3.5-35B 自部署 | 保留 Vast 部署、vLLM/FP8 配置及 auto 16/24/32 并发工具 |
| VideoJev 自部署 | 保留 Transformers 服务、小样本/PPE 对照及十二路测试工具 |
| 历史人物 ID 测试协议及调度 | 移到 `tools/benchmark_runtime/`，仅供独立实验使用 |
| 已有结果、录屏、模型与持久卷 | 不修改 |

实验工具引用 `tools.benchmark_runtime`；产品代码不引用它。
这意味着使用主程序运行 grounding 或 parallel 时，沿用 main 的 VLM 定位与 bbox
输出协议。测试工具保留历史协议，是为了保留 Decisions/自部署的既有实验与报告解析，
**不能把这些历史测试或旧测试入口的结果称作“恢复 main 后的产品测试结果”。**

原本地扩展的 `SCENE_GROUNDING_SOURCE`、`SCENE_DETECTOR_URL` 以及工作流
“人物定位”选择已从产品接入中撤回。它们不再用于选择产品 grounding 路径。
独立实验的 `person_display` 仅控制录屏是否显示 ID/框，不代表产品开关。

## 数据与复测边界

- 历史测试配置、六类标准、PPE 提示词、输入与报告不改写。
- 原 FP8 入口保留 16 个服务端序列 / 48 个客户端 admission 槽位；
  sweep 入口保留 auto + 16/24/32 配套并发，二者不是同一配置。
- 新源码包和新计划标记 `main-grounding-isolated-benchmarks-v3`，
  记录基线、commit、未提交状态及独立实验模块的逐文件 SHA256。
- API latency 仍指请求发出至完整响应；不把本地调度、检测或显示耗时混入其中。
- 本次只做代码恢复、打包/导入和离线回归，不运行真实推理、不租赁或重建实例。
- 最新 main 产品路径尚未重新进行十二路正式测试；旧数据保留原版本归属。

需要验证产品恢复后的性能时，应另建以 main 产品路径为入口的测试，明确输入、负载、
计时和版本，不能直接把独立人物 ID 实验入口当作最新产品链路。
