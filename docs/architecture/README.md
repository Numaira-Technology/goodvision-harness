# 架构与设计

[返回文档导航](../README.md) · [验证与排障记录](../validation/README.md)

这里收录系统职责、数据语义与设计方案。具体运行的测量和故障记录见验证目录；设计目标不等于已通过验收。

## 系统机制

| 文档 | 内容 |
| --- | --- |
| [持续人物与事件引擎](entity-event-engine.md) | 状态模型、持续分析与事件生命周期 |
| [事件运行时代码组织](event-runtime-organization.md) | 运行时、事件、告警与离线重放的职责和兼容导入 |
| [事件与视频证据](event-evidence.md) | 独立结果输出、身份连续性与证据修订 |
| [人物身份与陌生人证据](person-identity-evidence.md) | 身份关联、姓名与陌生人判断 |
| [网络视频连续显示](network-display.md) | 源画面、轮廓传播与显示统计 |
| [VLM 请求时序](vlm-request-timing.md) | 时序请求与处理边界 |
| [VLM 观察与跟踪关联](vlm-observation-association.md) | 单次观察的人物引用与跟踪模块负责的事件确认 |

## 带版本的设计与方案

以下文档保留原始日期和修订说明，阅读时结合[运行指南](../guides/runtime.md)及后续验证记录。
不同阶段的参数、路线和目标可能不同，不将历史方案自动视为当前默认设置。

| 文档 | 内容 |
| --- | --- |
| [SAM 3.1 架构评估](sam31-evaluation-20260912.md) | 原生时序跟踪的适用性与改造方案 |
| [SAM 3.1 实现说明](sam31-implementation-20260912.md) | 连续跟踪路径与验收边界 |
| [三层架构与并行优化](parallel-latency-design-20260916.md) | 显示、判断与后台整理的职责划分 |
| [SAM 实时队列方案](sam-realtime-speed-plan-20260920.md) | 有界队列实验及其限制 |
