# 文档导航

[项目概览](../README.md) · [开发与贡献](../CONTRIBUTING.md) · [测试说明](../tests/README.md) · [工具清单](../tools/README.md)

按任务选择入口。命令和未加链接的工程路径默认相对于仓库根目录。
带日期的方案、排障和验证记录只描述当时版本与环境；其中的“已部署”“通过”和性能数字不代表当前版本已重新验收。

<a id="delivery-and-integration"></a>
## 安装与接口集成

客户交付文档保留在仓库根目录，供交付包和启动流程使用。

| 任务 | 文档 |
| --- | --- |
| 安装客户交付包 | [部署说明](../DEPLOY.md) |
| 查看发布版本及已完成检查 | [发布记录](../RELEASE.md) |
| 对接结果、图片、SSE 与 Webhook | [公共 API](../API_DELIVERY.md) |
| 对接多路视频输出 | [N 路视频接口](reference/multi-camera-api.md) |
| 配置 Windows DirectML | [Windows 混合部署](../WINDOWS_HYBRID.md) |
| 检查结果数据结构 | [AnalysisResult v2 schema](../schemas/analysis-result-v2.schema.json) |

<a id="guides"></a>
## 使用与运维

| 任务 | 文档 |
| --- | --- |
| 配置调度、人脸与视觉大模型 | [运行指南](guides/runtime.md) |
| 了解设置优先级、推理路线与参考记忆 | [配置指南](guides/configuration.md) |
| 从源码部署、使用 Compose 与 dry-run | [源码部署](guides/deployment.md) |
| 在本机 NVIDIA 显卡上运行多路视频 | [本地 GPU 部署](guides/local-gpu.md) |
| 使用受管云 GPU 运行多路追踪 | [云 GPU 部署](guides/cloud-gpu.md) |
| 将主程序、追踪与大模型部署到不同机器 | [分机部署](guides/modular-services.md) |
| 使用持续场景分析 | [持续场景指南](guides/continuous-scene.md) |
| 检查时序请求和运行回放 | [时序诊断与回放](guides/diagnostics-replay.md) |
| 配置与验收 Contour 实时队列 | [Contour 验收指南](guides/contour-realtime-validation.md) |
| 构建和检查发布包 | [发布构建指南](guides/release.md) |

本地开发和人物登记见[贡献指南](../CONTRIBUTING.md#local-development)；
工作进程的安装与接口见 [SAM scene](../workers/sam_scene/README.md) 和 [Contour](../workers/contour/README.md)。

<a id="architecture-and-implementation"></a>
## 架构与设计

[架构索引](architecture/README.md)收录事件引擎、身份与证据、网络显示、VLM 时序和 SAM 方案。
架构目录保留机制说明和设计文档，仍有用的故障结论、修复边界及测量汇总放到验证目录。

[Generic event injection](architecture/generic-event-injection-20260926.md) describes runtime event definitions and question-driven event selection.

<a id="validation-records"></a>
## 验证与排障记录

[验证结论索引](validation/README.md)：网络与显示、事件与归档、追踪性能、识别质量、交付升级。

每个专题集中维护有效结论、关键证据和待验收项，删除被替代的过程记录。
必要的实测依据保存在 `validation/evidence/`；原始媒体及运行日志存放在忽略目录。

<a id="internal-reference"></a>
## 内部参考

[推理路线目录](internal/route-catalog.md)记录内部映射与历史供应商比较，与客户文档分开维护。

## 新文档放哪里

| 目录 | 内容 |
| --- | --- |
| `guides/` | 使用、部署、配置和发布操作 |
| `reference/` | 专项接口和字段参考 |
| `architecture/` | 系统机制与设计方案，不放单次运行日志 |
| `validation/` | 按专题维护的验证结论、未关闭问题和必要证据 |
| `internal/` | 内部参考 |

新增文档时更新对应索引；排障与验证优先更新已有专题，避免再次堆积日期流水账。
移动或删除文档时同步修正引用；不再有参考价值的附件一并删除。

<a id="continuous-scenes-and-multi-camera-services"></a>
持续场景和多路部署入口已归入上方“使用与运维”和“安装与接口集成”。
