# 事件运行时代码组织

实现按职责分组，保持当前事件判断、采样、调度和结果语义。

| 包 | 实现 | 职责 |
| --- | --- | --- |
| `runtime` | `scene.py`、`scheduling.py` | 场景编排、源证据保留、检查调度与预算 |
| `events` | `state_machine.py` | 人物与事件证据账本、确认、解除和修订 |
| `actions` | `alerts.py`、`results.py` | 告警门槛、身份策略与公开结果投影 |
| `evaluation` | `replay.py` | 用生产状态机离线重放源观察 |

运行时调用事件状态机及告警策略；状态机和调度组件不调用模型或服务入口。
外部视频、模型路由、显示、诊断与归档模块保留现有职责。这一整理改善实现定位和
包的复用，不改变算法、并发或延迟，也没有把大型类拆成新的算法组件。

旧的 `scene_runtime`、`scene_state`、`scene_evidence`、`scene_alerts`、
`scene_results` 和 `tools.scene_replay` 是同一实现模块的兼容别名。
旧导入和替换模块全局变量的测试保持有效，生产代码使用新的实现路径。
`perception.py` 和 `model_routing.py` 保持原有模块接口。

离线重放可以使用已安装包，无需仓库的 `tools` 目录：

```sh
python -m goodvision_harness.evaluation.replay observations.jsonl
goodvision-event-replay observations.jsonl
```

旧命令 `python tools/scene_replay.py observations.jsonl` 继续可用。

这次整理不包含 Supervision 组件、Observation schema 或新事件评测算法；
最新主线的 PPE、地面垃圾、诊断和源时序行为全部保留。
镜像预检校验嵌套的真实实现，基准报告源码哈希也包含嵌套模块。
