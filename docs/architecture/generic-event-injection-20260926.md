# 通用事件注入（Generic Event Injection）设计方案

设计基线：2026-09-26，基于 `pengyu/harness-offline-benchmark`（`7a83ce4`）。选这条分支而不是 `origin/main`，是因为离线 benchmark 通路只存在于此；代价见第 6 节。本文只描述设计与拆解，不代表已验收。相关：[持续人物与事件引擎](entity-event-engine.md)。

## 1. 问题

harness 当前只能检测固定的安全事件四件套 `fight / smoking / fall / weapon`（加上独立路径的 `stranger`）。benchmark 题库里大量 meva / caviar 类日常语义事件——`person talks to person`、`opens door`、`texts on phone`、`walking`、`vehicle turns`、`shop enter`——模型从未被要求观察，最终作答阶段自然答不出来，这批题目目前是被**剔除**后才跑的 benchmark。

已有的离线 benchmark 通路（`pengyu/harness-offline-benchmark`）已经能把问题文本喂进每个观察窗口，返回 `qa_observations`，再用纯文本证据做最终作答。实测结论是只有 `physical_conflict`（映射到 harness 的 `fight`）提升明显，其余事件与基线相当——因为事件检测分支仍然只问那四个事件，题目相关的证据从来没有被采集进事件时间线。

本方案要补的是缺失的那一层：**运行时可注入的事件定义**。事件词表和事件语义从配置或请求进入，而不是散落在 prompt 文本和各处 `frozenset` 里。

## 2. 现状盘点：哪些已经通用，哪些写死

整条管线**已经按 `event_labels` 元组参数化**，这是本方案成本可控的前提。事件生命周期本身与标签无关：

```1464:1478:src/goodvision_harness/scene_state.py
        if detected is True:
            event.last_positive_timestamp = timestamp
            event.positives += 1
            event.negatives = 0
            if event.positives >= self.confirm_observations:
                event.phase = "confirmed"
                if event.confirmed_since_timestamp is None:
                    event.confirmed_since_timestamp = timestamp
        elif detected is False:
            event.negatives += 1
            event.positives = 0
            if event.event_id and event.negatives >= self.clear_observations:
                event.phase = "ended"
                if event.ended_since_timestamp is None:
                    event.ended_since_timestamp = timestamp
```

`suspected → confirmed → ended` 的推进只依赖 `confirm_observations` / `clear_observations` 阈值和 `(event_label, participants)` 这个键，没有任何按标签的特判。归档、证据修订、时间线投影同理。

真正卡住的是六处白名单校验、一段写死的 prompt 文本，以及若干按标签枚举的展示与业务规则：

| 位置 | 内容 | 性质 |
| --- | --- | --- |
| `grounding.py:43-45` | `SAFETY_EVENT_LABELS` / `DETECTION_EVENT_LABELS` 常量 | 词表来源 |
| `grounding.py:2073` | `build_selected_features_grounding_client` 拒绝表外标签 | 校验 |
| `service.py:4400-4420` | `_normalize_features` 拒绝表外 feature，并按固定顺序归一 | 校验 |
| `scene_inspection.py:19,82` | `_EVENT_LABELS` + `unsupported temporal event labels` | 校验 |
| `scene_runtime.py:35,148` | `_EVENTS` 与默认 `event_labels` 元组 | 校验＋默认值 |
| `scene_state.py:21,1240` | `_EVENT_LABELS` + `observe_event` 的 `unsupported event_label` | 校验 |
| `scene_worker.py:54-56` | 日志覆盖统计与四件套取交集 | 统计口径 |
| `config.yaml:161` 起 | `multi_event_prompt`：四件套 JSON 示例 + 每事件取证规则 + 每事件出框规则 | prompt 文本 |
| `event_sensitivity.py:8-13` | 每标签的敏感度默认值与选项文案 | 按标签枚举 |
| `grounding.py:165` | `_EVENT_NAMES_ZH` 事件中文名 | 按标签枚举 |
| `overlay_fusion.py:69-70` | `_PERSON_TARGET_EVENTS` / `_OBJECT_TARGET_EVENTS` | 按标签枚举 |
| `event_archive.py:1384`、`person_registry.py:1357` | `fight` 多人特判 | 按标签特判 |

另外有一条**已经存在的动态通路**值得利用：`AIClubMultiGroundingClient` 在请求标签不等于固定四件套时，会放弃 `config.yaml` 的固定 prompt，改走通用 prompt + `build_multi_grounding_messages` 动态拼出的 contract。

```1350:1353:src/goodvision_harness/grounding.py
        use_fixed_prompt = (
            bool(multi_prompt) and self.event_labels == SAFETY_EVENT_LABELS
        )
        prompt = multi_prompt if use_fixed_prompt else self.settings.prompt
```

这就是要扩建的地基：保留固定四件套 prompt 供交付链路逐字使用，自定义事件走渲染路径。

## 3. 设计

### 3.1 事件定义注册表

新增 `src/goodvision_harness/event_registry.py`，把"一个事件是什么"收敛成一条结构化定义，取代分散的 `if label == ...`：

```python
@dataclass(frozen=True, slots=True)
class EventDefinition:
    label: str                      # 稳定的机器标签，snake_case
    zh_name: str                    # 中文显示名（原 _EVENT_NAMES_ZH）
    definition: str                 # 阳性判据：什么样的可见证据算这个事件
    counter_examples: str = ""      # 阴性判据：哪些相似情形不算
    target: str = "person"          # person | object | vehicle | scene
    target_hint: str = ""           # 出框对象的自然语言描述
    requires_subject: bool = True   # 是否必须绑定到 SAM 追踪的 subject_id
    default_sensitivity: str = "standard"
    sensitivity_options: tuple[SensitivityOption, ...] = ()
    builtin: bool = False           # 内置事件禁止被注入定义覆盖
```

现有五个事件（含 `stranger`）**先原样搬进注册表**，字段内容逐字沿用今天的 prompt 片段和敏感度文案。内置行为零变化是这个改动能不砸掉交付路径的前提，也是第一批单测要锁住的东西。

注册表内容 = 内置五件套 + 配置注入 + 请求注入，三者按优先级合并，内置项不可被覆盖（避免 benchmark 注入把 `smoking` 的取证下限冲掉）。

### 3.2 三处注入入口

- **配置注入**：`config.yaml` 新增 `events.definitions:` 列表，字段与 `EventDefinition` 对齐。适合长期固定的自定义事件词表。
- **会话选择**：`POST /v1/sessions/start` 已有的 `features` 数组从词表里挑本次要测哪些，词表外的标签一律 `unsupported feature`。
- **benchmark 注入**：离线 benchmark 按当前视频涉及的题目，自动带上对应事件定义（见 3.6）。

> 2026-09-26 阶段二实施修正：原计划的第三个入口——`POST /v1/sessions/start` 的 `event_definitions` 数组——没有做。部署级词表 + 会话级 `features` 子集选择已经覆盖了 benchmark 的需求（每个视频只需挑子集，不需要现场造新定义），而请求级定义注入会把一段自由文本从 HTTP 边界直送模型 prompt，风险与收益不成正比。若将来确有需要再加，`vocabulary_from_json` 的校验已经按"不可信输入"写好，接上即可。

### 3.3 校验改造

第 2 节表中六处白名单校验，统一改成"注册表里有没有这个 label"。要点：

- `_normalize_features` 现在同时承担校验和**排序归一**（`DETECTION_EVENT_LABELS` 的固定顺序）。自定义事件没有天然顺序，改成"内置事件保持原顺序在前，自定义事件按注入顺序在后"，保证同一请求的 prompt 与响应数组顺序稳定可比。
- `scene_state.observe_event` 的 `1 <= len(subject_ids)` 约束对 `requires_subject=False` 的事件放宽（见 3.5）。
- `scene_worker.py` 的覆盖统计不能再对四件套取交集，否则自定义事件在日志里永远是"未请求"。注意该函数注释明确写了"Only fixed public labels enter logs"——自定义标签进日志需要确认是否触碰隐私边界，见第 6 节。

### 3.4 Prompt 从定义渲染

`build_multi_grounding_messages` 现在把出框规则写死成"fight 和 fall 出人框、smoking 出烟具框、weapon 出武器框"。改为遍历本次请求的定义，按 `target` / `target_hint` 渲染；每个事件的判据来自 `definition` 与 `counter_examples`；`events` 数组的 JSON 示例按请求标签数生成。

同样的渲染器供 `scene_inspection.build_temporal_messages` 使用——它已经把 `allowed_event_labels` 作为 JSON context 传入，但每个标签的语义仍然只存在于 `_PROMPT` 常量和 `sensitivity_policy_text` 里，自定义事件送进去等于只给了一个模型没见过的词。

`config.yaml` 的 `multi_event_prompt` 保留不动，继续只在请求恰好等于四件套时使用。

### 3.5 场景级事件与非人主体

这是唯一需要动事件状态机的部分，也是 `vehicle turns`、`shop enter` 这类题目的前提。

现状：verdict 必须绑定到 SAM 追踪出来的 `subject_ids`，否则 `scene_inspection._validate_job` 直接以 `inspection target lacks a source-frame location` 拒绝，`scene_state.observe_event` 也要求非撤回记录至少一个 subject。SAM 只追人，车辆和门没有 subject。

方案：`requires_subject=False` 的事件走**场景级事件**路径，事件键从 `(event_label, participants)` 退化为 `(event_label, ())`，即整个场景一条时间线。生命周期完全复用现有阈值推进，因此 `first_evidence / confirmed_since / ended_since` 照常产出，temporal 类题目可答。这类事件不进 person registry，不参与身份合并，不做人物着色。

`scene_state.py:554` 的"关联人物全部消失即结束"规则对场景级事件不适用（没有关联人物），只保留连续 clear 计数这一条结束路径。这一点必须在实现时显式处理，否则场景级事件永不结束。

### 3.6 benchmark 侧衔接

`benchmark/bench/schema.py` 目前只有一行映射：

```42:44:benchmark/bench/schema.py
EVENT_TARGET_TO_HARNESS_LABEL = {
    "physical_conflict": "fight",
}
```

改成从一份**人工撰写**的事件词表（`benchmark/events/vocabulary.yaml`）生成。词表每条对应一个 `EventDefinition`，覆盖题库里出现的 meva / caviar 事件。定义文本必须人工写，不能让模型自己从题面造——否则 prompt 里的判据和评分标准会互相漂移，跑出来的分数不可解释。

离线 runner 在启动某个视频的 session 前，按该视频的题目集合求出需要的事件标签，把对应定义随 session 创建一起注入。

## 4. 分阶段拆解

每个阶段独立可合、可回滚，顺序不可调换（后一阶段依赖前一阶段的注册表）。

**阶段一：注册表落地，行为零变化。** 新增 `event_registry.py`，五个内置事件搬进去，`_EVENT_NAMES_ZH`、`event_sensitivity.py` 的默认值与选项改为从注册表读取。六处白名单改为查注册表，但注册表此时只有内置项，对外行为完全不变。验收：现有回归全绿，且新增一条"四件套请求渲染出的 messages 与改动前逐字节相同"的快照单测。

**阶段二：注入入口 + prompt 渲染。** 打通 `config.yaml` 与会话 payload 两个注入口，`build_multi_grounding_messages` 和 `build_temporal_messages` 改为从定义渲染。验收：用 `DryRunMultiGroundingClient` 注入一个自定义事件（如 `texts_on_phone`）跑通端到端，检查 prompt 里出现了该事件的判据与出框指引，响应能被解析进事件时间线并走完 `suspected → confirmed → ended`。

> 阶段二实际交付：`events.definitions` 配置注入、`EventVocabulary` 贯穿 `config → service → grounding`、`build_multi_grounding_messages` 渲染注入事件的判据/反例/出框指引、`_normalize_features` 与能力接口按词表放行。
>
> **未交付、且是阶段三第一件事**：生命周期还进不去。`scene_state.py:1240` 的 `_submit_event` 仍然拿 `_EVENT_LABELS`（只有内置五件套）卡 `event_label`，`scene_inspection._validate_job` 同理。也就是说注入事件现在能被提问、能渲进 prompt、能拿到模型判定，但那条判定会被事件状态机拒收，走不到 `confirmed / ended`，temporal 类题目答不了。把 `EventVocabulary` 接进 `SceneState` 与 `scene_inspection`，和场景级事件（`requires_subject=False`）改的是同一条路径，合并到阶段三一起做。`build_temporal_messages` 的定义渲染同理。

**阶段三：打通生命周期 + 场景级事件。** 先把 `EventVocabulary` 接进 `SceneState._submit_event` 与 `scene_inspection._validate_job`，让注入事件进得了状态机；再做 `requires_subject=False` 路径、事件键退化、结束规则调整，以及 `build_temporal_messages` 按定义渲染。验收：注入一个 person 类事件走完 `suspected → confirmed → ended` 且取证归档正常；再注入一个 `scene` 类事件，确认时间线三个时间戳齐全、不污染 person registry、能正常结束。

**阶段四：benchmark 接入。** 事件词表 YAML + runner 侧自动注入 + `EVENT_TARGET_TO_HARNESS_LABEL` 泛化。验收：对之前被剔除的那批题目跑一轮离线 benchmark，与"剔除前 / 剔除后"两个基线对比覆盖率与得分。

## 5. 明确不做

- 不新增第二套事件状态机、第二条模型路由、第二套证据归档。自定义事件复用现有全部下游。
- 不做自动事件发现（让模型自己说"我看到了什么事件"）。本方案只做"按给定定义去判定"，开放式描述是另一个问题。
- 不改 SAM 追踪范围。非人目标不做追踪，只做场景级判定。
- 不动 `config.yaml` 的四件套 `multi_event_prompt`，交付链路的 prompt 逐字保持。

## 6. 风险与待确认

**分支基线（已决策）。** 四个阶段全部在 `pengyu/harness-offline-benchmark` 上做，完成后一次性 rebase 回 main。选它是因为阶段四需要 `benchmark/` 目录，且阶段二要改的 `build_temporal_messages` 的 benchmark 分支也只在这条分支上存在——在 main 上无法验证"自定义事件定义是否真的进入了观察 prompt"。

以共同祖先 `b6e3f14` 为基准，两侧的改动分布如下（只列本方案要碰的文件）：

| 文件 | PengYu 侧 | main 侧 | 本方案的改动 |
| --- | --- | --- | --- |
| `grounding.py` | — | — | 词表常量、`build_multi_grounding_messages` 渲染 |
| `scene_state.py` | — | — | `_EVENT_LABELS`、`observe_event` 的 subject 约束 |
| `overlay_fusion.py` | — | — | 出框目标改为查 `target` |
| `scene_inspection.py` | +88 | — | `_EVENT_LABELS`、prompt 渲染 |
| `event_sensitivity.py` | — | +30 | 默认值与选项改为查注册表 |
| `scene_runtime.py` | +22 | +28 | `_EVENTS` 与默认元组 |
| `service.py` | +320 | +659 | `_normalize_features` + session 入参 |

前三个文件两侧都没动，rebase 时零冲突，而本方案的主要改动恰好集中在这里。真正的冲突面只有 `service.py` 和 `scene_runtime.py`——两侧都在改，从哪条分支起步都躲不掉。

已知代价：`event_sensitivity.py` 的新版本只在 main 上，本分支上是旧版。阶段一在这个文件上的改动是对着旧版做的，rebase 时需要按 main 的新版重做一遍，不要盲目接受任何一侧。

原则：改动尽量集中在新增文件（`event_registry.py`、渲染器模块）里，对既有文件只做"删掉 frozenset、改成查注册表"这类小切口，把 rebase 成本压在可控范围内。

**自定义标签入日志的隐私边界。** `scene_worker.py` 注释写明只有固定公共标签进日志。benchmark 注入的标签来自公开题库，应当安全，但需要确认这条约束的来源是隐私审计还是别的合规要求。

**敏感度 UI。** console 的事件敏感度选择器目前按四件套写死选项。自定义事件默认只给 `standard`，是否需要在 UI 里暴露，取决于 benchmark 是否要做敏感度扫描。

**词表规模。** 题库里被剔除的题目具体覆盖多少个不同事件、各占多少比例，会直接影响阶段三的优先级——如果非人主体事件占比很低，阶段三可以延后，先用阶段二把覆盖率从 4 个拉到二三十个。
