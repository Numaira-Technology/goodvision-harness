# Ashley 原生＋Siyu：有界 main 事件规则冻结发布

日期：2026-10-06。发布分支为 `codex/ashley-main-event-update`，基于原生提交 `fca08e118c013d053ff880153dc6008584805e64`。本分支不是 SAM Harness 最终汇总实验分支，也没有把完整最新 main 合入原生方法。

## 来源和冻结状态

|项目|来源或状态|
|---|---|
|原生基础|`fca08e118c013d053ff880153dc6008584805e64`，原生 SAM 人物关联与事件处理方法|
|Siyu 通用事件参考|PR33 的 `239bfd2921ae4f365108989f93e0e9498885ddb3`；此前独立兼容接入，不等于整支合并|
|有界 main 事件语义参考|`52d85145a3b06cdeb7352d4ca5f624bbd50eb9f1`；取六 builtin 判据、人物/场景绑定和垃圾场景投影|
|本次准确复制来源|`harness-experiments/main-event-supplement30-20261006/ashley/source` 的全部 `src/`、根 `config.yaml` 与 `event-definitions.json`|
|正式冻结清单 SHA256|`90d36a78a3c25ca3899d1dd9d326725aba89a56cb6df8f09705c4fcffb12385e`；这是 `ashley/source-manifest.json` 文件的哈希，不是 Git 提交或单个源码哈希|
|公开词表 SHA256|冻结清单记录为 `241f7243b52be5947d2f88ddbd6e3e89fcb3c793135e6bbeeac905cd5bf6aa22`|
|配置 SHA256|冻结清单中根与包内 `config.yaml` 均为 `f5703ad82b3cc1261a49ee076c473ad715c173711b435cb59f656ab3a0e8b9b5`|
|正式评测状态|主控制器于 2026-10-06 06:35:34 UTC 启动，PID33183；顺序 SAM44→Ashley44。此记录只说明控制器已启动，不表示 Ashley 已运行或答题完成|

评测范围为已有14道非 `physical_conflict` 事件题加新增30题（MEVA 事件22、位置8），共44题；完整题库范围由152扩为182。发布过程中不修改正式冻结目录，不重跑冻结或启动脚本，不将本分支替换正在执行的快照。

本次新提交只包含源码、配置、公开动作定义和本文；不新增题集、GT、标准答案、媒体、模型请求/回执、输出账本或凭据。基线已经跟踪的其他文件保持原样。

## 方法边界和实际语义

原生观察阶段不读取问题或选项，`observation_reads_questions=false`；公开事件标签仅用于会话功能选择，窗口不接收题目文本。原生事件生命周期沿用两次确认、两次消除、同源帧一票以及既有来源时间规则。最后用既有确定性提取器读取原生档案，不增加最终 LLM，`final_llm=false`。原生方法本身使用 SAM 人物关联，不能据此把它称为“有最终 LLM 的 SAM Harness 汇总版”。

本次 main 有界接入包含 `fight / smoking / fall / weapon / ppe_violation / ground_litter` 六项安全 builtin，`stranger` 仍沿独立身份分支：

- PPE 绑定可见 grounded 人物：可见缺背心或头盔任一成立；遮挡、模糊和距离不足不证明缺失。
- `ground_litter` 绑定场景、人物主体为空：需可见废物接地，无需人物或丢弃动作；手持物、桶内物、阴影和正常存放工具排除。发布 `scene-ground-litter` 整帧场景范围，并标注 `bbox_semantics=scene_extent_not_litter_localization`，不声称垃圾定位。
- `weapon` 的 standard/high 采用 main 严格判据：可辨识武器特征，或可见且定向的武器用途；普通搬放物和徒手攻击不是武器证据。
- 车辆场景阳性可沿已声明路径空 `object_ids` 发布；人物事件阳性仍要求有效主体。无关联的阴性/unknown 保留窗口档案，不伪造人物绑定。

不迁移 main 的 events-only、共享 admission、latest-only 或20秒 inspection deadline；保留冻结源既有320秒 inspection deadline、离线背压、来源帧保留及确定性计数/时间口径。已有离线文件源、原始帧时钟、收尾与审计适配也随完整 `src/` 一起发布，不能只复制六事件相关文件后声称得到同一个运行版本。

## 固定公开词表：38＋13

原38项内容与次序保留，尾部追加13项。定义来源是正式范围的公开动作名字及公开 session context，由人工规范化为 `definition / counter_examples / target_hint`；未使用 GT、标准答案或模型成绩，不宣称13项全是 main 已实现的原生标签。

新增项为 `hand_interacts_with_person`、`person_closes_trunk`、`person_closes_vehicle_door`、`person_loads_vehicle`、`person_opens_facility_door`、`person_opens_trunk`、`person_purchases`、`person_puts_down_object`、`person_reads_document`、`person_rides_bicycle`、`person_sits_down`、`person_talks_to_phone`、`vehicle_picks_up_person`。

`person_talks_to_phone` 保留公开措词为独立标签，用可见手机和靠耳/嘴的通话姿态定义，不 rename/alias 到 `person_talks_on_phone`。`vehicle_picks_up_person` 是无人物主体的 scene 事件，沿既有车辆名单处理；不借行人框虚构车辆或车辆轨迹。没有增加其他身份或事件别名。

可信冻结配置最多51项，逐项字段、重复和 builtin 保护仍保留；不可信外部 API 的 `MAX_INJECTED_EVENTS=32` 不变。`event_session.py` 在进程导入时通过 `ASHLEY_EVENT_DEFINITIONS` 加载该固定文件，必须显式指向本分支的 `event-definitions.json`；不会在会话中途替换词表。本分支是该离线冻结方法的代码发布，不是未经适配即可替代生产 main 的通用部署包。

## 运行审计与外置提取器

`native_event_audit.py` 保留完整请求哈希、成功回执严格匹配、最多3次尝试、未知请求封锁、402止损和审计 IO 全局退出。完整旧 requests/responses/attempt ledger 用于恢复累计次数及 blocked；未知只封锁相同完整输入，不因一条旧未知全局阻断其他输入。普通 `three_consecutive_observation_errors` 只保存首次停止原因，后续普通错误不覆写；publisher 全局失败可升级普通停止，IO 全局失败仍最高优先。本次发布原样保留冻结 Ashley 审计代码，不另改缓存或模型返回算法；本轮 SAM 最终答案缓存资格修正属于另外的 adapter，不归入此 Ashley 分支。

审计入口由外置原生启动脚本安装，使用 `ASHLEY_OUTPUT_ROOT` 与 `ASHLEY_PREVIOUS_RESPONSES` 指向各自档案；离线媒体只接受 `/offline-media` 内只读挂载文件，离线 capture 写到 `/run/output`。这些运行目录及历史档案不随代码发布。完整评测另依赖冻结控制器、原生入口、媒体和外置提取器，本分支不打包这些运行材料。

legacy 原生生命周期判据及确定性计数/时间提取算法未改。正式评测仍使用既有 `ashley/extractor-snapshot` 和 `event-extract-snapshot.py`，本次不提交或修改外置提取器；冻结回执记录 `extract.py` SHA256 为 `c543ed53174ac7ddc617fab86eb28c6c2a83a81128e6830ebb1377d9d3a43d56`。只读可见，`event-extract-snapshot.py` 的 scene 标签名单仍为原7项车辆标签，没有新增 `vehicle_picks_up_person`，可能使该新增标签的阴性覆盖判定保留为 unknown；这不是本次发布中修复过的能力，也不能把词表扩展等同于准确率提升。

## 发布审阅范围

发布前只读查看 Git 状态、文件清单和文本差异；基于 fca 的新工作树只暂存所述四类文件，保留其他工作树的用户修改。发布不 force、不建 PR，不改变运行资源或正在执行的源。本次先保存本地提交；远端目标为 `Numaira-Technology/goodvision-harness` 的 `codex/ashley-main-event-update` 分支，待产品所需的具体目的地披露授权后才能 push，本文不表示远端更新已经完成。

未运行测试（未获用户授权）。没有运行 lint、格式/类型检查、语法检查、编译、构建、Docker验证、独立复现、健康探针或额外模型/GPU调用。本分支记录代码与方法来源，不提供测试通过或正式准确率结论。
