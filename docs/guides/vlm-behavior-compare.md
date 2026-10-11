# 十二路事件质量与排队延迟对照

这个工具在十二路并发负载下，比较同一批原视频证据、同一规则下的 Qwen events 和
grounding 输出。它同时统计与 Astra 自动参考的一致性、疑似误报/漏报、不确定、
请求完成情况和排队延迟。默认只生成计划；只有显式 `--real` 才调用付费模型。

## 固定计划

- 只测十二路；每路最多 4 个在途任务、总上限 48、API 准入上限 20 次/秒。
  每次请求只包含一路的一个 3–4 帧证据窗口，48 不是 batch 大小。
- 每路每秒送入 1.25 个冻结窗口，各路源时钟错峰，总计划到达率 15 个窗口/秒。
  证据中的原视频时间戳不变；负载按固定节奏回放，不包含视频解码、摄像头上传或显示。
- 模型 `qwen/qwen3-vl-30b-a3b-instruct`；OpenRouter 外部推理，Harness 主机负责请求调度。
- **固定完整模型 ID `qwen/qwen3-vl-30b-a3b-instruct`**，脚本拒绝改成其他模型。
  请求只传这一个 `model`，不配置备用 `models` 列表或模型自动路由。
- 服务商不再限定 Alibaba。使用 `provider.sort=latency`、`require_parameters=true`、
  `allow_fallbacks=true`，允许 OpenRouter 在提供同一模型的服务商之间选择和回退。
  不再将公共目录是否列出 Alibaba 作为启动门槛，也不发起额外付费探测。
- 为隔离服务商变化，可显式传 `--provider-only deepinfra`（也支持 `novita`、
  `siliconflow`）；此时使用 `provider.only` 且禁用服务商回退，完整模型 ID 不变。
  默认仍为自动路由。所选服务商、回退策略和实际返回服务商都会保存；固定服务商
  不保证其永远可用，不应通过降低解析校验标准来提高返回率。
- 每次实际请求记录 `requested_model` 和返回的服务商分类；未报告服务商单列为
  `unreported`，未知名称归为 `other`。服务商变化本身不会让有效判断作废。
  这是自动路由条件下的模型对照，不能据此排除不同服务商/量化实现对延迟或输出的影响。
- 使用生产 `EventJSONInspector` 与 `CompactGroundingInspector`，六类事件共享同一策略。
  相同 JPEG、源时间戳、图像参数和敏感度；参考标签不进入提示词。
  `observed_detected` 是比较的原始场景判断，不将定位或确认阶段的结果当作模型原始判断。
- 分四阶段运行：events → grounding → grounding → events。每阶段十二路，每路 69 组，
  共 828 个计划窗口；四阶段最多 **3312 次请求**。默认硬预算 3500，不自动重试。
  两模式各重复两轮，反转顺序减少部分顺序影响，不能证明某项改动单独造成提升。
- 默认 events 最长 5 秒/160 tokens，grounding 最长 20 秒/1600 tokens，保留生产接口差异。
  没有额外付费预热。首次创建客户端耗时单列，包含在输入到结果发布的时间中。

公共路由资料：[服务商路由与 fallback](https://openrouter.ai/docs/guides/routing/provider-selection)，
[该模型端点目录](https://openrouter.ai/api/v1/models/qwen/qwen3-vl-30b-a3b-instruct/endpoints)。

## 队列规则与异常处理

复用生产 `SharedAdmission`，共享 SQLite 同时控制总并发、每路并发、请求发起间隔。
准入优先选择最久未获准的源；同一源/分支只保留最新未获准票据。外层每路最多保留一个
尚未交给工作线程的窗口，较新输入替换较旧待发窗口，工作线程不会无限排队。

外层待发和 admission 等待各自最多 1 秒，源截止时间为进入测试分支后 3 秒。
这个截止用于请求发起前的丢弃，不是响应必须在 3 秒内完成；输出中的 3 秒累计比例
也只是观察阈值，不自动当作业务验收要求。其他合法参数调整均记录在 manifest 中。

每一个队列替换、等待超时、源窗口错过、请求失败和未执行计划都保留在统计中。
若发送线程落后计划超过 0.1 秒，记录 `source_missed`，不在恢复后把旧窗口突发补发；
对应阶段输入时序无效，不把降低负载后得到的成绩称为十二路有效容量结果。
输入时序有效本身也不等于质量或容量达标，应结合完整率和逐路延迟判断。

HTTP 401/402/403/429、客户端异常，或同一路连续三次
不成功的已尝试请求，都会停止后续派发。已发出的请求在有限超时内收尾，排队者取消，
尚未到达的后续计划记为未执行。报告保留部分结果。输出目录必须是新目录；不自动续跑
中断任务，避免重复调用与不清楚的计费。

## 降低定位调用频率的可选对照

### 精简定位输出（不降低输入频率）

`--grounding-format lean-v1` 启用版本化的短字段格式。默认仍为 `legacy`，
生产默认配置不变。首轮对照使用 `--grounding-interval 0`，保留所有原始窗口，
单独观察输出格式的影响；队列、截止时间、模型、图像与事件策略均不改变。
可使用 `--provider-only deepinfra` 固定同模型服务商并禁用回退。

新格式示例（仅说明结构，不是模型或参考答案）：

```json
{"v":1,"s":[{"i":"p1","b":[100,100,500,900]}],"e":{"fight":[{"p":["p1"],"d":false,"c":0.8}],"smoking":[{"p":["p1"],"d":true,"c":0.8}],"fall":[{"p":["p1"],"d":null,"c":null}],"weapon":[{"p":["p1"],"d":false,"c":0.8}],"ppe_violation":[{"p":["p1"],"d":true,"c":0.8}],"ground_litter":[{"p":[],"d":false,"c":0.8}]}}
```

- `v=1` 明确声明格式版本与整数 0..1000 坐标；不猜测坐标单位。
- `s` 每个人只返回一次局部引用 `i` 和最新帧框 `b`；无法定位时 `b=null`。
- `e` 必须包含全部六类，每类至少一个 `{p,d,c}` 判断：人物引用、布尔/不确定、
  置信度。不同人员可以有不同结果；同一判断/置信度可以合并人物引用。
- 继续使用相同事件策略与策略指纹，保留正、负、不确定和置信度；内部仍转换成
  `IndependentVerdict`，现有场景汇总、人物关联与质量报告不改语义。
- 严格 JSON Schema 约束输出，应用端再次校验。重复字段/人物、事件缺失、额外字段、
  错误类型、重复事件参与组、超出容量、未正常结束/截断都拒绝；不会自动补阴性。
  坏框、未定位或引用不存在的人员保留事件原始判断，但标记定位失败，不生成可用框。
  地面垃圾必须为场景判断 `p=[]`。
- 新格式不接受旧的 `no_events` 快捷回复。因此完整定位回复通常更短，全阴性快捷
  回复则可能更长。结构示例的字节减少不等于实际 token、延迟或识别质量已经改善。
- manifest 保存 `grounding_format`、实际 adapter 与完整 response schema；报告展示格式。
  直连模型与预处理委托接口均识别 `lean-v1`，委托端也需部署此版本。

先生成计划或离线模拟，不发起付费调用：

```bash
python tools/vlm_behavior_compare.py \
  --packet outputs/behavior-annotation/astra-unified-v2-20261009/packet \
  --grounding-format lean-v1 --grounding-interval 0 --provider-only deepinfra
```

首次真实使用应先确认服务商支持该 schema，再对同一证据比较 legacy/lean-v1。
`--modes grounding` 可仅执行定位分支，避免为了格式比较重复调用未修改的 events。
此时只有实际选择的模式进入计划和统计，不生成虚构的 events 结果，两模式配对表为空。
格式对照可按 legacy、lean-v1、lean-v1、legacy 顺序，各跑一轮十二路完整窗口；
不同格式的判断质量另按相同视频、窗口、路号配对。该对照不代表两种模式同时运行。
按完整返回率、全部窗口分母、分段延迟、实际输出 token 和 Astra 参考一致性共同判断。
先前降频试验遇到的 HTTP 429 停止规则仍生效，本变更不重试或绕过限流。

### 降低定位采样频率

`--grounding-interval 4` 或 `5` 保持原视频时间戳以 1 倍速推进，events 仍选择全部
原证据；grounding 在每隔 4/5 秒的源时间点选择当时已经到达的最新证据窗口。
不会将 69 组依次每 4/5 秒发送，从而慢放原视频；也不会读取采样点之后的未来帧。
未选择的窗口标记 `not_sampled`，不调用模型、不算返回成功，也不当成模型阴性或漏报。
当前 69 组证据中，4 秒选中每路 14 组，5 秒选中每路 11 组；两轮总调用上限分别为
1992 和 1920（包括 events），原始窗口槽位仍分别为 3312。

报告同时给出完整结果/选中请求、完整结果/全部原窗口、主动未采样，以及参考阳性中
选中和未采样的数量。高返回率不等于高时间覆盖率，也不代表原输入负载下提速。
沿用同一 69 组原图和自动参考，两模式只在同源、同轮、同窗口均有明确结果的子集配对。
`evidence_to_publish_seconds` 另外计入选中证据在采样时已经过去的源时间，
用于真实运行的结果年龄诊断；不包含视频解码、显示，模拟加速结果不解释为真实时延。

如需在存在响应超时的条件下完成固定对照，可显式使用 `--continue-on-timeout`。
超时仍记为失败，仍占计划分母与调用预算，不延长请求期限、不重试、不记为阴性；
超时不再触发连续失败停止，并清零该源的连续其他失败计数。认证、额度、限流、
客户端异常以及连续三次其他不成功请求的停止规则保持生效。即使启用该选项，同一路
连续三次在 HTTP 请求上传完成之前的可观测传输超时仍会停止，避免将 DNS/连接故障
当作模型性能测量。该选项写入 manifest，
不能把这类完整计划与提前停止的部分运行直接当成同条件性能提升。
停止派发后仍逐个收集已完成的请求，避免等待最慢请求收尾时人为增加其他结果的发布延迟。

若实例的系统 DNS 查询有延迟，可独立验证进程级 `RES_OPTIONS=use-vc`（TCP DNS）。
这不会更换模型/API 域名、固定服务端 IP 或禁用证书验证；环境差异会以 `dns_transport`
写入 manifest。须先做无推理连接检查，不能仅凭超时就断言 DNS 有问题。

## 运行

从仓库根目录运行，素材包包含 `input-manifest.json`、`event-policy.json`、
`annotations.json` 和原始 JPEG 帧。当前包：
`outputs/behavior-annotation/astra-unified-v2-20261009/packet`。

只校验和生成计划（不读密钥、不联网、不调用模型）：

```bash
python tools/vlm_behavior_compare.py \
  --packet outputs/behavior-annotation/astra-unified-v2-20261009/packet
```

离线模拟（合成全阴性响应，通过生产解析器和队列；不是模型质量/性能结果）：

```bash
python tools/vlm_behavior_compare.py \
  --packet outputs/behavior-annotation/astra-unified-v2-20261009/packet \
  --simulate --simulation-time-scale 0.1
```

加速仅用于离线回归；源时钟、模拟模型等待、速率与队列超时同步缩放。
操作系统调度、客户端初始化和 SQLite 开销不会缩放，因此它也可能出现源错过或排队跳过；
不能把加速模拟的延迟当作真实推理速度。`--real` 禁止加速。

正式十二路对照：

```bash
python tools/vlm_behavior_compare.py \
  --packet outputs/behavior-annotation/astra-unified-v2-20261009/packet \
  --real
```

默认保留 OpenRouter；使用 `--backend dashscope` 可直接调用阿里云百炼的
`qwen3-vl-30b-a3b-instruct`，共用相同的证据包、六类规则、Astra 参考、十二路队列和计分。
可用 `--endpoint` 指定同地域的完整 Chat Completions URL；省略时为北京共享域名。
直连禁用其他服务商回退，不支持 `--provider-only`；使用 JSON Object 并保留本地严格校验。

```bash
python tools/vlm_behavior_compare.py \
  --packet outputs/behavior-annotation/astra-unified-v2-20261009/packet \
  --backend dashscope \
  --endpoint https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions \
  --grounding-format lean-v1
```

以上仅生成计划；`--simulate` 验证离线流程，`--real` 才会调用模型。
百炼只读取 `DASHSCOPE_API_KEY`，OpenRouter 只读取 `OPENROUTER_API_KEY`，不会互相回退；
先读所选渠道的环境变量，再读仓库 `.env`。可通过 `--env-file`
指定文件。不要把密钥写到命令参数。`--output` 指定尚不存在的目录；未指定则创建时间戳目录。
计划超出预算时，在读取密钥、联网前拒绝。实测与现有 VLM suite 共用锁，避免同时压测。

## 判断质量的统计口径

- 相对 Astra 的 TP/TN 是阳性/阴性一致，FP/FN 是疑似误报/疑似漏报；不是人工确认的错误。
- Astra `null` 单列并排除对错评分。模型不确定、失败、类别不完整、队列跳过和未执行分别报告。
  请求未返回不能当成阴性或漏报；不完整请求整组不参与二值评分，部分输出仍保留日志。
- 一致/全部明确参考 = (TP+TN) / 全部 Astra 明确项目。
  未返回、不确定、失败和跳过都仍在分母中，不能靠少回答提高该分数。
- 二值覆盖率 = 双方均有明确判断的项目 / 全部 Astra 明确项目。
- Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2TP/(2TP+FP+FN)，**只对双方均明确子集计算**。
  分母为零显示不可评估，必须和二值覆盖率一起看。
- 另外保留 TP/全部 Astra 阳性及 TN/全部 Astra 阴性，体现未完成或不确定的影响。
- 两模式配对必须是同一路、同一轮、同一窗口，并且 Astra 与两模型输出均明确。
  不把一条路的 events 与另一条路的 grounding 拼成一对。
- 提供总体与逐路逐类结果。十二路重复同一 69 组证据，不是 828 个独立事件样本。
  相邻窗口有重叠，两轮也重复同一视频；本轮不能推断独立场景泛化或并发相对单路的因果影响。
- 当前参考无武器阳性、防护违规阴性，跌倒阳性仅一个窗口；吸烟有大量不确定。
  不足以证明六类事件均准确。框定位/跟踪正确率需要额外人物框与对应关系标注，本轮不评分。

## 延迟的统计口径

全部时间点来自同一进程单调时钟，每条记录可重建各段：

1. 计划送入 → 实际送入：源输入迟到，不计入排队，单列检查负载有效性。
2. 实际送入 → 工作线程开始：外层等待。
3. 工作线程开始 → admission 获准：共享准入等待。
4. 获准 → inspect 开始：客户端准备。
5. inspect 开始 → inspect 返回/异常：图像构造、API 往返、模型响应和解析，含失败实际等待。
6. inspect 结束 → 结果整理和释放名额完成：结果整理。
7. 整理完成 → 测试收集器接收结果：发布等待。

排队 = 2+3；输入到发布 = 2+3+4+5+6+7；计划到发布再加 1。
“发布”指测试收集器收到结果，不代表用户已在浏览器看到；不含录像或浏览器显示。
API 内部细分沿用生产 `http_timing` 与 `inspection_timing`，不能将其中重叠阶段相加。

每阶段及每路提供 P50/P95/P99/最大值、样本数、直方图和 1/2/3/5 秒完整结果比例。
HTML 延迟表的排队只含已发起请求，调用耗时包含成功与失败，输入到发布只含完整结果。
跳过窗口的等待时间、所有已记录结果的耗时在 JSON/CSV 中另列，避免快速丢弃拉低完整结果延迟。
累计完成比例以全部计划窗口为分母；不同样本集合的分位数不能相加。

## 交付文件

- `report.html`：完成情况、总体/逐路延迟、逐类/逐路一致性、同源配对、原图和分歧。
- `manifest.json`：完整计划、固定模型和自动路由参数、并发/速率/时序、模式参数、代码和素材指纹。
- `results.jsonl`：已记录窗口（包括跳过）；实际请求附完整生产报告、事件/框/定位及阶段计时。
- `summary.json`：各轮各模式分别统计，不用总体平均掩盖某一路失败。
- `per-stream-quality.csv`、`request-latencies.csv`：逐路质量和所有计划窗口的阶段延迟。
- `comparisons.csv` / `.json`：逐项判断，包括缺失与不确定。
- `inputs/`：冻结原图、规则、自动参考，报告整体可搬移。

所有报告写“与 Astra 自动标注的一致性”，不称为已验证真实准确率。
