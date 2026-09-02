# VLM Rich Annotation 实验

本分支把 rich annotation 做成独立的第六项能力：在原有五类事件能力之外，额外调用一次 VLM，识别画面中的人和物品，返回边框和外观属性。它复用现有 rich-label 数据结构、画面叠加及事件框融合路径；原有事件的请求和结果格式保持兼容。

`rich_labeling.backend: vlm` 复用 `inference_service` 选择的推理来源、API 地址、密钥和线路参数。`rich_labeling.model` 未设置时使用该来源选定的模型。它支持 main 新增的 `INFERENCE_PROVIDER` 切换方式。这条 rich annotation 路径不需要启动 Paddle worker，也不需要安装 Paddle 的本地模型依赖。需要切回本地方案时可选择 `backend: paddle`。

## 属性与输出

属性沿用 Paddle worker 的命名和值：

| 内容 | 字段与示例 |
| --- | --- |
| 人与物品 | `label`、`bbox`、`confidence`、`track_id` |
| 外观性别估计 | `estimated_gender`: `female-presenting` / `male-presenting` |
| 大致年龄段 | `estimated_age_range`: `<18` / `18-60` / `>60` |
| 可见方向 | `estimated_direction`: `Front` / `Side` / `Back` |
| 袖长与衣服样式 | `sleeve_type`、`upper_style`、`lower_styles` |
| 衣服颜色 | `upper_color`、`lower_color` |
| 配饰、手持物 | `wearing_hat`、`wearing_glasses`、`bag_type`、`holding_object`、`wearing_boots` |
| 叠加显示摘要 | `appearance_attributes` |

年龄和性别仅表示模型根据画面作出的外观估计，不代表真实身份信息。提示词要求看不清、被遮挡或无法确定的属性省略，不要求模型补齐全部字段。返回的边框仍使用项目既有的归一化坐标格式。轨迹编号由本地代码按同类别框的重叠关联，超过 10 秒或物体消失后会分配新编号；它不是身份识别，快速移动或人物交叉时可能换号。

## 启动实验服务

在此 worktree 根目录、现有 Harness Python 环境中运行。实验默认使用 OpenRouter，请在 `.env` 配置 `OPENROUTER_API_KEY`，不要写进 YAML。main 的通用密钥 `INFERENCE_API_KEY` 仍具有最高优先级；切换来源时请同步核对该覆盖值。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
$env:INFERENCE_PROVIDER = "openrouter"
$env:HARNESS_CONFIG = (Join-Path (Get-Location) "config.vlm-rich.yaml")
$env:HARNESS_PORT = "7901"
python -m goodvision_harness.service
```

打开 [实验服务](http://127.0.0.1:7901)，按现有方式启动视频流或摄像头会话。实验配置保留原配置的视频裁剪和事件提示词。默认 `stream.frame_crop: left_half` 适用于左右并排的机器人画面；普通单目视频请改为 `none`。

OpenRouter 使用此前在 `4592cd8` 中经视觉延迟测试选定的 `qwen/qwen3.5-35b-a3b`，固定 `provider.only: [deepinfra]`、`allow_fallbacks: false`、`reasoning.effort: none`。事件检测和 rich annotation 均使用这条线路；rich annotation 仍有独立提示词和 4096-token 输出上限。`INFERENCE_PROVIDER=aiclub` 可切回 `qwen3.5-plus`，此时不会携带 OpenRouter 的线路参数。`INFERENCE_MODEL` 和 `INFERENCE_ENDPOINT` 仍可显式覆盖模型与地址。

Rich annotation 通过上述独立配置开启，启用后随视频会话持续采样；它不会作为第六个事件塞入 `features`，也不生成警报事件。现有界面仍选择五种事件，Rich annotation 的框与属性显示在同一标注视频中。

最新结构化结果位于 `GET /v1/rich-labels/latest`，运行状态位于 `GET /v1/status`。服务运行时新增的 rich annotation 请求独立处理，实验配置的请求超时为 90 秒；失败不会阻塞视频和事件能力。密钥通过部署面板更新时，rich annotation 也使用更新后的密钥。

`config.vlm-rich.yaml` 的起始设置：

- 最多每秒发起 1 次 rich annotation 请求，同一时刻只执行 1 次；正在等待处理的旧画面会被新画面替换。
- 每张模型输入图片最长边 1280 像素，输出最多 4096 tokens。
- `vlm_request_parameters` 增加 `response_format: {type: json_object}`，其余线路参数来自所选来源的配置。
- 当前提示词优先标注人物，再选择主要物品，最多返回 12 个对象，以限制输出耗时。
- 显示延迟 25 秒，框最长保留 25 秒，跨帧插值间隔最多 2 秒。

这些是实验起点。实际请求频率受 API 延迟限制；25 秒显示延迟也不保证覆盖所有慢请求。这份配置适合验证功能，不代表实时性能；原来的普通配置不受影响。

AIClub 配置保留 `enable_thinking: false`。此前网关的实际表现见下方实测记录。

## 少量样本检查

图片已经是最终视角，不再套用 `stream.frame_crop`。下面只发起 1 次请求：

```powershell
python tools/vlm_rich_smoke.py --image "D:\media\frame.jpg" --expected-endpoint https://openrouter.ai/api/v1/chat/completions --save-model-response
```

视频会先按实验配置裁剪，再依次取 0、2、4 秒附近的画面，最多请求 3 次：

```powershell
python tools/vlm_rich_smoke.py --video "D:\media\sample.mp4" --samples 3 --interval-seconds 2 --start-seconds 0 --expected-endpoint https://openrouter.ai/api/v1/chat/completions
```

可以通过 `--config` 选择修改后的实验配置，通过 `--output-dir` 为每次运行选择一个空目录。默认目录为 `output/vlm-rich-smoke`；已有结果不会被覆盖。

测试脚本保留原有的 AIClub 默认地址校验，因此调用 OpenRouter 时必须像上面一样显式传入 `--expected-endpoint`。目标地址不匹配时，会在读取图片和发起请求之前停止。调用目标写入本地 `metrics.json`，便于核对。切换到 AIClub 时使用 `AICLUB_API_KEY`，并去掉 OpenRouter 的地址参数。

输出包括：

- `snapshots.json`：每个成功样本的完整 rich-label 结果。
- `metrics.json`：所选来源、请求线路、返回的实际 provider、样本数、成功数、逐次请求耗时、平均耗时、标签计数和错误类型。
- `annotated-001.jpg` 等：使用现有 rich overlay 绘制的边框和标签，供肉眼对照。
- 使用 `--save-model-response` 时，额外保存 `response-001.json` 等模型回复文件，用于离线排查 JSON 格式与截断问题；只保存助手文本、结束原因、用量和返回的 provider，不保存请求头或密钥。这些文件包含图片的文字标注，保存在 Git 忽略的输出目录中。

工具依次请求、不做失败重试，首个请求失败后停止。实际样本数也可能因为视频结束而少于 `--samples`。它不记录密钥或原始服务错误。`--dry-run` 可只验证取帧及文件输出，不调用 API；这类输出不能用来判断标注质量。

这是功能与延迟的少量样本检查。它不评估实时吞吐、连续跟踪准确率或完整视频的标注质量，也不构成 Paddle 与 VLM 的正式基准测试。

## 2026-09-03 的 AIClub 实测

目标地址已精确校验为 AIClub，模型为 `qwen3.5-plus`。使用已有的 720×405 办公室测试截图，精简提示词后的单次请求成功，耗时 **20.29 秒**，返回 **4 个人、5 把办公椅、1 个电脑机箱、1 张桌子、1 个饮水机**，共 12 个对象。人物返回了年龄段、性别外观、Back/Side、袖长、衣服颜色等字段。原先较长的完整属性提示词两次在约 61 秒后调用失败。

成功回复的用量报告仍包含 123 个 reasoning tokens，因此不能认定网关已按预期关闭思考。人数与画面中的四人相符；被桌椅遮挡的下装属性仍需人工复核。单次耗时意味着当前无法达到每秒更新，且跨帧轨迹关联尚未通过真实视频验证。

本地结果保存在 `output/vlm-rich-aiclub-office-compact/`：`metrics.json`、`snapshots.json`、`response-001.json` 和标注图片。`annotated-readable.jpg` 使用同一份真实结果重新绘制：优先放置人物属性卡，错开放置物品标签，并用连线指向原框。空间不足时保留全部框和 JSON，省略低优先级物品文字。私有图片与输出均不提交到 Git。

## 2026-09-03 的 OpenRouter 复测

实验分支合入 main `91867ae`，接入 PR #7 的来源切换能力。使用同一张 720×405 办公室截图、相同 rich 提示词与图片设置，仅发起一次请求。模型为 `qwen/qwen3.5-35b-a3b`，请求固定 `provider.only: [deepinfra]` 且禁用 fallback，回复的 `provider` 确认为 **DeepInfra**。

请求成功，JSON 正常解析，共返回 **4 个人、5 把椅子、1 个电脑机箱、1 个柜子、1 个饮水机**。四人均有年龄段、性别外观、方向、袖长与上衣颜色；中间两人的遮挡下装属性被省略。回复以 `stop` 结束，共 1896 tokens（输入 821、输出 1075、reasoning 0）。单次耗时 **111.43 秒**，本轮没有调整延迟或再次选路；“此前最快”只说明历史配置来源，不代表这次速度结论。

本地结果保存在 `output/vlm-rich-openrouter-deepinfra/`：`annotated-001.jpg`、`snapshots.json`、`metrics.json` 和 `response-001.json`。这次人数与画面相符，人物身体被桌椅遮挡，部分框及服装细节仍需人工核对。一次静态图结果不代表连续视频跟踪质量。
