# VLM Rich Annotation 实验

本分支把 rich annotation 做成独立的第六项能力：在原有五类事件能力之外，额外调用一次 VLM，识别画面中的人和物品，返回边框和外观属性。它复用现有 rich-label 数据结构、画面叠加及事件框融合路径；原有事件的请求和结果格式保持兼容。

`rich_labeling.backend: vlm` 直接复用 `aiclub` / `inference_service` 的推理 API 地址与密钥。`rich_labeling.model` 未设置时使用 `inference.model`。这条 rich annotation 路径不需要启动 Paddle worker，也不需要安装 Paddle 的本地模型依赖。需要切回本地方案时可选择 `backend: paddle`。

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

在此 worktree 根目录、现有 Harness Python 环境中运行。密钥继续使用已有的 `INFERENCE_API_KEY`、配置指定的环境变量或 `.env`，不要写进 YAML。

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
$env:HARNESS_CONFIG = (Join-Path (Get-Location) "config.vlm-rich.yaml")
$env:HARNESS_PORT = "7901"
python -m goodvision_harness.service
```

打开 [实验服务](http://127.0.0.1:7901)，按现有方式启动视频流或摄像头会话。实验配置保留原配置的视频裁剪、事件提示词和模型设置。默认 `stream.frame_crop: left_half` 适用于左右并排的机器人画面；普通单目视频请改为 `none`。

Rich annotation 通过上述独立配置开启，启用后随视频会话持续采样；它不会作为第六个事件塞入 `features`，也不生成警报事件。现有界面仍选择五种事件，Rich annotation 的框与属性显示在同一标注视频中。

最新结构化结果位于 `GET /v1/rich-labels/latest`，运行状态位于 `GET /v1/status`。服务运行时新增的 rich annotation 请求独立处理，实验配置的请求超时为 90 秒；失败不会阻塞视频和事件能力。密钥通过部署面板更新时，rich annotation 也使用更新后的密钥。

`config.vlm-rich.yaml` 的起始设置：

- 最多每秒发起 1 次 rich annotation 请求，同一时刻只执行 1 次；正在等待处理的旧画面会被新画面替换。
- 每张模型输入图片最长边 1280 像素，输出最多 4096 tokens。
- `vlm_request_parameters` 使用 `enable_thinking: false` 和 `response_format: {type: json_object}`。
- 当前提示词优先标注人物，再选择主要物品，最多返回 12 个对象，以限制输出耗时。
- 显示延迟 25 秒，框最长保留 25 秒，跨帧插值间隔最多 2 秒。

这些是实验起点。实际请求频率受 API 延迟限制；25 秒显示延迟也不保证覆盖所有慢请求。这份配置适合验证功能，不代表实时性能；原来的普通配置不受影响。

Qwen3.5-Plus 默认开启思考模式；直接 HTTP 请求中，关闭开关位于 JSON 顶层。阿里云为非思考模式提供 JSON Object 输出支持。本配置采用这些参数，AIClub 网关的实际表现以测试为准。[思考模式文档](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)、[结构化输出文档](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)。

## 少量样本检查

图片已经是最终视角，不再套用 `stream.frame_crop`。下面只发起 1 次请求：

```powershell
python tools/vlm_rich_smoke.py --image "D:\media\frame.jpg" --save-model-response
```

视频会先按实验配置裁剪，再依次取 0、2、4 秒附近的画面，最多请求 3 次：

```powershell
python tools/vlm_rich_smoke.py --video "D:\media\sample.mp4" --samples 3 --interval-seconds 2 --start-seconds 0
```

可以通过 `--config` 选择修改后的实验配置，通过 `--output-dir` 为每次运行选择一个空目录。默认目录为 `output/vlm-rich-smoke`；已有结果不会被覆盖。

测试脚本默认要求目标地址精确匹配 AIClub；配置被切换到其他平台时，会在读取图片和发起请求之前停止。调用目标写入本地 `metrics.json`，便于核对。请配置 `AICLUB_API_KEY`；不会用其他平台的密钥替代它。

输出包括：

- `snapshots.json`：每个成功样本的完整 rich-label 结果。
- `metrics.json`：样本数、成功数、逐次请求耗时、平均耗时、标签计数和错误类型。
- `annotated-001.jpg` 等：使用现有 rich overlay 绘制的边框和标签，供肉眼对照。
- 使用 `--save-model-response` 时，额外保存 `response-001.json` 等模型回复文件，用于离线排查 JSON 格式与截断问题；只保存助手文本、结束原因和用量，不保存请求头或密钥。这些文件包含图片的文字标注，保存在 Git 忽略的输出目录中。

工具依次请求、不做失败重试，首个请求失败后停止。实际样本数也可能因为视频结束而少于 `--samples`。它不记录密钥或原始服务错误。`--dry-run` 可只验证取帧及文件输出，不调用 API；这类输出不能用来判断标注质量。

这是功能与延迟的少量样本检查。它不评估实时吞吐、连续跟踪准确率或完整视频的标注质量，也不构成 Paddle 与 VLM 的正式基准测试。

## 2026-09-03 的 AIClub 实测

目标地址已精确校验为 AIClub，模型为 `qwen3.5-plus`。使用已有的 720×405 办公室测试截图，精简提示词后的单次请求成功，耗时 **20.29 秒**，返回 **4 个人、5 把办公椅、1 个电脑机箱、1 张桌子、1 个饮水机**，共 12 个对象。人物返回了年龄段、性别外观、Back/Side、袖长、衣服颜色等字段。原先较长的完整属性提示词两次在约 61 秒后调用失败。

成功回复的用量报告仍包含 123 个 reasoning tokens，因此不能认定网关已按预期关闭思考。人数与画面中的四人相符；被桌椅遮挡的下装属性仍需人工复核。单次耗时意味着当前无法达到每秒更新，且跨帧轨迹关联尚未通过真实视频验证。

本地结果保存在 `output/vlm-rich-aiclub-office-compact/`：`metrics.json`、`snapshots.json`、`response-001.json` 和标注图片。`annotated-readable.jpg` 使用同一份真实结果重新绘制：优先放置人物属性卡，错开放置物品标签，并用连线指向原框。空间不足时保留全部框和 JSON，省略低优先级物品文字。私有图片与输出均不提交到 Git。
