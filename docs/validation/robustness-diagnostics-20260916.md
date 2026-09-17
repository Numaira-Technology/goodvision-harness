# 鲁棒性诊断与连续回放（2026-09-16）

目标：机器人实机表现未知时，先能分清一次“没报警”到底发生在哪一步——动作没被抽到、请求被丢弃、
请求失败（鉴权 / 超时 / 解析等），还是模型看到了但没认出来。本轮改了诊断、采样和测试工具，
并把行为事件送模图片固定为 360×202（第 5 节）；采样率、并发上限（当时为 6）、提示词和模型线路
没有改动。

后续配置更新：默认并发上限从 6 调整为 10，主配置、交付配置、VLM rich 配置、包内配置和代码默认值保持一致。
下文的并发 6 / 10 测试数据保留原始测试条件；此次调整未重新进行实机或付费模型测试。

## 1. 事件采样与人脸采样解耦（缺陷修复）

此前在多能力会话中勾选陌生人后，服务会把事件分支的采样率改成 `face.sample_fps`，并把滑动窗口
强制改成 1 帧。因此调整 `video.fps` 或 `sliding_window` 在勾选陌生人时不生效。

现在混合会话（行为事件 + 陌生人）只对源画面采样一次，采样率取两者较快者；随后各分支按自己的
节奏选帧、建窗口：

- 行为事件：`video.fps` + `sliding_window.size_frames / stride_frames`；
- 陌生人：`face.sample_fps`，始终单帧。

默认配置两者都是 2.5 FPS、单帧窗口，行为与之前一致。只选陌生人或只选行为事件时保持原逻辑。
浏览器摄像头推帧速率改用两者较快者（`/v1/ui/info` 的 `multi.sample_fps`），同时新增
`multi.safety_sample_fps`、`multi.safety_window`、`multi.face_sample_fps`。

## 2. 分析诊断

新增 `GET /v1/diagnostics`，控制台“分析诊断”面板每 2 秒刷新。会话停止后保留到下一次启动。
按分支（行为事件 / 陌生人）给出：

| 指标 | 回答的问题 |
| --- | --- |
| 最近送去分析的图片（`/v1/diagnostics/model-inputs/{id}/frames/{i}.jpg`）、实际尺寸、字节数、源帧时间、源画面尺寸、结果 | 模型到底看到了什么？ |
| `analyzed_per_second`（最近 10 秒）、`longest_unanalyzed_gap_seconds`、`current_unanalyzed_gap_seconds` | 快动作是否可能整段没被分析？ |
| `gap_scope`、`gap_history_size` | 最长间隔是整段会话（`session`）还是只统计最近 512 次结果（`recent`）？ |
| `failures`：auth / rate_limited / timeout / connection / server_error / http_error / bad_response / parse / input / internal / unknown，以及 `last_failure`（含 HTTP 状态码） | 是系统错误还是识别漏检？ |
| `latency_seconds.capture_to_result`、`capture_to_submit`、`model`（P50 / P95 / 最大） | 报警来不来得及？ |

诊断只保存有界的数字和最近 8 次送模图片，不保存供应商原文。分析时间只保留最近 512 个；长会话超出后，
最长间隔改为只在这 512 个结果之间计算，并标记 `gap_scope: recent`，面板显示“（最近 N 次）”，
不会把已淘汰的记录误算成一段很长的空白。面板上“当前未分析间隔”的告警阈值会再加上模型延迟 P95，
避免请求正常在途时误报。

## 3. 连续视频回放工具

`tools/replay_stress.py` 让本地视频按真实时间进入完整检测流程（采样、分支窗口、真实提示词构建与
解析、结果整理、诊断），只把远程分析服务换成模拟器：

- 模拟器是“神谕”：收到的帧落在标注区间内就报阳性，所以漏报只可能来自没抽到、被丢弃、请求失败或
  结果没发布，与模型认不认得无关；
- 可注入延迟分布、慢请求（造成乱序）、超时、401、不可解析回复和随机误报；
- `--speed N` 把时间轴压缩 N 倍，用来模拟快动作（`video.playback_speed` 只改变播放节奏，
  采样仍按源时间，不能用来模拟快动作）；
- `--real` 改为调用已配置的真实服务（会产生费用，陌生人分支仍为模拟），未识别的事件标记为
  `not_recognized`。事件灵敏度与线上服务默认一致（吸烟为 `high`），可用
  `--event-sensitivity smoking=standard` 逐项覆盖，报告里记录实际使用的 `event_sensitivities`；
- 默认和服务一样使用固定送模画布（第 5 节）；`--model-canvas none` 改为按 `--safety-side`
  原样送图，`--model-canvas 320x180` 可试其他画布，报告的 `settings.model_canvas` 记录实际值；
- `--real` 有请求上限 `--max-requests`（默认 300）：启动前按“每个采样窗口一次请求”估算上界，
  超过就直接退出、不创建客户端；运行中也按上限硬截断，超出的窗口记为 `budget_exhausted`
  （计入请求失败，不当作阴性），报告里的 `paid_requests` 是实际发出的请求数。

每个标注事件输出 `detected / not_sampled / dropped / request_failed / not_recognized`，以及首次报警
延迟、误报数、乱序结果数和整段诊断。

```bash
# 单段视频，两个标注事件，正常速度与 3 倍速对比，带故障注入
python tools/replay_stress.py clip.mp4 --event smoking:2.2-2.8 --event fight:4.0-4.3 \
  --speed 1 --speed 3 --latency-ms 700 --slow-rate 0.2 --timeout-rate 0.1 \
  --safety-side 360 --output tmp/replay/smoke.json

# 负例素材清单（无 events 字段即全阴性），五项能力全开
python tools/replay_stress.py --manifest tmp/robustness-media/negatives/robot_360x202/manifest.json \
  --features fight,smoking,fall,weapon,stranger --safety-side 360 --safety-quality 90 \
  --output tmp/replay/negatives.json
```

## 4. 机器人画幅素材（360×202）

素材都来自 Pexels 的摆拍库存视频，放在被 Git 忽略的 `tmp/robustness-media/` 下。清单里的路径
相对仓库根目录，请在仓库根目录运行命令。两份清单都能同时被 `replay_stress.py` 和
`low_resolution_model_eval.py` 使用；后者只读 `samples`，会忽略 `events` 和 `ignore`。

| 目录 | 内容 |
| --- | --- |
| `negatives/robot_360x202/` | 喝水、吃东西、摸嘴（打哈欠、咳嗽）各 2 段，全部为阴性；`eat_02` 是手指夹细长饼干棒，最接近香烟 |
| `positives/robot_360x202/` | 抽烟、打架、持械各 2 段，带事件区间 `events` 和过渡帧 `ignore` 区间 |

`ignore` 区间用于动作起止的过渡帧（例如香烟刚举起、枪刚拔出），落在这些区间里的阳性不计为误报。
`fight_02` 的 `key_actions` 标出了 3.25–4.5 秒的快速踢击，可以单独用来测快动作：

```bash
python tools/replay_stress.py tmp/robustness-media/positives/robot_360x202/fight_02_pexels7777758_360x202.mp4 \
  --event fight:3.25-4.5 --speed 1 --speed 2 --speed 4 --speed 6 --safety-side 360
```

模拟服务、默认 2.5 FPS 采样下的结果：踢击在 1 倍速（1.25 秒）和 2 倍速（0.63 秒）时能被抽到并报警；
4 倍速（0.31 秒）和 6 倍速（0.21 秒）时一帧都没被抽到（`not_sampled`）。六段正例在 1 倍速下全部
报警，六段负例没有误报；这两项在模拟服务下是预期结果，只证明素材和标注能被流程正确使用。

按机器人画幅比较单帧图片参数（真实服务，会产生费用；长边 `0` 表示保持 360×202 原尺寸）。
这个工具默认按 `--sizes` / `--quality` 原样送图，不套用服务的固定画布；要模拟服务，加
`--model-canvas 360x202`：

```bash
python tools/low_resolution_model_eval.py \
  --manifest tmp/robustness-media/positives/robot_360x202/manifest.json \
  --sizes 0,320 --quality 75 --workers 1 --max-requests 60 \
  --output tmp/eval/positives-q75.json
python tools/low_resolution_model_eval.py \
  --manifest tmp/robustness-media/positives/robot_360x202/manifest.json \
  --sizes 0 --quality 90 --workers 1 --max-requests 30 \
  --output tmp/eval/positives-q90.json
# 把 positives 换成 negatives 再各跑一次，看误报
```

## 5. 固定送模画布（360×202）

`video.model_image_size: [360, 202]`：行为事件图片先按 `video.max_image_side`（现为 360）缩放，
再等比放进 360×202 画布，不足处居中补黑边；模型返回的框换算回原始帧坐标。16:9 输入缩放后正好是
360×202，直接发送，不再解码或重新压缩。三份配置（根目录、包内默认、交付）保持一致。

- 图片无法解码时不发请求，诊断记为 `input`；
- 某个事件的框全部落在黑边里时，只把这个事件记为未完成（`missing_event_labels`，既不算阳性也不算
  阴性），同一帧其他事件照常发布；单事件结果或所有事件都这样时整帧作废，诊断记为 `parse`；
- 黑边里的人物框和辅助框直接丢弃。

同一批 6 段正例的真实服务回放（2.5 FPS、单帧窗口、长边 360、Q75，均在画布改动之前）：

| 运行 | 检出 | 丢弃窗口 | 解析失败 | 误报 | 请求数 |
| --- | --- | --- | --- | --- | --- |
| 当前线路，并发 6 | 6/6 | 43 | 0 | 0 | 101 |
| 当前线路，并发 10 | 6/6 | 3 | 0 | 1 | 141 |
| qwen8b，并发 6 | 5/6 | 15 | 51 | 10 | 129 |
| qwen8b，并发 10 | 4/6 | 0 | 64 | 14 | 144 |

并发 10 把最长未分析间隔从 2.0 秒降到 0.4–0.8 秒，但请求数多约四成，且同一上限也限制本地人脸
推理，默认仍保持 6。固定画布本身还没有用真实服务对比过。

注意：`low_resolution_model_eval.py` 没有灵敏度参数，用的是严格的吸烟证据规则（相当于
`smoking=standard`），而线上服务默认吸烟为 `high`。它适合在同一规则下比较图片尺寸和质量；
要贴近线上吸烟命中率，用 `replay_stress.py --real`（默认即 `high`）。

## 验证

- 全量回归：**2159 passed、6 skipped**（基线 2069 passed、6 skipped；新增 90 项）。
  浏览器控制台 JS 测试 6 个文件通过（新增诊断面板告警阈值测试）。全仓 Ruff 发现数与基线相同（89），`git diff --check` 通过。
- 新增测试覆盖：分支窗口与原滑动窗口语义一致、慢分支抗到帧抖动、服务端混合会话不再覆盖事件
  采样、实时管线中行为 3 帧窗口与陌生人半速同时生效；401 / 超时 / 解析失败分别计数、实际送模
  图片为 360×202（源 640×360）、HTTP 端点与会话结束后保留；模拟回复通过真实解析器、压缩时间轴下
  短事件变为 `not_sampled`、真实模式的未识别标记；长会话淘汰历史后最长间隔仍真实、面板当前间隔
  告警计入在途延迟；`--real` 默认吸烟为 `high`、可覆盖为 `standard`，请求上限在创建客户端前
  拦截、运行中硬截断；固定画布的补边、框还原、黑边内框按事件处理、已是画布尺寸时不重复解码，
  评测工具和回放工具的实际送模尺寸与 `--model-canvas` 一致。
- 以 360×202 机器人画幅素材做的模拟回放：3 倍速下 0.3 秒的动作压缩到 0.1 秒，没有被 2.5 FPS
  采样抽到；并发 1、单次 1.2 秒、5 FPS 三帧窗口时，43 个事件窗口中 37 个被丢弃，最长未分析间隔
  2.0 秒。这些是流程层面的结论，不代表真实模型的识别率。

## 边界

- 模拟器不评价识别准确率。模型能否认出吸烟，仍需带标注素材配合 `--real` 或
  `tools/low_resolution_model_eval.py` 另测。
- 固定 360×202 画布尚未用真实服务与原 320 长边对比；应先用同一批正负例
  （`tmp/eval/benchmark-12clips-360x202/manifest.json`）各跑一次再定稿。
- 并发 6 在单次约 3.5 秒的延迟下会丢弃约三分之一的窗口；是否提高需先确认费用和服务商并发限制。
