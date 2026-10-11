# Qwen3.5-35B / Alibaba 十二路三模式测试

固定入口：`python -m tools.vlm_benchmark.qwen35b_alibaba`。
默认离线模拟；必须显式使用 `--real` 才调用付费 API。

- OpenRouter 模型 `qwen/qwen3.5-35b-a3b`，关闭 thinking，temperature=0。
- 固定 Alibaba 的该模型使用 JSON Object 输出：2026-10-10 实测文字请求可接受
  JSON Schema，但包含视频帧时返回 HTTP 400（schema is None）。六类事件提示词、
  true/false/null、grounding lean-v1 字段及本地质量校验保持一致；不切换模型或供应商。
- 路由仅允许 `alibaba`，`allow_fallbacks=false`，不提供备用模型。
  成功结果必须有匹配的模型名和 Alibaba 返回标签；否则保留诊断并停止后续轮次。
  这是接口返回信息核对，不是独立验证服务端模型权重。
- 可显式添加 `--allow-provider-fallback`：改为 `order=["alibaba"]`、
  `allow_fallbacks=true`，移除 `only`，Alibaba 不可用时允许同一 35B 模型的其他
  供应商接替。仍核对每个返回的模型名称，不提供备用模型；JSON Object 和本地
  六类质量校验保持不变。每轮 `route_audit` 分别记录所有响应标签和完整结果的
  供应商计数，不能将这类报告称为“全部 Alibaba”。接口未报告供应商的完整结果
  仍使路由核对失败；未列入已知公共名称的供应商汇总为 `other`。
  此选项不改变负载、截止时间、录屏或最终 HTTP 429 的停止保护。
- 2026-10-10 JSON Object 预检发现 grounding 会返回空事件数组和错误人物字段。
  针对该 35B 路由在输出格式段补充六类 JSON 结构示例，并声明示例不是答案；
  判断规则、取帧和严格解析均不变。此修正增加提示词长度，manifest 显式记录
  `grounding_json_object_shape_example`，不能把修正前后差异全部归因于路由。
- events、grounding（lean-v1）、parallel，各十二路、两轮。
  第一轮正序，第二轮反序；parallel 已包含两条分支。
- 正式输入每轮 55.798021 秒；单路接口预检和预热各 10 秒。
  正式阶段每路 69 个窗口，各分支共 828 个；六轮请求计划上限 6624，
  另有预检和预热，准确预算以 `--plan-only` 输出为准。
- 每路共享并发 4、全局 48、每分支本地并发 4、请求上限 40/s，源时钟错峰。
  观察频率 1.25 Hz；排队超时 1 秒；events 请求超时 5 秒、grounding 20 秒。
- 六类事件、观察帧、720/JPEG90 输入和定位质量校验保持现有规则。
  每轮检查输入时序、排空、记录完整性和残留 admission 名额；
  HTTP 401/402/403/429 停止后续轮次。

## 监看与录屏

每路只有一个原速连续视频，预览目标 15 FPS。三个模式均无右侧证据帧、
无定位框、无帧间跟踪；浏览器不调用跟踪接口，也不下载用于右侧显示的证据图。
grounding 仍执行定位、保存原始框和质量校验，不能把隐藏框误解为关闭定位推理。

六类事件状态、证据时间、返回耗时与统计仍显示。未增加视频播放缓冲，
所以状态可能对应几秒前的画面。显示帧率与模型 1.25 Hz 请求频率不同。
录屏维持 1920×1080 的浏览器实际录制与现有 25 FPS 编码设置；
这不代表原视频有 25 个不同画面/秒。`recording.json` 记录逐路实际播放帧率、
丢帧和最长停顿，不能仅凭文件帧率声称视频流畅。
单画面录屏记录每一路的实测平均播放帧率（至少 1 秒播放测量），目标仍为 15 FPS。
低帧率或测量数据缺失只作记录，不增加中止条件，不改变已有录屏完整性检查或测试流程。
报告显示实际帧率范围；文件完整不代表严格达到 15 FPS。
云端先做离线录制检查，再发起付费模型请求；预检查失败时不会调用付费模型。

## 启动

只检查本地方案（无网络、无密钥读取、无付费调用）：

```bash
python -m tools.vlm_benchmark.qwen35b_alibaba --plan-only
```

只检查云端方案：

```bash
python -m tools.vlm_benchmark.qwen35b_alibaba --cloud --plan-only --real
```

云端运行时提供 `--host`、`--port`、`--identity` 和 `--real`；
默认视频为 `~/Downloads/output 3.mp4`，可用 `--video` 指定。
沿用现有云端上传、隐藏输入 API key、后台作业、恢复和结果下载流程。
Harness 与录屏在云实例执行，模型由 OpenRouter 的 Alibaba 服务执行。
本入口不允许覆盖模型、模式、并发和显示布局；供应商策略只允许上述显式选择。

本地真实运行使用 `--real`，从 `OPENROUTER_API_KEY` 或 `--env-file` 读取密钥。
不要将密钥写入命令、报告或 Git。可用 `--output` 指定新的本地结果目录。

## 交付与解读

三模式两轮共六个 `recording/viewport.mp4`，附 `final.png`、`recording.json`；
总报告 `report.html`、`summary.json`、`manifest.json`、逐请求日志和模型/供应商核对。
保留各分支完整率、缺失/失败、不确定判断和排队/API/发布延迟。
完整输出率不等于事件准确率；与 Astra 比较仍是自动参考一致率，不能当成人工真值。
本轮换了模型和显示布局，不能把与旧报告的差异全部归因于供应商。
