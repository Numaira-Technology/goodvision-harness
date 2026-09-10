# VLM rich annotation 复测：2026-09-06

**结论：修复了缓存边界、模型路由覆盖和单路连接复用，但本轮没有实现“完整、正确、稳定在 2.5 秒内”的 rich annotation。** 不能仅凭 JSON 可解析或首字快就判定达标。当前实验仍不适合替代实时 Paddle 标注。

## 范围与环境

- 所有代码和最终诊断记录位于 `codex/vlm-rich-annotation` / `.worktrees/vlm-rich-annotation`，没有改动 main 原有修改。
- 沿用先前获准的办公室截图 `server-live-left.jpg`，720×405；画面中有 4 人。交替使用原图和水平镜像，检验变化像素下的缓存与标注。镜像只是一项受控测试，不等同于真实动态视频或新场景泛化。
- 更新密钥后，正常运行环境的 OpenRouter trace 返回 `loc=HK, colo=HKG`。在该环境发出 46 次调用尝试，取得 43 份完整推理响应；另有超时/429。完整响应包括不符合标注合同的结果。
- 早期过期密钥的 8 次 401 和隔离环境的 8 次连接失败不计入推理测速。早期阿里云直连的 12 份响应单独保留，不与香港 OpenRouter 数据混为一组；后续新增直连测试被自动审批拒绝，未继续该路径。

## 找到并修复的问题

1. **单路每帧新建 urllib 请求，没有复用连接。** 现在单路和并行共用可复用的异步 HTTP 客户端与事件循环，并记录首字、响应结束和 token 用量。整帧时限覆盖准备、请求和解析，关闭客户端可以取消正在等待的请求，不自动重试。最终实测首帧本地准备/创建客户端等约 0.9–1.2 秒，后续帧请求前后开销约 3–4 毫秒；这不是对旧实现网络延迟的严格 A/B 估计。
2. **独立 rich 模型的上游配置被 event profile 覆盖。** 显式选择不同 rich 模型时，它的 provider/采样参数现在可覆盖 event profile；event 自身保持原配置。切换渠道仍清除不兼容的旧渠道字段。
3. **紧凑模式未启用，且原提示词鼓励缩到 300–400 tokens。** 原紧凑提示词曾以 1.87/2.22 秒返回 4 人，却完全漏掉物品、缺少人物属性。现在启用紧凑结构、严格 schema，要求全部 13 项人物字段；不可见值填 null。明确要求边缘人物和主要物品，去掉人为的短输出目标。
4. **缓存断点位置不适合变化帧。** 见下文。实现了可选 `vlm_prompt_cache`，没有在产品提示词中填充文本。

## 缓存实测

使用 Qwen 3.6 Flash / Alibaba，并固定上游、不允许 fallback。OpenRouter response caching 已在诊断请求中关闭；下面测的是模型前缀缓存。

| 请求结构 | 服务端报告 | 说明 |
|---|---|---|
| text+image 放同一条 user 消息，text 上加 cache_control | 换图 0 hit；再次原图命中 2,509 tokens | 这条线路把图片一起纳入该消息的断点，重复同一图片会造成乐观结果 |
| 固定 text 放 system 消息 | 0 cache read / write | 本次该线路没有报告有效显式缓存 |
| 独立 user 指令 + 固定 assistant `Ready.` + 新 user 图片 | 镜像图也命中 2,261 tokens | 诊断中用重复前缀越过门槛；证明变化帧可复用文本前缀 |
| 最终真实客户端，同样独立消息，普通长度指令+schema | 不同帧均命中 1,320 / 1,630 tokens（约81%） | 无文本填充；思考 tokens 为 0 |

最后一行仍需 **6.86 / 4.88 / 5.73 秒**，且三次均只识别到 3 人。不能把与关闭缓存时的速度差完全归因于缓存，因为消息布局同时改变了召回率和输出长度。缓存开关因此默认关闭，供后续测试使用。

缓存主要节省重复输入的计算与费用，无法省略逐 token 输出。阿里云文档说明重复前缀门槛及变化图片时应将文字放前面；OpenRouter 文档也要求检查 `cached_tokens` / `cache_write_tokens`，不能只看是否传了参数。[阿里云缓存说明](https://www.alibabacloud.com/help/en/model-studio/context-cache)、[OpenRouter 缓存说明](https://openrouter.ai/docs/guides/best-practices/prompt-caching)。

## 最终客户端实测

每组都是原图、镜像、原图，复用同一个客户端；13项字段由 schema 要求，null 在公共属性字典中仍按原合同省略。首帧包含客户端初始化开销。

| 模型 / 配置 | 完整调用耗时（秒） | 人 / 物品数 | 质量结论 |
|---|---|---|---|
| Qwen 3.6 Flash，缓存关闭 | 7.75 / 4.90 / 5.68 | 4/6、3/4、4/6 | 原图找全4人；镜像漏边缘人物；方向和衣着仍有错误 |
| Qwen 3.6 Flash，缓存开启 | 6.86 / 4.88 / 5.73 | 3/5、3/5、3/5 | 确认81%缓存命中，仍漏人、未达2.5秒 |
| Qwen3 VL 8B Instruct，缓存关闭 | 5.52 / 3.16 / 4.45 | 4/5、3/9、4/8 | 镜像漏人且超过提示词要求的8物品；衣着/方向错误更明显 |

Qwen3 VL 8B 最后一次首字约 **0.35秒**，总请求仍约 **4.44秒**，输出448 tokens。后面的约4.09秒是答案生成/流式返回阶段，缓存不能消除它。该系列 OpenRouter 响应报告的隐藏思考 tokens 均为0。

最激进的10字符属性压缩格式曾达到 **2.08–2.48秒**，但8B漏掉物品、改变行结构，3.6 Flash出现属性码错位，6次均不符合合同。没有接入产品。原35B / DeepInfra两次分别是25秒超时和429，本轮不能复现其过去速度，也不能把它作为精确的同日提速倍数基线。

这些是一个办公室画面的有限样本，没有宣称P95保证、跨场景准确率或真实视频连续追踪已通过。

## 保存的实验配置

`config.vlm-rich.yaml`：独立 rich 使用 Qwen 3.6 Flash / Alibaba，单路、紧凑完整属性、严格schema、1,536输出token上限、720像素、8秒请求截止。缓存默认关闭。视频缓冲由25秒改为10秒以配合实验截止时间；**10秒缓冲是显示延迟，不是模型提速**。event 模型和路由未更改。

所有方案仍保留原公共 RichLabelSnapshot / overlay 接口；模型不生成本地track ID和帧时间。

## 验证与文件

- 最终相关回归 **296/296通过**：VLM、流式API、配置、service。
- 较早完整测试运行：1003通过、7跳过、11失败。11个GPU smoke测试被pytest命令行参数污染；在清空参数的对照中全部通过。记录保留，没有把该次全量运行描述为全绿。
- `tmp/vlm-latency-20260906/native-no-cache.jsonl`、`native-verified.jsonl`、`native-8b.jsonl` 是上表依据。
- `cache-short.jsonl`、`cache-long.jsonl`、`cache-system.jsonl`、`cache-message.jsonl` 保留缓存对照。
- `native-final.jsonl` 是早期诊断记录，其中3次模型已成功但脚本错误调用了 `to_dict`；后续改为 `as_dict` 并重新测量，最终表不使用这些错误记录。
- `native-no-cache-overlay.jpg`、`native-8b-overlay.jpg` 通过现有产品 renderer 绘制，已人工查看，可见人物框和属性仍有误差。
- `tools/vlm_native_probe.py` 重跑最终客户端；`tools/vlm_latency_probe.py` 与 `tools/vlm_packed_probe.py` 重跑诊断方案。使用自己的已授权图片和凭据文件，不将密钥写入记录。
