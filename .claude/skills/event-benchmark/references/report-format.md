# 报告列格式(Edwin Huang 2026-10 Slack 确认)

**每一次模型对比都用这组列,不要自己改列。** Edwin 的唯一改动:硬件和价格合并成一列。

| 列 | 中文 | 英文 | 来源 |
|---|---|---|---|
| 1 | 单次推理速度 | single inference speed | `latency_bench.py` 单测(中位数) |
| 2 | 12 路并发 P50 / P95 延迟 | P50 / P95 latency @ 12 concurrency | `latency_bench.py` 并发 |
| 3 | 各子任务成功率 | per-subtask success rate | `run_eval.py`(如 fight x/N、smoking x/N…) |
| 4 | 总体准确率 | overall accuracy | `run_eval.py` OVERALL |
| 5 | 推理硬件 + 价格 | hardware + price | API / L4 / A100 等 + 单价(合并,Edwin 要求) |

## 标准表模板

```
| 模型(类) | 单次推理速度 | P50 / P95 @12路 | 各子任务成功率 | 总体准确率 | 硬件 + 价格 |
|---|---|---|---|---|---|
| gemini (language) | 0.XXs | 0.XX / X.XXs | fight A/B · smoking C/D · fall E/F · weapon G/H | NN/100 = NN% | OpenRouter API · $X/Mtok |
| qwen   (language) | … | … | … | … | … |
| luna   (decision) | … | … | … | … | … |
```

## 填表规则

- **第 1、2 列(速度)** 来自 `latency_bench.py`,**不是** `run_eval.py`。测的是"一帧一个 predicate,client→OpenRouter→模型→返回"的墙钟(含网络+排队),不是整条产品链路。三个模型测同一帧、同一道 predicate,才可比。
- **第 3 列** 列出每个计分事件的 x/N(当前四类:fight / smoking / fall / weapon)。N 是该类有效题数(扣掉 error)。
- **第 4 列** = `run_eval.py` 的 OVERALL(TP+TN)/总数。
- **第 5 列** 当前三个候选都走 OpenRouter API,写清 provider + 单价(价格会变,报告前查当时价,不要用记忆里的旧价)。将来接自托管(L4 / A100)时写硬件型号 + 摊算成本。
- GT 未定稿时,在表下注明"临时:N 条草稿答案待审",别把临时数当定稿发出去。
- 无负例的事件(特异度算不了)在明细里标 `—(无负例)`,不要外推误报率。
- **会议里举例用的数字(如随口说的 smoking x/15)不是实测**,别和实测表混在一起。

## 举例形式 vs 实测

Slack 原文特别点出:各子任务成功率那列"会议中仅以举例形式提及,非实测结果"。所以报告里凡是实测的,要能追到 `run_eval.py` 的输出 JSON;凡是口头举例的数字,明确标为示例、不进实测表。
