# Report column format (confirmed by Edwin Huang, Slack 2026-10)

Use this set of columns for every model comparison; don't invent your own. Edwin's one change: merge
hardware and price into a single column.

| Col | Name (EN) | Name (CN, as used in CN meeting tables) | Source |
|---|---|---|---|
| 1 | single inference speed | 单次推理速度 | `latency_bench.py` single (median) |
| 2 | P50 / P95 latency @ 12 concurrency | 12 路并发 P50 / P95 延迟 | `latency_bench.py` concurrent |
| 3 | per-subtask success rate | 各子任务成功率 | `run_eval.py` (e.g. fight x/N, smoking x/N…) |
| 4 | overall accuracy | 总体准确率 | `run_eval.py` OVERALL |
| 5 | hardware + price | 推理硬件 + 价格 | API / L4 / A100 + unit price (merged, per Edwin) |

## Standard template

```
| model (class) | single inf speed | P50 / P95 @12 | per-subtask success | overall acc | hardware + price |
|---|---|---|---|---|---|
| gemini (language) | 0.XXs | 0.XX / X.XXs | fight A/B · smoking C/D · fall E/F · weapon G/H | NN/100 = NN% | OpenRouter API · $X/Mtok |
| qwen   (language) | … | … | … | … | … |
| luna   (decision) | … | … | … | … | … |
```

(Chinese headers for CN meetings: 模型(类) · 单次推理速度 · P50 / P95 @12路 · 各子任务成功率 ·
总体准确率 · 硬件 + 价格.)

## Fill rules

- **Columns 1–2 (speed)** come from `latency_bench.py`, not `run_eval.py`. They measure "one frame, one
  predicate, client→OpenRouter→model→back" wall-clock (network + queueing included), not the full product
  pipeline. The three models must hit the same frame and the same predicate to be comparable.
- **Column 3** lists x/N for each scored event (currently fight / smoking / fall / weapon). N is the valid
  question count for that class (errors removed).
- **Column 4** = `run_eval.py` OVERALL, (TP+TN)/total.
- **Column 5**: the three current candidates all run on the OpenRouter API, so give provider + unit price
  (prices change — look up the current one at report time, don't use a remembered price). When something
  moves to self-hosting (L4 / A100), give the hardware model + amortized cost.
- When GT isn't final, note "provisional: N draft answers under review" below the table; don't ship
  provisional numbers as final.
- For an event with no negatives (specificity not computable), mark `— (no negatives)` in the detail;
  don't extrapolate a false-positive rate.
- **Numbers used as examples in a meeting (e.g. an off-hand smoking x/15) are not measured** — keep them
  out of the measured table.

## Illustrative vs measured

The Slack note called this out specifically: the per-subtask success-rate column was "mentioned only as an
example in the meeting, not a measured result." So in a report, every measured cell must trace back to a
`run_eval.py` output JSON; any number quoted verbally as an example is labeled as such and stays out of the
measured table.
