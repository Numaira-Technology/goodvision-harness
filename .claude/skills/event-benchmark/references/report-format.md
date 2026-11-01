# Benchmark report format

Keep these five measurement columns, plus model identity:

| Model / protocol / variant | Single inference | P50 / P95 @12 | Per-subtask success | Overall accuracy | Hardware + price |
|---|---|---|---|---|---|
| qwen / v2 / fixed | not run | not run | not run | not run | not measured |
| gemini / v2 / fixed | not run | not run | not run | not run | not measured |
| luna / v2 / fixed 0.5 | not run | not run | not run | not run | not measured |

Chinese labels: 单次推理速度 · 12路并发 P50/P95 延迟 · 各子任务成功率 ·
总体准确率 · 推理硬件+价格. Placeholders are not measurements.

- Speed comes from `scripts/latency_bench.py`, not accuracy-run wall time. Same
  normalized JPEG hash and question across candidates; one request per sample,
  no retry hiding. Report success/attempt counts, failures and whether the run is
  complete. Incomplete success-only percentiles are diagnostic, not ranked headline
  speed. This measures client-to-provider round trips, not full product latency.
- v2 accuracy: per event and overall `correct/valid`, `errors/attempted`, coverage.
  Never silently rank different successful subsets. `valid=0` means unavailable.
  No-negative events have unavailable specificity. Unknown is not false.
- Separate other runners or label their non-equivalent cadence, cohort, denominator
  and timing. Do not put sequential probe latency in @12 cells.
- Price: record date/source, currency and unit. API input/output prices ($/M tokens)
  and total GPU run cost ($) are distinct quantities; do not rank them as one cost
  metric or invent a blended token price. Unknown price stays blank.
- Attach dataset/manifest/result hashes, source commit, model revision/config,
  spec/parser version and per-block run dates. Fixed and tuned rows need distinct
  names and threshold-artifact hashes; tuning reports validation counts too.
- Historical/provisional evidence is explicitly labeled and separated from reviewed
  v2 comparisons. An existing draft manifest blocks an official v2 run.

The running Sheet has its own pivoted A:W layout, verified at publication time by
`bench-sheet-contribute`. It omits the single-inference column; keep that value in
the report/artifact, without shifting the Sheet's speed columns.
