# 测试视频人数参考答案

`output-2.reference.json` 是对原始 `output 2.mp4` 的 AI 视觉标注初稿，不是人工金标准。视频 SHA256 与 20260925 延迟扫描日志一致。每秒取首个不早于目标时间的真实解码帧，保存真实 PTS、源帧索引和逐帧人数。

共 56 个样本；1–8 秒和 15 秒的 9 个样本因前景遮挡中央后排人物而待复核，默认排除。这些是离散采样点，不能直接外推成人数变化的精确连续区间。其他 47 个样本由 AI 判断明确，但仍需人工复核。

计数包含局部可辨认的人，包括后排只露出头部的人；完全不可见者不计。原有描边率只衡量显示覆盖情况，本参考新增人数比例得分、人数完全一致率、MAE 和少报／多报计数，不评估轮廓形状或人员身份。人数比例得分逐帧按 `max(0, 1 - |显示人数 - 参考人数| / max(参考人数, 1))` 计算，再对成功对齐的明确样本求平均；例如参考 4 人时显示 3 人或 5 人都得 75%，不再因相差一人直接得 0。完全一致率仍单独保留，比例得分不代表识别到了正确的人。

在工作树根目录运行：

```sh
.venv-harness-benchmark/bin/python -m tools.seg_pipeline.count_accuracy \
  --batch outputs/seg-delay-sweep/20260925T215518Z-10236 \
  --reference docs/validation/person-count/output-2.reference.json \
  --frames outputs/person-count-reference \
  --output outputs/person-count-accuracy
```

输出 `report.html`、`reference-review.html`、`accuracy.csv`、`accuracy.json`。复核页面可以修改计数、标明不确定帧、勾选已复核并下载 JSON；把下载文件传给 `--reference` 再运行即可更新结果。截图通过 data URL 嵌入复核页，可单独分享 HTML。图片位于本地输出目录，不进入 Git。

对齐使用 capture.video_pts 与 present.sequence，不用显示缓冲后的墙钟。最多允许 0.1 秒偏差。缺失匹配与 0 人分开处理，并报告匹配分母；当前 40 次运行均匹配全部 47 个明确样本。计数读取显示 subject 条目数，重复条目也计入。所有模型共用同一份独立标注；标注不来自模型预测。

## 用户复核版（2026-09-26）

`output-2.reviewed.json` 已由用户确认全部 56 个采样点，全部明确，无排除点。更新报告时将上述命令的 `--reference` 改为 `docs/validation/person-count/output-2.reviewed.json`。AI 初稿保留供追溯。

`output-2.reviewed.json` 是当前正式复核版，与 `output-2.reviewed-v3.json` 一致。此前的复核版本保留用于结果追溯，最新报告保存在本地 `outputs/person-count-accuracy/`。
