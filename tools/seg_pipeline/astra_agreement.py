"""Score saved Test 3 display logs against the untouched AI count reference."""

import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from tools.seg_pipeline.count_accuracy import evaluate_trial, validate_reference


class ReportLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href.endswith("/report.html"):
                self.links.append(href)


def linked_zero_delay_runs(combined_report):
    parser = ReportLinks()
    parser.feed(combined_report.read_text())
    root = combined_report.parent.parent.resolve()
    runs = set()
    for link in parser.links:
        report = (combined_report.parent / link).resolve()
        if not report.is_relative_to(root) or not report.is_file():
            raise ValueError(f"Invalid combined-report link: {link}")
        runs.add(report.parent)
    if not runs:
        raise ValueError("Combined report contains no model runs")
    return sorted(runs)


def discover_trials(combined_report, sweep):
    sources = [("0-second batch", run) for run in linked_zero_delay_runs(combined_report)]
    sources.extend(("delay sweep", p.parent) for p in sorted(sweep.glob("*/delay-*/summary.json")))
    if not any(source == "delay sweep" for source, _ in sources):
        raise ValueError("Delay sweep contains no model runs")
    for source, run in sources:
        summary = json.loads((run / "summary.json").read_text())
        for trial in summary["trials"]:
            if not trial["valid"]:
                raise ValueError(f"Cannot score invalid trial: {run}/repeat-{trial['repeat']}")
            directory = run / f"repeat-{trial['repeat']}"
            yield source, directory, trial


def count_f1(samples):
    """Micro F1 of counts; min(counts) is agreement, not identity matching."""
    matched = [sample for sample in samples if sample["predicted"] is not None]
    details = [
        {
            "timestamp_seconds": sample["timestamp_seconds"],
            "astra_count": sample["expected"],
            "displayed_count": sample["predicted"],
            "count_overlap": min(sample["expected"], sample["predicted"])
            if sample["predicted"] is not None else None,
            "missed_people": max(sample["expected"] - sample["predicted"], 0)
            if sample["predicted"] is not None else None,
            "extra_people": max(sample["predicted"] - sample["expected"], 0)
            if sample["predicted"] is not None else None,
            "difference": sample["predicted"] - sample["expected"]
            if sample["predicted"] is not None else None,
            "matched_video_pts": sample["matched_video_pts"],
        }
        for sample in samples
    ]
    true_count = sum(row["count_overlap"] for row in details if row["count_overlap"] is not None)
    missed = sum(row["missed_people"] for row in details if row["missed_people"] is not None)
    extra = sum(row["extra_people"] for row in details if row["extra_people"] is not None)
    denominator = 2 * true_count + missed + extra
    return {
        "f1": (2 * true_count / denominator if denominator else 1.0) if matched else None,
        "count_overlap": true_count,
        "missed_people": missed,
        "extra_people": extra,
        "undercount_frames": sum(s["predicted"] < s["expected"] for s in matched),
        "overcount_frames": sum(s["predicted"] > s["expected"] for s in matched),
        "matched": len(matched),
        "clear_details": details,
    }


def score_trial(directory, reference, trial, source):
    clear = evaluate_trial(directory, reference)
    uncertain_labels = [s for s in reference["samples"] if s["status"] == "uncertain"]
    uncertain = []
    if uncertain_labels:
        uncertain_reference = {
            **reference,
            "samples": [{**sample, "status": "clear"} for sample in uncertain_labels],
        }
        aligned = evaluate_trial(directory, uncertain_reference)
        for label, sample in zip(uncertain_labels, aligned["samples"], strict=True):
            uncertain.append(
                {
                    "timestamp_seconds": label["timestamp_seconds"],
                    "possible_counts": label.get("possible_counts", [label["person_count"]]),
                    "displayed_count": sample["predicted"],
                    "matched_video_pts": sample["matched_video_pts"],
                }
            )
    counts = count_f1(clear["samples"])
    if counts["matched"] != clear["eligible"]:
        counts["f1"] = None
    if len(uncertain) != len(uncertain_labels) or any(row["displayed_count"] is None for row in uncertain):
        raise ValueError(f"Uncertain samples failed alignment: {directory}")
    return {
        "source": source,
        "model": trial["model"],
        "delay_seconds": trial["configured_delay_seconds"],
        "repeat": trial["repeat"],
        "directory": str(directory.resolve()),
        "clear_samples": clear["eligible"],
        "matched_clear_samples": counts["matched"],
        "contour_coverage": trial["coverage"],
        **counts,
        "uncertain": uncertain,
    }


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def render_report(path, result):
    esc = html.escape
    rows = []
    for trial in sorted(result["trials"], key=lambda r: (r["delay_seconds"], r["model"], r["repeat"])):
        f1 = "—" if trial["f1"] is None else f"{trial['f1']:.1%}"
        rows.append(
            "<tr>"
            f"<td>{esc(trial['model'])}</td><td>{trial['delay_seconds']:g}</td>"
            f"<td>{trial['repeat']}</td><td>{f1}</td>"
            f"<td>{trial['missed_people']} 人次 / {trial['undercount_frames']} 帧</td>"
            f"<td>{trial['extra_people']} 人次 / {trial['overcount_frames']} 帧</td>"
            f"<td>{trial['contour_coverage']:.1%}</td>"
            f"<td>{trial['matched_clear_samples']}/{trial['clear_samples']}</td>"
            f"<td>{esc(trial['source'])}</td>"
            "</tr>"
        )
    by_timestamp = {}
    for trial in result["trials"]:
        for sample in trial["uncertain"]:
            record = by_timestamp.setdefault(
                sample["timestamp_seconds"],
                {"possible": sample["possible_counts"], "displayed": Counter()},
            )
            record["displayed"][sample["displayed_count"]] += 1
    uncertain_rows = []
    for timestamp, record in sorted(by_timestamp.items()):
        distribution = ", ".join(
            f"{count} 人：{frequency} 次" for count, frequency in sorted(record["displayed"].items())
        )
        uncertain_rows.append(
            f"<tr><td>{timestamp:g}</td><td>{esc(str(record['possible']))}</td>"
            f"<td>{esc(distribution)}</td></tr>"
        )
    path.write_text(
        "<!doctype html><html lang='zh'><meta charset='utf-8'><title>测试 3 · 与 Astra 自动标注的人数一致性</title>"
        "<style>body{font:15px/1.6 system-ui;max-width:1300px;margin:32px auto;padding:0 20px;color:#193047}"
        "table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #d9e1e8;text-align:left}"
        "th{background:#edf3f7}section{margin:28px 0;overflow:auto}p{max-width:1000px}"
        "small{color:#526176}a{color:#146b63}</style>"
        "<h1>测试 3 · 与 Astra 自动标注的人数一致性</h1>"
        f"<p>同一原视频的 {result['reference_samples']} 个采样时间点中，"
        f"{result['clear_samples']} 个 Astra 明确标注用于评分，"
        f"{result['uncertain_samples']} 个不确定标注单独列出。"
        f"共评估 {len(result['trials'])} 轮，覆盖 {result['model_count']} 个模型。"
        "本报告使用未经人工复核的原始 AI 标注；分数表示与 Astra 自动标注的一致性，"
        "不是经过验证的真实准确率，也不衡量人物身份或轮廓形状。</p>"
        "<p>人数一致性 F1 按所有明确时间点汇总：人数重叠 = Σmin(显示人数, Astra 人数)，"
        "少报 = Σmax(Astra 人数 − 显示人数, 0)，多报 = Σmax(显示人数 − Astra 人数, 0)；"
        "F1 = 2×人数重叠 ÷ (2×人数重叠 + 少报 + 多报)。"
        "人数相同但认错人仍会得满分。描边率单独显示，不与 F1 相乘。"
        "所有轮次按视频 PTS 对齐同一组采样时间点，容差 0.1 秒。</p>"
        "<section><h2>各模型与缓冲档位</h2><table><thead><tr>"
        "<th>模型</th><th>显示缓冲 / 秒</th><th>轮次</th><th>人数一致性 F1</th>"
        "<th>少报</th><th>多报</th><th>描边率</th><th>对齐 / 明确样本</th><th>批次</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
        "<p><a href='clear-samples.csv'>下载明确时间点的逐帧人数明细</a>："
        "每行列出 Astra 人数、显示人数、重叠、少报、多报及差额。"
        "差额 = 显示人数 − Astra 人数；逐帧重叠、少报、多报各自求和后得到上表的 F1。</p>"
        "<section><h2>Astra 不确定标注</h2><p>以下时间点未纳入 F1。"
        "<a href='uncertain.csv'>下载各模型逐轮的显示人数</a>；"
        "<a href='results.csv'>下载全部轮次指标</a>。</p>"
        "<table><thead><tr><th>视频时间 / 秒</th><th>Astra 候选人数</th>"
        "<th>各轮显示人数分布</th></tr></thead><tbody>"
        + "".join(uncertain_rows)
        + "</tbody></table></section>"
        f"<p><small>原始标注 SHA-256：{result['reference_sha256']}。"
        "0 秒批次与延迟扫描是不同批次；不同缓冲档位不应直接当作同条件准确率排名。"
        "延迟扫描沿用原有停止规则，并非每个模型在每个缓冲档位都有结果。"
        "</small></p></html>"
    )


def run(args):
    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    validate_reference(reference)
    if reference.get("status") != "ai_draft_pending_human_review" or any(
        sample.get("human_reviewed") is not False for sample in reference["samples"]
    ):
        raise ValueError("Expected an untouched, unreviewed AI reference")
    trials = [
        score_trial(directory, reference, trial, source)
        for source, directory, trial in discover_trials(args.combined, args.sweep)
    ]
    if len({trial["model"] for trial in trials}) != 12:
        raise ValueError("Expected all 12 Test 3 models")
    clear_count = sum(sample["status"] == "clear" for sample in reference["samples"])
    uncertain_count = len(reference["samples"]) - clear_count
    result = {
        "title": "测试 3 · 与 Astra 自动标注的人数一致性",
        "reference_status": reference["status"],
        "reference_path": str(args.reference.resolve()),
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "video_sha256": reference["video_sha256"],
        "reference_samples": len(reference["samples"]),
        "clear_samples": clear_count,
        "uncertain_samples": uncertain_count,
        "model_count": len({trial["model"] for trial in trials}),
        "metric": "Astra automatic count agreement, not verified ground-truth accuracy",
        "trials": trials,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    fields = [
        "source", "model", "delay_seconds", "repeat", "f1", "count_overlap",
        "missed_people", "extra_people", "undercount_frames", "overcount_frames",
        "contour_coverage", "matched_clear_samples", "clear_samples", "directory",
    ]
    write_csv(args.output / "results.csv", trials, fields)
    clear_rows = [
        {
            "source": trial["source"], "model": trial["model"],
            "delay_seconds": trial["delay_seconds"], "repeat": trial["repeat"],
            **sample,
        }
        for trial in trials for sample in trial["clear_details"]
    ]
    write_csv(
        args.output / "clear-samples.csv", clear_rows,
        ["source", "model", "delay_seconds", "repeat", "timestamp_seconds",
         "astra_count", "displayed_count", "count_overlap", "missed_people",
         "extra_people", "difference", "matched_video_pts"],
    )
    uncertain_rows = [
        {
            "source": trial["source"], "model": trial["model"],
            "delay_seconds": trial["delay_seconds"], "repeat": trial["repeat"],
            "timestamp_seconds": sample["timestamp_seconds"],
            "possible_counts": "/".join(map(str, sample["possible_counts"])),
            "displayed_count": sample["displayed_count"],
            "matched_video_pts": sample["matched_video_pts"],
        }
        for trial in trials for sample in trial["uncertain"]
    ]
    write_csv(
        args.output / "uncertain.csv", uncertain_rows,
        ["source", "model", "delay_seconds", "repeat", "timestamp_seconds",
         "possible_counts", "displayed_count", "matched_video_pts"],
    )
    render_report(args.output / "report.html", result)
    print(args.output / "report.html")
    print(f"{len(trials)} trials; {clear_count} clear + {uncertain_count} uncertain timestamps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
