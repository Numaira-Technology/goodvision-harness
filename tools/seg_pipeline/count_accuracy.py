"""Offline displayed-person count evaluation against independent sampled labels."""
# ruff: noqa: E501, RUF001 -- embedded HTML/CSS/JS and Chinese report prose

import argparse
import base64
import csv
import html
import json
import math
from pathlib import Path


def validate_reference(reference):
    samples = reference.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Reference must contain sampled frames")
    timestamps = set()
    for label in samples:
        timestamp = label.get("timestamp_seconds")
        if (
            type(timestamp) not in (int, float)
            or not math.isfinite(timestamp)
            or timestamp < 0
            or timestamp in timestamps
        ):
            raise ValueError("Reference timestamps must be unique finite seconds")
        timestamps.add(timestamp)
        if label.get("status") not in {"clear", "uncertain"}:
            raise ValueError("Reference status must be clear or uncertain")
        if type(label.get("person_count")) is not int or label["person_count"] < 0:
            raise ValueError("Reference person count must be a nonnegative integer")


def evaluate_trial(directory, reference, tolerance=0.1):
    validate_reference(reference)
    directory = Path(directory)
    job = json.loads((directory / "job.json").read_text())
    if job.get("video_sha256") != reference["video_sha256"]:
        raise ValueError(f"Video SHA256 mismatch: {directory}")
    rows = [
        json.loads(line)
        for line in (directory / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    captures = {r["sequence"]: r for r in rows if r["kind"] == "capture"}
    presented = []
    for row in rows:
        if row["kind"] != "present":
            continue
        capture = captures.get(row["sequence"])
        if capture is None or "video_pts" not in capture:
            continue
        presented.append((capture["video_pts"], row))
    samples = []
    for label in reference["samples"]:
        if label["status"] != "clear":
            continue
        t = label["timestamp_seconds"]
        nearest = min(presented, key=lambda pair: abs(pair[0] - t), default=None)
        record = {
            "timestamp_seconds": t,
            "expected": label["person_count"],
            "predicted": None,
            "matched_video_pts": None,
            "sequence": None,
            "absolute_error": None,
            "signed_error": None,
            "scaled_count_score": None,
            "correct": None,
        }
        if nearest is not None and abs(nearest[0] - t) <= tolerance:
            source_pts, frame = nearest
            # Count displayed entries, including duplicate detections; do not deduplicate.
            predicted = len(frame["subjects"])
            record.update(
                predicted=predicted,
                matched_video_pts=source_pts,
                sequence=frame["sequence"],
                absolute_error=abs(predicted - label["person_count"]),
                signed_error=predicted - label["person_count"],
                scaled_count_score=max(
                    0.0,
                    1
                    - abs(predicted - label["person_count"])
                    / max(label["person_count"], 1),
                ),
                correct=predicted == label["person_count"],
            )
        samples.append(record)
    matched = [s for s in samples if s["predicted"] is not None]
    correct = sum(s["correct"] for s in matched)
    return {
        "model": job["model"],
        "delay_seconds": job.get("display_delay_seconds", job.get("delay")),
        "repeat": directory.name,
        "eligible": len(samples),
        "matched": len(matched),
        "correct": correct,
        "exact_count_accuracy": correct / len(matched) if matched else None,
        "scaled_count_score": sum(s["scaled_count_score"] for s in matched)
        / len(matched)
        if matched
        else None,
        "mae": sum(s["absolute_error"] for s in matched) / len(matched)
        if matched
        else None,
        "undercount_frames": sum(s["signed_error"] < 0 for s in matched),
        "overcount_frames": sum(s["signed_error"] > 0 for s in matched),
        "undercount_people": sum(max(-s["signed_error"], 0) for s in matched),
        "overcount_people": sum(max(s["signed_error"], 0) for s in matched),
        "complete_alignment": len(matched) == len(samples) and bool(samples),
        "samples": samples,
    }


STYLE = """body{font:16px system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 24px;color:#182333;background:#f5f7fa}h1{font-size:28px}p{line-height:1.7}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d9e0e9;padding:10px;text-align:left}th{background:#eaf0f7}small{color:#526176} .note{padding:16px;background:#fff4d6;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}.card{padding:15px;background:white;border-radius:10px}.card img{width:100%}input,select,button{font:inherit;padding:7px;margin:5px}.uncertain{border:2px solid #ca8210}.bright img{filter:brightness(1.6)}@media(max-width:750px){.grid{grid-template-columns:1fr}}"""


def write_review(reference, frames_dir, output):
    validate_reference(reference)
    frames_root = Path(frames_dir).resolve(strict=True)
    cards = []
    for i, label in enumerate(reference["samples"]):
        relative = Path(label["image"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() not in {".jpg", ".jpeg"}
        ):
            raise ValueError(f"Invalid review image path: {relative}")
        img = (frames_root / relative).resolve(strict=True)
        if not img.is_relative_to(frames_root) or not img.is_file():
            raise ValueError(f"Review image escapes frames directory: {relative}")
        data = base64.b64encode(img.read_bytes()).decode()
        cards.append(f'''<div class="card {label["status"]}"><b>{label["timestamp_seconds"]:.3f} 秒</b>
<img loading="lazy" src="data:image/jpeg;base64,{data}"><p>{html.escape(label["note"])}</p>
<label>人数 <input type="number" min="0" value="{label["person_count"]}" id="count-{i}"></label>
<select id="status-{i}"><option value="clear" {"selected" if label["status"] == "clear" else ""}>明确</option><option value="uncertain" {"selected" if label["status"] == "uncertain" else ""}>不确定（不评分）</option></select>
<label><input type="checkbox" id="review-{i}" {"checked" if label.get("human_reviewed") else ""}>已人工复核</label></div>''')
    payload = json.dumps(reference, ensure_ascii=False).replace("<", "\\u003c")
    script = """function download(){for(let i=0;i<reference.samples.length;i++){let r=reference.samples[i];let count=Number(document.getElementById(`count-${i}`).value);if(!Number.isInteger(count)||count<0){alert('人数必须是非负整数');return;}r.person_count=count;r.status=document.getElementById(`status-${i}`).value;r.human_reviewed=document.getElementById(`review-${i}`).checked;if(r.status==='clear')r.possible_counts=[count];}reference.status=reference.samples.every(r=>r.human_reviewed)?'human_reviewed':'ai_draft_pending_human_review';let a=document.createElement('a');let url=URL.createObjectURL(new Blob([JSON.stringify(reference,null,2)],{type:'application/json'}));a.href=url;a.download='output-2.reviewed.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}"""
    output.write_text(f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>视频人数标注 · 待复核</title><style>{STYLE}</style>
<h1>视频人数标准答案初稿</h1><p class="note">参考状态：{html.escape(reference["status"])}。{len(reference["samples"])} 个时间点，{sum(r["status"] != "clear" for r in reference["samples"])} 个不确定点。未勾选人工复核的标注仍为 AI 初稿。</p>
<p>{html.escape(reference["counting_rule"])}</p><p>{html.escape(reference["sampling"])}</p>
<p>可以修改每帧人数、确认状态并下载 JSON；下载后重新运行评估命令即可更新报告。页面修改不会自动保存。</p>
<button onclick="document.body.classList.toggle('bright')">切换提亮显示</button><button onclick="download()">下载复核结果 JSON</button>
<div class="grid">{"".join(cards)}</div><script>const reference={payload};{script}</script></html>""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    reviewed = (
        reference.get("status") == "human_reviewed"
        and bool(reference["samples"])
        and all(row.get("human_reviewed") is True for row in reference["samples"])
    )
    title = "人数计数评估 · 人工复核版" if reviewed else "人数计数评估 · 初步评估"
    notice = (
        "参考答案已由用户逐帧复核；以下结果使用复核后的明确标注计算。"
        if reviewed
        else "参考答案仍含 AI 初稿，人工复核后才能作为正式 benchmark 结论。"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    trials = [
        evaluate_trial(p.parent, reference)
        for p in sorted(args.batch.glob("*/delay-*/repeat-*/job.json"))
    ]
    if not trials:
        raise ValueError("No trial files found")
    summary = {
        "reference_status": reference["status"],
        "video_sha256": reference["video_sha256"],
        "metric": "displayed_person_count_not_raw_model_detection_or_mask_accuracy",
        "scaled_count_score_formula": "mean(max(0, 1 - abs(displayed-reference)/max(reference,1))) over aligned clear samples",
        "alignment": "Nearest captured video_pts within 0.1s, joined to presentation by sequence; missing matches excluded and reported.",
        "reference_samples": len(reference["samples"]),
        "excluded_uncertain": sum(s["status"] != "clear" for s in reference["samples"]),
        "trials": trials,
    }
    (args.output / "accuracy.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    with (args.output / "accuracy.csv").open("w", newline="") as stream:
        keys = [k for k in trials[0] if k != "samples"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: t[k] for k in keys} for t in trials)
    rows = []
    for trial in trials:
        score = trial["scaled_count_score"]
        accuracy = trial["exact_count_accuracy"]
        mae = trial["mae"]
        score_text = f"{score:.1%}" if score is not None else "—"
        acc_text = f"{accuracy:.1%}" if accuracy is not None else "—"
        mae_text = f"{mae:.2f}" if mae is not None else "—"
        rows.append(
            f"<tr><td>{html.escape(trial['model'])}</td><td>{trial['delay_seconds']:g}</td><td>{trial['repeat']}</td><td>{trial['matched']}/{trial['eligible']}</td><td>{score_text}</td><td>{acc_text}</td><td>{mae_text}</td><td>{trial['undercount_frames']} 帧 / {trial['undercount_people']} 人次</td><td>{trial['overcount_frames']} 帧 / {trial['overcount_people']} 人次</td></tr>"
        )
    (
        args.output / "report.html"
    ).write_text(f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>{title}</title><style>{STYLE}</style>
<h1>{title}</h1><p class="note">{notice}</p>
<p>原视频约 55.8 秒，每秒抽取一帧，共 56 帧。默认排除不确定帧，本次排除 {summary["excluded_uncertain"]} 帧。<a href="reference-review.html">查看并复核标准答案</a>。</p>
<p><b>人数比例得分</b>：每帧得分 = max(0, 1 − |显示人数 − 参考人数| ÷ max(参考人数, 1))，再对成功对齐的明确采样点取平均。例如参考 4 人、显示 3 人或 5 人均得 75%；两者均为 0 人得 100%。<b>完全一致率</b> = 显示人数与参考人数完全相同的采样点数 ÷ 成功对齐的明确采样点数。<b>MAE</b> = 平均人数绝对误差（越低越好）。少报与多报分别列出，其中“人次”是各采样点差额之和，同一人持续漏报会重复计入。
按 source video PTS 就近对齐，容差 0.1 秒，丢失匹配单独计数，不当成 0 人。启动阶段也计入。</p>
<p>统计现有浏览器日志中实际显示的 subject 条目数，包含追踪与显示管线的影响；不是原始模型检测准确率，不是轮廓 IoU，也不代表识别到了正确的人。原有 70% 描边率门槛未改变。</p>
<table><thead><tr><th>模型</th><th>显示缓冲（秒）</th><th>重复</th><th>对齐 / 明确样本</th><th>人数比例得分</th><th>完全一致率</th><th>MAE（人）</th><th>少报</th><th>多报</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p>这些模型的显示缓冲档位可能不同，不宜把混合档位直接当作公平速度排名。参考视频 SHA256：<small>{reference["video_sha256"]}</small></p></html>""")
    write_review(reference, args.frames, args.output / "reference-review.html")
    print(args.output / "report.html")
    print(
        f"{len(trials)} trials; {sum(t['complete_alignment'] for t in trials)} fully aligned"
    )


if __name__ == "__main__":
    main()
