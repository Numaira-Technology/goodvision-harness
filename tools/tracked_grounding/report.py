# ruff: noqa: RUF001
"""Offline summary; absent/failed/late windows remain in the denominator."""

import html
import json

from tools.decisions_benchmark.report import astra_agreement
from tools.vlm_benchmark.accounting import complete, delivery_counts
from tools.vlm_benchmark.model_comparison import safe_path
from tools.vlm_benchmark.report import write_json
from tools.vlm_benchmark.vision_timeliness import assess_timeliness
from tools.vlm_multistream_benchmark import planned_observations, report_lane, timing


def summarize_job(job, raw, reference):
    result = job["result"]
    cameras = result.get("streams", 12)
    expected = planned_observations(
        result.get("duration_seconds", 55.798021),
        result.get("observation_fps", 1.25),
        result.get("source_sample_fps"),
    )
    planned = expected * cameras
    timely = assess_timeliness(
        raw,
        set(result["lanes"]),
        planned_per_camera=expected,
        cameras=cameras,
    )
    lanes = {}
    for lane in job["result"]["lanes"]:
        selected = [item for item in raw if report_lane(item["report"]) == lane]
        records = [item["report"] for item in selected]
        good = [r for r in records if complete(r)]
        counts = delivery_counts(records, planned)
        latencies = [r["timing"]["source_to_publish_seconds"] for r in good]
        lanes[lane] = {
            "delivery": counts,
            "planned_windows": planned,
            "sent_fraction": counts["http_attempted"] / planned,
            "complete_fraction": len(good) / planned,
            "timely": timely["lanes"][lane],
            "end_to_end_seconds": timing(latencies),
            "detector_failed_windows": sum(
                r["timing"].get("inspection_timing", {}).get("detector_failed") == 1
                for r in records
            ),
            "astra": astra_agreement(selected, reference),
            "http_status_counts": job["result"]["lanes"][lane]["http_status_counts"],
            "phase_metrics": job["result"]["lanes"][lane]["request_phase_metrics"],
        }
    return {
        "id": job["id"],
        "mode": job["mode"],
        "repeat": job["repeat"],
        "lanes": lanes,
        "joint": timely["joint"],
        "detector_counts": job["detector_counts"],
        "recording": job.get("recording"),
        "valid": bool(
            job["result"]["all_drained"]
            and job["result"]["input_timing_valid"]
            and not job["result"]["result_records_missing"]
            and job["result"]["result_retention"].get("valid")
            and (job.get("recording") or {}).get("complete")
        ),
    }


def analyze(root, reference):
    summary = json.loads((root / "summary.json").read_text())
    baseline = summary.get("variant") == "qwen-own-grounding"
    title = "Qwen35B 原定位基线" if baseline else "YOLO26-S 人物 ID + Qwen35B"
    title = summary.get("report_title", title)
    if summary.get("quantization"):
        title += "（本地 FP8）"
    trials = []
    for job in summary["jobs"]:
        if job["phase"] != "measurement":
            continue
        raw = [
            json.loads(line)
            for line in safe_path(root, job["raw_reports"]).read_text().splitlines()
        ]
        trial = summarize_job(job, raw, reference)
        if summary.get("version") in {
            "vast-qwen35-fp8-yolo26s-v1",
            "vast-qwen35-fp8-yolo26s-v2-main",
            "vast-qwen35-fp8-yolo26s-v3-isolated",
        }:
            from tools.vast_vlm.fp8_adapter import stage_metrics

            for lane, data in trial["lanes"].items():
                data["tokens"] = stage_metrics(
                    [i["report"] for i in raw if report_lane(i["report"]) == lane],
                    job["result"],
                )
            trial["telemetry"] = job.get("telemetry")
        trials.append(trial)
    result = {
        "state": summary["state"],
        "variant": summary.get("variant"),
        "model": summary.get("model"),
        "provider": summary.get("provider"),
        "quantization": summary.get("quantization"),
        "trials": trials,
        "all_trials_complete": summary["state"] == "completed"
        and len(trials) == 6
        and all(t["valid"] for t in trials),
        "comparison_notes": summary.get("comparison_notes", []),
        "quality_scope": "Astra scene agreement; does not score person ID correctness",
    }
    if summary.get("provider") == "OpenAI direct; no fallback":
        from tools.decisions_benchmark.report import (
            audit_coverage,
            audit_rows,
            audit_usage,
        )

        records = []
        for job in summary["jobs"]:
            rows = [
                json.loads(line)["report"]
                for line in safe_path(root, job["raw_reports"]).read_text().splitlines()
            ]
            records.extend(rows)
            if job["phase"] == "measurement":
                trial = next(t for t in trials if t["id"] == job["id"])
                trial["audit_coverage"] = audit_coverage(rows, audit_rows(root, job))
                trial["valid"] &= trial["audit_coverage"]["valid"]
        result["decisions_usage"] = audit_usage(root, records)
        result["all_trials_complete"] &= result["decisions_usage"]["audit_coverage"][
            "valid"
        ] and all(t["valid"] for t in trials)
    write_json(root / "tracked-comparison.json", result)
    rows = []
    pct = lambda v: "—" if v is None else f"{v:.1%}"  # noqa: E731
    sec = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    for trial in trials:
        for lane, data in trial["lanes"].items():
            values = [
                "grounding" if trial["mode"] == "compact" else trial["mode"],
                trial["repeat"],
                lane,
                pct(data["sent_fraction"]),
                pct(data["complete_fraction"]),
                pct(data["timely"]["on_time_coverage"]),
                pct(data["timely"]["worst_camera_coverage"]),
                sec(data["end_to_end_seconds"]["p50"]),
                sec(data["end_to_end_seconds"]["p95"]),
                pct(data["astra"]["agreement"]),
                data["detector_failed_windows"],
            ]
            rows.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(v))}</td>" for v in values)
                + "</tr>"
            )
    videos = []
    for path in sorted(root.glob("*/recording/*.mp4")):
        relative = html.escape(path.relative_to(root).as_posix(), quote=True)
        videos.append(
            f"<p>{html.escape(path.parent.parent.name)}</p>"
            f'<video controls width="960" src="{relative}"></video>'
        )
    (root / "report.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>{title}</title>'
        "<style>body{font:16px "
        "system-ui;margin:32px}table{border-collapse:collapse}"
        "th,td{padding:10px;border:1px solid #bbb}</style>"
        f"<h1>{title}</h1>"
        f"<p>状态：{html.escape(summary['state'])}；每分支每轮 "
        f"{summary.get('planned_windows_per_branch_per_round', 828)} 个计划窗口。</p>"
        "<p>延迟为源帧进入后的端对端耗时；启用检测时包括检测等待。"
        "目标是每一路三秒内完整率 ≥95%，不能用总体平均替代。Astra "
        "只评价场景六类判断，不评价人物关联。检测失败不会作为无人或阴性。</p>"
        + "".join(f"<p>{html.escape(note)}</p>" for note in result["comparison_notes"])
        + "<table><tr><th>模式</th><th>轮次</th><th>分支</th>"
        "<th>发出率</th><th>完整率</th>"
        "<th>≤3s</th><th>最差一路≤3s</th><th>P50(s)</th><th>P95(s)</th>"
        "<th>Astra</th><th>检测失败窗口</th></tr>"
        + "".join(rows)
        + "</table>"
        + "".join(videos),
        encoding="utf-8",
    )
    return result
