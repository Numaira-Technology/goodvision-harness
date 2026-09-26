"""Display coverage acceptance, distinct from segmentation accuracy."""
# ruff: noqa: RUF001

import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_PRESENTATION_FPS_RETENTION = 0.8


def distribution(values):
    return {
        "count": len(values),
        **{
            name: float(np.percentile(values, percentile)) if values else None
            for name, percentile in (("p50", 50), ("p95", 95), ("max", 100))
        },
    }


def summarize_trial(directory):
    directory = Path(directory)
    report = json.loads((directory / "browser-report.json").read_text())
    job = json.loads((directory / "job.json").read_text())
    rows = [
        json.loads(line)
        for line in (directory / "frames.jsonl").read_text().splitlines()
    ]
    captured = [row for row in rows if row["kind"] == "capture"]
    presented = [row for row in rows if row["kind"] == "present"]
    captured_sequences = [row["sequence"] for row in captured]
    sequences = [row["sequence"] for row in presented]
    final = report.get("final", {})
    count = len(presented)
    continuous = sum(row.get("counted_continuous", False) for row in presented)
    complete = sum(row.get("counted_complete", False) for row in presented)
    info = report.get("config", {})
    actual = info.get("config", {})
    duration = info.get("video_duration")
    target_fps = actual.get("target_fps")
    source_span = (
        presented[-1]["pts"] - presented[0]["pts"] if len(presented) > 1 else 0
    )
    presentation_fps = (
        (count - 1) / source_span if source_span > 0 and count > 1 else None
    )
    problems = []
    if not report.get("completed") or not report.get("drained"):
        problems.append("Video/session did not finish and drain")
    if report.get("errors"):
        problems.append("Browser/transport errors were recorded")
    if not count or len(set(sequences)) != count:
        problems.append("Missing or duplicate presented source frames")
    if (
        not captured
        or len(set(captured_sequences)) != len(captured)
        or len(captured) != count
        or final.get("captured_frames") != len(captured)
    ):
        problems.append("Captured and presented source frames do not match")
    if any(
        final.get(key) != 0
        for key in (
            "display_buffer_expired",
            "display_catchup_expired",
            "rejected_source_frames",
        )
    ):
        problems.append("Display dropped or rejected source frames")
    if (
        type(duration) not in (int, float)
        or duration <= 0
        or type(target_fps) not in (int, float)
        or target_fps <= 0
        or not presented
        or presented[0]["pts"] > 1
        or presented[-1]["pts"] < duration - max(1, 2 / target_fps)
        or presentation_fps is None
        or presentation_fps < target_fps * MIN_PRESENTATION_FPS_RETENTION
    ):
        problems.append(
            "Presentation did not cover the full video at a valid frame rate"
        )
    started = report.get("play_started_at")
    ended = report.get("source_ended_at")
    playback_seconds = (
        (ended - started) / 1000
        if type(started) in (int, float)
        and type(ended) in (int, float)
        and ended > started
        else None
    )
    if (
        type(duration) in (int, float)
        and duration > 0
        and (playback_seconds is None or playback_seconds < duration * 0.9)
    ):
        problems.append("Source playback interval was incomplete or accelerated")
    if (count, continuous, complete) != (
        final.get("distinct_presented_frames"),
        final.get("continuous_contour_frames"),
        final.get("complete_contour_frames"),
    ):
        problems.append("Per-frame recount differs from production counters")
    if report.get("video_sha256") != job["video_sha256"]:
        problems.append("Source video identity changed")
    if actual.get("delay_seconds") != job["delay"]:
        problems.append("Effective display delay differs from fixed main configuration")
    if report.get("benchmark", {}).get("fixed_main") != job.get("fixed_main"):
        problems.append("Running service differs from frozen main configuration")
    coverage = continuous / count if count else None
    windows = defaultdict(list)
    for row in presented:
        windows[int(row["pts"] // 5) * 5].append(row)
    # Model/HTTP metadata arrival is separate from what appeared on screen.
    session_ids = set()
    ws_path = directory / "ws.jsonl"
    if ws_path.exists():
        for line in ws_path.read_text().splitlines():
            row = json.loads(line)
            payload = row["payload"]
            if payload.get("type") == "session":
                session_ids.add(payload.get("stream_session_id"))
    timing_path = directory / "worker-timings.json"
    model_times = []
    if timing_path.exists():
        model_times = [
            row["inference_seconds"]
            for row in json.loads(timing_path.read_text())["frames"]
            if row["stream_session_id"] in session_ids
        ]
    if not model_times:
        problems.append("No GPU inference timings matched the browser session")
    processing = {}
    status_path = directory / "status.jsonl"
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            value = json.loads(line)["payload"].get("processing")
            if value and value.get("source_to_result_seconds"):
                processing = value
    return {
        "model": job["model"],
        "configured_delay_seconds": job["delay"],
        "repeat": job["repeat"],
        "sample_fps": job["sample_fps"],
        "queue_policy": job["queue_policy"],
        "fixed_main": job["fixed_main"],
        "scope": "full_pipeline",
        "valid": not problems,
        "problems": problems,
        "captured_frames": len(captured),
        "presented_frames": count,
        "presentation_fps": presentation_fps,
        "playback_seconds": playback_seconds,
        "continuous_contour_frames": continuous,
        "complete_contour_frames": complete,
        "coverage": coverage,
        "strict_coverage": complete / count if count else None,
        "coverage_pass": not problems and coverage is not None and coverage >= 0.7,
        "segmentation_accuracy_measured": False,
        "display_delay_override_seconds": job.get("display_delay_seconds"),
        "source_to_present_seconds": distribution(
            [row["latency_seconds"] for row in presented]
        ),
        "model_inference_seconds": distribution(model_times),
        "processing": processing,
        "windows_5s": [
            {
                "start": start,
                "frames": len(items),
                "coverage": sum(row.get("counted_continuous", False) for row in items)
                / len(items),
            }
            for start, items in sorted(windows.items())
        ],
        "browser_version": report.get("browser_version"),
        "path": str(directory),
    }


def write_report(output, manifest, trials, baseline):
    output = Path(output)
    protocol = manifest["fixed_main"]
    passing = (
        manifest["repeats"] >= 2
        and len(trials) == manifest["repeats"]
        and {row["repeat"] for row in trials} == set(range(1, manifest["repeats"] + 1))
        and all(
            row.get("coverage_pass")
            and row.get("model") == manifest["model"]
            and row.get("fixed_main") == protocol
            and row["configured_delay_seconds"]
            == protocol["effective_display_delay_seconds"]
            for row in trials
        )
    )
    summary = {
        "model": manifest["model"],
        "trials": trials,
        "fixed_main": protocol,
        "repeated_coverage_pass": passing,
        "required_coverage": 0.7,
        "required_repeats": manifest["repeats"],
        "completed_repeats": len(trials),
        "sam_reference": baseline["recommendation"],
        "sam_reference_is_historical": True,
        "same_host_same_version_comparison": False,
        "segmentation_accuracy_measured": False,
        "metric_limitation": "Coverage considers only each model's known subjects. "
        "Missing detections can inflate it; inspect missed people separately.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    rows = []

    def number(value, *, scale=1, percent=False):
        if value is None:
            return "—"
        return f"{value:.1%}" if percent else f"{value * scale:.3f}"

    for row in trials:
        model = row.get("model_inference_seconds", {})
        display = row.get("source_to_present_seconds", {})
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    row["repeat"],
                    number(row.get("coverage"), percent=True),
                    number(row.get("strict_coverage"), percent=True),
                    number(model.get("p50"), scale=1000),
                    number(model.get("p95"), scale=1000),
                    number(display.get("p50")),
                    number(display.get("p95")),
                    number(row.get("presentation_fps")),
                    "通过" if row.get("coverage_pass") else "未通过 / 无效",
                    "; ".join(row.get("problems", [])),
                )
            )
            + "</tr>"
        )
    status = "通过" if passing else "尚未通过"
    delay = protocol["effective_display_delay_seconds"]
    fps = protocol["tracking"]["max_fps"]
    configuration_label = (
        "main 配置 + 显式显示缓冲实验"
        if manifest.get("display_delay_override_seconds") is not None
        else "固定 main 配置"
    )
    (output / "report.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        "<title>描边延迟评测</title>"
        "<style>body{font:16px system-ui;max-width:1200px;"
        "margin:40px auto;padding:24px;"
        "background:#f5f7fa;color:#182234}table{border-collapse:collapse;width:100%}"
        "td,th{padding:12px;border-bottom:1px solid #ccd3dd;text-align:left}</style>"
        f"<h1>{html.escape(manifest['model'])} · 完整显示链路</h1>"
        f"<p>{configuration_label}：显示缓冲 {delay:g} 秒，跟踪采样 {fps:g} FPS。</p>"
        f"<p>各轮描边率均 ≥70%：{status}（要求至少两轮）。本报告仅记录此档位。</p>"
        "<p>计入开头无描边帧及近似补全；每轮播放完整原片。"
        "模型耗时为含预处理、推理、掩码还原的 GPU worker 调用耗时，已同步 CUDA。"
        "呈现耗时为浏览器采集到画面显示的延迟，包含无描边帧；二者分别记录。</p>"
        "<p>SAM 历史结果随包保留，配置和运行环境可能不同，不据此计算加速倍数。"
        "应在固定配置下，从重复达标的模型中比较实际耗时。</p>"
        "<p>覆盖率不是分割准确率；漏检人物可能让覆盖率变高。必须结合截图和逐帧输出检查漏人。</p>"
        "<table><tr><th>轮次</th><th>描边率</th><th>严格描边率</th>"
        "<th>模型 P50 / ms</th><th>模型 P95 / ms</th>"
        "<th>呈现 P50 / s</th><th>呈现 P95 / s</th>"
        "<th>呈现 FPS</th>"
        "<th>覆盖门槛</th><th>问题</th></tr>" + "".join(rows) + "</table></html>"
    )
    return summary
