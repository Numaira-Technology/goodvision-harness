"""Separate cloud execution evidence, recording regression and per-lane coverage."""
# ruff: noqa: RUF001

import html
import json

from tools.seg_pipeline.report import distribution

UPLOAD_BACKPRESSURE_MESSAGE = (
    "分析上传积压，新的分析采样暂缓；画面继续播放，已接纳帧保留。"
)
SERVER_QUEUE_BACKPRESSURE_MESSAGE = (
    "服务器分析队列暂满；画面继续播放，已接纳帧保留并等待重试。"
)
BACKPRESSURE_MESSAGES = {
    UPLOAD_BACKPRESSURE_MESSAGE,
    SERVER_QUEUE_BACKPRESSURE_MESSAGE,
}


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def invariant(info):
    # Freeze hardware limits and model settings; usage/power draw may vary.
    host = info["host"]
    protocol = dict(info["fixed_main"])
    protocol.pop("effective_display_delay_seconds", None)
    devices = []
    for line in host["gpu_at_start"].splitlines():
        columns = [column.strip() for column in line.split(",")]
        devices.append([columns[index] for index in (0, 1, 2, 5, 8)])
    return {
        "host": host["hostname"],
        "gpu_uuid_name_vram_power_limit_driver": devices,
        "cpu_count": host["cpu_count"],
        "ram_bytes": host["ram_bytes"],
        "config_sha256": host["config_sha256"],
        "harness_source_sha256": host["harness_source_sha256"],
        "worker_source_sha256": host["worker_source_sha256"],
        "protocol": protocol,
    }


def lane_summary(directory, lane, job):
    browser = read(directory / f"lane-{lane}/browser-report.json")
    evidence = read(directory / f"cloud-evidence-{lane}.json")
    frames = rows(directory / f"lane-{lane}/frames.jsonl")
    presented = [row for row in frames if row["kind"] == "present"]
    captured = [row for row in frames if row["kind"] == "capture"]
    ws = rows(directory / f"lane-{lane}/ws.jsonl")
    sessions = {
        row["payload"].get("stream_session_id")
        for row in ws
        if row["payload"].get("type") == "session"
    }
    counts = (
        len(presented),
        sum(r.get("counted_continuous", False) for r in presented),
        sum(r.get("counted_complete", False) for r in presented),
    )
    final = browser.get("final", {})
    problems = []
    if not browser.get("completed") or not browser.get("drained"):
        problems.append("Browser did not finish cleanly and drain")
    errors = browser.get("errors", [])
    backlog_warnings = [
        error
        for error in errors
        if error.get("kind") == "failure"
        and error.get("message") in BACKPRESSURE_MESSAGES
    ]
    if len(backlog_warnings) != len(errors):
        problems.append(
            "Browser reported an unexpected error; inspect browser-report.json"
        )
    if browser["video_sha256"] != job["video_sha256"]:
        problems.append("Source video changed")
    if not counts[0] or len({r["sequence"] for r in presented}) != counts[0]:
        problems.append("Missing or duplicate presented frames")
    if counts != (
        final.get("distinct_presented_frames"),
        final.get("continuous_contour_frames"),
        final.get("complete_contour_frames"),
    ):
        problems.append("Per-frame recount differs from production counters")
    if browser["config"]["config"].get("delay_seconds") != job["delay"]:
        problems.append("Display buffer differs from the scheduled condition")
    if evidence["trial"] != {"id": job["id"], "delay": job["delay"]}:
        problems.append("Cloud evidence belongs to another trial")
    if len(sessions) != 1 or None in sessions:
        problems.append("Expected exactly one identified SAM session")
    if browser.get("benchmark", {}).get("fixed_main") != evidence["fixed_main"]:
        problems.append("Configuration changed during playback")
    timings = [
        row
        for row in evidence["timings"]
        if row.get("event") == "sam_coordinator_timing"
        and row.get("stream_session_id") in sessions
        and row.get("outcome") == "ok"
        and row.get("during_capture")
        and row.get("output_frames", 0) > 0
    ]
    appends = [row["append_seconds"] for row in timings if "append_seconds" in row]
    source_ages = [
        entry["seconds"] for row in timings for entry in row.get("source_to_result", [])
    ]
    if not appends or not source_ages:
        problems.append("No matched cloud processing timings during capture")
    if any(row.get("event") == "sam_pipeline_failure" for row in evidence["timings"]):
        problems.append("SAM pipeline failure recorded")
    samples = [row for row in evidence["telemetry"] if row.get("trial_id") == job["id"]]
    if not samples:
        problems.append("Cloud CPU telemetry is missing")
    start = (
        captured[0]["at"] if captured else browser.get("play_started_at", 0)
    ) / 1000
    if not captured:
        problems.append("Actual capture start timing is missing")
    end = browser.get("source_ended_at", 0) / 1000
    duration = browser["config"]["video_duration"]
    # Source playback is real-time. Reject incomplete/abnormally accelerated playback.
    if not start or end <= start or end - start < duration * 0.9:
        problems.append("Invalid complete-video playback interval")
    elapsed = (
        (presented[-1]["at"] - presented[0]["at"]) / 1000 if len(presented) > 1 else 0
    )
    if elapsed <= 0:
        problems.append("Insufficient frame timing to measure presentation FPS")
    return {
        "lane": lane,
        "valid": not problems,
        "problems": problems,
        # Upload backpressure is a measured performance failure, not a broken
        # experiment: playback can finish cleanly while contour coverage falls.
        "analysis_backpressure_warnings": len(backlog_warnings),
        "sessions": sorted(s for s in sessions if s),
        "presented_frames": counts[0],
        "coverage": counts[1] / counts[0] if counts[0] else None,
        "strict_coverage": counts[2] / counts[0] if counts[0] else None,
        "presentation_fps": (counts[0] - 1) / elapsed if elapsed > 0 else None,
        "source_to_present_seconds": distribution(
            [r["latency_seconds"] for r in presented]
        ),
        "sam_append_seconds": distribution(appends),
        "cloud_source_to_result_seconds": distribution(source_ages),
        "play_start": start,
        "play_end": end,
        "cloud_host": evidence["host"],
        "invariant": invariant(evidence),
        "cloud_process_cpu_percent": distribution(
            [r["process_cpu_percent"] for r in samples]
        ),
        "cloud_system_cpu_percent": distribution(
            [r["system_cpu_percent"] for r in samples]
        ),
        "cloud_available_ram_bytes": distribution(
            [r["available_ram_bytes"] for r in samples]
        ),
        "cloud_swap_used_bytes": distribution([r["swap_used_bytes"] for r in samples]),
    }


def summarize(directory):
    job = read(directory / "job.json")
    lanes = [
        lane_summary(directory, lane, job) for lane in range(1, job["streams"] + 1)
    ]
    problems = [
        f"Lane {lane['lane']}: {problem}"
        for lane in lanes
        for problem in lane["problems"]
    ]
    overlap, skew = None, None
    if len(lanes) == 2:
        overlap = max(
            0, min(r["play_end"] for r in lanes) - max(r["play_start"] for r in lanes)
        )
        longest = max(r["play_end"] - r["play_start"] for r in lanes)
        overlap = overlap / longest if longest > 0 else 0
        skew = abs(lanes[0]["play_start"] - lanes[1]["play_start"])
        if set(lanes[0]["sessions"]) & set(lanes[1]["sessions"]):
            problems.append("Both lanes reused the same session")
        if lanes[0]["invariant"] != lanes[1]["invariant"]:
            problems.append("Two lanes have different fixed configurations or hosts")
    recording = None
    if job["recording"]:
        recording = read(directory / "recording.json")
        required_seconds = max(
            recording["expected_source_seconds"],
            max(lane["play_end"] - lane["play_start"] for lane in lanes),
        )
        if recording["duration_seconds"] < required_seconds * 0.95:
            problems.append("Recording does not cover the full source playback")
        if recording["method"] != job["recording_method"]:
            problems.append("Recording method differs from the scheduled test")
    return {
        "job": job,
        "lanes": lanes,
        "valid": not problems,
        "problems": problems,
        "overlap_ratio": overlap,
        "start_skew_seconds": skew,
        "recording": recording,
    }


def comparisons(plan, trials):
    result = []
    reference = next(
        (t["lanes"][0]["invariant"] for t in trials if t["valid"] and t["lanes"]),
        None,
    )
    baseline_name, changed_name = [
        condition["name"] for condition in plan["conditions"]
    ]
    for delay in plan["delays_seconds"]:
        pairs = []
        for repeat in range(1, plan["repeats"] + 1):
            matches = [
                [
                    t
                    for t in trials
                    if t["job"]["delay"] == delay
                    and t["job"]["repeat"] == repeat
                    and t["job"]["name"] == name
                ]
                for name in (baseline_name, changed_name)
            ]
            if any(len(items) != 1 for items in matches):
                pairs.append(
                    {
                        "repeat": repeat,
                        "comparable": False,
                        "reason": "Missing or duplicate condition",
                    }
                )
                continue
            before, after = [items[0] for items in matches]
            if not before["valid"] or not after["valid"]:
                pairs.append(
                    {"repeat": repeat, "comparable": False, "reason": "Invalid trial"}
                )
                continue
            a = before["lanes"][0]
            if a["invariant"] != reference or any(
                row["invariant"] != a["invariant"] for row in after["lanes"]
            ):
                pairs.append(
                    {
                        "repeat": repeat,
                        "comparable": False,
                        "reason": "Host/configuration changed",
                    }
                )
                continue
            passed = all(
                row["coverage"] >= plan["coverage_target"] for row in after["lanes"]
            )
            fps_ok = all(
                row["presentation_fps"]
                >= a["presentation_fps"] * plan["display_fps_retention_min"]
                for row in after["lanes"]
            )
            entry = {
                "repeat": repeat,
                "comparable": True,
                "each_changed_lane_coverage_pass": passed,
                "display_fps_retained": fps_ok,
            }
            if plan["name"] == "test_1_recording":
                b = after["lanes"][0]
                drop = (a["coverage"] - b["coverage"]) * 100
                ratio = b["sam_append_seconds"]["p95"] / a["sam_append_seconds"]["p95"]
                entry.update(
                    coverage_drop_pp=drop,
                    sam_append_p95_ratio=ratio,
                    no_material_recording_regression=(
                        drop <= plan["coverage_drop_tolerance_pp"]
                        and ratio <= 1 + plan["sam_p95_increase_tolerance_ratio"]
                        and fps_ok
                    ),
                )
            else:
                baseline_coverage_pass = a["coverage"] >= plan["coverage_target"]
                simultaneous = (
                    after["overlap_ratio"] >= plan["minimum_overlap_ratio"]
                    and after["start_skew_seconds"]
                    <= plan["maximum_start_skew_seconds"]
                )
                entry.update(
                    baseline_coverage_pass=baseline_coverage_pass,
                    simultaneous=simultaneous,
                    dual_stream_acceptance=passed and fps_ok and simultaneous,
                )
            pairs.append(entry)
        criterion = (
            "no_material_recording_regression"
            if plan["name"] == "test_1_recording"
            else "dual_stream_acceptance"
        )
        result.append(
            {
                "delay_seconds": delay,
                "pairs": pairs,
                "repeated_acceptance": plan["repeats"] >= 2
                and all(pair.get(criterion) is True for pair in pairs),
            }
        )
    return result


def write_report(output, manifest, trials):
    plan = manifest["plan"]
    compared = comparisons(plan, trials)
    summary = {
        "test": plan["name"],
        "question": plan["question"],
        "comparisons": compared,
        "trials": trials,
        "cloud_migration_fully_proven": False,
        "historical_5_or_6_percent_is_only_context": True,
        "interpretation": "云端执行位置由主机/PID/配置/负载记录佐证。"
        "录屏无退化不等于完整迁移证明；"
        "双路失败不能排除 CPU 瓶颈。"
        "SAM append 耗时含云端排队、CPU 和 GPU，并非纯 GPU 时间。",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )

    def number(value, percent=False):
        return "—" if value is None else f"{value:.1%}" if percent else f"{value:.3f}"

    table = []
    for trial in trials:
        for lane in trial["lanes"]:
            values = [
                trial["job"]["delay"],
                trial["job"]["repeat"],
                trial["job"]["name"],
                lane["lane"],
                number(lane["coverage"], True),
                number(lane["strict_coverage"], True),
                number(lane["presentation_fps"]),
                number(lane["sam_append_seconds"]["p95"]),
                number(lane["cloud_source_to_result_seconds"]["p95"]),
                number(lane["source_to_present_seconds"]["p95"]),
                lane["analysis_backpressure_warnings"],
                "有效" if trial["valid"] else "无效",
            ]
            table.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(v))}</td>" for v in values)
                + "</tr>"
            )

    def outcome_label(row):
        if row["repeated_acceptance"]:
            return "重复通过"
        if any(not pair.get("comparable") for pair in row["pairs"]):
            return "证据不足"
        return "未达标"

    outcomes = "".join(
        f"<li>{row['delay_seconds']:g} 秒缓冲：{outcome_label(row)}</li>"
        for row in compared
    )
    problems = "".join(
        f"<li>{html.escape(t['job']['id'] + ': ' + '; '.join(t['problems']))}</li>"
        for t in trials
        if t["problems"]
    )
    title = html.escape(plan["question"])
    content = (
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        f"<title>{title}</title>"
        "<style>body{font:16px system-ui;margin:40px;line-height:1.6}"
        "table{border-collapse:collapse}"
        "th,td{padding:10px;border:1px solid #ccc}</style>"
        f"<h1>{title}</h1><ul>{outcomes}</ul>"
        "<p>测试一分别判断录屏退化与 70% 描边门槛；无退化不代表描边已达标。"
        "测试二要求两路同时运行、每一路达到 70%，"
        "并保留单路对照至少 90% 的呈现帧率。</p>"
        "<p>分析积压警告表示新分析采样曾暂停或服务端队列暂满；"
        "若视频完整播放并正常收尾，"
        "这属于实际性能结果，仍计入描边率及达标判断。</p>"
        f"<p>{html.escape(summary['interpretation'])}</p>"
        "<p>描边率包含启动无描边帧及 main 近似补全，"
        "只统计模型已知人物，并非人工真值准确率。"
        "呈现延迟包括无描边帧；浏览器仍在本机解码、上传、传播轮廓和显示。</p>"
        f"<p>录屏方式：{html.escape(manifest['recording_method'])}。"
        "canvas 只录制浏览器画布，不能代替 macOS 系统录屏结论。</p>"
        "<table><tr><th>缓冲/s</th><th>轮次</th><th>条件</th><th>路</th><th>描边率</th>"
        "<th>严格描边率</th><th>呈现 FPS</th><th>SAM append P95/s</th>"
        "<th>云端收到帧→结果 P95/s</th><th>采集→呈现 P95/s</th>"
        "<th>分析积压警告</th><th>状态</th></tr>"
        + "".join(table)
        + f"</table><ul>{problems}</ul></html>"
    )
    (output / "report.html").write_text(content)
    return summary
