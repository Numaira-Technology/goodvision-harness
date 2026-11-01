# ruff: noqa: RUF001
"""Re-score both runs against one Astra packet, without new model calls."""

import html
import json
from pathlib import Path

from tools.decisions_benchmark.report import audit_coverage, audit_rows
from tools.tracked_grounding.report import summarize_job
from tools.tracked_grounding.suite import POLICY_SHA, VIDEO_SHA
from tools.vlm_benchmark.model_comparison import load_reference, safe_path, sha
from tools.vlm_benchmark.report import write_json


def read_run(root, reference):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    if (
        summary.get("state") != "completed"
        or summary.get("variant") != "detector-person-judgments"
        or summary.get("source_sha256") != VIDEO_SHA
        or summary.get("event_policy", {}).get("sha256") != POLICY_SHA
    ):
        raise ValueError(
            "comparison requires tracked runs with the same video and six-class policy"
        )
    trials = {}
    for job in summary["jobs"]:
        if job["phase"] != "measurement":
            continue
        key = (job["mode"], job["repeat"])
        if key in trials:
            raise ValueError("duplicate measurement round")
        if job["streams"] != 12 or job["duration"] != 55.798021:
            raise ValueError("comparison requires the unchanged twelve-camera load")
        raw = [
            json.loads(line)
            for line in safe_path(root, job["raw_reports"]).read_text().splitlines()
        ]
        expected_lanes = {
            "events": {"events"},
            "compact": {"grounding"},
            "parallel": {"events", "grounding"},
        }.get(job["mode"])
        if set(job["result"]["lanes"]) != expected_lanes:
            raise ValueError("missing or unexpected result branches")
        trials[key] = summarize_job(job, raw, reference)
        if summary.get("model") == "gpt-6-luna":
            coverage = audit_coverage([r["report"] for r in raw], audit_rows(root, job))
            trials[key]["audit_coverage"] = coverage
            trials[key]["valid"] &= coverage["valid"]
    if set(trials) != {
        (mode, r) for mode in ("events", "compact", "parallel") for r in (1, 2)
    }:
        raise ValueError("both runs must contain all six formal rounds")
    return summary, trials


def compare(results, baseline, output, packet, video):
    output = Path(output)
    if output.exists():
        raise ValueError("choose a new comparison output directory")
    if sha(video) != VIDEO_SHA:
        raise ValueError("use the original output.mp4")
    reference = load_reference(packet, video)
    if reference["policy"]["sha256"] != POLICY_SHA:
        raise ValueError("Astra policy differs")
    new, new_trials = read_run(results, reference)
    old, old_trials = read_run(baseline, reference)
    if new.get("model") != "gpt-6-luna" or old.get("model") != "qwen/qwen3.5-35b-a3b":
        raise ValueError("select tracked Decisions and tracked Qwen35B results")
    strict_fields = (
        "streams",
        "repeats",
        "observation_fps",
        "sample_fps",
        "global_limit",
        "per_stream_limit",
        "parallel_scheduling",
        "arrival_mode",
        "detector",
    )
    if any(new.get(k) != old.get(k) for k in strict_fields):
        raise ValueError("load/detector settings differ; not a matched comparison")
    rows = []
    for key in sorted(new_trials):
        before, after = old_trials[key], new_trials[key]
        if not before["valid"] or not after["valid"]:
            raise ValueError(
                "input/recording/retention integrity failed in compared run"
            )
        for lane in sorted(after["lanes"]):
            a, b = before["lanes"][lane], after["lanes"][lane]
            rows.append(
                {
                    "mode": key[0],
                    "repeat": key[1],
                    "lane": lane,
                    "qwen35b": a,
                    "decisions": b,
                    "delta_percentage_points": {
                        "sent": 100 * (b["sent_fraction"] - a["sent_fraction"]),
                        "complete": 100
                        * (b["complete_fraction"] - a["complete_fraction"]),
                        "within_3s": 100
                        * (
                            b["timely"]["on_time_coverage"]
                            - a["timely"]["on_time_coverage"]
                        ),
                    },
                }
            )
    result = {
        "rows": rows,
        "same_astra_reference": True,
        "comparison_notes": new.get("comparison_notes", []),
        "http_timeouts": {
            "qwen35b": old.get("http_timeouts"),
            "decisions": new.get("http_timeouts"),
        },
        "joint_by_round": [
            {
                "mode": k[0],
                "repeat": k[1],
                "qwen35b": old_trials[k]["joint"],
                "decisions": new_trials[k]["joint"],
            }
            for k in sorted(new_trials)
            if k[0] == "parallel"
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "comparison.json", result)
    rendered = []

    def pct(value):
        return "—" if value is None else f"{value:.1%}"

    for row in rows:
        for model in ("qwen35b", "decisions"):
            item = row[model]
            cells = [
                row["mode"],
                row["repeat"],
                row["lane"],
                model,
                pct(item["sent_fraction"]),
                pct(item["complete_fraction"]),
                pct(item["timely"]["on_time_coverage"]),
                pct(item["timely"]["worst_camera_coverage"]),
                item["end_to_end_seconds"]["p50"],
                item["end_to_end_seconds"]["p95"],
                pct(item["astra"]["agreement"]),
            ]
            rendered.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells)
                + "</tr>"
            )
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>YOLO + Decisions / Qwen35B</title>'
        "<style>body{font:16px system-ui;margin:30px}"
        "td,th{border:1px solid #ccc;padding:8px}"
        "table{border-collapse:collapse}</style>"
        "<h1>YOLO26-S + Decisions 与 Qwen35B</h1>"
        "<p>同一 Astra 参考，事件一致率不评价人物 ID 正确性。</p>"
        + "".join(f"<p>{html.escape(n)}</p>" for n in result["comparison_notes"])
        + "<table><tr><th>模式</th><th>轮次</th><th>分支</th><th>模型</th>"
        "<th>发出率</th><th>完整率</th><th>≤3s</th><th>最差一路≤3s</th>"
        "<th>P50秒</th><th>P95秒</th><th>Astra</th></tr>"
        + "".join(rendered)
        + "</table>",
        encoding="utf-8",
    )
    return result
