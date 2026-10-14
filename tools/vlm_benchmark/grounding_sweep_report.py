# ruff: noqa: RUF001
"""Delivery accounting for the bounded twelve-stream grounding sweep."""

import csv
import html
import json
import math
from collections import Counter

from tools.vlm_benchmark.report import write_json
from tools.vlm_multistream_benchmark import timing


def distribution(values):
    values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    return {"count": len(values), **timing(values)}


def analyze(stage, raw_path, *, real):
    """Do not count admission failures as API calls or silently lose unknowns."""
    records = [json.loads(line)["report"]
               for line in raw_path.read_text(encoding="utf-8").splitlines()]
    lane = stage["lanes"]["grounding"]
    complete = [r for r in records if r["state"] == "completed"
                and r.get("analysis_complete") is True]
    attempts, not_attempted, admission, unknown = [], [], [], []
    for row in records:
        times = row.get("timing", {})
        if str(row.get("error_code", "")).startswith("admission_"):
            admission.append(row)
        # Timing includes request construction and may be absent even after a
        # send. Only the explicit marker proves an HTTP/inspection attempt.
        attempted = times.get(
            "http_request_attempted" if real else "inspection_attempted"
        )
        if attempted is True:
            attempts.append(row)
        elif attempted is False:
            not_attempted.append(row)
        else:
            # An inspector/setup failure or lost HTTP diagnostics is not proof
            # of a dispatched request. Keep it visible instead of guessing.
            unknown.append(row)
    planned = lane["expected_observations"]
    without_record = planned - len(records)
    accounting_valid = (
        without_record >= 0 and not unknown
        and len(records) == lane["reports"]
        and len(complete) == lane["complete_valid_results"]
        and all(r in attempts for r in complete)
        and stage.get("result_retention", {}).get("valid") is True
    )
    gate = stage.get("shared_admission") or {}
    valid = (accounting_valid and stage["input_timing_valid"]
             and stage["all_drained"] and stage["result_records_missing"] == 0
             and gate.get("active") == 0 and gate.get("waiting") == 0)

    def phase(rows, name, section=None):
        return distribution([
            (r["timing"].get(section, {}) if section else r["timing"]).get(name)
            for r in rows
        ])

    elapsed = (stage["input_schedule_span_seconds"]
               + stage["drain_elapsed_seconds"])
    return {
        "kind": "real_api" if real else "simulation_no_api",
        "valid": valid, "accounting_valid": accounting_valid,
        "planned": planned, "attempts": len(attempts),
        "not_issued": planned - len(attempts) if accounting_valid else None,
        "windows_without_record": without_record,
        "confirmed_not_attempted_records": len(not_attempted),
        "admission_rejected": len(admission), "unclassified_records": len(unknown),
        "complete": len(complete),
        "complete_fraction_of_planned": len(complete) / planned if planned else None,
        "complete_fraction_of_attempts": len(complete) / len(attempts)
        if attempts and accounting_valid else None,
        "attempt_errors": dict(Counter(r["error_code"] for r in attempts
                                       if r.get("error_code"))),
        "admission_errors": dict(Counter(r["error_code"] for r in admission)),
        "validation_errors": lane["validation_reason_counts"],
        "http_status_counts": lane["http_status_counts"],
        "localization_complete": lane["localization_complete_results"],
        "elapsed_input_and_drain_seconds": elapsed,
        "complete_per_wall_minute": len(complete) * 60 / elapsed if elapsed else None,
        "input_to_publish_complete": phase(complete, "source_to_publish_seconds"),
        "api_complete": phase(complete, "total_seconds", "http_timing"),
        "api_all_attempts": phase(attempts, "total_seconds", "http_timing"),
        "pre_request_wait_attempts": phase(attempts, "source_to_request_seconds"),
        "admission_wait_attempts": phase(attempts, "admission_wait_seconds"),
        "admission_wait_rejected": phase(admission, "admission_wait_seconds"),
        "publish_order_wait_complete": phase(complete, "reorder_wait_seconds"),
        "provider_response_labels": dict(Counter(
            r["timing"].get("response_diagnostics", {}).get(
                "provider_category", "unreported") for r in attempts)),
        "shared_admission": gate,
        "per_stream": [
            {"stream": i, "planned": r["lanes"]["grounding"]["expected_observations"],
             "complete": r["lanes"]["grounding"]["complete_valid_results"],
             "input_to_publish": r["lanes"]["grounding"]["source_to_publish_seconds"]}
            for i, r in enumerate(stage["per_stream"], 1)
        ],
    }


def render(output, document):
    """Keep one row per repetition; never average percentiles across trials."""
    rows = [{"id": j["id"], "repeat": j["repeat"], "limits": j["limits"],
             "raw_reports": j["raw_reports"], **j["capacity"]}
            for j in document["jobs"] if j["phase"] == "measurement"
            and "capacity" in j]
    settings = document["settings"]
    write_json(output / "grounding-concurrency.json", {
        "state": document["state"], "settings": settings,
        "request_budget": document.get("request_budget"), "trials": rows,
    })
    flat = []
    for r in rows:
        flat.append({
            "per_stream_concurrency": r["limits"]["per_stream_limit"],
            "global_concurrency": r["limits"]["global_limit"],
            **{k: r[k] for k in ["repeat", "kind", "valid", "planned", "attempts",
                                "not_issued", "unclassified_records", "complete",
                                "complete_fraction_of_planned",
                                "complete_fraction_of_attempts",
                                "complete_per_wall_minute"]},
            "timeouts": r["attempt_errors"].get("timeout", 0),
            "invalid_responses": r["attempt_errors"].get("invalid_response", 0),
            **{f"{metric}_{q}": r[metric][q] for metric in [
                "input_to_publish_complete", "api_complete", "api_all_attempts",
                "pre_request_wait_attempts", "admission_wait_attempts",
                "publish_order_wait_complete"] for q in ["p50", "p95"]},
        })
    if flat:
        with (output / "grounding-concurrency.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    e = lambda x: html.escape(str(x))  # noqa: E731
    f = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
    pct = lambda x: "—" if x is None else f"{x:.1%}"  # noqa: E731
    cells, details = [], []
    for r in rows:
        values = [r["limits"]["per_stream_limit"], r["limits"]["global_limit"],
                  r["repeat"], f'{r["complete"]}/{r["planned"]} '
                  f'({pct(r["complete_fraction_of_planned"])})', r["attempts"],
                  pct(r["complete_fraction_of_attempts"]),
                  r["not_issued"], r["attempt_errors"].get("timeout", 0),
                  f(r["complete_per_wall_minute"]),
                  f(r["input_to_publish_complete"]["p50"]),
                  f(r["input_to_publish_complete"]["p95"]),
                  f(r["api_complete"]["p50"]),
                  f(r["pre_request_wait_attempts"]["p95"]),
                  "有效" if r["valid"] else "无效：查看诊断"]
        cells.append("<tr>" + "".join(f"<td>{e(v)}</td>" for v in values) + "</tr>")
        details.append(f'<details><summary>{e(r["id"])}：分段与逐路统计</summary>'
                       f'<a href="{e(r["raw_reports"])}">逐请求记录</a>'
                       f'<pre>{e(json.dumps(r, ensure_ascii=False, indent=2))}</pre>'
                       '</details>')
    scope = "真实模型测试" if settings["real"] else "离线模拟，不代表模型性能"
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grounding 十二路并发对照</title><style>
body{{font:16px/1.7 system-ui;margin:32px auto;padding:0 24px;max-width:1500px;
color:#183048}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border:1px solid #ccd8e2;padding:8px}}th{{background:#edf3f8}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{margin:18px 0}}
.note{{background:#fff5dc;padding:16px}}.scroll{{overflow:auto}}</style>
<h1>Grounding 十二路：每路并发 4 / 6 / 8</h1>
<p class="note">{e(scope)}；状态：{e(document["state"])}。
{e(document.get("reason", ""))}</p>
<p>模型：{e(settings["model"])}；渠道：{e(settings["backend"])}；
供应商：{e(settings.get("provider_only") or "自动路由，供应商可能变化")}。
输入频率每路 {e(settings["observation_fps"])} 次/秒；请求发起上限
{e(settings["request_rate"])} 次/秒；API 超时 {e(settings["http_timeout"])} 秒。
固定 lean-v1、六类规则、相同视频和取帧；仅同时改变每路线程、每路名额及总名额。
奇数轮正序，偶数轮反序。预检、预热不计入正式结果。配置见 manifest。</p>
<p>优先比较完整结果/计划及每分钟完整结果，再看延迟和失败数；不按发出请求数选赢家。
自动路由与时段波动会影响比较，不能证明某一项修改的独立效果。
判断正确率、框是否跟对人需另做标注对照。</p>
<div class="scroll"><table><tr><th>每路并发</th><th>总并发</th><th>轮次</th>
<th>完整/计划</th><th>请求尝试</th><th>已尝试完整率</th><th>未发出</th><th>超时</th>
<th>完整结果/分钟</th><th>总延迟 P50</th><th>总延迟 P95</th>
<th>API P50</th><th>请求前等待 P95</th><th>统计有效性</th></tr>
{"".join(cells)}</table></div>
<p>延迟单位秒。总延迟和 API P50 均只统计完整结果，详情另列全部请求尝试的 API 分布。
请求前等待统计已尝试请求从证据进入分支到开始模型处理的时间；其中已包含名额等待，
不能相加。未发出窗口没有 API 延迟，不当成 0 秒成功。</p>
<p>请求尝试表示客户端进入 HTTP 调用，不保证供应商已接收；模拟时表示模拟处理任务。
只使用明确的调用标记，不以耗时推断。无法分类或留存未核对的记录标记无效，
无法据此推算未发出数量。每分钟完整结果按输入播放及排空耗时计算，包含排空后
拿到的迟到结果；不是实时达标率。定位结构完整也不代表定位正确。</p>
<p><a href="grounding-concurrency.json">对照 JSON</a> ·
<a href="grounding-concurrency.csv">对照 CSV</a> ·
<a href="report.html">全部阶段报告</a></p>{"".join(details)}</html>'''
    (output / "grounding-concurrency.html").write_text(page, encoding="utf-8")
