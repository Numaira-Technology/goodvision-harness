# ruff: noqa: RUF001
"""Exact-window delivery and latency comparison; never infer missing HTTP calls."""

import csv
import html
import json
from collections import Counter

from tools.vlm_benchmark.accounting import complete, delivery_counts
from tools.vlm_benchmark.report import write_json


def analyze(stage, raw_path, *, real):
    # Cloud bootstrap failures must still render without PIL/model dependencies.
    from tools.vlm_benchmark.grounding_sweep_report import distribution
    from tools.vlm_multistream_benchmark import report_lane

    reports = [
        json.loads(line)["report"]
        for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    lanes = {}
    for lane, summary in stage["lanes"].items():
        rows = [r for r in reports if report_lane(r) == lane]
        count = delivery_counts(rows, summary["expected_observations"])
        attempted = [
            r for r in rows if r["timing"].get("http_request_attempted") is True
        ]
        good = [r for r in rows if complete(r)]
        known = (
            not count["http_attempt_unknown_records"]
            and not count["records_exceed_plan"]
            and len(rows) == summary["reports"]
            and len(good) == summary["complete_valid_results"]
        )

        def phase(records, key, section=None):
            return distribution(
                [
                    (r["timing"].get(section, {}) if section else r["timing"]).get(key)
                    for r in records
                ]
            )

        planned = count["planned_windows"]
        lanes[lane] = {
            **count,
            "accounting_known": known,
            "http_attempt_fraction": len(attempted) / planned
            if planned and known
            else None,
            "complete_fraction": len(good) / planned if planned else None,
            "complete_fraction_of_http_attempts": count["http_attempted_complete"]
            / len(attempted)
            if attempted and known
            else None,
            "errors": dict(
                Counter(r["error_code"] for r in rows if r.get("error_code"))
            ),
            "http_status_counts": summary["http_status_counts"],
            "complete_latency": phase(good, "source_to_publish_seconds"),
            "api_all_attempts": phase(attempted, "total_seconds", "http_timing"),
            "queue_attempts": phase(attempted, "admission_wait_seconds"),
            "queue_not_attempted": phase(
                [r for r in rows if r["timing"].get("http_request_attempted") is False],
                "admission_wait_seconds",
            ),
            "completion_to_publish": phase(good, "completion_to_publish_seconds"),
            "provider_response_labels": dict(
                Counter(
                    r["timing"]
                    .get("response_diagnostics", {})
                    .get("provider_category", "unreported")
                    for r in attempted
                )
            ),
        }
    pair = stage["paired_windows"]
    accounting_valid = (
        bool(lanes)
        and all(r["accounting_known"] for r in lanes.values())
        and pair["http_pair_accounting_known"]
        and stage.get("result_retention", {}).get("valid") is True
    )
    gate = stage.get("shared_admission") or {}
    valid = (
        accounting_valid
        and stage["input_timing_valid"]
        and stage["all_drained"]
        and stage["result_records_missing"] == 0
        and gate.get("active") == 0
        and gate.get("waiting") == 0
    )
    elapsed = stage["input_schedule_span_seconds"] + stage["drain_elapsed_seconds"]
    return {
        "kind": "real_api" if real else "simulation_no_api",
        "accounting_valid": accounting_valid,
        "valid": valid,
        "lanes": lanes,
        "pairs": pair,
        "shared_admission": gate,
        "both_complete_per_wall_minute": pair["both_complete"] * 60 / elapsed
        if elapsed
        else None,
        "per_stream": [
            {"camera": i, "lanes": r["lanes"]}
            for i, r in enumerate(stage["per_stream"], 1)
        ],
    }


def render(output, document):
    trials = [
        {
            "id": j["id"],
            "repeat": j["repeat"],
            "limits": j["limits"],
            "parallel_scheduling": j["parallel_scheduling"],
            "raw_reports": j["raw_reports"],
            **j["capacity"],
            "route_audit": j.get("route_audit"),
        }
        for j in document["jobs"]
        if j["phase"] == "measurement" and "capacity" in j
    ]
    for r in trials:
        r["valid"] = r["valid"] and (r.get("route_audit") or {}).get("valid", True)
    write_json(
        output / "parallel-concurrency.json",
        {
            "state": document["state"],
            "settings": document["settings"],
            "request_budget": document.get("request_budget"),
            "trials": trials,
        },
    )
    flat = []
    for r in trials:
        pair = r["pairs"]
        row = {
            "id": r["id"],
            "scheduling": r["parallel_scheduling"],
            **r["limits"],
            "repeat": r["repeat"],
            "valid": r["valid"],
            "kind": r["kind"],
            **{
                k: pair[k]
                for k in (
                    "expected_source_windows",
                    "both_http_attempted",
                    "both_http_attempted_fraction",
                    "both_complete",
                    "both_complete_fraction",
                    "only_one_http_attempted",
                    "neither_http_attempted",
                    "one_complete_peer_http_not_attempted",
                    "one_complete_peer_http_attempted_without_complete",
                )
            },
            "both_complete_per_wall_minute": r["both_complete_per_wall_minute"],
            **{
                f"pair_latency_{q}": pair["both_published_seconds"][q]
                for q in ("p50", "p95")
            },
        }
        for lane, d in r["lanes"].items():
            row.update(
                {
                    f"{lane}_{k}": d[k]
                    for k in (
                        "http_attempted",
                        "http_attempt_fraction",
                        "complete_results",
                        "complete_fraction",
                        "http_attempted_without_complete",
                        "planned_without_record",
                        "http_not_attempted_records",
                        "http_attempt_unknown_records",
                    )
                }
            )
            row.update(
                {
                    f"{lane}_{m}_{q}": d[m][q]
                    for m in (
                        "complete_latency",
                        "api_all_attempts",
                        "queue_attempts",
                        "completion_to_publish",
                    )
                    for q in ("p50", "p95")
                }
            )
        flat.append(row)
    if flat:
        with (output / "parallel-concurrency.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    e = lambda x: html.escape(str(x))  # noqa: E731
    pct = lambda x: "未知" if x is None else f"{x:.1%}"  # noqa: E731
    sec = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
    cells, details = [], []
    for r in trials:
        p, ev, gr = r["pairs"], r["lanes"]["events"], r["lanes"]["grounding"]
        values = [
            r["parallel_scheduling"],
            r["limits"]["global_limit"],
            r["repeat"],
            pct(ev["http_attempt_fraction"]),
            pct(gr["http_attempt_fraction"]),
            pct(p["both_http_attempted_fraction"]),
            pct(ev["complete_fraction"]),
            pct(gr["complete_fraction"]),
            pct(p["both_complete_fraction"]),
            sec(p["both_published_seconds"]["p50"]),
            sec(p["both_published_seconds"]["p95"]),
            "有效" if r["valid"] else "无效，查看诊断",
        ]
        cells.append("<tr>" + "".join(f"<td>{e(v)}</td>" for v in values) + "</tr>")
        details.append(
            f"<details><summary>{e(r['id'])}：排队、失败、供应商及逐路明细</summary>"
            f'<a href="{e(r["raw_reports"])}">逐请求记录</a>'
            f"<pre>{e(json.dumps(r, ensure_ascii=False, indent=2))}</pre></details>"
        )
    s = document["settings"]
    scope = "真实 API 测试" if s["real"] else "离线模拟：没有 API 请求，不代表模型性能"
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parallel 配对与并发对照</title><style>
body{{font:16px/1.7 system-ui;max-width:1500px;margin:32px auto;
padding:0 24px;color:#183048}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccd8e2;padding:8px}}
th{{background:#edf3f8}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
details{{margin:20px 0}}.note{{padding:16px;background:#fff5dc}}</style>
<h1>Parallel 十二路：配对调度与 48 / 72 / 96 并发</h1>
<p class="note">{e(scope)}；状态 {e(document["state"])}。
{e(document.get("reason", ""))}</p>
<p>模型 {e(s["model"])}；{e(s["backend"])}；
优先服务商 {e(s.get("provider_prefer", ""))}，
实际返回的服务商分布见每轮明细。固定每分支每路 {e(s["observation_fps"])} Hz，
请求发起上限 {e(s["request_rate"])} 次/秒。同一输入、六类规则及输出格式保持不变。</p>
<p>第一轮：原调度 48 → 配对 48 → 配对 72 → 配对 96；第二轮反序。
每路共享名额 4/6/8，本地每分支上限同步为 4/6/8，名额计入两条分支。
配对模式先确认两分支都有本地位置，再提交同一证据窗口；一条获得名额后，优先放行另一条。
跨视频路保持公平，等待和证据过期限制不变；两条结果独立发布，不等待慢分支。
这会牺牲部分单独 events 机会，是否改善整体完成率需实测。</p>
<p>以下比例均以全部计划源窗口为分母。两条均完整才算整体完成；历史完整结果也计入。
模型返回不确定可属于结构完整，完整返回率不是判断准确率。预检、预热不计入正式表格。</p>
<div class="scroll"><table><tr><th>调度</th><th>总并发</th><th>轮次</th>
<th>Events 发出率</th>
<th>Grounding 发出率</th><th>两条均发出</th><th>Events 完整</th><th>Grounding 完整</th>
<th>两条均完整</th><th>配对 P50 秒</th><th>配对 P95 秒</th><th>统计</th></tr>
{"".join(cells)}</table></div>
<p>发出指客户端尝试 HTTP，不保证供应商接收；排空和留存核对通过后才能解释未发出数。
配对延迟取同一窗口两条发布延迟的较大值，仅统计两条均完整窗口；失败及未发出仍在完成率分母中。
详情保留所有 HTTP 尝试的 API 延迟、已发出/未发出记录的排队延迟以及完成到发布的等待。
不平均两轮 P95，不把未知记录当作未发出，也不把模拟任务当作 API 请求。</p>
<p>48 原调度与配对 48 用于比较调度；配对 48/72/96 用于比较容量。
优先同时看整体完成率、每路结果、API 超时和 P95；不能仅凭发出数选择配置。
供应商与时段波动仍会影响结果，不是业务 SLA 或长期稳定性保证。</p>
<p><a href="parallel-concurrency.json">JSON</a> ·
<a href="parallel-concurrency.csv">CSV</a> ·
<a href="report.html">全部阶段与录屏</a></p>{"".join(details)}</html>"""
    (output / "parallel-concurrency.html").write_text(page, encoding="utf-8")
