# ruff: noqa: RUF001
"""Reference agreement, with failure/uncertainty visible in fixed denominators."""

from __future__ import annotations

import csv
import html
import math
from collections import Counter
from pathlib import Path

from tools.vlm_benchmark.report import write_json

NAMES = {
    "fight": "打架",
    "smoking": "吸烟",
    "fall": "跌倒",
    "weapon": "武器",
    "ppe_violation": "防护违规",
    "ground_litter": "地面垃圾",
}
OUTCOMES = {
    "tp": "阳性一致",
    "tn": "阴性一致",
    "fp": "疑似误报",
    "fn": "疑似漏报",
    "reference_uncertain": "Astra 不确定",
    "model_uncertain": "模型不确定",
    "failed": "请求失败",
    "incomplete": "返回类别不完整",
    "not_run": "未执行",
    "skipped": "队列跳过",
    "not_sampled": "按间隔未采样",
}


def ratio(n, d):
    return n / d if d else None


def quantile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)] if values else None


def _summarize_source(document, manifest, reference, records):
    labels = document["event_policy"]["event_labels"]
    refs = {r["window_id"]: r for r in reference["rows"]}
    by_request = {r["request_id"]: r for r in records}
    if len(by_request) != len(records):
        raise ValueError("duplicate result request")
    planned_ids = {p["request_id"] for p in document["planned"]}
    if not set(by_request) <= planned_ids:
        raise ValueError("unplanned result request")
    details, groups, delivery = [], {}, {}
    for item in document["planned"]:
        record = by_request.get(item["request_id"])
        if record and any(
            record[k] != item[k] for k in ("mode", "repeat", "window_id", "stream_id")
        ):
            raise ValueError("result request does not match planned window")
        stage = item["repeat"], item["mode"]
        d = delivery.setdefault(
            stage,
            {
                "planned": 0,
                "selected": 0,
                "complete": 0,
                "failed": 0,
                "incomplete": 0,
                "not_run": 0,
                "skipped": 0,
                "not_sampled": 0,
                "latencies": [],
            },
        )
        d["planned"] += 1
        selected = item.get("sampling_selected", True)
        d["selected"] += selected
        state = record["state"] if record else "not_run" if selected else "not_sampled"
        if (state == "not_sampled") != (not selected):
            raise ValueError("result does not match sampling plan")
        if state not in {
            "complete",
            "failed",
            "incomplete",
            "not_run",
            "skipped",
            "not_sampled",
        }:
            raise ValueError("unexpected result state")
        d[state] += 1
        if record and record.get("inspect_seconds") is not None:
            elapsed = record["inspect_seconds"]
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("invalid inspection timing")
            d["latencies"].append(elapsed)
        for label in labels:
            ref = refs[item["window_id"]]["judgments"][label]
            present = bool(record and label in record["judgments"])
            predicted = record["judgments"].get(label) if present else None
            if type(predicted) not in (bool, type(None)):
                raise ValueError("invalid prediction type")
            if ref["detected"] is None:
                outcome = "reference_uncertain"
            elif state != "complete":
                outcome = state
            elif not present:
                outcome = "incomplete"
            elif predicted is None:
                outcome = "model_uncertain"
            elif ref["detected"]:
                outcome = "tp" if predicted else "fn"
            else:
                outcome = "fp" if predicted else "tn"
            group = groups.setdefault((*stage, label), Counter())
            group["planned"] += 1
            group["selected"] += selected
            if ref["detected"] is True:
                group["reference_positive_selected"] += selected
                group["reference_positive_not_sampled"] += not selected
            group[
                "reference_positive"
                if ref["detected"] is True
                else "reference_negative"
                if ref["detected"] is False
                else "reference_uncertain_support"
            ] += 1
            group[outcome] += 1
            if present and predicted is None and state == "complete":
                group["model_uncertain_all_references"] += 1
            details.append(
                {
                    **item,
                    "judgment_time_seconds": refs[item["window_id"]][
                        "judgment_time_seconds"
                    ],
                    "event_label": label,
                    "reference": ref["detected"],
                    "prediction": predicted,
                    "prediction_present": present,
                    "request_state": state,
                    "outcome": outcome,
                    "reference_reason": ref["reason"],
                    "error_code": record.get("error_code") if record else None,
                }
            )
    metrics = []
    for (repeat, mode, label), c in groups.items():
        certain = c["reference_positive"] + c["reference_negative"]
        binary = sum(c[x] for x in ("tp", "tn", "fp", "fn"))
        c.update({name: 0 for name in OUTCOMES if name not in c})
        metrics.append(
            {
                "repeat": repeat,
                "mode": mode,
                "event_label": label,
                **dict(c),
                "reference_certain": certain,
                "binary_comparable": binary,
                "binary_coverage": ratio(binary, certain),
                "agreement_over_all_certain": ratio(c["tp"] + c["tn"], certain),
                "agreement_decided_only": ratio(c["tp"] + c["tn"], binary),
                "precision_decided_only": ratio(c["tp"], c["tp"] + c["fp"]),
                "recall_decided_only": ratio(c["tp"], c["tp"] + c["fn"]),
                "f1_decided_only": ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]),
                "positive_recovery_over_reference": ratio(
                    c["tp"], c["reference_positive"]
                ),
                "negative_agreement_over_reference": ratio(
                    c["tn"], c["reference_negative"]
                ),
                "can_assess_false_negatives": c["reference_positive"] > 0,
                "can_assess_false_positives": c["reference_negative"] > 0,
            }
        )
    paired = []
    # Same eligible subset for both modes; differing response coverage must not
    # turn a mode's easier completed subset into a claimed quality improvement.
    lookup = {
        (r["repeat"], r["mode"], r["window_id"], r["event_label"]): r for r in details
    }
    for repeat in range(1, document["repeats"] + 1):
        for label in labels:
            common = []
            for w in manifest["windows"]:
                a = lookup.get((repeat, "events", w["window_id"], label))
                b = lookup.get((repeat, "grounding", w["window_id"], label))
                if a is None or b is None:
                    continue
                if a["outcome"] in {"tp", "tn", "fp", "fn"} and b["outcome"] in {
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                }:
                    common.append((a, b))
            if {p["mode"] for p in document["planned"]} != {"events", "grounding"}:
                continue
            paired.append(
                {
                    "repeat": repeat,
                    "event_label": label,
                    "both_binary_comparable": len(common),
                    "events_agreements": sum(
                        a["outcome"] in {"tp", "tn"} for a, b in common
                    ),
                    "grounding_agreements": sum(
                        b["outcome"] in {"tp", "tn"} for a, b in common
                    ),
                    "prediction_disagreements": sum(
                        a["prediction"] != b["prediction"] for a, b in common
                    ),
                }
            )
    delivery_rows = []
    for (repeat, mode), data in delivery.items():
        latencies = data.pop("latencies")
        delivery_rows.append(
            {
                "repeat": repeat,
                "mode": mode,
                **data,
                "complete_fraction": ratio(data["complete"], data["planned"]),
                "complete_fraction_selected": ratio(data["complete"], data["selected"]),
                "sampling_fraction": ratio(data["selected"], data["planned"]),
                "attempt_latency_p50": quantile(latencies, 0.5),
                "attempt_latency_p95": quantile(latencies, 0.95),
            }
        )
    return {
        "measurement_kind": document["measurement_kind"],
        "state": document["state"],
        "claim": "与 Astra 自动标注的一致性；不是已验证真实准确率",
        "unique_windows": len(manifest["windows"]),
        "metrics": metrics,
        "delivery": delivery_rows,
        "paired": paired,
    }, details


def latency_stats(values):
    return {
        "count": len(values),
        "p50": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values) if values else None,
    }


def timing_summary(planned, records, document):
    n = len(planned)
    phases = (
        "input_release_lag_seconds",
        "dispatch_wait_seconds",
        "admission_wait_seconds",
        "queue_wait_seconds",
        "queue_residence_seconds",
        "inspector_setup_seconds",
        "inspect_seconds",
        "result_finalize_seconds",
        "publication_wait_seconds",
        "input_to_publish_seconds",
        "scheduled_to_publish_seconds",
        "evidence_to_publish_seconds",
    )
    distributions = {}
    for name in phases:
        values = [
            r["latency"][name]
            for r in records
            if r["state"] != "not_sampled"
            if r.get("latency", {}).get(name) is not None
        ]
        distributions[name] = latency_stats(values)
        if name in {
            "queue_wait_seconds",
            "inspect_seconds",
            "input_to_publish_seconds",
        }:
            bounds = (
                (0.05, 0.1, 0.25, 0.5, 1, 2, 3)
                if name == "queue_wait_seconds"
                else (0.5, 1, 1.5, 2, 3, 5, 10, 20)
            )
            previous = -1
            bins = []
            for upper in (*bounds, float("inf")):
                bins.append(
                    {
                        "greater_than": max(0, previous),
                        "up_to_inclusive": upper if math.isfinite(upper) else None,
                        "count": sum(previous < v <= upper for v in values),
                    }
                )
                previous = upper
            distributions[name]["bins"] = bins
    threshold = document.get("input_lag_tolerance_seconds", 0.1) * document.get(
        "simulation_time_scale", 1
    )
    missed = sum(r.get("error_code") == "source_missed" for r in records)
    late = sum(
        r.get("latency", {}).get("input_release_lag_seconds", 0) > threshold
        for r in records
    )
    provider_counts = Counter(
        r.get("response_diagnostics", {}).get("provider_category") or "unreported"
        for r in records
        if r.get("attempted")
    )
    return {
        "complete_result_latency": latency_stats(
            [
                r["latency"]["input_to_publish_seconds"]
                for r in records
                if r["state"] == "complete"
                and r.get("latency", {}).get("input_to_publish_seconds") is not None
            ]
        ),
        "complete_evidence_age": latency_stats(
            [
                r["latency"]["evidence_to_publish_seconds"]
                for r in records
                if r["state"] == "complete"
                and r.get("latency", {}).get("evidence_to_publish_seconds") is not None
            ]
        ),
        "planned": n,
        "selected": sum(p.get("sampling_selected", True) for p in planned),
        "recorded_windows": len(records),
        "attempted_requests": sum(r.get("attempted", False) for r in records),
        "source_missed": missed,
        "late_input_windows": late,
        "input_timing_valid": len(records) == n and not missed and not late,
        "skipped_reasons": dict(
            Counter(r["error_code"] for r in records if r["state"] == "skipped")
        ),
        "provider_counts": dict(provider_counts),
        "phases": distributions,
        "within_seconds": {
            str(t): {
                "count": sum(
                    r["state"] == "complete"
                    and r.get("latency", {}).get("input_to_publish_seconds", math.inf)
                    <= t
                    for r in records
                ),
                "fraction_of_planned": ratio(
                    sum(
                        r["state"] == "complete"
                        and r.get("latency", {}).get(
                            "input_to_publish_seconds", math.inf
                        )
                        <= t
                        for r in records
                    ),
                    n,
                ),
            }
            for t in (1, 2, 3, 5)
        },
    }


def summarize(document, manifest, reference, records):
    streams = sorted({p.get("stream_id", 1) for p in document["planned"]})

    # Qualify window IDs by source for pooled statistics; then restore the
    # canonical evidence ID. Never pair s1/events with s2/grounding.
    def qualified(item):
        return {
            **item,
            "stream_id": item.get("stream_id", 1),
            "window_id": f"{item.get('stream_id', 1)}:{item['window_id']}",
        }

    combined_manifest = {
        **manifest,
        "windows": [
            {**w, "window_id": f"{stream}:{w['window_id']}"}
            for stream in streams
            for w in manifest["windows"]
        ],
    }
    combined_reference = {
        **reference,
        "rows": [
            {**row, "window_id": f"{stream}:{row['window_id']}"}
            for stream in streams
            for row in reference["rows"]
        ],
    }
    combined_doc = {**document, "planned": [qualified(p) for p in document["planned"]]}
    summary, details = _summarize_source(
        combined_doc,
        combined_manifest,
        combined_reference,
        [qualified(r) for r in records],
    )
    summary["unique_windows"] = len(manifest["windows"])
    summary["streams"] = len(streams)
    for row in details:
        row["window_id"] = row["window_id"].split(":", 1)[1]
    summary["per_stream"] = []
    for stream in streams:
        plan = [p for p in document["planned"] if p.get("stream_id", 1) == stream]
        rows = [r for r in records if r.get("stream_id", 1) == stream]
        src_summary, _ = _summarize_source(
            {**document, "planned": [{**p, "stream_id": stream} for p in plan]},
            manifest,
            reference,
            [{**r, "stream_id": stream} for r in rows],
        )
        for delivery in src_summary["delivery"]:
            key = delivery["repeat"], delivery["mode"]
            delivery["timing"] = timing_summary(
                [p for p in plan if (p["repeat"], p["mode"]) == key],
                [r for r in rows if (r["repeat"], r["mode"]) == key],
                document,
            )
        summary["per_stream"].append({"stream_id": stream, **src_summary})
    for delivery in summary["delivery"]:
        key = delivery["repeat"], delivery["mode"]
        delivery["timing"] = timing_summary(
            [p for p in document["planned"] if (p["repeat"], p["mode"]) == key],
            [r for r in records if (r["repeat"], r["mode"]) == key],
            document,
        )
    return summary, details


def render(output, document, manifest, reference, records):
    output = Path(output)
    summary, details = summarize(document, manifest, reference, records)
    write_json(output / "summary.json", summary)
    write_json(output / "comparisons.json", details)
    with (output / "comparisons.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    per_source = [
        {"stream_id": src["stream_id"], **row}
        for src in summary["per_stream"]
        for row in src["metrics"]
    ]
    with (output / "per-stream-quality.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fields = list(dict.fromkeys(k for row in per_source for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_source)
    by_request = {r["request_id"]: r for r in records}
    latency_rows = [
        {
            **p,
            "state": row["state"]
            if row
            else "not_run"
            if p.get("sampling_selected", True)
            else "not_sampled",
            "error_code": row.get("error_code") if row else None,
            "attempted": row.get("attempted", False) if row else False,
            **(row.get("latency", {}) if row else {}),
        }
        for p in document["planned"]
        for row in [by_request.get(p["request_id"])]
    ]
    with (output / "request-latencies.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fields = list(dict.fromkeys(k for row in latency_rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(latency_rows)
    e = lambda v: html.escape(str(v))  # noqa: E731
    percent = lambda v: "不可评估" if v is None else f"{v:.1%}"  # noqa: E731
    number = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    kind = document["measurement_kind"]
    title = (
        "仅准备测试 · 未调用模型"
        if kind == "plan_only"
        else "离线模拟 · 不是模型质量结果"
        if kind == "simulation_not_model_quality"
        else "Qwen 与 Astra 自动标注的一致性"
    )
    table = []
    for m in summary["metrics"]:
        table.append(
            f"<tr><td>{m['repeat']} / {m['mode']}</td>"
            f"<td>{NAMES[m['event_label']]}</td>"
            f"<td>{m.get('reference_positive', 0)} / "
            f"{m.get('reference_negative', 0)} / "
            f"{m.get('reference_uncertain_support', 0)}</td>"
            f"<td>{m['tp']} / {m['tn']}</td>"
            f"<td>{m['fp']} / {m['fn']}</td><td>{m['model_uncertain']}</td>"
            f"<td>{m.get('reference_positive_selected', 0)} / "
            f"{m.get('reference_positive_not_sampled', 0)}</td>"
            f"<td>{m['failed'] + m['incomplete'] + m['not_run'] + m['skipped']}</td>"
            f"<td>{percent(m['binary_coverage'])}</td>"
            f"<td>{percent(m['agreement_over_all_certain'])}</td>"
            f"<td>{percent(m['precision_decided_only'])}</td>"
            f"<td>{percent(m['recall_decided_only'])}</td>"
            f"<td>{percent(m['f1_decided_only'])}</td></tr>"
        )
    delivery = "".join(
        f"<tr><td>{d['repeat']} / {d['mode']}</td>"
        f"<td>{d['complete']}/{d['planned']}</td>"
        f"<td>{d['complete']}/{d['selected']} "
        f"({percent(d['complete_fraction_selected'])})</td>"
        f"<td>{d['not_sampled']}</td>"
        f"<td>{d['failed']}/{d['incomplete']}/{d['skipped']}/{d['not_run']}</td>"
        f"<td>{number(d['attempt_latency_p50'])}</td>"
        f"<td>{number(d['attempt_latency_p95'])}</td></tr>"
        for d in summary["delivery"]
    )
    pair_rows = "".join(
        f"<tr><td>{p['repeat']}</td><td>{NAMES[p['event_label']]}</td>"
        f"<td>{p['both_binary_comparable']}</td><td>{p['events_agreements']}</td>"
        f"<td>{p['grounding_agreements']}</td><td>{p['prediction_disagreements']}</td></tr>"
        for p in summary["paired"]
    )
    timing_rows = []
    for scope, deliveries in [
        ("全部", summary["delivery"]),
        *[
            (f"第 {src['stream_id']} 路", src["delivery"])
            for src in summary["per_stream"]
        ],
    ]:
        for d in deliveries:
            t = d["timing"]
            phase = t["phases"]
            pairs = "".join(
                f"<td>{number(phase[name]['p50'])} / {number(phase[name]['p95'])}</td>"
                for name in (
                    "queue_wait_seconds",
                    "inspect_seconds",
                )
            )
            timing_rows.append(
                f"<tr><td>{scope}</td><td>{d['repeat']}/{d['mode']}</td>"
                f"<td>{d['complete']}/{d['planned']}</td>{pairs}"
                f"<td>{number(t['complete_result_latency']['p50'])} / "
                f"{number(t['complete_result_latency']['p95'])}</td>"
                f"<td>{t['source_missed']}</td>"
                f"<td>{'有效' if t['input_timing_valid'] else '无效或未完成'}</td></tr>"
            )
    timing_html = "".join(timing_rows)
    source_quality = "".join(
        f"<tr><td>{src['stream_id']}</td><td>{m['repeat']}/{m['mode']}</td>"
        f"<td>{NAMES[m['event_label']]}</td><td>{m['fp']}/{m['fn']}</td>"
        f"<td>{m['model_uncertain']}</td>"
        f"<td>{percent(m['binary_coverage'])}</td>"
        f"<td>{percent(m['agreement_over_all_certain'])}</td>"
        f"<td>{percent(m['f1_decided_only'])}</td></tr>"
        for src in summary["per_stream"]
        for m in src["metrics"]
    )
    provider_rows = "".join(
        f"<tr><td>{d['repeat']}/{d['mode']}</td>"
        f"<td>{e(provider)}</td><td>{count}</td></tr>"
        for d in summary["delivery"]
        for provider, count in d["timing"]["provider_counts"].items()
    )
    stage_errors = e(document.get("stop_reason") or "无")
    mode_order = "、".join(document.get("modes", ["events", "grounding"]))
    routing_policy = (
        "直接调用阿里云百炼，不经过 OpenRouter，不回退至其他服务商；"
        if document.get("provider") == "dashscope"
        else f"固定服务商 {e(document['provider_only'])}，禁用服务商回退；"
        if document.get("provider_only")
        else "自动选择低延迟服务商，允许同模型服务商回退；"
    )
    sampling_policy = (
        f"grounding 每 {document['grounding_interval_seconds']} 秒采样一次；"
        "按原视频时间戳以 1 倍速推进，events 保留原证据窗口。"
        "仅使用采样时已到达的最新证据，不提前读取未来帧。完整/选中衡量请求可靠性，完整/全部原窗口衡量覆盖；主动不采样不算成功，也不算模型漏报。"
        if document.get("grounding_interval_seconds")
        else "两模式均选择全部原证据窗口。"
    )
    timeout_policy = (
        "普通请求超时计为失败后继续固定计划，不延长期限、不重试；认证、额度、限流和其他停止规则仍生效。"
        if document.get("continue_on_timeout")
        else "同一路连续三次不成功请求会停止后续测试，超时也计入连续失败。"
    )
    cards = []
    for w in manifest["windows"]:
        rows = [d for d in details if d["window_id"] == w["window_id"]]
        imgs = "".join(
            f'<figure><img loading="lazy" src="inputs/{e(f["path"])}">'
            f"<figcaption>{f['media_seconds']:.3f} 秒</figcaption></figure>"
            for f in w["frames"]
        )
        values = {True: "符合", False: "不符合", None: "不确定"}
        body = "".join(
            f"<tr><td>{r['repeat']}/{r['mode']}/s{r['stream_id']}</td>"
            f"<td>{NAMES[r['event_label']]}</td>"
            f"<td>{values[r['reference']]}</td>"
            "<td>"
            f"{values[r['prediction']] if r['prediction_present'] else '未返回'}"
            "</td>"
            f"<td>{OUTCOMES[r['outcome']]}</td><td>{e(r['request_state'])}</td>"
            f"<td>{e(r['reference_reason'])}</td></tr>"
            for r in rows
        )
        different = any(r["outcome"] in {"fp", "fn"} for r in rows)
        cards.append(
            f'<details class="window" data-difference="{str(different).lower()}">'
            f"<summary>{w['window_id']} · {w['judgment_time_seconds']:.3f} 秒"
            f"{' · 有分歧' if different else ''}</summary>"
            f'<div class="images">{imgs}</div>'
            '<div class="scroll"><table><tr><th>轮次/模式/路</th>'
            "<th>类别</th><th>Astra</th>"
            "<th>Qwen</th><th>比较</th><th>请求状态</th>"
            "<th>Astra 可见证据依据</th></tr>"
            f"{body}</table></div></details>"
        )
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;background:#f5f7fa;color:#182235;
max-width:1450px;margin:24px auto;padding:0 20px}}
header,section,details{{background:#fff;
padding:20px;margin:18px 0;border:1px solid #dbe2ec;border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;
border-bottom:1px solid #e3e7ee;text-align:left}}th{{background:#eef3f9}}
.scroll{{overflow:auto}}.notice{{padding:14px;background:#fff1d6}}.images{{display:grid;
grid-template-columns:repeat(2,1fr)}}figure{{margin:8px}}img{{width:100%}}
summary{{cursor:pointer}}button{{padding:9px 14px}}
code{{overflow-wrap:anywhere}}</style>
<header><h1>{title}</h1><p>模型：{e(document["model"])}
 · 渠道：{e(document.get("provider", "openrouter"))}
 · 状态：{e(document["state"])}</p>
<p class="notice">所有分数只表示与 Astra 自动参考的一致性，不是人工验证的真实准确率。
本次仅测试十二路固定证据工作负载，使用生产共享 admission 队列；
不含摄像头解码、显示或框跟踪。
固定 Qwen3-VL-30B-A3B-Instruct，{routing_policy}
不配置备用模型。实际服务商分类记录在 JSON 中。
输入时序不合格时，保留质量诊断，不认定容量有效。
十二路和多轮都重复同一批证据，不会增加独立事件样本数，也不代表十二个独立场景。</p>
<p>{document["unique_windows"]} 个唯一窗口，每组判断六类事件；{document["repeats"]} 轮，
每阶段十二路，每路 {document["unique_windows"]} 组；
计划最多 {document["planned_requests"]} 次请求。events 使用现有事件接口，
grounding 使用现有
compact 定位接口。</p>
<p>本次实际选择模式：{e(mode_order)}；仅执行所选模式，多模式时第二轮反向。
只选一种模式时，两模式配对表为空。</p>
<p>grounding 传输格式：{e(document.get("grounding_format", "legacy"))}。
精简格式保留六类判断及定位校验；格式缩短不等于已经验证延迟或准确率提升。</p>
<p>本轮超时策略：{timeout_policy}</p>
<p class="notice">采样策略：{sampling_policy}
不能将降频后的返回率称为原负载下的性能提升。</p>
<p>比较原始场景判断 observed_detected。缺框、低置信度、尚未确认不自动改成阴性。
框坐标和定位状态保存在 results.jsonl 中；
没有人物框参考标签，本报告不评价框是否跟对人。</p>
<p><a href="manifest.json">测试参数和版本</a> · <a href="summary.json">统计 JSON</a> ·
<a href="per-stream-quality.csv">逐路质量 CSV</a> ·
<a href="request-latencies.csv">逐窗口延迟 CSV</a> ·
<a href="comparisons.csv">逐项 CSV</a> · <a href="comparisons.json">逐项 JSON</a> ·
<a href="inputs/event-policy.json">共同规则</a></p></header>
<section><h2>请求完成与耗时</h2><p>耗时从 inspect 调用到完成解析，含失败请求实际等待；
此表仅为模型调用耗时；排队及从输入到发布的耗时单列于下方。
不同接口沿用生产的超时与输出长度，参数详见 manifest。</p>
<table><tr><th>轮次/模式</th><th>完整/计划</th>
<th>完整/选中请求（返回率）</th><th>主动未采样</th>
<th>失败/不完整/跳过/未执行</th>
<th>P50 秒</th><th>P95 秒</th></tr>{delivery}</table></section>
<section><h2>排队与逐路延迟</h2><p>停止原因：{stage_errors}。
下表时长为 P50 / P95（秒）。
排队仅针对实际发起请求；调用包括失败等待；
表中输入到发布仅统计完整结果，跳过窗口的等待时间在 JSON 中另列。
这些样本集合不同，分位数不能直接相加。每路完整率分母均保留所有计划窗口。</p>
<p>完整分段、P99、队列跳过原因、延迟直方图和 1/2/3/5 秒累计分布见 summary.json；
3 秒仅为参考阈值。模拟加速时长不属于真实性能。聚合窗口不是独立事件数。</p>
<div class="scroll"><table><tr><th>范围</th><th>轮次/模式</th><th>完整/计划</th>
<th>排队 P50/P95</th><th>模型调用 P50/P95</th><th>完整结果 输入→发布 P50/P95</th>
<th>源窗口错过</th><th>输入时序</th></tr>{timing_html}</table></div></section>
<section><h2>实际服务商记录</h2><p>仅统计已尝试请求，不把跳过/未执行当作某家服务商结果。
未报告分类记为 unreported，未知分类为 other；离线模拟为 simulation。
混合服务商的结果不能用来证明服务商差异对质量或延迟没有影响。</p>
<table><tr><th>轮次/模式</th><th>服务商分类</th><th>已尝试请求数</th></tr>
{provider_rows}</table></section>
<section><h2>逐类一致性</h2><p>Astra 不确定不进入对错评分。
模型不确定、失败、跳过、主动未采样、未执行保留在
“全部明确参考”分母中，并单列；二值覆盖率说明有多少明确参考实际得到了模型是/否判断。</p>
<p>Precision / Recall / F1 仅在双方均明确的子集上计算：TP 为阳性一致，TN 为阴性一致，
FP/FN 为相对 Astra 的疑似误报/漏报，F1=2TP/(2TP+FP+FN)。分母为零显示不可评估，
必须结合覆盖率看。没有阳性无法验证漏报，没有阴性无法验证误报。</p>
<div class="scroll"><table><tr><th>轮次/模式</th><th>类别</th><th>参考阳/阴/不确定</th>
<th>TP/TN</th><th>FP/FN</th><th>模型不确定*</th><th>参考阳性 选中/未采样</th>
<th>失败/缺失*</th><th>二值覆盖率</th>
<th>一致/全部明确参考</th><th>Precision†</th><th>Recall†</th><th>F1†</th></tr>
{"".join(table)}</table></div><p>*仅统计 Astra 明确的项目；
所有请求状态见上表及逐项明细。
†仅双方明确子集。</p></section>
<details><summary>逐路、逐类判断质量</summary>
<p>疑似误报/漏报与不确定仅按明确的 Astra 参考计分；不同路重复同一素材。</p>
<div class="scroll"><table><tr><th>路</th><th>轮次/模式</th><th>类别</th>
<th>FP/FN</th><th>模型不确定</th><th>二值覆盖率</th>
<th>一致/全部明确参考</th><th>F1（双方明确）</th></tr>
{source_quality}</table></div></details>
<section><h2>两模式共同可比较子集</h2><p>
限定 Astra、events、grounding 三者均为明确判断，
避免两模式只比较各自返回成功的不同样本。</p><table><tr><th>轮次</th><th>类别</th>
<th>共同样本</th><th>events 一致数</th><th>grounding 一致数</th><th>两模式分歧数</th>
</tr>{pair_rows}</table></section>
<section><h2>逐窗口证据与判断</h2>
<button id="toggle">只看有明确分歧的窗口</button></section>
{"".join(cards)}<script>let only=false;
document.getElementById('toggle').onclick=function(){{
only=!only;document.querySelectorAll('.window').forEach(x=>x.hidden=only&&x.dataset.difference!=='true');
this.textContent=only?'显示全部窗口':'只看有明确分歧的窗口';}};</script></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
