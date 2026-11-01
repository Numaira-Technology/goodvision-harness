"""Offline delivery and quality report; fixed denominators include every loss."""

import html
import json
import math
from collections import Counter
from pathlib import Path

from tools.vlm_benchmark.accounting import complete, delivery_counts
from tools.vlm_benchmark.model_comparison import quality, safe_path
from tools.vlm_benchmark.report import write_json
from tools.vlm_benchmark.vision_timeliness import assess_timeliness
from tools.vlm_multistream_benchmark import report_lane, timing

from . import profile as p


def distribution(values):
    values = [
        v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0
    ]
    return {"count": len(values), **timing(values)}


def astra_agreement(raw, reference):
    rows, unaligned = [], 0
    for item in raw:
        report = item["report"]
        if report["event_policy"]["sha256"] != p.POLICY_SHA256:
            raise ValueError("result policy mismatch")
        if not complete(report):
            continue
        evidence = report["evidence"]
        ticks = tuple(e["frame_sequence"] for e in evidence)
        wid = reference["lookup"].get(ticks)
        if wid is None or any(
            type(e["frame_sequence"]) is not int
            or not math.isfinite(e["timestamp_seconds"])
            or abs(e["timestamp_seconds"] - e["frame_sequence"] / 2.5) > 1e-6
            for e in evidence
        ):
            unaligned += 1
            continue
        judgments = report["judgments"]
        predictions = {j["event_label"]: j["observed_detected"] for j in judgments}
        if len(judgments) != 6 or set(predictions) != set(p.LABELS):
            raise ValueError("invalid six-label result")
        for label, pred in predictions.items():
            truth = reference["refs"][wid]["judgments"][label]["detected"]
            if type(pred) not in (bool, type(None)):
                raise ValueError("invalid prediction")
            outcome = (
                "reference_uncertain"
                if truth is None
                else "model_uncertain"
                if pred is None
                else "tp"
                if truth and pred
                else "fn"
                if truth
                else "fp"
                if pred
                else "tn"
            )
            rows.append({"outcome": outcome, "label": label})
    return {
        **quality(rows),
        "unaligned_complete_windows": unaligned,
        "per_label": {
            label: quality([r for r in rows if r["label"] == label])
            for label in p.LABELS
        },
    }


def audit_rows(root, job=None):
    root = Path(root)
    folders = (
        [safe_path(root, job["id"]) / "api-audit"]
        if job
        else [root / "api-audit", *root.glob("*/api-audit")]
    )
    return [
        json.loads(line)
        for folder in folders
        for path in sorted(folder.glob("*.jsonl"))
        for line in path.read_text().splitlines()
    ]


def audit_coverage(records, audit):
    """Match actual sends by identity; equal totals alone can hide lost requests."""

    def identity(row):
        # A parallel window deliberately shares its request ID across branches.
        # Older events-only audits predate this field and default to events.
        if row.get("vlm_lane") not in {
            None,
            "events",
            "information",
            "grounding",
            "combined",
        }:
            return None
        key = (
            row.get("stream_session_id"),
            row.get("request_id"),
            report_lane(row),
        )
        return key if all(isinstance(v, str) and v for v in key) else None

    attempted = [
        r for r in records if r["timing"].get("http_request_attempted") is True
    ]
    sent = [r for r in audit if r.get("http_request_attempted") is True]
    expected, observed = Counter(map(identity, attempted)), Counter(map(identity, sent))
    by_id = {identity(row): row for row in sent}
    mismatched = sum(
        complete(r) != by_id[identity(r)].get("complete")
        or (
            r["timing"].get("http_status") is not None
            and r["timing"]["http_status"] != by_id[identity(r)].get("http_status")
        )
        for r in attempted
        if identity(r) is not None and identity(r) in by_id
    )
    result = {
        "expected_attempts": len(attempted),
        "audited_attempts": len(sent),
        "missing_attempts": sum((expected - observed).values()),
        "unexpected_attempts": sum((observed - expected).values()),
        "duplicate_identities": sum(v - 1 for v in observed.values() if v > 1)
        + sum(v - 1 for v in expected.values() if v > 1),
        "invalid_identities": expected[None] + observed[None],
        "unknown_attempt_records": sum(
            type(r["timing"].get("http_request_attempted")) is not bool for r in records
        ),
        "mismatched_outcomes": mismatched,
    }
    result["valid"] = all(
        result[k] == 0
        for k in (
            "missing_attempts",
            "unexpected_attempts",
            "duplicate_identities",
            "invalid_identities",
            "unknown_attempt_records",
            "mismatched_outcomes",
        )
    )
    return result


def audit_usage(root, records=None):
    audit = audit_rows(root)
    sent = [r for r in audit if r["http_request_attempted"] is True]
    known = [
        r["input_tokens"]
        for r in sent
        if type(r.get("input_tokens")) is int and r["input_tokens"] >= 0
    ]
    coverage = audit_coverage(records, audit) if records is not None else None
    usage_complete = len(known) == len(sent) and (
        coverage["valid"] if coverage is not None else bool(sent)
    )
    known_cost = sum(known) * p.BASE_INPUT_PRICE_PER_MILLION / 1e6
    return {
        "scope": "all retained HTTP attempts including preflight and warmups",
        "attempts": len(sent),
        "usage_reported_requests": len(known),
        "usage_missing_requests": len(sent) - len(known),
        "reported_input_tokens": sum(known),
        "estimated_base_cost_usd": known_cost if usage_complete else None,
        "reported_usage_base_cost_usd": known_cost,
        "usage_complete": usage_complete,
        "audit_coverage": coverage,
        "billed_cost_verified": False,
        "http_statuses": dict(
            Counter(str(r.get("http_status", "unknown")) for r in sent)
        ),
        "error_codes": dict(
            Counter(
                r.get("error_code", "unreported")
                for r in sent
                if r.get("http_status", 0) >= 400
            )
        ),
        "openai_request_ids_retained": sum(
            "openai_request_id" in r.get("headers", {}) for r in sent
        ),
        "openai_processing_ms": distribution(
            [r.get("headers", {}).get("openai-processing-ms") for r in sent]
        ),
        "limitations": "Missing usage is unknown, not zero. "
        "Estimate excludes pricing premiums. "
        "Processing headers are server-reported, not pure GPU inference.",
    }


def analyze(root, reference):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (
        any(manifest["settings"].get(k) != v for k, v in p.FROZEN.items())
        or manifest["settings"].get("real") is not True
        or manifest.get("decisions_profile") != p.VERSION
        or manifest.get("transport") != "decisions-json-v1"
        or manifest["event_policy"]["sha256"] != p.POLICY_SHA256
        or reference["policy"]["sha256"] != p.POLICY_SHA256
        or manifest["decoded_evidence"] != [reference["decoded_evidence"]]
        or [v["sha256"] for v in manifest["videos"]] != [p.VIDEO_SHA256]
        or manifest.get("http_timeouts") != {"events": 20}
    ):
        raise ValueError("not the frozen real Decisions run or evidence mismatch")
    trials, seen, all_records = [], set(), []
    for job in summary["jobs"]:
        raw = [
            json.loads(line)
            for line in safe_path(root, job["raw_reports"]).read_text().splitlines()
        ]
        records = [i["report"] for i in raw]
        all_records.extend(records)
        if job["phase"] != "measurement":
            continue
        if (
            job["mode"] != "events"
            or job["repeat"] not in (1, 2)
            or job["repeat"] in seen
            or job["streams"] != 12
        ):
            raise ValueError("unexpected or duplicate Decisions trial")
        seen.add(job["repeat"])
        good = [r for r in records if complete(r)]
        stage = job["result"]
        if (
            set(stage["lanes"]) != {"events"}
            or stage["lanes"]["events"]["expected_observations"] != 828
            or stage["lanes"]["events"]["reports"] != len(records)
            or stage["lanes"]["events"]["complete_valid_results"] != len(good)
        ):
            raise ValueError("raw records do not reconcile with trial summary")
        delivery = delivery_counts(records, 828)
        coverage = audit_coverage(records, audit_rows(root, job))
        timed = assess_timeliness(raw, {"events"})["lanes"]["events"]
        valid = bool(
            stage["all_drained"]
            and stage["input_timing_valid"]
            and stage["result_records_missing"] == 0
            and stage.get("result_retention", {}).get("valid") is True
            and stage["shared_admission"]["active"] == 0
            and stage["shared_admission"]["waiting"] == 0
            and job.get("recording", {}).get("complete")
            and coverage["valid"]
        )
        token_values = [
            r["timing"].get("http_timing", {}).get("prompt_tokens") for r in records
        ]
        tokens = [v for v in token_values if type(v) is int and v >= 0]
        span = stage["input_schedule_span_seconds"] + stage["drain_elapsed_seconds"]
        trials.append(
            {
                "mode": "events",
                "repeat": job["repeat"],
                "valid": valid,
                "planned": 828,
                "delivery": delivery,
                "audit_coverage": coverage,
                "sent_rate": delivery["http_attempted"] / 828,
                "complete_rate": len(good) / 828,
                "complete_among_sent": delivery["http_attempted_complete"]
                / delivery["http_attempted"]
                if delivery["http_attempted"]
                else None,
                "within_three_seconds": timed,
                "latency_complete_results": distribution(
                    [r["timing"].get("source_to_publish_seconds") for r in good]
                ),
                "http_latency_all_attempts": distribution(
                    [
                        r["timing"].get("http_timing", {}).get("total_seconds")
                        for r in records
                        if r["timing"].get("http_request_attempted") is True
                    ]
                ),
                "admission_wait_all_records": distribution(
                    [r["timing"].get("admission_wait_seconds") for r in records]
                ),
                "http_429": sum(r["timing"].get("http_status") == 429 for r in records),
                "tokens": {
                    "reported_input_tokens": sum(tokens),
                    "reported_requests": len(tokens),
                    "input_tokens_per_second": sum(tokens) / span if span > 0 else None,
                    "complete_windows_per_second": len(good) / span
                    if span > 0
                    else None,
                    "accounting_span_seconds": span,
                    "output_tps": None,
                    "output_tps_reason": "native classification, not generated text",
                },
                "astra": astra_agreement(raw, reference),
                "recording": job.get("recording"),
            }
        )
    usage = audit_usage(root, all_records)
    report = {
        "profile": p.VERSION,
        "state": summary["state"],
        "trials": trials,
        "all_two_trials": seen == {1, 2},
        "passes_target": (
            summary["state"] == "completed"
            and seen == {1, 2}
            and usage["audit_coverage"]["valid"]
            and all(
                t["valid"] and t["within_three_seconds"]["all_cameras_pass"]
                for t in trials
            )
        ),
        "reference_sha256": reference["reference_sha256"],
        "usage": usage,
        "limitations": [
            "Astra agreement is not human-confirmed accuracy or person-label accuracy.",
            "No boxes or person IDs are produced; this is a scene-events benchmark.",
            "Latency includes local queue, HTTP and publication; not pure inference.",
            "Historical Qwen routing, timeouts and input layout may differ.",
        ],
    }
    write_json(root / "decisions-comparison.json", report)
    coverage = usage["audit_coverage"]
    estimated_cost = usage["estimated_base_cost_usd"]
    cost_text = (
        f"${estimated_cost:.6f}"
        if estimated_cost is not None
        else "未知（日志或用量不全）"  # noqa: RUF001
    )
    rows = []
    for t in trials:
        latency = t["latency_complete_results"]
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(v))}</td>"
                for v in (
                    t["repeat"],
                    f"{t['sent_rate']:.1%}",
                    f"{t['complete_rate']:.1%}",
                    f"{t['within_three_seconds']['on_time_coverage']:.1%}",
                    latency.get("p50"),
                    latency.get("p95"),
                    t["http_429"],
                    f"{t['astra']['agreement']:.1%}"
                    if t["astra"]["agreement"] is not None
                    else "N/A",
                    t["astra"]["binary_comparable"],
                    t["valid"],
                )
            )
            + "</tr>"
        )
    (root / "decisions-comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Decisions 十二路结果</title>'
        "<style>body{font:16px system-ui;margin:40px}"
        "td,th{padding:12px;border:1px solid #ccc}"
        "table{border-collapse:collapse}</style>"
        "<h1>OpenAI Decisions · 十二路 events</h1>"
        f"<p>测试状态：{html.escape(str(report['state']))}；"  # noqa: RUF001
        f"逐路三秒目标：{'达标' if report['passes_target'] else '未通过验收'}。</p>"  # noqa: RUF001
        f"<p>请求日志核对：{'完整' if coverage['valid'] else '不完整或不一致'}；"  # noqa: RUF001
        f"结果记载 {coverage['expected_attempts']} 次，"  # noqa: RUF001
        f"请求日志记载 {coverage['audited_attempts']} 次，"  # noqa: RUF001
        f"缺失 {coverage['missing_attempts']} 次。详情见 JSON。</p>"
        f"<p>整批基价估算：{cost_text}；"  # noqa: RUF001
        f"已知用量小计 ${usage['reported_usage_base_cost_usd']:.6f}。"
        "小计不代表全部费用，估算不是已核对账单。</p>"  # noqa: RUF001
        "<p>每轮 828 个计划窗口。失败、未发出、迟到都保留在分母。"
        "输出 TPS 不适用。Astra 一致率不等于人工准确率。</p>"
        "<table><tr><th>轮次</th><th>发出率</th><th>整体完整率</th><th>三秒内完整率</th>"
        "<th>P50 秒</th><th>P95 秒</th><th>429</th><th>Astra 一致率</th>"
        "<th>可比判断数</th><th>运行有效</th></tr>"
        + "".join(rows)
        + '</table><p><a href="report.html">原始报告与录屏</a> · '
        '<a href="decisions-comparison.json">全部统计和 Astra 对比</a></p>',
        encoding="utf-8",
    )
    return report
