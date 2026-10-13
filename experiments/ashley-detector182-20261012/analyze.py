"""已保存 outcome 的本地 PR63 增量评分；不启动模型、服务或 GPU。"""

from __future__ import annotations

import importlib.util
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from html import escape
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
ARMS = ("yolo26-s", "rfdetr-l")
OLD_METHODS = ("raw", "sam", "ashley")
ALL_METHODS = (*ARMS, *OLD_METHODS)
PRIOR = ROOT.parent / "main-event-supplement30-ashley1-resume10-20261009"
SCOREABLE = {"answered", "extract_unknown_or_ambiguous", "extract_error_blank", "terminal_failed"}
NUMERIC_KEYS = (
    "Counting_MRA", "Counting_MAE", "Temporal_Duration_MRA", "Temporal_Duration_MAE",
    "Temporal_TimePoint_Accuracy", "Temporal_TimePoint_MAE",
)
ACCURACY_KEYS = (
    "Event_Recognition_Accuracy", "Location_Tracking_Accuracy", "Temporal_TimePoint_Accuracy",
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scoring_module():
    spec = importlib.util.spec_from_file_location("pr63_scoring", ROOT / "scoring/scoring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def save_lines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def decimal_amount(value) -> Decimal | None:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError):
        return None
    return amount if amount.is_finite() and amount >= 0 else None


def finance() -> dict:
    response_path = ROOT / "output/model-responses.jsonl"
    request_path = ROOT / "output/model-requests.jsonl"
    ledger_path = ROOT / "output/attempt-ledger.jsonl"
    baseline_path = ROOT / "baseline-response-ids.json"
    prior_ledger_path = PRIOR / "ashley/output/attempt-ledger.jsonl"
    baseline_ids = set(json.loads(baseline_path.read_text())) if baseline_path.exists() else set()
    prior_call_ids = {row["call_id"] for row in read_jsonl(prior_ledger_path)} if prior_ledger_path.exists() else set()
    responses = read_jsonl(response_path) if response_path.exists() else []
    requests = read_jsonl(request_path) if request_path.exists() else []
    ledger = read_jsonl(ledger_path) if ledger_path.exists() else []
    current_requests = {row["call_id"]: row for row in requests if row["call_id"] not in prior_call_ids}
    current_terminal = {row["call_id"]: row for row in ledger
                        if row["call_id"] not in prior_call_ids and row.get("state") != "dispatching"}
    current_responses = {row["call_id"]: row for row in responses if row["call_id"] not in prior_call_ids}
    by_response_id = {}
    no_cost_receipt = []
    for call_id, row in current_responses.items():
        response = row.get("response") or {}
        response_id = response.get("id")
        if response_id in baseline_ids:
            continue
        amount = decimal_amount((response.get("usage") or {}).get("cost"))
        if not response_id or amount is None:
            no_cost_receipt.append(call_id)
            continue
        if response_id in by_response_id and by_response_id[response_id] != amount:
            raise ValueError(f"同一response.id费用回执冲突: {response_id}")
        by_response_id[response_id] = amount
    no_receipt = sorted(set(current_requests) - set(current_responses))
    unknown_terminal = sorted(call_id for call_id, row in current_terminal.items()
                              if row.get("state") == "outcome_unknown")
    unknown_ids = sorted(set(no_receipt) | set(no_cost_receipt) | set(unknown_terminal))
    gpu_path = ROOT / "gpu/output/gpu-cost.json"
    gpu = json.loads(gpu_path.read_text(encoding="utf-8")) if gpu_path.exists() else {}
    actual = decimal_amount(gpu.get("cumulative_known_charge_usd", gpu.get("actual_charge_usd")))
    estimate = decimal_amount(gpu.get("rental_estimate_usd_not_actual"))
    return {
        "model": {
            "new_dispatches": len(current_requests), "new_receipts": len(current_responses),
            "unique_new_response_ids_with_cost": len(by_response_id),
            "known_usage_cost_usd": str(sum(by_response_id.values(), Decimal(0))),
            "no_receipt_call_ids": no_receipt, "receipt_without_id_or_cost_call_ids": sorted(no_cost_receipt),
            "outcome_unknown_call_ids": unknown_terminal,
            "unknown_fee_attempt_call_ids": unknown_ids,
            "unknown_fee_attempt_count": len(unknown_ids),
            "baseline_response_ids_excluded": len(baseline_ids),
            "prior_call_ids_excluded": len(prior_call_ids),
        },
        "gpu": {
            "actual_provider_snapshot_usd": str(actual) if actual is not None else None,
            "actual_charge_status": gpu.get("charge_status", "not_observed"),
            "current_instance_charge_usd": gpu.get("actual_charge_usd"),
            "prior_instance_snapshots": gpu.get("prior_instance_snapshots", []),
            "scope": "本次检测器对照跨恢复轮的已知供应商收费；空账单或读取失败部分仍未知",
            "rental_estimate_usd_not_actual": str(estimate) if estimate is not None else None,
            "final_invoice": gpu.get("final_invoice", False),
            "source": str(gpu_path) if gpu_path.exists() else None,
            "checked_epoch": gpu.get("checked_epoch"),
            "estimate_excludes_unreported_transfer_and_other_fees": gpu.get(
                "estimate_excludes_unreported_transfer_and_other_fees"),
        },
        "known_model_plus_gpu_actual_usd": str(sum(by_response_id.values(), Decimal(0)) + actual)
        if actual is not None else None,
        "known_total_is_not_final_bill": True,
    }


def times(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        for arm in ARMS:
            result = row["methods"][arm]
            value = result.get("elapsed_seconds")
            if result.get("status") in SCOREABLE and isinstance(value, (int, float)) and value >= 0:
                mode = "geometry_only" if row["geometry_only"] else "event_vlm"
                groups[(arm, mode)].append(float(value))
    details = {}
    for arm in ARMS:
        for mode in ("geometry_only", "event_vlm"):
            values = sorted(groups[(arm, mode)])
            details[f"{arm}/{mode}"] = {
                "completed_with_time": len(values), "sum_seconds": sum(values),
                "mean_seconds": mean(values) if values else None,
                "median_seconds": median(values) if values else None,
                "p95_seconds": values[max(0, int((len(values) - 1) * .95))] if values else None,
                "min_seconds": values[0] if values else None,
                "max_seconds": values[-1] if values else None,
            }
    launch_path, final_path = ROOT / "launch-intent.json", ROOT / "round-outcome.json"
    launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
    final = json.loads(final_path.read_text()) if final_path.exists() else {}
    start, end = launch.get("started_epoch"), final.get("finished_epoch")
    return {
        "per_question": details,
        "whole_round_wall_seconds_including_cold_start_and_cleanup": end - start if start and end else None,
        "whole_round_complete": bool(start and end),
        "meaning": "每题elapsed_seconds为已保存控制器端到端题目用时，不含该组容器冷启动；整轮wall包含启动、冷启动和清理。均非GPU kernel时间。",
    }


def save_svg(name: str, title: str, labels: list[str], series: list[tuple[str, list[float | None], str]],
             maximum: float, unit: str) -> None:
    width, height = 900, max(260, 95 + len(labels) * 62)
    left, right = 230, 100
    span = width - left - right
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="24" y="35" font-family="Arial,sans-serif" font-size="22" fill="#19232d">{escape(title)}</text>',
             f'<line x1="{left}" y1="58" x2="{left}" y2="{height - 28}" stroke="#8293a3"/>']
    for index, label in enumerate(labels):
        y = 82 + index * 62
        parts.append(f'<text x="16" y="{y + 12}" font-family="Arial,sans-serif" font-size="14" fill="#263746">{escape(label)}</text>')
        for series_index, (legend, values, color) in enumerate(series):
            value = values[index]
            if value is None:
                continue
            bar_y = y - 7 + series_index * 18
            length = max(0, min(span, span * value / maximum))
            parts.append(f'<rect x="{left}" y="{bar_y}" width="{length:.2f}" height="14" fill="{color}"/>')
            parts.append(f'<text x="{left + length + 7:.2f}" y="{bar_y + 12}" font-family="Arial,sans-serif" font-size="12" fill="#263746">{value:.1f}{escape(unit)}</text>')
    lx = 24
    for legend, _, color in series:
        parts.append(f'<rect x="{lx}" y="{height - 23}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx + 17}" y="{height - 12}" font-family="Arial,sans-serif" font-size="12" fill="#263746">{escape(legend)}</text>')
        lx += max(135, len(legend) * 11 + 35)
    parts.append('</svg>\n')
    path = REPORT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(parts), encoding="utf-8")
    temporary.replace(path)


def charts(summary: dict, groups: dict, paired: list[dict]) -> None:
    capability_order = ("counting", "location_tracking", "temporal_questions", "event_recognition")
    paired_groups = grouped(paired, "capability")
    labels = [f"{label} (n={paired_groups.get(label, {}).get('question_count', 0)})" for label in capability_order]
    score_series = []
    for arm, color in ((ARMS[0], "#2677b9"), (ARMS[1], "#e28137")):
        values = [100 * paired_groups[label]["methods"][arm]["overall_score_mean"]
                  if label in paired_groups and paired_groups[label]["methods"][arm]["overall_score_mean"] is not None
                  else None for label in capability_order]
        score_series.append((arm, values, color))
    save_svg("能力同题分.svg", "两组共同可评分题：按能力综合分", labels, score_series, 100, "/100")
    counts = [summary["all_methods"][arm]["scored"] for arm in ARMS]
    save_svg("可评分覆盖.svg", "182题每组可评分覆盖", list(ARMS),
             [("可评分", [float(v) for v in counts], "#2677b9"),
              ("未评分", [float(182 - v) for v in counts], "#b8c3cd")], 182, "题")
    timing = summary["time"]["per_question"]
    time_labels = [f"{arm} / {mode}" for arm in ARMS for mode in ("geometry_only", "event_vlm")]
    means = [timing[label]["mean_seconds"] for label in time_labels]
    save_svg("每题端到端时间.svg", "已完成题平均端到端时间（非GPU kernel）", time_labels,
             [("平均秒数", means, "#3a8d70")], max([v for v in means if v is not None] or [1]), "秒")


def classify_failure(status: str, reason: str | None) -> str:
    if status in {"not_started", "pending"}:
        if reason and any(token in reason.lower() for token in ("request", "http", "model", "unknown")):
            return "pending_request_or_response"
        return "pending_incomplete_or_unstarted"
    if status == "terminal_failed":
        return "terminal_request_failure"
    if status == "extract_error_blank":
        return "deterministic_extraction_error"
    if status == "extract_unknown_or_ambiguous":
        return "observation_or_option_ambiguity"
    if status == "answered":
        return "answered"
    return "unrecognized_status"


def new_result(qa_id: str, arm: str, question: dict, truth: dict, scorer) -> dict:
    path = ROOT / "results" / arm / "questions" / qa_id / "outcome.json"
    if not path.exists():
        return {"status": "not_started", "score_status": "unscored", "prediction": None,
                "metrics": None, "reason": None, "failure_kind": "pending_incomplete_or_unstarted", "source": None}
    outcome = json.loads(path.read_text(encoding="utf-8"))
    if outcome.get("qa_id") != qa_id:
        raise ValueError(f"outcome qa_id mismatch: {path}")
    if outcome.get("detector_model") not in (None, arm):
        raise ValueError(f"outcome detector mismatch: {path}")
    status = outcome.get("status", "unknown")
    reason = outcome.get("reason")
    prediction = outcome.get("prediction")
    result = {"status": status, "score_status": "unscored", "prediction": prediction,
              "metrics": None, "reason": reason, "failure_kind": classify_failure(status, reason),
              "source": str(path), "stream_session_id": outcome.get("stream_session_id"),
              "journal_path": outcome.get("journal_path") or outcome.get("capture_path"),
              "elapsed_seconds": outcome.get("elapsed_seconds")}
    if status not in SCOREABLE:
        return result
    document = {"capability": question["capability"],
                "public_context_json": json.dumps(question.get("public_context") or {}, ensure_ascii=False),
                "ground_truth_json": json.dumps(truth, ensure_ascii=False)}
    result["metrics"] = scorer.benchmark_v1_process_results(document, [str(prediction or "")])
    result["score_status"] = "scored"
    return result


def summarize(question_rows: list[dict], method: str) -> dict:
    states = Counter(row["methods"][method]["status"] for row in question_rows)
    scored = [row["methods"][method]["metrics"] for row in question_rows
              if row["methods"][method]["score_status"] == "scored"]
    summary = {
        "total_questions": len(question_rows), "scored": len(scored),
        "unscored": len(question_rows) - len(scored), "statuses": dict(states),
        "overall_score_sum": sum(m["Overall_Score"] for m in scored),
        "overall_score_mean": (sum(m["Overall_Score"] for m in scored) / len(scored)) if scored else None,
    }
    for key in (*NUMERIC_KEYS, *ACCURACY_KEYS):
        values = [m[key] for m in scored if key in m]
        if not values:
            continue
        summary[key] = {"sum": sum(values), "denominator": len(values),
                        "mean": sum(values) / len(values)}
        if key in ACCURACY_KEYS:
            summary[key]["correct"] = sum(value == 1.0 for value in values)
    return summary


def grouped(rows: list[dict], key: str) -> dict:
    values = defaultdict(list)
    for row in rows:
        raw = row.get(key)
        labels = raw if isinstance(raw, list) else [raw]
        for label in labels:
            values[str(label if label not in (None, "") else "未标明")].append(row)
    return {label: {"question_count": len(group),
                    "methods": {method: summarize(group, method) for method in ALL_METHODS}}
            for label, group in sorted(values.items())}


def fmt_score(summary: dict) -> str:
    mean = summary["overall_score_mean"]
    return f"{mean * 100:.2f}/100（{summary['scored']}/{summary['total_questions']}）" if mean is not None else "未评分"


def fmt_metric(summary: dict, key: str) -> str:
    metric = summary.get(key)
    if not metric:
        return "—"
    if key in ACCURACY_KEYS:
        return f"{metric['correct']}/{metric['denominator']}（{metric['mean'] * 100:.1f}%）"
    return f"{metric['mean']:.3f}（n={metric['denominator']}）"


def markdown(rows: list[dict], summary: dict, groups: dict, paired: list[dict]) -> str:
    lines = ["# 检测器182题正式离线增量报告", "",
             f"生成时间：{summary['generated_at']}。评分版本：Numaira-Technology/goodvision PR63，合并提交 `253935fc944fd8ff05ae84c1f3a987c035141578`。",
             "", "两款新方法均以Ashley原生确定性提取器答题：90题几何关闭VLM，92题事件使用Qwen3-VL-8B原生观察；每组182题。旧Raw、SAM及Ashley为历史输入和版本，PR63仅对保存的旧原答离线重评分。新方法间只用两边均可评分的同题集合比较；历史分数单列，不据此推断检测器因果效应。", "",
             "未评分状态（包括 pending 和尚未启动）不填零；已完整观察但提取空答或终局失败按固定评分规则计入零分。数值题 MRA 的分母为该类可评分题；MAE 只在有效数值回答上计算，并标明有效分母。", "",
             "## 进度与同题比较", "",
             "| 方法 | 全量可评分 / 182 | 全量综合分 | 新组共同题综合分 | 未评分 |", "|---|---:|---:|---:|---:|"]
    pair_stats = summary["paired_methods"]
    for method in ALL_METHODS:
        all_stat = summary["all_methods"][method]
        lines.append(f"| {method} | {all_stat['scored']}/182 | {fmt_score(all_stat)} | {fmt_score(pair_stats[method])} | {all_stat['unscored']} |")
    lines += ["", f"新组共同可评分题：{len(paired)}/182。先行20题计入全量，不重复计数。各分层表按每种方法各自可评分题汇总，括号内分母可能不同；只有共同题列和共同题能力图是严格同题比较。", "",
              "## 四能力", "", "| 能力 | 题数 | YOLO26-S | RF-DETR-L | 旧Raw | 旧SAM | 旧Ashley |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, item in groups["capability"].items():
        cells = " | ".join(fmt_score(item["methods"][method]) for method in ALL_METHODS)
        lines.append(f"| {label} | {item['question_count']} | {cells} |")
    lines += ["", "## 数值题与选择题分母", "",
              "| 方法 | 计数MRA | 计数有效MAE | 时长MRA | 时长有效MAE | 时间点正确 | 时间点有效MAE | 事件正确 | 位置正确 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    metric_order = ("Counting_MRA", "Counting_MAE", "Temporal_Duration_MRA", "Temporal_Duration_MAE",
                    "Temporal_TimePoint_Accuracy", "Temporal_TimePoint_MAE", "Event_Recognition_Accuracy",
                    "Location_Tracking_Accuracy")
    for method in ALL_METHODS:
        item = summary["all_methods"][method]
        lines.append("| " + method + " | " + " | ".join(fmt_metric(item, key) for key in metric_order) + " |")
    cost = summary["cost"]
    model_cost, gpu_cost = cost["model"], cost["gpu"]
    lines += ["", "## 费用与耗时", "",
              f"模型新请求：{model_cost['new_dispatches']}次；回执：{model_cost['new_receipts']}次；按新 `response.id` 去重且排除冻结基线后的已知 `usage.cost` 为 {model_cost['known_usage_cost_usd']} 美元（{model_cost['unique_new_response_ids_with_cost']} 个唯一响应）。无回执/无费用/结局未知的请求合并去重为 {model_cost['unknown_fee_attempt_count']} 次，费用未知，不当作零。",
              f"GPU供应商实际收费快照：{gpu_cost['actual_provider_snapshot_usd'] if gpu_cost['actual_provider_snapshot_usd'] is not None else '未知'} 美元，状态 `{gpu_cost['actual_charge_status']}`；按报价估算的租用金额：{gpu_cost['rental_estimate_usd_not_actual'] if gpu_cost['rental_estimate_usd_not_actual'] is not None else '未知'} 美元。两者分别记录，收费快照不是最终发票；已知模型费加实际GPU快照为 {cost['known_model_plus_gpu_actual_usd'] if cost['known_model_plus_gpu_actual_usd'] is not None else '未能合计'} 美元，仍不代表完整账单。",
              "", "| 检测器 / 路径 | 有用时记录题数 | 平均端到端秒 | 中位秒 | P95秒 |", "|---|---:|---:|---:|---:|"]
    for key, item in summary["time"]["per_question"].items():
        def shown(value):
            return f"{value:.2f}" if value is not None else "—"
        lines.append(f"| {key} | {item['completed_with_time']} | {shown(item['mean_seconds'])} | {shown(item['median_seconds'])} | {shown(item['p95_seconds'])} |")
    whole = summary["time"]["whole_round_wall_seconds_including_cold_start_and_cleanup"]
    lines += ["", f"整轮控制器实际墙钟时长（含冷启动与资源清理）：{whole:.1f}秒。" if whole is not None else "整轮尚未结束，冷启动与资源清理后的总墙钟时长待定。",
              "每题 `elapsed_seconds` 是控制器端到端题目用时，不含该组容器冷启动；整轮墙钟包含部署及清理。这些都不是GPU kernel纯推理延迟。",
              "", "![共同题能力分](能力同题分.svg)", "", "![可评分覆盖](可评分覆盖.svg)", "", "![每题端到端时间](每题端到端时间.svg)"]
    for key, title in (("source_dataset", "数据源"), ("question_form", "问法"),
                       ("event_label", "事件或目标"), ("event_polarity", "事件正负")):
        lines += ["", f"## 按{title}拆分", "",
                  "| 类别 | 题数 | YOLO26-S | RF-DETR-L | 旧Raw | 旧SAM | 旧Ashley |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for label, item in groups[key].items():
            cells = " | ".join(fmt_score(item["methods"][method]) for method in ALL_METHODS)
            lines.append(f"| {label.replace('|', '/')} | {item['question_count']} | {cells} |")
    lines += ["", "## 失败与证据边界", "",
              "下表区分请求/响应未决、观察不完整、确定性提取失败及选项歧义；具体题号、GT、原答、文件路径与每题分数见 `逐题PR63.jsonl`。未检查原视频或标注，不把零分直接归因于视觉识别。", "",
              "| 方法 | 状态或原因类别 | 题数 |", "|---|---|---:|"]
    for method in ARMS:
        counts = Counter(row["methods"][method]["failure_kind"] for row in rows)
        for kind, number in sorted(counts.items()):
            lines.append(f"| {method} | {kind} | {number} |")
    actual = defaultdict(list)
    for row in rows:
        for method in ARMS:
            item = row["methods"][method]
            if item["source"] and item["status"] != "answered":
                actual[(method, item["status"], str(item.get("reason") or "未记录具体原因"))].append(row["qa_id"])
    lines += ["", "### 已保存结果中的具体原因", ""]
    if actual:
        lines += ["| 方法 | 状态 | 原始原因 | 题数 | 题号示例 |",
                  "|---|---|---|---:|---|"]
        for (method, status, reason), ids in sorted(actual.items()):
            lines.append(f"| {method} | {status} | {reason.replace('|', '/')} | {len(ids)} | {', '.join(ids[:5])} |")
    else:
        lines.append("当前尚无已保存的非回答结果。")
    lines += ["", "数据文件：`分类汇总.json` 包含每个分层的各指标分母；`逐题PR63.jsonl` 保留同题旧分和原始状态。历史Ashley未评分9题仍保留为未评分。", ""]
    return "\n".join(lines)


def main() -> None:
    public = read_jsonl(ROOT / "public/questions.jsonl")
    private = {row["qa_id"]: row["ground_truth"] for row in read_jsonl(ROOT / "private/answers.jsonl")}
    old = {(row["qa_id"], row["method"]): row for row in read_jsonl(ROOT / "scoring/old-pr63-rescore.jsonl")}
    metadata_path = ROOT.parent / "main-event-supplement30-20261006/report/182题逐题记录.json"
    metadata = {row["qa_id"]: row for row in json.loads(metadata_path.read_text(encoding="utf-8"))}
    if len(public) != 182 or set(private) != {row["qa_id"] for row in public}:
        raise ValueError("182题公开与私有答案集合不一致")
    scorer = scoring_module()
    rows = []
    for question in public:
        qa_id = question["qa_id"]
        meta = metadata[qa_id]
        methods = {arm: new_result(qa_id, arm, question, private[qa_id], scorer) for arm in ARMS}
        for method in OLD_METHODS:
            item = old[(qa_id, method)]
            methods[method] = {"status": item["status"],
                               "score_status": item["rescore_status"],
                               "prediction": item["prediction"],
                               "metrics": item["pr63_metrics"],
                               "old_metrics": item["old_metrics"],
                               "source": item["old_source"],
                               "reason": None, "failure_kind": "historical_record"}
        event_label = meta.get("event_or_target") or question.get("target_label")
        rows.append({"qa_id": qa_id, "question": question["question"],
                     "response_options": question.get("response_options"),
                     "ground_truth": private[qa_id], "capability": question["capability"],
                     "question_form": question["question_form"],
                     "source_dataset": question["source_dataset"],
                     "source_group": question.get("source_group"),
                     "event_label": event_label, "event_polarity": meta.get("event_polarity"),
                     "video_id": question["video_id"],
                     "geometry_only": question["capability"] in {"counting", "location_tracking"}
                     or question["question_form"] in {"first_exact_visible_count", "last_exact_visible_count", "longest_at_or_above_threshold"},
                     "methods": methods})
    paired = [row for row in rows if all(row["methods"][arm]["score_status"] == "scored" for arm in ARMS)]
    categories = {key: grouped(rows, key) for key in
                  ("capability", "source_dataset", "question_form", "event_label", "event_polarity")}
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scorer": "Numaira-Technology/goodvision PR63 merge 253935fc944fd8ff05ae84c1f3a987c035141578",
        "model": {"yolo26-s": "Qwen3-VL-8B only for event path",
                  "rfdetr-l": "Qwen3-VL-8B only for event path"},
        "scope": {"per_arm": 182, "geometry_only": 90, "event_vlm": 92,
                  "pilot_included": 20, "paired_new_scored": len(paired)},
        "all_methods": {method: summarize(rows, method) for method in ALL_METHODS},
        "paired_methods": {method: summarize(paired, method) for method in ALL_METHODS},
        "new_pair_qa_ids": [row["qa_id"] for row in paired],
        "cost": finance(), "time": times(rows),
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    save_lines(REPORT / "逐题PR63.jsonl", rows)
    save(REPORT / "分类汇总.json", {"summary": summary, "groups": categories})
    charts(summary, categories, paired)
    text = markdown(rows, summary, categories, paired)
    temp = REPORT / "完整分析.md.tmp"
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(REPORT / "完整分析.md")
    print(json.dumps({"paired_new_scored": len(paired),
                      "new_scored": {arm: summary["all_methods"][arm]["scored"] for arm in ARMS},
                      "report": str(REPORT / "完整分析.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
