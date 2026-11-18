"""Results table for the latency investigation.

Columns: run date, location, GPU, CPU model, vLLM version, concurrency,
input-size bucket, P50, P95, P95/P50, then the same three numbers measured
from actual send, then notes. There is no P99 column.

Closed loop (fixed 12-way concurrency): both clocks are actual send.
Open loop (recorded intervals): the primary P50/P95 add AIPerf's schedule lag
back onto dispatch latency, so the clock starts at the scheduled send. The
actual-send numbers stay in the secondary columns.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from tools.latency_investigation.percentiles import p50, p95, ratio, token_bucket

COLUMNS = (
    "run_date",
    "location",
    "gpu",
    "cpu_model",
    "vllm_version",
    "concurrency",
    "input_size_bucket",
    "p50_seconds",
    "p95_seconds",
    "p95_over_p50",
    "p50_actual_send_seconds",
    "p95_actual_send_seconds",
    "p95_over_p50_actual_send",
    "notes",
)


def rows_from_vllm_bench(
    result: dict, manifest: dict, *, host: dict | None = None
) -> list[dict]:
    """Join ``--save-detailed`` latencies to the manifest in file order.

    ``--disable-shuffle`` is required for that order to match. Latencies are
    the tool's E2EL from actual send.
    """
    latencies = result.get("latencies")
    texts = result.get("generated_texts") or []
    output_lens = result.get("output_lens") or []
    requests = manifest.get("requests") or []
    if not isinstance(latencies, list) or len(latencies) != len(requests):
        raise ValueError("bench latencies do not match the manifest order")
    errors = result.get("errors")
    if not requests or not isinstance(errors, list) or len(errors) != len(requests):
        raise ValueError("bench success metadata does not match the manifest")
    if (
        type(result.get("completed")) is not int
        or type(result.get("failed")) is not int
        or result["completed"] != len(requests)
        or result["failed"] != 0
        or any(error != "" for error in errors)
    ):
        raise ValueError("bench report requires all manifest requests to succeed")
    if any(not _valid_seconds(value) for value in latencies):
        raise ValueError("bench latencies must be finite non-negative seconds")
    host = host or {}
    samples = []
    for index, item in enumerate(requests):
        samples.append(
            {
                "latency": latencies[index],
                "latency_actual": latencies[index],
                "bucket": item.get("bucket") or token_bucket(item.get("prompt_tokens")),
                "output_text": texts[index] if index < len(texts) else None,
                "output_tokens": output_lens[index]
                if index < len(output_lens)
                else None,
                "baseline_output_text": item.get("baseline_output_text"),
                "request_id": item.get("request_id"),
            }
        )
    return _bucket_rows(
        samples, result=result, manifest=manifest, host=host, clock="actual_send"
    )


def rows_from_aiperf_jsonl(
    path, manifest: dict, *, host: dict | None = None
) -> list[dict]:
    """Read AIPerf ``profile_export.jsonl``.

    ``request_latency`` is dispatch-to-last-content. The primary table adds
    the least-late schedule lag so the clock starts at the scheduled send.
    The actual-send numbers stay in the secondary columns.
    """
    path = Path(path)
    requests = manifest.get("requests") or []
    by_session = {item.get("request_id"): item for item in requests}
    samples = []
    record_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_count += 1
        if record.get("error"):
            continue
        metrics = record.get("metrics") or {}
        latency = _metric_seconds(metrics.get("request_latency"))
        if latency is None:
            continue
        metadata = record.get("metadata") or {}
        session = metadata.get("conversation_id") or metadata.get("session_id")
        item = by_session.get(session, {})
        start_ns = metadata.get("request_start_ns")
        scheduled = item.get("arrival_offset_seconds")
        samples.append(
            {
                "latency": latency,
                "bucket": item.get("bucket") or "unknown",
                "output_tokens": _metric_number(metrics.get("output_token_count")),
                "output_text": record.get("response_text"),
                "baseline_output_text": item.get("baseline_output_text"),
                "request_id": item.get("request_id") or session,
                "start_ns": start_ns,
                "scheduled_offset_s": scheduled,
            }
        )
    if not samples:
        raise ValueError("AIPerf run has no successful requests with valid latency")
    if (
        record_count != len(requests)
        or len(samples) != len(requests)
        or len(by_session) != len(requests)
        or {sample["request_id"] for sample in samples} != set(by_session)
    ):
        raise ValueError(
            "AIPerf report requires every manifest request exactly once and successful"
        )
    anchored = _add_schedule_lag(samples)
    return _bucket_rows(
        samples,
        result={"date": None, "max_concurrency": 12},
        manifest=manifest,
        host=host or {},
        clock="scheduled_send" if anchored else "actual_send_lag_unavailable",
    )


def _add_schedule_lag(samples) -> bool:
    """Add AIPerf's least-late schedule lag onto dispatch latency.

    ``lag_i = (start_i - scheduled_i) - min_j(start_j - scheduled_j)``.
    ``start`` is ``request_start_ns``. ``scheduled`` is the dataset offset.
    The minimum is the same anchor AIPerf uses because it does not record
    schedule zero. Returns False when a start time or offset is missing.
    """
    if not samples:
        return False
    offsets = []
    for sample in samples:
        start = sample.get("start_ns")
        scheduled = sample.get("scheduled_offset_s")
        if (
            isinstance(start, bool)
            or isinstance(scheduled, bool)
            or not isinstance(start, (int, float))
            or not isinstance(scheduled, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(scheduled)
        ):
            for item in samples:
                item["latency_actual"] = item["latency"]
            return False
        offsets.append(float(start) - float(scheduled) * 1e9)
    anchor = min(offsets)
    for sample, offset in zip(samples, offsets, strict=True):
        lag = (offset - anchor) / 1e9
        sample["latency_actual"] = sample["latency"]
        sample["lag_seconds"] = lag
        sample["latency"] = sample["latency"] + lag
    return True


def _valid_seconds(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _metric_seconds(metric):
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    if not _valid_seconds(value):
        return None
    unit = metric.get("unit")
    if unit == "ms":
        return value / 1000.0
    if unit in {"s", "sec", "seconds"}:
        return float(value)
    return None


def _metric_number(metric):
    if isinstance(metric, dict):
        metric = metric.get("value")
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        return None
    return metric


def _bucket_rows(samples, *, result, manifest, host, clock) -> list[dict]:
    groups = {"overall": samples}
    for sample in samples:
        groups.setdefault(sample["bucket"], []).append(sample)
    order = ["overall", "<500", "500-999", ">=1000", "unknown"]
    notes = _notes(samples, manifest, clock)
    rows = []
    for name in order:
        group = groups.get(name)
        if not group:
            continue
        latencies = [sample["latency"] for sample in group]
        actual = [sample.get("latency_actual", sample["latency"]) for sample in group]
        p50_value = p50(latencies)
        p95_value = p95(latencies)
        p50_actual = p50(actual)
        p95_actual = p95(actual)
        rows.append(
            {
                "run_date": host.get("run_date") or result.get("date") or "",
                "location": host.get("location") or "",
                "gpu": host.get("gpu") or "",
                "cpu_model": host.get("cpu_model") or "",
                "vllm_version": manifest.get("vllm_version")
                or host.get("vllm_version")
                or "",
                "concurrency": result.get("max_concurrency")
                or host.get("concurrency")
                or "",
                "input_size_bucket": name,
                "p50_seconds": p50_value,
                "p95_seconds": p95_value,
                "p95_over_p50": ratio(p95_value, p50_value),
                "p50_actual_send_seconds": p50_actual,
                "p95_actual_send_seconds": p95_actual,
                "p95_over_p50_actual_send": ratio(p95_actual, p50_actual),
                "notes": notes,
            }
        )
    return rows


def _notes(samples, manifest, clock) -> str:
    compared = [
        sample
        for sample in samples
        if isinstance(sample.get("baseline_output_text"), str)
        and isinstance(sample.get("output_text"), str)
    ]
    mismatches = sum(
        1
        for sample in compared
        if sample["output_text"] != sample["baseline_output_text"]
    )
    text = (
        f"clock={clock}; compared_output_text={len(compared)}; "
        f"output_text_mismatches={mismatches}; "
        "setup time excluded; no P99"
    )
    extra = manifest.get("clock")
    if isinstance(extra, str) and extra:
        text = text + "; " + extra
    return text


def render_csv(rows) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in COLUMNS})
    return buffer.getvalue()


def render_markdown(rows) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    rule = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = []
    for row in rows:
        cells = []
        for key in COLUMNS:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            cells.append(str(value).replace("|", "/").replace("\n", " "))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body]) + "\n"
