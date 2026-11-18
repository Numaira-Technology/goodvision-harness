"""Summarize saved eval ledgers by input-token bucket.

Reads ``attempts.jsonl`` plus sibling ``receipts/`` when those files exist.
Does not invent rows. Uses the shared formal request-latency definition,
including transport timing, explicit legacy fallback and failure accounting.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.latency_investigation.percentiles import ratio, token_bucket
from tools.unified_benchmark.workflow import timings


def _prompt_tokens(receipt: dict):
    response = receipt.get("response") if isinstance(receipt, dict) else None
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None
    value = usage.get("prompt_tokens")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def summarize_phase(phase_dir) -> dict:
    phase_dir = Path(phase_dir)
    ledger_path = phase_dir / "attempts.jsonl"
    if not ledger_path.is_file():
        return {
            "available": False,
            "reason": "attempts.jsonl missing",
            "path": str(phase_dir),
        }
    rows = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    results = [row for row in rows if row.get("type") == "result"]
    receipts = {}
    receipt_dir = phase_dir / "receipts"
    if receipt_dir.is_dir():
        for path in receipt_dir.glob("*.json"):
            try:
                receipts[path.name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    buckets = {}
    missing_tokens = 0
    for row in results:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        receipt_name = metadata.get("receipt")
        name = Path(str(receipt_name)).name if receipt_name else ""
        tokens = _prompt_tokens(receipts.get(name, {}))
        if tokens is None and row.get("complete") is True:
            missing_tokens += 1
        buckets[id(row)] = token_bucket(tokens)
    by_attempt = {
        (row.get("unit"), row.get("attempt")): buckets[id(row)] for row in results
    }
    groups = {"overall": rows}
    for row in rows:
        if row.get("type") not in {"intent", "result"}:
            continue
        bucket = buckets.get(
            id(row), by_attempt.get((row.get("unit"), row.get("attempt")), "unknown")
        )
        groups.setdefault(bucket, []).append(row)
    table = []
    for name in ("overall", "<500", "500-999", ">=1000", "unknown"):
        group = groups.get(name)
        if group is None:
            continue
        stats = timings(group, split=phase_dir.name)
        table.append(
            {
                "input_size_bucket": name,
                **stats,
                "p95_over_p50": ratio(stats["p95_seconds"], stats["p50_seconds"]),
            }
        )
    return {
        "available": True,
        "path": str(phase_dir),
        "complete_attempts": sum(row.get("complete") is True for row in results),
        "missing_prompt_tokens": missing_tokens,
        "table": table,
    }


def summarize_tree(root) -> list[dict]:
    root = Path(root)
    found = []
    if (root / "attempts.jsonl").is_file():
        found.append(summarize_phase(root))
    for path in sorted(root.rglob("attempts.jsonl")):
        if path.parent == root:
            continue
        found.append(summarize_phase(path.parent))
    return found
