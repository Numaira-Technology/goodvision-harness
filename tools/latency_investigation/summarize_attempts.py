"""Summarize saved eval ledgers by input-token bucket.

Reads ``attempts.jsonl`` plus sibling ``receipts/`` when those files exist.
Does not invent rows. P50 and P95 use the formal ceil-rank on complete
attempts' ``elapsed_seconds``, which is the benchmark's own latency field.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.latency_investigation.percentiles import p50, p95, ratio, token_bucket


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
    results = [
        row
        for row in rows
        if row.get("type") == "result"
        and row.get("complete") is True
        and isinstance(row.get("elapsed_seconds"), (int, float))
        and not isinstance(row.get("elapsed_seconds"), bool)
    ]
    receipts = {}
    receipt_dir = phase_dir / "receipts"
    if receipt_dir.is_dir():
        for path in receipt_dir.glob("*.json"):
            try:
                receipts[path.name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    paired = []
    missing_tokens = 0
    for row in results:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        receipt_name = metadata.get("receipt")
        name = Path(str(receipt_name)).name if receipt_name else ""
        tokens = _prompt_tokens(receipts.get(name, {}))
        if tokens is None:
            missing_tokens += 1
        paired.append(
            {"elapsed_seconds": row["elapsed_seconds"], "prompt_tokens": tokens}
        )
    groups = {"overall": [item["elapsed_seconds"] for item in paired]}
    for item in paired:
        groups.setdefault(token_bucket(item["prompt_tokens"]), []).append(
            item["elapsed_seconds"]
        )
    table = []
    for name in ("overall", "<500", "500-999", ">=1000", "unknown"):
        values = groups.get(name) or []
        if not values and name != "overall":
            continue
        p50_value = p50(values)
        p95_value = p95(values)
        table.append(
            {
                "input_size_bucket": name,
                "n": len(values),
                "p50_seconds": p50_value,
                "p95_seconds": p95_value,
                "p95_over_p50": ratio(p95_value, p50_value),
            }
        )
    return {
        "available": True,
        "path": str(phase_dir),
        "complete_attempts": len(results),
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
