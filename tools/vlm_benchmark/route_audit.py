"""Check response routing evidence without preserving arbitrary API text."""

import json
from collections import Counter

from tools.vlm_benchmark.monitor import lane_name


def audit(path, stage, *, allow_other_providers=False):
    lanes = {
        name: {
            "providers": Counter(),
            "complete_providers": Counter(),
            "models": Counter(),
            "verified": 0,
            "mismatch": 0,
            "unverified_complete": 0,
        }
        for name in stage["lanes"]
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        report = json.loads(line)["report"]
        row = lanes[lane_name(report)]
        d = report.get("timing", {}).get("response_diagnostics", {})
        provider, model = d.get("provider_category"), d.get("model_match")
        if provider:
            row["providers"][provider] += 1
        if model:
            row["models"][model] += 1
        known_provider = provider in {"Alibaba", "Alibaba Cloud Int."}
        provider_allowed = bool(provider) if allow_other_providers else known_provider
        verified = provider_allowed and model == "matched"
        row["verified"] += int(verified)
        row["mismatch"] += int(
            (provider is not None and not provider_allowed) or model == "mismatch"
        )
        complete = (
            report.get("state") == "completed"
            and report.get("analysis_complete") is True
        )
        row["unverified_complete"] += int(complete and not verified)
        if complete:
            row["complete_providers"][provider or "unreported"] += 1
    return {
        "valid": bool(lanes)
        and all(
            r["verified"] > 0 and r["mismatch"] == 0 and r["unverified_complete"] == 0
            for r in lanes.values()
        ),
        "lanes": lanes,
        "provider_policy": (
            "alibaba_first_with_fallback" if allow_other_providers else "alibaba_only"
        ),
        "scope": "OpenRouter response labels; not independent weight attestation",
    }
