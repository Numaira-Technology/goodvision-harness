"""Freeze routing from a public endpoint snapshot; never sends inference requests."""

import argparse
import hashlib
import importlib.util
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_spec = importlib.util.spec_from_file_location("provider_evidence", Path(__file__).with_name("evidence.py"))
evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evidence)


def latency_value(endpoint):
    value = endpoint.get("latency_last_30m")
    if isinstance(value, dict):
        value = value.get("p50")
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def select_provider(catalog, model, backend, digest, selected_at, slug=None):
    if catalog.get("data", {}).get("id") != model:
        raise ValueError("endpoint catalog model differs")
    endpoints = [e for e in catalog["data"].get("endpoints", []) if e.get("status") == 0]
    if backend == "language":
        endpoints = [e for e in endpoints if {"temperature", "max_tokens"} <= set(e.get("supported_parameters", []))]
    if not endpoints:
        raise ValueError("no active endpoint supports the fixed request parameters")
    if slug:
        matches = [e for e in endpoints if e.get("tag") == slug]
        if len(matches) != 1:
            raise ValueError("explicit endpoint slug is absent or ambiguous")
        chosen, method = matches[0], "explicit_endpoint_override"
    elif len(endpoints) == 1:
        chosen, method = endpoints[0], "only_eligible_endpoint_not_a_speed_comparison"
    elif any(latency_value(e) is None for e in endpoints):
        raise ValueError("fastest provider unresolved: latency metadata missing; approved timing or explicit selection required")
    else:
        chosen = min(endpoints, key=lambda e: (latency_value(e), e["tag"]))
        method = "lowest_published_latency_last_30m_p50_then_slug"
    if not chosen.get("tag") or not chosen.get("provider_name"):
        raise ValueError("endpoint catalog lacks routing identity")
    return {"schema_version": 1, "model": model, "backend": backend,
            "selected_at": selected_at,
            "provider": {"only": [chosen["tag"]], "allow_fallbacks": False, "require_parameters": True},
            "expected_provider_name": chosen["provider_name"],
            "catalog_quantization": chosen.get("quantization"), "catalog_pricing": chosen.get("pricing"),
            "selection": {"method": method, "catalog_sha256": digest,
                          "eligible_endpoints": [e["tag"] for e in endpoints],
                          "published_latency": latency_value(chosen)},
            "limits": ["Published latency is not this benchmark's measured end-to-end latency.",
                       "Actual provider/model must be checked from inference receipts.",
                       "Catalog precision is not independent proof of actual served weights or precision."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["language", "decision"], required=True)
    parser.add_argument("--catalog", type=Path, help="read a saved catalog instead of HTTP GET")
    parser.add_argument("--catalog-out", type=Path, required=True)
    parser.add_argument("--provider-slug", help="explicit override, never labeled fastest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.catalog_out.exists():
        parser.error("outputs must be new paths")
    try:
        if args.catalog:
            raw = args.catalog.read_bytes()
        else:
            url = "https://openrouter.ai/api/v1/models/" + args.model + "/endpoints"
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("oversized catalog")
        catalog = json.loads(raw)
        args.catalog_out.parent.mkdir(parents=True, exist_ok=True)
        with args.catalog_out.open("xb") as handle:
            handle.write(raw)
        plan = select_provider(catalog, args.model, args.backend, hashlib.sha256(raw).hexdigest(),
                               datetime.now(timezone.utc).isoformat(), args.provider_slug)
        evidence.write_new(args.out, plan)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(plan, indent=2))
    return plan


if __name__ == "__main__":
    main()
