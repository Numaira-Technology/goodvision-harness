"""Bounded evidence about HTTP 402/429 limits; never retain upstream text."""

from __future__ import annotations

import asyncio
import json

import httpx


def sanitize_rate_limit_diagnostics(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for name, allowed in {
        "limit_source": {
            "upstream_provider",
            "upstream_provider_shared_pool",
            "openrouter",
            "openrouter_rate_limit",
            "openrouter_free_model",
            "openrouter_in_flight_budget",
            "openrouter_key_limit",
            "openrouter_credits",
            "openrouter_guardrail_budget",
            "openrouter_workspace_budget",
        },
        "reason": {"in_flight_budget_exhausted", "weight_exceeds_budget"},
        "provider_name": {
            "Cerebras",
            "Alibaba",
            "Alibaba Cloud Int.",
            "DeepInfra",
            "SiliconFlow",
            "Novita",
            "Parasail",
            "AtlasCloud",
        },
        "error_type": {"rate_limit_exceeded", "overloaded", "provider_unavailable"},
        "provider_code": {
            "rate_limited",
            "rate_limit_exceeded",
            "too_many_requests",
            "overloaded",
            "rate_limit_error",
        },
    }.items():
        item = value.get(name)
        if isinstance(item, str) and item in allowed:
            result[name] = item
    if type(value.get("is_byok")) is bool:
        result["is_byok"] = value["is_byok"]
    # Public results redact vendor identifiers. Keep a vendor-neutral category
    # as well as the exact allowlisted source in the internal transport log.
    categories = {
        "openrouter_in_flight_budget": "in_flight_spending_budget",
        "openrouter_credits": "account_credits",
        "openrouter_key_limit": "key_credit_limit",
        "openrouter_guardrail_budget": "guardrail_budget",
        "openrouter_workspace_budget": "workspace_budget",
    }
    category = categories.get(result.get("limit_source"))
    if category is not None:
        result["limit_category"] = category
    return result


async def read_rate_limit_diagnostics(response):
    # Optional diagnostics must not turn an already received 402/429 into a slow
    # body read. Preserve cancellation of the enclosing request; only our own
    # short diagnostic deadline is swallowed. Retry-After is read separately.
    body = bytearray()
    try:
        async with asyncio.timeout(0.1):
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > 16 * 1024:
                    return {}
                body.extend(chunk)
        payload = json.loads(body)
    except (TimeoutError, httpx.HTTPError, ValueError, RecursionError):
        return {}
    error = payload.get("error") if isinstance(payload, dict) else None
    metadata = error.get("metadata") if isinstance(error, dict) else None
    # Never retain raw/message, arbitrary header values, request IDs, org IDs,
    # or unknown provider fields. Missing evidence stays unknown.
    return sanitize_rate_limit_diagnostics(metadata)
