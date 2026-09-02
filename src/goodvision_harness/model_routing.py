"""Resolve one complete model route before any inference client is built."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .model_api import PROVIDER_DEFAULTS, ModelAPISettings


def _section(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return dict(value)


def resolve_model_route(
    raw: Mapping[str, Any], inference: Mapping[str, Any]
) -> tuple[ModelAPISettings, str, dict[str, Any]]:
    """Select connection, model, and channel-specific request parameters together.

    The old top-level aiclub/openrouter sections still work. Their global
    inference.model and request parameters belong to that original channel;
    switching channels must not carry its model ID or private routing options.
    """

    service = _section(raw, "inference_service")
    legacy_provider = (
        "openrouter" if "openrouter" in raw and "aiclub" not in raw else "aiclub"
    )
    configured_provider = str(service.get("provider") or legacy_provider).strip().lower()
    provider = (
        os.environ.get("INFERENCE_PROVIDER", "").strip().lower() or configured_provider
    )
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(
            "inference_service.provider must be aiclub, openrouter, or dashscope"
        )
    profiles = _section(service, "providers")
    legacy_route = _section(raw, provider)
    selected_profile = _section(profiles, provider)
    route = {**legacy_route, **selected_profile}

    # A provider-less delivery file uses the existing neutral credential name.
    # Legacy or explicit route configurations retain the channel-specific name.
    shared_connection = dict(service)
    if "inference_service" in raw and "provider" not in service:
        shared_connection.setdefault("api_key_env", "INFERENCE_API_KEY")
    if provider != configured_provider:
        # Legacy neutral files may store their original channel's connection
        # directly in inference_service. Keep shared tuning, not that address
        # or credential name, when the environment selects a different channel.
        shared_connection.pop("endpoint", None)
        shared_connection.pop("api_key_env", None)
        if "providers" not in service:
            shared_connection.pop("extra_headers", None)
    connection = {
        **legacy_route,
        **shared_connection,
        **selected_profile,
        "provider": provider,
    }
    endpoint = os.environ.get("INFERENCE_ENDPOINT", "").strip()
    if endpoint:
        connection["endpoint"] = endpoint
    headers = {
        **_section(legacy_route, "extra_headers"),
        **_section(shared_connection, "extra_headers"),
        **_section(selected_profile, "extra_headers"),
    }
    if provider == "openrouter":
        for key, header in (("http_referer", "HTTP-Referer"), ("app_title", "X-Title")):
            if route.get(key):
                headers[header] = str(route[key])
    connection["extra_headers"] = headers
    settings = ModelAPISettings.from_mapping(connection)

    profile = inference.get("profile")
    if profile is not None and str(profile).strip().lower() != "standard":
        raise ValueError("inference.profile must be standard")
    configured_model = inference.get("model") if provider == configured_provider else None
    model = (
        os.environ.get("INFERENCE_MODEL", "").strip()
        or str(route.get("model") or "").strip()
        or str(configured_model or "").strip()
        or PROVIDER_DEFAULTS[provider][2]
    )
    parameters = _section(inference, "request_parameters")
    if provider != configured_provider:
        # Legacy files kept channel-specific options in the global block.
        for key in ("provider", "reasoning", "enable_thinking"):
            parameters.pop(key, None)
    # These defaults keep the shipped vision routes in non-thinking mode.
    # Custom model IDs need their own documented per-route options.
    if model == PROVIDER_DEFAULTS[provider][2]:
        defaults = {
            "openrouter": {"reasoning": {"enabled": False}},
            "dashscope": {"enable_thinking": False},
        }.get(provider, {})
        parameters = {**defaults, **parameters}
    parameters.update(_section(route, "request_parameters"))
    return settings, model, parameters
