"""Independent, console-managed OpenAI-compatible summary connection."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from urllib.parse import urlsplit


def summary_service_settings(value: object = None) -> dict:
    if value is None:
        value = {}
    if not isinstance(value, Mapping) or set(value) - {
        "mode",
        "endpoint",
        "model",
        "request_parameters",
        "api_key",
    }:
        raise ValueError("invalid summary service settings")
    result = {
        "mode": value.get("mode", "inherit"),
        "endpoint": value.get("endpoint", ""),
        "model": value.get("model", ""),
        "request_parameters": copy.deepcopy(value.get("request_parameters", {})),
        "api_key": value.get("api_key", ""),
    }
    if result["mode"] not in ("inherit", "custom"):
        raise ValueError("summary service mode must be inherit or custom")
    for name, limit in (("endpoint", 2048), ("model", 512), ("api_key", 4096)):
        item = result[name]
        if (
            not isinstance(item, str)
            or len(item) > limit
            or any(ord(char) < 32 or ord(char) == 127 for char in item)
        ):
            raise ValueError(f"invalid summary service {name}")
        result[name] = item.strip()
    if result["endpoint"]:
        try:
            url = urlsplit(result["endpoint"])
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username is not None
                or url.password is not None
                or url.fragment
                or url.query
                or (url.port is not None and not 1 <= url.port <= 65535)
                or any(c.isspace() or c == "\\" for c in result["endpoint"])
            ):
                raise ValueError
        except ValueError:
            raise ValueError(
                "summary endpoint must be a complete HTTP(S) chat-completions URL "
                "without credentials, queries or fragments"
            ) from None
    if result["mode"] == "custom" and (not result["endpoint"] or not result["model"]):
        raise ValueError("summary service requires an endpoint and model")
    parameters = result["request_parameters"]
    if not isinstance(parameters, dict) or set(parameters) & {
        "model",
        "models",
        "messages",
        "stream",
        "endpoint",
        "api_key",
        "api_key_env",
        "route",
    }:
        raise ValueError("invalid summary request parameters")
    for name in ("max_tokens", "max_completion_tokens"):
        if name in parameters and (
            type(parameters[name]) is not int or not 1 <= parameters[name] <= 4096
        ):
            raise ValueError("summary token budget must be within 1 through 4096")
    if {"max_tokens", "max_completion_tokens"} <= parameters.keys():
        raise ValueError("choose one summary token budget parameter")
    try:
        if len(json.dumps(parameters, allow_nan=False)) > 8192:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            "summary request parameters must be a bounded JSON object"
        ) from None
    return result


def public_summary_service(value: dict, *, persistence_available: bool) -> dict:
    return {
        **{key: item for key, item in value.items() if key != "api_key"},
        "api_key_present": bool(value["api_key"]),
        "persistence_available": persistence_available,
    }
