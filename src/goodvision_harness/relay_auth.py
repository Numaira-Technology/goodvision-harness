"""Authentication for the private gateway-to-analysis relay channel."""

from __future__ import annotations

import hmac
import os
from collections.abc import Mapping
from pathlib import Path

from fastapi import HTTPException

RELAY_AUTH_HEADER = "X-Goodvision-Relay-Key"
_MAX_KEY_LENGTH = 512


def _valid_key(value: str) -> bool:
    return (
        32 <= len(value) <= _MAX_KEY_LENGTH
        and value.isascii()
        and all(33 <= ord(character) <= 126 for character in value)
    )


def relay_key() -> str | None:
    value = os.environ.get("HARNESS_RELAY_KEY", "").strip()
    if value:
        return value if _valid_key(value) else None
    filename = os.environ.get("HARNESS_RELAY_KEY_FILE", "").strip()
    if not filename:
        return None
    try:
        with Path(filename).open("r", encoding="utf-8") as source:
            value = source.read(_MAX_KEY_LENGTH + 2)
    except (OSError, UnicodeError, ValueError):
        return None
    if len(value) > _MAX_KEY_LENGTH + 1:
        return None
    value = value.strip()
    return value if _valid_key(value) else None


def strip_relay_header(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value for key, value in headers.items()
        if key.lower() != RELAY_AUTH_HEADER.lower()
    }


def relay_authorized(headers: Mapping[str, str]) -> bool:
    expected = relay_key()
    supplied = [
        value for key, value in headers.items()
        if key.lower() == RELAY_AUTH_HEADER.lower()
    ]
    return (
        expected is not None and len(supplied) == 1
        and isinstance(supplied[0], str) and supplied[0].isascii()
        and hmac.compare_digest(expected, supplied[0])
    )


def require_relay_auth(headers: Mapping[str, str]) -> None:
    if not relay_authorized(headers):
        raise HTTPException(403, "Relay authentication required")


def relay_headers(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    key = relay_key()
    if key is None:
        raise HTTPException(503, "Local video relay authentication is not configured")
    result = strip_relay_header(headers or {})
    result[RELAY_AUTH_HEADER] = key
    return result
