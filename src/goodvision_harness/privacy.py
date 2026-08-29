"""Public-output privacy helpers.

The service deliberately keeps implementation/vendor identifiers internal.  All
records that can leave the process should pass through these helpers, while raw
third-party failures are reduced to stable, operator-safe messages at the point
where they are caught.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

PUBLIC_ANALYSIS_ERROR = "analysis request failed"
PUBLIC_FACE_ERROR = "local face analysis failed"

_PRIVATE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"https?://aiclub\.zimaocloud\.com(?:/[^\s\"']*)?",
        r"\bai[\s_-]*club\b",
        r"https?://(?:www\.)?openrouter\.ai(?:/[^\s\"']*)?",
        r"\bopen[\s_-]*router\b(?:\.ai)?",
        r"\bqwen(?:[\w./:+-]*)?\b",
        r"\bcoreweave\b",
        r"\bdeepinfra\b",
        r"\binsight[\s_-]*face\b",
        r"\bbuffalo(?:[_-][\w.-]+)?\b",
        r"\bantelope(?:v\d+)?\b",
        r"\bonnx(?:runtime)?(?:-gpu)?\b",
        r"\b(?:cpu|cuda)executionprovider\b",
        r"\bfaceanalysis\b",
    )
)


def contains_private_implementation_detail(value: object) -> bool:
    """Return whether text contains a known private engine/provider marker."""

    rendered = str(value)
    return any(pattern.search(rendered) for pattern in _PRIVATE_PATTERNS)


def redact_public_text(value: object, *, fallback: str = "analysis engine") -> str:
    """Redact known implementation identifiers from otherwise useful text."""

    rendered = str(value)
    for pattern in _PRIVATE_PATTERNS:
        rendered = pattern.sub(fallback, rendered)
    return rendered


def public_error_message(
    _error: object | None = None,
    *,
    fallback: str = PUBLIC_ANALYSIS_ERROR,
) -> str:
    """Return a stable public failure without echoing an underlying exception."""

    return fallback


def public_safe_value(value: Any) -> Any:
    """Recursively redact strings in JSON-shaped public records."""

    if isinstance(value, str):
        return redact_public_text(value)
    if isinstance(value, Mapping):
        return {
            redact_public_text(key): public_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(public_safe_value(item) for item in value)
    if isinstance(value, list):
        return [public_safe_value(item) for item in value]
    return value
