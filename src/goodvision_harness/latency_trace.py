"""Small structured timing records on the server's existing stderr log channel."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone


def trace_id(value: object) -> str | None:
    """Only opaque protocol identifiers, never arbitrary URLs or user text."""
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value):
        return value
    return None


def configure_worker_timing() -> None:
    """Spawned model workers do not inherit Uvicorn's logging configuration."""
    logger = logging.getLogger("uvicorn.error")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("INFO:     %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.disabled = False
    logger.propagate = False


def log_timing(event: str, **fields: object) -> None:
    """Diagnostics cannot turn a successful model/display operation into failure.

    Uvicorn configures this logger even when application INFO loggers are muted.
    Container logs / the launcher stderr file retain these records. No images,
    endpoint URLs, response bodies, exception messages or identity labels belong
    in fields. Wall-clock durations use monotonic clocks at their call sites.
    """
    try:
        logging.getLogger("uvicorn.error").info(
            "%s %s", event,
            json.dumps(
                {"recorded_at": datetime.now(timezone.utc).isoformat(), **fields},
                ensure_ascii=True, allow_nan=False, separators=(",", ":"),
            ),
        )
    except Exception:
        pass
