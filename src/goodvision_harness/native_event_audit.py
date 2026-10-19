"""Session-scoped append-only audit for explicitly invoked research runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock


def redact_images(value):
    if isinstance(value, dict):
        return {
            key: (
                {
                    "image_omitted": True,
                    "base64_characters": len(item),
                }
                if key in {"jpeg_base64", "image_base64"} and isinstance(item, str)
                else redact_images(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact_images(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {
            "image_omitted": True,
            "data_url_characters": len(value),
        }
    return value


class NativeAudit:
    """Audit errors abort the evaluator before another request can be made."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")
        self._lock = RLock()
        self.failed = False
        self.rows = []

    def record(self, kind, **payload):
        with self._lock:
            if self.failed:
                raise RuntimeError("native audit is unavailable")
            row = {"kind": kind, **payload}
            try:
                data = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                self._handle.write(data)
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except (OSError, ValueError, TypeError):
                self.failed = True
                raise
            self.rows.append(row)
            return row

    def close(self):
        self._handle.close()
