"""Explicit evaluation settings, with bookkeeping and credentials excluded."""
from __future__ import annotations

import copy
from urllib.parse import urlsplit, urlunsplit


def settings_record(value):
    """Keep declared values without carrying legacy checksum gates forward."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = key.lower()
            if lowered in {"sha256", "sha1", "md5", "hash", "checksum", "fingerprint"} or lowered.endswith(
                    ("_sha256", "_sha1", "_md5", "_hash")):
                continue
            if lowered in {"api_key", "authorization", "password", "secret", "token", "headers"}:
                result[key] = "[REDACTED]"
            else:
                result[key] = settings_record(item)
        return result
    if isinstance(value, (list, tuple)):
        return [settings_record(item) for item in value]
    if isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https", "rtsp", "rtsps"}:
            return urlunsplit((parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", ""))
    return copy.deepcopy(value)


def source_settings(config):
    """Ledger placement and generated snapshots do not alter source observations."""
    return settings_record({key: value for key, value in config.items()
                            if key not in {"model_ledger", "effective_harness_config"}})
