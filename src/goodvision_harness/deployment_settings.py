"""Persistent deployment settings stored outside the immutable image layer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import uuid4


SETTINGS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DeploymentSettings:
    """Console-managed defaults that survive container replacement."""

    stream_url: str
    dry_run: bool
    autostart: bool


class DeploymentSettingsStore:
    """Atomically persist deployment defaults and a write-only model API key."""

    def __init__(self, settings_path: str | Path) -> None:
        self.path = Path(settings_path).expanduser().resolve()
        self.api_key_path = self.path.with_name("model-api-key")
        self._legacy_api_key_path = self.path.with_name("openrouter-api-key")
        self._lock = RLock()

    def load(self) -> DeploymentSettings | None:
        with self._lock:
            if not self.path.exists():
                return None
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError("deployment settings must be a JSON object")
        if payload.get("schema_version") != SETTINGS_SCHEMA_VERSION:
            raise ValueError("unsupported deployment settings schema_version")
        stream_url = payload.get("stream_url")
        dry_run = payload.get("dry_run")
        autostart = payload.get("autostart")
        if not isinstance(stream_url, str):
            raise TypeError("deployment stream_url must be a string")
        if not isinstance(dry_run, bool):
            raise TypeError("deployment dry_run must be boolean")
        if not isinstance(autostart, bool):
            raise TypeError("deployment autostart must be boolean")
        return DeploymentSettings(
            stream_url=stream_url.strip(),
            dry_run=dry_run,
            autostart=autostart,
        )

    def save(self, settings: DeploymentSettings) -> None:
        payload = {
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "stream_url": settings.stream_url,
            "dry_run": settings.dry_run,
            "autostart": settings.autostart,
        }
        self._atomic_write(
            self.path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )

    def load_api_key(self) -> str | None:
        with self._lock:
            if self.api_key_path.exists():
                value = self.api_key_path.read_text(encoding="utf-8").strip()
                self._legacy_api_key_path.unlink(missing_ok=True)
                return value or None
            if not self._legacy_api_key_path.exists():
                return None
            # One-time migration keeps the secret while removing the old
            # provider-branded record from persistent storage.
            value = self._legacy_api_key_path.read_text(encoding="utf-8").strip()
            if value:
                self._atomic_write(self.api_key_path, value + "\n")
            self._legacy_api_key_path.unlink(missing_ok=True)
        return value or None

    def save_api_key(self, api_key: str) -> None:
        rendered = api_key.strip()
        if not rendered:
            raise ValueError("API key must not be empty")
        self._atomic_write(self.api_key_path, rendered + "\n")
        with self._lock:
            self._legacy_api_key_path.unlink(missing_ok=True)

    def delete_api_key(self) -> bool:
        with self._lock:
            existed = self.api_key_path.exists() or self._legacy_api_key_path.exists()
            self.api_key_path.unlink(missing_ok=True)
            self._legacy_api_key_path.unlink(missing_ok=True)
            return existed

    def _atomic_write(self, destination: Path, content: str) -> None:
        with self._lock:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.name}.{uuid4().hex}.tmp"
            )
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
