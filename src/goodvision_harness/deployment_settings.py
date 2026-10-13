"""Persistent deployment settings stored outside the immutable image layer."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .rich_label_models import validate_rich_label_model
from .service_locations import SERVICE_LOCATION_FIELDS, validate_service_location

SETTINGS_SCHEMA_VERSION = 1
# Stable, neutral names keep each route's credentials independent. The original
# record belongs to the default route so existing deployments remain usable.
_API_KEY_FILENAMES = {
    "aiclub": "model-api-key",
    "openrouter": "model-api-key-2",
    "dashscope": "model-api-key-3",
    "local": "local-model-api-key",
    "tracking": "tracking-node-key",
}


@dataclass(frozen=True, slots=True)
class DeploymentSettings:
    """Console-managed defaults that survive container replacement."""

    stream_url: str
    dry_run: bool
    autostart: bool
    buffer_delay_seconds: float | None = None
    camera_layout: str | None = None
    processing_method: str | None = None
    local_inference_enabled: bool | None = None
    local_tracking_enabled: bool | None = None
    inference_mode: str | None = None
    tracking_mode: str | None = None
    inference_server_url: str | None = None
    inference_server_model: str | None = None
    tracking_server_url: str | None = None
    tracking_server_capacity: int | None = None
    tracking_server_protocol: str | None = None
    tracking_sample_fps: float | None = None
    inference_server_protocol: str | None = None
    # Console-selected rich-label model key (brand-neutral: contour / det-balanced
    # / det-fast / det-accurate); see rich_label_models.RICH_LABEL_MODELS.
    rich_label_model: str | None = None


def _load_rich_label_model(payload: dict) -> str | None:
    """Read a persisted rich-label model key, tolerating an unknown value.

    The save path validates strictly, but a stored key may be absent, or stale
    from an earlier build / hand edit. Treat an unrecognized key as unset
    (falls back to the default) rather than failing the whole settings load.
    """
    value = payload.get("rich_label_model")
    if value is None:
        return None
    try:
        return validate_rich_label_model(value)
    except ValueError:
        return None


def validate_local_gpu_enabled(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def validate_processing_method(value: object) -> str:
    """SAM continuous analysis is the only supported production method."""
    if not isinstance(value, str) or value not in {"sam_live", "person_tracking"}:
        raise ValueError("processing_method must be person_tracking")
    return "sam_live"


def validate_camera_layout(value: object) -> str:
    if not isinstance(value, str) or value not in {"mono", "stereo"}:
        raise ValueError("camera_layout must be mono or stereo")
    return value


def validate_buffer_delay_seconds(value: object) -> float:
    """Bound console buffering without coercing strings or booleans."""
    message = "buffer_delay_seconds must be a finite number between 0 and 30"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    try:
        seconds = float(value)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(seconds) or not 0 <= seconds <= 30:
        raise ValueError(message)
    return seconds


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
        buffer_delay_seconds = (
            validate_buffer_delay_seconds(payload["buffer_delay_seconds"])
            if "buffer_delay_seconds" in payload
            else None
        )
        return DeploymentSettings(
            **{
                name: validate_service_location(name, payload[name])
                for name in SERVICE_LOCATION_FIELDS if name in payload
            },
            stream_url=stream_url.strip(),
            dry_run=dry_run,
            autostart=autostart,
            buffer_delay_seconds=buffer_delay_seconds,
            local_inference_enabled=(
                validate_local_gpu_enabled(
                    payload["local_inference_enabled"], "local_inference_enabled",
                ) if "local_inference_enabled" in payload else None
            ),
            local_tracking_enabled=(
                validate_local_gpu_enabled(
                    payload["local_tracking_enabled"], "local_tracking_enabled",
                ) if "local_tracking_enabled" in payload else None
            ),
            camera_layout=(
                validate_camera_layout(payload["camera_layout"])
                if "camera_layout" in payload
                else None
            ),
            processing_method=(
                # Migrate a previously saved choice without exposing the retired
                # route to new writes or requiring manual settings repair.
                "sam_live"
                if payload.get("processing_method") == "hybrid"
                else validate_processing_method(payload["processing_method"])
                if "processing_method" in payload
                else None
            ),
            rich_label_model=_load_rich_label_model(payload),
        )

    def save(self, settings: DeploymentSettings) -> None:
        payload = {
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "stream_url": settings.stream_url,
            "dry_run": settings.dry_run,
            "autostart": settings.autostart,
        }
        if settings.buffer_delay_seconds is not None:
            payload["buffer_delay_seconds"] = validate_buffer_delay_seconds(
                settings.buffer_delay_seconds
            )
        if settings.camera_layout is not None:
            payload["camera_layout"] = validate_camera_layout(settings.camera_layout)
        if settings.processing_method is not None:
            payload["processing_method"] = validate_processing_method(
                settings.processing_method
            )
        if settings.rich_label_model is not None:
            payload["rich_label_model"] = validate_rich_label_model(
                settings.rich_label_model
            )
        for name in ("local_inference_enabled", "local_tracking_enabled"):
            value = getattr(settings, name)
            if value is not None:
                payload[name] = validate_local_gpu_enabled(value, name)
        for name in SERVICE_LOCATION_FIELDS:
            value = getattr(settings, name)
            if value is not None:
                payload[name] = validate_service_location(name, value)
        self._atomic_write(
            self.path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )

    def _provider_api_key_path(self, provider: str) -> Path:
        try:
            filename = _API_KEY_FILENAMES[provider]
        except KeyError as exc:
            raise ValueError("unsupported inference provider") from exc
        return self.path.with_name(filename)

    def load_api_key(self, *, provider: str = "aiclub") -> str | None:
        key_path = self._provider_api_key_path(provider)
        with self._lock:
            if key_path.exists():
                value = key_path.read_text(encoding="utf-8").strip()
                if provider == "aiclub":
                    self._legacy_api_key_path.unlink(missing_ok=True)
                return value or None
            if provider != "aiclub" or not self._legacy_api_key_path.exists():
                return None
            # One-time migration keeps the secret while removing the old
            # provider-branded record from persistent storage.
            value = self._legacy_api_key_path.read_text(encoding="utf-8").strip()
            if value:
                self._atomic_write(self.api_key_path, value + "\n")
            self._legacy_api_key_path.unlink(missing_ok=True)
        return value or None

    def save_api_key(self, api_key: str, *, provider: str = "aiclub") -> None:
        key_path = self._provider_api_key_path(provider)
        rendered = api_key.strip()
        if not rendered:
            raise ValueError("API key must not be empty")
        self._atomic_write(key_path, rendered + "\n")
        with self._lock:
            if provider == "aiclub":
                self._legacy_api_key_path.unlink(missing_ok=True)

    def delete_api_key(self, *, provider: str = "aiclub") -> bool:
        key_path = self._provider_api_key_path(provider)
        with self._lock:
            existed = key_path.exists()
            key_path.unlink(missing_ok=True)
            if provider == "aiclub":
                existed = existed or self._legacy_api_key_path.exists()
                self._legacy_api_key_path.unlink(missing_ok=True)
            return existed

    def _atomic_write(self, destination: Path, content: str) -> None:
        with self._lock:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)

    def load_model_route(self) -> str | None:
        """Only an opaque catalog ID is persisted; never an arbitrary route."""
        with self._lock:
            path = self.path.with_name("selected-route.json")
            if not path.exists():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        from .route_probe import ROUTES_BY_ID

        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("invalid saved route")
        route_id = payload.get("route_id")
        if not isinstance(route_id, str) or route_id not in ROUTES_BY_ID:
            raise ValueError("invalid saved route")
        return route_id

    def load_analysis_settings(self) -> dict | None:
        """Camera-local analysis choices are independent of shared deployment routes."""
        with self._lock:
            path = self.path.with_name("analysis-settings.json")
            if not path.exists():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("invalid analysis settings")
        return {name: payload[name] for name in ("features", "event_sensitivities")
                if name in payload}

    def save_analysis_settings(self, settings: dict) -> None:
        self._atomic_write(
            self.path.with_name("analysis-settings.json"),
            json.dumps({"schema_version": 1, **settings}, sort_keys=True) + "\n",
        )

    def save_model_route(self, route_id: str) -> None:
        from .route_probe import ROUTES_BY_ID

        if route_id not in ROUTES_BY_ID:
            raise ValueError("invalid route")
        self._atomic_write(
            self.path.with_name("selected-route.json"),
            json.dumps({"schema_version": 1, "route_id": route_id}) + "\n",
        )
