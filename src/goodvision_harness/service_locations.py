"""Independent execution locations for inference and person tracking services."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from .config import HarnessConfig
    from .deployment_settings import DeploymentSettings
    from .local_gpu import LocalGPUDeployment

SERVICE_LOCATION_FIELDS = frozenset(
    {
        "inference_mode",
        "tracking_mode",
        "inference_server_url",
        "inference_server_model",
        "inference_server_protocol",
        "tracking_server_url",
        "tracking_server_capacity",
        "tracking_server_protocol",
        "tracking_sample_fps",
    }
)
_ENVIRONMENT_NAMES = {
    "inference_mode": "INFERENCE_EXECUTION_MODE",
    "tracking_mode": "TRACKING_EXECUTION_MODE",
    "inference_server_url": "INFERENCE_SERVER_URL",
    "inference_server_model": "INFERENCE_SERVER_MODEL",
    "inference_server_protocol": "INFERENCE_SERVER_PROTOCOL",
    "tracking_server_url": "TRACKING_SERVER_URL",
    "tracking_server_capacity": "TRACKING_SERVER_CAPACITY",
    "tracking_server_protocol": "TRACKING_SERVER_PROTOCOL",
    "tracking_sample_fps": "LOCAL_TRACKING_FPS",
}


def validate_service_location(name: str, value: object) -> Any:
    """Validate one persisted field without consulting environment settings."""
    if name == "inference_server_protocol":
        if value not in ("compatible", "goodvision"):
            raise ValueError(
                "inference_server_protocol must be compatible or goodvision"
            )
        return value
    if name == "tracking_sample_fps":
        message = "tracking_sample_fps must be a finite number between 0.5 and 30"
        if type(value) not in (int, float):
            raise ValueError(message)
        try:
            fps = float(value)
        except OverflowError as exc:
            raise ValueError(message) from exc
        if not math.isfinite(fps) or not 0.5 <= fps <= 30:
            raise ValueError(message)
        return fps
    if name == "tracking_server_protocol":
        if value not in ("legacy", "contour"):
            raise ValueError("tracking_server_protocol must be legacy or contour")
        return value
    if name in {"inference_mode", "tracking_mode"}:
        accepted = (
            {"cloud", "local", "server"}
            if name == "inference_mode"
            else {
                "legacy",
                "cloud",
                "local",
                "server",
            }
        )
        if not isinstance(value, str) or value not in accepted:
            raise ValueError(f"{name} must be one of {', '.join(sorted(accepted))}")
        return value
    if name in {"inference_server_url", "tracking_server_url"}:
        return _service_origin(value, name)
    if name == "inference_server_model":
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 512
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("inference_server_model must be a nonempty model name")
        return value.strip()
    if name == "tracking_server_capacity":
        if type(value) is not int or value < 1:
            raise ValueError("tracking_server_capacity must be a positive integer")
        return value
    raise ValueError("unknown service location setting")


def _service_origin(value: object, name: str) -> str:
    message = f"{name} must be an HTTP(S) origin without credentials or a path"
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError(message)
    rendered = value.strip()
    if not rendered:
        return ""
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in rendered
    ) or any(character in rendered for character in "\\?#%"):
        raise ValueError(message)
    try:
        parsed = urlsplit(rendered)
        port = parsed.port
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.netloc.endswith(":")
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError(message)
        # A bare authority must not smuggle another URI scheme or separators.
        hostname = parsed.hostname.encode("idna").decode("ascii").rstrip(".")
        if ":" not in hostname and (
            not hostname
            or len(hostname) > 253
            or any(
                not re.fullmatch(
                    r"[a-zA-Z0-9_](?:[a-zA-Z0-9_-]{0,61}[a-zA-Z0-9_])?", label
                )
                for label in hostname.split(".")
            )
        ):
            raise ValueError(message)
    except (ValueError, UnicodeError):
        raise ValueError(message) from None
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), "", "", ""))


@dataclass(frozen=True, slots=True)
class ServiceLocations:
    inference_mode: str = "cloud"
    tracking_mode: str = "legacy"
    inference_server_url: str = ""
    inference_server_model: str = "local-vision"
    tracking_server_url: str = ""
    tracking_server_capacity: int = 1
    tracking_server_protocol: str = "legacy"
    tracking_sample_fps: float = 2.0
    inference_server_protocol: str = "compatible"

    def __post_init__(self) -> None:
        for field in fields(self):
            value = validate_service_location(field.name, getattr(self, field.name))
            object.__setattr__(self, field.name, value)
        for role in ("inference", "tracking"):
            if getattr(self, f"{role}_mode") == "server" and not getattr(
                self,
                f"{role}_server_url",
            ):
                raise ValueError(f"{role}_server_url is required in server mode")

    @classmethod
    def resolve(
        cls,
        saved: DeploymentSettings | None,
        *,
        local_inference: bool,
        local_tracking: bool,
    ) -> ServiceLocations:
        defaults = {
            "inference_mode": "local" if local_inference else "cloud",
            "tracking_mode": "local" if local_tracking else "legacy",
            "inference_server_url": "",
            "inference_server_model": "local-vision",
            "inference_server_protocol": "compatible",
            "tracking_server_url": "",
            "tracking_server_capacity": 1,
            "tracking_server_protocol": "legacy",
            "tracking_sample_fps": 2.0,
        }
        values = {}
        for name, default in defaults.items():
            value = getattr(saved, name, None)
            if value is None:
                variable = _ENVIRONMENT_NAMES[name]
                value = os.environ.get(variable, "").strip() or default
                if name == "tracking_server_capacity":
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        raise ValueError(
                            "TRACKING_SERVER_CAPACITY must be a positive integer",
                        ) from None
                elif name == "tracking_sample_fps":
                    try:
                        value = float(value)
                    except (TypeError, ValueError, OverflowError):
                        raise ValueError(
                            "LOCAL_TRACKING_FPS must be within 0.5..30"
                        ) from None
            values[name] = value
        # Existing cloud deployments saved "legacy" before there was a cloud mode.
        if (
            values["tracking_mode"] == "legacy"
            and os.environ.get("CLOUD_GPU_CONTROLLER_URL")
        ):
            values["tracking_mode"] = "cloud"
        return cls(**values)

    def updated(self, payload: Mapping[str, Any]) -> ServiceLocations:
        values = {
            name: payload[name] for name in SERVICE_LOCATION_FIELDS if name in payload
        }
        for role, default_mode in (("inference", "cloud"), ("tracking", "legacy")):
            old_name, mode_name = f"local_{role}_enabled", f"{role}_mode"
            if old_name not in payload:
                continue
            enabled = payload[old_name]
            if not isinstance(enabled, bool):
                raise ValueError(f"{old_name} must be boolean")
            if mode_name in values:
                mode = validate_service_location(mode_name, values[mode_name])
                if enabled != (mode == "local"):
                    raise ValueError(f"{old_name} contradicts {mode_name}")
            else:
                values[mode_name] = "local" if enabled else default_mode
        return replace(self, **values)

    def public_status(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def tracking_capacity(self, local_capacity: int, gpu: LocalGPUDeployment) -> int:
        if self.tracking_mode == "cloud":
            return max(1, local_capacity)
        if self.tracking_mode == "server":
            return self.tracking_server_capacity
        if self.tracking_mode == "local" and gpu.available and gpu.tracking_available:
            return max(1, local_capacity)
        return 1

    def configure(
        self,
        base: HarnessConfig,
        gpu: LocalGPUDeployment,
        camera_index: int = 1,
    ) -> HarnessConfig:
        from .model_api import ModelAPISettings

        if type(camera_index) is not int or camera_index < 1:
            raise ValueError("camera_index must be a positive integer")
        config = gpu.configure(
            base,
            inference=self.inference_mode == "local",
            tracking=self.tracking_mode == "local",
            tracking_fps=self.tracking_sample_fps,
        )
        if self.inference_mode == "server":
            max_tokens = int(os.environ.get("LOCAL_INFERENCE_MAX_TOKENS", "1600"))
            if not 1 <= max_tokens <= 16384:
                raise ValueError("LOCAL_INFERENCE_MAX_TOKENS must be within 1..16384")
            config = replace(
                config,
                aiclub=ModelAPISettings(
                    provider="local",
                    endpoint=self.inference_server_url + "/v1/chat/completions",
                    timeout_seconds=base.model_api.timeout_seconds,
                    max_retries=base.model_api.max_retries,
                    retry_backoff_seconds=base.model_api.retry_backoff_seconds,
                    inspection_endpoint=(
                        self.inference_server_url + "/v1/inspections"
                        if self.inference_server_protocol == "goodvision" else ""
                    ),
                ),
                inference=replace(
                    config.inference,
                    model=self.inference_server_model,
                    request_parameters={"temperature": 0, "max_tokens": max_tokens},
                ),
                runtime=replace(
                    config.runtime, scene_max_tokens=max_tokens,
                    scene_mode=(
                        "active" if self.inference_server_protocol == "goodvision"
                        else config.runtime.scene_mode
                    ),
                ),
            )
            if (config.rich_labeling.backend == "vlm"
                    and self.inference_server_model not in {
                        "ZhuYuhan/VideoJev-4B", "vllm-sr/d3-nano",
                    }):
                config = replace(
                    config,
                    rich_labeling=replace(
                        config.rich_labeling,
                        model=self.inference_server_model,
                        vlm_request_parameters=config.inference.request_parameters,
                    ),
                )
        if self.tracking_mode == "cloud":
            origin = _service_origin(
                os.environ.get("CLOUD_TRACKING_URL", "http://sam-remote:8092"),
                "CLOUD_TRACKING_URL",
            )
            config = replace(
                config,
                rich_labeling=replace(
                    config.rich_labeling, enabled=True, tracking_enabled=True,
                    backend="sam31_stream", processing_method="sam_live",
                    max_fps=self.tracking_sample_fps,
                    endpoint=f"{origin}/v1/cameras/camera-{camera_index:02d}/perception/sam_live/sessions",
                ),
                runtime=replace(config.runtime, scene_mode="active"),
            )
        if self.tracking_mode == "server":
            path = (
                f"/v1/cameras/camera-{camera_index:02d}/perception/sam_live/sessions"
                if self.tracking_server_capacity > 1
                else "/v1/video/sessions"
            )
            if self.tracking_server_protocol == "contour":
                path = (
                    f"/v1/cameras/camera-{camera_index:02d}/contour/sessions"
                    if self.tracking_server_capacity > 1
                    else "/v1/contour/sessions"
                )
            config = replace(
                config,
                rich_labeling=replace(
                    config.rich_labeling,
                    enabled=True,
                    tracking_enabled=True,
                    backend="sam31_stream",
                    processing_method="sam_live",
                    max_fps=self.tracking_sample_fps,
                    endpoint=self.tracking_server_url + path,
                ),
                runtime=replace(config.runtime, scene_mode="active"),
            )
        if config.rich_labeling.backend == "sam31_stream":
            config = replace(config, rich_labeling=replace(
                config.rich_labeling, max_fps=self.tracking_sample_fps,
            ))
        if (config.model_api.provider == "local"
                and config.inference.model in {
                    "ZhuYuhan/VideoJev-4B", "vllm-sr/d3-nano",
                }):
            if self.inference_mode == "server":
                config = replace(config, aiclub=replace(
                    config.model_api, endpoint=self.inference_server_url + "/predict",
                    inspection_endpoint="",
                ))
            # This classifier emits scene choices, not chat or person grounding.
            config = replace(config, runtime=replace(
                config.runtime, scene_mode="active", scene_independent_vlm=True,
                scene_vlm_output_mode="events", scene_observation_fps=1.25,
                scene_max_in_flight=1, scene_max_image_side=720,
                scene_jpeg_quality=90, scene_timeout_seconds=20.0,
            ))
            if config.rich_labeling.backend == "vlm":
                # Native scene choices do not implement person-attribute chat.
                config = replace(config, rich_labeling=replace(
                    config.rich_labeling, enabled=False, tracking_enabled=False,
                ))
        return config
