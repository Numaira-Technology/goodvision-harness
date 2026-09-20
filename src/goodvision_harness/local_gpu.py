"""Local inference routes for a launcher-provisioned NVIDIA deployment.

GPU IDs here are host IDs. Each worker container exposes only its assigned
device, so its inference libraries consistently use device zero internally.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from urllib.parse import urlparse

from .config import HarnessConfig
from .deployment_settings import DeploymentSettings, validate_local_gpu_enabled
from .model_api import ModelAPISettings


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true or false")
    return value in {"true", "1"}


def local_choices(saved: DeploymentSettings | None) -> tuple[bool, bool]:
    choices = []
    for name in ("local_inference_enabled", "local_tracking_enabled"):
        value = getattr(saved, name, None)
        choices.append(
            _env_flag(name.upper())
            if value is None
            else validate_local_gpu_enabled(value, name)
        )
    return choices[0], choices[1]


@dataclass(frozen=True, slots=True)
class LocalGPUDeployment:
    available: bool = False
    gpu_count: int = 0
    inference_device: int = 0
    tracking_device: int = 0

    @classmethod
    def from_environment(cls) -> LocalGPUDeployment:
        # Only the NVIDIA-aware launcher declares this capability. An unrelated
        # graphics adapter or a CUDA Python package is not evidence of a GPU.
        available = _env_flag("LOCAL_GPU_AVAILABLE")
        count = int(os.environ.get("LOCAL_GPU_COUNT", "0") or "0")
        if count < 0 or (available and count == 0):
            raise ValueError("LOCAL_GPU_COUNT must identify available NVIDIA GPUs")
        inference = int(os.environ.get("LOCAL_INFERENCE_GPU_ID", "0"))
        tracking = int(os.environ.get("LOCAL_TRACKING_GPU_ID", str(int(count > 1))))
        if inference < 0 or tracking < 0:
            raise ValueError("local GPU device IDs must be non-negative")
        if available and ((count == 1) != (inference == tracking)):
            raise ValueError("single GPU must be shared; two GPUs must be separated")
        return cls(available, count, inference, tracking)

    def public_status(self) -> dict[str, object]:
        return {
            "available": self.available,
            "gpu_count": self.gpu_count if self.available else 0,
            "inference_device": self.inference_device if self.available else None,
            "tracking_device": self.tracking_device if self.available else None,
            "assignment": ("shared" if self.gpu_count == 1 else "split")
            if self.available
            else "unavailable",
        }

    def validate_choices(self, inference: bool, tracking: bool) -> None:
        validate_local_gpu_enabled(inference, "local_inference_enabled")
        validate_local_gpu_enabled(tracking, "local_tracking_enabled")
        if (inference or tracking) and not self.available:
            raise ValueError("本地 GPU 服务未配置。请先运行本地 GPU 启动程序。")

    def configure(
        self,
        base: HarnessConfig,
        *,
        inference: bool,
        tracking: bool,
        tracking_fps: float | None = None,
    ) -> HarnessConfig:
        self.validate_choices(inference, tracking)
        config = base
        if inference:
            endpoint = _endpoint(
                "LOCAL_INFERENCE_ENDPOINT",
                "http://127.0.0.1:8000/v1/chat/completions",
            )
            model = os.environ.get("LOCAL_INFERENCE_MODEL", "local-vision").strip()
            if not model:
                raise ValueError("LOCAL_INFERENCE_MODEL must not be empty")
            max_tokens = int(os.environ.get("LOCAL_INFERENCE_MAX_TOKENS", "1600"))
            if not 1 <= max_tokens <= 16384:
                raise ValueError("LOCAL_INFERENCE_MAX_TOKENS must be within 1..16384")
            # Start a fresh route so cloud-specific options, models, credentials
            # and extra headers cannot follow the local selection.
            config = replace(
                config,
                aiclub=ModelAPISettings(
                    provider="local",
                    endpoint=endpoint,
                    timeout_seconds=base.model_api.timeout_seconds,
                    max_retries=base.model_api.max_retries,
                    retry_backoff_seconds=base.model_api.retry_backoff_seconds,
                ),
                inference=replace(
                    base.inference,
                    model=model,
                    request_parameters={
                        "temperature": 0,
                        "max_tokens": max_tokens,
                    },
                ),
                runtime=replace(config.runtime, scene_max_tokens=max_tokens),
            )
            if base.rich_labeling.backend == "vlm":
                config = replace(
                    config,
                    rich_labeling=replace(
                        config.rich_labeling,
                        model=model,
                        vlm_request_parameters=config.inference.request_parameters,
                    ),
                )
        if tracking:
            fps = (tracking_fps if tracking_fps is not None
                   else float(os.environ.get("LOCAL_TRACKING_FPS", "2")))
            if not math.isfinite(fps) or not 0.5 <= fps <= 30:
                raise ValueError("LOCAL_TRACKING_FPS must be within 0.5..30")
            config = replace(
                config,
                rich_labeling=replace(
                    config.rich_labeling,
                    enabled=True,
                    tracking_enabled=True,
                    backend="sam31_stream",
                    processing_method="sam_live",
                    max_fps=fps,
                    endpoint=_endpoint(
                        "LOCAL_TRACKING_ENDPOINT",
                        "http://127.0.0.1:8092/v1/video/sessions",
                    ),
                ),
                runtime=replace(config.runtime, scene_mode="active"),
            )
        return config


def _endpoint(name: str, default: str) -> str:
    endpoint = os.environ.get(name, default).strip()
    parsed = urlparse(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
        or parsed.query
    ):
        raise ValueError(f"{name} must be an HTTP(S) service URL without credentials")
    return endpoint
