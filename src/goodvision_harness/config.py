"""Load and validate the streaming-harness YAML configuration."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv

from .aiclub import AIClubSettings
from .reference_memory import MemoryContext, load_memory_context

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True, slots=True)
class StreamSettings:
    """Optional network video source and WHEP request headers."""

    url: str | None = None
    whep_headers: Mapping[str, str] = field(default_factory=dict)
    frame_crop: str = "none"


@dataclass(frozen=True, slots=True)
class VideoSettings:
    """Model-facing frame sampling parameters."""

    fps: float = 2.0
    realtime: bool = True
    playback_speed: float = 1.0
    max_sampled_frames: int | None = None
    max_image_side: int = 1280
    jpeg_quality: int = 85


@dataclass(frozen=True, slots=True)
class DisplaySettings:
    """Browser-facing live feed parameters used when streaming a file."""

    fps: float = 15.0
    max_image_side: int = 720
    jpeg_quality: int = 70
    delay_seconds: float = 2.5
    bbox_max_gap_seconds: float = 0.8


@dataclass(frozen=True, slots=True)
class RichLabelSettings:
    """Optional independent visual annotation worker or VLM API client."""

    enabled: bool = False
    backend: str = "paddle"
    model: str | None = None
    vlm_max_tokens: int = 4096
    vlm_max_image_side: int = 1280
    vlm_request_parameters: Mapping[str, Any] = field(default_factory=dict)
    endpoint: str = "http://rich-label-worker:8091/v1/labels"
    timeout_seconds: float = 2.0
    max_fps: float = 7.5
    history_size: int = 128
    max_interpolation_gap_seconds: float = 2.0
    max_hold_seconds: float = 2.0


@dataclass(frozen=True, slots=True)
class SlidingWindowSettings:
    size_frames: int = 4
    stride_frames: int = 2
    include_partial_final_window: bool = False


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    model: str
    event_label: str
    system_prompt: str
    prompt: str
    request_parameters: Mapping[str, Any] = field(default_factory=dict)
    multi_event_prompt: str = ""


@dataclass(frozen=True, slots=True)
class FaceSettings:
    """Local InsightFace recognition and enrollment settings."""

    model_name: str = "buffalo_l"
    model_root: Path = Path(".models/insightface")
    allow_model_download: bool = False
    provider: str = "CPUExecutionProvider"
    live_scheduler: str = "auto"
    det_size: int = 640
    threshold: float = 0.4
    min_face_size: int = 20
    enrollment_min_face_size: int = 80
    sample_fps: float = 2.5
    max_image_side: int = 1280
    jpeg_quality: int = 90
    gallery_path: Path = Path(".goodvision/face-gallery.json")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    max_concurrent_requests: int = 4


@dataclass(frozen=True, slots=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 7900


@dataclass(frozen=True, slots=True)
class WebhookSettings:
    url: str | None = None
    secret_env: str = "HARNESS_WEBHOOK_SECRET"
    timeout_seconds: float = 5.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    queue_size: int = 100


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    source_id: str = "external-camera"
    result_history: int = 100
    ready_max_frame_age_seconds: float = 10.0
    ready_max_result_age_seconds: float = 120.0
    overlay_jpeg_quality: int = 85
    webhook: WebhookSettings = field(default_factory=WebhookSettings)


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    path: Path
    default_video: Path | None
    stream: StreamSettings
    video: VideoSettings
    display: DisplaySettings
    rich_labeling: RichLabelSettings
    sliding_window: SlidingWindowSettings
    aiclub: AIClubSettings
    inference: InferenceSettings
    memory: MemoryContext | None
    face: FaceSettings
    runtime: RuntimeSettings
    server: ServerSettings
    service: ServiceSettings


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> HarnessConfig:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError("config root must be a YAML mapping")

    input_section = _mapping_section(raw, "input")
    stream_section = _mapping_section(raw, "stream")
    video_section = _mapping_section(raw, "video")
    display_section = _mapping_section(raw, "display")
    rich_label_section = _mapping_section(raw, "rich_labeling")
    window_section = _mapping_section(raw, "sliding_window")
    aiclub_section = _mapping_section(raw, "aiclub")
    if "inference_service" in raw:
        aiclub_section = {
            **aiclub_section,
            "api_key_env": "INFERENCE_API_KEY",
            **_mapping_section(raw, "inference_service"),
        }
    inference_section = _mapping_section(raw, "inference")
    memory_section = _mapping_section(raw, "memory")
    face_section = _mapping_section(raw, "face")
    runtime_section = _mapping_section(raw, "runtime")
    server_section = _mapping_section(raw, "server")
    service_section = _mapping_section(raw, "service")

    default_video_value = input_section.get("default_video")
    default_video = None
    if default_video_value:
        default_video = (config_path.parent / str(default_video_value)).resolve()

    stream_url_value = stream_section.get("url")
    if stream_url_value is not None and not isinstance(stream_url_value, str):
        raise TypeError("stream.url must be a string or null")
    stream_url = (
        stream_url_value.strip() or None if stream_url_value is not None else None
    )

    whep_headers_value = stream_section.get("whep_headers", {})
    if whep_headers_value is None:
        whep_headers_value = {}
    if not isinstance(whep_headers_value, Mapping):
        raise TypeError("stream.whep_headers must be a mapping")
    whep_headers: dict[str, str] = {}
    for raw_key, raw_value in whep_headers_value.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("stream.whep_headers keys must not be empty")
        if raw_value is None:
            raise TypeError(f"stream.whep_headers.{key} must be a string")
        whep_headers[key] = str(raw_value)
    frame_crop = _required_string(
        stream_section.get("frame_crop", "none"), "stream.frame_crop"
    ).lower()
    if frame_crop not in {"none", "left_half", "right_half"}:
        raise ValueError(
            "stream.frame_crop must be none, left_half, or right_half"
        )
    stream = StreamSettings(
        url=stream_url,
        whep_headers=whep_headers,
        frame_crop=frame_crop,
    )

    video = VideoSettings(
        fps=_positive_float(video_section.get("fps", 2.0), "video.fps"),
        realtime=_bool(video_section.get("realtime", True), "video.realtime"),
        playback_speed=_positive_float(
            video_section.get("playback_speed", 1.0), "video.playback_speed"
        ),
        max_sampled_frames=_optional_positive_int(
            video_section.get("max_sampled_frames"), "video.max_sampled_frames"
        ),
        max_image_side=_positive_int(
            video_section.get("max_image_side", 1280), "video.max_image_side"
        ),
        jpeg_quality=_bounded_int(
            video_section.get("jpeg_quality", 85),
            "video.jpeg_quality",
            minimum=1,
            maximum=100,
        ),
    )
    display = DisplaySettings(
        fps=_positive_float(display_section.get("fps", 15.0), "display.fps"),
        max_image_side=_positive_int(
            display_section.get("max_image_side", 720), "display.max_image_side"
        ),
        jpeg_quality=_bounded_int(
            display_section.get("jpeg_quality", 70),
            "display.jpeg_quality",
            minimum=1,
            maximum=100,
        ),
        delay_seconds=_non_negative_float(
            display_section.get("delay_seconds", 2.5),
            "display.delay_seconds",
        ),
        bbox_max_gap_seconds=_positive_float(
            display_section.get("bbox_max_gap_seconds", 0.8),
            "display.bbox_max_gap_seconds",
        ),
    )
    rich_label_enabled = _environment_bool(
        "RICH_LABEL_ENABLED",
        _bool(rich_label_section.get("enabled", False), "rich_labeling.enabled"),
    )
    rich_label_backend = (
        os.environ.get("RICH_LABEL_BACKEND", "").strip()
        or _required_string(
            rich_label_section.get("backend", "paddle"), "rich_labeling.backend"
        )
    ).lower()
    if rich_label_backend not in {"paddle", "vlm"}:
        raise ValueError("rich_labeling.backend must be paddle or vlm")
    rich_label_model = rich_label_section.get("model")
    if rich_label_model is not None and not isinstance(rich_label_model, str):
        raise TypeError("rich_labeling.model must be a string or null")
    if rich_label_model is not None:
        rich_label_model = rich_label_model.strip() or None
    rich_label_endpoint = (
        os.environ.get("RICH_LABEL_ENDPOINT", "").strip()
        or _required_string(
            rich_label_section.get(
                "endpoint", "http://rich-label-worker:8091/v1/labels"
            ),
            "rich_labeling.endpoint",
        )
    )
    if not rich_label_endpoint.startswith(("http://", "https://")):
        raise ValueError("rich_labeling.endpoint must use http or https")
    rich_label_history_size = _positive_int(
        rich_label_section.get("history_size", 128),
        "rich_labeling.history_size",
    )
    if rich_label_history_size < 2:
        raise ValueError("rich_labeling.history_size must be at least 2")
    rich_labeling = RichLabelSettings(
        enabled=rich_label_enabled,
        backend=rich_label_backend,
        model=rich_label_model,
        vlm_max_tokens=_positive_int(
            rich_label_section.get("vlm_max_tokens", 4096),
            "rich_labeling.vlm_max_tokens",
        ),
        vlm_max_image_side=_positive_int(
            rich_label_section.get("vlm_max_image_side", 1280),
            "rich_labeling.vlm_max_image_side",
        ),
        vlm_request_parameters=dict(
            _mapping_section(rich_label_section, "vlm_request_parameters")
        ),
        endpoint=rich_label_endpoint,
        timeout_seconds=_positive_float(
            rich_label_section.get("timeout_seconds", 2.0),
            "rich_labeling.timeout_seconds",
        ),
        max_fps=_positive_float(
            rich_label_section.get("max_fps", 7.5), "rich_labeling.max_fps"
        ),
        history_size=rich_label_history_size,
        max_interpolation_gap_seconds=_positive_float(
            rich_label_section.get("max_interpolation_gap_seconds", 2.0),
            "rich_labeling.max_interpolation_gap_seconds",
        ),
        max_hold_seconds=_positive_float(
            rich_label_section.get("max_hold_seconds", 2.0),
            "rich_labeling.max_hold_seconds",
        ),
    )
    sliding_window = SlidingWindowSettings(
        size_frames=_positive_int(
            window_section.get("size_frames", 4), "sliding_window.size_frames"
        ),
        stride_frames=_positive_int(
            window_section.get("stride_frames", 2),
            "sliding_window.stride_frames",
        ),
        include_partial_final_window=_bool(
            window_section.get("include_partial_final_window", False),
            "sliding_window.include_partial_final_window",
        ),
    )

    request_parameters = inference_section.get("request_parameters") or {}
    if not isinstance(request_parameters, Mapping):
        raise TypeError("inference.request_parameters must be a mapping")
    model = (
        _profile_model(inference_section["profile"], kind="inference")
        if "profile" in inference_section
        else str(inference_section.get("model", "")).strip()
    )
    event_label = str(inference_section.get("event_label", "")).strip()
    system_prompt = str(inference_section.get("system_prompt", "")).strip()
    prompt = str(inference_section.get("prompt", "")).strip()
    multi_event_prompt = str(
        inference_section.get("multi_event_prompt", "")
    ).strip()
    for key, value in {
        "inference.model": model,
        "inference.event_label": event_label,
        "inference.system_prompt": system_prompt,
        "inference.prompt": prompt,
    }.items():
        if not value:
            raise ValueError(f"{key} must not be empty")

    inference = InferenceSettings(
        model=model,
        event_label=event_label,
        system_prompt=system_prompt,
        prompt=prompt,
        request_parameters=dict(request_parameters),
        multi_event_prompt=multi_event_prompt,
    )
    memory = load_memory_context(
        memory_section,
        config_directory=config_path.parent,
    )
    face = FaceSettings(
        model_name=(
            _profile_model(face_section["model_profile"], kind="face")
            if "model_profile" in face_section
            else _required_string(
                face_section.get("model_name", "buffalo_l"), "face.model_name"
            )
        ),
        model_root=_configured_path(
            os.environ.get("FACE_MODEL_ROOT")
            or face_section.get("model_root", ".models/insightface"),
            config_path=config_path,
            name="face.model_root",
        ),
        allow_model_download=_bool(
            face_section.get("allow_model_download", False),
            "face.allow_model_download",
        ),
        provider=_face_provider(face_section),
        live_scheduler=_face_live_scheduler(
            os.environ.get("FACE_LIVE_SCHEDULER")
            or face_section.get("live_scheduler", "auto")
        ),
        det_size=_positive_int(face_section.get("det_size", 640), "face.det_size"),
        threshold=_bounded_float(
            face_section.get("threshold", 0.4),
            "face.threshold",
            minimum=-1.0,
            maximum=1.0,
        ),
        min_face_size=_positive_int(
            face_section.get("min_face_size", 20), "face.min_face_size"
        ),
        enrollment_min_face_size=_positive_int(
            face_section.get("enrollment_min_face_size", 80),
            "face.enrollment_min_face_size",
        ),
        sample_fps=_positive_float(
            face_section.get("sample_fps", 2.5), "face.sample_fps"
        ),
        max_image_side=_positive_int(
            face_section.get("max_image_side", 1280), "face.max_image_side"
        ),
        jpeg_quality=_bounded_int(
            face_section.get("jpeg_quality", 90),
            "face.jpeg_quality",
            minimum=1,
            maximum=100,
        ),
        gallery_path=_configured_path(
            os.environ.get("FACE_GALLERY_PATH")
            or face_section.get("gallery_path", ".goodvision/face-gallery.json"),
            config_path=config_path,
            name="face.gallery_path",
        ),
    )
    runtime = RuntimeSettings(
        max_concurrent_requests=_positive_int(
            runtime_section.get("max_concurrent_requests", 4),
            "runtime.max_concurrent_requests",
        )
    )
    server = ServerSettings(
        host=str(server_section.get("host", "127.0.0.1")),
        port=_positive_int(server_section.get("port", 7900), "server.port"),
    )

    webhook_value = service_section.get("webhook") or {}
    if not isinstance(webhook_value, Mapping):
        raise TypeError("service.webhook must be a mapping")
    webhook_url_value = webhook_value.get("url")
    if webhook_url_value is not None and not isinstance(webhook_url_value, str):
        raise TypeError("service.webhook.url must be a string or null")
    webhook_url = (
        webhook_url_value.strip() or None if webhook_url_value is not None else None
    )
    webhook = WebhookSettings(
        url=webhook_url,
        secret_env=_required_string(
            webhook_value.get("secret_env", "HARNESS_WEBHOOK_SECRET"),
            "service.webhook.secret_env",
        ),
        timeout_seconds=_positive_float(
            webhook_value.get("timeout_seconds", 5.0),
            "service.webhook.timeout_seconds",
        ),
        max_retries=_non_negative_int(
            webhook_value.get("max_retries", 2),
            "service.webhook.max_retries",
        ),
        retry_backoff_seconds=_non_negative_float(
            webhook_value.get("retry_backoff_seconds", 0.5),
            "service.webhook.retry_backoff_seconds",
        ),
        queue_size=_positive_int(
            webhook_value.get("queue_size", 100),
            "service.webhook.queue_size",
        ),
    )
    service = ServiceSettings(
        source_id=_required_string(
            service_section.get("source_id", "external-camera"),
            "service.source_id",
        ),
        result_history=_positive_int(
            service_section.get("result_history", 100),
            "service.result_history",
        ),
        ready_max_frame_age_seconds=_positive_float(
            service_section.get("ready_max_frame_age_seconds", 10.0),
            "service.ready_max_frame_age_seconds",
        ),
        ready_max_result_age_seconds=_positive_float(
            service_section.get("ready_max_result_age_seconds", 120.0),
            "service.ready_max_result_age_seconds",
        ),
        overlay_jpeg_quality=_bounded_int(
            service_section.get("overlay_jpeg_quality", 85),
            "service.overlay_jpeg_quality",
            minimum=1,
            maximum=100,
        ),
        webhook=webhook,
    )

    return HarnessConfig(
        path=config_path,
        default_video=default_video,
        stream=stream,
        video=video,
        display=display,
        rich_labeling=rich_labeling,
        sliding_window=sliding_window,
        aiclub=AIClubSettings.from_mapping(aiclub_section),
        inference=inference,
        memory=memory,
        face=face,
        runtime=runtime,
        server=server,
        service=service,
    )


def _mapping_section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = root.get(name) or {}
    if not isinstance(section, Mapping):
        raise TypeError(f"{name} must be a YAML mapping")
    return section


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be true or false")
    return value


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive number")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered <= 0:
        raise ValueError(f"{name} must be a positive number")
    return rendered


def _non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative number")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return rendered


def _required_string(value: Any, name: str) -> str:
    rendered = str(value).strip()
    if not rendered:
        raise ValueError(f"{name} must not be empty")
    return rendered


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _bounded_float(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    rendered = float(value)
    if not math.isfinite(rendered) or not minimum <= rendered <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return rendered


def _configured_path(value: Any, *, config_path: Path, name: str) -> Path:
    rendered = str(value).strip()
    if not rendered:
        raise ValueError(f"{name} must not be empty")
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _execution_provider(value: Any) -> str:
    provider = _required_string(value, "face.provider")
    if provider not in {
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
    }:
        raise ValueError(
            "face.provider must be CPUExecutionProvider, CUDAExecutionProvider, "
            "or DmlExecutionProvider"
        )
    return provider


def _profile_model(value: Any, *, kind: str) -> str:
    name = "inference.profile" if kind == "inference" else "face.model_profile"
    profile = _required_string(value, name).lower()
    if profile != "standard":
        raise ValueError(f"{name} must be standard")
    return "qwen3.5-plus" if kind == "inference" else "buffalo_l"


def _face_provider(section: Mapping[str, Any]) -> str:
    # Explicit neutral environment takes precedence; preserve legacy settings
    # for existing deployments and local experiments that have no new setting.
    accelerator = os.environ.get("FACE_ACCELERATOR")
    legacy_provider = os.environ.get("FACE_PROVIDER")
    if not accelerator and legacy_provider:
        return _execution_provider(legacy_provider)
    if not accelerator:
        accelerator = section.get("accelerator")
    if accelerator is not None:
        providers = {
            "cpu": "CPUExecutionProvider",
            "gpu": "CUDAExecutionProvider",
            "windows-gpu": "DmlExecutionProvider",
        }
        key = _required_string(accelerator, "face.accelerator").lower()
        if key not in providers:
            raise ValueError("face.accelerator must be cpu, gpu, or windows-gpu")
        return providers[key]
    return _execution_provider(section.get("provider", "CPUExecutionProvider"))


def _face_live_scheduler(value: Any) -> str:
    scheduler = _required_string(value, "face.live_scheduler")
    if scheduler not in {"auto", "full", "single_flight"}:
        raise ValueError(
            "face.live_scheduler must be auto, full, or single_flight"
        )
    return scheduler
