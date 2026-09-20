"""HTTP delivery service for the independent streaming harness."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import struct
import time
import unicodedata
from collections.abc import AsyncIterator, Callable, Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import timezone
from functools import partial
from html import escape
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Annotated, Any
from uuid import uuid4

import anyio
import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.websockets import WebSocketDisconnect

from .camera_display import (
    CameraDisplayMetadata,
    CameraSocketChannel,
    decode_camera_packet,
)
from .config import (
    DEFAULT_CONFIG_PATH,
    DisplaySettings,
    HarnessConfig,
    ServiceSettings,
    load_config,
)
from .contracts import PUBLIC_MODEL_NAME, AnalysisResult
from .delayed_overlay import (
    DelayedOverlayComposer,
    DisplayAnalysis,
    OverlaySelection,
    is_face_object,
)
from .delivery import WebhookDelivery
from .deployment_settings import (
    DeploymentSettings,
    DeploymentSettingsStore,
    validate_buffer_delay_seconds,
    validate_camera_layout,
    validate_processing_method,
)
from .display_telemetry import validate_display_metrics
from .event_archive import EventArchive
from .event_routes import install_event_routes
from .event_sensitivity import (
    EVENT_SENSITIVITY_DEFAULTS,
    normalize_event_sensitivities,
    sensitivity_ui_info,
)
from .face_fast import FaceFastResults
from .face_recognition import (
    FaceRecognitionRuntime,
    FaceRecognitionSession,
    FaceRecognizerConfig,
)
from .fleet import install_fleet_routes
from .grounding import (
    DETECTION_EVENT_LABELS,
    SAFETY_EVENT_LABELS,
    SELECTED_FEATURES_ANALYSIS_KIND,
    build_grounding_client,
    build_selected_features_grounding_client,
)
from .identity_review_worker import IdentityReviewWorker
from .latency_trace import log_timing, trace_id
from .local_gpu import LocalGPUDeployment, local_choices
from .model_api import (
    DEFAULT_OPENROUTER_ENDPOINT,
    MissingAPIKeyError,
    ModelAPISettings,
    PooledModelAPIClient,
    environment_api_key,
)
from .overlay import (
    draw_event_rich_overlay,
    draw_overlay,
    draw_rich_overlay,
)
from .overlay_fusion import OverlayFusionMatcher
from .perception import (
    create_tracking_client,
    create_tracking_coordinator,
    is_contour_endpoint,
    processing_methods,
    processing_status,
)
from .pipeline import CAMERA_MODE, STREAM_MODE, SessionCallbacks, StreamingSession
from .privacy import (
    PUBLIC_ANALYSIS_ERROR,
    PUBLIC_FACE_ERROR,
    PUBLIC_INSPECTION_ERROR,
    PUBLIC_MISSING_API_KEY_ERROR,
    PUBLIC_TRACKING_ERROR,
    public_error_message,
    public_safe_value,
    redact_public_text,
)
from .reference_memory import (
    FACE_RECOGNITION_PURPOSE,
    UNKNOWN_FACE_EVENT_LABEL,
)
from .result_bus import FrameBus, FramePacket, ResultBus
from .rich_labeling import (
    DeferredRichLabelHistory,
    RichLabelClient,
    RichLabelCoordinator,
    RichLabelHTTPClient,
)
from .route_probe import (
    MAX_PROBE_CONCURRENCY,
    PROBE_TRIALS,
    PUBLIC_ROUTE_IDS,
    ROUTES,
    ROUTES_BY_ID,
    choose_route,
    empty_results,
    finish_result,
    probe_sample,
)
from .run_postprocess import RunPostprocess
from .scene_archive import SceneArchive
from .scene_capture import SceneCaptureClient
from .scene_delivery import SceneDisplayWorker, TrackedSceneDisplayWorker
from .scene_model import TemporalVLMInspector
from .scene_results import SceneResultPublisher
from .scene_runtime import SceneRuntime
from .scene_worker import SceneInspectionWorker
from .service_locations import SERVICE_LOCATION_FIELDS, ServiceLocations
from .stream_adapter import parse_stream_source
from .temporal_client_pool import TemporalClientPool
from .video_connection import VideoConnection
from .video_to_stream import encode_jpeg
from .video_tracking import (
    BACKEND as NATIVE_VIDEO_BACKEND,
)
from .video_tracking import (
    TrackingCommitBackpressure,
    VideoTrackingBatch,
    VideoTrackingCoordinator,
)
from .video_tracking_session import (
    VideoSessionTrackingCoordinator,
    continuous_source_history_seconds,
)
from .visual_memory import (
    VisualMemoryRecord,
    VisualMemorySink,
)
from .vlm_rich_labeling import RICH_ANNOTATION_CAPABILITY, VLMRichLabelClient

LOGGER = logging.getLogger(__name__)
MJPEG_BOUNDARY = "goodvision-frame"
CAMERA_FRAME_HEADER = struct.Struct(">HH")
CAMERA_FRAME_MAX_BYTES = 2_000_000
EVENT_LABEL_MAX_CHARS = 80
PERSON_FIELD_MAX_CHARS = 120
FACE_ENROLLMENT_MAX_BYTES = 10_000_000
DEPLOYMENT_VALUE_MAX_CHARS = 4096
DEPLOYMENT_REQUEST_MAX_BYTES = 64 * 1024
RICH_READINESS_RETRY_SECONDS = 5.0
EVENT_ANALYSIS_MODE = "event"
FACE_ANALYSIS_MODE = "face"
MULTI_ANALYSIS_MODE = "multi"
ANALYSIS_MODES = frozenset(
    {EVENT_ANALYSIS_MODE, FACE_ANALYSIS_MODE, MULTI_ANALYSIS_MODE}
)
INDEPENDENT_FACE_ANALYSIS_KINDS = frozenset(
    {"face_recognition", "face_reference_match"}
)


class ProcessingMethodUnavailableError(RuntimeError):
    """A chosen perception plugin must be ready before opening a video source."""

    def __init__(self) -> None:
        super().__init__(
            "人物处理服务尚未就绪或旧会话仍在收尾。当前检测未开始。请检查服务器连接后重试。"
        )


def _resolved_face_live_scheduler(*, provider: str, configured: str) -> str:
    """Resolve auto without coupling the scheduler to a particular GPU vendor."""

    if configured == "auto":
        return (
            "single_flight"
            if provider == "CPUExecutionProvider"
            else "full"
        )
    if configured not in {"full", "single_flight"}:
        raise ValueError("face live scheduler must be auto, full, or single_flight")
    return configured


class HarnessRuntime:
    """Own one active stream or browser-camera session."""

    def __init__(
        self,
        config: HarnessConfig | None,
        *,
        config_error: str | None = None,
        memory_sink: VisualMemorySink | None = None,
        face_runtime: FaceRecognitionRuntime | None = None,
        deployment_store: DeploymentSettingsStore | None = None,
        rich_label_coordinator: RichLabelCoordinator | None = None,
    ) -> None:
        self._fleet_camera_index = int(
            os.environ.get("HARNESS_FLEET_CAMERA_INDEX", "1"),
        )
        self._remote_config = config
        self._base_config_error = config_error
        model_api = config.model_api if config is not None else ModelAPISettings()
        self._inference_provider = model_api.provider
        self._deployment_store = deployment_store
        self._deployment_settings: DeploymentSettings | None = None
        self._deployment_error: str | None = None
        self._persisted_api_key: str | None = None
        self._tracking_api_key: str | None = None
        if deployment_store is not None:
            try:
                self._deployment_settings = deployment_store.load()
                self._tracking_api_key = deployment_store.load_api_key(provider="tracking")
                self._persisted_api_key = deployment_store.load_api_key(
                    provider=self._inference_provider,
                )
            except Exception:
                LOGGER.error("could not load persistent deployment settings")
                self._deployment_error = "persistent settings could not be loaded"
        self._local_gpu = LocalGPUDeployment()
        self._local_inference_enabled = False
        self._local_tracking_enabled = False
        self._service_locations = ServiceLocations()
        self._local_configuration_error: str | None = None
        try:
            self._local_gpu = LocalGPUDeployment.from_environment()
            (self._local_inference_enabled,
             self._local_tracking_enabled) = local_choices(self._deployment_settings)
            self._service_locations = ServiceLocations.resolve(
                self._deployment_settings,
                local_inference=self._local_inference_enabled,
                local_tracking=self._local_tracking_enabled,
            )
            self._local_inference_enabled = (
                self._service_locations.inference_mode == "local"
            )
            self._local_tracking_enabled = (
                self._service_locations.tracking_mode == "local"
            )
            if (config is not None and self._deployment_settings is not None
                    and self._service_locations.tracking_mode == "legacy"
                    and self._deployment_settings.processing_method is not None):
                config = replace(config, rich_labeling=replace(
                    config.rich_labeling,
                    processing_method=self._deployment_settings.processing_method,
                ))
                self._remote_config = config
            if config is not None:
                config = self._service_locations.configure(
                    config, self._local_gpu, self._fleet_camera_index,
                )
                model_api = config.model_api
                self._inference_provider = model_api.provider
                self._persisted_api_key = (
                    deployment_store.load_api_key(provider=model_api.provider)
                    if deployment_store is not None else None
                )
        except (ValueError, TypeError):
            self._local_configuration_error = (
                "模型服务配置不可用。请检查运行位置、服务地址和 GPU 部署。"
            )
            config_error = config_error or self._local_configuration_error
        try:
            dry_run = (
                self._deployment_settings.dry_run
                if self._deployment_settings is not None
                else _env_bool("HARNESS_DRY_RUN", False)
            )
            autostart = (
                self._deployment_settings.autostart
                if self._deployment_settings is not None
                else _env_bool("HARNESS_AUTOSTART", True)
            )
        except ValueError:
            dry_run = False
            autostart = False
            config_error = config_error or "configuration error"
        processing_method = (
            self._deployment_settings.processing_method
            if self._deployment_settings is not None
            and self._deployment_settings.processing_method is not None
            else config.rich_labeling.processing_method if config is not None else None
        )
        if processing_method is not None and config is not None:
            config = replace(config, rich_labeling=replace(
                config.rich_labeling,
                processing_method=validate_processing_method(processing_method),
            ))
            if (config.rich_labeling.backend != "sam31_stream"
                    or config.runtime.scene_mode != "active"):
                config_error = config_error or (
                    "processing methods require active scene mode and sam31_stream"
                )
        self.config = config
        self._rich_config_revision = 0
        self._retiring_rich_labels = None
        self._rich_retirement_thread: Thread | None = None
        self._last_processing_stats: dict[str, Any] | None = None
        self._scene_mode = (
            config.runtime.scene_mode if config is not None else "disabled"
        )
        if self._scene_mode not in {"disabled", "active"}:
            raise ValueError("runtime.scene_mode must be disabled or active")
        self._scene_enabled = self._scene_mode == "active"
        self._scene: SceneRuntime | None = None
        self._scene_worker: SceneInspectionWorker | None = None
        self._temporal_clients = TemporalClientPool()
        self._identity_worker: IdentityReviewWorker | None = None
        self._pending_archive_subject_merges: dict[str, str] = {}
        self._pending_archive_scene: dict[str, Any] | None = None
        self._scene_display = None
        self._scene_archive: SceneArchive | None = None
        self._last_scene_inference_monotonic: float | None = None
        self._scene_errors = 0
        self.config_error = config_error
        self.dry_run = dry_run
        self.autostart = autostart
        self._environment_api_key = environment_api_key(model_api) or None
        self._aiclub_api_key = (
            self._persisted_api_key or self._environment_api_key
        )
        self.settings = config.service if config is not None else ServiceSettings()
        self.source_id = (
            os.environ.get("HARNESS_SOURCE_ID", "").strip() or self.settings.source_id
        )
        self.results = ResultBus(self.settings.result_history)
        self.scene_results = ResultBus(min(self.settings.result_history, 32))
        self.frames = FrameBus()
        self.preview_frames = FrameBus()
        self.annotated_frames = FrameBus()
        display = config.display if config is not None else DisplaySettings()
        if processing_method == "sam_live":
            # Causal contours are ready on the returned source frame. Their
            # presentation must not inherit the legacy VLM interpolation wait.
            display = replace(display, delay_seconds=display.sam_live_delay_seconds)
        if (
            self._deployment_settings is not None
            and self._deployment_settings.buffer_delay_seconds is not None
        ):
            display = replace(
                display,
                delay_seconds=self._deployment_settings.buffer_delay_seconds,
            )
        if (
            config is not None
            and config.rich_labeling.backend == "sam31_stream"
            and self._scene_mode == "active"
        ):
            # Match the console's 0..30-second promise before allocating clients
            # or starting a capture. Source retention remains independently capped.
            validate_buffer_delay_seconds(display.delay_seconds)
        self._display_settings = display
        self._rich_disabled_reason: str | None = None
        self._rich_requested = rich_label_coordinator is not None or (
            config is not None and config.rich_labeling.enabled
        )
        self._tracking_requested = self._rich_requested or (
            config is not None and config.rich_labeling.tracking_enabled
        ) or processing_method is not None
        self._events: EventArchive | None = None
        self._rich_probe_thread: Thread | None = None
        self._rich_probe_stop = Event()
        self._active_rich_session_id: str | None = None
        if rich_label_coordinator is not None:
            self._rich_labels = rich_label_coordinator
        elif config is not None and config_error is None and (
            config.rich_labeling.enabled or processing_method is not None
        ):
            rich = config.rich_labeling
            rich_client: RichLabelClient | Any
            if rich.backend == "vlm":
                rich_client = VLMRichLabelClient(
                    replace(
                        config.model_api,
                        timeout_seconds=rich.timeout_seconds,
                        max_retries=0,
                    ),
                    model=rich.model or config.inference.model,
                    api_key=lambda: self._aiclub_api_key,
                    max_tokens=rich.vlm_max_tokens,
                    max_image_side=rich.vlm_max_image_side,
                    wire_format=rich.vlm_wire_format,
                    execution_mode=rich.vlm_execution_mode,
                    parallel_strategy=rich.vlm_parallel_strategy,
                    structured_output=rich.vlm_structured_output,
                    require_complete=rich.vlm_require_complete,
                    prompt_cache=rich.vlm_prompt_cache,
                    http2=rich.vlm_http2,
                    trust_env=rich.vlm_trust_env,
                    keepalive_expiry_seconds=rich.vlm_keepalive_expiry_seconds,
                    request_parameters=rich.vlm_request_parameters,
                    dry_run=self._rich_annotation_dry_run,
                )
            elif rich.backend in {"sam31", "sam31_stream"}:
                rich_client = self._create_tracking_client(rich)
            else:
                rich_client = RichLabelHTTPClient(
                    rich.endpoint,
                    timeout_seconds=rich.timeout_seconds,
                    mode="tracks", include_attributes=self._rich_requested,
                )
            if rich.backend == "vlm" or rich_client.gpu_ready():
                self._rich_labels = self._tracking_coordinator(rich_client)
            else:
                rich_client.close()
                self._rich_labels = None
                self._rich_disabled_reason = "gpu_required"
                LOGGER.info("rich annotation disabled: usable GPU required")
        else:
            self._rich_labels = None
        self._rich_history = DeferredRichLabelHistory(lambda: self._rich_labels)
        self._overlay_fusion = OverlayFusionMatcher()
        self._annotated_output_fps = (
            max(display.fps, config.video.fps, config.face.sample_fps)
            if config is not None
            else display.fps
        )
        self._delayed_overlay = DelayedOverlayComposer(
            delay_seconds=display.delay_seconds,
            max_gap_seconds=display.bbox_max_gap_seconds,
            output_fps=self._annotated_output_fps,
            render=lambda jpeg, analysis: _render_analysis_jpeg(
                jpeg,
                analysis,
                quality=self.settings.overlay_jpeg_quality,
            ),
            publish=self.annotated_frames.publish,
            enrich=(
                self._render_rich_label_jpeg
                if self._rich_requested
                else None
            ),
            compose=(
                partial(self._render_fused_label_jpeg, interpolate=False)
                if self._rich_requested
                else None
            ),
        )
        self._state_lock = RLock()
        self._scene_result_publisher = SceneResultPublisher()
        self._camera_display_metadata = CameraDisplayMetadata()
        self._run_postprocess: dict[str, RunPostprocess] = {}
        self._identity_control_lock = RLock()
        self._last_camera_receipt = None
        self._postprocess_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="run-postprocess",
        )
        self._control_lock = RLock()
        self._video_connection: VideoConnection | None = None
        self._connection_camera_layout: str | None = None
        self._analysis_settings = {
            "features": list(DETECTION_EVENT_LABELS),
            "event_sensitivities": dict(EVENT_SENSITIVITY_DEFAULTS),
        }
        if deployment_store is not None:
            try:
                saved_analysis = deployment_store.load_analysis_settings()
                if saved_analysis is not None:
                    features = _normalize_features(saved_analysis.get("features"))
                    self._analysis_settings = {
                        "features": list(features),
                        "event_sensitivities": normalize_event_sensitivities(
                            saved_analysis.get("event_sensitivities"), features,
                        ),
                    }
            except (ValueError, TypeError, OSError):
                self._deployment_error = "analysis settings could not be loaded"
        self._session_start_pending = False
        self._session_start_cancel = Event()
        self._pending_start_owner: str | None = None
        self._session: StreamingSession | None = None
        self._session_mode: str | None = None
        self._camera_owner: str | None = None
        self._camera_frame_sequence = 0
        self._camera_sample_period = 0.4
        self._camera_next_sample_at = 0.0
        self._camera_max_image_side = (
            config.video.max_image_side if config is not None else 0
        )
        self._session_started_monotonic: float | None = None
        self._generation = 0
        self._delivery: WebhookDelivery | None = None
        self._delivery_initialized = False
        self._memory_sink = memory_sink
        self._face_runtime = face_runtime
        self.face_fast_results = FaceFastResults()
        self._last_frame_monotonic: float | None = None
        self._last_result_monotonic: float | None = None
        self._closed = False
        self._route_transport: PooledModelAPIClient | None = None
        self._route_pool_enabled = False
        self._route_probe_thread: Thread | None = None
        self._route_probe_stop = Event()
        self._selected_route_id: str | None = next((
            route.id for route in ROUTES_BY_ID.values()
            if config is not None and config.inference.model == route.request_model
            and config.inference.request_parameters.get("provider")
            == route.parameters({})["provider"]
        ), None)
        self._route_persisted = False
        self._route_selection_method = "default"
        self._route_probe: dict[str, Any] = {
            "state": "idle", "results": empty_results(), "reason": None,
            "started_at": None, "finished_at": None, "error": None,
        }
        if self._route_probe_supported() and deployment_store is not None:
            try:
                saved_route = deployment_store.load_model_route()
                if saved_route is not None:
                    self.config = replace(
                        self.config,
                        inference=ROUTES_BY_ID[saved_route].inference(
                            self.config.inference,
                        ),
                    )
                    self._selected_route_id = saved_route
                    self._route_persisted = True
                    self._route_selection_method = "saved"
                    self._route_pool_enabled = True
                    self._remote_config = replace(
                        self._remote_config, inference=self.config.inference,
                    )
            except Exception:
                LOGGER.error("could not load saved route selection")
                self._route_probe["error"] = "saved_route_unavailable"
        self._state: dict[str, Any] = {
            "service_started_at": _utc_now(),
            "source_id": self.source_id,
            "pipeline_state": (
                "configuration_error" if config_error else "idle"
            ),
            "input_mode": None,
            "analysis_mode": MULTI_ANALYSIS_MODE,
            "generation": 0,
            "ready": False,
            "last_frame_at": None,
            "last_result_at": None,
            "source_frames_received": 0,
            "analysis_frames_published": 0,
            "results_published": 0,
            "last_error": config_error,
            "end_reason": None,
            "scene_drain": None,
            "stream": None,
            "stream_session_id": None,
            "event_label": (
                ",".join(DETECTION_EVENT_LABELS) if config is not None else None
            ),
            "event_labels": list(DETECTION_EVENT_LABELS),
            "features": list(DETECTION_EVENT_LABELS),
            "event_sensitivities": dict(EVENT_SENSITIVITY_DEFAULTS),
            "model": PUBLIC_MODEL_NAME if config is not None else None,
            "dry_run": self.dry_run,
            "webhook_enabled": False,
            "memory_enabled": memory_sink is not None,
            "memory_records_accepted": 0,
            "memory_records_dropped": 0,
            "rich_labeling_enabled": (
                self._rich_requested and self._rich_labels is not None
            ),
            "pipeline": None,
        }

    @classmethod
    def from_environment(cls) -> HarnessRuntime:
        config_path = Path(os.environ.get("HARNESS_CONFIG", str(DEFAULT_CONFIG_PATH)))
        deployment_path = os.environ.get(
            "HARNESS_DEPLOYMENT_SETTINGS_PATH", ""
        ).strip()
        deployment_store = (
            DeploymentSettingsStore(deployment_path) if deployment_path else None
        )
        try:
            config = load_config(config_path)
        except Exception:
            LOGGER.error("could not load harness configuration")
            return cls(
                None,
                config_error="configuration error",
                deployment_store=deployment_store,
            )
        return cls(config, deployment_store=deployment_store)

    def configured_stream_url(self) -> str:
        if self.config is None:
            return ""
        if self._deployment_settings is not None:
            return self._deployment_settings.stream_url
        return (
            os.environ.get("HARNESS_STREAM_URL", "").strip()
            or os.environ.get("RTSP_URL", "").strip()
            or (self.config.stream.url or "")
        )

    def _rich_annotation_dry_run(self) -> bool:
        """Use the active session override, or the current deployment default."""

        if self._active_rich_session_id is not None:
            return bool(self._state["dry_run"])
        return self.dry_run

    def configured_frame_crop(self) -> str:
        if self._deployment_settings is not None:
            layout = self._deployment_settings.camera_layout
            if layout is not None:
                return "left_half" if layout == "stereo" else "none"
        crop = self.config.stream.frame_crop if self.config is not None else "none"
        return crop if crop in {"left_half", "right_half"} else "none"

    def configured_camera_layout(self) -> str:
        return "mono" if self.configured_frame_crop() == "none" else "stereo"

    def _route_probe_supported(self) -> bool:
        return self.config is not None and (
            self.config.model_api.provider == "openrouter"
            and self.config.model_api.endpoint == DEFAULT_OPENROUTER_ENDPOINT
        )

    def route_probe_status(self) -> dict[str, Any]:
        """Dedicated operator contract: opaque IDs, safe errors, and timings only."""
        with self._control_lock:
            supported = self._route_probe_supported()
            active = self._session_start_pending or (
                self._session is not None and self._session.running
            )
            selected = (
                PUBLIC_ROUTE_IDS.get(self._selected_route_id) if supported else None
            )
            probe = copy.deepcopy(self._route_probe)
            for row in probe["results"]:
                row["id"] = PUBLIC_ROUTE_IDS[row["id"]]
                row["label"] = f"线路 {row['id']}"
            return {
                **probe,
                "supported": supported,
                "can_probe": supported and bool(self._aiclub_api_key)
                and not active and not self._closed
                and self._route_probe["state"] != "running",
                "can_select": supported and not active and not self._closed
                and self._route_probe["state"] != "running",
                "selected_id": selected,
                "selected_label": f"线路 {selected}" if selected else None,
                "selection_persisted": self._route_persisted,
                "selection_method": self._route_selection_method,
                "trials_per_route": PROBE_TRIALS,
                "selection_policy": "lowest_median_latency",
            }

    def _apply_model_route(self, route_id: str, *, method: str) -> None:
        """Caller holds the control lock; persistence must succeed first."""
        inference = ROUTES_BY_ID[route_id].inference(self.config.inference)
        if self._deployment_store is not None:
            self._deployment_store.save_model_route(route_id)
        route_changed = self.config.inference != inference
        self.config = replace(self.config, inference=inference)
        if route_changed:
            self._temporal_clients.invalidate()
        if self._remote_config is not None:
            self._remote_config = replace(self._remote_config, inference=inference)
        self._selected_route_id = route_id
        self._route_persisted = self._deployment_store is not None
        self._route_selection_method = method
        self._route_pool_enabled = True
        if self._route_transport is not None:
            self._route_transport.settings = self.config.model_api

    def select_model_route(self, public_route_id: str) -> dict[str, Any]:
        """Select only a displayed route, never an arbitrary model or endpoint."""
        with self._control_lock:
            if self._closed or not self._route_probe_supported():
                raise ValueError("route selection is unavailable for this deployment")
            if self._processing_lifecycle_busy():
                raise RuntimeError("stop the active session before selecting a route")
            if self._route_probe["state"] == "running":
                raise RuntimeError("wait for the route check before selecting a route")
            route_id = next((
                internal for internal, public in PUBLIC_ROUTE_IDS.items()
                if public == public_route_id
            ), None)
            if route_id is None:
                raise ValueError("route_id must be one of the displayed route IDs")
            self._apply_model_route(route_id, method="manual")
            self._route_probe["error"] = None
            return self.route_probe_status()

    def _pooled_transport(self) -> PooledModelAPIClient:
        if self._route_transport is None:
            self._route_transport = PooledModelAPIClient(
                self.config.model_api, api_key=self._aiclub_api_key,
            )
        return self._route_transport

    def start_route_probe(self) -> dict[str, Any]:
        with self._control_lock:
            if self._closed or not self._route_probe_supported():
                raise ValueError("route checking is unavailable for this deployment")
            if not self._aiclub_api_key:
                raise ValueError("save an API key before checking routes")
            if self._processing_lifecycle_busy():
                raise RuntimeError("stop the active session before checking routes")
            if self._route_probe["state"] == "running":
                raise RuntimeError("a route check is already running")
            transport = self._pooled_transport()
            transport.settings = replace(
                self.config.model_api, max_retries=0,
                timeout_seconds=min(20.0, self.config.model_api.timeout_seconds),
            )
            self._route_probe_stop.clear()
            self._route_probe = {
                "state": "running", "results": empty_results(), "reason": None,
                "started_at": _utc_now(), "finished_at": None, "error": None,
            }
            self._route_probe_thread = Thread(
                target=self._run_route_probe, name="route-check", daemon=True,
            )
            self._route_probe_thread.start()
            return self.route_probe_status()

    def _run_route_probe(self) -> None:
        def check(index: int) -> None:
            route = ROUTES[index]
            with self._control_lock:
                row = self._route_probe["results"][index]
                row["state"] = "warming"
            # One unscored request warms the connection and route before scoring.
            if not self._route_probe_stop.is_set():
                probe_sample(self._route_transport, self.config.inference, route, 0)
            for trial in range(PROBE_TRIALS):
                if self._route_probe_stop.is_set():
                    return
                with self._control_lock:
                    row["state"] = "testing"
                sample = probe_sample(
                    self._route_transport, self.config.inference, route, trial + 1,
                )
                with self._control_lock:
                    row["samples"].append(sample)
            with self._control_lock:
                finish_result(row)

        try:
            # Queue routes here, before timing, rather than inside the HTTP pool.
            workers = min(MAX_PROBE_CONCURRENCY, len(ROUTES))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(check, range(len(ROUTES))))
            with self._control_lock:
                if self._closed or self._route_probe_stop.is_set():
                    self._route_probe["state"] = "cancelled"
                    return
                route_id, reason = choose_route(self._route_probe["results"])
                if route_id is None:
                    self._route_probe.update(state="failed", error=reason)
                    return
                self._apply_model_route(route_id, method="automatic")
                self._route_probe.update(state="complete", reason=reason)
        except Exception:
            LOGGER.error("route check could not complete")
            with self._control_lock:
                self._route_probe.update(state="failed", error="route_check_failed")
        finally:
            with self._control_lock:
                self._route_probe["finished_at"] = _utc_now()
                if self._route_transport is not None:
                    self._route_transport.settings = self.config.model_api
                    if self._closed:
                        self._route_transport.close()
                        self._route_transport = None

    def service_execution_settings(self) -> dict[str, Any]:
        """Return the shared service topology, excluding credentials."""
        return self._service_locations.public_status()

    def tracking_camera_capacity(self) -> int:
        """Bound tracking lanes by both the service and provisioned main workers."""
        provisioned = max(1, int(os.environ.get("HARNESS_FLEET_CAPACITY", "1")))
        return min(provisioned, self._service_locations.tracking_capacity(
            provisioned, self._local_gpu,
        ))

    def deployment_settings(self) -> dict[str, Any]:
        """Return persistent deployment metadata without returning any secret."""

        stream_url = self.configured_stream_url()
        protocol = None
        if stream_url:
            try:
                protocol = parse_stream_source(stream_url).protocol
            except ValueError:
                protocol = None
        if self._deployment_settings is not None:
            source = "persistent"
        elif os.environ.get("HARNESS_STREAM_URL", "").strip() or os.environ.get(
            "RTSP_URL", ""
        ).strip():
            source = "environment"
        else:
            source = "image_default"
        return public_safe_value({
            "persistence_available": self._deployment_store is not None,
            "settings_saved": self._deployment_settings is not None,
            "settings_source": source,
            "configured_stream": bool(stream_url),
            "configured_stream_protocol": protocol,
            "stream_frame_crop": self.configured_frame_crop(),
            "camera_layout": self.configured_camera_layout(),
            "processing_method": self._processing_method(),
            "processing_methods": self._processing_methods(),
            "processing_cleanup": self._state.get("processing_cleanup"),
            "dry_run": self.dry_run,
            "autostart": self.autostart,
            "buffer_delay_seconds": self._display_settings.delay_seconds,
            "api_key_present": bool(self._aiclub_api_key),
            "api_key_persisted": bool(self._persisted_api_key),
            "tracking_key_present": bool(self._tracking_api_key),
            "local_inference_enabled": self._local_inference_enabled,
            "local_tracking_enabled": self._local_tracking_enabled,
            **self.service_execution_settings(),
            "api_key_required": (
                self.config is None or self.config.model_api.provider != "local"
            ),
            "local_gpu": {
                **self._local_gpu.public_status(),
                "error": self._local_configuration_error,
            },
            "storage_error": self._deployment_error,
        })

    def _create_tracking_client(self, rich):
        if is_contour_endpoint(rich.endpoint):
            return create_tracking_client(rich, api_key=self._tracking_api_key)
        return create_tracking_client(rich)

    def _processing_method(self) -> str | None:
        return (
            self.config.rich_labeling.processing_method
            if self.config is not None else None
        )

    def _processing_methods(self) -> list[dict[str, Any]]:
        if self.config is None:
            return []
        methods = processing_methods(self.config.rich_labeling)
        if self._scene_mode != "active":
            return [
                dict(
                    method, available=False,
                    reason="此部署需要启用持续人物与事件分析。",
                )
                for method in methods
            ]
        return methods

    def _processing_lifecycle_busy(self, *, include_pending_start: bool = True) -> bool:
        """Capture end and native cleanup both belong to the old session."""
        if include_pending_start and self._session_start_pending:
            return True
        if self._retiring_rich_labels is not None:
            return True
        if self._session is not None and self._session.running:
            return True
        if self._state.get("pipeline_state") in {
            "starting", "running", "stopping", "draining",
        }:
            return True
        drain = getattr(self, "_end_drain_thread", None)
        if drain is not None and drain.is_alive():
            return True
        coordinator = self._rich_labels
        return isinstance(coordinator, VideoSessionTrackingCoordinator) and not (
            coordinator.wait_until_idle(timeout_seconds=0)
        )

    def _replace_processing_method(
        self, method: str | None, *, config: HarnessConfig | None = None,
    ) -> None:
        """Detach the old plugin under its revision before creating another."""
        with self._state_lock:
            previous_method = self._processing_method()
            self._rich_config_revision += 1
            self._generation += 1
            self.face_fast_results.stop()
            self.face_fast_results = FaceFastResults(generation=self._generation)
            previous = self._rich_labels
            self._rich_labels = None
            self._retiring_rich_labels = previous
            self._active_rich_session_id = None
            self._last_processing_stats = None
            self._session = None
            self._session_mode = None
            self._camera_owner = None
            self.config = config or replace(self.config, rich_labeling=replace(
                self.config.rich_labeling, processing_method=method,
            ))
            if (getattr(self, "_service_locations", ServiceLocations()).tracking_mode
                    == "legacy" and self._remote_config is not None):
                self._remote_config = replace(
                    self._remote_config, rich_labeling=replace(
                        self._remote_config.rich_labeling, processing_method=method,
                    ),
                )
            self._scene_mode = self.config.runtime.scene_mode
            self._scene_enabled = self._scene_mode == "active"
            self._rich_requested = self.config.rich_labeling.enabled
            if (self._deployment_settings is None
                    or self._deployment_settings.buffer_delay_seconds is None):
                delay = (
                    self.config.display.sam_live_delay_seconds if method == "sam_live"
                    else self.config.display.delay_seconds
                )
                self._display_settings = replace(
                    self._display_settings, delay_seconds=delay,
                )
                self._delayed_overlay.set_delay_seconds(delay)
            self._tracking_requested = (
                self._rich_requested or self.config.rich_labeling.tracking_enabled
                or method is not None
            )
            self._rich_disabled_reason = "gpu_required"
            self._state.update(
                pipeline_state="stopping" if previous is not None else "idle",
                input_mode=None, ready=False, generation=self._generation,
                stream_session_id=None, scene_drain=None,
                processing_method=method, rich_labeling_enabled=False,
                processing_cleanup="closing" if previous is not None else None,
                processing_retiring_method=(
                    previous_method if previous is not None else None
                ),
            )
        if previous is not None:
            previous.close(timeout_seconds=0)
            if previous.close_complete():
                self._complete_processing_retirement(previous)
            else:
                self._rich_retirement_thread = Thread(
                    target=self._wait_processing_retirement, args=(previous,),
                    name="perception-retirement", daemon=True,
                )
                self._rich_retirement_thread.start()

    def _apply_local_execution(
        self, configuration: HarnessConfig, *, locations: ServiceLocations,
    ) -> None:
        """Activate validated routes only after persistence succeeds, while idle."""
        previous = self.config
        self._temporal_clients.invalidate()
        if self._route_transport is not None:
            self._route_transport.close()
            self._route_transport = None
        self._service_locations = locations
        self._local_inference_enabled = locations.inference_mode == "local"
        self._local_tracking_enabled = locations.tracking_mode == "local"
        self._local_configuration_error = None
        self.config_error = self._base_config_error
        self._inference_provider = configuration.model_api.provider
        self._persisted_api_key = self._deployment_store.load_api_key(
            provider=self._inference_provider,
        )
        self._environment_api_key = environment_api_key(configuration.model_api) or None
        self._aiclub_api_key = self._persisted_api_key or self._environment_api_key
        self._route_pool_enabled = False
        if configuration.model_api.provider != "local":
            self._selected_route_id = next((
                route.id for route in ROUTES_BY_ID.values()
                if configuration.inference.model == route.request_model
                and configuration.inference.request_parameters.get("provider")
                == route.parameters({})["provider"]
            ), None)
            self._route_persisted = (
                self._selected_route_id is not None
                and self._deployment_store.load_model_route() == self._selected_route_id
            )
            self._route_pool_enabled = self._route_persisted
        if previous.rich_labeling != configuration.rich_labeling:
            self._replace_processing_method(
                configuration.rich_labeling.processing_method, config=configuration,
            )
        else:
            self.config = configuration
        self._state.update(
            last_error=self.config_error,
            pipeline_state="configuration_error" if self.config_error else
            "stopping" if self._retiring_rich_labels is not None else "idle",
        )

    def _wait_processing_retirement(self, previous: Any) -> None:
        while not self._rich_probe_stop.wait(0.05):
            if previous.close_complete():
                with self._control_lock:
                    self._complete_processing_retirement(previous)
                return

    def _complete_processing_retirement(self, previous: Any) -> None:
        with self._state_lock:
            if self._retiring_rich_labels is not previous:
                return
            self._retiring_rich_labels = None
            self._state.update(
                pipeline_state="idle", processing_cleanup="remote_check_pending",
                processing_retiring_method=None,
            )

    def update_deployment_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist defaults; buffer-only updates also apply to the active session."""

        if self._deployment_store is None:
            raise RuntimeError("persistent deployment storage is not configured")
        allowed = SERVICE_LOCATION_FIELDS | {
            "stream_url",
            "dry_run",
            "autostart",
            "api_key",
            "clear_api_key",
            "tracking_api_key",
            "clear_tracking_api_key",
            "buffer_delay_seconds",
            "camera_layout",
            "processing_method",
            "local_inference_enabled",
            "local_tracking_enabled",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("unknown deployment setting")
        with self._control_lock:
            buffer_only = set(payload) == {"buffer_delay_seconds"}
            if self._processing_lifecycle_busy() and not buffer_only:
                raise RuntimeError("stop the active session before saving settings")
            if self._route_probe["state"] == "running":
                raise RuntimeError("wait for the route check before saving settings")

            setting_fields = SERVICE_LOCATION_FIELDS | {
                "stream_url", "dry_run", "autostart", "buffer_delay_seconds",
                "camera_layout",
                "processing_method",
                "local_inference_enabled", "local_tracking_enabled",
            }
            settings: DeploymentSettings | None = None
            local_configuration = None
            locations = self._service_locations
            if (SERVICE_LOCATION_FIELDS | {
                "local_inference_enabled", "local_tracking_enabled",
            }).intersection(payload):
                locations = locations.updated(payload)
                if self._remote_config is None:
                    raise ValueError("configuration unavailable")
                local_configuration = locations.configure(
                    self._remote_config, self._local_gpu, self._fleet_camera_index,
                )
                # Preserve any explicitly saved cloud route across a restart
                # that initially selected local inference.
                if (locations.inference_mode == "cloud"
                        and local_configuration.model_api.provider == "openrouter"):
                    route = self._deployment_store.load_model_route()
                    if route is not None:
                        local_configuration = replace(
                            local_configuration,
                            inference=ROUTES_BY_ID[route].inference(
                                local_configuration.inference,
                            ),
                        )
            if setting_fields.intersection(payload):
                stream_url = payload.get("stream_url", self.configured_stream_url())
                dry_run = payload.get("dry_run", self.dry_run)
                autostart = payload.get("autostart", self.autostart)
                if not isinstance(stream_url, str):
                    raise ValueError("stream_url must be a string")
                stream_url = stream_url.strip()
                if len(stream_url) > DEPLOYMENT_VALUE_MAX_CHARS:
                    raise ValueError("stream_url is too long")
                if stream_url:
                    parse_stream_source(stream_url)
                if not isinstance(dry_run, bool):
                    raise ValueError("dry_run must be boolean")
                if not isinstance(autostart, bool):
                    raise ValueError("autostart must be boolean")
                buffer_delay_seconds = (
                    self._deployment_settings.buffer_delay_seconds
                    if self._deployment_settings is not None else None
                )
                if "buffer_delay_seconds" in payload:
                    buffer_delay_seconds = validate_buffer_delay_seconds(
                        payload["buffer_delay_seconds"]
                    )
                processing_method = (
                    self._deployment_settings.processing_method
                    if self._deployment_settings is not None else None
                )
                if "processing_method" in payload:
                    processing_method = validate_processing_method(
                        payload["processing_method"]
                    )
                    methods = (processing_methods(local_configuration.rich_labeling)
                               if local_configuration is not None
                               else self._processing_methods())
                    if not any(item["id"] == processing_method and item["available"]
                               for item in methods):
                        raise ValueError("此部署暂不支持所选人物处理方式。")
                if (local_configuration is not None
                        and "processing_method" not in payload):
                    processing_method = (
                        local_configuration.rich_labeling.processing_method
                    )
                if (local_configuration is not None and processing_method is not None):
                    # Apply one complete prospective route. Detaching twice can
                    # overwrite the handle of a worker still retiring.
                    local_configuration = replace(local_configuration, rich_labeling=
                        replace(local_configuration.rich_labeling,
                                processing_method=processing_method))
                settings = DeploymentSettings(
                    stream_url=stream_url,
                    dry_run=dry_run,
                    autostart=autostart,
                    buffer_delay_seconds=buffer_delay_seconds,
                    processing_method=processing_method,
                    local_inference_enabled=locations.inference_mode == "local",
                    local_tracking_enabled=locations.tracking_mode == "local",
                    **locations.public_status(),
                    camera_layout=(
                        validate_camera_layout(payload["camera_layout"])
                        if "camera_layout" in payload else (
                            self._deployment_settings.camera_layout
                            if self._deployment_settings is not None else None
                        )
                    ),
                )

            api_key = payload.get("api_key")
            clear_api_key = payload.get("clear_api_key", False)
            rendered_key: str | None = None
            if not isinstance(clear_api_key, bool):
                raise ValueError("clear_api_key must be boolean")
            if api_key is not None and clear_api_key:
                raise ValueError("cannot set and clear the API key together")
            if api_key is not None:
                if not isinstance(api_key, str):
                    raise ValueError("api_key must be a string")
                if any(
                    unicodedata.category(character) == "Cc"
                    for character in api_key
                ):
                    raise ValueError("api_key must not contain control characters")
                rendered_key = api_key.strip()
                if not rendered_key:
                    raise ValueError("api_key must not be empty")
                if len(rendered_key) > DEPLOYMENT_VALUE_MAX_CHARS:
                    raise ValueError("api_key is too long")

            tracking_key = payload.get("tracking_api_key")
            clear_tracking_key = payload.get("clear_tracking_api_key", False)
            if not isinstance(clear_tracking_key, bool):
                raise ValueError("clear_tracking_api_key must be boolean")
            if tracking_key is not None:
                if (clear_tracking_key or not isinstance(tracking_key, str)
                        or not 1 <= len(tracking_key.strip()) <= 4096
                        or any(not 33 <= ord(char) <= 126 for char in tracking_key.strip())):
                    raise ValueError("invalid tracking node key")
                tracking_key = tracking_key.strip()

            # Do not mutate disk or runtime state until every supplied field has
            # passed validation. A mixed request must fail as one unit when, for
            # example, its stream settings are valid but its API key is not.
            tracking_revision = self._rich_config_revision
            tracking_key_changed = False
            if tracking_key is not None or clear_tracking_key:
                new_tracking_key = None if clear_tracking_key else tracking_key
                tracking_key_changed = new_tracking_key != self._tracking_api_key
                if new_tracking_key is None:
                    self._deployment_store.delete_api_key(provider="tracking")
                else:
                    self._deployment_store.save_api_key(new_tracking_key, provider="tracking")
                self._tracking_api_key = new_tracking_key
            if settings is not None:
                self._deployment_store.save(settings)
                self._deployment_settings = settings
                self.dry_run = settings.dry_run
                self.autostart = settings.autostart
                if local_configuration is not None:
                    self._apply_local_execution(
                        local_configuration, locations=locations,
                    )
                if (settings.processing_method is not None
                        and settings.processing_method != self._processing_method()):
                    self._replace_processing_method(settings.processing_method)
                if "buffer_delay_seconds" in payload:
                    assert buffer_delay_seconds is not None
                    self._delayed_overlay.set_delay_seconds(buffer_delay_seconds)
                    if isinstance(self._scene_display, TrackedSceneDisplayWorker):
                        self._scene_display.set_delay_seconds(buffer_delay_seconds)
                    self._display_settings = replace(
                        self._display_settings, delay_seconds=buffer_delay_seconds,
                    )
            previous_api_key = self._aiclub_api_key
            if rendered_key is not None:
                self._deployment_store.save_api_key(
                    rendered_key, provider=self._inference_provider,
                )
                self._persisted_api_key = rendered_key
            elif clear_api_key:
                self._deployment_store.delete_api_key(
                    provider=self._inference_provider,
                )
                self._persisted_api_key = None
            self._aiclub_api_key = (
                self._persisted_api_key or self._environment_api_key
            )
            if self._aiclub_api_key != previous_api_key:
                self._temporal_clients.invalidate()
            if (rendered_key is not None or clear_api_key) and self._route_transport:
                self._route_transport.close()
                self._route_transport = None
            self._deployment_error = None
            if (tracking_key_changed and self.config is not None
                    and self._rich_config_revision == tracking_revision
                    and is_contour_endpoint(self.config.rich_labeling.endpoint)):
                self._replace_processing_method(self._processing_method())
            if not buffer_only:
                self._update_state(dry_run=self.dry_run)
        return self.deployment_settings()

    def video_display_config(self) -> dict[str, Any]:
        """Shared presentation policy for browser capture and network sources."""
        delay = _env_bounded_float("LOCAL_CAMERA_DISPLAY_DELAY_SECONDS", 6.0, 0.0, 8.0)
        return {
            "protocol": 1, "transport": "websocket",
            "enabled": (self._scene_mode == "active" and self.config.rich_labeling.backend == "sam31_stream"
                        and os.environ.get("LOCAL_CAMERA_DISPLAY_ENABLED", "false").lower() in {"1", "true", "yes"}),
            "metadata_url": "/v1/video/display/metadata", "metrics_url": "/v1/video/display/metrics",
            "target_fps": _env_bounded_int("LOCAL_CAMERA_TARGET_FPS", 20, 15, 30),
            "delay_seconds": delay,
            "motion_width": _env_bounded_int("LOCAL_CAMERA_MOTION_WIDTH", 160, 160, 320),
            "motion_backend": "dense", "anchor_policy": "nearest_arrived",
            "allow_partial_boundaries": False,
            "max_history_seconds": max(8.0, delay + 4.0), "max_propagation_seconds": 1.5,
            "analysis_max_image_side": _env_bounded_int("LOCAL_CAMERA_ANALYSIS_MAX_SIDE", 720, 160, 1280),
            "analysis_jpeg_quality": _env_bounded_int("LOCAL_CAMERA_ANALYSIS_JPEG_QUALITY", 80, 30, 95),
            "analysis_sample_fps": max(self.config.rich_labeling.max_fps, self.config.face.sample_fps),
        }

    def ui_info(self) -> dict[str, Any]:
        if self.config is None:
            result = {
                "configured": False,
                "error": "configuration unavailable",
            }
            safe = public_safe_value(result)
            assert isinstance(safe, dict)
            return safe
        scene_active = self._scene_mode == "active"
        scene_source_side = self.config.runtime.scene_max_image_side
        camera_side = max(
            self.config.video.max_image_side,
            scene_source_side if scene_active else self.config.video.max_image_side,
        )
        face_side = max(
            self.config.face.max_image_side,
            scene_source_side if scene_active else self.config.face.max_image_side,
        )
        stream_url = self.configured_stream_url()
        protocol = None
        if stream_url:
            try:
                protocol = parse_stream_source(stream_url).protocol
            except ValueError:
                protocol = None
        result = {
            "configured": True,
            "scene_engine": {
                "enabled": self._scene_enabled,
                "mode": self._scene_mode,
                "snapshot_url": "/v1/scene",
                "events_url": "/v1/scene/events",
            },
            "model": PUBLIC_MODEL_NAME,
            "event_label": self.config.inference.event_label,
            "event_labels": list(DETECTION_EVENT_LABELS),
            "features": list(DETECTION_EVENT_LABELS),
            "supported_features": list(DETECTION_EVENT_LABELS),
            **sensitivity_ui_info(),
            "local_camera_display": self.video_display_config(),
            "sample_fps": self.config.video.fps,
            "max_image_side": camera_side,
            "jpeg_quality": self.config.video.jpeg_quality,
            "window": {
                "size_frames": self.config.sliding_window.size_frames,
                "stride_frames": self.config.sliding_window.stride_frames,
            },
            "max_concurrent_requests": self.config.runtime.max_concurrent_requests,
            "api_key_present": bool(self._aiclub_api_key),
            "local_inference_enabled": self._local_inference_enabled,
            "local_tracking_enabled": self._local_tracking_enabled,
            **self.service_execution_settings(),
            "api_key_required": self.config.model_api.provider != "local",
            "dry_run": self.dry_run,
            "configured_stream": bool(stream_url),
            "configured_stream_protocol": protocol,
            "stream_frame_crop": self.configured_frame_crop(),
            "camera_layout": self.configured_camera_layout(),
            "processing_method": self._processing_method(),
            "processing_methods": self._processing_methods(),
            "deployment": self.deployment_settings(),
            "analysis_modes": sorted(ANALYSIS_MODES),
            "reference_memory": (
                {
                    "purpose": self.config.memory.purpose,
                    "reference_count": len(self.config.memory.references),
                    "newest_frame_only": self.config.memory.newest_frame_only,
                }
                if self.config.memory is not None
                else None
            ),
            "face": {
                "engine": "local",
                "sample_fps": self.config.face.sample_fps,
                "max_image_side": face_side,
                "jpeg_quality": self.config.face.jpeg_quality,
                "threshold": self.config.face.threshold,
                "min_face_size": self.config.face.min_face_size,
                "enrollment_min_face_size": (self.config.face.enrollment_min_face_size),
            },
            "multi": {
                "sample_fps": self.config.face.sample_fps,
                "max_image_side": face_side,
                "jpeg_quality": self.config.face.jpeg_quality,
                "safety_max_image_side": (
                    scene_source_side
                    if scene_active else self.config.video.max_image_side
                ),
            },
            "annotated_output": {
                **({"mode": "scene", "max_hold_seconds": 1.0} if scene_active else {}),
                "output_fps": self._annotated_output_fps,
                "max_image_side": self.config.display.max_image_side,
                "jpeg_quality": (
                    min(95, self.settings.overlay_jpeg_quality)
                    if scene_active else self.config.display.jpeg_quality
                ),
                "buffer_delay_seconds": (
                    0.0 if (
                        scene_active
                        and self.config.rich_labeling.backend != "sam31_stream"
                    )
                    else self._display_settings.delay_seconds
                ),
                "bbox_max_gap_seconds": (
                    None if scene_active else self.config.display.bbox_max_gap_seconds
                ),
            },
            "rich_labeling": {
                "enabled": self._rich_requested and self._rich_labels is not None,
                "capability": RICH_ANNOTATION_CAPABILITY,
                "max_fps": self.config.rich_labeling.max_fps,
                "track_identity": "session_scoped",
                "age": "estimated_range",
                "gender": "estimated_presentation",
            },
        }
        safe = public_safe_value(result)
        assert isinstance(safe, dict)
        return safe

    def face_gallery(self) -> dict[str, Any]:
        """Return an embedding-free view of the local known-person gallery."""

        try:
            people = self._ensure_face_runtime().list_people()
        except Exception:
            LOGGER.warning("could not open face gallery")
            return {
                "ready": False,
                "people": [],
                "sample_count": 0,
                "error": PUBLIC_FACE_ERROR,
            }
        public_people = [
            {
                "person_id": person.person_id,
                "name": person.name,
                "samples": person.sample_count,
                "sample_count": person.sample_count,
            }
            for person in people
        ]
        result = {
            "ready": True,
            "people": public_people,
            "sample_count": sum(person.sample_count for person in people),
        }
        safe = public_safe_value(result)
        assert isinstance(safe, dict)
        return safe

    def event_archive(self) -> EventArchive:
        """Open persistence outside the frame/inference path, once per service."""
        with self._control_lock:
            if self._events is None:
                if self.config is None or self._closed:
                    raise RuntimeError("event archive unavailable")
                self._events = EventArchive(self.config.events)
            return self._events

    def _tracking_coordinator(self, client: Any, rich: Any = None) -> Any:
        rich = rich or self.config.rich_labeling
        if rich.backend in {"sam31", "sam31_stream"}:
            revision = self._rich_config_revision
            coordinator = create_tracking_coordinator(
                client, rich,
                on_batch=lambda batch: self._on_processing_batch(revision, batch),
            )
            if hasattr(coordinator, "on_native_batch"):
                coordinator.on_native_batch = lambda batch, generation: (
                    self._camera_display_metadata.observe_batch(
                        batch, native_token=(revision, id(coordinator), generation),
                    )
                )
            return coordinator
        return RichLabelCoordinator(
            client, max_fps=rich.max_fps, history_size=rich.history_size,
            max_interpolation_gap_seconds=rich.max_interpolation_gap_seconds,
            max_hold_seconds=rich.max_hold_seconds,
            on_snapshot=self._on_tracking_snapshot,
        )

    def _on_processing_batch(self, revision: int, batch: VideoTrackingBatch) -> None:
        with self._state_lock:
            if revision != self._rich_config_revision:
                return
            self._on_tracking_batch(batch)

    def _on_tracking_batch(self, batch: VideoTrackingBatch) -> None:
        # Bounded in-memory commit. Lifecycle lock excludes source replacement
        # between epoch activation, canonical mapping and scene submission.
        with self._state_lock:
            session = self._session
            if session is None or batch.stream_session_id != session.stream_session_id:
                return
            registry = session.person_registry

            def canonicalize(snapshots):
                if not registry.begin_native_epoch(
                    getattr(batch, "backend", NATIVE_VIDEO_BACKEND),
                    batch.predictor_epoch,
                ):
                    raise ValueError("native predictor epoch rejected")
                canonical = []
                for snapshot in snapshots:
                    registry.observe_snapshot(snapshot)
                    mapped = registry.snapshot_at(
                        snapshot.timestamp_seconds, snapshot.frame_sequence,
                    )
                    if mapped is None or (
                        mapped.timestamp_seconds, mapped.frame_sequence
                    ) != (snapshot.timestamp_seconds, snapshot.frame_sequence):
                        raise ValueError("native frame could not be mapped to source")
                    canonical.append(mapped)
                merges = registry.drain_subject_merges()
                if scene is not None:
                    scene.apply_subject_merges(merges)
                self._queue_archive_subject_merges(session, merges)
                return tuple(canonical)

            scene = getattr(self, "_scene", None)
            if scene is not None:
                retained_sources = getattr(self._rich_labels, "retained_sources", None)
                if not scene.commit_native_tracking_batch(
                    batch.frames,
                    completed_timestamp_seconds=batch.completed_timestamp_seconds,
                    completed_frame_sequence=batch.completed_frame_sequence,
                    canonicalize=canonicalize,
                    retained_sources=(
                        retained_sources(batch) if retained_sources else ()
                    ),
                ):
                    if scene.tracking_last_rejection == "inspection_capacity":
                        raise TrackingCommitBackpressure("inspection capacity")
                    raise ValueError(
                        "native batch lacks complete retained source evidence"
                    )
            else:
                canonicalize(batch.frames)
            self._camera_display_metadata.observe_batch(batch)
            for corrected in registry.drain_reassociated_results():
                self._on_timeline_result(corrected)
                if session.archive is not None:
                    session.archive.submit_result(corrected)
            worker = getattr(self, "_scene_worker", None)
            if worker is not None:
                worker.wake()
            display = getattr(self, "_scene_display", None)
            if isinstance(display, TrackedSceneDisplayWorker):
                display.notify_tracking()
            self._wake_identity_review()

    def _on_tracking_snapshot(self, snapshot: Any) -> None:
        with self._state_lock:
            session = self._session
            if (
                session is None
                or snapshot.stream_session_id != session.stream_session_id
            ):
                return
            registry = session.person_registry
        registry.observe_snapshot(snapshot)
        canonical = (
            registry.snapshot_at(snapshot.timestamp_seconds, snapshot.frame_sequence)
            if getattr(self, "_scene", None) is not None else None
        )
        corrections = registry.drain_reassociated_results()
        with self._state_lock:
            if self._session is not session:
                return
            if canonical is not None and (
                canonical.timestamp_seconds, canonical.frame_sequence
            ) == (snapshot.timestamp_seconds, snapshot.frame_sequence):
                self._observe_scene("observe_tracks", canonical)
            for corrected in corrections:
                self._on_timeline_result(corrected)
                if session.archive is not None:
                    session.archive.submit_result(corrected)
            self._wake_identity_review()

    def enroll_face(
        self,
        person_id: str,
        name: str,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        with self._identity_control_lock:
            person = self._ensure_face_runtime().enroll(person_id, name, image_bytes)
            self._invalidate_fast_face_gallery()
            with self._state_lock:
                session = self._session
            registry = getattr(session, "person_registry", None)
            if registry is not None:
                registry.allow_person(person_id)
            with self._state_lock:
                self._observe_scene("allow_identity", person_id)
            self._camera_display_metadata.allow_identity(person_id)
        result = {
            "ready": True,
            "person": {
                "person_id": person.person_id,
                "name": person.name,
                "samples": person.sample_count,
                "sample_count": person.sample_count,
            },
        }
        safe = public_safe_value(result)
        assert isinstance(safe, dict)
        return safe

    def delete_face(self, person_id: str) -> bool:
        with self._identity_control_lock:
            deleted = self._ensure_face_runtime().delete_person(person_id)
            if deleted:
                self._invalidate_fast_face_gallery()
                with self._state_lock:
                    targets = [(self._session, self._scene)]
                    targets.extend(
                        (run.session, run.scene)
                        for run in self._run_postprocess.values()
                        if run.state != "finalized" and run.session is not None
                    )
                seen = set()
                for session, scene in targets:
                    if session is None or id(session) in seen:
                        continue
                    seen.add(id(session))
                    registry = getattr(session, "person_registry", None)
                    if registry is not None:
                        registry.forget_person(person_id)
                    if scene is not None:
                        scene.forget_identity(person_id)
                self._camera_display_metadata.forget_identity(person_id)
            return deleted

    def _invalidate_fast_face_gallery(self) -> None:
        if isinstance(self._face_runtime, FaceRecognitionRuntime):
            self.face_fast_results.require_gallery(
                self._face_runtime.gallery.private_snapshot().version,
            )

    def _ensure_face_runtime(self) -> FaceRecognitionRuntime:
        if self.config is None:
            raise RuntimeError(self.config_error or "configuration unavailable")
        with self._control_lock:
            if self._face_runtime is None:
                face = self.config.face
                self._face_runtime = FaceRecognitionRuntime(
                    FaceRecognizerConfig(
                        model_name=face.model_name,
                        inference_instances=face.inference_instances,
                        inference_threads=face.inference_threads,
                        model_root=face.model_root,
                        allow_model_download=face.allow_model_download,
                        provider=face.provider,
                        det_size=face.det_size,
                        threshold=face.threshold,
                        min_face_size=face.min_face_size,
                        enrollment_min_face_size=face.enrollment_min_face_size,
                        gallery_path=face.gallery_path,
                    )
                )
            return self._face_runtime

    def start(self) -> None:
        """Initialize delivery and optionally connect the configured preview."""

        if self.config is None or self.config_error is not None:
            return
        self._ensure_rich_readiness_probe()
        try:
            with self._control_lock:
                self._ensure_delivery()
            stream_url = self.configured_stream_url()
            # Auto-connect only provides a preview. Analysis always starts with
            # the explicit global control, including single-camera deployments.
            # Additional cameras connect through the primary fleet controller.
            if self.autostart and stream_url and self._fleet_camera_index == 1:
                self.connect_stream(stream_url)
                return
            self._update_state(
                pipeline_state="idle",
                last_error=None,
                webhook_enabled=self._delivery is not None,
            )
        except MissingAPIKeyError:
            self._update_state(
                pipeline_state="idle", last_error=PUBLIC_MISSING_API_KEY_ERROR,
                ready=False,
            )
        except ProcessingMethodUnavailableError as exc:
            self._update_state(pipeline_state="idle", last_error=str(exc), ready=False)
        except Exception:
            LOGGER.error("could not start harness runtime")
            self._update_state(
                pipeline_state="error",
                last_error=PUBLIC_ANALYSIS_ERROR,
                ready=False,
            )

    def _ensure_rich_readiness_probe(self) -> None:
        """Retry optional GPU warmup without blocking video or status requests."""

        with self._control_lock:
            if (
                self._closed
                or self.config_error is not None
                or self._retiring_rich_labels is not None
                or not self._tracking_requested
                or self._rich_labels is not None
                or (self._rich_probe_thread is not None
                    and self._rich_probe_thread.is_alive())
            ):
                return
            self._rich_probe_thread = Thread(
                target=self._run_rich_readiness_probe,
                name="rich-readiness",
                daemon=True,
            )
            self._rich_probe_thread.start()

    def _run_rich_readiness_probe(self) -> None:
        while not self._rich_probe_stop.is_set():
            if self._try_enable_rich_labels():
                return
            self._rich_probe_stop.wait(RICH_READINESS_RETRY_SECONDS)

    def _try_enable_rich_labels(self) -> bool:
        """Transfer a verified client to one coordinator, or retry later."""

        with self._control_lock:
            if self._closed or self._rich_labels is not None:
                return True
            if self._retiring_rich_labels is not None:
                return False
            if self.config is None or not self._tracking_requested:
                return True
            rich = self.config.rich_labeling
            revision = self._rich_config_revision
        client = None
        try:
            if rich.backend in {"sam31", "sam31_stream"}:
                client = self._create_tracking_client(rich)
            else:
                client = RichLabelHTTPClient(
                    rich.endpoint, timeout_seconds=rich.timeout_seconds,
                    mode="tracks", include_attributes=self._rich_requested,
                )
            if not client.gpu_ready():
                return False
            # Network I/O never holds either lifecycle lock. The active session
            # or shutdown state may have changed while the worker warmed up.
            with self._control_lock:
                if self._closed or self._rich_labels is not None:
                    return True
                if (revision != self._rich_config_revision
                        or self._retiring_rich_labels is not None):
                    return False
                coordinator = self._tracking_coordinator(client, rich)
                with self._state_lock:
                    coordinator.reset(self._active_rich_session_id)
                    self._rich_labels = coordinator
                    self._camera_display_metadata.bind_native_run(
                        self._active_rich_session_id,
                        (revision, id(coordinator), coordinator._generation)
                        if hasattr(coordinator, "on_native_batch") else None,
                    )
                    self._rich_disabled_reason = None
                    self._state["rich_labeling_enabled"] = self._rich_requested
                client = None  # The coordinator now owns and closes it.
            LOGGER.info("rich annotation enabled: remote GPU is ready")
            return True
        except Exception:
            LOGGER.warning("rich annotation GPU readiness check failed")
            return False
        finally:
            if client is not None:
                client.close()

    def _require_processing_ready(self) -> tuple[int, Any] | None:
        """Check a selected plugin without holding locks during network I/O."""
        with self._control_lock:
            if self._processing_method() is None:
                return None
            if self._closed:
                raise RuntimeError("harness service is stopping")
            if self._processing_lifecycle_busy(include_pending_start=False):
                raise RuntimeError(
                    "stop the active session and wait for cleanup before starting"
                )
            revision = self._rich_config_revision
            rich = self.config.rich_labeling
            coordinator = self._rich_labels
        if coordinator is None:
            # The transfer inside this helper is revision-checked, including
            # when a background warmup is trying to enable the same method.
            self._try_enable_rich_labels()
        # A new health-only client avoids competing with native cleanup or
        # mutating an existing client's session state during a start check.
        probe = self._create_tracking_client(rich)
        try:
            if not probe.ready_for_new_session():
                raise ProcessingMethodUnavailableError()
        finally:
            probe.close()
        with self._control_lock:
            if revision != self._rich_config_revision:
                raise RuntimeError("人物处理方式已变化。请重新开始检测。")
            if self._rich_labels is None:
                raise ProcessingMethodUnavailableError()
            return revision, self._rich_labels

    def analysis_settings(self) -> dict[str, Any]:
        return {**copy.deepcopy(self._analysis_settings), **sensitivity_ui_info()}

    def update_analysis_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) - {"features", "event_sensitivities"}:
            raise ValueError("unknown camera analysis setting")
        with self._control_lock:
            if self._processing_lifecycle_busy():
                raise RuntimeError("stop the active session before saving settings")
            if self._deployment_store is None:
                raise RuntimeError("persistent deployment storage is not configured")
            features = _normalize_features(payload.get(
                "features", self._analysis_settings["features"],
            ))
            settings = {
                "features": list(features),
                "event_sensitivities": normalize_event_sensitivities(payload.get(
                    "event_sensitivities",
                    self._analysis_settings["event_sensitivities"],
                ), features),
            }
            self._deployment_store.save_analysis_settings(settings)
            self._analysis_settings = settings
        return self.analysis_settings()

    def connect_stream(
        self, stream_url: str | None = None, *, camera_layout: str | None = None,
    ) -> dict[str, Any]:
        """Decode for preview; no analysis session or model client is created."""
        with self._control_lock:
            if self.config is None or self._closed:
                raise RuntimeError("configuration unavailable")
            if self._processing_lifecycle_busy():
                raise RuntimeError("stop the active session before connecting video")
            connection = self._video_connection
            if connection is not None and connection.running:
                raise RuntimeError("video is already connected or disconnecting")
            url = (stream_url or "").strip() or self.configured_stream_url()
            parse_stream_source(url)
            layout = validate_camera_layout(
                camera_layout or self.configured_camera_layout(),
            )
            connection = VideoConnection(
                url, output_fps=max(
                    self.config.display.fps, self.config.video.fps,
                    self.config.face.sample_fps, self.config.events.fps,
                ),
                whep_headers=self.config.stream.whep_headers,
                frame_crop="left_half" if layout == "stereo" else "none",
            )
            self._video_connection = connection
            self._connection_camera_layout = layout
            connection.start()
        return self.status()

    def disconnect_stream(self) -> dict[str, Any]:
        with self._control_lock:
            if self._processing_lifecycle_busy():
                raise RuntimeError("stop the active session before disconnecting video")
            if self._video_connection is not None:
                self._video_connection.stop()
        return self.status()

    def prepare_analysis(self) -> dict[str, Any]:
        with self._control_lock:
            if self.config is None or self.config_error is not None or self._closed:
                raise RuntimeError(self.config_error or "configuration unavailable")
            connection = self._video_connection
            if connection is None or not connection.status()["connected"]:
                raise RuntimeError("请先连接视频并等待画面就绪。")
            if self._processing_lifecycle_busy():
                raise RuntimeError("stop the active session before starting analysis")
            if self._route_probe["state"] == "running":
                raise RuntimeError("wait for the route check before starting analysis")
            features = self._analysis_settings["features"]
            if (not self.dry_run
                    and any(item in SAFETY_EVENT_LABELS for item in features)
                    and not self._aiclub_api_key
                    and self.config.model_api.provider != "local"):
                raise MissingAPIKeyError(PUBLIC_MISSING_API_KEY_ERROR)
        self._require_processing_ready()
        if not self.dry_run and "stranger" in features:
            self._ensure_face_runtime().warmup()
        with self._control_lock:
            if (
                self._video_connection is not connection
                or not connection.status()["connected"]
            ):
                raise RuntimeError("视频连接已中断。请重新连接。")
        return {"prepared": True, "connection_id": connection.status()["connection_id"]}

    def start_connected_analysis(self) -> dict[str, Any]:
        self.prepare_analysis()
        with self._control_lock:
            connection = self._video_connection
            if connection is None or not connection.status()["connected"]:
                raise RuntimeError("视频连接已中断。请重新连接。")
            choices = copy.deepcopy(self._analysis_settings)
            layout = self._connection_camera_layout
        return self.start_stream(
            connection.stream_url, camera_layout=layout,
            _connection_id=connection.connection_id, **choices,
        )

    def start_stream(
        self,
        stream_url: str | None = None,
        *,
        dry_run: bool | None = None,
        event_label: str | None = None,
        analysis_mode: str = MULTI_ANALYSIS_MODE,
        features: Collection[str] | None = None,
        event_sensitivities: Mapping[str, str] | None = None,
        camera_layout: str | None = None,
        _connection_id: str | None = None,
    ) -> dict[str, Any]:
        if camera_layout is not None:
            validate_camera_layout(camera_layout)
        url = (stream_url or "").strip() or self.configured_stream_url()
        if not url:
            raise ValueError("provide an RTSP, RTMP, or WebRTC/WHEP stream URL")
        source = parse_stream_source(url)
        return self._start_session(
            STREAM_MODE,
            stream_url=url,
            stream_protocol=source.protocol,
            stream_host=source.host,
            dry_run=dry_run,
            event_label=event_label,
            analysis_mode=analysis_mode,
            features=features,
            event_sensitivities=event_sensitivities,
            camera_layout=camera_layout,
            video_connection_id=_connection_id,
        )

    def start_camera(
        self,
        *,
        dry_run: bool | None = None,
        event_label: str | None = None,
        analysis_mode: str = MULTI_ANALYSIS_MODE,
        features: Collection[str] | None = None,
        event_sensitivities: Mapping[str, str] | None = None,
        owner: str | None = None,
    ) -> dict[str, Any]:
        return self._start_session(
            CAMERA_MODE,
            dry_run=dry_run,
            event_label=event_label,
            analysis_mode=analysis_mode,
            features=features,
            event_sensitivities=event_sensitivities,
            camera_owner=owner,
        )

    def _start_session(
        self, mode: str, **kwargs: Any,
    ) -> dict[str, Any]:
        # Admission remains visible while readiness checks run outside the
        # control lock, even if the caller's HTTP connection disappears.
        with self._control_lock:
            if self._session_start_pending:
                raise RuntimeError("a harness session is already starting")
            self._session_start_pending = True
            self._session_start_cancel.clear()
            self._pending_start_owner = kwargs.get("camera_owner")
        try:
            return self._start_session_impl(mode, **kwargs)
        finally:
            with self._control_lock:
                self._session_start_pending = False
                self._pending_start_owner = None

    def _start_session_impl(
        self,
        mode: str,
        *,
        stream_url: str | None = None,
        stream_protocol: str | None = None,
        stream_host: str | None = None,
        dry_run: bool | None = None,
        event_label: str | None = None,
        analysis_mode: str = MULTI_ANALYSIS_MODE,
        features: Collection[str] | None = None,
        event_sensitivities: Mapping[str, str] | None = None,
        camera_owner: str | None = None,
        camera_layout: str | None = None,
        video_connection_id: str | None = None,
    ) -> dict[str, Any]:
        if self.config is None or self.config_error is not None:
            raise RuntimeError(self.config_error or "configuration unavailable")
        if self._fleet_camera_index > self.tracking_camera_capacity():
            raise RuntimeError("该视频通道超出人员追踪模块的配置容量。")
        self._ensure_rich_readiness_probe()
        if mode not in {CAMERA_MODE, STREAM_MODE}:
            raise ValueError("unsupported input mode")

        selected_analysis = str(analysis_mode).strip().lower()
        if selected_analysis not in ANALYSIS_MODES:
            raise ValueError("analysis_mode must be event, face, or multi")
        if features is not None and selected_analysis != MULTI_ANALYSIS_MODE:
            raise ValueError("features can only be used with analysis_mode=multi")

        if selected_analysis == MULTI_ANALYSIS_MODE:
            selected_features = _normalize_features(features)
            label = ",".join(selected_features)
            use_dry_run = self.dry_run if dry_run is None else bool(dry_run)
        elif selected_analysis == FACE_ANALYSIS_MODE:
            selected_features = None
            label = UNKNOWN_FACE_EVENT_LABEL
            use_dry_run = False
        else:
            selected_features = None
            face_reference_memory = (
                self.config.memory is not None
                and self.config.memory.purpose == FACE_RECOGNITION_PURPOSE
            )
            if face_reference_memory:
                label = UNKNOWN_FACE_EVENT_LABEL
            else:
                label = (event_label or self.config.inference.event_label).strip()
                if not label:
                    label = self.config.inference.event_label
            label = label[:EVENT_LABEL_MAX_CHARS]
            use_dry_run = self.dry_run if dry_run is None else bool(dry_run)

        sensitivity_labels = selected_features or (label,)
        if selected_analysis == EVENT_ANALYSIS_MODE and self.config.memory is not None:
            sensitivity_labels = ()
        selected_sensitivities = normalize_event_sensitivities(
            event_sensitivities, sensitivity_labels,
        )

        processing_ready = self._require_processing_ready()

        with self._control_lock:
            if self._closed:
                raise RuntimeError("harness service is stopping")
            if video_connection_id is not None and (
                self._video_connection is None
                or self._video_connection.connection_id != video_connection_id
                or not self._video_connection.status()["connected"]
            ):
                raise RuntimeError("视频连接已变化。请重新连接后统一开始分析。")
            if self._session_start_cancel.is_set():
                raise RuntimeError("session start cancelled")
            if self._route_probe["state"] == "running":
                raise RuntimeError("wait for the route check before starting a session")
            if self._state.get("pipeline_state") == "draining" or (
                self._session is not None and self._session.running
            ):
                raise RuntimeError(
                    "a harness session is already running; stop it first"
                )
            if processing_ready is not None and (
                processing_ready != (self._rich_config_revision, self._rich_labels)
                or self._processing_lifecycle_busy(include_pending_start=False)
            ):
                raise RuntimeError("人物处理方式或会话状态已变化。请重新开始检测。")
            needs_model_key = selected_analysis == EVENT_ANALYSIS_MODE or (
                selected_analysis == MULTI_ANALYSIS_MODE
                and any(feature in SAFETY_EVENT_LABELS for feature in selected_features)
            )
            if (needs_model_key and not use_dry_run and not self._aiclub_api_key
                    and self.config.model_api.provider != "local"):
                raise MissingAPIKeyError(
                    "missing model API key; set " + self.config.model_api.api_key_env
                )
            self._ensure_delivery()

            face_session = None
            scene_active = self._scene_mode == "active"
            if scene_active and selected_analysis == EVENT_ANALYSIS_MODE:
                raise ValueError(
                    "continuous scene mode requires multi-feature or face analysis"
                )
            if (
                scene_active
                and not use_dry_run
                and (
                    self._rich_labels is None
                    or self.config.rich_labeling.backend == "vlm"
                )
            ):
                raise RuntimeError(
                    "continuous scene mode requires a ready person tracking module"
                )
            if scene_active and self._scene_archive is None:
                self._scene_archive = SceneArchive(
                    self.config.events.directory / "scene",
                    retention_days=self.config.events.retention_days,
                )
            if selected_analysis == FACE_ANALYSIS_MODE:
                face_runtime = self._ensure_face_runtime()
                warmup_seconds = face_runtime.warmup()
                if warmup_seconds:
                    LOGGER.info(
                        "warmed local face analysis in %.3fs", warmup_seconds
                    )
                face = self.config.face
                face_live_scheduler = _resolved_face_live_scheduler(
                    provider=face.provider,
                    configured=face.live_scheduler,
                )
                session_config = replace(
                    self.config,
                    video=replace(
                        self.config.video,
                        fps=face.sample_fps,
                        max_image_side=face.max_image_side,
                        jpeg_quality=face.jpeg_quality,
                    ),
                    sliding_window=replace(
                        self.config.sliding_window,
                        size_frames=1,
                        stride_frames=1,
                        include_partial_final_window=False,
                    ),
                    inference=face_runtime.settings,
                    runtime=(
                        replace(
                            self.config.runtime,
                            max_concurrent_requests=1,
                        )
                        if face_live_scheduler == "single_flight"
                        else self.config.runtime
                    ),
                )
                if isinstance(face_runtime, FaceRecognitionRuntime):
                    face_session = FaceRecognitionSession(
                        face_runtime,
                        observation_store_path=(
                            self.config.events.directory / "private-face"
                        ),
                    )
                client = face_session or face_runtime
            elif selected_analysis == MULTI_ANALYSIS_MODE:
                face = self.config.face
                face_live_scheduler = _resolved_face_live_scheduler(
                    provider=face.provider,
                    configured=face.live_scheduler,
                )
                include_stranger = "stranger" in selected_features
                face_runtime = None
                if include_stranger and not use_dry_run:
                    face_runtime = self._ensure_face_runtime()
                    warmup_seconds = face_runtime.warmup()
                    if warmup_seconds:
                        LOGGER.info(
                            "warmed local face analysis in %.3fs", warmup_seconds
                        )
                    if isinstance(face_runtime, FaceRecognitionRuntime):
                        face_session = FaceRecognitionSession(
                            face_runtime,
                            observation_store_path=(
                                self.config.events.directory / "private-face"
                            ),
                        )
                session_config = self.config
                if include_stranger:
                    session_config = replace(
                        self.config,
                        video=replace(
                            self.config.video,
                            fps=face.sample_fps,
                            max_image_side=face.max_image_side,
                            jpeg_quality=face.jpeg_quality,
                        ),
                        sliding_window=replace(
                            self.config.sliding_window,
                            size_frames=1,
                            stride_frames=1,
                            include_partial_final_window=False,
                        ),
                        # CPU single-flight keeps stranger-only sessions at one
                        # request and drops only busy face branches in mixed
                        # sessions. Full mode keeps the configured request
                        # concurrency and evaluates InsightFace for every
                        # admitted request, including DirectML and CUDA auto.
                        runtime=(
                            replace(
                                self.config.runtime,
                                max_concurrent_requests=1,
                            )
                            if (
                                selected_features == ("stranger",)
                                and face_live_scheduler == "single_flight"
                            )
                            else self.config.runtime
                        ),
                    )
                frame_features = (
                    (("stranger",) if include_stranger else ())
                    if scene_active
                    else selected_features
                )
                client = (
                    build_selected_features_grounding_client(
                        session_config.inference,
                        session_config.model_api,
                        features=frame_features,
                        event_sensitivities=selected_sensitivities,
                        stranger_client=face_session or face_runtime,
                        safety_max_image_side=self.config.video.max_image_side,
                        safety_jpeg_quality=self.config.video.jpeg_quality,
                        face_live_scheduler=face_live_scheduler,
                        dry_run=use_dry_run,
                        api_key=self._aiclub_api_key,
                        transport=(
                            self._pooled_transport()
                            if self._route_pool_enabled
                            and not use_dry_run
                            and any(item != "stranger" for item in frame_features)
                            else None
                        ),
                    )
                    if frame_features
                    else SceneCaptureClient(session_config.inference, selected_features)
                )
            else:
                session_config = self.config
                if label != self.config.inference.event_label:
                    session_config = replace(
                        self.config,
                        inference=replace(self.config.inference, event_label=label),
                    )
                client = build_grounding_client(
                    session_config.inference,
                    session_config.model_api,
                    memory_context=session_config.memory,
                    event_sensitivities=selected_sensitivities,
                    dry_run=use_dry_run,
                    api_key=self._aiclub_api_key,
                    transport=(
                        self._pooled_transport()
                        if self._route_pool_enabled and not use_dry_run else None
                    ),
                )
            if scene_active:
                # The continuous tracker and temporal inspector consume these
                # source pictures. Do not inherit the legacy 320px safety crop
                # budget for browser capture or source-only session admission.
                session_config = replace(
                    session_config,
                    video=replace(
                        session_config.video,
                        max_image_side=max(
                            session_config.video.max_image_side,
                            self.config.runtime.scene_max_image_side,
                        ),
                    ),
                )
            if mode == STREAM_MODE:
                crop = self.configured_frame_crop() if camera_layout is None else (
                    "left_half" if camera_layout == "stereo" else "none"
                )
                session_config = replace(
                    session_config,
                    stream=replace(session_config.stream, frame_crop=crop),
                )
            generation = self._generation + 1
            callbacks = SessionCallbacks(
                on_display_frame=(
                    (
                        lambda jpeg, timestamp, sequence: self._guarded_callback(
                            generation,
                            self._on_display_frame,
                            jpeg,
                            timestamp,
                            sequence,
                        )
                    )
                    if mode == STREAM_MODE
                    else None
                ),
                on_state=lambda state: self._guarded_callback(
                    generation, self._on_pipeline_state, state
                ),
                on_end=lambda reason, error: self._guarded_callback(
                    generation, self._on_end, reason, error
                ),
                on_analysis_frame=lambda result, jpeg: self._guarded_callback(
                    generation, self._on_analysis_frame, result, jpeg
                ),
                on_analysis_result=lambda result: self._guarded_callback(
                    generation, self._on_analysis_result, result
                ),
                on_timeline_result=lambda result: self._guarded_callback(
                    generation, self._on_timeline_result, result
                ),
            )
            if mode == STREAM_MODE:
                assert stream_url is not None
                connection = self._video_connection
                stream_options = {}
                if video_connection_id is not None and (
                    connection is None
                    or connection.connection_id != video_connection_id
                    or not connection.status()["connected"]
                ):
                    raise RuntimeError("视频连接已变化。请重新连接后统一开始分析。")
                if connection is not None and connection.running:
                    if stream_url != connection.stream_url:
                        raise RuntimeError(
                            "disconnect the current video before changing source",
                        )
                    if not connection.status()["connected"]:
                        raise RuntimeError("video connection is not ready")
                    stream_options["video_stream"] = connection.subscribe()
                session = StreamingSession.for_stream(
                    session_config,
                    client,
                    callbacks,
                    stream_url=stream_url,
                    whep_headers=session_config.stream.whep_headers,
                    source_id=self.source_id,
                    archive=self.event_archive(),
                    **stream_options,
                )
            else:
                session = StreamingSession.for_camera(
                    session_config,
                    client,
                    callbacks,
                    source_id=self.source_id,
                    archive=self.event_archive(),
                )

            session.defer_archive_end = True
            self._archive_run(session)
            if face_session is not None:
                face_session.bind_rich_labels(
                    session.stream_session_id,
                    self._rich_history if self._tracking_requested else None,
                )

            scene_labels = tuple(
                item
                for item in (selected_features or ())
                if item in SAFETY_EVENT_LABELS
            )
            if (
                selected_analysis == EVENT_ANALYSIS_MODE
                and label in SAFETY_EVENT_LABELS
            ):
                scene_labels = (label,)
            next_scene = (
                SceneRuntime(
                    session.stream_session_id,
                    self.source_id,
                    mode=self._scene_mode,
                    event_labels=scene_labels,
                    stranger_enabled=(
                        selected_analysis == FACE_ANALYSIS_MODE
                        or "stranger" in (selected_features or ())
                    ),
                    event_sensitivities=selected_sensitivities,
                    temporal_observation_fps=(
                        self.config.runtime.scene_observation_fps
                        if self.config.rich_labeling.backend == "sam31_stream"
                        and scene_labels else None
                    ),
                    max_pending_inspections=(
                        8 if self.config.rich_labeling.backend == "sam31_stream" else 4
                    ),
                    tracking_mode=(
                        "batch" if self.config.rich_labeling.backend in {
                            "sam31", "sam31_stream"
                        }
                        else "immediate"
                    ),
                    native_memory_scope=(
                        "session" if self.config.rich_labeling.backend == "sam31_stream"
                        else None
                    ),
                    tracking_history_seconds=(
                        continuous_source_history_seconds(
                            self.config.rich_labeling.max_fps
                        )
                        if self.config.rich_labeling.backend == "sam31_stream" else 12.0
                    ),
                )
                if self._scene_enabled
                else None
            )
            next_worker = None
            if scene_active and not use_dry_run and scene_labels:
                model_settings = replace(
                    session_config.model_api,
                    timeout_seconds=self.config.runtime.scene_timeout_seconds,
                    max_retries=0,
                    extra_headers=copy.deepcopy(dict(
                        session_config.model_api.extra_headers
                    )),
                )
                parameters = copy.deepcopy(session_config.inference.request_parameters)
                # Freeze credentials and routing for this generation. Retired
                # requests must never pick up a later run's replacement key.
                api_key = self._aiclub_api_key or (
                    environment_api_key(model_settings)
                    if model_settings.provider == "local" else ""
                )
                model = session_config.inference.model
                options = {
                    "max_tokens": self.config.runtime.scene_max_tokens,
                    "max_image_side": self.config.runtime.scene_max_image_side,
                    "jpeg_quality": self.config.runtime.scene_jpeg_quality,
                }

                def build_inspector():
                    return TemporalVLMInspector(
                        model_settings, model=model, api_key=api_key,
                        request_parameters=parameters, **options,
                    )

                parallel = self.config.rich_labeling.backend == "sam31_stream"
                size = self.config.runtime.scene_max_in_flight if parallel else 1
                fingerprint = hashlib.sha256(json.dumps({
                    "settings": asdict(model_settings), "model": model,
                    "parameters": parameters, "api_key": api_key, **options,
                    "transport_environment": {
                        name: os.environ.get(name) for name in (
                            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                            "http_proxy", "https_proxy", "all_proxy", "no_proxy",
                            "SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE",
                        )
                    },
                }, sort_keys=True, allow_nan=False).encode()).hexdigest()
                leases = iter(self._temporal_clients.prepare(
                    fingerprint, build_inspector, size,
                    accept=lambda job, scene=next_scene: scene.inspection_active(
                        job.request.request_id
                    ),
                ))
                next_worker = SceneInspectionWorker(
                    next_scene, next(leases), max_in_flight=size,
                    inspector_factory=(lambda: next(leases)) if parallel else None,
                    on_update=lambda: self._guarded_callback(
                        generation, self._on_scene_update
                    ),
                )
                next_scene.automatic_temporal_inference = True
            display_class = (
                TrackedSceneDisplayWorker
                if self.config.rich_labeling.backend == "sam31_stream"
                else SceneDisplayWorker
            )
            display_options = (
                {"delay_seconds": self._display_settings.delay_seconds}
                if display_class is TrackedSceneDisplayWorker else {}
            )
            next_display = (
                display_class(
                    next_scene,
                    lambda packet: self._guarded_callback(
                        generation, self.annotated_frames.publish, packet
                    ),
                    quality=min(95, self.settings.overlay_jpeg_quality),
                    max_fps=self._annotated_output_fps,
                    **display_options,
                )
                if scene_active
                else None
            )

            with self._state_lock:
                self._generation = generation
                self.face_fast_results.stop()
                self.face_fast_results = FaceFastResults(
                    stream_session_id=(session.stream_session_id
                                       if face_session is not None else None),
                    source_id=self.source_id, generation=generation,
                )
                if face_session is not None:
                    face_session.fast_results = self.face_fast_results
                # Reset the compositor before its public bus. A frame already
                # being published finishes first, then the bus reset clears it;
                # the new generation cannot leak an old session frame.
                if self._rich_labels is not None:
                    self._rich_labels.reset(session.stream_session_id)
                self._active_rich_session_id = (
                    session.stream_session_id if self._tracking_requested else None
                )
                self._overlay_fusion.reset()
                self._delayed_overlay.reset()
                self.annotated_frames.reset()
                self.results.reset()
                self._scene_result_publisher.reset(session.stream_session_id)
                self.scene_results.reset()
                self.frames.reset()
                self.preview_frames.reset()
                self._session = session
                self._camera_display_metadata.reset(
                    session.stream_session_id,
                    native_token=(self._rich_config_revision, id(self._rich_labels),
                                  self._rich_labels._generation)
                    if hasattr(self._rich_labels, "on_native_batch") else None,
                )
                self._last_camera_receipt = None
                if self._scene is not None:
                    self._scene.close()
                self._close_scene_workers()
                self._scene = next_scene
                self._last_scene_snapshot = None
                self._scene_worker = next_worker
                self._identity_worker = IdentityReviewWorker(
                    lambda: self._review_session_identities(generation, session)
                )
                self._scene_display = next_display
                self._last_scene_inference_monotonic = None
                self._scene_success_count = 0
                self._last_scene_publish_monotonic = 0.0
                self._scene_errors = 0
                self._session_mode = mode
                self._camera_owner = camera_owner if mode == CAMERA_MODE else None
                self._camera_frame_sequence = 0
                self._camera_sample_period = 1.0 / session_config.video.fps
                self._camera_next_sample_at = 0.0
                self._camera_max_image_side = session_config.video.max_image_side
                self._session_started_monotonic = time.monotonic()
                self._last_frame_monotonic = None
                self._last_result_monotonic = None
                self._last_processing_stats = None
                self._last_identity_review = None
                self._pending_archive_subject_merges = {}
                self._pending_archive_scene = None
                self._state.update(
                    {
                        "pipeline_state": "starting",
                        "input_mode": mode,
                        "analysis_mode": selected_analysis,
                        "generation": generation,
                        "last_frame_at": None,
                        "last_result_at": None,
                        "source_frames_received": 0,
                        "analysis_frames_published": 0,
                        "results_published": 0,
                        "last_error": None,
                        "end_reason": None,
                        "scene_drain": None,
                        "local_camera_display": None,
                        "stream": (
                            {
                                "protocol": stream_protocol,
                                "host": stream_host,
                                "frame_crop": session_config.stream.frame_crop,
                                "camera_layout": (
                                    "mono" if session_config.stream.frame_crop == "none"
                                    else "stereo"
                                ),
                            }
                            if mode == STREAM_MODE
                            else None
                        ),
                        "stream_session_id": session.stream_session_id,
                        "event_label": label,
                        "event_labels": (
                            list(selected_features)
                            if selected_analysis == MULTI_ANALYSIS_MODE
                            else [label]
                        ),
                        "features": (
                            list(selected_features)
                            if selected_analysis == MULTI_ANALYSIS_MODE
                            else None
                        ),
                        "event_sensitivities": dict(selected_sensitivities),
                        "model": PUBLIC_MODEL_NAME,
                        "dry_run": use_dry_run,
                        "processing_method": self._processing_method(),
                        "processing_cleanup": None,
                        "webhook_enabled": self._delivery is not None,
                        "pipeline": None,
                    }
                )
            try:
                session.start()
                if next_worker is not None:
                    next_worker.start()
                if next_display is not None:
                    next_display.start()
            except Exception:
                with self._state_lock:
                    self._generation += 1
                    self.face_fast_results.stop()
                    if self._rich_labels is not None:
                        self._rich_labels.reset(None)
                    self._active_rich_session_id = None
                    self._session = None
                    if self._scene is not None:
                        self._scene.close()
                    self._scene = None
                    self._close_scene_workers()
                    self._session_mode = None
                    self._camera_owner = None
                    self._state.update(
                        {
                            "pipeline_state": "error",
                            "input_mode": None,
                            "last_error": "could not start streaming session",
                        }
                    )
                session.stop(drain=False)
                session.join(timeout=1.0)
                control = self._archive_run(session)
                with control.lock:
                    control.state, control.error = "failed", "startup_failed"
                if session.archive is not None:
                    session.archive.end_session(
                        session.stream_session_id, "startup_failed",
                    )
                raise

            response: dict[str, Any] = {
                "type": "session",
                "mode": mode,
                "analysis_mode": selected_analysis,
                "generation": generation,
                "stream_session_id": session.stream_session_id,
                "event_label": label,
                "model": PUBLIC_MODEL_NAME,
                "dry_run": use_dry_run,
                "processing_method": self._processing_method(),
                "event_sensitivities": dict(selected_sensitivities),
            }
            if selected_analysis == MULTI_ANALYSIS_MODE:
                response["event_labels"] = list(selected_features)
                response["features"] = list(selected_features)
            if stream_protocol is not None:
                response["stream_protocol"] = stream_protocol
                response["camera_layout"] = (
                    "mono" if session_config.stream.frame_crop == "none" else "stereo"
                )
            if (
                selected_analysis == EVENT_ANALYSIS_MODE
                and session_config.memory is not None
            ):
                response["reference_memory"] = {
                    "purpose": session_config.memory.purpose,
                    "reference_count": len(session_config.memory.references),
                    "newest_frame_only": session_config.memory.newest_frame_only,
                    "memory_ids": list(session_config.memory.memory_ids),
                }
            safe_response = public_safe_value(response)
            assert isinstance(safe_response, dict)
            return safe_response

    def stop_session(self, *, camera_owner: str | None = None) -> bool:
        """Stop capture, then finish accepted review and evidence in the background."""

        with self._control_lock:
            if getattr(self, "_session_start_pending", False) and (
                camera_owner is None or camera_owner == self._pending_start_owner
            ):
                self._session_start_cancel.set()
                if self._session is None:
                    return True
            with self._state_lock:
                if (
                    camera_owner is not None
                    and (
                        self._session_mode != CAMERA_MODE
                        or self._camera_owner != camera_owner
                    )
                ):
                    return False
                session = self._session
                if session is None:
                    if self._rich_labels is not None:
                        self._rich_labels.reset(None)
                    self._overlay_fusion.reset()
                    self._active_rich_session_id = None
                    self._state.update(
                        {"pipeline_state": "idle", "input_mode": None, "ready": False}
                    )
                    return False
                if self._session_mode == CAMERA_MODE and isinstance(
                    self._rich_labels, VideoSessionTrackingCoordinator
                ):
                    self._rich_labels.mark_capture_finished(
                        stream_session_id=session.stream_session_id,
                        generation=self._rich_labels._generation,
                    )
                self._stop_requested_session = session
                if self._state.get("pipeline_state") not in {
                    "draining", "ended", "error",
                }:
                    self._update_state(
                        pipeline_state="draining", ready=False,
                        scene_drain={"state": "running", "stage": "detection"},
                    )
                if self._state.get("pipeline_state") in {"ended", "error"}:
                    self._release_stopped_session(session)
                    return True

            session.stop(drain=True)
            # Quick empty runs finish inline; slow model/encoding work stays
            # visible as draining and cannot be replaced by a new generation.
            session.join(timeout=0.1)
            thread = getattr(self, "_end_drain_thread", None)
            if thread is not None:
                thread.join(timeout=0.15)
            return True

    def _release_stopped_session(self, session: StreamingSession) -> None:
        with self._state_lock:
            if session is not self._session:
                return
            if self._processing_method() is not None:
                self._last_processing_stats = processing_status(
                    self._rich_labels, self._processing_method(),
                )
            self._generation += 1
            self._delayed_overlay.reset()
            self.face_fast_results.stop()
            self._overlay_fusion.reset()
            if self._rich_labels is not None:
                self._rich_labels.reset(None)
            self.annotated_frames.reset()
            self._active_rich_session_id = None
            self._last_identity_review = session.person_registry.identity_review_stats()
            self._session = None
            self._stop_requested_session = None
            self._session_mode = None
            self._camera_owner = None
            self._session_started_monotonic = None
            self._camera_max_image_side = self.config.video.max_image_side
            self._update_state(
                pipeline_state=(
                    "error" if self._state.get("last_error") else "idle"
                ),
                input_mode=None, ready=False, stream=None,
                generation=self._generation,
            )

    def stop_camera_owned_by(self, owner: str) -> None:
        self.stop_session(camera_owner=owner)

    def stop(self) -> None:
        with self._control_lock:
            self._closed = True
            self._route_probe_stop.set()
            self._rich_probe_stop.set()
        self.stop_session()
        connection = getattr(self, "_video_connection", None)
        if connection is not None:
            connection.stop(timeout=1.0)
        session = self._session
        if session is not None:
            session.join()
        drain_thread = getattr(self, "_end_drain_thread", None)
        if drain_thread is not None:
            drain_thread.join()
        self._postprocess_pool.shutdown(wait=True)
        self._temporal_clients.close(timeout_seconds=3.0)
        if self._rich_probe_thread is not None:
            self._rich_probe_thread.join(timeout=3.0)
        if self._rich_retirement_thread is not None:
            self._rich_retirement_thread.join(timeout=1.0)
        probe_thread = self._route_probe_thread
        if probe_thread is not None:
            probe_thread.join(timeout=1.0)
        with self._control_lock:
            if self._route_transport is not None and (
                probe_thread is None or not probe_thread.is_alive()
            ):
                self._route_transport.close()
                self._route_transport = None
        self._delayed_overlay.close(timeout_seconds=5.0)
        if self._rich_labels is not None:
            self._rich_labels.close(timeout_seconds=5.0)
        if self._delivery is not None:
            self._delivery.close(timeout_seconds=5.0)
        if self._events is not None:
            self._events.close(timeout=15.0)
        if self._scene_archive is not None:
            self._scene_archive.close(timeout_seconds=1.0)
        self._update_state(pipeline_state="stopped", ready=False)

    def push_camera_frame(
        self,
        jpeg: bytes,
        *,
        width: int,
        height: int,
        owner: str | None = None,
        stream_session_id: str | None = None,
        generation: int | None = None,
        crop_version: int | None = None,
        source_sequence: int | None = None,
        media_timestamp: float | None = None,
    ) -> bool:
        started = time.monotonic()
        timing = {
            "stream_session_id": trace_id(stream_session_id),
            "generation": generation if type(generation) is int else None,
            "source_sequence": (
                source_sequence if type(source_sequence) is int else None
            ),
            "input_bytes": len(jpeg) if isinstance(jpeg, bytes) else None,
            "width": width if type(width) is int else None,
            "height": height if type(height) is int else None,
            "outcome": "rejected",
        }

        def phase(name):
            now = time.monotonic()
            previous = timing.pop("_phase", None)
            phase_started = timing.pop("_phase_started", None)
            if previous is not None:
                timing[previous + "_seconds"] = now - phase_started
            if name is not None:
                timing["_phase"] = name
                timing["_phase_started"] = now

        phase("validation")
        try:
            accepted = self._push_camera_frame_timed(
                jpeg, width=width, height=height, owner=owner,
                stream_session_id=stream_session_id, generation=generation,
                crop_version=crop_version, source_sequence=source_sequence,
                media_timestamp=media_timestamp, timing=timing, phase=phase,
            )
            if accepted and timing["outcome"] != "duplicate":
                timing["outcome"] = "accepted"
            return accepted
        except Exception as exc:
            timing.update(outcome="error", error_type=type(exc).__name__)
            raise
        finally:
            phase(None)
            timing["total_seconds"] = time.monotonic() - started
            log_timing("camera_ingress_timing", **timing)

    def _push_camera_frame_timed(
        self, jpeg: bytes, *, width: int, height: int, owner: str | None,
        stream_session_id: str | None, generation: int | None,
        crop_version: int | None, source_sequence: int | None,
        media_timestamp: float | None, timing: dict, phase: Callable,
    ) -> bool:
        if self.config is None:
            return False
        if not jpeg or len(jpeg) > CAMERA_FRAME_MAX_BYTES:
            raise ValueError("camera JPEG is empty or too large")
        if width < 1 or height < 1:
            raise ValueError("camera frame dimensions must be positive")
        max_image_side = self._camera_max_image_side
        if max(width, height) > max_image_side:
            raise ValueError(
                f"camera frame exceeds {max_image_side}px"
            )
        phase("decode")
        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("camera payload is not a valid JPEG")
        actual_height, actual_width = image.shape[:2]
        if (actual_width, actual_height) != (width, height):
            raise ValueError("camera JPEG dimensions do not match its header")
        phase("display_encode")
        # The local GVC2 display already owns the frozen source canvas. With no
        # requested server resize, preserve the validated source JPEG directly;
        # another lossy encode changes neither geometry nor display cadence.
        passthrough = (
            stream_session_id is not None
            and self.config.rich_labeling.backend == "sam31_stream"
            and self._scene_is_active()
            and self._display_settings.max_image_side is None
            and os.environ.get("LOCAL_CAMERA_DISPLAY_ENABLED", "false").lower()
            in {"1", "true", "yes"}
        )
        timing["display_source_reused"] = passthrough
        if passthrough:
            display_jpeg = jpeg
        else:
            display_jpeg, _, _ = encode_jpeg(
                image,
                max_image_side=self._display_settings.max_image_side,
                quality=self._display_settings.jpeg_quality,
            )
        timing["display_bytes"] = len(display_jpeg)

        phase("control_lock_wait")
        with self._control_lock:
            phase("source_admission")
            session = self._session
            if (
                session is None
                or self._session_mode != CAMERA_MODE
                or self._state.get("pipeline_state") == "draining"
                or not session.running
                or (self._camera_owner is not None and owner != self._camera_owner)
            ):
                return False
            source_header = None
            receipt = None
            if stream_session_id is not None:
                source_header = {
                    "stream_session_id": stream_session_id, "generation": generation,
                    "crop_version": crop_version, "source_sequence": source_sequence,
                    "media_timestamp": media_timestamp,
                }
                receipt = (
                    stream_session_id, generation, crop_version, source_sequence,
                    media_timestamp, hashlib.sha256(jpeg).digest(),
                )
                # A lost ACK may resend the same immutable source. Do not admit
                # it to either analysis branch a second time.
                if self._last_camera_receipt == receipt:
                    timing["outcome"] = "duplicate"
                    return True
                if not self._camera_display_metadata.accept_source(
                    source_header, commit=False,
                ):
                    return False
                sequence, timestamp = source_sequence, media_timestamp
            else:
                sequence = self._camera_frame_sequence
                started = self._session_started_monotonic or time.monotonic()
                timestamp = max(0.0, time.monotonic() - started)
            timing.update(
                stream_session_id=trace_id(session.stream_session_id),
                source_sequence=sequence, source_timestamp_seconds=timestamp,
            )
            should_sample = timestamp + 1e-9 >= self._camera_next_sample_at
            timing["sampled"] = should_sample
            if should_sample:
                phase("source_queue")
                if not session.push_camera_frame(
                    jpeg, width=width, height=height,
                    timestamp_seconds=timestamp, source_frame_index=sequence,
                ):
                    return False
                phase("source_metadata")
                while self._camera_next_sample_at <= timestamp + 1e-9:
                    self._camera_next_sample_at += self._camera_sample_period
            if source_header is not None:
                self._camera_display_metadata.accept_source(source_header)
                self._last_camera_receipt = receipt
            self._camera_frame_sequence = sequence + 1
            generation = self._generation
            phase("display_callback_state_wait")
            with self._state_lock:
                phase("display_callback")
                self._guarded_callback(
                    generation,
                    self._on_display_frame,
                    display_jpeg,
                    timestamp,
                    sequence,
                )
            return True

    def record_display_metrics(self, report: dict) -> dict:
        with self._state_lock:
            current = self._camera_display_metadata.source_identity()
            if any(report[key] != current[key] for key in (
                "stream_session_id", "generation", "crop_version",
            )):
                return {"accepted": False, "reason": "obsolete_source"}
            previous = self._state.get("local_camera_display") or {}
            if report.get("report_sequence", -1) <= previous.get("report_sequence", -1):
                return {"accepted": False, "reason": "obsolete_report"}
            self._state["local_camera_display"] = report
        network = getattr(self._session, "mode", None) != CAMERA_MODE
        log_timing("network_video_display_timing" if network else "local_camera_display_timing", **report)
        return {"accepted": True}

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            payload = copy.deepcopy(self._state)
            postprocess = tuple(self._run_postprocess.values())
            rich_labels = self._rich_labels
            method = self._processing_method()
            last_processing_stats = (
                dict(self._last_processing_stats)
                if self._last_processing_stats is not None else None
            )
            active_rich_session_id = self._active_rich_session_id
            session = self._session
            scene_worker = self._scene_worker
            retiring = self._retiring_rich_labels is not None
            payload["processing_method"] = method
            rich_disabled_reason = self._rich_disabled_reason
            payload["rich_labeling_enabled"] = (
                self._rich_requested and rich_labels is not None
            )
            payload["person_tracking_enabled"] = rich_labels is not None
            selector = getattr(
                getattr(self, "_session", None), "frame_crop_selector", None,
            )
            if selector is not None and isinstance(payload.get("stream"), dict):
                payload["stream"]["crop_selection"] = selector.status()
        payload["postprocessing_runs"] = [run.status() for run in postprocess]
        payload["session_start_pending"] = self._session_start_pending
        if self._session_start_pending and payload.get("pipeline_state") not in {
            "running", "draining",
        }:
            payload["pipeline_state"] = (
                "stopping" if self._session_start_cancel.is_set() else "starting"
            )
        payload["configured_stream"] = bool(self.configured_stream_url())
        payload["camera_layout"] = self.configured_camera_layout()
        payload["event_archive"] = (
            self._events.status() if self._events else {"state": "idle"}
        )
        scene_display = self._scene_display
        if isinstance(scene_display, TrackedSceneDisplayWorker):
            output = scene_display.stats()
            latest = self.annotated_frames.latest()
            output["annotation_mode"] = (
                latest[1].annotation_mode if latest is not None else "none"
            )
            output["output_fps"] = min(
                self._annotated_output_fps, self.config.rich_labeling.max_fps,
            )
            payload["annotated_output"] = output
        else:
            payload["annotated_output"] = self._delayed_overlay.stats()
        payload["rich_labeling"] = (
            dict(rich_labels.stats())
            if rich_labels is not None
            else {
                "state": "disabled",
                **({"disabled_reason": rich_disabled_reason}
                   if rich_disabled_reason else {}),
            }
        )
        tracking = payload["rich_labeling"]
        if tracking.get("last_error"):
            tracking["last_error"] = PUBLIC_TRACKING_ERROR
        live_failure = None
        if session is not None:
            # Report the current run's terminal branch failure on the next
            # status poll. This does not stop producers or discard accepted work.
            if active_rich_session_id == session.stream_session_id and (
                tracking.get("state") == "failed"
                or tracking.get("recovery_required") is True
            ):
                live_failure = (PUBLIC_TRACKING_ERROR, "person_tracking_failed")
            elif (
                scene_worker is not None
                and scene_worker.stats().get("state") == "failed"
            ):
                live_failure = (PUBLIC_INSPECTION_ERROR, "event_analysis_failed")
        payload["processing"] = (
            last_processing_stats
            if active_rich_session_id is None
            and last_processing_stats is not None
            and last_processing_stats.get("method") == method
            else processing_status(rich_labels, method)
        )
        if retiring or (method is not None and session is None
                and isinstance(rich_labels, VideoSessionTrackingCoordinator)
                and not rich_labels.wait_until_idle(timeout_seconds=0)):
            payload["pipeline_state"] = "stopping"
            payload["processing_cleanup"] = "closing"
        ready, reason = self.readiness()
        payload["ready"] = ready
        payload["readiness_reason"] = reason
        if live_failure is not None:
            payload["last_error"], payload["readiness_reason"] = live_failure
            payload["ready"] = False
        connection = getattr(self, "_video_connection", None)
        payload.update(connection.status() if connection is not None else {
            "connection_state": "disconnected", "connected": False,
            "connection_id": None,
        })
        if connection is not None and connection.running:
            payload["camera_layout"] = self._connection_camera_layout
        pipeline_state = payload.get("pipeline_state")
        payload["analysis_state"] = (
            "stopping" if pipeline_state in {"stopping", "draining"}
            else pipeline_state if pipeline_state in {"starting", "running", "error"}
            else "idle"
        )
        if self._delivery is not None:
            payload["webhook"] = self._delivery.stats()
        safe = public_safe_value(payload)
        assert isinstance(safe, dict)
        return safe

    def latest_rich_labels(self) -> dict[str, Any] | None:
        """Return the latest worker snapshot without exposing worker internals."""

        if self._rich_labels is None or not self._rich_requested:
            return None
        with self._state_lock:
            session_id = self._active_rich_session_id
        if session_id is None:
            return None
        snapshot = self._rich_labels.latest(session_id)
        if snapshot is None:
            return None
        safe = public_safe_value(snapshot.as_dict())
        assert isinstance(safe, dict)
        return safe

    def scene_snapshot(self, *, include_history: bool = False) -> dict[str, Any]:
        """Experimental scene state; no images or automatic inference side effects."""
        with self._state_lock:
            payload = (
                self._scene.snapshot(include_history=include_history)
                if self._scene is not None else copy.deepcopy(
                    getattr(self, "_last_scene_snapshot", None)
                ) or {
                    "enabled": self._scene_enabled,
                    "mode": self._scene_mode,
                    "state": "idle" if self._scene_enabled else "disabled",
                    "automatic_temporal_inference": False,
                    "scene": None, "inspections": [],
                }
            )
            payload["observer_errors"] = self._scene_errors
            payload["scene_drain"] = copy.deepcopy(self._state.get("scene_drain"))
            if self._session is not None:
                payload["identity_review"] = (
                    self._session.person_registry.identity_review_stats()
                )
                if self._identity_worker is not None:
                    payload["identity_review"].update(self._identity_worker.stats())
            elif getattr(self, "_last_identity_review", None) is not None:
                payload["identity_review"] = self._last_identity_review
            if self._scene_archive is not None:
                payload["archive"] = self._scene_archive.stats()
            if self._scene_worker is not None:
                payload["worker"] = self._scene_worker.stats()
            if isinstance(self._scene_display, TrackedSceneDisplayWorker):
                payload["display"] = self._scene_display.stats()
            if payload.get("scene") and not include_history:
                for item in payload["scene"]["entities"] + payload["scene"]["events"]:
                    item.pop("history", None)
        safe = public_safe_value(payload)
        assert isinstance(safe, dict)
        return safe

    def _close_scene_workers(self) -> None:
        # May hold the state lock: never join a callback waiting on that lock.
        for name in ("_scene_worker", "_scene_display", "_identity_worker"):
            worker = getattr(self, name, None)
            if worker is not None:
                worker.close(timeout_seconds=0)
                setattr(self, name, None)

    def _wake_identity_review(self) -> None:
        # Long-term identity consolidation belongs to postprocessing. Necessary
        # gallery/stranger classification remains in the online face branch.
        session = getattr(self, "_session", None)
        if isinstance(session, StreamingSession):
            return
        worker = getattr(self, "_identity_worker", None)
        if worker is not None:
            worker.wake()

    def _archive_run(self, session) -> RunPostprocess:
        with self._state_lock:
            return self._run_postprocess.setdefault(
                session.stream_session_id, RunPostprocess(session.stream_session_id),
            )

    def _queue_archive_subject_merges(self, session, merges=()) -> bool:
        """Retain control updates when the bounded archive ingress is full."""
        if session.archive is None:
            return True
        control = self._archive_run(session)
        with control.lock:
            control.pending_merges.update(merges)
            if not control.pending_merges:
                return True
            if session.archive.reconcile_subjects(
                session.stream_session_id, control.pending_merges,
            ):
                control.pending_merges.clear()
                return True
            return False

    def _review_session_identities(self, generation, session) -> None:
        registry = session.person_registry
        with self._state_lock:
            if generation != self._generation or session is not self._session:
                return
            scene = self._scene
            registry.review_pending_identities(accept_merge=(
                (lambda old, new: scene.apply_identity_merges(((old, new),)) > 0)
                if scene is not None else None
            ))
            merges = registry.drain_identity_merges()
            if scene is not None:
                scene.apply_identity_review(registry.identity_review_states())
            admitted = self._queue_archive_subject_merges(session, merges)
            if scene is not None:
                self._on_scene_update()
        while not admitted:
            archive = session.archive
            if archive.status()["state"] in {"error", "stopped"}:
                raise RuntimeError("identity archive update rejected")
            archive.flush(timeout=0.1)
            with self._state_lock:
                if generation != self._generation or session is not self._session:
                    return
                admitted = self._queue_archive_subject_merges(session)

    def _queue_archive_scene(self, session, snapshot=None) -> bool:
        """Keep accepted temporal episodes until the video archive admits them."""
        if session.archive is None:
            return True
        control = self._archive_run(session)
        with control.lock:
            pending = control.pending_scene
            if snapshot is not None:
                current = snapshot.get("scene")
                if not current or current.get("stream_session_id") != (
                    session.stream_session_id
                ):
                    return False
                previous = pending.get("scene", {}) if pending else {}
                events = {
                    event["event_id"]: event
                    for event in previous.get("events", ())
                }
                transitions = {
                    transition["event_id"]: transition
                    for transition in previous.get("transitions", ())
                }
                current_events = {
                    event["event_id"]: event
                    for event in current.get("events", ())
                    if event["event_label"] in SAFETY_EVENT_LABELS
                }
                for transition in current.get("transitions", ()):
                    if transition.get("event_label") not in SAFETY_EVENT_LABELS:
                        continue
                    transitions[transition["event_id"]] = transition
                for event_id, transition in transitions.items():
                    if event_id in current_events:
                        continue
                    if transition.get("type") == "retract":
                        events.pop(event_id, None)
                    elif (
                        transition.get("type") == "end"
                        or transition.get("phase") == "ended"
                    ) and event_id in events:
                        events[event_id] = {
                            **events[event_id],
                            "phase": "ended",
                            "last_evidence_timestamp_seconds": transition[
                                "timestamp_seconds"
                            ],
                            "frame_sequence": transition["frame_sequence"],
                        }
                events.update(current_events)
                pending = {
                    **snapshot,
                    "scene": {
                        **current,
                        "events": list(events.values()),
                        "transitions": list(transitions.values()),
                    },
                }
                control.pending_scene = pending
            if pending is None:
                return True
            if session.archive.submit_scene(pending):
                control.pending_scene = None
                return True
            return False

    def _on_scene_update(self) -> None:
        scene = getattr(self, "_scene", None)
        if scene is None:
            return
        archive_payload = self.scene_snapshot(include_history=False)
        payload = {
            **archive_payload,
            "scene": {
                **archive_payload["scene"],
                **{
                    key: [
                        {
                            name: value for name, value in item.items()
                            if name != "history"
                        }
                        for item in archive_payload["scene"][key]
                    ]
                    for key in ("entities", "events")
                },
            },
        }
        self.scene_results.publish({"type": "scene", **payload})
        self._publish_scene_result(payload)
        self._camera_display_metadata.observe_scene(payload)
        if isinstance(self._scene_display, TrackedSceneDisplayWorker):
            self._scene_display.notify_events(
                payload.get("scene", {}).get("events", [])
            )
        if self._scene_is_active():
            session = self._session
            if session is not None:
                self._queue_archive_scene(session, archive_payload)
            if self._scene_archive is not None:
                self._scene_archive.submit(payload)
        self._last_scene_publish_monotonic = time.monotonic()
        worker = getattr(self, "_scene_worker", None)
        if worker is not None:
            succeeded = worker.stats()["succeeded_requests"]
            if succeeded > getattr(self, "_scene_success_count", 0):
                self._last_scene_inference_monotonic = time.monotonic()
                self._scene_success_count = succeeded

    def _scene_is_active(self) -> bool:
        scene = getattr(self, "_scene", None)
        return scene is not None and scene.mode == "active"

    def _publish_scene_result(self, snapshot: Mapping[str, Any]) -> None:
        """Publish confirmed scene state through JSON/SSE/webhook, without feedback."""
        with self._state_lock:
            result = self._scene_result_publisher.project(
                snapshot,
                labels=tuple(self._state.get("event_labels", ())),
                emitted_at=_utc_now(),
            )
            if result is not None:
                self._publish_analysis_result(result)
                log_timing(
                    "scene_result_published",
                    stream_session_id=trace_id(result.stream_session_id),
                    message_id=trace_id(result.message_id),
                    frame_sequence=result.frame.source_frame_index,
                    timestamp_seconds=result.frame.timestamp_seconds,
                    judgments={j.event_label: j.detected for j in result.judgments},
                )

    def _observe_scene(self, method: str, *args: Any, **kwargs: Any) -> None:
        scene = getattr(self, "_scene", None)
        if scene is not None:
            try:
                getattr(scene, method)(*args, **kwargs)
                worker = getattr(self, "_scene_worker", None)
                if worker is not None:
                    worker.wake()
            except Exception:
                self._scene_errors += 1
                if self._scene_errors == 1:
                    LOGGER.warning(
                        "experimental scene observer rejected an observation"
                    )

    def readiness(self) -> tuple[bool, str]:
        with self._state_lock:
            state = self._state["pipeline_state"]
            last_frame = self._last_frame_monotonic
            last_result = self._last_result_monotonic
            session = self._session
        if state in {
            "configuration_error",
            "error",
            "stopped",
            "idle",
            "ended",
            "stopping",
            "draining",
        }:
            return False, str(state)
        if session is None or not session.running:
            return False, "pipeline_not_running"
        if last_frame is None:
            return False, "waiting_for_first_frame"
        now = time.monotonic()
        if now - last_frame > self.settings.ready_max_frame_age_seconds:
            return False, "frame_stale"
        if self._scene_is_active() and self._scene.event_labels:
            last_result = self._last_scene_inference_monotonic
        if last_result is None:
            return False, "waiting_for_first_analysis"
        if now - last_result > self.settings.ready_max_result_age_seconds:
            return False, "analysis_stale"
        return True, "ready"

    def _ensure_delivery(self) -> None:
        if self._delivery_initialized:
            return
        self._delivery = self._build_delivery()
        if self._delivery is not None:
            self._delivery.start()
        self._delivery_initialized = True

    def _build_delivery(self) -> WebhookDelivery | None:
        assert self.config is not None
        webhook = self.settings.webhook
        url = os.environ.get("RESULT_WEBHOOK_URL", "").strip() or webhook.url
        if not url:
            return None
        secret = os.environ.get(webhook.secret_env)
        return WebhookDelivery(
            url,
            secret=secret,
            timeout_seconds=webhook.timeout_seconds,
            max_retries=webhook.max_retries,
            retry_backoff_seconds=webhook.retry_backoff_seconds,
            queue_size=webhook.queue_size,
        )

    def _guarded_callback(
        self,
        generation: int,
        callback: Callable[..., None],
        *args: Any,
    ) -> None:
        with self._state_lock:
            if generation != self._generation:
                return
            # Keep generation validation and callback submission atomic with
            # session reset. Callback state updates are safe because this is an
            # RLock.
            callback(*args)

    def _on_analysis_frame(self, result: AnalysisResult, jpeg: bytes) -> None:
        source_jpeg = bytes(jpeg)
        scene = getattr(self, "_scene", None)
        if scene is not None and (
            result.stream_session_id == scene.stream_session_id
            and result.source_id == scene.source_id
        ):
            self._observe_scene(
                "observe_frame", source_jpeg, result.frame.timestamp_seconds,
                result.frame.source_frame_index, force=True,
            )
        if self._scene_is_active():
            # Mixed sessions still produce exact local-face observations. Keep
            # their original JPEG and analysis ID available without painting
            # current scene positions over an older source frame. Source-only
            # sessions never enter this callback and cannot invent evidence.
            if (
                result.stream_session_id != scene.stream_session_id
                or result.source_id != scene.source_id
            ):
                return
            latest = self.frames.latest()
            if latest is not None and (
                result.frame.timestamp_seconds, result.frame.source_frame_index
            ) < (latest[1].media_timestamp, latest[1].frame_sequence):
                return
            self.frames.publish(FramePacket(
                jpeg=source_jpeg,
                media_timestamp=result.frame.timestamp_seconds,
                frame_sequence=result.frame.source_frame_index,
                observed_at=result.frame.observed_at,
                analysis_message_id=result.message_id,
                annotation_mode="none",
            ))
            self._update_state(
                analysis_frames_published=(
                    int(self._state["analysis_frames_published"]) + 1
                ),
                last_analysis_frame_at=_utc_now(),
            )
            return
        if self._memory_sink is not None:
            try:
                accepted = self._memory_sink.submit(
                    VisualMemoryRecord(
                        observation=result.as_dict(),
                        evidence_jpeg=source_jpeg,
                    )
                )
            except Exception:
                LOGGER.warning("visual memory sink rejected an observation")
                accepted = False
            memory_key = (
                "memory_records_accepted" if accepted else "memory_records_dropped"
            )
            self._update_state(**{memory_key: int(self._state[memory_key]) + 1})
        latest = self.frames.latest()
        if latest is not None:
            _, packet = latest
            if (
                result.frame.timestamp_seconds, result.frame.source_frame_index
            ) < (packet.media_timestamp, packet.frame_sequence):
                # An independently completed branch retains its exact frame for
                # memory and the private timeline, but cannot rewind the preview.
                return
        rendered: bytes | None = None
        if self._rich_labels is not None and self._rich_requested:
            try:
                rendered = self._render_fused_label_jpeg(
                    source_jpeg,
                    OverlaySelection(result, "exact", result.message_id),
                    result.frame.timestamp_seconds,
                    result.frame.source_frame_index,
                )
            except Exception:
                LOGGER.warning("could not render fused labels on analysis frame")
        if rendered is None:
            # A failed fusion must not draw unassociated rich and event boxes
            # in succession. Preserve the event on the original source frame.
            try:
                rendered = _render_analysis_jpeg(
                    source_jpeg,
                    result,
                    quality=self.settings.overlay_jpeg_quality,
                )
            except (cv2.error, TypeError, ValueError):
                LOGGER.warning("could not render analysis overlay")
                rendered = source_jpeg
        self.frames.publish(
            FramePacket(
                jpeg=rendered,
                media_timestamp=result.frame.timestamp_seconds,
                frame_sequence=result.frame.source_frame_index,
                observed_at=result.frame.observed_at,
                analysis_message_id=result.message_id,
            )
        )
        frame_count = int(self._state["analysis_frames_published"]) + 1
        self._update_state(
            analysis_frames_published=frame_count,
            last_analysis_frame_at=_utc_now(),
        )

    def _on_analysis_result(self, result: AnalysisResult) -> None:
        if self._scene_is_active():
            # Raw face observations remain inputs to identity/scene and the fast
            # face endpoint. Do not let a candidate overwrite a gated verdict.
            self._scene_result_publisher.observe_reference(
                result.stream_session_id, result.frame,
            )
            self._on_scene_update()
            return
        self._publish_analysis_result(result)

    def _publish_analysis_result(self, result: AnalysisResult) -> None:
        payload = result.as_dict()
        self.results.publish(payload)
        result_count = int(self._state["results_published"]) + 1
        self._last_result_monotonic = time.monotonic()
        self._update_state(
            last_result_at=_utc_now(),
            results_published=result_count,
            stream_session_id=result.stream_session_id,
        )
        if self._delivery is not None:
            self._delivery.submit(payload)

    def _on_timeline_result(self, result: AnalysisResult) -> None:
        with self._state_lock:
            scene = getattr(self, "_scene", None)
            if scene is not None:
                session = self._session
                if (
                    session is not None
                    and session.stream_session_id == result.stream_session_id
                ):
                    registry = session.person_registry
                    merges = registry.drain_subject_merges()
                    scene.apply_subject_merges(merges)
                    self._queue_archive_subject_merges(session, merges)
                    canonical = registry.snapshot_at(
                        result.frame.timestamp_seconds, result.frame.source_frame_index,
                    )
                    if canonical is not None and (
                        canonical.timestamp_seconds, canonical.frame_sequence
                    ) == (
                        result.frame.timestamp_seconds, result.frame.source_frame_index,
                    ):
                        self._observe_scene("observe_tracks", canonical)
                self._observe_scene("observe_result", result)
            self._wake_identity_review()
        if self._scene_is_active():
            self._on_scene_update()
            return
        coordinator = self._rich_labels
        if (
            coordinator is not None
            and self._rich_requested
            and result.analysis_kind not in INDEPENDENT_FACE_ANALYSIS_KINDS
        ):
            with self._state_lock:
                session_id = self._active_rich_session_id
            if session_id == result.stream_session_id:
                try:
                    rich_selection = coordinator.select(
                        session_id,
                        result.frame.timestamp_seconds,
                        frame_sequence=result.frame.source_frame_index,
                    )
                    self._overlay_fusion.plan(
                        result,
                        rich_selection.objects,
                        association_message_id=result.message_id,
                    )
                except Exception:
                    LOGGER.warning("could not prime event-to-rich association")
        self._delayed_overlay.submit_result(result)

    def _on_display_frame(
        self,
        jpeg: bytes,
        media_timestamp: float,
        frame_sequence: int,
    ) -> None:
        observed_at = _utc_now()
        source_jpeg = bytes(jpeg)
        # Browser frames bypass StreamingSession's source decoder. Preserve all
        # presentation frames for evidence, not only the model samples.
        session = self._session
        if session is not None and getattr(session, "mode", None) != CAMERA_MODE:
            # Network capture owns the same source identity as its raw display
            # stream. Do not renumber frames or wait for SAM to bind metadata.
            self._camera_display_metadata.accept_source({
                "stream_session_id": session.stream_session_id,
                "generation": self._generation, "crop_version": 0,
                "source_sequence": int(frame_sequence),
                "media_timestamp": float(media_timestamp),
            })
        # Retain the exact samples admitted to a continuous model session. A
        # high-rate preview must not evict its slower, still-pending source JPEGs.
        # The callback takes this same lock, so it cannot precede source retention.
        continuous_queued = False
        with self._state_lock:
            if self._scene_is_active() and session is not None:
                self._scene_result_publisher.observe_frame(
                    session.stream_session_id, source_jpeg, media_timestamp,
                    frame_sequence, observed_at,
                )
            coordinator = self._rich_labels
            if isinstance(coordinator, VideoSessionTrackingCoordinator):
                continuous_queued = True
                admitted = False
                session_id = self._active_rich_session_id
                if session_id is not None:
                    try:
                        admitted = coordinator.submit(
                            source_jpeg,
                            stream_session_id=session_id,
                            frame_sequence=int(frame_sequence),
                            timestamp_seconds=float(media_timestamp),
                        )
                    except Exception:
                        LOGGER.warning("could not queue continuous tracking frame")
                self._observe_scene(
                    "observe_frame", source_jpeg, media_timestamp, frame_sequence,
                    retain_source=admitted,
                )
            else:
                self._observe_scene(
                    "observe_frame", source_jpeg, media_timestamp, frame_sequence,
                )
        if (
            session is not None and getattr(session, "mode", None) == CAMERA_MODE
            and getattr(session, "archive", None) is not None
        ):
            session.archive.submit_frame(
                source_jpeg, media_timestamp, frame_sequence, session.stream_session_id,
            )
        if self._rich_labels is not None and not continuous_queued:
            with self._state_lock:
                session_id = self._active_rich_session_id
            if session_id is not None:
                try:
                    self._rich_labels.submit(
                        source_jpeg,
                        stream_session_id=session_id,
                        frame_sequence=int(frame_sequence),
                        timestamp_seconds=float(media_timestamp),
                    )
                except Exception:
                    LOGGER.warning("could not queue frame for rich labeling")
        packet = FramePacket(
            jpeg=source_jpeg,
            media_timestamp=float(media_timestamp),
            frame_sequence=int(frame_sequence),
            observed_at=observed_at,
        )
        self.preview_frames.publish(packet)
        if self._scene_is_active():
            display = getattr(self, "_scene_display", None)
            if isinstance(display, TrackedSceneDisplayWorker):
                display.submit(packet, retain_source=continuous_queued and admitted)
            elif display is not None:
                display.submit(packet)
            else:
                self.annotated_frames.publish(packet)
            if time.monotonic() - self._last_scene_publish_monotonic >= 0.5:
                self._on_scene_update()
        else:
            self._delayed_overlay.submit_frame(packet)
        frame_count = int(self._state["source_frames_received"]) + 1
        self._last_frame_monotonic = time.monotonic()
        self._update_state(
            pipeline_state=(
                "draining" if self._state.get("pipeline_state") == "draining"
                else "running"
            ),
            last_frame_at=observed_at,
            source_frames_received=frame_count,
            last_media_timestamp=round(float(media_timestamp), 3),
            last_frame_sequence=int(frame_sequence),
        )

    def _render_rich_label_jpeg(
        self,
        jpeg: bytes,
        media_timestamp: float,
        frame_sequence: int,
    ) -> bytes:
        coordinator = self._rich_labels
        if coordinator is None:
            return jpeg
        with self._state_lock:
            session_id = self._active_rich_session_id
        if session_id is None:
            return jpeg
        selection = coordinator.select(
            session_id,
            float(media_timestamp),
            frame_sequence=int(frame_sequence),
            interpolate=False,
        )
        if not selection.objects:
            return jpeg
        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("rich-label frame did not contain a valid JPEG")
        rendered = draw_rich_overlay(image, selection.objects)
        ok, output = cv2.imencode(
            ".jpg",
            rendered,
            [cv2.IMWRITE_JPEG_QUALITY, self.settings.overlay_jpeg_quality],
        )
        if not ok:
            raise ValueError("could not encode rich-label JPEG")
        return output.tobytes()

    def _render_fused_label_jpeg(
        self,
        jpeg: bytes,
        overlay_selection: OverlaySelection,
        media_timestamp: float,
        frame_sequence: int,
        *,
        interpolate: bool = True,
    ) -> bytes:
        """Render Rich and Event annotations together with one JPEG codec pass."""

        coordinator = self._rich_labels
        analysis = overlay_selection.analysis
        rich_objects = ()
        if coordinator is not None:
            with self._state_lock:
                session_id = self._active_rich_session_id
            if session_id is not None and (
                analysis is None or analysis.stream_session_id == session_id
            ):
                rich_selection = coordinator.select(
                    session_id,
                    float(media_timestamp),
                    frame_sequence=int(frame_sequence),
                    interpolate=interpolate,
                )
                rich_objects = rich_selection.objects

        if not rich_objects and analysis is None:
            return jpeg

        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("fused-label frame did not contain a valid JPEG")

        if analysis is None:
            rendered = draw_rich_overlay(image, rich_objects)
        elif (
            analysis.analysis_kind in INDEPENDENT_FACE_ANALYSIS_KINDS
            and not _has_safety_annotations(analysis)
        ):
            rendered = (
                draw_rich_overlay(image, rich_objects) if rich_objects else image
            )
            rendered = _draw_analysis_overlay(rendered, analysis)
        else:
            plan = self._overlay_fusion.plan(
                analysis,
                rich_objects,
                association_message_id=(
                    overlay_selection.left_message_id or analysis.message_id
                ),
            )
            rendered = draw_rich_overlay(
                image,
                rich_objects,
                alerts_by_track_id=plan.alerts_by_track_id(),
            )
            fallback_ids = set(plan.unmatched_event_object_ids)
            fallback_objects = tuple(
                item for item in analysis.objects if item.object_id in fallback_ids
            )
            rendered = draw_event_rich_overlay(
                rendered,
                fallback_objects,
                alerts_by_object_id=plan.alerts_by_unmatched_object_id(),
            )
            arguments = _analysis_overlay_arguments(analysis)
            arguments["objects"] = ()
            rendered = draw_overlay(rendered, **arguments)

        ok, output = cv2.imencode(
            ".jpg",
            rendered,
            [cv2.IMWRITE_JPEG_QUALITY, self.settings.overlay_jpeg_quality],
        )
        if not ok:
            raise ValueError("could not encode fused-label JPEG")
        return output.tobytes()

    def _on_pipeline_state(self, state: dict[str, Any]) -> None:
        with self._state_lock:
            current = self._state.get("pipeline") or {}
            revision = state.get("revision")
            previous_revision = current.get("revision")
            if (
                type(revision) is int
                and type(previous_revision) is int
                and revision <= previous_revision
            ):
                return
            values: dict[str, Any] = {
                "pipeline": public_safe_value(copy.deepcopy(state))
            }
            if "last_error" in state:
                pipeline_error = state["last_error"]
                values["last_error"] = (
                    public_error_message(pipeline_error) if pipeline_error else None
                )
            self._update_state(**values)

    def _on_end(self, reason: str, error: str | None) -> None:
        # This callback also runs under _guarded_callback's state lock. Never
        # wait here: native results need that same lock to reach the scene.
        with self._state_lock:
            coordinator = self._rich_labels
            scene = getattr(self, "_scene", None)
            if self._session is not None:
                generation = self._generation
                if getattr(self, "_end_drain_generation", None) == generation:
                    return
                self._end_drain_generation = generation
                self._update_state(
                    pipeline_state="draining", ready=False, end_reason=reason,
                    scene_drain={"state": "running", "stage": "tracking"},
                )
                if isinstance(coordinator, VideoTrackingCoordinator):
                    coordinator.flush()
                thread = Thread(
                    target=self._drain_native_end,
                    args=(generation, self._session, scene, coordinator, reason, error),
                    name="scene-source-end", daemon=True,
                )
                thread.start()
                # Shutdown reads this reference without the state lock. Only
                # publish a started thread, which is safe for that owner to join.
                self._end_drain_thread = thread
                return
            self._finish_scene_end(reason, error)

    def _drain_native_end(
        self, generation, session, scene, coordinator, reason, error=None,
    ):
        """Drain accepted online work, release live resources, then organize."""
        control = self._archive_run(session)

        def current():
            with self._state_lock:
                return generation == self._generation and session is self._session

        failure_stage = "tracking"
        try:
            while current() and coordinator is not None:
                if coordinator.stats().get("state") in {"failed", "closed", "stopped"}:
                    raise RuntimeError("tracking drain failed")
                if coordinator.wait_until_idle(timeout_seconds=0.1):
                    if coordinator.stats().get("state") == "failed":
                        raise RuntimeError("tracking drain failed")
                    break
            if not current():
                return
            failure_stage = "inspection"
            self._update_state(scene_drain={"state": "running", "stage": "inspection"})
            if scene is not None:
                scene.finish_inspection_sources()
            worker = self._scene_worker
            if worker is not None:
                worker.wake()
                while current() and scene.has_pending_inspections():
                    if worker.stats()["state"] in {"failed", "stopped", "stopping"}:
                        raise RuntimeError("inspection drain failed")
                    time.sleep(0.05)
            failure_stage = "display"
            display = self._scene_display
            if isinstance(display, TrackedSceneDisplayWorker):
                display.notify_tracking()
                while current() and not display.wait_until_idle(timeout_seconds=0.1):
                    pass
            failure_stage = "finalization"
            with self._state_lock:
                if not current():
                    return
                self._close_scene_workers()
                # The old scene remains owned by its background task. Starting
                # another run cannot close it or redirect its identity updates.
                stopped = getattr(self, "_stop_requested_session", None) is session
                self._last_scene_snapshot = (
                    scene.snapshot(include_history=False) if scene is not None else None
                )
                if self._last_scene_snapshot is not None:
                    self._last_scene_snapshot["state"] = "postprocessing"
                    self._last_scene_snapshot["automatic_temporal_inference"] = False
                self._scene = None
                self._release_stopped_session(session)
                self._update_state(
                    pipeline_state="error" if error else "idle" if stopped else "ended",
                    scene_drain={"state": "running", "stage": "postprocessing"},
                    end_reason=reason,
                    last_error=public_error_message(error) if error else None,
                )
                with control.lock:
                    control.state, control.stage = "queued", "identity_review"
                    control.session, control.scene = session, scene
                self._postprocess_pool.submit(
                    self._finalize_run_background, session, scene, control, reason,
                )
        except Exception:
            LOGGER.exception("online session drain failed")
            failure_message = {
                "tracking": PUBLIC_TRACKING_ERROR,
                "inspection": PUBLIC_INSPECTION_ERROR,
            }.get(failure_stage, public_error_message(error))
            with control.lock:
                control.state, control.error = "failed", "online_drain_failed"
            with self._state_lock:
                if current():
                    self._update_state(scene_drain={"state": "failed"})
                    self._finish_scene_end(reason, failure_message)
                    if getattr(self, "_stop_requested_session", None) is session:
                        self._release_stopped_session(session)

    def _finalize_run_background(self, session, scene, control, reason):
        """L3 owns the retired run; no live global lock during review or I/O."""
        try:
            with control.lock:
                control.state, control.stage = "running", "identity_review"
            with self._identity_control_lock:
                registry = session.person_registry
                registry.review_pending_identities(accept_merge=(
                    (lambda old, new: scene.apply_identity_merges(((old, new),)) > 0)
                    if scene is not None else None
                ))
                merges = registry.drain_identity_merges()
                if scene is not None:
                    scene.apply_identity_review(registry.identity_review_states())
                admitted = self._queue_archive_subject_merges(session, merges)
                while not admitted:
                    if session.archive.status()["state"] in {"error", "stopped"}:
                        raise RuntimeError("archive rejected identity facts")
                    time.sleep(0.05)
                    admitted = self._queue_archive_subject_merges(session)
                if scene is not None:
                    scene.close()
                    final_snapshot = scene.snapshot()
                    admitted = self._queue_archive_scene(session, final_snapshot)
                    while not admitted:
                        if session.archive.status()["state"] in {"error", "stopped"}:
                            raise RuntimeError("archive rejected final scene")
                        time.sleep(0.05)
                        admitted = self._queue_archive_scene(session)
                else:
                    final_snapshot = None
            with control.lock:
                control.stage = "evidence"
            archive = session.archive
            if archive is not None:
                if not archive.end_session(session.stream_session_id, reason):
                    raise RuntimeError("archive rejected session end")
                while not archive.wait_until_idle(
                    timeout_seconds=0.1, stream_session_id=session.stream_session_id,
                ):
                    if archive.status()["state"] in {"error", "stopped"}:
                        raise RuntimeError("archive finalization failed")
                if archive.status()["state"] == "error":
                    raise RuntimeError("archive finalization failed")
            with self._identity_control_lock:
                # Gallery deletion may have changed the retired scene while
                # evidence encoding ran. Publish a fresh, serialized snapshot.
                if scene is not None:
                    final_snapshot = scene.snapshot()
                scene_archive = self._scene_archive
                if scene_archive is not None and final_snapshot is not None:
                    while not scene_archive.submit(final_snapshot, timeout_seconds=0.1):
                        if not final_snapshot.get("scene", {}).get("events"):
                            break
                        if scene_archive.stats()["state"] in {"failed", "closed"}:
                            raise RuntimeError("scene archive failed")
                    while not scene_archive.flush(timeout_seconds=0.1):
                        if scene_archive.stats()["state"] in {"failed", "closed"}:
                            raise RuntimeError("scene archive failed")
                    if scene_archive.stats()["state"] == "failed" or (
                        scene_archive.failed_for_run(session.stream_session_id)
                    ):
                        raise RuntimeError("scene archive write failed")
                with control.lock:
                    control.state, control.stage = "finalized", "complete"
                    control.session = control.scene = None
                with self._state_lock:
                    if (
                        self._session is None
                        and self._state.get("stream_session_id")
                        == session.stream_session_id
                    ):
                        self._last_scene_snapshot = final_snapshot
                        self._publish_scene_result(final_snapshot)
                        self._last_identity_review = registry.identity_review_stats()
                        self._update_state(scene_drain={"state": "completed"})
        except Exception:
            LOGGER.exception("background run finalization failed")
            with control.lock:
                control.state, control.error = "failed", "postprocess_failed"
            with self._state_lock:
                if self._session is None and self._state.get("stream_session_id") == (
                    session.stream_session_id
                ):
                    self._update_state(scene_drain={"state": "failed"})

    def _finish_scene_end(self, reason: str, error: str | None) -> None:
        self._observe_scene("close")
        self._close_scene_workers()
        if getattr(self, "_scene", None) is not None:
            self._on_scene_update()
        next_state = "error" if error else "ended"
        self._update_state(
            pipeline_state=next_state,
            ready=False,
            end_reason=reason,
            last_error=(public_error_message(error) if error else None),
        )

    def _update_state(self, **values: Any) -> None:
        with self._state_lock:
            safe = public_safe_value(values)
            assert isinstance(safe, dict)
            self._state.update(safe)


def _render_analysis_jpeg(
    jpeg: bytes,
    analysis: DisplayAnalysis,
    *,
    quality: int,
) -> bytes:
    encoded = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("display callback did not contain a valid JPEG")
    rendered = _draw_analysis_overlay(image, analysis)
    ok, output = cv2.imencode(
        ".jpg",
        rendered,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise ValueError("could not encode annotated JPEG")
    return output.tobytes()


def _draw_analysis_overlay(image: Any, analysis: DisplayAnalysis) -> Any:
    return draw_overlay(image, **_analysis_overlay_arguments(analysis))


def _has_safety_annotations(analysis: DisplayAnalysis) -> bool:
    return any(
        judgment.event_label in SAFETY_EVENT_LABELS
        for judgment in (
            *analysis.judgments, *getattr(analysis, "pending_clear_judgments", ()),
        )
    )


def _analysis_overlay_arguments(analysis: DisplayAnalysis) -> dict[str, Any]:
    face_objects = ()
    analysis_kind = analysis.analysis_kind
    if (
        analysis_kind in INDEPENDENT_FACE_ANALYSIS_KINDS
        and _has_safety_annotations(analysis)
    ):
        analysis_kind = SELECTED_FEATURES_ANALYSIS_KIND
    if analysis_kind == "face_recognition":
        detected_judgments = tuple(
            judgment for judgment in analysis.judgments if judgment.detected
        )
        displayed_judgments = detected_judgments or analysis.judgments
        objects = analysis.objects
        event_label = " | ".join(
            judgment.event_label for judgment in displayed_judgments
        )
        detected = bool(detected_judgments)
        confidence = max(
            (judgment.confidence for judgment in displayed_judgments),
            default=0.0,
        )
        description = "; ".join(
            judgment.description for judgment in displayed_judgments
        )
    else:
        observed_positives = tuple(
            judgment for judgment in analysis.judgments if judgment.detected
        )
        observed_labels = {judgment.event_label for judgment in observed_positives}
        detected_judgments = (*observed_positives, *(
            judgment for judgment in getattr(analysis, "pending_clear_judgments", ())
            if judgment.event_label not in observed_labels
        ))
        labels_by_object: dict[str, list[str]] = {}
        # A held event keeps its last positive box until clear confirmation.
        for judgment in detected_judgments:
            for object_id in judgment.object_ids:
                labels = labels_by_object.setdefault(object_id, [])
                if judgment.event_label not in labels:
                    labels.append(judgment.event_label)
        objects = tuple(
            replace(
                item,
                label=(
                    f"{' / '.join(labels_by_object[item.object_id])} | {item.label}"
                ),
            )
            for item in analysis.objects
            if item.object_id in labels_by_object and not is_face_object(item)
        )
        face_objects = tuple(item for item in analysis.objects if is_face_object(item))
        detected = bool(detected_judgments)
        if detected_judgments:
            event_label = " | ".join(
                judgment.event_label for judgment in detected_judgments
            )
            confidence = max(
                judgment.confidence for judgment in detected_judgments
            )
            description = "; ".join(
                f"{judgment.event_label}: {judgment.description}"
                for judgment in detected_judgments
            )
        else:
            event_label = " | ".join(
                judgment.event_label for judgment in analysis.judgments
            )
            confidence = min(
                (judgment.confidence for judgment in analysis.judgments),
                default=0.0,
            )
            description = "No target events detected."
    return {
        "event_label": event_label,
        "detected": detected,
        "confidence": confidence,
        "description": description,
        "objects": objects,
        "face_objects": face_objects,
        "analysis_kind": analysis_kind,
        "incomplete": bool(
            analysis.missing_event_labels
            or getattr(analysis, "expired_event_labels", ())
        ),
    }


def _normalize_features(features: Collection[str] | None) -> tuple[str, ...]:
    if features is None:
        return DETECTION_EVENT_LABELS
    if not isinstance(features, (list, tuple)):
        raise ValueError("features must be an array")
    raw = tuple(features)
    if not raw:
        raise ValueError("features must not be empty")
    if any(not isinstance(feature, str) or not feature.strip() for feature in raw):
        raise ValueError("features must contain non-empty strings")
    rendered = tuple(feature.strip().lower() for feature in raw)
    if len(set(rendered)) != len(rendered):
        raise ValueError("features must not contain duplicates")
    unsupported = tuple(
        feature for feature in rendered if feature not in DETECTION_EVENT_LABELS
    )
    if unsupported:
        raise ValueError("unsupported feature")
    return tuple(
        feature for feature in DETECTION_EVENT_LABELS if feature in rendered
    )


def _api_reference_html(schema: dict[str, Any]) -> str:
    """Render a self-contained customer reference without external assets."""

    rows: list[str] = []
    for path, operations in schema.get("paths", {}).items():
        for method, operation in operations.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            summary = operation.get("summary", "")
            rows.append(
                "<tr><td><strong>"
                + escape(method.upper())
                + "</strong></td><td><code>"
                + escape(redact_public_text(path))
                + "</code></td><td>"
                + escape(redact_public_text(summary))
                + "</td></tr>"
            )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Goodvision API Reference</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;
padding:0 20px;color:#17202e;line-height:1.6}
h1{margin-bottom:8px}a{color:#1558b0}
table{width:100%;border-collapse:collapse;margin:24px 0}
th,td{text-align:left;padding:10px;border-bottom:1px solid #dbe1e8;vertical-align:top}
code{overflow-wrap:anywhere}th{background:#f4f6f8}nav{display:flex;gap:20px;flex-wrap:wrap}
</style></head><body><h1>Goodvision API Reference</h1>
<p>Live detection, evidence retrieval, video output, and service controls.</p>
<nav><a href="/">Control console</a>
<a href="/openapi.json">Machine-readable API specification</a></nav>
<h2>Response formats</h2>
<p>Result snapshots use JSON. Result event streams use Server-Sent Events.
Video snapshots use JPEG; continuous video uses multipart MJPEG.
Continuous event evidence uses versioned MP4 clips with Range support.
The specification describes parameters, request bodies, and response codes.</p>
<p>Analysis results remain immediate. Archive event revisions and video encoding
run independently. Retired stranger photo endpoints return 410.</p>
<h2>HTTP endpoints</h2><table><thead><tr>
<th>Method</th><th>Path</th><th>Operation</th></tr></thead><tbody>
""" + "".join(rows) + """</tbody></table>
<h2>Live control</h2><p><code>/ws</code> provides the live control connection.
Use the console to manage the selected detection features and video source.</p>
<h2>Operational notes</h2><p>Result and image endpoints return 404 before data exists.
Stopping a session does not delete enrolled identities or stored evidence.
Protect deployment and enrollment controls using your deployment's access policy.</p>
</body></html>"""


def create_app(
    runtime_factory: Callable[[], HarnessRuntime] | None = None,
) -> FastAPI:
    factory = runtime_factory or HarnessRuntime.from_environment

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime = factory()
        application.state.runtime = runtime
        runtime.start()
        try:
            yield
        finally:
            runtime.stop()
            await application.state.fleet.aclose()

    application = FastAPI(
        title="Goodvision Streaming Harness",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        swagger_ui_oauth2_redirect_url=None,
    )
    install_event_routes(application)
    install_fleet_routes(application)

    @application.exception_handler(StarletteHTTPException)
    async def public_http_error(
        _request: Request, exc: StarletteHTTPException
    ) -> Response:
        if exc.status_code < 200 or exc.status_code in {204, 205, 304}:
            return Response(status_code=exc.status_code, headers=exc.headers)
        return JSONResponse(
            {"detail": public_safe_value(jsonable_encoder(exc.detail))},
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def public_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Keep the validation contract and location without reflecting request
        # bodies, query values, or exception objects into public responses.
        errors = []
        for item in exc.errors():
            error = dict(item)
            if "input" in error:
                error["input"] = "[redacted]"
            errors.append(error)
        return JSONResponse(
            public_safe_value(jsonable_encoder({"detail": errors})),
            status_code=422,
        )

    @application.get("/docs", response_class=HTMLResponse, include_in_schema=False)
    @application.get("/redoc", response_class=HTMLResponse, include_in_schema=False)
    async def api_reference() -> HTMLResponse:
        return HTMLResponse(
            _api_reference_html(application.openapi()),
            headers={
                "Content-Security-Policy": (
                    "default-src 'none'; style-src 'unsafe-inline'"
                ),
                "Cache-Control": "no-store",
            },
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")

    @application.get("/v1/ui/info")
    def ui_info(request: Request) -> dict[str, Any]:
        return _runtime(request).ui_info()

    @application.get("/v1/deployment/settings")
    def deployment_settings(request: Request) -> dict[str, Any]:
        return _runtime(request).deployment_settings()

    @application.get("/v1/deployment/routes")
    def route_status(request: Request) -> dict[str, Any]:
        return _runtime(request).route_probe_status()

    @application.post("/v1/deployment/routes/check", status_code=202)
    async def check_routes(request: Request) -> dict[str, Any]:
        try:
            async with application.state.fleet.control_lock:
                await application.state.fleet.assert_can_disable_multicamera()
                return await asyncio.to_thread(_runtime(request).start_route_probe)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="could not start route check",
            ) from exc

    @application.post("/v1/deployment/routes/select")
    async def select_route(request: Request) -> dict[str, Any]:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > 1024:
                raise HTTPException(status_code=413, detail="route selection too large")
            body.extend(chunk)
        try:
            payload = json.loads(body)
            if (
                not isinstance(payload, dict) or set(payload) != {"route_id"}
                or not isinstance(payload["route_id"], str)
            ):
                raise ValueError
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=400, detail="expected one route_id string",
            ) from exc
        try:
            async with application.state.fleet.control_lock:
                await application.state.fleet.assert_can_disable_multicamera()
                return await asyncio.to_thread(
                    _runtime(request).select_model_route, payload["route_id"],
                )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="could not save route selection",
            ) from exc

    @application.put("/v1/deployment/settings")
    async def update_deployment_settings(request: Request) -> dict[str, Any]:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="Content-Length must be an integer"
                ) from exc
            if declared_length < 0:
                raise HTTPException(
                    status_code=400, detail="Content-Length must not be negative"
                )
            if declared_length > DEPLOYMENT_REQUEST_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="deployment settings body exceeds 64 KiB",
                )

        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > DEPLOYMENT_REQUEST_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="deployment settings body exceeds 64 KiB",
                )
            body.extend(chunk)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=400, detail="deployment settings body must be JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400, detail="deployment settings body must be an object"
            )
        try:
            async with application.state.fleet.control_lock:
                runtime = _runtime(request)
                current_locations = runtime.service_execution_settings()
                # Legacy false also changes server mode back to cloud/legacy.
                # Compare the resolved topology, not just the boolean metadata.
                mode_changed = (
                    ServiceLocations(**current_locations).updated(payload).public_status()
                    != current_locations
                )
                dry_run_changed = (
                    "dry_run" in payload and payload["dry_run"] != runtime.dry_run
                )
                if (mode_changed or dry_run_changed
                        or {"api_key", "clear_api_key", "tracking_api_key",
                            "clear_tracking_api_key"}.intersection(payload)):
                    await application.state.fleet.assert_can_disable_multicamera()
                return await asyncio.to_thread(
                    runtime.update_deployment_settings, payload,
                )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=redact_public_text(exc)
            ) from exc
        except RuntimeError as exc:
            status_code = 409 if "active session" in str(exc) else 503
            raise HTTPException(
                status_code=status_code, detail=redact_public_text(exc)
            ) from exc

    @application.websocket("/ws")
    async def control_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        runtime: HarnessRuntime = websocket.app.state.runtime
        owner = str(uuid4())
        channel = CameraSocketChannel(websocket, runtime._camera_display_metadata)
        channel.start()
        metrics_task = None
        try:
            await channel.send({
                "type": "hello", **await asyncio.to_thread(runtime.ui_info),
            })
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text_message = message.get("text")
                if text_message is not None:
                    try:
                        payload = json.loads(text_message)
                    except json.JSONDecodeError:
                        await channel.send(
                            {"type": "error", "message": "control message must be JSON"}
                        )
                        continue
                    if not isinstance(payload, dict):
                        await channel.send(
                            {
                                "type": "error",
                                "message": "control message must be an object",
                            }
                        )
                        continue
                    command = payload.get("type")
                    if (command in {"start", "stop"}
                            and application.state.fleet.capacity > 1):
                        await channel.send({
                            "type": "error",
                            "message": "请使用控制台统一开始或停止所有视频的分析。",
                        })
                        continue
                    if command == "camera_display_metrics":
                        try:
                            report = validate_display_metrics({
                                key: value for key, value in payload.items()
                                if key != "type"
                            })
                        except (ValueError, TypeError):
                            continue
                        log_timing(
                            "local_camera_display_metrics_ingress",
                            stream_session_id=report["stream_session_id"],
                            report_sequence=report.get("report_sequence"),
                            recorder_busy=(
                                metrics_task is not None and not metrics_task.done()
                            ),
                        )
                        if metrics_task is None or metrics_task.done():
                            metrics_task = asyncio.create_task(asyncio.to_thread(
                                runtime.record_display_metrics, report,
                            ))
                        continue
                    if command == "start":
                        mode = str(payload.get("mode") or "")
                        analysis_mode = str(
                            payload.get("analysis_mode") or MULTI_ANALYSIS_MODE
                        )
                        requested_features = payload.get("features")
                        requested_dry_run = payload.get("dry_run")
                        if requested_dry_run is not None and not isinstance(
                            requested_dry_run, bool
                        ):
                            await channel.send(
                                {"type": "error", "message": "dry_run must be boolean"}
                            )
                            continue
                        event_label = str(payload.get("event_label") or "").strip()
                        try:
                            if mode == CAMERA_MODE:
                                response = await asyncio.to_thread(
                                    runtime.start_camera,
                                    dry_run=requested_dry_run,
                                    event_label=event_label,
                                    analysis_mode=analysis_mode,
                                    features=requested_features,
                                    event_sensitivities=payload.get("event_sensitivities"),
                                    owner=owner,
                                )
                            elif mode == STREAM_MODE:
                                response = await asyncio.to_thread(
                                    runtime.start_stream,
                                    str(payload.get("stream_url") or ""),
                                    dry_run=requested_dry_run,
                                    event_label=event_label,
                                    analysis_mode=analysis_mode,
                                    features=requested_features,
                                    event_sensitivities=payload.get("event_sensitivities"),
                                    camera_layout=payload.get("camera_layout"),
                                )
                            else:
                                raise ValueError("mode must be camera or stream")
                        except MissingAPIKeyError:
                            await channel.send(
                                {"type": "error",
                                 "message": PUBLIC_MISSING_API_KEY_ERROR}
                            )
                        except ProcessingMethodUnavailableError as exc:
                            await channel.send(
                                {"type": "error", "message": str(exc)}
                            )
                        except ValueError as exc:
                            await channel.send(
                                {
                                    "type": "error",
                                    "message": redact_public_text(exc),
                                }
                            )
                        except Exception as exc:
                            await channel.send(
                                {
                                    "type": "error",
                                    "message": public_error_message(
                                        exc, fallback="could not start session"
                                    ),
                                }
                            )
                        else:
                            await channel.send(response)
                            if mode == CAMERA_MODE:
                                channel.enable(response.get("stream_session_id"))
                            else:
                                channel.enable(None)
                    elif command == "stop":
                        await asyncio.to_thread(runtime.stop_session)
                        await channel.flush_available()
                        await channel.send(
                            {"type": "ended", "reason": "stopped", "error": None}
                        )
                    else:
                        await channel.send(
                            {
                                "type": "error",
                                "message": "unknown command",
                            }
                        )
                    continue

                data = message.get("bytes")
                if data is None:
                    continue
                if len(data) <= CAMERA_FRAME_HEADER.size:
                    await channel.send(
                        {"type": "error", "message": "camera frame is truncated"}
                    )
                    continue
                try:
                    jpeg, header = decode_camera_packet(data)
                    accepted = await asyncio.to_thread(
                        runtime.push_camera_frame, jpeg, owner=owner, **header,
                    )
                    if "source_sequence" in header:
                        current = (None if accepted else await asyncio.to_thread(
                            runtime._camera_display_metadata.source_identity,
                        ))
                        obsolete = (
                            not accepted and (
                            header["stream_session_id"] != current["stream_session_id"]
                            or (current["generation"] is not None and (
                                header["generation"], header["crop_version"]
                            ) != (current["generation"], current["crop_version"])))
                        )
                        await channel.send({
                            "type": "camera_frame_ack",
                            **{key: header[key] for key in (
                                "stream_session_id", "generation", "crop_version",
                                "source_sequence",
                            )},
                            "accepted": accepted,
                            "reason": None if accepted else (
                                "obsolete_source" if obsolete else "capacity"
                            ),
                        })
                except ValueError as exc:
                    await channel.send(
                        {"type": "error", "message": redact_public_text(exc)}
                    )
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            # ASGI disconnect/shutdown may cancel the surrounding scope while
            # we join our writer. Complete owned-task and camera cleanup first.
            with anyio.CancelScope(shield=True):
                await channel.close()
                if metrics_task is not None:
                    metrics_task.cancel()
                    await asyncio.gather(metrics_task, return_exceptions=True)
                await asyncio.to_thread(runtime.stop_camera_owned_by, owner)

    @application.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready")
    def health_ready(request: Request) -> JSONResponse:
        runtime = _runtime(request)
        ready, reason = runtime.readiness()
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "reason": reason},
            status_code=200 if ready else 503,
        )

    @application.get("/v1/status")
    def status(request: Request) -> dict[str, Any]:
        return _runtime(request).status()

    @application.get("/v1/faces/fast")
    async def fast_faces(request: Request) -> JSONResponse:
        # No service status/state lock or shared synchronous endpoint pool.
        return JSONResponse(
            _runtime(request).face_fast_results.snapshot(),
            headers={"Cache-Control": "no-store"},
        )

    def require_coordinator(request: Request) -> None:
        # Worker ports are private. The primary forwards coordinated commands
        # with this marker; it is routing metadata, not an authentication token.
        if (application.state.fleet.capacity > 1
                and request.headers.get("X-Fleet-Coordinator") != "1"):
            raise HTTPException(
                409, "请使用主控制台统一管理所有视频的分析。",
            )

    async def camera_payload(request: Request) -> dict[str, Any]:
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 64 * 1024:
                raise HTTPException(413, "camera control body exceeds 64 KiB")
        try:
            value = json.loads(body or b"{}")
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(400, "camera control body must be JSON") from None
        if not isinstance(value, dict):
            raise HTTPException(400, "camera control body must be an object")
        return value

    async def camera_action(action: Callable, *args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.to_thread(action, *args, **kwargs)
        except MissingAPIKeyError:
            raise HTTPException(400, PUBLIC_MISSING_API_KEY_ERROR) from None
        except ProcessingMethodUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, redact_public_text(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, redact_public_text(exc)) from exc

    @application.post("/v1/video/connect")
    async def connect_video(request: Request) -> dict[str, Any]:
        require_coordinator(request)
        payload = await camera_payload(request)
        if set(payload) - {"stream_url", "camera_layout"}:
            raise HTTPException(400, "unknown video connection setting")
        if "stream_url" in payload and not isinstance(payload["stream_url"], str):
            raise HTTPException(400, "stream_url must be a string")
        return await camera_action(
            _runtime(request).connect_stream,
            payload.get("stream_url"), camera_layout=payload.get("camera_layout"),
        )

    @application.post("/v1/video/disconnect")
    async def disconnect_video(request: Request) -> dict[str, Any]:
        require_coordinator(request)
        if await camera_payload(request):
            raise HTTPException(400, "disconnect does not accept settings")
        return await camera_action(_runtime(request).disconnect_stream)

    @application.get("/v1/analysis/settings")
    def analysis_settings(request: Request) -> dict[str, Any]:
        return _runtime(request).analysis_settings()

    @application.put("/v1/analysis/settings")
    async def update_analysis_settings(request: Request) -> dict[str, Any]:
        require_coordinator(request)
        payload = await camera_payload(request)
        return await camera_action(_runtime(request).update_analysis_settings, payload)

    @application.post("/v1/analysis/prepare")
    async def prepare_analysis(request: Request) -> dict[str, Any]:
        require_coordinator(request)
        if await camera_payload(request):
            raise HTTPException(400, "prepare uses saved camera settings")
        return await camera_action(_runtime(request).prepare_analysis)

    @application.post("/v1/analysis/start")
    async def start_analysis(request: Request) -> dict[str, Any]:
        require_coordinator(request)
        if await camera_payload(request):
            raise HTTPException(400, "analysis uses saved camera settings")
        runtime = _runtime(request)
        await camera_action(runtime.start_connected_analysis)
        return runtime.status()

    @application.post("/v1/analysis/stop")
    async def stop_analysis(request: Request) -> dict[str, Any]:
        require_coordinator(request)
        if await camera_payload(request):
            raise HTTPException(400, "stop does not accept settings")
        runtime = _runtime(request)
        await camera_action(runtime.stop_session)
        return runtime.status()

    @application.post("/v1/sessions/start")
    async def start_session(
        request: Request,
        payload: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> dict[str, Any]:
        """Start one network-stream session with any selected feature subset."""
        require_coordinator(request)
        payload = payload or {}
        mode = payload.get("mode", STREAM_MODE)
        if not isinstance(mode, str) or mode not in {CAMERA_MODE, STREAM_MODE}:
            raise HTTPException(status_code=400, detail="mode must be camera or stream")
        if mode == CAMERA_MODE:
            raise HTTPException(
                status_code=400,
                detail="camera sessions must be started over /ws",
            )
        analysis_mode = payload.get("analysis_mode", MULTI_ANALYSIS_MODE)
        if not isinstance(analysis_mode, str):
            raise HTTPException(
                status_code=400,
                detail="analysis_mode must be a string",
            )
        dry_run = payload.get("dry_run")
        if dry_run is not None and not isinstance(dry_run, bool):
            raise HTTPException(status_code=400, detail="dry_run must be boolean")
        event_label = payload.get("event_label")
        if event_label is not None and not isinstance(event_label, str):
            raise HTTPException(status_code=400, detail="event_label must be a string")
        stream_url = payload.get("stream_url", "")
        if not isinstance(stream_url, str):
            raise HTTPException(status_code=400, detail="stream_url must be a string")
        runtime = _runtime(request)
        try:
            return await asyncio.to_thread(
                runtime.start_stream,
                stream_url,
                dry_run=dry_run,
                event_label=event_label,
                analysis_mode=analysis_mode,
                features=payload.get("features"),
                event_sensitivities=payload.get("event_sensitivities"),
                camera_layout=payload.get("camera_layout"),
            )
        except MissingAPIKeyError as exc:
            raise HTTPException(
                status_code=400, detail=PUBLIC_MISSING_API_KEY_ERROR,
            ) from exc
        except ProcessingMethodUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=redact_public_text(exc),
            ) from exc
        except RuntimeError as exc:
            status_code = 409 if "already running" in str(exc) else 503
            raise HTTPException(
                status_code=status_code,
                detail=public_error_message(exc, fallback="could not start session"),
            ) from exc

    @application.post("/v1/sessions/stop")
    async def stop_session(request: Request) -> dict[str, bool]:
        require_coordinator(request)
        return {
            "stopped": await asyncio.to_thread(_runtime(request).stop_session)
        }

    @application.get("/v1/faces/gallery")
    async def face_gallery(request: Request) -> dict[str, Any]:
        return await asyncio.to_thread(_runtime(request).face_gallery)

    @application.get("/v1/faces/strangers")
    async def stranger_captures(request: Request) -> dict[str, Any]:
        raise HTTPException(410, {
            "code": "stranger_photos_removed", "events_url": "/v1/events",
            "message": "Stranger photos were replaced by event video evidence.",
        })

    @application.get("/v1/faces/strangers/{stranger_id}/photo")
    async def stranger_photo(stranger_id: str, request: Request) -> Response:
        raise HTTPException(410, {
            "code": "stranger_photos_removed", "events_url": "/v1/events",
            "message": "Stranger photos were replaced by event video evidence.",
        })

    @application.post("/v1/faces/gallery/{person_id}")
    async def enroll_face(
        person_id: str,
        request: Request,
        name: str = Query(min_length=1, max_length=PERSON_FIELD_MAX_CHARS),
    ) -> dict[str, Any]:
        rendered_id = person_id.strip()
        if not rendered_id or len(rendered_id) > PERSON_FIELD_MAX_CHARS:
            raise HTTPException(status_code=400, detail="invalid person_id")
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > FACE_ENROLLMENT_MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="enrollment image is too large",
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="invalid Content-Length",
                ) from None
        image_buffer = bytearray()
        async for chunk in request.stream():
            if len(image_buffer) + len(chunk) > FACE_ENROLLMENT_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="enrollment image is too large",
                )
            image_buffer.extend(chunk)
        image_bytes = bytes(image_buffer)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="enrollment image is empty")
        try:
            return await asyncio.to_thread(
                _runtime(request).enroll_face,
                rendered_id,
                name.strip(),
                image_bytes,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=redact_public_text(exc)
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=public_error_message(exc, fallback=PUBLIC_FACE_ERROR),
            ) from exc

    @application.delete("/v1/faces/gallery/{person_id}")
    async def delete_face(person_id: str, request: Request) -> dict[str, Any]:
        try:
            deleted = await asyncio.to_thread(
                _runtime(request).delete_face,
                person_id,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=redact_public_text(exc)
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=public_error_message(exc, fallback=PUBLIC_FACE_ERROR),
            ) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="person not found")
        result = public_safe_value({"deleted": True, "person_id": person_id})
        assert isinstance(result, dict)
        return result

    @application.get("/v1/results/latest")
    async def latest_result(request: Request) -> JSONResponse:
        return _latest_result_response(_runtime(request))

    @application.get("/v1/rich-labels/latest")
    def latest_rich_labels(request: Request) -> JSONResponse:
        payload = _runtime(request).latest_rich_labels()
        if payload is None:
            raise HTTPException(status_code=404, detail="no rich labels yet")
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @application.get("/v1/scene")
    @application.get("/v1/experimental/scene", include_in_schema=False)
    def experimental_scene(request: Request) -> JSONResponse:
        return JSONResponse(
            _runtime(request).scene_snapshot(),
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/v1/results/events")
    async def result_events(
        request: Request,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        return _result_events_response(request, _runtime(request), after=after)

    @application.get("/v1/scene/evidence")
    def scene_evidence(request: Request) -> JSONResponse:
        runtime = _runtime(request)
        with runtime._state_lock:
            scene = runtime._scene
            job = scene.latest_inspection() if scene else None
        if job is None:
            raise HTTPException(
                status_code=404, detail="no completed temporal evidence"
            )
        payload = {
            "stream_session_id": job.stream_session_id,
            "request_id": job.request.request_id,
            "event_labels": list(job.request.event_labels),
            "source_interval": [
                job.frames[0].timestamp_seconds,
                job.frames[-1].timestamp_seconds,
            ],
            "frames": [
                {
                    "frame_sequence": frame.frame_sequence,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "url": (
                        f"/v1/scene/evidence/{job.request.request_id}/"
                        f"{frame.frame_sequence}.jpg"
                    ),
                }
                for frame in job.frames
            ],
        }
        return JSONResponse(
            public_safe_value(payload), headers={"Cache-Control": "no-store"}
        )

    @application.get("/v1/scene/evidence/{request_id}/{frame_sequence}.jpg")
    def scene_evidence_frame(
        request: Request, request_id: str, frame_sequence: int
    ) -> Response:
        runtime = _runtime(request)
        with runtime._state_lock:
            scene = runtime._scene
            job = scene.latest_inspection() if scene else None
            frame = (
                next(
                    (
                        item
                        for item in job.frames
                        if item.frame_sequence == frame_sequence
                    ),
                    None,
                )
                if job and job.request.request_id == request_id
                else None
            )
        if frame is None:
            raise HTTPException(
                status_code=404, detail="temporal evidence expired or unavailable"
            )
        return Response(
            frame.jpeg_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Frame-Sequence": str(frame.frame_sequence),
                "X-Media-Timestamp": str(frame.timestamp_seconds),
            },
        )

    @application.get("/v1/scene/history")
    async def scene_history(
        request: Request, limit: int = Query(default=100, ge=1, le=1000)
    ) -> JSONResponse:
        runtime = _runtime(request)
        archive = runtime._scene_archive
        events = (
            await asyncio.to_thread(archive.list_events, limit=limit) if archive else []
        )
        return JSONResponse({"events": events}, headers={"Cache-Control": "no-store"})

    @application.get("/v1/scene/events")
    async def scene_events(request: Request) -> StreamingResponse:
        runtime = _runtime(request)

        async def generate() -> AsyncIterator[bytes]:
            # Always begin with a current session snapshot on reconnect.
            while not await request.is_disconnected():
                payload = await asyncio.to_thread(runtime.scene_snapshot)
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                yield f"event: scene\ndata: {data}\n\n".encode()
                await asyncio.sleep(0.5)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/v1/results/overall/latest")
    async def latest_overall_description(request: Request) -> JSONResponse:
        return _latest_result_response(_runtime(request), overall_only=True)

    @application.get("/v1/results/overall/events")
    async def overall_description_events(
        request: Request,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        return _result_events_response(
            request, _runtime(request), after=after, overall_only=True
        )

    def feature_latest_endpoint(
        feature: str,
    ) -> Callable[[Request], Any]:
        async def endpoint(request: Request) -> JSONResponse:
            return _latest_result_response(_runtime(request), feature=feature)

        return endpoint

    def feature_events_endpoint(feature: str) -> Callable[..., Any]:
        async def endpoint(
            request: Request,
            after: int = Query(default=0, ge=0),
        ) -> StreamingResponse:
            return _result_events_response(
                request,
                _runtime(request),
                after=after,
                feature=feature,
            )

        return endpoint

    # Literal paths give downstream consumers five independently addressable
    # APIs while all of them reuse the same correlated result history.
    for feature in DETECTION_EVENT_LABELS:
        application.add_api_route(
            f"/v1/results/{feature}/latest",
            feature_latest_endpoint(feature),
            methods=["GET"],
            name=f"latest_{feature}_result",
        )
        application.add_api_route(
            f"/v1/results/{feature}/events",
            feature_events_endpoint(feature),
            methods=["GET"],
            name=f"{feature}_result_events",
        )

    @application.get("/assets/cloud_gpu_console.js")
    async def cloud_gpu_console_script() -> Response:
        return Response(
            Path(__file__).with_name("cloud_gpu_console.js").read_text(encoding="utf-8"),
            media_type="text/javascript",
        )

    @application.get("/assets/fleet_console.js")
    async def fleet_console_script() -> Response:
        return Response(
            Path(__file__).with_name("fleet_console.js").read_text(encoding="utf-8"),
            media_type="application/javascript",
        )

    @application.get("/assets/camera_display.js")
    def camera_display_script() -> Response:
        return Response(
            Path(__file__).with_name("camera_display.js").read_text(encoding="utf-8"),
            media_type="application/javascript", headers={"Cache-Control": "no-cache"},
        )

    @application.get("/assets/camera_display_dense.js")
    def camera_display_dense_script() -> FileResponse:
        return FileResponse(
            Path(__file__).with_name("camera_display_dense.js"),
            media_type="application/javascript", headers={"Cache-Control": "no-cache"},
        )

    @application.get("/assets/display_motion_runtime.js", include_in_schema=False)
    @application.get("/assets/opencv-4.13.0.js", include_in_schema=False)
    def camera_display_opencv_script() -> FileResponse:
        return FileResponse(
            Path(__file__).with_name("assets") / "opencv-4.13.0.js",
            media_type="application/javascript",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @application.get("/v1/video/display/metadata")
    def camera_display_metadata(request: Request) -> JSONResponse:
        return JSONResponse(
            request.app.state.runtime._camera_display_metadata.snapshot(),
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/v1/video/display/frames")
    def network_display_frames(request: Request, after: int = -1,
                               connection_id: str = "", revision: int = -1,
                               stream_session_id: str = "") -> Response:
        from .network_display import display_response
        return display_response(_runtime(request), after=after,
                                connection_id=connection_id, revision=revision,
                                stream_session_id=stream_session_id)

    @application.post("/v1/video/display/metrics")
    async def camera_display_metrics(request: Request) -> dict:
        body = await request.body()
        if len(body) > 8192:
            raise HTTPException(413, "display report too large")
        try:
            report = validate_display_metrics(json.loads(body))
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(400, "invalid display metrics") from exc
        return await asyncio.to_thread(
            request.app.state.runtime.record_display_metrics, report,
        )

    @application.get("/v1/video/latest.jpg")
    async def latest_frame(request: Request) -> Response:
        return _snapshot_response(
            _runtime(request).frames,
            missing_detail="no analyzed frame yet",
        )

    @application.get("/v1/video/stream.mjpeg")
    async def video_stream(request: Request) -> StreamingResponse:
        return _mjpeg_response(request, _runtime(request).frames)

    @application.get("/v1/video/live/latest.jpg")
    async def latest_live_frame(request: Request) -> Response:
        return _snapshot_response(
            _runtime(request).preview_frames,
            missing_detail="no live source frame yet",
        )

    @application.get("/v1/video/live/stream.mjpeg")
    async def live_video_stream(request: Request) -> StreamingResponse:
        return _mjpeg_response(request, _runtime(request).preview_frames)

    @application.get("/v1/video/annotated/latest.jpg")
    async def latest_annotated_frame(request: Request) -> Response:
        runtime = _runtime(request)
        connection = getattr(runtime, "_video_connection", None)
        if connection is not None:
            analyzing = runtime._state.get("pipeline_state") in {"starting", "running"}
            if not analyzing or runtime.annotated_frames.latest() is None:
                jpeg = connection.latest_jpeg()
                if jpeg is None:
                    raise HTTPException(404, "video connection has no preview yet")
                return Response(
                    jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"},
                )
        return _snapshot_response(
            runtime.annotated_frames,
            missing_detail="no delayed annotated frame yet",
        )

    @application.get("/v1/video/annotated/stream.mjpeg")
    async def annotated_video_stream(request: Request) -> StreamingResponse:
        return _mjpeg_response(request, _runtime(request).annotated_frames)

    return application


def _runtime(request: Request) -> HarnessRuntime:
    return request.app.state.runtime


def _feature_result_payload(
    payload: dict[str, Any],
    feature: str,
) -> dict[str, Any] | None:
    accepted_labels = {feature}
    if feature == "stranger":
        accepted_labels.add(UNKNOWN_FACE_EVENT_LABEL)
    judgments = payload.get("judgments")
    if not isinstance(judgments, list):
        return None
    selected = next(
        (
            copy.deepcopy(judgment)
            for judgment in judgments
            if isinstance(judgment, dict)
            and judgment.get("event_label") in accepted_labels
        ),
        None,
    )
    if selected is None:
        return None
    selected["event_label"] = feature
    object_ids = {
        object_id
        for object_id in selected.get("object_ids", [])
        if isinstance(object_id, str)
    }
    objects = payload.get("objects")
    filtered_objects = (
        [
            copy.deepcopy(item)
            for item in objects
            if isinstance(item, dict) and (
                item.get("object_id") in object_ids
                or (
                    feature == "stranger"
                    and isinstance(item.get("attributes"), dict)
                    and item["attributes"].get("status") in {
                        "known", "unknown", "unusable",
                    }
                )
            )
        ]
        if isinstance(objects, list)
        else []
    )
    result = copy.deepcopy(payload)
    result["judgments"] = [selected]
    result["objects"] = filtered_objects
    return result


def _overall_description_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Project the shared result into a frame-correlated description response."""

    return {
        "type": "overall_description",
        **{
            key: payload[key]
            for key in (
                "schema_version",
                "message_id",
                "observation_id",
                "source_id",
                "stream_session_id",
                "sequence",
                "emitted_at",
                "frame",
                "description",
                "analysis_complete",
                "missing_event_labels",
            )
        },
    }


def _latest_result_response(
    runtime: HarnessRuntime,
    *,
    feature: str | None = None,
    overall_only: bool = False,
) -> JSONResponse:
    record = runtime.results.latest(feature=feature)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"no {feature} result yet" if feature else "no analysis result yet",
        )
    payload = record.payload
    if feature is not None:
        selected = _feature_result_payload(payload, feature)
        if selected is None:
            raise HTTPException(
                status_code=404,
                detail=f"no {feature} result in the latest analysis",
            )
        payload = selected
    if overall_only:
        payload = _overall_description_payload(payload)
    headers = {
        "Cache-Control": "no-store",
        "X-Result-Cursor": str(record.cursor),
        "X-Result-Stream": runtime.results.epoch,
    }
    if feature is not None:
        headers["X-Result-Feature"] = feature
    return JSONResponse(payload, headers=headers)


def _result_events_response(
    request: Request,
    runtime: HarnessRuntime,
    *,
    after: int,
    feature: str | None = None,
    overall_only: bool = False,
) -> StreamingResponse:
    header_cursor = request.headers.get("last-event-id")
    cursor = after
    if header_cursor and after == 0:
        epoch, separator, raw_cursor = header_cursor.rpartition(":")
        if separator and epoch == runtime.results.epoch:
            try:
                cursor = max(0, int(raw_cursor))
            except ValueError:
                cursor = 0
        else:
            cursor = 0

    async def generate() -> AsyncIterator[bytes]:
        current = cursor
        while not await request.is_disconnected():
            records = await asyncio.to_thread(
                runtime.results.records_after,
                current,
                timeout_seconds=15.0,
            )
            if not records:
                yield b": heartbeat\n\n"
                continue
            emitted = False
            for record in records:
                current = record.cursor
                payload = record.payload
                if feature is not None:
                    selected = _feature_result_payload(payload, feature)
                    if selected is None:
                        continue
                    payload = selected
                if overall_only:
                    payload = _overall_description_payload(payload)
                data = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                emitted = True
                event_name = (
                    "overall_description" if overall_only else "analysis_result"
                )
                yield (
                    f"id: {runtime.results.epoch}:{record.cursor}\n"
                    f"event: {event_name}\ndata: {data}\n\n"
                ).encode()
            if not emitted:
                yield b": heartbeat\n\n"

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    if feature is not None:
        headers["X-Result-Feature"] = feature
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=headers,
    )


def _frame_headers(packet: FramePacket) -> dict[str, str]:
    headers = {
        "Cache-Control": "no-store",
        "X-Frame-Sequence": str(packet.frame_sequence),
        "X-Media-Timestamp": f"{packet.media_timestamp:.3f}",
        "X-Observed-At": packet.observed_at,
    }
    if packet.analysis_message_id is not None:
        headers["X-Analysis-Message-Id"] = packet.analysis_message_id
    if packet.next_analysis_message_id is not None:
        headers["X-Next-Analysis-Message-Id"] = packet.next_analysis_message_id
    if packet.annotation_mode is not None:
        headers["X-Annotation-Mode"] = packet.annotation_mode
    if packet.stream_delay_seconds is not None:
        headers["X-Stream-Delay"] = f"{packet.stream_delay_seconds:.3f}"
    return headers


def _snapshot_response(bus: FrameBus, *, missing_detail: str) -> Response:
    latest = bus.latest()
    if latest is None:
        raise HTTPException(status_code=404, detail=missing_detail)
    _, packet = latest
    return Response(
        packet.jpeg,
        media_type="image/jpeg",
        headers=_frame_headers(packet),
    )


def _mjpeg_response(request: Request, bus: FrameBus) -> StreamingResponse:
    async def generate() -> AsyncIterator[bytes]:
        version = 0
        latest = bus.latest()
        if latest is not None:
            version, packet = latest
            yield _mjpeg_part(packet)
        while not await request.is_disconnected():
            update = await asyncio.to_thread(
                bus.wait_after,
                version,
                timeout_seconds=15.0,
            )
            if update is None:
                continue
            version, packet = update
            yield _mjpeg_part(packet)

    return StreamingResponse(
        generate(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )


def _mjpeg_part(packet: FramePacket) -> bytes:
    optional_headers: list[str] = []
    if packet.analysis_message_id is not None:
        optional_headers.append(
            f"X-Analysis-Message-Id: {packet.analysis_message_id}\r\n"
        )
    if packet.next_analysis_message_id is not None:
        optional_headers.append(
            f"X-Next-Analysis-Message-Id: {packet.next_analysis_message_id}\r\n"
        )
    if packet.annotation_mode is not None:
        optional_headers.append(f"X-Annotation-Mode: {packet.annotation_mode}\r\n")
    if packet.stream_delay_seconds is not None:
        optional_headers.append(
            f"X-Stream-Delay: {packet.stream_delay_seconds:.3f}\r\n"
        )
    rendered_optional_headers = "".join(optional_headers)
    headers = (
        f"--{MJPEG_BOUNDARY}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(packet.jpeg)}\r\n"
        f"X-Frame-Sequence: {packet.frame_sequence}\r\n"
        f"X-Media-Timestamp: {packet.media_timestamp:.3f}\r\n"
        f"X-Observed-At: {packet.observed_at}\r\n"
        f"{rendered_optional_headers}\r\n"
    ).encode("ascii")
    return headers + packet.jpeg + b"\r\n"


def _utc_now() -> str:
    from datetime import datetime

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _env_bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def _env_bounded_float(
    name: str, default: float, minimum: float, maximum: float,
) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number in {minimum}..{maximum}") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a number in {minimum}..{maximum}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    rendered = value.strip().lower()
    if rendered in {"1", "true", "yes", "on"}:
        return True
    if rendered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def main() -> None:
    capture_options = os.environ.get("VIDEO_CAPTURE_OPTIONS")
    if capture_options is not None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = capture_options
    host = os.environ.get("HARNESS_HOST", "0.0.0.0")
    port = int(os.environ.get("HARNESS_PORT", "8080"))
    # MJPEG/SSE clients are intentionally long-lived. Bound their shutdown so
    # Docker can continue into application cleanup instead of waiting forever.
    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=10,
        server_header=False,
    )


app = create_app()


if __name__ == "__main__":
    main()
