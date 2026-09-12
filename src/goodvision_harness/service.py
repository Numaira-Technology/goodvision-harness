"""HTTP delivery service for the independent streaming harness."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import struct
import time
import unicodedata
from collections.abc import AsyncIterator, Callable, Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timezone
from html import escape
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Annotated, Any
from uuid import uuid4

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.websockets import WebSocketDisconnect

from .config import (
    DEFAULT_CONFIG_PATH,
    DisplaySettings,
    HarnessConfig,
    ServiceSettings,
    load_config,
)
from .contracts import PUBLIC_MODEL_NAME, AnalysisResult
from .delayed_overlay import DelayedOverlayComposer, DisplayAnalysis, OverlaySelection
from .delivery import WebhookDelivery
from .deployment_settings import (
    DeploymentSettings,
    DeploymentSettingsStore,
    validate_buffer_delay_seconds,
    validate_camera_layout,
)
from .event_archive import EventArchive
from .event_routes import install_event_routes
from .event_sensitivity import (
    EVENT_SENSITIVITY_DEFAULTS,
    normalize_event_sensitivities,
    sensitivity_ui_info,
)
from .face_recognition import (
    FaceRecognitionRuntime,
    FaceRecognitionSession,
    FaceRecognizerConfig,
)
from .grounding import (
    DETECTION_EVENT_LABELS,
    FIXED_FIVE_ANALYSIS_KIND,
    SAFETY_EVENT_LABELS,
    SELECTED_FEATURES_ANALYSIS_KIND,
    build_grounding_client,
    build_selected_features_grounding_client,
)
from .model_api import (
    DEFAULT_OPENROUTER_ENDPOINT,
    MissingAPIKeyError,
    ModelAPISettings,
    PooledModelAPIClient,
    environment_api_key,
)
from .overlay import (
    draw_event_rich_overlay,
    draw_face_boxes,
    draw_overlay,
    draw_rich_overlay,
)
from .overlay_fusion import OverlayFusionMatcher
from .person_event_overlay import PersonOverlayAnalysis
from .pipeline import CAMERA_MODE, STREAM_MODE, SessionCallbacks, StreamingSession
from .privacy import (
    PUBLIC_ANALYSIS_ERROR,
    PUBLIC_FACE_ERROR,
    PUBLIC_MISSING_API_KEY_ERROR,
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
from .scene_runtime import SceneRuntime
from .stream_adapter import parse_stream_source
from .video_to_stream import encode_jpeg
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
        model_api = config.model_api if config is not None else ModelAPISettings()
        self._inference_provider = model_api.provider
        self._deployment_store = deployment_store
        self._deployment_settings: DeploymentSettings | None = None
        self._deployment_error: str | None = None
        self._persisted_api_key: str | None = None
        if deployment_store is not None:
            try:
                self._deployment_settings = deployment_store.load()
                self._persisted_api_key = deployment_store.load_api_key(
                    provider=self._inference_provider,
                )
            except Exception:
                LOGGER.error("could not load persistent deployment settings")
                self._deployment_error = "persistent settings could not be loaded"
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
        self.config = config
        self._scene_enabled = config is not None and config.runtime.scene_enabled
        self._scene: SceneRuntime | None = None
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
        self.frames = FrameBus()
        self.preview_frames = FrameBus()
        self.annotated_frames = FrameBus()
        display = config.display if config is not None else DisplaySettings()
        if (
            self._deployment_settings is not None
            and self._deployment_settings.buffer_delay_seconds is not None
        ):
            display = replace(
                display,
                delay_seconds=self._deployment_settings.buffer_delay_seconds,
            )
        self._display_settings = display
        self._rich_disabled_reason: str | None = None
        self._rich_requested = rich_label_coordinator is not None or (
            config is not None and config.rich_labeling.enabled
        )
        self._tracking_requested = self._rich_requested or (
            config is not None and config.rich_labeling.tracking_enabled
        )
        self._events: EventArchive | None = None
        self._rich_probe_thread: Thread | None = None
        self._rich_probe_stop = Event()
        self._active_rich_session_id: str | None = None
        if rich_label_coordinator is not None:
            self._rich_labels = rich_label_coordinator
        elif config is not None and config.rich_labeling.enabled:
            rich = config.rich_labeling
            rich_client: RichLabelClient
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
            else:
                rich_client = RichLabelHTTPClient(
                    rich.endpoint,
                    timeout_seconds=rich.timeout_seconds,
                    mode="tracks", include_attributes=self._rich_requested,
                )
            if rich.backend == "vlm" or rich_client.gpu_ready():
                self._rich_labels = RichLabelCoordinator(
                    rich_client,
                    max_fps=rich.max_fps,
                    history_size=rich.history_size,
                    max_interpolation_gap_seconds=(
                        rich.max_interpolation_gap_seconds
                    ),
                    max_hold_seconds=rich.max_hold_seconds,
                    on_snapshot=self._on_tracking_snapshot,
                )
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
                self._render_fused_label_jpeg
                if self._rich_requested
                else None
            ),
        )
        self._state_lock = RLock()
        self._control_lock = RLock()
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
            active = self._session is not None and self._session.running
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
        self.config = replace(self.config, inference=inference)
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
            if self._session is not None and self._session.running:
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
            if self._session is not None and self._session.running:
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
        return {
            "persistence_available": self._deployment_store is not None,
            "settings_saved": self._deployment_settings is not None,
            "settings_source": source,
            "configured_stream": bool(stream_url),
            "configured_stream_protocol": protocol,
            "stream_frame_crop": self.configured_frame_crop(),
            "camera_layout": self.configured_camera_layout(),
            "dry_run": self.dry_run,
            "autostart": self.autostart,
            "buffer_delay_seconds": self._display_settings.delay_seconds,
            "api_key_present": bool(self._aiclub_api_key),
            "api_key_persisted": bool(self._persisted_api_key),
            "storage_error": self._deployment_error,
        }

    def update_deployment_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist defaults; buffer-only updates also apply to the active session."""

        if self._deployment_store is None:
            raise RuntimeError("persistent deployment storage is not configured")
        allowed = {
            "stream_url",
            "dry_run",
            "autostart",
            "api_key",
            "clear_api_key",
            "buffer_delay_seconds",
            "camera_layout",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("unknown deployment setting")
        with self._control_lock:
            buffer_only = set(payload) == {"buffer_delay_seconds"}
            if self._session is not None and self._session.running and not buffer_only:
                raise RuntimeError("stop the active session before saving settings")
            if self._route_probe["state"] == "running":
                raise RuntimeError("wait for the route check before saving settings")

            setting_fields = {
                "stream_url", "dry_run", "autostart", "buffer_delay_seconds",
                "camera_layout",
            }
            settings: DeploymentSettings | None = None
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
                settings = DeploymentSettings(
                    stream_url=stream_url,
                    dry_run=dry_run,
                    autostart=autostart,
                    buffer_delay_seconds=buffer_delay_seconds,
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

            # Do not mutate disk or runtime state until every supplied field has
            # passed validation. A mixed request must fail as one unit when, for
            # example, its stream settings are valid but its API key is not.
            if settings is not None:
                self._deployment_store.save(settings)
                self._deployment_settings = settings
                self.dry_run = settings.dry_run
                self.autostart = settings.autostart
                if "buffer_delay_seconds" in payload:
                    assert buffer_delay_seconds is not None
                    self._delayed_overlay.set_delay_seconds(buffer_delay_seconds)
                    self._display_settings = replace(
                        self._display_settings, delay_seconds=buffer_delay_seconds,
                    )
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
            if (rendered_key is not None or clear_api_key) and self._route_transport:
                self._route_transport.close()
                self._route_transport = None
            self._deployment_error = None
            if not buffer_only:
                self._update_state(dry_run=self.dry_run)
        return self.deployment_settings()

    def ui_info(self) -> dict[str, Any]:
        if self.config is None:
            result = {
                "configured": False,
                "error": "configuration unavailable",
            }
            safe = public_safe_value(result)
            assert isinstance(safe, dict)
            return safe
        stream_url = self.configured_stream_url()
        protocol = None
        if stream_url:
            try:
                protocol = parse_stream_source(stream_url).protocol
            except ValueError:
                protocol = None
        result = {
            "configured": True,
            "model": PUBLIC_MODEL_NAME,
            "event_label": self.config.inference.event_label,
            "event_labels": list(DETECTION_EVENT_LABELS),
            "features": list(DETECTION_EVENT_LABELS),
            "supported_features": list(DETECTION_EVENT_LABELS),
            **sensitivity_ui_info(),
            "sample_fps": self.config.video.fps,
            "max_image_side": self.config.video.max_image_side,
            "jpeg_quality": self.config.video.jpeg_quality,
            "window": {
                "size_frames": self.config.sliding_window.size_frames,
                "stride_frames": self.config.sliding_window.stride_frames,
            },
            "max_concurrent_requests": self.config.runtime.max_concurrent_requests,
            "api_key_present": bool(self._aiclub_api_key),
            "dry_run": self.dry_run,
            "configured_stream": bool(stream_url),
            "configured_stream_protocol": protocol,
            "stream_frame_crop": self.configured_frame_crop(),
            "camera_layout": self.configured_camera_layout(),
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
                "max_image_side": self.config.face.max_image_side,
                "jpeg_quality": self.config.face.jpeg_quality,
                "threshold": self.config.face.threshold,
                "min_face_size": self.config.face.min_face_size,
                "enrollment_min_face_size": (
                    self.config.face.enrollment_min_face_size
                ),
            },
            "multi": {
                "sample_fps": self.config.face.sample_fps,
                "max_image_side": self.config.face.max_image_side,
                "jpeg_quality": self.config.face.jpeg_quality,
                "safety_max_image_side": self.config.video.max_image_side,
            },
            "annotated_output": {
                "output_fps": self._annotated_output_fps,
                "max_image_side": self.config.display.max_image_side,
                "jpeg_quality": self.config.display.jpeg_quality,
                "buffer_delay_seconds": self._display_settings.delay_seconds,
                "bbox_max_gap_seconds": (
                    self.config.display.bbox_max_gap_seconds
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

    def enroll_face(
        self,
        person_id: str,
        name: str,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        person = self._ensure_face_runtime().enroll(person_id, name, image_bytes)
        with self._state_lock:
            session = self._session
        registry = getattr(session, "person_registry", None)
        if registry is not None:
            registry.allow_person(person_id)
        with self._state_lock:
            self._observe_scene("allow_identity", person_id)
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
        deleted = self._ensure_face_runtime().delete_person(person_id)
        if deleted:
            with self._state_lock:
                session = self._session
            registry = getattr(session, "person_registry", None)
            if registry is not None:
                registry.forget_person(person_id)
            with self._state_lock:
                self._observe_scene("forget_identity", person_id)
        return deleted

    def _ensure_face_runtime(self) -> FaceRecognitionRuntime:
        if self.config is None:
            raise RuntimeError(self.config_error or "configuration unavailable")
        with self._control_lock:
            if self._face_runtime is None:
                face = self.config.face
                self._face_runtime = FaceRecognitionRuntime(
                    FaceRecognizerConfig(
                        model_name=face.model_name,
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
        """Initialize delivery and optionally start the configured network stream."""

        if self.config is None or self.config_error is not None:
            return
        self._ensure_rich_readiness_probe()
        try:
            with self._control_lock:
                self._ensure_delivery()
            stream_url = self.configured_stream_url()
            if self.autostart and stream_url:
                self.start_stream(stream_url)
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
                or not self._tracking_requested
                or self._rich_labels is not None
                or self._rich_probe_thread is not None
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
            if self.config is None or not self._tracking_requested:
                return True
            rich = self.config.rich_labeling
        client = None
        try:
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
                coordinator = RichLabelCoordinator(
                    client,
                    max_fps=rich.max_fps,
                    history_size=rich.history_size,
                    max_interpolation_gap_seconds=(
                        rich.max_interpolation_gap_seconds
                    ),
                    max_hold_seconds=rich.max_hold_seconds,
                    on_snapshot=self._on_tracking_snapshot,
                )
                with self._state_lock:
                    coordinator.reset(self._active_rich_session_id)
                    self._rich_labels = coordinator
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
    ) -> dict[str, Any]:
        if self.config is None or self.config_error is not None:
            raise RuntimeError(self.config_error or "configuration unavailable")
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

        with self._control_lock:
            if self._closed:
                raise RuntimeError("harness service is stopping")
            if self._route_probe["state"] == "running":
                raise RuntimeError("wait for the route check before starting a session")
            if self._session is not None and self._session.running:
                raise RuntimeError(
                    "a harness session is already running; stop it first"
                )
            needs_model_key = selected_analysis == EVENT_ANALYSIS_MODE or (
                selected_analysis == MULTI_ANALYSIS_MODE
                and any(feature in SAFETY_EVENT_LABELS for feature in selected_features)
            )
            if needs_model_key and not use_dry_run and not self._aiclub_api_key:
                raise MissingAPIKeyError(
                    "missing model API key; set " + self.config.model_api.api_key_env
                )
            self._ensure_delivery()

            face_session = None
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
                    face_session = FaceRecognitionSession(face_runtime)
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
                        face_session = FaceRecognitionSession(face_runtime)
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
                client = build_selected_features_grounding_client(
                    session_config.inference,
                    session_config.model_api,
                    features=selected_features,
                    event_sensitivities=selected_sensitivities,
                    stranger_client=face_session or face_runtime,
                    safety_max_image_side=self.config.video.max_image_side,
                    safety_jpeg_quality=self.config.video.jpeg_quality,
                    face_live_scheduler=face_live_scheduler,
                    dry_run=use_dry_run,
                    api_key=self._aiclub_api_key,
                    transport=(
                        self._pooled_transport()
                        if self._route_pool_enabled and not use_dry_run
                        and any(item != "stranger" for item in selected_features)
                        else None
                    ),
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
                session = StreamingSession.for_stream(
                    session_config,
                    client,
                    callbacks,
                    stream_url=stream_url,
                    whep_headers=session_config.stream.whep_headers,
                    source_id=self.source_id,
                    archive=self.event_archive(),
                )
            else:
                session = StreamingSession.for_camera(
                    session_config,
                    client,
                    callbacks,
                    source_id=self.source_id,
                    archive=self.event_archive(),
                )

            if face_session is not None:
                face_session.bind_rich_labels(
                    session.stream_session_id,
                    self._rich_history if self._tracking_requested else None,
                )

            with self._state_lock:
                self._generation = generation
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
                self.frames.reset()
                self.preview_frames.reset()
                self._session = session
                if self._scene is not None:
                    self._scene.close()
                self._scene = (
                    SceneRuntime(session.stream_session_id, self.source_id)
                    if self._scene_enabled else None
                )
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
                        "webhook_enabled": self._delivery is not None,
                        "pipeline": None,
                    }
                )
            try:
                session.start()
            except Exception:
                with self._state_lock:
                    if self._rich_labels is not None:
                        self._rich_labels.reset(None)
                    self._active_rich_session_id = None
                    self._session = None
                    if self._scene is not None:
                        self._scene.close()
                    self._scene = None
                    self._session_mode = None
                    self._camera_owner = None
                    self._state.update(
                        {
                            "pipeline_state": "error",
                            "input_mode": None,
                            "last_error": "could not start streaming session",
                        }
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
        """Stop the current run but keep the service available for another."""

        with self._control_lock:
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
                self._generation += 1
                if self._scene is not None:
                    self._scene.close()
                self._state.update(
                    {
                        "pipeline_state": "stopping",
                        "generation": self._generation,
                        "ready": False,
                    }
                )

            session.stop()
            session.join(timeout=10.0)
            if session.running:
                self._update_state(
                    pipeline_state="error",
                    last_error="the previous video session did not stop in time",
                )
                return False

            self._delayed_overlay.reset()
            self._overlay_fusion.reset()
            if self._rich_labels is not None:
                self._rich_labels.reset(None)
            self.annotated_frames.reset()

            with self._state_lock:
                self._active_rich_session_id = None
                self._session = None
                self._session_mode = None
                self._camera_owner = None
                self._session_started_monotonic = None
                self._camera_max_image_side = self.config.video.max_image_side
                self._state.update(
                    {
                        "pipeline_state": "idle",
                        "input_mode": None,
                        "ready": False,
                        "stream": None,
                    }
                )
            return True

    def stop_camera_owned_by(self, owner: str) -> None:
        self.stop_session(camera_owner=owner)

    def stop(self) -> None:
        with self._control_lock:
            self._closed = True
            self._route_probe_stop.set()
            self._rich_probe_stop.set()
        self.stop_session()
        if self._rich_probe_thread is not None:
            self._rich_probe_thread.join(timeout=3.0)
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
        self._update_state(pipeline_state="stopped", ready=False)

    def push_camera_frame(
        self,
        jpeg: bytes,
        *,
        width: int,
        height: int,
        owner: str | None = None,
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
        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("camera payload is not a valid JPEG")
        actual_height, actual_width = image.shape[:2]
        if (actual_width, actual_height) != (width, height):
            raise ValueError("camera JPEG dimensions do not match its header")
        display_jpeg, _, _ = encode_jpeg(
            image,
            max_image_side=self._display_settings.max_image_side,
            quality=self._display_settings.jpeg_quality,
        )

        with self._control_lock:
            session = self._session
            if (
                session is None
                or self._session_mode != CAMERA_MODE
                or not session.running
                or (self._camera_owner is not None and owner != self._camera_owner)
            ):
                return False
            sequence = self._camera_frame_sequence
            self._camera_frame_sequence += 1
            started = self._session_started_monotonic or time.monotonic()
            timestamp = max(0.0, time.monotonic() - started)
            should_sample = timestamp + 1e-9 >= self._camera_next_sample_at
            if should_sample:
                while self._camera_next_sample_at <= timestamp + 1e-9:
                    self._camera_next_sample_at += self._camera_sample_period
                session.push_camera_frame(
                    jpeg,
                    width=width,
                    height=height,
                    timestamp_seconds=timestamp,
                    source_frame_index=sequence,
                )
            generation = self._generation
            self._guarded_callback(
                generation,
                self._on_display_frame,
                display_jpeg,
                timestamp,
                sequence,
            )
            return True

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            payload = copy.deepcopy(self._state)
            rich_labels = self._rich_labels
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
        payload["event_archive"] = (
            self._events.status() if self._events else {"state": "idle"}
        )
        payload["annotated_output"] = self._delayed_overlay.stats()
        payload["rich_labeling"] = (
            rich_labels.stats()
            if rich_labels is not None
            else {
                "state": "disabled",
                **({"disabled_reason": rich_disabled_reason}
                   if rich_disabled_reason else {}),
            }
        )
        ready, reason = self.readiness()
        payload["ready"] = ready
        payload["readiness_reason"] = reason
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

    def scene_snapshot(self) -> dict[str, Any]:
        """Experimental scene state; no images or automatic inference side effects."""
        with self._state_lock:
            payload = (
                self._scene.snapshot() if self._scene is not None else {
                    "enabled": self._scene_enabled,
                    "mode": "shadow",
                    "state": "idle" if self._scene_enabled else "disabled",
                    "automatic_temporal_inference": False,
                    "scene": None, "inspections": [],
                }
            )
            payload["observer_errors"] = self._scene_errors
        safe = public_safe_value(payload)
        assert isinstance(safe, dict)
        return safe

    def _observe_scene(self, method: str, *args: Any, **kwargs: Any) -> None:
        scene = getattr(self, "_scene", None)
        if scene is not None:
            try:
                getattr(scene, method)(*args, **kwargs)
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
        }:
            return False, str(state)
        if session is None or not session.running:
            return False, "pipeline_not_running"
        if last_frame is None:
            return False, "waiting_for_first_frame"
        now = time.monotonic()
        if now - last_frame > self.settings.ready_max_frame_age_seconds:
            return False, "frame_stale"
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
        self._delayed_overlay.submit_analysis_frame(result, source_jpeg)
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
        if getattr(self, "_scene", None) is not None:
            session = self._session
            if (
                session is not None
                and session.stream_session_id == result.stream_session_id
            ):
                canonical = session.person_registry.snapshot_at(
                    result.frame.timestamp_seconds, result.frame.source_frame_index,
                )
                if canonical is not None and (
                    canonical.timestamp_seconds, canonical.frame_sequence
                ) == (result.frame.timestamp_seconds, result.frame.source_frame_index):
                    self._observe_scene("observe_tracks", canonical)
            self._observe_scene("observe_result", result)
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
        if self._rich_labels is not None:
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
        self._delayed_overlay.submit_frame(packet)
        frame_count = int(self._state["source_frames_received"]) + 1
        self._last_frame_monotonic = time.monotonic()
        self._update_state(
            pipeline_state="running",
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
            isinstance(analysis, PersonOverlayAnalysis)
            or analysis.analysis_kind in INDEPENDENT_FACE_ANALYSIS_KINDS
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
        self._observe_scene("close")
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
    if isinstance(analysis, PersonOverlayAnalysis):
        return draw_face_boxes(image, analysis.objects)
    return draw_overlay(image, **_analysis_overlay_arguments(analysis))


def _analysis_overlay_arguments(analysis: DisplayAnalysis) -> dict[str, Any]:
    known_faces = ()
    if analysis.analysis_kind == "face_recognition":
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
        # Pending clears affect the event title only. Their historical object
        # IDs and coordinates cannot be attached to the current negative frame.
        for judgment in observed_positives:
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
            if item.object_id in labels_by_object
        )
        # Hide identity boxes during a positive safety event, but do not hide
        # observed known people just because the safety branch is still pending.
        if (
            analysis.analysis_kind in (
                FIXED_FIVE_ANALYSIS_KIND, SELECTED_FEATURES_ANALYSIS_KIND,
            )
            and not any(
                judgment.event_label in SAFETY_EVENT_LABELS
                for judgment in detected_judgments
            )
        ):
            known_faces = tuple(
                item for item in analysis.objects
                if item.object_id not in labels_by_object
                and item.label == "known_person"
                and item.attributes.get("status") == "known"
                and item.attributes.get("person_id")
            )
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
        "known_faces": known_faces,
        "analysis_kind": analysis.analysis_kind,
        "incomplete": bool(analysis.missing_event_labels),
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

    application = FastAPI(
        title="Goodvision Streaming Harness",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        swagger_ui_oauth2_redirect_url=None,
    )
    install_event_routes(application)

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
    async def ui_info(request: Request) -> dict[str, Any]:
        return _runtime(request).ui_info()

    @application.get("/v1/deployment/settings")
    async def deployment_settings(request: Request) -> dict[str, Any]:
        return _runtime(request).deployment_settings()

    @application.get("/v1/deployment/routes")
    async def route_status(request: Request) -> dict[str, Any]:
        return _runtime(request).route_probe_status()

    @application.post("/v1/deployment/routes/check", status_code=202)
    async def check_routes(request: Request) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(_runtime(request).start_route_probe)
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
            return await asyncio.to_thread(
                _runtime(request).select_model_route, payload["route_id"],
            )
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
            return await asyncio.to_thread(
                _runtime(request).update_deployment_settings,
                payload,
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
        await websocket.send_json({"type": "hello", **runtime.ui_info()})
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text_message = message.get("text")
                if text_message is not None:
                    try:
                        payload = json.loads(text_message)
                    except json.JSONDecodeError:
                        await websocket.send_json(
                            {"type": "error", "message": "control message must be JSON"}
                        )
                        continue
                    if not isinstance(payload, dict):
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "control message must be an object",
                            }
                        )
                        continue
                    command = payload.get("type")
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
                            await websocket.send_json(
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
                            await websocket.send_json(
                                {"type": "error",
                                 "message": PUBLIC_MISSING_API_KEY_ERROR}
                            )
                        except ValueError as exc:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": redact_public_text(exc),
                                }
                            )
                        except Exception as exc:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": public_error_message(
                                        exc, fallback="could not start session"
                                    ),
                                }
                            )
                        else:
                            await websocket.send_json(response)
                    elif command == "stop":
                        await asyncio.to_thread(runtime.stop_session)
                        await websocket.send_json(
                            {"type": "ended", "reason": "stopped", "error": None}
                        )
                    else:
                        await websocket.send_json(
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
                    await websocket.send_json(
                        {"type": "error", "message": "camera frame is truncated"}
                    )
                    continue
                width, height = CAMERA_FRAME_HEADER.unpack_from(data)
                try:
                    await asyncio.to_thread(
                        runtime.push_camera_frame,
                        data[CAMERA_FRAME_HEADER.size :],
                        width=width,
                        height=height,
                        owner=owner,
                    )
                except ValueError as exc:
                    await websocket.send_json(
                        {"type": "error", "message": redact_public_text(exc)}
                    )
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            await asyncio.to_thread(runtime.stop_camera_owned_by, owner)

    @application.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "live"}

    @application.get("/health/ready")
    async def health_ready(request: Request) -> JSONResponse:
        runtime = _runtime(request)
        ready, reason = runtime.readiness()
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "reason": reason},
            status_code=200 if ready else 503,
        )

    @application.get("/v1/status")
    async def status(request: Request) -> dict[str, Any]:
        return _runtime(request).status()

    @application.post("/v1/sessions/start")
    async def start_session(
        request: Request,
        payload: Annotated[dict[str, Any] | None, Body()] = None,
    ) -> dict[str, Any]:
        """Start one network-stream session with any selected feature subset."""

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
    async def latest_rich_labels(request: Request) -> JSONResponse:
        payload = _runtime(request).latest_rich_labels()
        if payload is None:
            raise HTTPException(status_code=404, detail="no rich labels yet")
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @application.get("/v1/experimental/scene", include_in_schema=False)
    async def experimental_scene(request: Request) -> JSONResponse:
        return JSONResponse(
            _runtime(request).scene_snapshot(), headers={"Cache-Control": "no-store"},
        )

    @application.get("/v1/results/events")
    async def result_events(
        request: Request,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        return _result_events_response(request, _runtime(request), after=after)

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
        return _snapshot_response(
            _runtime(request).annotated_frames,
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
