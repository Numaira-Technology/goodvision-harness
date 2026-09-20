"""Replaceable perception routes behind the shared source/tracking contract.

The application selects one descriptor per session. It never imports detector
or SAM code: removing a route does not change person, event or display logic.
Legacy video clients keep their original strict protocol when no route is set.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlparse

import httpx

from .rich_labeling import RichLabelProtocolError
from .sam_tracking_pipeline import PipelinedVideoSessionTrackingCoordinator
from .video_tracking import VideoTrackingCoordinator, VideoTrackingHTTPClient
from .video_tracking_session import (
    VideoSessionTrackingCoordinator,
    VideoSessionTrackingHTTPClient,
    _limits,
    _require,
)


@dataclass(frozen=True, slots=True)
class ProcessingMethod:
    id: str
    label: str
    description: str
    backend: str
    confirmation_policy: str
    native_sam_video_tracking: bool


METHODS = {
    method.id: method
    for method in (
        ProcessingMethod(
            "sam_live",
            "人员追踪模块",
            "人员追踪模块持续跟踪。先返回当前观察，再确认新人物。",  # noqa: RUF001
            "sam3.1_video",
            "causal_native_three_detections",
            True,
        ),
    )
}


class PerceptionSessionHTTPClient(VideoSessionTrackingHTTPClient):
    """The route extension is explicit; legacy endpoints never accept it."""

    append_metadata_fields = frozenset({"route_metadata"})
    # The causal SAM route replays the last identical request without inference.
    # Legacy session clients retain their fail-closed, non-retrying behavior.
    append_transport_retries = 1

    def __init__(self, endpoint, *, method: str, **kwargs):
        self.method = METHODS[method]
        self.backend = self.method.backend
        self.native_sam_video_tracking = self.method.native_sam_video_tracking
        self.session_path = f"/v1/perception/{method}/sessions"
        self.health_path = f"/health/perception/{method}"
        path = urlparse(endpoint).path
        if path.startswith("/v1/cameras/"):
            lane = re.fullmatch(
                rf"/v1/cameras/(camera-(?:0[1-9]|[1-9][0-9]+))/perception/{method}/sessions/?",
                path,
            )
            if lane is None:
                raise ValueError("invalid local camera tracking endpoint")
            camera = lane.group(1)
            self.session_path = f"/v1/cameras/{camera}/perception/{method}/sessions"
            self.health_path = f"/health/cameras/{camera}/perception/{method}"
        super().__init__(endpoint, **kwargs)
        self._capabilities = {"startup_frames": 1, "max_append_frames": 4}

    @staticmethod
    def validate_limits(value):
        result = _limits(value, causal=True)
        _require(
            result.get("hotstart_delay") == 1
            and result.get("max_pending_frames") == 0
            and result["startup_frames"] == 1
            and result["max_append_frames"] == 4,
            "unsupported causal route limits",
        )
        return result

    def ready_for_new_session(self):
        # Route health reports the worker's global ownership, including a
        # sibling plugin whose asynchronous teardown has not finished.
        return self.gpu_ready() and self._health_active_session is False

    def prepare_snapshots(self, payload, snapshots):
        metadata = payload["route_metadata"]
        _require(
            isinstance(metadata, dict)
            and {"route", "confirmation_policy", "observations"} <= set(metadata)
            and set(metadata)
            <= {"route", "confirmation_policy", "observations", "diagnostics"}
            and metadata["route"] == self.method.id
            and metadata["confirmation_policy"] == self.method.confirmation_policy,
            "invalid perception route metadata",
        )
        observations = metadata["observations"]
        _require(
            isinstance(observations, list) and len(observations) == len(snapshots),
            "missing perception confirmation provenance",
        )
        diagnostics = metadata.get("diagnostics", {})
        _require(
            isinstance(diagnostics, dict)
            and len(diagnostics) <= 32
            and len(json.dumps(diagnostics, allow_nan=True)) <= 4096,
            "unbounded perception diagnostics",
        )
        for key, value in diagnostics.items():
            _require(
                isinstance(key, str)
                and len(key) <= 80
                and (
                    value is None
                    or type(value) is bool
                    or (type(value) in (int, float) and math.isfinite(value))
                    or (isinstance(value, str) and len(value) <= 160)
                ),
                "invalid perception diagnostic value",
            )
        result = []
        for observation, snapshot in zip(observations, snapshots, strict=True):
            _require(
                isinstance(observation, dict)
                and set(observation) == {"frame_sequence", "provisional_object_ids"}
                and type(observation["frame_sequence"]) is int
                and observation["frame_sequence"] == snapshot.frame_sequence,
                "confirmation is not from the emitted source frame",
            )
            ids = observation["provisional_object_ids"]
            native_ids = {
                obj.attributes["native_track"]["object_id"] for obj in snapshot.objects
            }
            _require(
                isinstance(ids, list)
                and len(ids) <= len(native_ids)
                and all(isinstance(item, str) for item in ids)
                and len(set(ids)) == len(ids)
                and set(ids) <= native_ids,
                "provisional identities do not match observed objects",
            )
            result.append(
                replace(
                    snapshot,
                    objects=tuple(
                        replace(
                            obj,
                            attributes={
                                **obj.attributes,
                                "tracking_confirmation": (
                                    "provisional"
                                    if obj.attributes["native_track"]["object_id"]
                                    in ids
                                    else "confirmed"
                                ),
                            },
                        )
                        for obj in snapshot.objects
                    ),
                )
            )
        return tuple(result)


def is_contour_endpoint(endpoint):
    return re.fullmatch(
        r"/v1/(?:cameras/camera-(?:0[1-9]|[1-9][0-9]+)/)?contour/sessions/?",
        urlparse(endpoint).path,
    ) is not None


class ContourSessionHTTPClient(PerceptionSessionHTTPClient):
    """Authenticated public node contract, retaining native source validation."""

    def __init__(self, endpoint, *, api_key=None, **kwargs):
        base = httpx.URL(endpoint)
        super().__init__(str(base.copy_with(path="/")), method="sam_live", **kwargs)
        self.method = replace(self.method, id="contour", backend="goodvision_contour")
        self.backend = self.method.backend
        self.session_path = "/v1/contour/sessions"
        self.health_path = "/api/status"
        if is_contour_endpoint(endpoint):
            self.session_path = base.path.rstrip("/")
            if self.session_path.startswith("/v1/cameras/"):
                camera = self.session_path.split("/")[3]
                self.health_path = f"/api/cameras/{camera}/status"
        self.endpoint = str(base.copy_with(path=self.session_path))
        self._health = str(base.copy_with(path=self.health_path))
        self._trust_env = False
        if api_key:
            if (not isinstance(api_key, str) or len(api_key) > 4096
                    or any(not 33 <= ord(char) <= 126 for char in api_key)):
                raise ValueError("invalid tracking node key")
            self._request_headers = {"Authorization": "Bearer " + api_key}

    @staticmethod
    def validate_limits(value):
        result = _limits(value, causal=True)
        _require(
            result.get("hotstart_delay") == 1
            and result.get("max_pending_frames") == 0
            and result["startup_frames"] == 1
            and result["max_append_frames"] in (1, 4),
            "unsupported Contour node limits",
        )
        return result

    def gpu_ready(self):
        if not self._request_headers:
            return False
        try:
            value = self._request(
                "GET", self._health,
                deadline=time.monotonic() + min(self._timeout_seconds, 5),
            )
            _require(value.get("model_id") == "goodvision-contour-1",
                     "unexpected tracking node")
            if not (value.get("ready") is True and value.get("gpu_ready") is True
                    and value.get("accepting_sessions") is True
                    and type(value.get("active_session")) is bool):
                return False
            limits = self.validate_limits(value["limits"])
            if self._transport_codec == "h264" and limits.get("h264_transport") != 1:
                return False
            with self._session_state:
                self._capabilities = limits
                self._health_active_session = value["active_session"]
            return True
        except (httpx.HTTPError, ValueError, KeyError, RichLabelProtocolError):
            return False


def create_tracking_client(rich, *, api_key=None):
    options: dict[str, Any] = {
        "timeout_seconds": rich.video_timeout_seconds,
        "transport_max_side": rich.video_transport_max_side,
        "transport_jpeg_quality": rich.video_transport_jpeg_quality,
    }
    method = rich.processing_method
    if method is not None and rich.backend != "sam31_stream":
        raise ValueError("processing methods require the continuous video backend")
    if rich.backend == "sam31_stream":
        options.update(
            transport_codec=rich.video_transport_codec,
            transport_crf=rich.video_transport_crf,
        )
        if is_contour_endpoint(rich.endpoint):
            return ContourSessionHTTPClient(rich.endpoint, api_key=api_key, **options)
        if method is not None:
            return PerceptionSessionHTTPClient(rich.endpoint, method=method, **options)
        return VideoSessionTrackingHTTPClient(rich.endpoint, **options)
    if rich.backend == "sam31":
        return VideoTrackingHTTPClient(rich.endpoint, **options)
    raise ValueError("unsupported video tracking backend")


def create_tracking_coordinator(client, rich, on_batch):
    options = dict(
        max_fps=rich.max_fps,
        history_size=rich.history_size,
        on_batch=on_batch,
        batch_frames=rich.video_batch_frames,
        batch_seconds=rich.video_batch_seconds,
    )
    if rich.processing_method is not None:
        # Dispatch as soon as HTTP is idle; prepare at most four accumulated
        # sources ahead. SAM's durable ingress never overwrites pending samples.
        options.update(batch_frames=1, batch_seconds=0.1, pending_capacity=4)
    coordinator = (
        PipelinedVideoSessionTrackingCoordinator
        if rich.processing_method == "sam_live"
        else VideoSessionTrackingCoordinator
        if rich.backend == "sam31_stream"
        else VideoTrackingCoordinator
    )
    if coordinator is PipelinedVideoSessionTrackingCoordinator:
        options.update(
            queue_policy=rich.video_queue_policy,
            realtime_append_frames=rich.video_realtime_append_frames,
            realtime_pending_frames=rich.video_realtime_pending_frames,
            realtime_max_age_seconds=rich.video_realtime_max_age_seconds,
            result_deadline_seconds=rich.video_result_deadline_seconds,
        )
    return coordinator(client, **options)


def processing_methods(rich):
    supported = rich is not None and rich.backend == "sam31_stream"
    return [
        dict(
            id=method.id,
            label=method.label,
            description=method.description,
            available=supported,
            reason=(
                "启动时检查服务器和模型是否就绪。"
                if supported
                else "需要启用连续视频处理服务。"
            ),
        )
        for method in METHODS.values()
    ]


def processing_status(coordinator, method):
    stats = coordinator.stats() if coordinator is not None else {}
    return {
        "method": method,
        "label": METHODS[method].label if method in METHODS else "现有处理方式",
        "source_to_result_seconds": stats.get("source_to_result_seconds"),
        "completed_analysis_fps": stats.get("completed_analysis_fps"),
        "completed_during_capture": stats.get("completed_during_capture", 0),
        "completed_after_capture": stats.get("completed_after_capture", 0),
        "capture_elapsed_seconds": stats.get("capture_elapsed_seconds", 0),
        "queue_frames": stats.get("pending_frames", 0),
        "processed_frames": stats.get("processed_frames", 0),
        "dropped_frames": stats.get("dropped_backpressure", 0),
        "queue_policy": stats.get("queue_policy"),
        "effective_max_append_frames": stats.get("effective_max_append_frames"),
        "skipped_by_reason": stats.get("skipped_by_reason", {}),
        "latest_result_age_seconds": stats.get("latest_result_age_seconds"),
        "oldest_pending_seconds": stats.get("oldest_pending_seconds"),
        "result_deadline_seconds": stats.get("result_deadline_seconds"),
        "results_within_deadline": stats.get("results_within_deadline", 0),
        "results_after_deadline": stats.get("results_after_deadline", 0),
        "admitted_deadline_coverage": stats.get("admitted_deadline_coverage"),
    }
