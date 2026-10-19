"""Public response allowlists. Internal metadata never passes through wholesale."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,160}$")]
Token = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
Number = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Sequence = Annotated[int, Field(ge=0)]
Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)


class Limits(Record):
    max_frames: int
    max_request_bytes: int
    max_jpeg_bytes: int
    max_total_jpeg_bytes: int
    max_image_pixels: int
    max_total_image_pixels: int
    inference_max_side: int
    max_objects: int
    request_timeout_seconds: Number
    max_inflight_requests: Literal[1]
    startup_frames: Literal[1]
    max_append_frames: Literal[1, 4]
    hotstart_delay: Literal[1]
    max_pending_frames: Literal[0]
    history_frames: int
    max_session_frames: int
    max_cumulative_native_ids: int
    idle_seconds: Number
    max_active_sessions: Literal[1]
    h264_transport: Literal[0, 1]


class Identity(Record):
    video_session_id: Token
    stream_session_id: Identifier
    predictor_epoch: Token
    backend: Literal["goodvision_contour"]
    memory_scope: Literal["session"]
    state: Literal["active", "closed", "invalidated"]


class SessionCreated(Identity):
    limits: Limits


class SessionDeleted(Identity):
    discarded_pending_frames: Sequence


class Box(Record):
    x1: Coordinate
    y1: Coordinate
    x2: Coordinate
    y2: Coordinate


class NativeTrack(Record):
    backend: Literal["goodvision_contour"]
    epoch: Token
    object_id: Annotated[str, Field(pattern=r"^\d{1,20}$")]


class Contours(Record):
    coordinate_system: Literal["normalized"]
    source_frame_sequence: Sequence
    source_timestamp_seconds: Number
    polygons: list[list[Annotated[list[Coordinate], Field(min_length=2, max_length=2)]]]


class MaskGeometry(Record):
    max_ignored_component_area_pixels: Sequence
    ignored_component_count: Sequence
    ignored_component_pixels: Sequence


class Attributes(Record):
    native_track: NativeTrack
    mask_geometry: MaskGeometry
    mask_contours: Contours | None = None


class TrackedObject(Record):
    track_id: Annotated[str, Field(pattern=r"^ct-[a-f0-9]{32}:\d{1,20}$")]
    label: Literal["person"]
    bbox: Box
    confidence: Coordinate
    attributes: Attributes
    observation_state: Literal["observed"]
    last_observed_frame_sequence: Sequence
    last_observed_timestamp_seconds: Number


class Frame(Record):
    stream_session_id: Identifier
    frame_sequence: Sequence
    timestamp_seconds: Number
    objects: Annotated[list[TrackedObject], Field(max_length=16)]


class Observation(Record):
    frame_sequence: Sequence
    provisional_object_ids: list[Annotated[str, Field(pattern=r"^\d{1,20}$")]]


class RouteMetadata(Record):
    route: Literal["contour"]
    confirmation_policy: Literal["causal_native_three_detections"]
    observations: list[Observation]


class SessionAppended(Identity):
    request_id: Identifier
    frames: Annotated[list[Frame], Field(max_length=4)]
    accepted_frame_sequence: Sequence
    accepted_timestamp_seconds: Number
    processed_frame_sequence: Sequence
    processed_timestamp_seconds: Number
    emitted_frame_sequence: Sequence | None
    emitted_timestamp_seconds: Number | None
    pending_frame_count: Literal[0]
    route_metadata: RouteMetadata


class CacheReclaim(Record):
    time: Sequence
    released_bytes: Sequence


class GPUMemoryStatus(Record):
    available: bool
    reason: Literal["cuda_not_initialized", "telemetry_unavailable"] | None
    reclaim_pending: bool
    reclaim_count: Sequence
    last_reclaim: CacheReclaim | None
    error_code: Literal["cache_reclaim_failed"] | None
    allocated_bytes: Sequence | None = None
    reserved_bytes: Sequence | None = None
    unused_reserved_bytes: Sequence | None = None
    peak_allocated_bytes: Sequence | None = None
    device_free_bytes: Sequence | None = None
    device_total_bytes: Sequence | None = None
