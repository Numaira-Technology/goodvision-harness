"""Validate image-free browser presentation measurements."""

import math
import re

COUNTERS = frozenset({
    "network_requests", "network_failures", "network_bytes_received",
    "network_frames_received", "network_source_gap_frames",
    "captured_frames", "distinct_presented_frames", "display_buffer_expired",
    "display_catchup_expired", "network_queue_dropped_frames", "video_callback_frames", "video_callback_gap_frames",
    "rejected_source_frames", "buffered_frames", "analysis_admitted", "analysis_sent",
    "analysis_pending", "analysis_upload_failures", "analysis_capture_pauses",
    "analysis_prepared", "analysis_prepared_pending",
    "motion_failures", "metadata_failures", "frames_without_contours",
    "dense_motion_ready", "motion_initialization_failures", "motion_warmup_reused",
    "relocalized_contour_frames", "partial_contour_frames", "complete_contour_frames",
    "continuous_contour_frames", "approximate_contour_frames", "approximate_subjects",
    "geometry_cache_hits", "geometry_cache_misses",
    "pending_present_queries", "source_width", "source_height", "motion_input_width", "motion_input_height",
    "backward_contour_frames",
    "observed_contour_frames", "propagated_contour_frames", "report_sequence",
    "window_presented_frames",
    "metadata_requests", "metadata_received", "metadata_timeouts",
    "metadata_http_errors", "metadata_decode_errors", "metadata_run_mismatches",
    "metadata_anchors_received", "worker_frames_completed", "worker_metadata_accepted",
    "metadata_ws_received", "metadata_duplicate_revisions", "metadata_invalid_revisions",
    "worker_anchor_rejections", "worker_results_received", "worker_results_late",
    "worker_present_expired",
    "motion_frames_admitted", "motion_frames_skipped", "motion_frames_expired",
    "motion_metadata_coalesced", "worker_pending_present_queries",
    "worker_pending_metadata",
    "worker_pending_frames", "coverage_not_ready", "coverage_no_anchor",
    "preprocess_frames_completed", "preprocess_failures", "preprocess_pending_frames", "motion_pending_frames",
    "coverage_anchor_expired", "coverage_motion_gap", "coverage_motion_unreliable",
    "coverage_missing_source",
    "warp_reject_invalid_input", "warp_reject_internal_support",
    "warp_reject_support_span", "warp_reject_affine_geometry",
    "warp_reject_component_visibility", "warp_reject_component_appearance",
    "warp_reject_region_occluded", "warp_reject_boundary_occluded",
    "warp_reject_crop_continuation", "warp_reject_invalid_geometry",
    "warp_reject_invalid_polygon",
    "page_hidden", "visibility_changes", "longtask_count",
    "window_callback_frames", "window_callback_gap_frames",
    "estimated_display_buffer_bytes", "required_display_buffer_bytes",
    "display_buffer_capacity_failures",
})
MEASUREMENTS = frozenset({
    "distinct_presented_fps", "last_source_to_present_seconds", "elapsed_seconds",
    "source_callback_fps",
    "window_seconds", "window_presented_fps", "source_to_present_p50_seconds",
    "source_to_present_p95_seconds", "source_to_present_max_seconds",
    "contour_frame_ratio", "complete_contour_frame_ratio", "continuous_contour_frame_ratio", "max_present_gap_seconds",
    "metadata_latency_max_seconds", "last_metadata_received_age_seconds",
    "worker_frame_latency_max_seconds", "worker_result_latency_max_seconds",
    "motion_ready_elapsed_seconds", "last_metadata_applied_age_seconds",
    "motion_runtime_ready_elapsed_seconds", "motion_runtime_load_seconds",
    "motion_startup_drain_seconds",
    "configured_display_delay_seconds", "presentation_prefetch_seconds",
    "latest_received_anchor_age_seconds", "latest_applied_anchor_age_seconds",
    "hidden_seconds", "longtask_total_seconds",
})
# Bounded per-window aggregate timings only; never browser URLs, pixels or labels.
TIMING_STAGES = (
    "network_download", "network_decode",
    "callback_gap", "callback_lateness", "source_freeze", "display_capture",
    "motion_capture", "motion_preprocess", "analysis_capture", "capture_total", "analysis_encode",
    "motion_preprocess_queue", "motion_compute", "motion_frame_queue",
    "motion_present_queue", "motion_present_compute",
    "analysis_ack", "metadata_apply", "presentation", "longtask",
)
COUNTERS |= frozenset(
    f"window_{stage}_{stat}" for stage in TIMING_STAGES for stat in ("count", "sample_count")
)
MEASUREMENTS |= frozenset(
    f"window_{stage}_{stat}_seconds"
    for stage in TIMING_STAGES
    for stat in ("mean", "p95", "max")
)
IDENTITY = frozenset({"stream_session_id", "generation", "crop_version"})


def validate_display_metrics(payload: object) -> dict:
    if not isinstance(payload, dict) or not IDENTITY <= payload.keys() or (
        payload.keys() - IDENTITY - COUNTERS - MEASUREMENTS - {"display_instance_id"}
    ):
        raise ValueError("invalid display metrics fields")
    sid = payload["stream_session_id"]
    if not isinstance(sid, str) or not 1 <= len(sid) <= 256:
        raise ValueError("invalid display run")
    if "display_instance_id" in payload and (
        not isinstance(payload["display_instance_id"], str)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,96}", payload["display_instance_id"]) is None
    ):
        raise ValueError("invalid display instance")
    for key in (COUNTERS | {"generation", "crop_version"}) & payload.keys():
        if type(payload[key]) is not int or not 0 <= payload[key] <= 2**53 - 1:
            raise ValueError("invalid display counter")
    for key in MEASUREMENTS & payload.keys():
        value = payload[key]
        if value is None:
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid display measurement")
        if key in {"contour_frame_ratio", "complete_contour_frame_ratio", "continuous_contour_frame_ratio"} and value > 1:
            raise ValueError("invalid contour ratio")
    return dict(payload)
