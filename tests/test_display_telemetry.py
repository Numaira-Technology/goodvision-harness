"""Numeric browser diagnostics retain the image-free reporting boundary."""

import pytest

from goodvision_harness.display_telemetry import validate_display_metrics


def test_receive_apply_and_browser_stall_diagnostics_are_accepted():
    report = {
        "stream_session_id": "run", "generation": 2, "crop_version": 0,
        "metadata_ws_received": 3, "metadata_duplicate_revisions": 1,
        "metadata_invalid_revisions": 0, "last_metadata_applied_age_seconds": 0.02,
        "latest_received_anchor_age_seconds": 1.3, "latest_applied_anchor_age_seconds": 1.35,
        "page_hidden": 0, "visibility_changes": 2, "hidden_seconds": 0.4,
        "longtask_count": 1, "longtask_total_seconds": 0.08,
        "window_capture_total_count": 100, "window_capture_total_mean_seconds": 0.004,
        "window_capture_total_sample_count": 100,
        "window_capture_total_p95_seconds": 0.01, "window_capture_total_max_seconds": 0.015,
        "window_metadata_apply_count": 3, "window_metadata_apply_mean_seconds": 0.006,
        "window_longtask_max_seconds": 0.08, "motion_ready_elapsed_seconds": 0.3,
        "complete_contour_frames": 90, "continuous_contour_frames": 96,
        "approximate_contour_frames": 6, "approximate_subjects": 9,
        "continuous_contour_frame_ratio": 0.96,
        "display_catchup_expired": 9,
        "motion_frames_admitted": 2, "motion_frames_skipped": 1300,
        "motion_frames_expired": 1,
        "worker_pending_frames": 1, "worker_pending_present_queries": 2,
        "worker_pending_metadata": 2, "motion_metadata_coalesced": 500,
        "geometry_cache_hits": 10, "geometry_cache_misses": 20,
        "pending_present_queries": 2, "source_width": 640, "source_height": 480,
        "motion_input_width": 138, "motion_input_height": 103, "presentation_prefetch_seconds": 1.5,
    }
    assert validate_display_metrics(report) == report


def test_display_instance_is_optional_and_cannot_contain_arbitrary_text():
    report = {"stream_session_id": "run", "generation": 2, "crop_version": 0}
    assert validate_display_metrics(report) == report
    report["display_instance_id"] = "viewer-abc_123"
    assert validate_display_metrics(report) == report
    for value in ("https://private.example/path", "x" * 97, "", 3):
        with pytest.raises(ValueError):
            validate_display_metrics({**report, "display_instance_id": value})


@pytest.mark.parametrize("field,value", [
    ("longtask_entries", [{"name": "private page"}]),
    ("window_capture_total_count", 0.5),
    ("window_capture_total_max_seconds", float("nan")),
    ("latest_applied_anchor_age_seconds", -1),
    ("continuous_contour_frame_ratio", 1.1),
    ("approximate_subjects", 0.5),
    ("display_catchup_expired", -1),
])
def test_diagnostics_do_not_admit_raw_entries_or_invalid_measurements(field, value):
    with pytest.raises(ValueError):
        validate_display_metrics({
            "stream_session_id": "run", "generation": 2, "crop_version": 0,
            field: value,
        })
