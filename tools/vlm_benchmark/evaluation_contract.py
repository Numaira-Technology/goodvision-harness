"""Fixed per-camera deadline contract shared by tracked benchmark reports."""


def evaluation_contract():
    """A scoring deadline, independent of the HTTP cancellation timeout."""
    return {
        "deadline_seconds": 3,
        "minimum_on_time_coverage": 0.95,
        "cameras": 12,
        "planned_windows_per_camera": 69,
        "required_timely_per_camera": 66,
        "latency_field": "source_to_publish_seconds",
        "scope": "latest evidence enters VLM branch through result publication",
        "denominator": "all planned windows, including unsent/failed/missing/late",
        "parallel": "same camera/window_id; both complete within deadline",
        "acceptance": "every camera in each mode and each repeat independently",
    }
