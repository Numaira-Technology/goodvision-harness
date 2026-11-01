"""Per-camera scoring; never use a pooled average to accept a twelve-stream run."""

import math

from tools.vlm_benchmark.accounting import complete
from tools.vlm_benchmark.evaluation_contract import evaluation_contract
from tools.vlm_multistream_benchmark import report_lane


def _latency(report):
    value = report.get("timing", {}).get("source_to_publish_seconds")
    return (
        value
        if type(value) in (int, float) and math.isfinite(value) and value >= 0
        else None
    )


def assess_timeliness(raw, lanes, *, planned_per_camera=None, cameras=None):
    """Pair exact production window IDs, including failures and historical replies.

    Missing source windows still count against 69. A missing/invalid timestamp
    cannot be used to claim timely completion. Duplicate windows are corruption,
    not additional coverage. For parallel, both branches must meet the deadline.
    """
    contract = evaluation_contract()
    if planned_per_camera is not None:
        if type(planned_per_camera) is not int or planned_per_camera < 1:
            raise ValueError("invalid planned windows")
        contract["planned_windows_per_camera"] = planned_per_camera
        contract["required_timely_per_camera"] = math.ceil(
            planned_per_camera * contract["minimum_on_time_coverage"]
        )
    if cameras is not None:
        if type(cameras) is not int or not 1 <= cameras <= 12:
            raise ValueError("invalid cameras")
        contract["cameras"] = cameras
    expected = contract["planned_windows_per_camera"]
    deadline = contract["deadline_seconds"]
    lanes = set(lanes)
    if not lanes or not lanes <= {"events", "grounding"}:
        raise ValueError("invalid timeliness lanes")
    windows = {camera: {} for camera in range(1, contract["cameras"] + 1)}
    for item in raw:
        camera, report = item["camera"], item["report"]
        lane, wid = report_lane(report), report.get("window_id")
        if (
            type(camera) is not int
            or camera not in windows
            or lane not in lanes
            or not isinstance(wid, str)
            or not wid
        ):
            raise ValueError("invalid camera/lane/window_id for timeliness")
        pair = windows[camera].setdefault(wid, {})
        if lane in pair:
            raise ValueError("duplicate camera/window/lane in timeliness")
        pair[lane] = report
        if len(windows[camera]) > expected:
            raise ValueError("camera windows exceed frozen plan")

    def assess(required):
        cameras = []
        for camera, pairs in windows.items():
            # A pair is retained if either branch exists. It is only sent or
            # complete if ALL required branches are sent or complete.
            retained = [
                [p.get(lane) for lane in required]
                for p in pairs.values()
                if any(lane in p for lane in required)
            ]
            good = [rows for rows in retained if all(r and complete(r) for r in rows)]
            latencies = [
                max(values) if all(v is not None for v in values) else None
                for rows in good
                for values in [[_latency(r) for r in rows]]
            ]
            timely = sum(v is not None and v <= deadline for v in latencies)
            cameras.append(
                {
                    "camera": camera,
                    "planned_windows": expected,
                    "recorded_windows": len(retained),
                    "missing_windows": expected - len(retained),
                    "sent_windows": sum(
                        all(
                            r
                            and r.get("timing", {}).get("http_request_attempted")
                            is True
                            for r in rows
                        )
                        for rows in retained
                    ),
                    "complete_windows": len(good),
                    "incomplete_or_missing_windows": expected - len(good),
                    "timely_complete_windows": timely,
                    "late_complete_windows": sum(
                        v is not None and v > deadline for v in latencies
                    ),
                    "complete_without_valid_latency": sum(v is None for v in latencies),
                    "not_timely_windows": expected - timely,
                    "on_time_coverage": timely / expected,
                    "passes_target": timely >= contract["required_timely_per_camera"],
                }
            )
        total_timely = sum(c["timely_complete_windows"] for c in cameras)
        return {
            "scope": "both_branches" if len(required) == 2 else required[0],
            "deadline_seconds": deadline,
            "minimum_on_time_coverage": contract["minimum_on_time_coverage"],
            "planned_windows": expected * len(cameras),
            "timely_complete_windows": total_timely,
            "on_time_coverage": total_timely / (expected * len(cameras)),
            "worst_camera_coverage": min(c["on_time_coverage"] for c in cameras),
            "cameras_passing": sum(c["passes_target"] for c in cameras),
            "all_cameras_pass": all(c["passes_target"] for c in cameras),
            "per_camera": cameras,
        }

    return {
        "lanes": {lane: assess([lane]) for lane in sorted(lanes)},
        "joint": assess(["events", "grounding"]) if len(lanes) == 2 else None,
    }
