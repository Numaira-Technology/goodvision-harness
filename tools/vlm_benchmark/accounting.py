"""Delivery accounting independent of live-result promotion and model quality."""


def complete(report):
    return (
        report.get("state") == "completed" and report.get("analysis_complete") is True
    )


def delivery_counts(reports, expected):
    attempted = [
        r for r in reports if r.get("timing", {}).get("http_request_attempted") is True
    ]
    not_attempted = sum(
        r.get("timing", {}).get("http_request_attempted") is False for r in reports
    )
    unknown = len(reports) - len(attempted) - not_attempted
    history = [r for r in reports if r.get("historical_only") is True]
    good = sum(complete(r) for r in reports)
    historical_good = sum(complete(r) for r in history)
    return {
        "planned_windows": expected,
        "recorded_results": len(reports),
        # Missing scheduled work is not automatically an unsent request. It may
        # be unrecorded/outstanding work; retention reconciliation is separate.
        "planned_without_record": max(0, expected - len(reports)),
        "records_exceed_plan": len(reports) > expected,
        "http_attempted": len(attempted),
        "http_not_attempted_records": not_attempted,
        "http_attempt_unknown_records": unknown,
        "http_attempted_complete": sum(complete(r) for r in attempted),
        "http_attempted_without_complete": sum(not complete(r) for r in attempted),
        "inspection_attempted": sum(
            r.get("timing", {}).get("inspection_attempted") is True for r in reports
        ),
        "complete_results": good,
        "historical_reports": len(history),
        "historical_complete_results": historical_good,
        "nonhistorical_complete_results": good - historical_good,
        "historical_failed_reports": sum(r.get("state") == "failed" for r in history),
        "scope": (
            "HTTP attempted means client send invoked, not provider receipt; "
            "historical results count as returned, not current display. "
            "Missing HTTP metadata remains unknown."
        ),
    }


def reconcile_retention(reports_by_camera, stats, lanes, lane_name):
    """Check each source/lane after drain; matching grand totals can hide losses."""
    rows, missing_instrumentation = [], False
    for camera, (reports, state) in enumerate(
        zip(reports_by_camera, stats, strict=True), 1
    ):
        for lane in lanes:
            worker = state.get("request_accounting", {}).get(lane)
            recorded = [r for r in reports if lane_name(r) == lane]
            delivery = delivery_counts(recorded, len(recorded))
            discarded = (
                state.get("late_information_reports_discarded", 0)
                if lane == "events"
                else 0
            )
            if worker is None:
                missing_instrumentation = True
                valid = False if discarded else None
            else:
                ids = [r.get("request_id") for r in recorded]
                valid = (
                    worker.get("submitted")
                    == worker.get("finished")
                    == worker.get("published")
                    == len(recorded)
                    and worker.get("inspection_attempted", 0)
                    == delivery["inspection_attempted"]
                    and worker.get("http_attempted", 0) == delivery["http_attempted"]
                    and worker.get("http_not_attempted", 0)
                    == delivery["http_not_attempted_records"]
                    and worker.get("http_attempt_unknown", 0)
                    == delivery["http_attempt_unknown_records"]
                    and not discarded
                    and all(ids)
                    and len(set(ids)) == len(ids)
                )
            rows.append(
                {
                    "camera": camera,
                    "lane": lane,
                    "valid": valid,
                    "worker": worker,
                    "recorded_results": len(recorded),
                    "legacy_discarded_results": discarded,
                }
            )
    return {
        "valid": False
        if any(r["valid"] is False for r in rows)
        else None
        if missing_instrumentation
        else True,
        "per_stream_lane": rows,
        "scope": (
            "Submitted evaluations must finish, publish and be retained exactly "
            "once per camera/lane; no claim that every planned window was submitted."
        ),
    }
