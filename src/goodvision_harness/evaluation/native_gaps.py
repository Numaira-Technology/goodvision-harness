"""Describe native skipped observations without repairing or replacing evidence."""
from __future__ import annotations

import json
from pathlib import Path

from .failures import native_report_failure_policy


def is_association_skip(failure):
    return (failure.get("stage") == "fusion" and bool(failure.get("reference"))
            and failure.get("reason") in {"expired_reports", "capacity_dropped_reports"}
            and not failure.get("error"))


def is_local_gap(failure, evidence):
    if is_association_skip(failure):
        return True
    if (not failure.get("window_id") or not failure.get("reference")
            or failure.get("stage") not in {"independent_vlm", "fusion"}
            or failure.get("vlm_lane") not in {"information", "grounding", "combined"}
            or failure.get("reason")):
        return False
    status = failure.get("status_code")
    if failure.get("error") == "http_error" and status is None:
        # Older captures omitted the HTTP status from the projection. Read its
        # already-saved report, not the provider, and never modify that capture.
        try:
            archived = json.loads(Path(failure["reference"]).read_bytes())
            report = archived["payload"]
            if (archived.get("kind") != "independent"
                    or failure["window_id"] != f"native-{archived['sequence']:08d}"
                    or report.get("stream_session_id") != evidence.get("stream_session_id")
                    or report.get("source_id") != evidence.get("source_id")
                    or report.get("vlm_lane", "combined") != failure["vlm_lane"]
                    or report.get("error_code") != failure["error"]
                    or report.get("state") != "failed"):
                return False
            status = report.get("timing", {}).get("http_status")
        except (OSError, ValueError, KeyError, TypeError):
            return False
    policy = native_report_failure_policy({"state": "failed", "error_code": failure.get("error"),
        "timing": {"http_status": status}})
    return policy["scope"] == "window"


def disclose_skipped_observations(messages, evidence):
    """Add source-time notes outside the immutable evidence and original artifact."""
    windows = {row["window_id"]: row for row in
               evidence.get("observation_manifest", {}).get("window_plan", [])}
    gaps = []
    for failure in evidence.get("failures", []):
        if not is_local_gap(failure, evidence):
            continue
        detail = failure.get("detail")
        interval = (detail if is_association_skip(failure) and isinstance(detail, dict)
                    else windows.get(failure.get("window_id"), {}))
        start, end = interval.get("start"), interval.get("end")
        gaps.append({"status": "skipped", "start": start, "end": end,
            "time_known": start is not None and end is not None,
            "affected_lane": "person_association" if is_association_skip(failure)
                else failure["vlm_lane"],
            "reference": failure["reference"]})
    payload = json.loads(messages[1]["content"])
    payload["skipped_observations"] = gaps
    return [
        {**messages[0], "content": messages[0]["content"] +
         " The skipped_observations list identifies times with skipped work in the named lane. "
         "A person_association queue skip may leave some observations unassociated; preserve "
         "every successful association present in the saved evidence. Do not infer zero people "
         "or a negative event from a skip. A person_association skip does not erase saved "
         "independent VLM observations or person detections. "
         "Historical entries with time_known=false have unknown times; do not invent intervals. "
         "Answer from the remaining saved evidence despite these local gaps, or report "
         "insufficient_evidence if it cannot support the requested answer."},
        {**messages[1], "content": json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False)},
    ]
