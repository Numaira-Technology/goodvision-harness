"""Export explicitly approved partial archives without claiming a completed capture."""
from __future__ import annotations

import json
from pathlib import Path

from .configuration import settings_record
from .evidence import EvidenceCapture
from .privacy import reject_private
from .recording import publish_once

APPROVAL_SCHEMA = "partial-saved-observation-approval-v1"
POLICY = "answer_saved_partial_observations_without_recapture"


def approved_source(approval, video_id):
    if (approval.get("schema_version") != APPROVAL_SCHEMA
            or approval.get("approved_by") != "human_user"
            or approval.get("decision") != POLICY):
        raise ValueError("Explicit scoped human approval is required for partial observations")
    matches = [row for row in approval["sources"] if row["video_id"] == video_id]
    if len(matches) != 1 or not matches[0]["question_ids"]:
        raise ValueError("Partial-observation approval must identify exactly one source")
    return matches[0]


def partial_answer_eligible(evidence, qa_id, approval):
    """Authorization changes answer eligibility, never source execution history."""
    recovery = evidence.get("partial_recovery")
    if approval is None or recovery is None:
        return False
    selected = approved_source(approval, evidence["video_id"])
    if (qa_id not in selected["question_ids"]
            or recovery.get("policy") != POLICY
            or recovery.get("record_directory") != selected["record_directory"]
            or recovery.get("capture_complete") is not False
            or evidence.get("observation_manifest", {}).get("complete") is not False):
        raise ValueError("Partial evidence is outside the explicitly approved source/question scope")
    return bool(evidence.get("frames") or evidence.get("independent_events")
                or evidence.get("fused_episodes"))


def export_partial_recording(directory, *, output, video_id, approval):
    """Use existing adapters on archived callbacks; never call a model or recorder finish."""
    selected = approved_source(approval, video_id)
    root, output = Path(directory).resolve(), Path(output).resolve()
    if root != Path(selected["record_directory"]).resolve() or output.is_relative_to(root):
        raise ValueError("Partial export must read the approved archive and write elsewhere")
    manifest = json.loads((root / "manifest.json").read_bytes())
    provenance = manifest["provenance"]
    if (manifest.get("schema_version") != "native-stream-recording-v1"
            or manifest.get("video_id") != video_id or manifest.get("face_enabled") is not False
            or provenance.get("question_conditioned") is not False
            or provenance.get("scheduling_mode") != "native_streaming"
            or provenance.get("tracking_model") != "yolo26-s"
            or provenance.get("vlm_enabled") is not True):
        raise ValueError("Only the declared face-disabled native streaming archives are supported")
    records = []
    for path in sorted((root / "records").glob("*.json")):
        row = json.loads(path.read_bytes())
        if row["sequence"] != len(records) + 1:
            raise ValueError("Archived record order is incomplete or ambiguous")
        records.append((path, row))
    frame_plan = [settings_record(row["payload"]) for _, row in records if row["kind"] == "source"]
    source_frames = {row["frame"]: row["t"] for row in frame_plan}
    windows, report_windows, failed_windows = [], {}, []
    for _, row in records:
        if row["kind"] != "independent":
            continue
        report = row["payload"]
        stamps = report["evidence"]
        if not stamps or any(source_frames.get(stamp["frame_sequence"]) != stamp["timestamp_seconds"]
                             for stamp in stamps):
            raise ValueError("Published partial report is not linked to archived input frames")
        key = (report["request_id"], report.get("vlm_lane", "combined"))
        if key in report_windows:
            raise ValueError("Duplicate published terminal report in partial archive")
        window = {"window_id": f"native-{row['sequence']:08d}",
            "start": min(stamp["timestamp_seconds"] for stamp in stamps),
            "end": max(stamp["timestamp_seconds"] for stamp in stamps),
            "event_labels": provenance["event_profile"]["labels"],
            "report_lanes": [key[1]], "request_ids": {key[1]: key[0]}}
        windows.append(window)
        report_windows[key] = window["window_id"]
        if report.get("state") != "completed" or report.get("error_code"):
            failed_windows.append({"lane": key[1], "start": window["start"], "end": window["end"]})
    capture = EvidenceCapture(video_id=video_id, stream_session_id=manifest["stream_session_id"],
        source_id=manifest["source_id"], source=settings_record(manifest["source"]),
        provenance=settings_record(provenance), frame_plan=frame_plan, window_plan=windows)
    for path, row in records:
        kind, payload = row["kind"], row["payload"]
        if kind == "tracking":
            capture.tracking_frame(**{key: value for key, value in payload.items()
                                      if key not in {"raw_response", "jpeg_sha256"}}, raw_reference=str(path))
        elif kind == "independent":
            key = (payload["request_id"], payload.get("vlm_lane", "combined"))
            capture.independent_report(payload, window_id=report_windows[key],
                raw_reference=str(path), publication_cursor=row["sequence"])
        elif kind == "scene":
            capture.scene_snapshot(payload["snapshot"], association_links=payload["association_links"],
                                   raw_reference=str(path))
        elif kind == "failure":
            capture.request_failure(stage=payload["stage"], reason=payload["reason"],
                                    reference=str(path), detail=payload.get("detail"))
    # Do not call seal/finish with invented drain flags or scheduler counters.
    evidence = capture.snapshot()
    source_ends = [row["payload"] for _, row in records if row["kind"] == "source-end"]
    eof = True if any(row.get("reason") == "eof" for row in source_ends) else None
    processed = {row["frame"] for row in evidence["frames"]}
    recovery = {"policy": POLICY, "record_directory": str(root), "record_count": len(records),
        "capture_complete": False, "source_eof_recorded": eof, "normal_queue_drain_recorded": False,
        "first_saved_input_time": frame_plan[0]["t"] if frame_plan else None,
        "last_saved_input_time": frame_plan[-1]["t"] if frame_plan else None,
        "saved_input_count": len(frame_plan), "saved_person_result_count": len(processed),
        "person_result_unavailable_times": [row["t"] for row in frame_plan if row["frame"] not in processed],
        "failed_published_windows": failed_windows,
        "unpublished_window_times": None,
        "unobserved_tail_after_last_input": eof is not True,
        "tail_end_original_metadata_seconds": manifest["source"].get("duration_seconds"),
        "note": "Only saved results are available. Queue drain and unpublished windows are unknown. "
                "Missing observations are not negative events or zero counts. Metadata duration is not corrected."}
    evidence.update(source_eof=eof, tracking_drained=None, events_drained=None, fusion_drained=None,
                    partial_recovery=recovery)
    evidence["observation_manifest"].update(complete=False, event_windows_complete=False,
        person_frames_accounted=False)
    reject_private(evidence, observation=True)
    publish_once(output, evidence)
    return evidence
