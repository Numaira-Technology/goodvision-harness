"""统一答案记录的首个实现：人物路计数；不读取GT、不调用最终LLM。

其他能力在证据适配完成前显式unsupported，不回退到旧Ashley或猜答案。
"""
from __future__ import annotations

import math
from typing import Any

PARSER_VERSION = "unified-evidence-count-v1"
PEAK_FORMS = frozenset({
    "vscrowd_full_sequence_peak_annotated_heads",
    "full_sequence_peak_visible_pedestrians",
})


def answer_question(question: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "qa_id": question["qa_id"], "final_answer": None,
        "status": "unsupported", "evidence_refs": [],
        "parser_version": PARSER_VERSION,
    }
    if evidence.get("video_id") != question.get("video_id"):
        raise ValueError("evidence video identity mismatch")
    if evidence.get("request_failure"):
        return {**result, "status": "request_failure"}
    if not evidence.get("source_eof") or not evidence.get("tracking_drained"):
        return {**result, "status": "incomplete_video"}
    if question.get("capability") != "counting" or question.get("question_form") not in PEAK_FORMS:
        return result
    frames = evidence.get("frames", [])
    if not frames:
        return {**result, "status": "insufficient_evidence", "reason": "no_observation_frames"}
    counts = []
    previous = (-1, -1.0)
    for frame in frames:
        index, timestamp = frame["frame"], frame["t"]
        if (type(index) is not int or index <= previous[0] or
                isinstance(timestamp, bool) or not math.isfinite(timestamp) or
                timestamp < 0 or timestamp <= previous[1]):
            raise ValueError("non-monotonic evidence frames")
        previous = index, timestamp
        ids = set()
        for subject in frame["subjects"]:
            if subject.get("observation_state") == "predicted":
                continue
            if subject.get("observation_state") != "observed":
                raise ValueError("explicit observation state required")
            if subject.get("label") != "person":
                continue
            sid = subject["id"]
            if not isinstance(sid, str) or not sid or sid in ids:
                raise ValueError("invalid or duplicate anonymous track ID")
            box = subject["bbox"]
            x1, y1, x2, y2 = (box[key] for key in ("x1", "y1", "x2", "y2"))
            if not (all(math.isfinite(v) for v in (x1, y1, x2, y2))
                    and 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("normalized observed bbox required")
            ids.add(sid)
        counts.append((len(ids), index, timestamp))
    peak = max(row[0] for row in counts)
    refs = [{"file": "capture.json", "frame": index, "timestamp_seconds": timestamp}
            for count, index, timestamp in counts if count == peak]
    return {**result, "status": "answered", "final_answer": str(peak),
            "evidence_refs": refs, "operator": "peak_observed_person_count",
            "observation_target": "person", "question_target": question.get("target_label"),
            "target_semantics_mismatch": question.get("target_label") == "person_head"}
