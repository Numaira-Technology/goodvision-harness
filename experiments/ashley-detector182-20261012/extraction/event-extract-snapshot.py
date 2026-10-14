"""只读取公开题目及本次原生档案，确定性转换选项；不读取GT或调用模型。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCENE_LABELS = {"vehicle_stops", "vehicle_turns_right", "vehicle_makes_u_turn",
                "vehicle_drops_off_person", "vehicle_turns_left", "vehicle_starts", "vehicle_reverses"}
NONE = "none of these events or actions"


def canonical(label):
    return "_".join(str(label).lower().split())


def public_labels(question):
    context = question.get("public_context", {})
    if context.get("response_format") == "event_presence_combination":
        return list(context["event_labels"])
    return [label for label in question["response_options"] if label != NONE]


def session_features(question):
    """主控制器只把这组公开标签送入会话，不能把question/options送入窗口。"""
    return list(dict.fromkeys(canonical(label) for label in public_labels(question)))


def jsonl(path):
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ref(path, line, row=None):
    value = {"path": str(path), "line": line}
    if row:
        for key in ("request_id", "start_seconds", "end_seconds", "timestamp_seconds", "evidence_id"):
            if key in row:
                value[key] = row[key]
    return value


def read_session(root, session):
    sid = session["stream_session_id"]
    capture_path = Path(session["capture_path"])
    if not capture_path.is_absolute():
        capture_path = root / capture_path
    journal = jsonl(capture_path)
    if any(row.get("stream_session_id") not in (None, sid) for row in journal):
        raise ValueError("原生capture包含其他会话，禁止混用")
    kinds = {row.get("kind") for row in journal}
    complete = {"source_eof", "tracking_finished", "final_scene"} <= kinds
    if "source_error" in kinds:
        complete = False
    tracking = next((row.get("stats", {}) for row in reversed(journal)
                     if row.get("kind") == "tracking_finished"), {})
    eof = next((row for row in reversed(journal) if row.get("kind") == "source_eof"), {})
    frames = [row["frame"] for row in journal if row.get("kind") == "frame"]
    if (not frames or len({row["frame"] for row in frames}) != len(frames)
            or tracking.get("processed_frames") != len(frames)
            or tracking.get("session_admitted_frames") != len(frames)
            or tracking.get("session_source_frames") != eof.get("decoded_frames")
            or tracking.get("last_error") or tracking.get("recovery_required")
            or any(tracking.get(key, 0) for key in
                   ("rejected_order", "session_dropped_backpressure", "rejected_capacity"))):
        complete = False
    events = {}
    for index, row in enumerate(journal, 1):
        if row.get("kind") == "event":
            events[row["event_id"]] = {**row["event"], "reference": _ref(capture_path, index)}
    data = {"complete": complete, "events": list(events.values()), "capture_path": str(capture_path)}
    for name, key in (("native-window-starts.jsonl", "starts"), ("native-windows.jsonl", "windows"),
                      ("native-completions.jsonl", "completions"), ("native-evidence.jsonl", "evidence")):
        path = root / name
        data[key] = [{**row, "reference": _ref(path, index, row)}
                     for index, row in enumerate(jsonl(path), 1) if row.get("stream_session_id") == sid]
    return data


def event_state(label, data):
    events = [event for event in data["events"] if event.get("event_label") == label]
    confirmed = [event for event in events
                 if event.get("confirmed_since_timestamp_seconds", event.get("confirmed_since")) is not None
                 and event.get("phase") in {"confirmed", "ended"}]
    positive_refs = [event["reference"] for event in confirmed]
    if confirmed:
        return {"state": "present", "reason": "native_confirmed_event",
                "references": positive_refs, "events": confirmed}

    starts = {row["request_id"]: row for row in data["starts"]}
    windows = {row["request_id"]: row for row in data["windows"]}
    completions = {row["request_id"]: row for row in data["completions"]}
    evidence = {row["evidence_id"]: row for row in data["evidence"]}
    issues = []
    refs = []
    if not starts:
        issues.append("no_native_window")
    for request_id, started in starts.items():
        window, completion = windows.get(request_id), completions.get(request_id)
        if not window or window.get("status") != "completed":
            issues.append("window_failed_or_unfinished")
            refs.append(started["reference"])
            continue
        refs.append(window["reference"])
        if (label not in window.get("requested_event_labels", []) or not completion
                or not completion.get("completed") or not completion.get("success")):
            issues.append("window_not_accepted_or_label_not_requested")
            continue
        matches = [(index, row) for index, row in enumerate(window.get("verdicts", []))
                   if row.get("event_label") == label]
        if not matches:
            issues.append("label_omitted")
            continue
        allowed = set(window.get("allowed_subject_ids", []))
        covered = {sid for _, row in matches for sid in row.get("subject_ids", [])}
        if label not in SCENE_LABELS and (not allowed or not allowed <= covered):
            issues.append("person_coverage_unknown")
        if label in SCENE_LABELS and any(row.get("subject_ids") for _, row in matches):
            issues.append("invalid_scene_subject_binding")
        for index, verdict in matches:
            accepted = evidence.get(f"inspection:{request_id}:{index}")
            if not accepted or not accepted.get("accepted"):
                issues.append("verdict_not_accepted_by_native_scene")
            elif accepted.get("detected") is not False:
                issues.append("unconfirmed_positive" if accepted.get("detected") is True else "unknown_verdict")
            elif verdict.get("detected") is not False:
                issues.append("raw_and_accepted_verdict_disagree")
    if any(event.get("phase") == "suspected" for event in events):
        issues.append("native_event_unconfirmed")
    if not data["complete"]:
        issues.append("native_session_incomplete")
    if issues:
        return {"state": "unknown", "reason": sorted(set(issues)), "references": refs, "events": events}
    return {"state": "absent", "reason": "all_completed_native_windows_explicitly_negative",
            "references": refs, "events": events,
            "coverage_scope": "仅原生抽样窗口及已跟踪人物；不证明未采样时间或未跟踪人物不存在事件"}


def _combination(option, count):
    tokens = [token.strip() for token in option.split(";")]
    if len(tokens) != count:
        raise ValueError("公开组合选项项数不一致")
    states = []
    for index, token in enumerate(tokens):
        match = re.fullmatch(r"([A-Z]):\s*(present|absent)", token)
        if not match or match.group(1) != chr(ord("A") + index):
            raise ValueError("不支持的公开组合选项格式")
        states.append(match.group(2))
    return states


def extract_question(question, output_root, session):
    root = Path(output_root)
    result = {"qa_id": question["qa_id"], "video_id": question["video_id"],
              "method": "ashley_native_siyu_event_extension", "prediction": "",
              "status": "pending", "public_response_options": question["response_options"],
              "final_llm": False, "observation_reads_questions": False}
    if not session:
        result["reason"] = "native_session_not_recorded"
        return result
    data = read_session(root, session)
    result.update(stream_session_id=session["stream_session_id"], capture_path=data["capture_path"])
    if not data["complete"]:
        result["reason"] = "native_eof_tracking_and_final_scene_not_complete"
        return result
    labels = public_labels(question)
    result["event_states"] = {label: event_state(canonical(label), data) for label in labels}
    states = [result["event_states"][label]["state"] for label in labels]
    options = question["response_options"]
    context = question.get("public_context", {})
    selected = []
    if context.get("response_format") == "event_presence_combination":
        if "unknown" not in states:
            selected = [option for option in options if _combination(option, len(labels)) == states]
        reason = "combination_requires_all_event_states"
    else:
        # 某个候选已获原生确认即为可选证据；其他 unknown 保留，不改成 absent。
        # 两个已确认候选（或另一候选已有未确认阳性）不擅自决定哪一个更显著。
        present = [label for label, state in zip(labels, states) if state == "present"]
        unresolved_positive = [label for label in labels
                              if "unconfirmed_positive" in result["event_states"][label].get("reason", [])
                              or "native_event_unconfirmed" in result["event_states"][label].get("reason", [])]
        if len(present) == 1 and not unresolved_positive:
            selected = present
        elif not present and all(state == "absent" for state in states) and NONE in options:
            selected = [NONE]
        reason = "single_choice_multiple_positive_or_insufficient_evidence"
    if len(selected) == 1:
        result.update(status="answered", prediction=selected[0], reason="deterministic_exact_public_option")
    else:
        result.update(status="extract_unknown_or_ambiguous", reason=reason,
                      ambiguity={"matching_options": selected, "event_states": states})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--sessions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sessions = json.loads(args.sessions.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    question_dir = args.output_dir / "questions"
    question_dir.mkdir(exist_ok=True)
    rows = []
    for question in jsonl(args.questions):
        try:
            result = extract_question(question, args.captures, sessions.get(question["video_id"]))
        except (ValueError, KeyError, OSError, TypeError) as error:
            result = {"qa_id": question["qa_id"], "video_id": question["video_id"],
                      "status": "extract_error", "prediction": "", "error": str(error)}
        (question_dir / (question["qa_id"] + ".json")).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        rows.append(result)
    (args.output_dir / "outcomes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    # pending不生成答案行；完成但提取unknown保留空答，并以明确状态交给正式评分器。
    (args.output_dir / "predictions.jsonl").write_text("".join(
        json.dumps({"qa_id": row["qa_id"], "prediction": row["prediction"]}, ensure_ascii=False) + "\n"
        for row in rows if row["status"] != "pending"))
    print(json.dumps({"outcomes": len(rows), "answered": sum(row["status"] == "answered" for row in rows),
                      "pending": sum(row["status"] == "pending" for row in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
