"""按题目独立会话，从原生档案做确定性答案提取；不读取私有答案。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "extraction"))
from bench.extract import answer_question
from bench.schema import Capture, EVENT_TARGET_TO_HARNESS_LABEL, Question

event_spec = importlib.util.spec_from_file_location(
    "frozen_event_extract", ROOT / "extraction/event-extract-snapshot.py")
events = importlib.util.module_from_spec(event_spec)
event_spec.loader.exec_module(events)

METADATA_FILES = (
    ROOT / "cohort/core116-media-metadata.jsonl",
    ROOT / "cohort/supplement-media-metadata.json",
)


def session_features(question: dict) -> list[str]:
    """沿用冻结入口的公开标签选择；几何会话由服务层关闭 VLM。"""
    capability = question["capability"]
    if capability in ("counting", "location_tracking"):
        return ["fight", "smoking", "fall", "weapon"]
    if capability == "temporal_questions":
        if question["question_form"] in {
            "first_exact_visible_count", "last_exact_visible_count", "longest_at_or_above_threshold"
        }:
            return ["fight", "smoking", "fall", "weapon"]
        target = question["public_context"]["event_label"]
        return [EVENT_TARGET_TO_HARNESS_LABEL.get(target, target)]
    if capability == "event_recognition":
        if question.get("question_form") == "event_presence_binary":
            # 冻结 core116 路径：公开 physical_conflict 对应原生 fight，
            # yes/no 是回答选项，不能当作窗口事件标签。
            target = question.get("public_context", {}).get("queried_event_label") or question.get("target_label")
            return [EVENT_TARGET_TO_HARNESS_LABEL.get(target, target)]
        return events.session_features(question)
    raise RuntimeError("GLOBAL_SCOPE_MISMATCH: unknown capability")


def _metadata(video_id: str, session: dict, eof: dict) -> dict:
    metadata = {}
    for path in METADATA_FILES:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("video_id") == video_id:
                        metadata.update({key: row.get(key) for key in ("width", "height", "fps", "duration_seconds")})
                        metadata["metadata_source"] = str(path)
                        break
        else:
            row = json.loads(path.read_text(encoding="utf-8")).get(video_id)
            if row:
                metadata.update({key: row.get(key) for key in ("width", "height", "fps", "duration_seconds")})
                metadata["metadata_source"] = str(path)
    # 仅保留公开媒体属性；EOF 自带的帧率和时长是当前输入的直接事实。
    for key in ("width", "height", "fps", "duration_seconds"):
        if session.get("media_metadata", {}).get(key) is not None:
            metadata[key] = session["media_metadata"][key]
    if metadata.get("fps") is None:
        metadata["fps"] = eof.get("file_fps") or 0.0
    if metadata.get("duration_seconds") is None:
        metadata["duration_seconds"] = eof.get("last_timestamp_seconds")
    return metadata


def _base(question: dict, session: dict | None) -> dict:
    return {
        "qa_id": question["qa_id"], "video_id": question["video_id"],
        "detector_model": session.get("detector_model") if session else None,
        "method": "ashley_native_siyu_main_rules_detector_ablation",
        "prediction": "", "status": "pending", "final_llm": False,
        "observation_reads_questions": False,
        "extractor_source": "frozen_ashley_native_bench_and_event_snapshots",
        "extractor_unchanged": True,
    }


def _path(output_root: Path, session: dict) -> Path:
    capture_path = Path(session["capture_path"])
    return capture_path if capture_path.is_absolute() else output_root / capture_path


def _complete_journal(path: Path) -> list[dict]:
    """读取已提交的整行；并发 append 尚未写完的尾部留待下一轮。"""
    if not path.exists():
        return []
    content = path.read_bytes()
    if b"\n" not in content:
        return []
    committed = content[:content.rfind(b"\n") + 1]
    rows = []
    for number, line in enumerate(committed.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"GLOBAL_AUDIT_CORRUPT_COMPLETE_JSONL: {path}:{number}") from error
    return rows


# 冻结事件提取算法不变；其输入读取器与外层使用同一套 append 整行边界。
# final_scene 已写入后其他累计审计文件仍可能恰在追加半行。
events.jsonl = _complete_journal


def _journal_complete(journal: list[dict], sid: str, geometry_only: bool) -> tuple[bool, str, dict, dict]:
    if not journal:
        return False, "native_journal_empty", {}, {}
    if any(row.get("stream_session_id") not in (None, sid) for row in journal):
        return False, "native_stream_session_id_mismatch", {}, {}
    kinds = {row.get("kind") for row in journal}
    if "source_error" in kinds:
        return False, "source_error", {}, {}
    eof = next((row for row in reversed(journal) if row.get("kind") == "source_eof"), {})
    tracking_row = next((row for row in reversed(journal) if row.get("kind") == "tracking_finished"), {})
    tracking = tracking_row.get("stats") or {}
    if not eof or not tracking_row:
        return False, "source_eof_or_tracking_finished_missing", eof, tracking
    if tracking_row.get("stream_session_id") not in (None, sid):
        return False, "tracking_session_id_mismatch", eof, tracking
    if tracking.get("stream_session_id") not in (None, sid):
        return False, "tracking_stats_session_id_mismatch", eof, tracking
    eof_index = max(index for index, row in enumerate(journal) if row.get("kind") == "source_eof")
    tracking_index = max(index for index, row in enumerate(journal) if row.get("kind") == "tracking_finished")
    if tracking_index <= eof_index:
        return False, "tracking_finished_before_source_eof", eof, tracking
    frames = [row["frame"] for row in journal if row.get("kind") == "frame"]
    if not frames or len({frame["frame"] for frame in frames}) != len(frames):
        return False, "native_frames_missing_or_duplicate", eof, tracking
    if (tracking.get("processed_frames") != len(frames)
            or tracking.get("session_admitted_frames") != len(frames)
            or tracking.get("session_source_frames") != eof.get("decoded_frames")):
        return False, "native_tracking_frame_totals_mismatch", eof, tracking
    if (tracking.get("last_error") or tracking.get("recovery_required")
            or tracking.get("pending_frame_count", 0) or tracking.get("pending_frames", 0)
            or tracking.get("in_flight") or tracking.get("pending")
            or any(tracking.get(key, 0) for key in
                   ("rejected_order", "rejected_session", "session_dropped_backpressure",
                    "rejected_capacity"))):
        return False, "native_tracking_not_drained_or_rejected", eof, tracking
    if geometry_only:
        marker = next((row for row in reversed(journal) if row.get("kind") == "geometry_finalized"), {})
        if not marker or marker.get("stream_session_id") != sid:
            return False, "geometry_finalized_missing_or_wrong_session", eof, tracking
        marker_eof = marker.get("source_eof")
        if marker_eof is not True and not (isinstance(marker_eof, dict) and marker_eof.get("reason") == "eof"):
            return False, "geometry_source_eof_not_confirmed", eof, tracking
        if marker.get("tracking_drained") is not True or marker.get("scene_inference") != "disabled":
            return False, "geometry_tracking_or_scene_disable_not_confirmed", eof, tracking
        final_index = max(index for index, row in enumerate(journal) if row.get("kind") == "geometry_finalized")
        if final_index <= tracking_index:
            return False, "geometry_finalized_before_tracking_finished", eof, tracking
    else:
        marker = next((row for row in reversed(journal) if row.get("kind") == "final_scene"), {})
        if not marker or marker.get("stream_session_id") != sid:
            return False, "native_final_scene_missing_or_wrong_session", eof, tracking
        final_index = max(index for index, row in enumerate(journal) if row.get("kind") == "final_scene")
        if final_index <= tracking_index:
            return False, "native_final_scene_before_tracking_finished", eof, tracking
    return True, "native_complete", eof, tracking


def extract_question(question: dict, output_root: str | Path,
                     session: dict | None, geometry_only: bool) -> dict:
    """返回 answered、真实完成空答或 pending；任何路径都只使用公开题。"""
    result = _base(question, session)
    result["geometry_only"] = bool(geometry_only)
    if question["capability"] == "event_recognition" and geometry_only:
        raise ValueError("event question cannot run geometry_only")
    if not session:
        result["reason"] = "native_session_not_recorded"
        return result
    if bool(session.get("geometry_only")) != bool(geometry_only):
        result["reason"] = "session_geometry_mode_mismatch"
        return result
    sid = session.get("stream_session_id")
    if not sid:
        result["reason"] = "stream_session_id_missing"
        return result
    path = _path(Path(output_root), session)
    result.update(stream_session_id=sid, journal_path=str(path), capture_path=str(path))
    journal = _complete_journal(path)
    complete, reason, eof, tracking = _journal_complete(journal, sid, geometry_only)
    if not complete:
        result["reason"] = reason
        return result

    if (question["capability"] == "event_recognition"
            and question.get("question_form") != "event_presence_binary"):
        event_result = events.extract_question(question, output_root, session)
        result.update(event_result)
        result["method"] = "ashley_native_siyu_main_rules_detector_ablation"
        result["detector_model"] = session.get("detector_model")
        result["geometry_only"] = False
        result["journal_path"] = str(path)
        return result

    data = None if geometry_only else events.read_session(Path(output_root), session)
    if data is not None and not data["complete"]:
        result["reason"] = "native_event_session_incomplete"
        return result
    frames = sorted((row["frame"] for row in journal if row.get("kind") == "frame"),
                    key=lambda frame: frame["frame"])
    if any(right["t"] <= left["t"] for left, right in zip(frames, frames[1:])):
        result.update(status="extract_error_blank", reason="native_frame_time_not_strictly_increasing")
        return result
    capture = {
        **_metadata(question["video_id"], session, eof),
        "video_id": question["video_id"], "frames": frames,
        "events": [] if geometry_only else data["events"],
        "input_mode": "offline_file", "timestamp_basis": eof.get("timestamp_basis"),
        "stream_session_id": sid, "source_eof": eof, "tracking_finished": tracking,
        "scene_inference": "disabled" if geometry_only else "enabled",
        "capture_method": "canonical_commit_and_scene_update_hooks",
    }
    result["capture_summary"] = {
        "frames": len(frames), "events": len(capture["events"]),
        "metadata_source": capture.get("metadata_source"),
        "source_eof": eof, "tracking_finished": tracking,
    }
    if question["capability"] == "counting" and question.get("target_label") == "person_head":
        result["known_method_limitation"] = "原生算法计人物框峰值；公开题目要求头部数。"
    elif question["capability"] == "temporal_questions" and not geometry_only:
        result["known_method_limitation"] = "原生算法返回第一个确认事件绝对结束秒；不修正为时长。"
    try:
        prediction = answer_question(Question.from_json(question), Capture.from_json(capture))
        result["prediction"] = prediction
        result["status"] = "answered" if prediction != "" else "extract_unknown_or_ambiguous"
    except (ValueError, KeyError, TypeError, IndexError) as error:
        result.update(status="extract_error_blank", reason="deterministic_extractor_error",
                      extraction_error=str(error), error_type=type(error).__name__)
    return result
