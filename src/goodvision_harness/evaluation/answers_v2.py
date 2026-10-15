"""公开14问法的确定性证据解析；不读取GT、不调用模型、不改计数v1。"""
from __future__ import annotations

import math
import re
from typing import Any

from .vocab import SUPPORTED_EVENT_LABELS, canonical_event_label

PARSER_VERSION = "unified-evidence-v2"
EVIDENCE_SCHEMA = "unified-evidence-v2"
# 全局预注册；不能根据题号、答案或本视频采样分布重新估计。
TIME_NEIGHBOR_SECONDS = 0.51
CONTIGUOUS_GAP_SECONDS = 0.51
LOCATOR_MIN_IOU = 0.30
LOCATOR_AMBIGUITY_MARGIN = 0.10
SAME_POSITION_FRACTION = 0.03
EPSILON = 1e-6
REGIONS = ("左侧", "中央", "右侧")
FORM_OPERATORS = {
    "vscrowd_full_sequence_peak_annotated_heads": "peak_observed_person_count",
    "full_sequence_peak_visible_pedestrians": "peak_observed_person_count",
    "bbox_locator_horizontal_displacement_track_id_withheld": "located_subject_displacement",
    "bbox_locator_vertical_displacement_track_id_withheld": "located_subject_displacement",
    "full_sequence_cumulative_region": "cumulative_observed_person_region",
    "interval_cumulative_region": "cumulative_observed_person_region",
    "event_presence_binary": "published_independent_event_presence",
    "event_single_action_selection": "exact_single_event_option",
    "event_presence_combination": "exact_event_presence_combination",
    "fixed_conflict_first_segment_end_timestamp": "first_confirmed_episode_end",
    "source_native_interval_elapsed_seconds": "associated_episode_elapsed_seconds",
    "first_exact_visible_count": "first_observed_exact_count_timestamp",
    "last_exact_visible_count": "last_observed_exact_count_timestamp",
    "longest_at_or_above_threshold": "longest_observed_threshold_span",
}
EVENT_FORMS = frozenset({
    "event_presence_binary", "event_single_action_selection", "event_presence_combination",
    "fixed_conflict_first_segment_end_timestamp", "source_native_interval_elapsed_seconds",
})
SCENE_PRESENCE_FORMS = frozenset({
    "event_presence_binary", "event_single_action_selection", "event_presence_combination",
})
FAILURE_STAGE_PATHS = {
    "tracking": "tracking", "person": "tracking", "detector": "tracking",
    "independent_vlm": "events", "events": "events", "vlm": "events",
    "fusion": "fusion",
}
VLM_LANES = frozenset({"information", "grounding", "combined"})
INFORMATION_LANES = frozenset({"information", "combined"})
GROUNDING_LANES = frozenset({"grounding", "combined"})


class EvidenceInsufficient(ValueError):
    """合法但不足的证据；转换为空答，不制造默认阴性。"""


class SemanticUnsupported(ValueError):
    """本版本不支持或公开字段冲突且无法明确裁定的语义。"""


def _number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return float(value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be nonempty text")
    return value


def _box(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        values = [value[k] for k in ("x1", "y1", "x2", "y2")]
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        values = value
    else:
        raise ValueError("bbox must be normalized xyxy")
    coords = tuple(_number(v, "bbox coordinate") for v in values)
    if not (0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1):
        raise ValueError("bbox must have positive area within normalized canvas")
    return coords


def _iou(a: tuple, b: tuple) -> float:
    area = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union


def _frame_ref(frame: dict, subject_id: str | None = None) -> dict:
    ref = {"file": "capture.json", "frame": frame["frame"], "timestamp_seconds": frame["t"]}
    if subject_id is not None:
        ref["subject_id"] = subject_id
    return ref


def _event_ref(event: dict) -> dict:
    return {"file": "capture.json", "evidence_id": event["evidence_id"],
            "window_id": event["window_id"], "vlm_lane": event.get("vlm_lane", "combined")}


def _episode_ref(episode: dict) -> dict:
    return {"file": "capture.json", "episode_id": episode["episode_id"],
            "evidence_ids": list(episode["evidence_ids"]),
            "start_seconds": episode["start"], "end_seconds": episode.get("end")}


def _frames(evidence: dict) -> list[dict]:
    frames = evidence.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("frames must be a list")
    previous = (-1, -1.0)
    for frame in frames:
        index = frame["frame"]
        timestamp = _number(frame["t"], "frame timestamp")
        if type(index) is not int or index <= previous[0] or timestamp <= previous[1]:
            raise ValueError("non-monotonic evidence frames")
        previous = index, timestamp
        seen = set()
        for subject in frame["subjects"]:
            sid = _text(subject["id"], "subject id")
            if sid in seen:
                raise ValueError("duplicate subject id in frame")
            seen.add(sid)
            _text(subject["label"], "subject label")
            if subject.get("observation_state") not in ("observed", "predicted"):
                raise ValueError("explicit observed/predicted state required")
            _box(subject["bbox"])
    if not frames:
        raise EvidenceInsufficient("no_observation_frames")
    return frames


def _frame_coverage(evidence: dict, frames: list[dict]) -> None:
    manifest = evidence.get("observation_manifest", {})
    plan = manifest.get("frame_plan")
    if manifest.get("person_frames_complete") is not True or not isinstance(plan, list) or not plan:
        raise EvidenceInsufficient("missing_complete_frame_manifest")
    expected = {}
    for row in plan:
        index, timestamp = row["frame"], _number(row["t"], "planned timestamp")
        if type(index) is not int or index < 0 or index in expected:
            raise ValueError("invalid or duplicate planned source frame")
        expected[index] = timestamp
    actual = {f["frame"]: f["t"] for f in frames}
    if actual.keys() != expected.keys() or any(abs(actual[k] - t) > EPSILON for k, t in expected.items()):
        raise EvidenceInsufficient("frame_plan_not_fully_observed")


def _observed(frame: dict, label: str = "person") -> list[dict]:
    return [s for s in frame["subjects"]
            if s["observation_state"] == "observed" and s["label"] == label]


def _third(subject: dict) -> int:
    box = _box(subject["bbox"])
    return min(2, int(((box[0] + box[2]) / 2) * 3))


def _near_frame(frames: list[dict], timestamp: float) -> dict:
    # 在所有源帧上找相邻时刻，而非先丢掉没有所需人物的帧再寻找远帧。
    distances = sorted((abs(f["t"] - timestamp), f["frame"], f) for f in frames)
    if not distances or distances[0][0] > TIME_NEIGHBOR_SECONDS + EPSILON:
        raise EvidenceInsufficient("no_frame_within_registered_time_tolerance")
    if len(distances) > 1 and abs(distances[0][0] - distances[1][0]) <= EPSILON:
        raise EvidenceInsufficient("ambiguous_equally_near_source_frames")
    return distances[0][2]


def _source_box(evidence: dict, frame: dict, pixels: list) -> tuple:
    source = frame.get("source", evidence.get("source", {}))
    width, height = source.get("width"), source.get("height")
    if width is None or height is None:
        raise EvidenceInsufficient("source_dimensions_required_for_pixel_locator")
    width, height = _number(width, "source width"), _number(height, "source height")
    if width <= 0 or height <= 0 or len(pixels) != 4:
        raise ValueError("source size and pixel xyxy must be valid")
    return _box([_number(v, "source pixel") / (width if i % 2 == 0 else height)
                 for i, v in enumerate(pixels)])


def _locate(evidence: dict, frames: list[dict], locator: dict) -> tuple[list[str], list[dict]]:
    if locator.get("bbox_coordinate_system") != "source_absolute_pixel_xyxy":
        raise SemanticUnsupported("locator_coordinate_system_not_supported")
    frame = _near_frame(frames, _number(locator["timestamp_sec"], "locator timestamp"))
    ids, refs = [], []
    for ref in locator["boxes"]:
        label = ref.get("object_label")
        if label not in ("person", "group", "vehicle"):
            raise SemanticUnsupported("locator_object_label_not_supported")
        reference = _source_box(evidence, frame, ref["bbox_xyxy"])
        matches = sorted(((_iou(reference, _box(s["bbox"])), s["id"], s)
                          for s in _observed(frame, label)), reverse=True)
        if not matches or matches[0][0] < LOCATOR_MIN_IOU:
            raise EvidenceInsufficient("no_observed_subject_matching_locator")
        if len(matches) > 1 and matches[0][0] - matches[1][0] < LOCATOR_AMBIGUITY_MARGIN:
            raise EvidenceInsufficient("ambiguous_subject_locator")
        sid = matches[0][1]
        if sid in ids:
            raise EvidenceInsufficient("multiple_locators_resolve_to_same_subject")
        ids.append(sid)
        refs.append(_frame_ref(frame, sid))
    if not ids:
        raise EvidenceInsufficient("empty_subject_locator")
    return ids, refs


def _at_subject_time(frames: list[dict], sid: str, timestamp: float) -> tuple[dict, dict]:
    frame = _near_frame(frames, timestamp)
    matches = [s for s in frame["subjects"]
               if s["id"] == sid and s["observation_state"] == "observed"]
    if len(matches) != 1:
        raise EvidenceInsufficient("located_subject_not_observed_at_required_time")
    return frame, matches[0]


def _duration(evidence: dict) -> float:
    value = evidence.get("source", {}).get("duration_seconds")
    if value is None:
        raise EvidenceInsufficient("source_duration_required")
    return _number(value, "source duration")


def _negative_video_end(evidence: dict) -> float:
    """阴性覆盖真实末解码帧，不把帧显示时长的结束点伪造成观察帧。

    新采集必须实际纳入首尾帧，并成功处理全部固定采样窗口。该结论仍是
    固定采样观察的阴性，不声称覆盖了采样帧之间未观看的图像。
    """
    source = evidence.get("source", {})
    if source.get("negative_coverage_domain") != "first_to_last_decoded_frame_pts":
        return _duration(evidence)  # 旧证据保持旧判定，不追改历史分数。
    terminal = source.get("terminal_frame")
    plan = evidence.get("observation_manifest", {}).get("frame_plan", [])
    if (not isinstance(terminal, dict) or not plan or evidence.get("source_eof") is not True
            or abs(plan[0]["t"]) > EPSILON
            or plan[-1]["frame"] != terminal.get("frame")
            or plan[-1]["jpeg_sha256"] != terminal.get("jpeg_sha256")
            or abs(plan[-1]["t"] - _number(terminal.get("t"), "terminal frame time")) > EPSILON):
        raise EvidenceInsufficient("real_terminal_frame_not_observed")
    return _number(terminal["t"], "terminal frame time")


def _window_plan(evidence: dict, *, require_complete: bool = True) -> list[dict]:
    manifest = evidence.get("observation_manifest", {})
    plan = manifest.get("window_plan")
    if (require_complete and manifest.get("event_windows_complete") is not True) or not isinstance(plan, list) or not plan:
        raise EvidenceInsufficient("missing_complete_event_window_manifest")
    seen = set()
    for window in plan:
        wid = _text(window["window_id"], "window id")
        if wid in seen:
            raise ValueError("duplicate planned window id")
        seen.add(wid)
        if _number(window["end"], "window end") < _number(window["start"], "window start"):
            raise ValueError("reversed event window")
        if not window.get("event_labels"):
            raise ValueError("planned window must declare observation vocabulary")
        lanes = window.get("report_lanes", ["combined"])
        if (not isinstance(lanes, list) or not lanes
                or any(not isinstance(v, str) or v not in VLM_LANES for v in lanes)
                or len(lanes) != len(set(lanes))):
            raise ValueError("planned report lanes must be unique information/grounding/combined")
    return plan


def _events(evidence: dict) -> list[dict]:
    events = evidence.get("independent_events", [])
    if not isinstance(events, list):
        raise ValueError("independent_events must be a list")
    seen = set()
    planned = {w["window_id"]: w for w in _window_plan(evidence, require_complete=False)}
    for event in events:
        eid = _text(event["evidence_id"], "evidence id")
        if eid in seen:
            raise ValueError("duplicate independent evidence id")
        seen.add(eid)
        _text(event["window_id"], "window id")
        canonical_event_label(event["event_label"])
        if type(event.get("detected")) not in (bool, type(None)):
            raise ValueError("event detected must be true/false/null")
        if event.get("scope") not in ("scene", "subject"):
            raise ValueError("event scope must be scene or subject")
        if event.get("publication_status") not in ("published", "pending", "failed"):
            raise ValueError("explicit independent publication status required")
        lane = event.get("vlm_lane", "combined")
        if not isinstance(lane, str) or lane not in VLM_LANES:
            raise ValueError("vlm_lane must be information/grounding/combined")
        if _number(event["end"], "event end") < _number(event["start"], "event start"):
            raise ValueError("reversed independent window")
        window = planned.get(event["window_id"])
        if (window is None or abs(event["start"] - window["start"]) > EPSILON
                or abs(event["end"] - window["end"]) > EPSILON
                or canonical_event_label(event["event_label"]) not in {
                    canonical_event_label(v) for v in window["event_labels"]}):
            raise ValueError("independent verdict must match planned window and vocabulary")
        if lane not in window.get("report_lanes", ["combined"]):
            raise ValueError("independent verdict lane not declared in planned window")
    return events


def _native_lane_complete(evidence: dict) -> bool:
    """information已发布不能代替每个固定窗口的grounding原生终态。"""
    events = _events(evidence)
    for window in _window_plan(evidence, require_complete=False):
        if not set(window.get("report_lanes", ["combined"])) & GROUNDING_LANES:
            return False
        for label in window["event_labels"]:
            rows = [e for e in events if e["window_id"] == window["window_id"]
                    and e.get("vlm_lane", "combined") in GROUNDING_LANES
                    and canonical_event_label(e["event_label"]) == canonical_event_label(label)]
            if not rows or any(e["publication_status"] != "published" for e in rows):
                return False
    return True


def _covers(intervals: list[tuple[float, float]], start: float, end: float) -> bool:
    cursor = start
    for left, right in sorted(intervals):
        if right < cursor - EPSILON:
            continue
        if left > cursor + EPSILON:
            return False
        cursor = max(cursor, right)
        if cursor >= end - EPSILON:
            return True
    return False


def _presence(evidence: dict, label: str) -> tuple[bool | None, list[dict]]:
    canonical = canonical_event_label(label)
    if canonical not in SUPPORTED_EVENT_LABELS:
        raise SemanticUnsupported("event_label_not_in_fixed_vocabulary")
    plan = _window_plan(evidence)
    events = _events(evidence)
    relevant = [e for e in events if canonical_event_label(e["event_label"]) == canonical
                and e.get("vlm_lane", "combined") in INFORMATION_LANES]
    refs = [_event_ref(e) for e in relevant]
    planned = {w["window_id"]: w for w in plan}
    for event in relevant:
        window = planned.get(event["window_id"])
        if window is None or abs(event["start"] - window["start"]) > EPSILON or abs(event["end"] - window["end"]) > EPSILON:
            raise ValueError("independent verdict must match planned window")
        if canonical not in {canonical_event_label(v) for v in window["event_labels"]}:
            raise ValueError("independent verdict label not declared in planned vocabulary")
    # 场景存在性使用已发布独立正判定；不伪造native确认，也不要求人物关联成功。
    if any(e["detected"] is True and e["publication_status"] == "published" for e in relevant):
        return True, refs
    if any(e["detected"] is not False or e["publication_status"] != "published" for e in relevant):
        return None, refs
    negative_intervals = []
    for window in plan:
        if canonical not in {canonical_event_label(v) for v in window["event_labels"]}:
            continue
        rows = [e for e in relevant if e["window_id"] == window["window_id"]]
        # 主体阴性只否定该主体，不能推导全场景阴性。遗漏、unknown及未发布正判定均不为阴性。
        if rows and any(e["scope"] == "scene" for e in rows) and all(
                e["detected"] is False and e["publication_status"] == "published" for e in rows):
            negative_intervals.append((window["start"], window["end"]))
    if _covers(negative_intervals, 0.0, _negative_video_end(evidence)):
        return False, refs
    return None, refs


def _negative_coverage(evidence: dict, label: str, start: float, end: float,
                       subject_ids: list[str] | None = None) -> bool:
    """证明某时段显式阴性；unknown不能被一个重叠阴性窗口遮盖。"""
    if end <= start + EPSILON:
        return True
    canonical = canonical_event_label(label)
    plan = _window_plan(evidence, require_complete=False)
    events = [e for e in _events(evidence) if e.get("vlm_lane", "combined") in GROUNDING_LANES]
    intervals = []
    for window in plan:
        if window["end"] <= start + EPSILON or window["start"] >= end - EPSILON:
            continue
        if canonical not in {canonical_event_label(v) for v in window["event_labels"]}:
            continue
        rows = [e for e in events if e["window_id"] == window["window_id"]
                and canonical_event_label(e["event_label"]) == canonical]
        if subject_ids is not None:
            matched = [e for e in rows if e.get("association_status") == "matched"
                       and set(e.get("associated_subject_ids", [])) == set(subject_ids)]
            # 其他主体的场景positive不推翻本组已关联的explicit negative。
            rows = matched or [e for e in rows if e.get("scope") == "scene"]
        if not rows or not all(e["detected"] is False and e["publication_status"] == "published" for e in rows):
            return False
        if subject_ids is None and not any(e["scope"] == "scene" for e in rows):
            return False
        intervals.append((window["start"], window["end"]))
    return _covers(intervals, start, end)


def _verdict_ids(event: dict) -> set[str]:
    """场景聚合保留原始子观察ID，避免把同一已关联positive误当新事件。"""
    return set(event.get("raw_observation_ids") or [event["evidence_id"]])


def _positive_verdict_ids(event: dict, known: dict[str, dict]) -> set[str]:
    """聚合positive只由原始positive支持；unknown/缺引用仍保留供阻断。"""
    refs = _verdict_ids(event)
    if event.get("raw_observation_ids") and event["detected"] is True:
        contributors = {eid for eid in refs if eid not in known or known[eid]["detected"] is not False}
        return contributors or refs
    return refs


def _prior_episode_coverage(evidence: dict, label: str, episode: dict, *,
                            anchor: float = 0.0, subject_ids: list[str] | None = None) -> bool:
    """起点前的窗口须阴性，或其正判定已经明确属于同一个首段。"""
    end = episode["start"]
    if end <= anchor + EPSILON:
        return True
    canonical = canonical_event_label(label)
    events = [e for e in _events(evidence) if e.get("vlm_lane", "combined") in GROUNDING_LANES]
    intervals = []
    known = {e["evidence_id"]: e for e in events}
    refs = set(episode["evidence_ids"])
    for window in _window_plan(evidence, require_complete=False):
        if window["start"] >= end - EPSILON or window["end"] <= anchor + EPSILON:
            continue
        rows = [e for e in events if e["window_id"] == window["window_id"]
                and canonical_event_label(e["event_label"]) == canonical]
        if subject_ids is not None:
            matched = [e for e in rows if e.get("association_status") == "matched"
                       and set(e.get("associated_subject_ids", [])) == set(subject_ids)]
            unbound = [e for e in rows if e["scope"] == "subject"
                       and e.get("association_status") != "matched"]
            scenes = [e for e in rows if e["scope"] == "scene"]
            if matched:
                relevant_ids = {e["evidence_id"] for e in matched + unbound}
                # 有本主体阴性时不以其他已关联主体的聚合positive推翻它；
                # 无法定位的判定仍可能属于本主体，不能绕过。
                scenes = [e for e in scenes if not e.get("raw_observation_ids")
                          or bool(_positive_verdict_ids(e, known) & relevant_ids)]
            rows = matched + unbound + scenes
        if not rows or any(e["publication_status"] != "published" or e["detected"] is None for e in rows):
            return False
        positives = [e for e in rows if e["detected"] is True]
        if positives:
            if not all(_positive_verdict_ids(e, known) <= refs for e in positives):
                return False
            if not all(eid in known and known[eid]["detected"] is True
                       and known[eid]["publication_status"] == "published"
                       for event in positives for eid in _positive_verdict_ids(event, known)):
                return False
            if subject_ids is not None and not all(
                    eid in known and known[eid].get("association_status") == "matched"
                    and set(known[eid].get("associated_subject_ids", [])) == set(subject_ids)
                    for event in positives for eid in _positive_verdict_ids(event, known)):
                return False
        elif subject_ids is None and not any(e["scope"] == "scene" for e in rows):
            return False
        intervals.append((window["start"], window["end"]))
    return _covers(intervals, anchor, end)


def _episode_resolution_window(evidence: dict, label: str, episode: dict) -> dict:
    """披露首个支持窗口及原生终帧边界；不插值动作起点。"""
    canonical = canonical_event_label(label)
    refs = set(episode["evidence_ids"])
    events = _events(evidence)
    known = {e["evidence_id"]: e for e in events}
    positives = [e for e in events
                 if e["detected"] is True and e["publication_status"] == "published"
                 and e.get("vlm_lane", "combined") in GROUNDING_LANES
                 and canonical_event_label(e["event_label"]) == canonical
                 and _positive_verdict_ids(e, known) <= refs]
    if not positives:
        raise EvidenceInsufficient("episode_missing_supporting_positive_window")
    first = min(positives, key=lambda e: (e["start"], e["end"], e["window_id"]))
    return {"window_id": first["window_id"], "start_seconds": first["start"],
            "end_seconds": first["end"], "native_boundary_seconds": episode["start"]}


def _options(question: dict) -> list[str]:
    options = question.get("response_options")
    if not isinstance(options, list) or not options or any(not isinstance(v, str) for v in options):
        raise SemanticUnsupported("public_response_options_required")
    if len(set(options)) != len(options):
        raise SemanticUnsupported("duplicate_response_options")
    return options


def _choose(question: dict, answer: str) -> str:
    if answer not in _options(question):
        raise EvidenceInsufficient("derived_answer_not_in_public_options")
    return answer


def _region_spec(question: dict) -> int | None:
    params = question["public_context"].get("parameters", {})
    region = params.get("region")
    mentioned = [v for v in REGIONS if v in question.get("question", "")]
    if len(mentioned) > 1 or (mentioned and region is not None and region != mentioned[0]):
        raise SemanticUnsupported("conflicting_public_region_fields")
    if region is None and mentioned:
        region = mentioned[0]
    if region is None and params.get("scope") == "full":
        return None
    if region not in REGIONS:
        raise SemanticUnsupported("unknown_public_count_region")
    return REGIONS.index(region)


def _window_frames(question: dict, frames: list[dict]) -> list[dict]:
    ctx = question["public_context"]
    clip = ctx.get("clip", {})
    start, end = _number(clip["start_sec"], "clip start"), _number(clip["end_sec"], "clip end")
    if question["question_form"] == "interval_cumulative_region":
        params = ctx["parameters"]
        # 公开第k秒是[k-1,k)的序号，clip明确给源时间；不用第k秒字段覆盖源PTS。
        if abs(_number(params["start_second"], "start second") - 1 - start) > EPSILON or abs(
                _number(params["end_second"], "end second") - end) > EPSILON:
            raise SemanticUnsupported("conflicting_public_second_interval_and_clip")
    if end < start:
        raise SemanticUnsupported("reversed_public_clip")
    rows = [f for f in frames if start - EPSILON <= f["t"] <= end + EPSILON]
    if not rows or rows[0]["t"] - start > TIME_NEIGHBOR_SECONDS or end - rows[-1]["t"] > TIME_NEIGHBOR_SECONDS:
        raise EvidenceInsufficient("required_clip_endpoints_not_observed")
    return rows


def _count_temporal(question: dict, frames: list[dict]) -> tuple[str, list[dict], dict]:
    form = question["question_form"]
    pattern = r"恰好出现\s*(\d+)\s*名" if form != "longest_at_or_above_threshold" else r"达到或超过\s*(\d+)\s*名"
    matches = re.findall(pattern, question.get("question", ""))
    if len(matches) != 1:
        raise SemanticUnsupported("public_count_threshold_not_uniquely_parsed")
    threshold, region = int(matches[0]), _region_spec(question)
    rows = _window_frames(question, frames)
    counts = [(f, sum(1 for s in _observed(f) if region is None or _third(s) == region)) for f in rows]
    if form != "longest_at_or_above_threshold":
        hits = [f for f, count in counts if count == threshold]
        if not hits:
            raise EvidenceInsufficient("exact_count_not_observed")
        chosen = hits[0] if form == "first_exact_visible_count" else hits[-1]
        return str(int(round(chosen["t"]))), [_frame_ref(chosen)], {"unrounded_seconds": chosen["t"], "threshold": threshold}
    if any(b["t"] - a["t"] > CONTIGUOUS_GAP_SECONDS + EPSILON for a, b in zip(rows, rows[1:])):
        raise EvidenceInsufficient("threshold_duration_has_unobserved_temporal_gaps")
    best, best_rows, run = 0.0, [], []
    for frame, count in counts:
        if count >= threshold:
            if run and frame["t"] - run[-1]["t"] > CONTIGUOUS_GAP_SECONDS + EPSILON:
                run = []
            run.append(frame)
            span = run[-1]["t"] - run[0]["t"]
            if not best_rows or span > best:
                best, best_rows = span, list(run)
        else:
            run = []
    if not best_rows:
        # 完整采样中无达标帧可报告观察到的最长span=0，不推断采样间隔中的真实动作。
        return "0", [_frame_ref(f) for f in rows], {"unrounded_seconds": 0.0, "threshold": threshold}
    return str(int(round(best))), [_frame_ref(f) for f in best_rows], {"unrounded_seconds": best, "threshold": threshold}


def _episodes(evidence: dict, label: str, subject_ids: list[str] | None = None) -> list[dict]:
    canonical = canonical_event_label(label)
    if canonical not in SUPPORTED_EVENT_LABELS:
        raise SemanticUnsupported("event_label_not_in_fixed_vocabulary")
    events = _events(evidence)
    known = {e["evidence_id"]: e for e in events}
    result, seen = [], set()
    for episode in evidence.get("fused_episodes", []):
        eid = _text(episode["episode_id"], "episode id")
        if eid in seen:
            raise ValueError("duplicate episode id")
        seen.add(eid)
        if canonical_event_label(episode["event_label"]) != canonical:
            continue
        start = _number(episode["start"], "episode start")
        if episode.get("end") is not None and _number(episode["end"], "episode end") < start:
            raise ValueError("episode end precedes start")
        ids = episode.get("subject_ids", [])
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError("episode subject_ids must be unique")
        if subject_ids is not None and set(ids) != set(subject_ids):
            continue
        refs = episode.get("evidence_ids", [])
        if not refs or any(r not in known or canonical_event_label(known[r]["event_label"]) != canonical for r in refs):
            raise EvidenceInsufficient("episode_missing_matching_independent_evidence")
        if subject_ids is not None and episode.get("association_status") != "matched":
            continue
        if episode.get("confirmed") is not True:
            continue
        if not any(known[r]["detected"] is True and known[r]["publication_status"] == "published"
                   and known[r].get("vlm_lane", "combined") in GROUNDING_LANES for r in refs):
            raise EvidenceInsufficient("confirmed_episode_missing_published_positive_verdict")
        if start > _duration(evidence) + EPSILON or (episode.get("end") is not None
                and episode["end"] > _duration(evidence) + EPSILON):
            raise ValueError("episode boundary beyond source duration")
        result.append(episode)
    return sorted(result, key=lambda e: (e["start"], e["episode_id"]))


def _ended(evidence: dict, episode: dict) -> float:
    if episode.get("end") is None or episode.get("end_reason") not in ("explicit_negative", "observed_boundary"):
        raise EvidenceInsufficient("episode_end_not_observed")
    end = _number(episode["end"], "episode end")
    boundary_ids = episode.get("end_evidence_ids", [])
    events = {e["evidence_id"]: e for e in _events(evidence)}
    if not boundary_ids or any(eid not in episode["evidence_ids"] or eid not in events for eid in boundary_ids):
        raise EvidenceInsufficient("episode_end_missing_boundary_evidence")
    rows = [events[eid] for eid in boundary_ids
            if events[eid].get("vlm_lane", "combined") in GROUNDING_LANES]
    if episode["end_reason"] == "explicit_negative":
        supported = any(e["detected"] is False and e["publication_status"] == "published"
                        and abs(e["end"] - end) <= EPSILON for e in rows)
    else:
        supported = any(e.get("boundary_kind") == "end" and e["publication_status"] == "published"
                        and e.get("observed_boundary_seconds") == end for e in rows)
    if not supported:
        raise EvidenceInsufficient("episode_end_not_supported_by_explicit_boundary_verdict")
    return end


def _dispatch(question: dict, evidence: dict, result: dict) -> dict:
    form, ctx = question["question_form"], question.get("public_context", {})
    if form not in EVENT_FORMS:
        frames = _frames(evidence)
        _frame_coverage(evidence, frames)
    else:
        _window_plan(evidence, require_complete=form in SCENE_PRESENCE_FORMS)
    if form in ("vscrowd_full_sequence_peak_annotated_heads", "full_sequence_peak_visible_pedestrians"):
        peak = max(len(_observed(f)) for f in frames)
        return {"final_answer": str(peak), "evidence_refs": [_frame_ref(f) for f in frames if len(_observed(f)) == peak],
                "observation_target": "person", "question_target": question.get("target_label"),
                "target_semantics_mismatch": question.get("target_label") == "person_head"}
    if form.startswith("bbox_locator_"):
        axis = ctx.get("displacement_axis")
        if axis not in ("horizontal", "vertical"):
            raise SemanticUnsupported("explicit_public_displacement_axis_required")
        nominal_axis = "horizontal" if "horizontal" in form else "vertical"
        rulings = []
        if axis != nominal_axis:
            rulings.append({"field": "displacement_axis", "rule": "explicit_public_axis_over_form_name",
                            "form_axis": nominal_axis, "applied_axis": axis})
        locator = {"bbox_coordinate_system": ctx.get("bbox_coordinate_system"),
                   "timestamp_sec": ctx["first_timestamp_sec"],
                   "boxes": [{"bbox_xyxy": ctx["first_bbox_xyxy"], "object_label": ctx.get("object_label", "person")}]}
        ids, refs = _locate(evidence, frames, locator)
        start_frame, start = _at_subject_time(frames, ids[0], _number(ctx["first_timestamp_sec"], "first timestamp"))
        end_frame, end = _at_subject_time(frames, ids[0], _number(ctx["last_timestamp_sec"], "last timestamp"))
        if end_frame["t"] < start_frame["t"]:
            raise SemanticUnsupported("reversed_public_displacement_times")
        offset = 0 if axis == "horizontal" else 1
        a, b = _box(start["bbox"]), _box(end["bbox"])
        delta = (b[offset] + b[offset + 2] - a[offset] - a[offset + 2]) / 2
        words = ("left", "right", "approximately_same_horizontal_position") if axis == "horizontal" else (
            "up", "down", "approximately_same_vertical_position")
        answer = words[2] if abs(delta) < SAME_POSITION_FRACTION else words[1] if delta > 0 else words[0]
        return {"final_answer": _choose(question, answer), "evidence_refs": refs + [_frame_ref(end_frame, ids[0])],
                "subject_ids": ids, "normalized_displacement": delta, "semantic_rulings": rulings}
    if form in ("full_sequence_cumulative_region", "interval_cumulative_region"):
        rows, totals = _window_frames(question, frames), [0, 0, 0]
        for frame in rows:
            for subject in _observed(frame):
                totals[_third(subject)] += 1
        largest = max(totals)
        winners = [i for i, count in enumerate(totals) if count == largest]
        if len(winners) != 1:
            raise EvidenceInsufficient("cumulative_regions_tied")
        return {"final_answer": _choose(question, REGIONS[winners[0]]),
                "evidence_refs": [_frame_ref(f) for f in rows], "region_observation_totals": totals,
                "observation_unit": "observed_frame_person_pair"}
    if form in ("first_exact_visible_count", "last_exact_visible_count", "longest_at_or_above_threshold"):
        answer, refs, details = _count_temporal(question, frames)
        return {"final_answer": answer, "evidence_refs": refs, **details}
    if form == "event_presence_binary":
        presence, refs = _presence(evidence, ctx["queried_event_label"])
        result["evidence_refs"] = refs
        if presence is None:
            raise EvidenceInsufficient("event_presence_unknown_or_incomplete_negative_coverage")
        return {"final_answer": _choose(question, "yes" if presence else "no"), "evidence_refs": refs,
                "positive_semantics": "published_independent_verdict_not_native_confirmation"}
    if form in ("event_single_action_selection", "event_presence_combination"):
        options = _options(question)
        labels = ctx.get("event_labels") if form == "event_presence_combination" else options
        if not isinstance(labels, list) or not labels:
            raise SemanticUnsupported("public_event_labels_required")
        presence, refs = [], []
        for label in labels:
            verdict, label_refs = _presence(evidence, label)
            presence.append(verdict)
            refs.extend(label_refs)
        result["evidence_refs"] = refs
        if any(v is None for v in presence):
            raise EvidenceInsufficient("one_or_more_candidate_events_unknown")
        if form == "event_single_action_selection":
            positives = [label for label, verdict in zip(labels, presence) if verdict is True]
            if len(positives) != 1:
                raise EvidenceInsufficient("single_choice_has_zero_or_multiple_positive_events")
            answer = positives[0]
        else:
            answer = "; ".join(f"{chr(65 + i)}: {'present' if v else 'absent'}" for i, v in enumerate(presence))
        return {"final_answer": _choose(question, answer), "evidence_refs": refs, "event_presence": dict(zip(labels, presence))}
    if form == "fixed_conflict_first_segment_end_timestamp":
        if ctx.get("event_ordinal") != 1 or ctx.get("queried_boundary") != "end" or ctx.get("timestamp_origin") != "video_start":
            raise SemanticUnsupported("unsupported_public_event_boundary")
        episodes = _episodes(evidence, ctx["event_label"])
        if not episodes:
            raise EvidenceInsufficient("no_confirmed_event_episode")
        first = episodes[0]
        result["evidence_refs"] = [_episode_ref(first)]
        # 原始正判定未落入任何确认episode时，无法断言第一个确认段就是第一个真实段。
        covered_ids = {r for e in episodes for r in e["evidence_ids"]}
        events = [e for e in _events(evidence) if e.get("vlm_lane", "combined") in GROUNDING_LANES]
        known = {e["evidence_id"]: e for e in events}
        for event in events:
            if (canonical_event_label(event["event_label"]) == canonical_event_label(ctx["event_label"])
                    and event["detected"] is not False and event["start"] < first["start"] - EPSILON
                    and not _positive_verdict_ids(event, known) <= covered_ids):
                raise EvidenceInsufficient("earlier_event_verdict_not_resolved_to_episode")
        if not _prior_episode_coverage(evidence, ctx["event_label"], first):
            raise EvidenceInsufficient("first_episode_has_unresolved_prior_windows")
        if len(episodes) > 1 and abs(first["start"] - episodes[1]["start"]) <= EPSILON:
            raise EvidenceInsufficient("multiple_first_episodes_at_same_timestamp")
        end = _ended(evidence, first)
        return {"final_answer": f"{end:.1f}", "evidence_refs": [_episode_ref(first)], "unrounded_seconds": end,
                "first_boundary_resolution_window": _episode_resolution_window(evidence, ctx["event_label"], first)}
    if form == "source_native_interval_elapsed_seconds":
        frames = _frames(evidence)
        _frame_coverage(evidence, frames)
        ids, refs = _locate(evidence, frames, ctx["subject_locator"])
        anchor = _number(ctx["interval_start_sec"], "public interval start")
        for sid in ids:
            frame, _subject = _at_subject_time(frames, sid, anchor)
            refs.append(_frame_ref(frame, sid))
        episodes = _episodes(evidence, ctx["event_label"], ids)
        departure = canonical_event_label(ctx["event_label"]) == "person_leaves_shop_window"
        if departure:
            candidates = [e for e in episodes if e["start"] >= anchor - EPSILON]
            result["operator"] = "elapsed_until_associated_departure_start"
        else:
            candidates = [e for e in episodes if abs(e["start"] - anchor) <= TIME_NEIGHBOR_SECONDS]
        if not candidates or (not departure and len(candidates) != 1):
            raise EvidenceInsufficient("interval_not_uniquely_associated_with_located_subject")
        episode = candidates[0]
        refs.append(_episode_ref(episode))
        result["evidence_refs"] = refs
        if departure:
            if len(candidates) > 1 and abs(candidates[0]["start"] - candidates[1]["start"]) <= EPSILON:
                raise EvidenceInsufficient("multiple_first_departures_at_same_timestamp")
            if not _prior_episode_coverage(evidence, ctx["event_label"], episode, anchor=anchor, subject_ids=ids):
                raise EvidenceInsufficient("first_departure_has_unresolved_prior_windows")
        boundary = episode["start"] if departure else _ended(evidence, episode)
        if boundary < anchor:
            raise EvidenceInsufficient("event_boundary_precedes_public_interval_start")
        elapsed = boundary - anchor
        if departure:
            result["first_boundary_resolution_window"] = _episode_resolution_window(evidence, ctx["event_label"], episode)
        return {"final_answer": f"{elapsed:.1f}", "evidence_refs": refs, "subject_ids": ids,
                "unrounded_seconds": elapsed, "public_start_seconds": anchor, "observed_boundary_seconds": boundary,
                "semantic_rulings": ([{"rule": "public_departure_question_asks_onset", "applied_boundary": "start"}] if departure else [])}
    raise SemanticUnsupported("question_form_not_supported")


def answer_question(question: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """只接公开题目与同视频统一证据；格式错误抛ValueError，合法空答保留原因。"""
    form = question.get("question_form")
    result = {"qa_id": question["qa_id"], "final_answer": None, "status": "unsupported",
              "evidence_refs": [], "parser_version": PARSER_VERSION,
              "operator": FORM_OPERATORS.get(form, "unsupported_public_question_form"),
              "protocol": {"time_neighbor_seconds": TIME_NEIGHBOR_SECONDS,
                           "contiguous_gap_seconds": CONTIGUOUS_GAP_SECONDS,
                           "locator_min_iou": LOCATOR_MIN_IOU,
                           "locator_ambiguity_margin": LOCATOR_AMBIGUITY_MARGIN,
                           "same_position_fraction": SAME_POSITION_FRACTION}}
    if evidence.get("video_id") != question.get("video_id"):
        raise ValueError("evidence video identity mismatch")
    if form in SCENE_PRESENCE_FORMS:
        paths = frozenset({"events"})
    elif form in EVENT_FORMS:
        paths = frozenset({"tracking", "events", "fusion"})
    else:
        paths = frozenset({"tracking"})
    result["required_paths"] = sorted(paths)
    failures = evidence.get("failures", [])
    if not isinstance(failures, list):
        raise ValueError("failures must be a list with explicit stages")
    for failure in failures:
        if not isinstance(failure, dict):
            raise ValueError("each failure must be an object")
        lane = failure.get("vlm_lane", "combined")
        if not isinstance(lane, str) or lane not in VLM_LANES:
            raise ValueError("failure vlm_lane must be information/grounding/combined")
    relevant_failures = [f for f in failures if FAILURE_STAGE_PATHS.get(f.get("stage"), "global") in paths
                         or FAILURE_STAGE_PATHS.get(f.get("stage"), "global") == "global"]
    selected_lanes = INFORMATION_LANES if form in SCENE_PRESENCE_FORMS else GROUNDING_LANES
    if form in EVENT_FORMS:
        result["required_vlm_lanes"] = sorted(selected_lanes)
    relevant_failures = [f for f in relevant_failures
                         if FAILURE_STAGE_PATHS.get(f.get("stage"), "global") != "events"
                         or f.get("vlm_lane", "combined") in selected_lanes]
    # 老式全局失败没有分路记录时保守阻断；有记录时只阻断所需路径及全局I/O/hash失败。
    if relevant_failures or (evidence.get("request_failure") and not failures):
        return {**result, "status": "request_failure", "reason": "recorded_request_failure"}
    if evidence.get("source_eof") is not True:
        return {**result, "status": "incomplete_video", "reason": "source_not_finished"}
    for path in sorted(paths):
        if evidence.get(f"{path}_drained") is not True:
            return {**result, "status": "incomplete_video", "reason": f"{path}_queue_not_drained"}
    if form not in FORM_OPERATORS:
        return {**result, "reason": "question_form_not_supported"}
    try:
        if evidence.get("schema_version") != EVIDENCE_SCHEMA:
            raise EvidenceInsufficient("explicit_v2_evidence_schema_required")
        if form in EVENT_FORMS and form not in SCENE_PRESENCE_FORMS and not _native_lane_complete(evidence):
            return {**result, "status": "incomplete_video", "reason": "grounding_lane_not_complete"}
        details = _dispatch(question, evidence, result)
        return {**result, **details, "status": "answered"}
    except EvidenceInsufficient as exc:
        return {**result, "status": "insufficient_evidence", "reason": str(exc)}
    except SemanticUnsupported as exc:
        return {**result, "status": "unsupported", "reason": str(exc)}
