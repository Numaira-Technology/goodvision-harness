"""Pure answer extractors that turn a Capture into benchmark predictions.

Each function implements one benchmark measurement semantics exactly as recorded
in bench_v1_sample/media.jsonl `taxonomy_cells`. They are deterministic and have
no harness or GPU dependency, so they are fully unit-tested.
"""

from __future__ import annotations

import re

from .schema import EVENT_TARGET_TO_HARNESS_LABEL, Capture, Question

# Column boundaries for horizontal thirds on the source-normalized x axis.
_LEFT_EDGE = 1.0 / 3.0
_RIGHT_EDGE = 2.0 / 3.0
_THIRD_LABELS = ("左侧", "中央", "右侧")

# region token in the question -> horizontal-third index, or None for full frame.
_REGION_THIRD = {"左侧": 0, "中央": 1, "右侧": 2}


def event_recognition(capture: Capture, target_label: str) -> str:
    """Q: does the video contain <event>?  -> "yes" / "no".

    Semantics: human_reviewed_positive_event_presence. We answer "yes" when the
    harness confirmed at least one matching event episode (its two-source-frame
    confirmation bar), "no" otherwise.
    """
    harness_label = EVENT_TARGET_TO_HARNESS_LABEL.get(target_label, target_label)
    confirmed = any(e.is_confirmed for e in capture.events_for_label(harness_label))
    return "yes" if confirmed else "no"


def temporal_event_end(
    capture: Capture, target_label: str, ordinal: int = 1
) -> float:
    """Q: when does the <ordinal>-th <event> end?  -> seconds (1 decimal).

    Uses the confirmed episode's `ended_since` timestamp. If the ordinal event
    never ended in-capture (still active at EOF), fall back to the last observed
    source time, then the capture duration.
    """
    harness_label = EVENT_TARGET_TO_HARNESS_LABEL.get(target_label, target_label)
    episodes = [e for e in capture.events_for_label(harness_label) if e.is_confirmed]
    episodes.sort(key=lambda e: (e.confirmed_since, e.first_evidence or 0.0))
    if ordinal < 1 or ordinal > len(episodes):
        raise ValueError(f"no confirmed {harness_label} episode #{ordinal}")
    episode = episodes[ordinal - 1]
    end = episode.ended_since
    if end is None:
        end = _last_observed_time(capture)
    return round(float(end), 1)


def peak_visible_count(capture: Capture) -> int:
    """Q: max simultaneously visible pedestrians (bbox centers inside image)."""
    peak = 0
    for frame in capture.frames:
        visible = sum(1 for s in frame.subjects if s.center_inside_image())
        peak = max(peak, visible)
    return peak


def busiest_horizontal_third(
    capture: Capture, start_s: float, end_s: float
) -> str:
    """Q: which third has the most cumulative visible pedestrian observations.

    Semantics: within_sequence_cumulative_visible_pedestrian_observations_by
    _horizontal_third over [start_s, end_s]. Each (frame, visible subject) is one
    observation, binned by center-x third; the third with the greatest total
    wins. Ties resolve left -> center -> right (stable argmax).
    """
    totals = [0, 0, 0]
    for frame in capture.frames_in_window(start_s, end_s):
        for subject in frame.subjects:
            if not subject.center_inside_image():
                continue
            cx, _ = subject.center
            totals[_third_index(cx)] += 1
    best = max(range(3), key=lambda i: totals[i])
    return _THIRD_LABELS[best]


def _region_count(frame, third: int | None) -> int:
    """Visible pedestrians in a frame, optionally restricted to one third."""
    total = 0
    for s in frame.subjects:
        if not s.center_inside_image():
            continue
        if third is None or _third_index(s.center[0]) == third:
            total += 1
    return total


def parse_count_temporal(question_text: str) -> dict | None:
    """Parse a MOT-style count-over-time temporal question.

    These questions (mot17/mot20 `temporal_questions`, target person) encode the
    threshold N, the region, and the boundary only in the natural-language text —
    the structured `parameters.region` field is unreliable in the source bank, so
    we read the region from the text too. Returns None when the text is not one
    of the known count-temporal shapes (e.g. a fight-event temporal).

    kinds:
      first_exact / last_exact  -> "第一次/最后一次恰好出现N名 … 第几秒" (timestamp)
      max_duration              -> "连续达到或超过N名的最长时段持续多少秒" (duration)
      first_max                 -> "第一次达到单画面最大值是在第几秒" (timestamp)
    """
    q = question_text
    if "行人" not in q and "人数" not in q:  # count-temporal questions are about people
        return None
    region = next((t for t in _REGION_THIRD if t in q), None)  # None => full frame
    third = _REGION_THIRD.get(region) if region else None

    if "最大值" in q:
        return {"kind": "first_max", "n": None, "third": third}
    m_n = re.search(r"(\d+)\s*名", q)
    n = int(m_n.group(1)) if m_n else None
    if ("持续" in q or "最长时段" in q) and n is not None:
        return {"kind": "max_duration", "n": n, "third": third}
    if "恰好出现" in q and n is not None:
        boundary = "last" if "最后一次" in q else "first"
        return {"kind": f"{boundary}_exact", "n": n, "third": third}
    return None


def count_exact_time(capture: Capture, n: int, third: int | None, boundary: str) -> float:
    """First/last time the (region) visible pedestrian count is exactly N."""
    hits = [f.t for f in capture.frames if _region_count(f, third) == n]
    if not hits:
        raise ValueError(f"count never exactly {n} in region {third}")
    return round(hits[0] if boundary == "first" else hits[-1])


def count_max_duration(capture: Capture, n: int, third: int | None) -> float:
    """Longest contiguous span (seconds) with region count >= N."""
    best = 0.0
    run_start: float | None = None
    prev_t: float | None = None
    for f in capture.frames:  # capture frames are in source order
        if _region_count(f, third) >= n:
            if run_start is None:
                run_start = f.t
            prev_t = f.t
        else:
            if run_start is not None and prev_t is not None:
                best = max(best, prev_t - run_start)
            run_start = None
    if run_start is not None and prev_t is not None:
        best = max(best, prev_t - run_start)
    return round(best)


def count_first_reaches_max(capture: Capture, third: int | None) -> float:
    """First time the region count reaches its own per-frame maximum."""
    counts = [(f.t, _region_count(f, third)) for f in capture.frames]
    if not counts:
        raise ValueError("no frames")
    peak = max(c for _, c in counts)
    return round(next(t for t, c in counts if c == peak))


def _third_index(cx: float) -> int:
    if cx < _LEFT_EDGE:
        return 0
    if cx < _RIGHT_EDGE:
        return 1
    return 2


# Relative-displacement location: option vocabularies keyed by axis.
_REL_OPTIONS = {
    "horizontal": ("left", "right", "approximately_same_horizontal_position"),
    "vertical": ("up", "down", "approximately_same_vertical_position"),
}
# A net center move below this fraction of the frame counts as "approximately
# the same position". The benchmark's own threshold lives in its private
# ground-truth generator; this is a documented approximation on the harness side.
_SAME_TOL_FRAC = 0.03


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _subject_at(capture: Capture, t: float, ref_box: tuple[float, float, float, float]) -> Subject | None:
    """The subject in the frame nearest time t whose box best matches ref_box."""
    frames = [f for f in capture.frames if f.subjects]
    if not frames:
        return None
    frame = min(frames, key=lambda f: abs(f.t - t))
    best, best_score = None, 0.0
    for s in frame.subjects:
        score = _iou(s.bbox, ref_box)
        if score > best_score:
            best, best_score = s, score
    if best is not None and best_score > 0.0:
        return best
    # No overlap (tracker box drifted): fall back to nearest center.
    rcx = (ref_box[0] + ref_box[2]) / 2.0
    rcy = (ref_box[1] + ref_box[3]) / 2.0
    return min(frame.subjects, key=lambda s: (s.center[0] - rcx) ** 2 + (s.center[1] - rcy) ** 2)


def _track_center_at(capture: Capture, subject_id: str, t: float, axis_idx: int) -> float | None:
    """Center coordinate of a tracked subject in the frame nearest time t."""
    frames = [f for f in capture.frames if any(s.id == subject_id for s in f.subjects)]
    if not frames:
        return None
    frame = min(frames, key=lambda f: abs(f.t - t))
    s = next(s for s in frame.subjects if s.id == subject_id)
    return s.center[axis_idx]


def relative_displacement(
    capture: Capture,
    ref_box_px: tuple[float, float, float, float],
    first_t: float,
    last_t: float,
    axis: str,
) -> str:
    """Net movement of one referenced person along an axis -> option string.

    The reference box is in source pixels; normalize it with the capture frame
    size, match it to a track at first_t, follow that track to last_t, and report
    the sign of the net center displacement (with a dead-band for "same").
    """
    if capture.width and capture.height:
        w, h = float(capture.width), float(capture.height)
        ref = (ref_box_px[0] / w, ref_box_px[1] / h, ref_box_px[2] / w, ref_box_px[3] / h)
    else:
        ref = ref_box_px  # assume already normalized
    axis_idx = 0 if axis == "horizontal" else 1
    subject = _subject_at(capture, first_t, ref)
    if subject is None:
        raise ValueError("no tracked subject to match the reference box")
    start = _track_center_at(capture, subject.id, first_t, axis_idx)
    end = _track_center_at(capture, subject.id, last_t, axis_idx)
    if start is None or end is None:
        raise ValueError("referenced track not observed at both endpoints")
    delta = end - start
    less, more, same = _REL_OPTIONS[axis]
    if abs(delta) < _SAME_TOL_FRAC:
        return same
    # _REL_OPTIONS orders each axis as (negative-dir, positive-dir, same):
    # horizontal (left, right) with +x = right; vertical (up, down) with +y = down.
    return more if delta > 0 else less


def _last_observed_time(capture: Capture) -> float:
    times = [f.t for f in capture.frames]
    if times:
        return max(times)
    if capture.duration_seconds is not None:
        return capture.duration_seconds
    return 0.0


def answer_question(question: Question, capture: Capture) -> str:
    """Dispatch a single question to its extractor and format the prediction."""
    ctx = question.public_context
    capability = question.capability

    if capability == "event_recognition":
        target = str(ctx.get("queried_event_label") or question.raw.get("target_label"))
        return event_recognition(capture, target)

    if capability == "temporal_questions":
        # MOT-style count-over-time questions are computed from tracked frames;
        # everything else is an event-boundary question over scene events.
        spec = parse_count_temporal(question.question)
        if spec is not None:
            if spec["kind"] == "first_max":
                return _fmt_int(count_first_reaches_max(capture, spec["third"]))
            if spec["kind"] == "max_duration":
                return _fmt_int(count_max_duration(capture, spec["n"], spec["third"]))
            boundary = "last" if spec["kind"] == "last_exact" else "first"
            return _fmt_int(count_exact_time(capture, spec["n"], spec["third"], boundary))
        target = str(ctx.get("event_label") or question.raw.get("target_label"))
        ordinal = int(ctx.get("event_ordinal", 1))
        return _fmt_number(temporal_event_end(capture, target, ordinal))

    if capability == "counting":
        return str(peak_visible_count(capture))

    if capability == "location_tracking":
        opts = set(question.response_options or ())
        # Sub-form 1: relative displacement of one referenced person.
        if "approximately_same_horizontal_position" in opts:
            return relative_displacement(
                capture, tuple(ctx["first_bbox_xyxy"]),
                float(ctx["first_timestamp_sec"]), float(ctx["last_timestamp_sec"]),
                "horizontal")
        if "approximately_same_vertical_position" in opts:
            return relative_displacement(
                capture, tuple(ctx["first_bbox_xyxy"]),
                float(ctx["first_timestamp_sec"]), float(ctx["last_timestamp_sec"]),
                "vertical")
        # Sub-form 2: busiest horizontal third over a window.
        params = ctx.get("parameters", {})
        start_s = float(params.get("start_second", ctx.get("clip", {}).get("start_sec", 0.0)))
        end_s = float(params.get("end_second", ctx.get("clip", {}).get("end_sec", 0.0)))
        return busiest_horizontal_third(capture, start_s, end_s)

    raise ValueError(f"unsupported capability: {capability}")


def _fmt_number(value: float) -> str:
    """Trim trailing zeros but keep the one-decimal contract readable (18.0)."""
    text = f"{value:.1f}"
    return text


def _fmt_int(value: float) -> str:
    """MOT count-temporal questions ask for whole seconds (请只回答整数秒)."""
    return str(int(round(value)))
