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
