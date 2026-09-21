"""Pure answer extractors that turn a Capture into benchmark predictions.

Each function implements one benchmark measurement semantics exactly as recorded
in bench_v1_sample/media.jsonl `taxonomy_cells`. They are deterministic and have
no harness or GPU dependency, so they are fully unit-tested.
"""

from __future__ import annotations

from .schema import EVENT_TARGET_TO_HARNESS_LABEL, Capture, Question

# Column boundaries for horizontal thirds on the source-normalized x axis.
_LEFT_EDGE = 1.0 / 3.0
_RIGHT_EDGE = 2.0 / 3.0
_THIRD_LABELS = ("左侧", "中央", "右侧")


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
