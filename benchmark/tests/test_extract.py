"""Unit tests for the pure answer extractors (no harness/GPU needed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.extract import (  # noqa: E402
    answer_question,
    busiest_horizontal_third,
    count_exact_time,
    count_first_reaches_max,
    count_max_duration,
    event_recognition,
    parse_count_temporal,
    peak_visible_count,
    relative_displacement,
    temporal_event_end,
)
from bench.schema import Capture, Question  # noqa: E402


def _cap(**kw):
    base = {"video_id": "v", "fps": 30.0, "duration_seconds": 20.0}
    base.update(kw)
    return Capture.from_json(base)


def _subj(sid, cx, cy, w=0.05, h=0.1):
    return {"id": sid, "bbox": [cx - w, cy - h, cx + w, cy + h]}


# ----- event_recognition (Q1) -----

def test_event_recognition_yes_when_confirmed():
    cap = _cap(events=[{"event_label": "fight", "confirmed_since": 12.3, "ended_since": 18.0}])
    assert event_recognition(cap, "physical_conflict") == "yes"


def test_event_recognition_no_when_only_suspected():
    cap = _cap(events=[{"event_label": "fight", "first_evidence": 10.0}])  # never confirmed
    assert event_recognition(cap, "physical_conflict") == "no"


def test_event_recognition_no_when_absent():
    assert event_recognition(_cap(events=[]), "physical_conflict") == "no"


# ----- temporal_event_end (Q4) -----

def test_temporal_end_uses_ended_since():
    cap = _cap(events=[{"event_label": "fight", "confirmed_since": 12.3, "ended_since": 18.04}])
    assert temporal_event_end(cap, "physical_conflict", 1) == 18.0


def test_temporal_end_picks_first_by_confirmation_order():
    cap = _cap(events=[
        {"event_label": "fight", "confirmed_since": 15.0, "ended_since": 19.0},
        {"event_label": "fight", "confirmed_since": 5.0, "ended_since": 8.0},
    ])
    assert temporal_event_end(cap, "physical_conflict", 1) == 8.0


def test_temporal_end_fallbacks_to_last_frame_when_open():
    cap = _cap(
        events=[{"event_label": "fight", "confirmed_since": 12.0, "ended_since": None}],
        frames=[{"t": 17.5, "frame": 525, "subjects": []}],
    )
    assert temporal_event_end(cap, "physical_conflict", 1) == 17.5


# ----- peak_visible_count (Q3) -----

def test_peak_count_is_max_simultaneous():
    cap = _cap(frames=[
        {"t": 0.0, "frame": 1, "subjects": [_subj("a", 0.2, 0.5), _subj("b", 0.8, 0.5)]},
        {"t": 1.0, "frame": 31, "subjects": [_subj("a", 0.2, 0.5), _subj("b", 0.8, 0.5), _subj("c", 0.5, 0.5)]},
        {"t": 2.0, "frame": 61, "subjects": [_subj("a", 0.2, 0.5)]},
    ])
    assert peak_visible_count(cap) == 3


def test_peak_count_excludes_centers_outside_image():
    cap = _cap(frames=[
        {"t": 0.0, "frame": 1, "subjects": [_subj("a", 0.5, 0.5), _subj("ghost", 1.4, 0.5)]},
    ])
    assert peak_visible_count(cap) == 1


# ----- busiest_horizontal_third (Q2) -----

def test_location_third_sums_over_window():
    cap = _cap(frames=[
        # inside window 9-13: right side has more cumulative observations
        {"t": 9.0, "frame": 271, "subjects": [_subj("a", 0.9, 0.5), _subj("b", 0.85, 0.5)]},
        {"t": 10.0, "frame": 301, "subjects": [_subj("a", 0.9, 0.5), _subj("c", 0.1, 0.5)]},
        # outside window: should be ignored even though left is crowded
        {"t": 2.0, "frame": 61, "subjects": [_subj("x", 0.1, 0.5), _subj("y", 0.1, 0.5), _subj("z", 0.1, 0.5)]},
    ])
    assert busiest_horizontal_third(cap, 9.0, 13.0) == "右侧"


def test_location_third_boundaries():
    cap = _cap(frames=[
        {"t": 10.0, "frame": 301, "subjects": [
            _subj("l", 0.30, 0.5),   # < 1/3 -> 左侧
            _subj("c", 0.50, 0.5),   # middle -> 中央
            _subj("c2", 0.60, 0.5),  # middle -> 中央
        ]},
    ])
    assert busiest_horizontal_third(cap, 9.0, 13.0) == "中央"


# ----- dispatch via answer_question -----

def _q(**kw):
    base = {"qa_id": "q", "capability": "counting", "question": "?", "video_id": "v"}
    base.update(kw)
    return Question.from_json(base)


def test_dispatch_counting():
    cap = _cap(frames=[{"t": 0.0, "frame": 1, "subjects": [_subj("a", 0.5, 0.5)]}])
    q = _q(capability="counting", answer_type="numeric_count")
    assert answer_question(q, cap) == "1"


def test_dispatch_location_reads_parameters():
    cap = _cap(frames=[{"t": 10.0, "frame": 301, "subjects": [_subj("a", 0.9, 0.5)]}])
    q = _q(
        capability="location_tracking",
        response_options=["左侧", "中央", "右侧"],
        public_context={"parameters": {"start_second": 9, "end_second": 13}},
    )
    assert answer_question(q, cap) == "右侧"


def test_dispatch_event_and_temporal():
    cap = _cap(events=[{"event_label": "fight", "confirmed_since": 12.0, "ended_since": 18.0}])
    q1 = _q(capability="event_recognition", response_options=["yes", "no"],
            public_context={"queried_event_label": "physical_conflict"})
    q4 = _q(capability="temporal_questions",
            public_context={"event_label": "physical_conflict", "event_ordinal": 1})
    assert answer_question(q1, cap) == "yes"
    assert answer_question(q4, cap) == "18.0"


# ----- count-over-time temporal (mot17/mot20) -----

def test_parse_count_temporal_shapes():
    p = parse_count_temporal("浏览完整视频后，全画面第一次恰好出现15名可见有效行人是在第几秒？")
    assert p == {"kind": "first_exact", "n": 15, "third": None}
    p = parse_count_temporal("画面左侧三等分区域最后一次恰好出现3名可见有效行人是在第几秒？")
    assert p == {"kind": "last_exact", "n": 3, "third": 0}
    p = parse_count_temporal("画面右侧三等分区域内可见有效行人连续达到或超过4名的最长时段持续多少秒？")
    assert p == {"kind": "max_duration", "n": 4, "third": 2}
    p = parse_count_temporal("画面中央三等分区域人数第一次达到单画面最大值是在第几秒？")
    assert p == {"kind": "first_max", "n": None, "third": 1}
    # a fight-event temporal must NOT match the count parser
    assert parse_count_temporal("When does the first physical conflict end?") is None


def _f(t, frame, centers):
    return {"t": t, "frame": frame, "subjects": [_subj(f"s{i}", cx, 0.5) for i, cx in enumerate(centers)]}


def test_count_exact_first_and_last():
    cap = _cap(frames=[
        _f(1.0, 1, [0.5]),            # 1
        _f(2.0, 2, [0.2, 0.5]),       # 2  <- first exactly 2
        _f(3.0, 3, [0.2, 0.5, 0.8]),  # 3
        _f(4.0, 4, [0.2, 0.5]),       # 2  <- last exactly 2
    ])
    assert count_exact_time(cap, 2, None, "first") == 2
    assert count_exact_time(cap, 2, None, "last") == 4


def test_count_exact_respects_region():
    cap = _cap(frames=[_f(5.0, 1, [0.1, 0.5, 0.9])])  # 1 per third
    assert count_exact_time(cap, 1, 1, "first") == 5   # 中央 third has exactly 1


def test_count_max_duration_longest_run():
    cap = _cap(frames=[
        _f(0.0, 1, [0.5, 0.6]),        # >=2
        _f(1.0, 2, [0.5, 0.6, 0.7]),   # >=2  run 0..1 = 1.0s
        _f(2.0, 3, [0.5]),             # <2 break
        _f(3.0, 4, [0.5, 0.6]),        # >=2
        _f(5.0, 5, [0.5, 0.6]),        # >=2  run 3..5 = 2.0s (longest)
    ])
    assert count_max_duration(cap, 2, None) == 2


def test_count_first_reaches_max():
    cap = _cap(frames=[_f(0.0, 1, [0.5]), _f(1.0, 2, [0.5, 0.6]), _f(2.0, 3, [0.5, 0.6])])
    assert count_first_reaches_max(cap, None) == 1  # peak=2 first seen at t=1


def test_dispatch_count_temporal_integer_seconds():
    cap = _cap(frames=[_f(1.0, 1, [0.5]), _f(2.0, 2, [0.2, 0.5]), _f(3.0, 3, [0.2, 0.5, 0.8])])
    q = _q(capability="temporal_questions", answer_type="numeric_seconds",
           question="浏览完整视频后，全画面第一次恰好出现2名可见有效行人是在第几秒？请只回答整数秒。")
    assert answer_question(q, cap) == "2"


# ----- relative-displacement location (relative_horizontal / vertical) -----

def _capwh(frames, w=100.0, h=100.0):
    return Capture.from_json({"video_id": "v", "fps": 30.0, "duration_seconds": 50.0,
                              "width": w, "height": h, "frames": frames})


def _mv(t, sid, cx, cy):
    return {"t": t, "frame": int(t * 30), "subjects": [{"id": sid, "bbox": [cx - .05, cy - .05, cx + .05, cy + .05]}]}


def test_relative_horizontal_right():
    cap = _capwh([_mv(0.0, "p", 0.2, 0.4), _mv(40.0, "p", 0.8, 0.4)])
    ref = (15.0, 35.0, 25.0, 45.0)  # px -> norm center (0.2,0.4), matches p at t=0
    assert relative_displacement(cap, ref, 0.0, 40.0, "horizontal") == "right"


def test_relative_horizontal_same_within_deadband():
    cap = _capwh([_mv(0.0, "p", 0.50, 0.4), _mv(40.0, "p", 0.51, 0.4)])
    ref = (45.0, 35.0, 55.0, 45.0)
    assert relative_displacement(cap, ref, 0.0, 40.0, "horizontal") == \
        "approximately_same_horizontal_position"


def test_relative_vertical_up_is_negative_y():
    cap = _capwh([_mv(0.0, "p", 0.5, 0.8), _mv(40.0, "p", 0.5, 0.2)])
    ref = (45.0, 75.0, 55.0, 85.0)
    assert relative_displacement(cap, ref, 0.0, 40.0, "vertical") == "up"


def test_relative_matches_correct_track_of_two():
    # two people; reference box overlaps the RIGHT one, which then moves left
    f0 = {"t": 0.0, "frame": 0, "subjects": [
        {"id": "L", "bbox": [0.05, 0.35, 0.15, 0.45]},
        {"id": "R", "bbox": [0.75, 0.35, 0.85, 0.45]}]}
    f1 = {"t": 40.0, "frame": 1200, "subjects": [
        {"id": "L", "bbox": [0.05, 0.35, 0.15, 0.45]},
        {"id": "R", "bbox": [0.25, 0.35, 0.35, 0.45]}]}
    cap = _capwh([f0, f1])
    ref = (75.0, 35.0, 85.0, 45.0)  # overlaps R
    assert relative_displacement(cap, ref, 0.0, 40.0, "horizontal") == "left"


def test_relative_displacement_refuses_pixel_box_without_dims():
    # no width/height + a pixel-space reference box -> must refuse, not guess
    cap = Capture.from_json({"video_id": "v", "fps": 30.0, "duration_seconds": 50.0,
                             "frames": [_mv(0.0, "a", 0.2, 0.4), _mv(40.0, "a", 0.8, 0.4)]})
    import pytest
    with pytest.raises(ValueError):
        relative_displacement(cap, (100.0, 100.0, 200.0, 300.0), 0.0, 40.0, "horizontal")


def test_relative_displacement_normalized_box_without_dims_ok():
    cap = Capture.from_json({"video_id": "v", "fps": 30.0, "duration_seconds": 50.0,
                             "frames": [_mv(0.0, "a", 0.2, 0.4), _mv(40.0, "a", 0.8, 0.4)]})
    # already-normalized box is fine even without dims
    assert relative_displacement(cap, (0.15, 0.35, 0.25, 0.45), 0.0, 40.0, "horizontal") == "right"


def test_count_max_duration_does_not_stitch_across_empty_frame():
    # 0.5s cadence; an explicit empty frame at 1.5 splits two 1s runs -> max 1s,
    # NOT one stitched 3s span across the gap.
    cap = _cap(frames=[_f(0.0, 0, [0.5]), _f(0.5, 15, [0.5]), _f(1.0, 30, [0.5]),
                       _f(1.5, 45, []),
                       _f(2.0, 60, [0.5]), _f(2.5, 75, [0.5]), _f(3.0, 90, [0.5])])
    assert count_max_duration(cap, 1, None) == 1


def test_count_max_duration_breaks_on_large_unobserved_gap():
    # steady 0.5s cadence then a 5s jump (unobserved) -> the late frame is not
    # stitched onto the earlier run.
    cap = _cap(frames=[_f(0.0, 0, [0.5]), _f(0.5, 15, [0.5]), _f(1.0, 30, [0.5]),
                       _f(6.0, 180, [0.5])])
    assert count_max_duration(cap, 1, None) == 1


def test_count_max_duration_real_contiguous_run():
    cap = _cap(frames=[_f(0.0, 0, [0.5]), _f(0.5, 15, [0.5]), _f(1.0, 30, [0.5]),
                       _f(1.5, 45, [0.5]), _f(2.0, 60, [0.5])])
    assert count_max_duration(cap, 1, None) == 2


def test_dispatch_relative_displacement_via_options():
    cap = _capwh([_mv(0.0, "p", 0.2, 0.4), _mv(40.0, "p", 0.8, 0.4)])
    q = _q(capability="location_tracking",
           response_options=["left", "right", "approximately_same_horizontal_position"],
           public_context={"first_bbox_xyxy": [15.0, 35.0, 25.0, 45.0],
                           "first_timestamp_sec": 0.0, "last_timestamp_sec": 40.0})
    assert answer_question(q, cap) == "right"
