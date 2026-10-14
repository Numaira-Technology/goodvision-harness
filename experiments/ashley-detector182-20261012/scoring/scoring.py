"""Benchmark V1 scoring, independent of model and dataset dependencies."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from typing import Any

import numpy as np

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

# Time points use absolute tolerance so a shared timeline shift does not change
# their score. Durations and counts use MRA with an MAE companion.
_TIME_POINT_UNITS = {"seconds_from_video_start", "second_1_based"}
# Time-point tolerance = max(floor, fraction * video_duration). The tolerance
# scales with video length (2%), so short clips stay strict and long clips get
# proportional slack; +/-1s of forgiveness begins around 50s. The 0.5s floor is
# a sanity minimum for continuous timestamps on very short clips -- it is below
# 1s, so it does not grant integer "which second" answers any off-by-one credit.
# video duration comes from the question's public_context clip window; a time
# point without one (e.g. the conflict-end timestamp) uses the floor.
_TIME_POINT_TOLERANCE_FRACTION = Decimal("0.02")
_TIME_POINT_TOLERANCE_FLOOR = Decimal("0.5")
_CHOICE_METRICS = {
    "event_recognition": "Event_Recognition_Accuracy",
    "location_tracking": "Location_Tracking_Accuracy",
}

# Location questions are asked in Chinese or English depending on the source, and
# their option labels match. A model may answer in the other language, so map
# every accepted form to one language-neutral token and compare on that. The
# "approximately same" options collapse across both axes: the question fixes the
# axis, so a bare "same"/"基本不变" can only mean this question's same-position
# option.
_LOCATION_CANON: dict[str, str] = {}
for _canonical, _forms in {
    "left": ["left", "左", "左侧", "左边", "左方", "向左", "偏左", "靠左"],
    "right": ["right", "右", "右侧", "右边", "右方", "向右", "偏右", "靠右"],
    "up": ["up", "upward", "上", "上方", "上边", "向上", "偏上", "靠上"],
    "down": ["down", "downward", "下", "下方", "下边", "向下", "偏下", "靠下"],
    "center": ["center", "centre", "middle", "中央", "中间", "中部", "正中"],
    "approximately_same": [
        "approximately_same_horizontal_position",
        "approximately_same_vertical_position",
        "approximately the same", "approximately same", "about the same",
        "same", "unchanged", "no change",
        "基本不变", "大致不变", "位置基本不变", "水平位置基本不变", "垂直位置基本不变",
    ],
}.items():
    for _form in _forms:
        _LOCATION_CANON[_form.casefold()] = _canonical


def _canonical_location(text: str) -> str:
    """Language-neutral token for a location answer; unknown text folds to itself."""
    stripped = str(text).strip()
    return _LOCATION_CANON.get(stripped.casefold(), stripped.casefold())


# This floor modifies standard VSI-Bench MRA for targets below 0.5. Keep the
# protocol explicit; it is not a human-calibrated temporal tolerance.
_MRA_DENOM_FLOOR = 0.5

# MRA functions from Thinking in Space / VSI-Bench (Apache-2.0):
# https://github.com/vision-x-nyu/thinking-in-space/blob/main/lmms_eval/tasks/vsibench/utils.py
def abs_dist_norm(pred, target):
    return abs(pred - target) / max(target, _MRA_DENOM_FLOOR)


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def _parse_magnitude(response: str) -> float | None:
    # Preserve VSI-Bench's first-word numeric parsing for magnitude estimates.
    try:
        prediction = float(response.split(" ")[0].rstrip(".").strip())
    except ValueError:
        return None
    return prediction if math.isfinite(prediction) and prediction >= 0 else None


def magnitude_mra(response: str, target: int | float) -> float:
    prediction = _parse_magnitude(response)
    if prediction is None:
        return 0.0
    return float(mean_relative_accuracy(prediction, float(target), start=.5, end=.95, interval=.05))


def _absolute_error(response: str, target: int | float) -> float | None:
    prediction = _parse_magnitude(response)
    return None if prediction is None else abs(prediction - float(target))


def _parse_time_point(response: str, unit: str) -> Decimal | None:
    if not _NUMBER.fullmatch(response):
        return None
    prediction = Decimal(response)
    minimum = 1 if unit == "second_1_based" else 0
    if not prediction.is_finite() or prediction < minimum:
        return None
    return prediction


def _video_duration_sec(doc: dict[str, Any]) -> Decimal | None:
    """Video duration in seconds from the question's public_context clip window.

    Only the clip window is used: it is public (already in questions.jsonl) and
    carries no answer. Time points without a clip window (the conflict-end
    timestamps) fall back to the tolerance floor rather than an annotation-derived
    duration, which could recover the private conflict-end answer."""
    try:
        clip = (json.loads(doc.get("public_context_json") or "{}") or {}).get("clip") or {}
        start, end = clip.get("start_sec"), clip.get("end_sec")
        if start is not None and end is not None:
            duration = Decimal(str(end)) - Decimal(str(start))
            if duration > 0:
                return duration
    except (TypeError, ValueError):
        return None
    return None


def benchmark_v1_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    response = str(results[0]).strip() if results else ""
    truth = json.loads(doc["ground_truth_json"])
    rule = truth["scoring_rule"]
    capability = doc["capability"]

    if capability == "counting":
        # Counts are magnitudes: graded MRA, plus an absolute-error companion.
        score = magnitude_mra(response, truth["value"])
        metrics = {"Overall_Score": score, "Counting_MRA": score}
        mae = _absolute_error(response, truth["value"])
        metrics["Counting_Valid_Rate"] = float(mae is not None)
        if mae is not None:
            metrics["Counting_MAE"] = mae
        return metrics

    if capability == "temporal_questions":
        if truth["unit"] in _TIME_POINT_UNITS:
            # Absolute tolerance = max(floor, 1% of video duration); position on
            # the timeline does not change it. MAE reports the raw gap for analysis.
            duration = _video_duration_sec(doc)
            tolerance = (
                max(_TIME_POINT_TOLERANCE_FLOOR, _TIME_POINT_TOLERANCE_FRACTION * duration)
                if duration is not None
                else _TIME_POINT_TOLERANCE_FLOOR
            )
            prediction = _parse_time_point(response, truth["unit"])
            target = Decimal(str(truth["value"]))
            score = float(prediction is not None and target - tolerance <= prediction <= target + tolerance)
            metrics = {
                "Overall_Score": score,
                "Temporal_TimePoint_Accuracy": score,
                "Temporal_TimePoint_Valid_Rate": float(prediction is not None),
            }
            if prediction is not None:
                metrics["Temporal_TimePoint_MAE"] = float(abs(prediction - target))
            return metrics
        if truth["unit"] != "seconds":
            raise ValueError(f"Unsupported Benchmark V1 temporal unit: {truth['unit']}")
        # Durations are magnitudes: MRA plus an absolute-error companion.
        score = magnitude_mra(response, truth["value"])
        metrics = {"Overall_Score": score, "Temporal_Duration_MRA": score}
        mae = _absolute_error(response, truth["value"])
        metrics["Temporal_Duration_Valid_Rate"] = float(mae is not None)
        if mae is not None:
            metrics["Temporal_Duration_MAE"] = mae
        return metrics

    if rule in {"exact_option_label", "casefolded_exact_label"}:
        if capability == "location_tracking":
            # Accept the Chinese or English form of the direction/region.
            score = float(_canonical_location(response) == _canonical_location(truth["value"]))
        elif rule == "casefolded_exact_label":
            score = float(response.casefold() == str(truth["value"]).casefold())
        else:
            score = float(response == truth["value"])
        return {"Overall_Score": score, _CHOICE_METRICS[capability]: score}

    raise ValueError(f"Unsupported Benchmark V1 scoring rule: {rule}")


def aggregate_mean(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0
