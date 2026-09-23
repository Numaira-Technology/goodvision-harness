"""Event-level metrics, separate from detection mAP and inference completion order."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..observations.schema import seconds, text


@dataclass(frozen=True, slots=True)
class EventEpisode:
    event_id: str
    source_id: str
    stream_session_id: str
    event_label: str
    subject_ids: tuple[str, ...]
    start_seconds: float
    end_seconds: float
    alert_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("event_id", "source_id", "stream_session_id", "event_label"):
            text(getattr(self, name), name)
        seconds(self.start_seconds, "start_seconds")
        seconds(self.end_seconds, "end_seconds")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("episodes must have positive duration")
        if (
            not isinstance(self.subject_ids, tuple)
            or not self.subject_ids
            or len(set(self.subject_ids)) != len(self.subject_ids)
        ):
            raise ValueError("episodes require unique actor IDs")
        for subject in self.subject_ids:
            text(subject, "subject_id")
        if self.alert_seconds is not None:
            seconds(self.alert_seconds, "alert_seconds")

    @property
    def key(self) -> tuple:
        return self.source_id, self.stream_session_id, self.event_id


def _overlap(left: EventEpisode, right: EventEpisode) -> float:
    return max(
        0.0,
        min(left.end_seconds, right.end_seconds)
        - max(left.start_seconds, right.start_seconds),
    )


def evaluate_events(
    predictions: list[EventEpisode],
    targets: list[EventEpisode],
    *,
    camera_hours: float,
    min_temporal_iou: float = 0.3,
    match_actors: bool = True,
) -> dict:
    """Maximum-cardinality one-to-one matching within camera, session and label.

    Edges prefer higher interval IoU, then earlier alert and stable IDs. An
    unmatched prediction is a false alert (including a duplicate); duplicates
    also get a separate count. Exposure must be supplied, including quiet time.
    Alert times must use seconds since the same session origin as source time.
    """
    seconds(camera_hours, "camera_hours")
    seconds(min_temporal_iou, "min_temporal_iou")
    if camera_hours <= 0 or not 0 < min_temporal_iou <= 1:
        raise ValueError("positive exposure and 0 < temporal IoU <= 1 are required")
    if type(match_actors) is not bool:
        raise ValueError("match_actors must be boolean")
    for episodes in (predictions, targets):
        if len(episodes) > 4096:
            raise ValueError("evaluate at most 4096 episodes per recording shard")
        if len({item.key for item in episodes}) != len(episodes):
            raise ValueError("episode IDs must be unique within each source/session")
    if any(item.alert_seconds is None for item in predictions):
        raise ValueError("predictions require observed alert times")
    predictions = sorted(predictions, key=lambda p: (p.alert_seconds, p.key))
    targets = sorted(targets, key=lambda p: (p.start_seconds, p.key))
    edges: list[list[int]] = []
    for prediction in predictions:
        candidates = []
        for j, target in enumerate(targets):
            if (
                prediction.source_id,
                prediction.stream_session_id,
                prediction.event_label,
            ) != (target.source_id, target.stream_session_id, target.event_label) or (
                match_actors and set(prediction.subject_ids) != set(target.subject_ids)
            ):
                continue
            intersection = _overlap(prediction, target)
            union = (
                prediction.end_seconds
                - prediction.start_seconds
                + target.end_seconds
                - target.start_seconds
                - intersection
            )
            iou = intersection / union
            if iou >= min_temporal_iou:
                candidates.append((-iou, target.key, j))
        edges.append([j for _, _, j in sorted(candidates)])

    # Iterative augmenting paths avoid greedy under-counting and recursion limits.
    owner: dict[int, int] = {}
    for i in range(len(predictions)):
        pending = [i]
        previous: dict[int, tuple[int, int] | None] = {i: None}
        found = None
        for current in pending:
            for target in edges[current]:
                if target not in owner:
                    found = (current, target)
                    break
                next_prediction = owner[target]
                if next_prediction not in previous:
                    previous[next_prediction] = (current, target)
                    pending.append(next_prediction)
            if found is not None:
                break
        if found is not None:
            current, target = found
            while True:
                owner[target] = current
                step = previous[current]
                if step is None:
                    break
                current, target = step

    matched = set(owner.values())
    delays, start_errors, end_errors = [], [], []
    missed_duration = sum(t.end_seconds - t.start_seconds for t in targets)
    matches = []
    for j, i in sorted(owner.items()):
        prediction, target = predictions[i], targets[j]
        delays.append(max(0.0, prediction.alert_seconds - target.start_seconds))
        start_errors.append(prediction.start_seconds - target.start_seconds)
        end_errors.append(prediction.end_seconds - target.end_seconds)
        missed_duration -= _overlap(prediction, target)
        matches.append({"prediction": list(prediction.key), "target": list(target.key)})
    tp = len(owner)
    fp = len(predictions) - tp
    duplicates = sum(
        i not in matched and bool(edges[i]) for i in range(len(predictions))
    )

    def percentile(values, quantile):
        return float(np.percentile(values, quantile)) if values else None

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": len(targets) - tp,
        "precision": tp / len(predictions) if predictions else None,
        "recall": tp / len(targets) if targets else None,
        "false_alerts_per_camera_hour": fp / camera_hours,
        "duplicate_alerts": duplicates,
        "duplicates_per_target_event": duplicates / len(targets) if targets else None,
        "time_to_detect_median_seconds": percentile(delays, 50),
        "time_to_detect_p95_seconds": percentile(delays, 95),
        "start_error_median_seconds": percentile(start_errors, 50),
        "end_error_median_seconds": percentile(end_errors, 50),
        "end_delay_p95_seconds": percentile([max(0.0, e) for e in end_errors], 95),
        "missed_event_duration_seconds": max(0.0, missed_duration),
        "camera_hours": camera_hours,
        "min_temporal_iou": min_temporal_iou,
        "match_actors": match_actors,
        "matches": matches,
    }
