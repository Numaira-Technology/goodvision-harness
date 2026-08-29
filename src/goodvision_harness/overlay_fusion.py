"""Associate sparse event objects with stable rich-label tracks.

The event model and the local detector intentionally keep separate object ID
namespaces.  This module performs a conservative spatial association at an
event keyframe, then remembers the selected rich ``track_id`` while that event
state is held or interpolated by the display compositor.  Ambiguous objects are
left unmatched so the legacy event rectangle can remain as a safe fallback.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from threading import RLock

from .contracts import AnalysisResult, GroundedObject, NormalizedBBox
from .rich_labeling import RichLabelObject

_PERSON_LABELS = frozenset(
    {
        "adult",
        "boy",
        "child",
        "female",
        "fighter",
        "fighters",
        "girl",
        "human",
        "knownperson",
        "male",
        "man",
        "men",
        "pedestrian",
        "people",
        "person",
        "persons",
        "smoker",
        "unknownperson",
        "woman",
        "women",
        "worker",
    }
)
_GROUP_PERSON_LABELS = frozenset(
    {"fighters", "men", "people", "persons", "women"}
)
_LABEL_ALIASES = {
    "cellphone": "phone",
    "mobilephone": "phone",
    "motorbike": "motorcycle",
    "purse": "handbag",
    "rucksack": "backpack",
    "sofa": "couch",
    "television": "tv",
}
_FACE_PERSON_LABELS = frozenset(
    {
        "face",
        "knownface",
        "knownperson",
        "unknownface",
        "unknownperson",
    }
)
_PERSON_TARGET_EVENTS = frozenset({"fight", "fall", "stranger"})
_OBJECT_TARGET_EVENTS = frozenset({"smoking", "weapon"})
_MIN_MATCH_SCORE = 0.42
_AMBIGUITY_MARGIN = 0.08


@dataclass(frozen=True, slots=True)
class RichAlert:
    """One event badge displayed on a rich-label track."""

    event_label: str
    confidence: float


@dataclass(frozen=True, slots=True)
class TrackAlerts:
    """All active event badges for one stable rich-label track."""

    track_id: str
    alerts: tuple[RichAlert, ...]


@dataclass(frozen=True, slots=True)
class ObjectAlerts:
    """All active event badges for one unmatched model object."""

    object_id: str
    alerts: tuple[RichAlert, ...]


@dataclass(frozen=True, slots=True)
class OverlayFusionPlan:
    """Conservative render plan for one event/rich display frame."""

    track_alerts: tuple[TrackAlerts, ...]
    matched_event_object_ids: tuple[str, ...]
    unmatched_event_object_ids: tuple[str, ...]
    unmatched_object_alerts: tuple[ObjectAlerts, ...]

    def alerts_by_track_id(self) -> dict[str, tuple[RichAlert, ...]]:
        return {item.track_id: item.alerts for item in self.track_alerts}

    def alerts_by_unmatched_object_id(self) -> dict[str, tuple[RichAlert, ...]]:
        return {
            item.object_id: item.alerts for item in self.unmatched_object_alerts
        }


class OverlayFusionMatcher:
    """Match event objects to rich tracks and retain stable associations."""

    def __init__(self, *, max_cached_associations: int = 512) -> None:
        if max_cached_associations < 1:
            raise ValueError("max_cached_associations must be positive")
        self._max_cached_associations = int(max_cached_associations)
        self._associations: OrderedDict[
            tuple[str, str, str, str], str
        ] = OrderedDict()
        self._lock = RLock()

    def reset(self) -> None:
        with self._lock:
            self._associations.clear()

    def plan(
        self,
        analysis: AnalysisResult,
        rich_objects: Sequence[RichLabelObject],
        *,
        association_message_id: str | None = None,
    ) -> OverlayFusionPlan:
        """Return matched badges and unmatched event fallback object IDs.

        Only objects referenced by positive judgments participate.  Cached
        associations take precedence while the rich track remains visible;
        otherwise a new association requires a clear, class-compatible spatial
        match.  The cache key includes the stream session and event keyframe.
        """

        event_alerts = _event_alerts_by_object(analysis)
        relevant_objects = tuple(
            item for item in analysis.objects if item.object_id in event_alerts
        )
        if not relevant_objects:
            return OverlayFusionPlan((), (), (), ())

        rich = tuple(rich_objects)
        if not rich:
            return OverlayFusionPlan(
                (),
                (),
                tuple(item.object_id for item in relevant_objects),
                tuple(
                    ObjectAlerts(item.object_id, event_alerts[item.object_id])
                    for item in relevant_objects
                ),
            )

        anchor_id = association_message_id or analysis.message_id
        with self._lock:
            assignments = self._assign_locked(
                analysis.stream_session_id,
                anchor_id,
                relevant_objects,
                rich,
                event_alerts,
            )

        alerts_by_track: dict[str, list[RichAlert]] = {}
        matched_ids: list[str] = []
        unmatched_ids: list[str] = []
        unmatched_alerts: list[ObjectAlerts] = []
        for event_index, event_object in enumerate(relevant_objects):
            object_matched = False
            object_unmatched: list[RichAlert] = []
            for alert in event_alerts[event_object.object_id]:
                rich_index = assignments.get((event_index, alert.event_label))
                if rich_index is None:
                    object_unmatched.append(alert)
                    continue
                object_matched = True
                track_id = rich[rich_index].track_id
                badges = alerts_by_track.setdefault(track_id, [])
                previous = next(
                    (
                        index
                        for index, existing in enumerate(badges)
                        if existing.event_label == alert.event_label
                    ),
                    None,
                )
                if previous is None:
                    badges.append(alert)
                elif badges[previous].confidence < alert.confidence:
                    badges[previous] = alert
            if object_matched:
                matched_ids.append(event_object.object_id)
            if object_unmatched:
                unmatched_ids.append(event_object.object_id)
                unmatched_alerts.append(
                    ObjectAlerts(event_object.object_id, tuple(object_unmatched))
                )

        track_alerts = tuple(
            TrackAlerts(
                item.track_id,
                tuple(alerts_by_track[item.track_id]),
            )
            for item in rich
            if item.track_id in alerts_by_track
        )
        return OverlayFusionPlan(
            track_alerts,
            tuple(matched_ids),
            tuple(unmatched_ids),
            tuple(unmatched_alerts),
        )

    def _assign_locked(
        self,
        session_id: str,
        anchor_id: str,
        event_objects: tuple[GroundedObject, ...],
        rich_objects: tuple[RichLabelObject, ...],
        event_alerts: dict[str, tuple[RichAlert, ...]],
    ) -> dict[tuple[int, str], int]:
        rich_by_track = {
            item.track_id: index for index, item in enumerate(rich_objects)
        }
        assignments: dict[tuple[int, str], int] = {}
        occupied: set[tuple[str, int]] = set()

        for event_index, event_object in enumerate(event_objects):
            for alert in event_alerts[event_object.object_id]:
                assignment_key = (event_index, alert.event_label)
                cache_key = (
                    session_id,
                    anchor_id,
                    event_object.object_id,
                    alert.event_label,
                )
                track_id = self._associations.get(cache_key)
                rich_index = (
                    rich_by_track.get(track_id) if track_id is not None else None
                )
                occupancy_key = (alert.event_label, rich_index)
                if rich_index is None or occupancy_key in occupied:
                    continue
                if not _target_classes_compatible(
                    event_object,
                    rich_objects[rich_index],
                ):
                    continue
                assignments[assignment_key] = rich_index
                occupied.add((alert.event_label, rich_index))
                self._associations.move_to_end(cache_key)

        candidates: dict[tuple[int, str], list[tuple[float, int]]] = {}
        for event_index, event_object in enumerate(event_objects):
            for alert in event_alerts[event_object.object_id]:
                assignment_key = (event_index, alert.event_label)
                if assignment_key in assignments:
                    continue
                matches: list[tuple[float, int]] = []
                for rich_index, rich_object in enumerate(rich_objects):
                    if (alert.event_label, rich_index) in occupied:
                        continue
                    score = _association_score(event_object, rich_object)
                    if score >= _MIN_MATCH_SCORE:
                        matches.append((score, rich_index))
                matches.sort(key=lambda item: item[0], reverse=True)
                if matches:
                    candidates[assignment_key] = matches

        proposals: dict[tuple[int, str], tuple[float, int]] = {}
        for assignment_key, matches in candidates.items():
            best_score, best_rich = matches[0]
            if (
                len(matches) > 1
                and best_score - matches[1][0] < _AMBIGUITY_MARGIN
            ):
                continue
            proposals[assignment_key] = (best_score, best_rich)

        proposals_by_slot: dict[
            tuple[str, int], list[tuple[float, tuple[int, str]]]
        ] = {}
        for assignment_key, (score, rich_index) in proposals.items():
            slot = (assignment_key[1], rich_index)
            proposals_by_slot.setdefault(slot, []).append((score, assignment_key))

        for slot, slot_proposals in proposals_by_slot.items():
            slot_proposals.sort(key=lambda item: item[0], reverse=True)
            best_score, assignment_key = slot_proposals[0]
            if (
                len(slot_proposals) > 1
                and best_score - slot_proposals[1][0] < _AMBIGUITY_MARGIN
            ):
                continue
            event_index, event_label = assignment_key
            rich_index = slot[1]
            assignments[assignment_key] = rich_index
            occupied.add(slot)
            cache_key = (
                session_id,
                anchor_id,
                event_objects[event_index].object_id,
                event_label,
            )
            self._associations[cache_key] = rich_objects[rich_index].track_id
            self._associations.move_to_end(cache_key)

        while len(self._associations) > self._max_cached_associations:
            self._associations.popitem(last=False)
        return assignments


def _event_alerts_by_object(
    analysis: AnalysisResult,
) -> dict[str, tuple[RichAlert, ...]]:
    objects_by_id = {item.object_id: item for item in analysis.objects}
    alerts: dict[str, list[RichAlert]] = {}
    for judgment in analysis.judgments:
        if not judgment.detected:
            continue
        for object_id in judgment.object_ids:
            event_object = objects_by_id.get(object_id)
            if event_object is None or not _event_targets_object(
                judgment.event_label,
                event_object,
            ):
                continue
            object_alerts = alerts.setdefault(object_id, [])
            existing = next(
                (
                    index
                    for index, alert in enumerate(object_alerts)
                    if alert.event_label == judgment.event_label
                ),
                None,
            )
            next_alert = RichAlert(judgment.event_label, judgment.confidence)
            if existing is None:
                object_alerts.append(next_alert)
            elif object_alerts[existing].confidence < judgment.confidence:
                object_alerts[existing] = next_alert
    return {key: tuple(value) for key, value in alerts.items()}


def _event_targets_object(event_label: str, event_object: GroundedObject) -> bool:
    event = "".join(
        character
        for character in str(event_label).casefold()
        if character.isalnum()
    )
    is_person = _normalized_label(event_object.label) == "person"
    if event in _PERSON_TARGET_EVENTS:
        return is_person
    if event in _OBJECT_TARGET_EVENTS:
        return not is_person
    return True


def _normalized_label(value: str) -> str:
    normalized = "".join(
        character for character in value.casefold() if character.isalnum()
    )
    if normalized in _PERSON_LABELS:
        return "person"
    return _LABEL_ALIASES.get(normalized, normalized)


def _labels_compatible(left: str, right: str) -> bool:
    return _normalized_label(left) == _normalized_label(right)


def _target_classes_compatible(
    event_object: GroundedObject,
    rich_object: RichLabelObject,
) -> bool:
    event_is_person = _normalized_label(event_object.label) == "person"
    rich_is_person = _normalized_label(rich_object.label) == "person"
    return event_is_person == rich_is_person


def _association_score(
    event_object: GroundedObject,
    rich_object: RichLabelObject,
) -> float:
    if not _target_classes_compatible(event_object, rich_object):
        return 0.0
    raw_event_label = "".join(
        character
        for character in event_object.label.casefold()
        if character.isalnum()
    )
    if raw_event_label in _GROUP_PERSON_LABELS:
        return 0.0
    event_is_person = _normalized_label(event_object.label) == "person"
    if event_is_person and _is_face_person_label(event_object.label):
        return _nested_person_match_score(event_object.bbox, rich_object.bbox)
    if event_is_person or _labels_compatible(
        event_object.label,
        rich_object.label,
    ):
        return _match_score(event_object.bbox, rich_object.bbox)
    return _cross_label_object_match_score(event_object.bbox, rich_object.bbox)


def _is_face_person_label(value: str) -> bool:
    normalized = "".join(
        character for character in value.casefold() if character.isalnum()
    )
    return normalized in _FACE_PERSON_LABELS


@dataclass(frozen=True, slots=True)
class _OverlapMetrics:
    iou: float
    left_containment: float
    right_containment: float
    area_similarity: float
    center_affinity: float


def _overlap_metrics(
    left: NormalizedBBox,
    right: NormalizedBBox,
) -> _OverlapMetrics | None:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return None

    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    if union <= 0.0 or left_area <= 0.0 or right_area <= 0.0:
        return None
    left_center = ((left.x1 + left.x2) * 0.5, (left.y1 + left.y2) * 0.5)
    right_center = ((right.x1 + right.x2) * 0.5, (right.y1 + right.y2) * 0.5)
    center_distance = math.hypot(
        left_center[0] - right_center[0],
        left_center[1] - right_center[1],
    )
    return _OverlapMetrics(
        iou=intersection / union,
        left_containment=intersection / left_area,
        right_containment=intersection / right_area,
        area_similarity=min(left_area, right_area) / max(left_area, right_area),
        center_affinity=max(0.0, 1.0 - center_distance / 0.35),
    )


def _match_score(left: NormalizedBBox, right: NormalizedBBox) -> float:
    metrics = _overlap_metrics(left, right)
    if metrics is None:
        return 0.0
    containment = max(metrics.left_containment, metrics.right_containment)
    if metrics.iou < 0.18 and (
        containment < 0.65 or metrics.area_similarity < 0.3
    ):
        return 0.0
    return (
        metrics.iou * 0.62
        + containment * 0.18
        + metrics.area_similarity * 0.12
        + metrics.center_affinity * 0.08
    )


def _nested_person_match_score(
    face: NormalizedBBox,
    person: NormalizedBBox,
) -> float:
    metrics = _overlap_metrics(face, person)
    if metrics is None or metrics.left_containment < 0.85:
        return 0.0
    person_width = person.x2 - person.x1
    person_height = person.y2 - person.y1
    face_center_x = (face.x1 + face.x2) * 0.5
    face_center_y = (face.y1 + face.y2) * 0.5
    relative_x = abs(face_center_x - (person.x1 + person.x2) * 0.5) / person_width
    relative_y = (face_center_y - person.y1) / person_height
    if relative_x > 0.45 or not 0.0 <= relative_y <= 0.68:
        return 0.0
    return (
        metrics.left_containment * 0.58
        + metrics.center_affinity * 0.25
        + metrics.iou * 0.12
        + metrics.area_similarity * 0.05
    )


def _cross_label_object_match_score(
    event_bbox: NormalizedBBox,
    rich_bbox: NormalizedBBox,
) -> float:
    metrics = _overlap_metrics(event_bbox, rich_bbox)
    if metrics is None:
        return 0.0
    containment = max(metrics.left_containment, metrics.right_containment)
    if metrics.iou < 0.45 and (
        containment < 0.85 or metrics.area_similarity < 0.25
    ):
        return 0.0
    return (
        metrics.iou * 0.55
        + containment * 0.20
        + metrics.area_similarity * 0.15
        + metrics.center_affinity * 0.10
    )
