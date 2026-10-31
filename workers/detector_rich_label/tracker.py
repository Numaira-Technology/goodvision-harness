"""Detector-agnostic IoU tracker that assigns stable track_ids to per-frame boxes.

Copied (unchanged logic) from workers/paddle_rich_label so the YOLO / RF-DETR
worker produces the exact same track_id behaviour the harness expects. The
harness never assigns track_ids itself (see rich_labeling.RichLabelObject,
which requires track_id and only associates across frames by it), so any
single-frame detector must be wrapped by this tracker.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

PERSON_LABELS = frozenset({"person", "pedestrian"})
MaskPolygons = tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # pixel x1,y1,x2,y2
    mask_polygons: MaskPolygons = ()  # source-normalized polygons, observed frame only


@dataclass(slots=True)
class Track:
    track_id: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    last_frame_sequence: int
    last_update_tick: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    last_attribute_frame: int | None = None
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    motion_updates: int = 0
    last_observed_timestamp_seconds: float | None = None
    mask_polygons: MaskPolygons = ()


class IoUTracker:
    """Small deterministic single-camera tracker around per-frame detections."""

    def __init__(
        self,
        *,
        match_iou: float = 0.3,
        max_idle_frames: int = 20,
        visible_idle_frames: int = 1,
        max_idle_seconds: float = 3.0,
    ) -> None:
        if not 0.0 < match_iou <= 1.0:
            raise ValueError("match_iou must be between 0 and 1")
        if max_idle_frames < 1:
            raise ValueError("max_idle_frames must be positive")
        if visible_idle_frames < 0:
            raise ValueError("visible_idle_frames cannot be negative")
        if not math.isfinite(max_idle_seconds) or max_idle_seconds <= 0:
            raise ValueError("max_idle_seconds must be finite and positive")
        self.match_iou = float(match_iou)
        self.max_idle_frames = int(max_idle_frames)
        self.visible_idle_frames = int(visible_idle_frames)
        self.max_idle_seconds = float(max_idle_seconds)
        self._tracks: dict[str, Track] = {}
        self._next_id = 1
        self._update_tick = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._update_tick = 0

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_sequence: int,
        timestamp_seconds: float | None = None,
    ) -> tuple[Track, ...]:
        if timestamp_seconds is not None and (
            not math.isfinite(timestamp_seconds) or timestamp_seconds < 0
        ):
            raise ValueError("timestamp_seconds must be finite and non-negative")
        self._update_tick += 1
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if (
                0 <= timestamp_seconds - track.last_observed_timestamp_seconds
                <= self.max_idle_seconds
                if timestamp_seconds is not None
                and track.last_observed_timestamp_seconds is not None
                else self._update_tick - track.last_update_tick <= self.max_idle_frames
            )
        }
        candidates: list[tuple[float, float, float, str, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                if track.label != detection.label:
                    continue
                association = _association_score(
                    track, detection,
                    frame_sequence=frame_sequence, match_iou=self.match_iou,
                )
                if association is None:
                    continue
                score, predicted_overlap, normalized_distance = association
                candidates.append(
                    (score, predicted_overlap, normalized_distance, track_id,
                     detection_index)
                )
        candidates.sort(
            key=lambda item: (-item[0], -item[1], item[2], item[3], item[4])
        )
        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, Track] = {}
        for _score, _overlap, _distance, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = self._tracks[track_id]

        current: list[Track] = []
        for detection_index, detection in enumerate(detections):
            track = assignments.get(detection_index)
            if track is not None:
                _update_track_motion(track, detection, frame_sequence=frame_sequence)
                track.last_update_tick = self._update_tick
            else:
                prefix = "P" if detection.label in PERSON_LABELS else "O"
                track_id = f"{prefix}{self._next_id:04d}"
                self._next_id += 1
                track = Track(
                    track_id=track_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    last_frame_sequence=frame_sequence,
                    last_update_tick=self._update_tick,
                )
                self._tracks[track_id] = track
            track.last_observed_timestamp_seconds = timestamp_seconds
            track.mask_polygons = detection.mask_polygons
            current.append(track)
        visible = {track.track_id: track for track in current}
        for track_id, track in self._tracks.items():
            if (
                track_id not in visible
                and self._update_tick - track.last_update_tick
                <= self.visible_idle_frames
            ):
                visible[track_id] = track
        return tuple(sorted(visible.values(), key=lambda item: item.track_id))


def valid_pixel_bbox(bbox: Sequence[float], width: int, height: int) -> bool:
    x1, y1, x2, y2 = bbox
    return (
        all(math.isfinite(value) for value in bbox)
        and 0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    )


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _association_score(
    track: Track, detection: Detection, *, frame_sequence: int, match_iou: float,
) -> tuple[float, float, float] | None:
    gap = max(1, frame_sequence - track.last_frame_sequence)
    predicted_bbox = _predicted_bbox(track, frame_sequence)
    predicted_overlap = _iou(predicted_bbox, detection.bbox)
    previous_overlap = _iou(track.bbox, detection.bbox)
    predicted_center = _bbox_center(predicted_bbox)
    detection_center = _bbox_center(detection.bbox)
    center_distance = math.hypot(
        predicted_center[0] - detection_center[0],
        predicted_center[1] - detection_center[1],
    )
    distance_scale = max(
        _bbox_diagonal(predicted_bbox), _bbox_diagonal(detection.bbox), 1.0,
    )
    normalized_distance = center_distance / distance_scale
    predicted_area = _bbox_area(predicted_bbox)
    detection_area = _bbox_area(detection.bbox)
    size_similarity = (
        min(predicted_area, detection_area) / max(predicted_area, detection_area)
        if predicted_area > 0 and detection_area > 0 else 0.0
    )
    center_limit = min(2.5, 1.25 + (0.12 * max(0, gap - 1)))
    if not (
        predicted_overlap >= max(0.05, match_iou * 0.5)
        or previous_overlap >= match_iou
        or (normalized_distance <= center_limit and size_similarity >= 0.25)
    ):
        return None
    proximity = max(0.0, 1.0 - (normalized_distance / center_limit))
    direction = _direction_alignment(track, detection.bbox)
    score = (
        (3.5 * predicted_overlap) + (0.5 * previous_overlap) + (1.5 * proximity)
        + (0.35 * size_similarity)
        + ((0.8 * direction) if track.motion_updates else 0.0)
    )
    return score, predicted_overlap, normalized_distance


def _predicted_bbox(
    track: Track, frame_sequence: int,
) -> tuple[float, float, float, float]:
    gap = max(0, frame_sequence - track.last_frame_sequence)
    offset_x = track.velocity_x * gap
    offset_y = track.velocity_y * gap
    return (
        track.bbox[0] + offset_x, track.bbox[1] + offset_y,
        track.bbox[2] + offset_x, track.bbox[3] + offset_y,
    )


def _update_track_motion(
    track: Track, detection: Detection, *, frame_sequence: int,
) -> None:
    gap = frame_sequence - track.last_frame_sequence
    if gap > 0:
        old_center = _bbox_center(track.bbox)
        new_center = _bbox_center(detection.bbox)
        observed_x = (new_center[0] - old_center[0]) / gap
        observed_y = (new_center[1] - old_center[1]) / gap
        if track.motion_updates:
            track.velocity_x = (0.35 * track.velocity_x) + (0.65 * observed_x)
            track.velocity_y = (0.35 * track.velocity_y) + (0.65 * observed_y)
        else:
            track.velocity_x = observed_x
            track.velocity_y = observed_y
        track.motion_updates += 1
    track.confidence = detection.confidence
    track.bbox = detection.bbox
    track.last_frame_sequence = frame_sequence


def _direction_alignment(
    track: Track, detection_bbox: tuple[float, float, float, float],
) -> float:
    if not track.motion_updates:
        return 0.0
    current_center = _bbox_center(track.bbox)
    detection_center = _bbox_center(detection_bbox)
    observed_x = detection_center[0] - current_center[0]
    observed_y = detection_center[1] - current_center[1]
    expected_norm = math.hypot(track.velocity_x, track.velocity_y)
    observed_norm = math.hypot(observed_x, observed_y)
    if expected_norm <= 1e-9 or observed_norm <= 1e-9:
        return 0.5
    cosine = (
        (track.velocity_x * observed_x) + (track.velocity_y * observed_y)
    ) / (expected_norm * observed_norm)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _bbox_center(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_diagonal(bbox: tuple[float, float, float, float]) -> float:
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0
