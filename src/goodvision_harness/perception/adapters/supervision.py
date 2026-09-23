"""The only live-path dependency boundary for Supervision's Detections API."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ...contracts import NormalizedBBox
from ...observations.normalization import is_observed
from ...rich_labeling import RichLabelObject, RichLabelSnapshot

_INDEX = "harness_index"


def supervision() -> Any:
    try:
        import supervision as sv
    except ImportError as exc:
        raise RuntimeError(
            "Supervision experiment requires: pip install '.[supervision]'"
        ) from exc
    return sv


def resolution(value: tuple[int, int]) -> tuple[int, int]:
    if (
        len(value) != 2
        or any(type(v) is not int or not 1 <= v <= 8192 for v in value)
        or value[0] * value[1] > 16 * 1024 * 1024
    ):
        raise ValueError("resolution must contain two positive bounded pixel sizes")
    return value


class SupervisionAdapter:
    """Bounded session-local string IDs and class vocabulary for CV primitives.

    Original objects stay outside sv.data; selection/NMS carries a row index so
    identity, masks, unknown confidence and provenance survive round trips.
    """

    def __init__(self, *, max_tracks: int = 2048) -> None:
        if type(max_tracks) is not int or not 1 <= max_tracks <= 65536:
            raise ValueError("max_tracks must be within 1..65536")
        self.max_tracks = max_tracks
        self._tracks: dict[str, int] = {}
        self._classes: dict[str, int] = {}

    def to_detections(
        self,
        snapshot: RichLabelSnapshot,
        *,
        resolution_wh: tuple[int, int],
        observed_only: bool = True,
    ):
        sv = supervision()
        resolution(resolution_wh)
        if len(snapshot.objects) > 128:
            raise ValueError("at most 128 detections per source frame")
        rows = [
            (i, obj)
            for i, obj in enumerate(snapshot.objects)
            if not observed_only or is_observed(obj, snapshot)
        ]
        new_tracks = {obj.track_id for _, obj in rows} - self._tracks.keys()
        new_classes = {obj.label for _, obj in rows} - self._classes.keys()
        if len(self._tracks) + len(new_tracks) > self.max_tracks:
            raise ValueError("Supervision track capacity exceeded; start a new epoch")
        if len(self._classes) + len(new_classes) > 256:
            raise ValueError("Supervision class capacity exceeded")
        for key in sorted(new_tracks):
            self._tracks[key] = len(self._tracks)
        for key in sorted(new_classes):
            self._classes[key] = len(self._classes)
        boxes = np.asarray(
            [(obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2) for _, obj in rows],
            dtype=float,
        ).reshape(-1, 4)
        return sv.Detections(
            xyxy=sv.denormalize_boxes(boxes, resolution_wh=resolution_wh),
            confidence=np.array(
                [
                    obj.confidence if obj.confidence is not None else 0.0
                    for _, obj in rows
                ],
                dtype=float,
            ),
            class_id=np.array([self._classes[obj.label] for _, obj in rows], dtype=int),
            tracker_id=np.array(
                [self._tracks[obj.track_id] for _, obj in rows], dtype=int
            ),
            data={
                _INDEX: np.array([i for i, _ in rows], dtype=int),
                "class_name": np.array([obj.label for _, obj in rows], dtype=str),
            },
        )

    @staticmethod
    def select_snapshot(snapshot: RichLabelSnapshot, detections) -> RichLabelSnapshot:
        """Select original measured objects; never write smoothed geometry back."""
        return replace(
            snapshot,
            objects=tuple(
                snapshot.objects[int(index)] for index in detections.data[_INDEX]
            ),
        )


def from_detections(
    detections,
    *,
    resolution_wh: tuple[int, int],
    stream_session_id: str,
    frame_sequence: int,
    timestamp_seconds: float,
    classes: tuple[str, ...],
    track_namespace: str,
) -> RichLabelSnapshot:
    """Adapt any upstream sv.Detections (YOLO/SAM/VLM/Trackers) to the harness.

    Without a confirmed tracker ID, IDs are frame-local. Only the measured boxes
    from the current inference may be supplied here, never tracker predictions.
    """
    sv = supervision()
    width, height = resolution(resolution_wh)
    if not track_namespace or len(track_namespace) > 128 or len(detections) > 128:
        raise ValueError("a bounded track namespace and at most 128 rows are required")
    boxes = np.asarray(detections.xyxy, dtype=float)
    if not np.isfinite(boxes).all():
        raise ValueError("detections contain non-finite geometry")
    boxes = sv.clip_boxes(boxes.copy(), resolution_wh) / (width, height, width, height)
    objects = []
    for index, box in enumerate(boxes):
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("detection has no visible source area")
        class_id = (
            int(detections.class_id[index]) if detections.class_id is not None else 0
        )
        if not 0 <= class_id < len(classes):
            raise ValueError("detection class is outside the declared vocabulary")
        tracker_id = (
            int(detections.tracker_id[index])
            if detections.tracker_id is not None
            else -1
        )
        track_id = (
            f"{track_namespace}:{tracker_id}"
            if tracker_id >= 0
            else f"{track_namespace}:frame:{frame_sequence}:{index}"
        )
        attributes = {
            "perception_adapter": "supervision",
            "tracker_confirmed": tracker_id >= 0,
        }
        if detections.mask is not None:
            mask = np.asarray(detections.mask[index])
            if mask.shape != (height, width):
                raise ValueError("mask resolution must match the source frame")
            polygons = []
            # Keep bounded outer contours; the harness contract has no holes.
            for polygon in sv.mask_to_polygons(mask.astype(bool))[:16]:
                if len(polygon) > 64:
                    polygon = sv.approximate_polygon(polygon, 1 - 64 / len(polygon))
                if 3 <= len(polygon) <= 64:
                    polygons.append((polygon / (width, height)).tolist())
            if polygons:
                attributes["mask_contours"] = {
                    "coordinate_system": "normalized",
                    "source_frame_sequence": frame_sequence,
                    "source_timestamp_seconds": timestamp_seconds,
                    "polygons": polygons,
                }
        objects.append(
            RichLabelObject(
                track_id=track_id,
                label=classes[class_id],
                bbox=NormalizedBBox(*box),
                confidence=(
                    float(detections.confidence[index])
                    if detections.confidence is not None
                    else None
                ),
                attributes=attributes,
                observation_state="observed",
                last_observed_timestamp_seconds=timestamp_seconds,
                last_observed_frame_sequence=frame_sequence,
            )
        )
    return RichLabelSnapshot(
        stream_session_id, frame_sequence, timestamp_seconds, tuple(objects)
    )
