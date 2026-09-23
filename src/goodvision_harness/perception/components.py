"""Supervision primitives, scoped to one source/session and ordered by capture.

Filtering and NMS select measured rows. Smoothed geometry is display-only.
Polygon membership and line crossing are observations, never safety verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..observations.schema import Observation
from .adapters.supervision import SupervisionAdapter, resolution, supervision
from .settings import SupervisionSettings


@dataclass(frozen=True)
class PerceptionResult:
    observation: Observation
    detections: Any
    display_detections: Any
    geometry: dict
    resolution_wh: tuple[int, int]


class SupervisionPipeline:
    def __init__(
        self,
        *,
        source_id: str,
        stream_session_id: str,
        settings: SupervisionSettings,
        resolution_wh: tuple[int, int] = (1000, 1000),
    ) -> None:
        self.sv = supervision()
        self.source_id = source_id
        self.stream_session_id = stream_session_id
        self.settings = settings
        self.resolution_wh = resolution(resolution_wh)
        self._last = (-1.0, -1)
        self._native_epoch: tuple = ()
        self._generation = 0
        self._reset()

    def _reset(self) -> None:
        sv = self.sv
        self._adapter = SupervisionAdapter(max_tracks=self.settings.max_tracks)
        self._smoother = sv.DetectionsSmoother(length=self.settings.smoothing_length)
        self._zones = {
            zone.name: sv.PolygonZone(
                polygon=np.rint(np.array(zone.points) * self.resolution_wh).astype(int),
                triggering_anchors=(sv.Position.BOTTOM_CENTER,),
            )
            for zone in self.settings.zones
        }
        self._lines = {}
        for line in self.settings.lines:
            points = np.array(line.points) * self.resolution_wh
            self._lines[line.name] = sv.LineZone(
                start=sv.Point(*points[0]),
                end=sv.Point(*points[1]),
                triggering_anchors=(sv.Position.BOTTOM_CENTER,),
            )
        self._generation += 1

    def process(
        self, observation: Observation, *, preserve_tracks: bool = False
    ) -> PerceptionResult:
        snapshot = observation.snapshot
        if (
            observation.source_id != self.source_id
            or snapshot.stream_session_id != self.stream_session_id
        ):
            raise ValueError("observation belongs to another source/session")
        at = (snapshot.timestamp_seconds, snapshot.frame_sequence)
        if at[0] < self._last[0] or at[1] <= self._last[1]:
            raise ValueError("CV primitives require distinct source-ordered frames")
        epochs = tuple(
            sorted(
                {
                    (
                        obj.attributes["native_track"].get("backend"),
                        obj.attributes["native_track"].get("epoch"),
                    )
                    for obj in snapshot.objects
                    if obj.attributes.get("native_track")
                }
            )
        )
        reset_reason = None
        if self._last[1] >= 0 and at[0] - self._last[0] > self.settings.max_gap_seconds:
            reset_reason = "source_gap"
        if epochs and self._native_epoch and epochs != self._native_epoch:
            reset_reason = "native_epoch"
        if reset_reason:
            self._reset()
        if epochs:
            self._native_epoch = epochs
        detections = self._adapter.to_detections(
            snapshot,
            resolution_wh=self.resolution_wh,
        )
        # Native IDs/masks and complete presence frames are authoritative; do not
        # suppress one actor as a duplicate or erase it with a confidence filter.
        if not preserve_tracks and not epochs:
            if self.settings.confidence_threshold is not None:
                detections = detections[
                    detections.confidence >= self.settings.confidence_threshold
                ]
            if self.settings.nms_threshold is not None:
                # Unknown confidence is preserved but cannot suppress a scored row.
                detections = detections.with_nms(self.settings.nms_threshold)
        selected = self._adapter.select_snapshot(snapshot, detections)
        subjects = [obj.track_id for obj in selected.objects]
        zones = {}
        for name, zone in self._zones.items():
            inside = zone.trigger(detections)
            zones[name] = {
                "count": int(inside.sum()),
                "track_ids": [
                    sid
                    for sid, present in zip(subjects, inside, strict=True)
                    if present
                ],
            }
        lines = {}
        for name, line in self._lines.items():
            entered, exited = line.trigger(detections)
            lines[name] = {
                "in_count": line.in_count,
                "out_count": line.out_count,
                "crossed_in": [
                    sid for sid, yes in zip(subjects, entered, strict=True) if yes
                ],
                "crossed_out": [
                    sid for sid, yes in zip(subjects, exited, strict=True) if yes
                ],
            }
        display = self._smoother.update_with_detections(detections)
        self._last = at
        return PerceptionResult(
            Observation(
                observation.source_id,
                selected,
                observation.model,
                observation.model_version,
                observation.inference_seconds,
            ),
            detections,
            display,
            {
                "source_id": self.source_id,
                "stream_session_id": self.stream_session_id,
                "frame_sequence": at[1],
                "timestamp_seconds": at[0],
                "generation": self._generation,
                "reset_reason": reset_reason,
                "zones": zones,
                "lines": lines,
                "observed_count": len(detections),
            },
            self.resolution_wh,
        )


class SupervisionRenderer:
    """Reusable optional debug/replay renderer; never mutates source images."""

    def __init__(self, *, heatmap: bool = False) -> None:
        self._heatmap_enabled = heatmap
        self._scope = None
        self._reset()

    def _reset(self):
        sv = supervision()
        self._boxes = sv.BoxAnnotator()
        self._labels = sv.LabelAnnotator()
        self._heatmap = sv.HeatMapAnnotator() if self._heatmap_enabled else None

    def annotate(self, frame, result: PerceptionResult):
        if (frame.shape[1], frame.shape[0]) != result.resolution_wh:
            raise ValueError("renderer needs the same source resolution as detections")
        scope = (
            result.observation.source_id,
            result.observation.snapshot.stream_session_id,
            result.resolution_wh,
            result.geometry["generation"],
        )
        if self._scope != scope:
            self._reset()
            self._scope = scope
        rendered = frame.copy()
        if self._heatmap is not None:
            rendered = self._heatmap.annotate(rendered, result.detections)
        rendered = self._boxes.annotate(rendered, result.display_detections)
        return self._labels.annotate(rendered, result.display_detections)
