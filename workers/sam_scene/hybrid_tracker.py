"""Bounded person association; model-independent, source-time observations only.

Predictions are used to associate detections, never emitted as observations.
The color descriptor is an association aid, not a person identity embedding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]


class TrackerCapacityError(RuntimeError):
    """A new session is required before allocating further track identifiers."""


@dataclass(frozen=True, slots=True)
class PersonDetection:
    bbox: BBox
    confidence: float
    appearance: tuple[float, ...] = ()

    def __post_init__(self):
        if (
            len(self.bbox) != 4
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.bbox)
            or self.bbox[0] >= self.bbox[2]
            or self.bbox[1] >= self.bbox[3]
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
            or len(self.appearance) > 64
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.appearance)
        ):
            raise ValueError("invalid person detection")


def overlap(a: BBox, b: BBox) -> float:
    area = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0.0


def appearance_similarity(a, b) -> float | None:
    if not a or len(a) != len(b):
        return None
    # Histograms for each body half/channel have unit mass. Normalize by the
    # number of histograms so the intersection lies within [0, 1].
    scale = max(sum(a), sum(b))
    return sum(min(x, y) for x, y in zip(a, b, strict=True)) / scale if scale else 0.0


def describe_person(image, bbox: BBox) -> tuple[float, ...]:
    import numpy as np

    width, height = image.size
    region = image.crop(
        (
            int(bbox[0] * width),
            int(bbox[1] * height),
            max(int(bbox[0] * width) + 1, int(bbox[2] * width)),
            max(int(bbox[1] * height) + 1, int(bbox[3] * height)),
        )
    )
    try:
        with region.resize((24, 48)) as resized:
            pixels = np.asarray(resized)
            result = []
            for half in (pixels[:24], pixels[24:]):
                for channel in range(3):
                    hist = np.bincount(
                        (half[:, :, channel].reshape(-1) // 32), minlength=8
                    )
                    result.extend((hist / hist.sum()).tolist())
            return tuple(result)
    finally:
        region.close()


@dataclass
class Track:
    object_id: int
    detection: PersonDetection
    timestamp: float
    frame_sequence: int
    velocity: BBox = (0.0, 0.0, 0.0, 0.0)
    observations: int = 1
    calibration: dict | None = None

    def predict(self, timestamp: float) -> BBox:
        dt = min(max(0.0, timestamp - self.timestamp), 0.75)
        return tuple(
            min(1.0, max(0.0, p + dt * v))
            for p, v in zip(self.detection.bbox, self.velocity, strict=True)
        )


@dataclass(frozen=True, slots=True)
class ObservedPerson:
    object_id: int
    detection: PersonDetection
    confirmed: bool
    calibration: dict | None = None


@dataclass(frozen=True, slots=True)
class CalibrationSeed:
    detection: PersonDetection
    frame_sequence: int
    timestamp: float


@dataclass
class PersonTracker:
    max_objects: int = 16
    max_ids: int = 2048
    ttl_seconds: float = 2.0
    high_threshold: float = 0.45
    low_threshold: float = 0.20
    ambiguity_margin: float = 0.12
    tracks: dict[int, Track] = field(default_factory=dict)
    seeds: list[CalibrationSeed] = field(default_factory=list)
    next_id: int = 0
    last_timestamp: float = -1.0
    last_sequence: int = -1
    ambiguous_count: int = 0
    seed_matches: int = 0

    def _score(self, track, detection, timestamp):
        predicted = track.predict(timestamp)
        iou = overlap(predicted, detection.bbox)
        similarity = appearance_similarity(
            track.detection.appearance, detection.appearance
        )
        if similarity is not None and similarity < 0.35:
            return 0.0
        if iou < 0.10:
            return 0.0
        return iou if similarity is None else 0.75 * iou + 0.25 * similarity

    def _associate(self, tracks, detections, timestamp):
        scores = {
            (tid, index): self._score(track, detection, timestamp)
            for tid, track in tracks.items()
            for index, detection in detections.items()
        }
        matches = []
        for index in detections:
            candidates = sorted(
                ((score, tid) for (tid, di), score in scores.items() if di == index),
                reverse=True,
            )
            if not candidates or candidates[0][0] < 0.30:
                continue
            best, tid = candidates[0]
            rivals = sorted(
                (
                    score
                    for (ti, di), score in scores.items()
                    if ti == tid and di != index
                ),
                reverse=True,
            )
            if (
                len(candidates) > 1 and best - candidates[1][0] < self.ambiguity_margin
            ) or (rivals and best - rivals[0] < self.ambiguity_margin):
                self.ambiguous_count += 1
                continue
            matches.append((tid, index))
        return matches

    def update(self, detections, *, timestamp, frame_sequence):
        if (
            not math.isfinite(timestamp)
            or timestamp < 0
            or timestamp <= self.last_timestamp
            or type(frame_sequence) is not int
            or frame_sequence <= self.last_sequence
        ):
            raise ValueError("person tracker source must strictly increase")
        self.last_timestamp, self.last_sequence = timestamp, frame_sequence
        self.tracks = {
            tid: track
            for tid, track in self.tracks.items()
            if timestamp - track.timestamp <= self.ttl_seconds
        }
        self.seeds = [
            seed
            for seed in self.seeds
            if 0 <= timestamp - seed.timestamp <= self.ttl_seconds
        ]
        accepted = sorted(
            (d for d in detections if d.confidence >= self.low_threshold),
            key=lambda detection: detection.confidence,
            reverse=True,
        )[: self.max_objects]
        high = {
            i: d for i, d in enumerate(accepted) if d.confidence >= self.high_threshold
        }
        matches = self._associate(self.tracks, high, timestamp)
        matched_tracks = {tid for tid, _ in matches}
        matched_detections = {index for _, index in matches}
        # Low scores may continue tracks corroborated by repeated detector
        # evidence or same-source SAM. They cannot independently create people.
        remaining = {
            tid: track
            for tid, track in self.tracks.items()
            if tid not in matched_tracks
            and (track.observations >= 2 or self._sam_confirmed(track))
        }
        low = {
            i: d
            for i, d in enumerate(accepted)
            if i not in high and i not in matched_detections
        }
        low_matches = self._associate(remaining, low, timestamp)
        matches.extend(low_matches)
        matched_detections.update(index for _, index in low_matches)
        result = []
        for tid, index in matches:
            track = self.tracks[tid]
            detection = accepted[index]
            dt = timestamp - track.timestamp
            track.velocity = tuple(
                max(-1.0, min(1.0, 0.65 * ((new - old) / dt) + 0.35 * previous))
                for new, old, previous in zip(
                    detection.bbox, track.detection.bbox, track.velocity, strict=True
                )
            )
            track.detection = detection
            track.timestamp, track.frame_sequence = timestamp, frame_sequence
            track.observations += 1
            result.append(self._observed(track))
        for index, detection in enumerate(accepted):
            if index in matched_detections:
                continue
            seed = self._seed_for(
                detection,
                competitors=tuple(
                    other
                    for di, other in enumerate(accepted)
                    if di != index and di not in matched_detections
                ),
            )
            if detection.confidence < self.high_threshold and seed is None:
                continue
            if len(self.tracks) >= self.max_objects * 4:
                oldest = min(self.tracks, key=lambda tid: self.tracks[tid].timestamp)
                del self.tracks[oldest]
            if self.next_id >= self.max_ids:
                raise TrackerCapacityError("hybrid person identifier budget exhausted")
            track = Track(self.next_id, detection, timestamp, frame_sequence)
            self.next_id += 1
            if seed is not None:
                track.calibration = self._calibration(
                    "confirmed", seed.frame_sequence, seed.timestamp, seed.detection
                )
                self.seeds.remove(seed)
                self.seed_matches += 1
            self.tracks[track.object_id] = track
            result.append(self._observed(track))
        return tuple(sorted(result, key=lambda item: item.object_id))

    def _seed_for(self, detection, *, competitors=()):
        eligible = []
        for seed in self.seeds:
            similarity = appearance_similarity(
                seed.detection.appearance, detection.appearance
            )
            iou = overlap(seed.detection.bbox, detection.bbox)
            if iou >= 0.5 and (similarity is None or similarity >= 0.65):
                eligible.append((iou, seed))
        eligible.sort(key=lambda item: item[0], reverse=True)
        if not eligible or (
            len(eligible) > 1
            and eligible[0][0] - eligible[1][0] < self.ambiguity_margin
        ):
            return None
        best, seed = eligible[0]
        for other in competitors:
            similarity = appearance_similarity(
                seed.detection.appearance, other.appearance
            )
            rival = overlap(seed.detection.bbox, other.bbox)
            if (
                rival >= 0.5
                and (similarity is None or similarity >= 0.65)
                and best - rival < self.ambiguity_margin
            ):
                return None
        return seed

    @staticmethod
    def _sam_confirmed(track):
        return (
            track.calibration is not None and track.calibration["status"] == "confirmed"
        )

    def _observed(self, track):
        uncertain = (
            track.calibration is not None and track.calibration["status"] == "uncertain"
        )
        return ObservedPerson(
            track.object_id,
            track.detection,
            not uncertain and (track.observations >= 2 or self._sam_confirmed(track)),
            dict(track.calibration) if track.calibration is not None else None,
        )

    @staticmethod
    def _calibration(status, sequence, timestamp, detection):
        return {
            "status": status,
            "frame_sequence": sequence,
            "timestamp_seconds": timestamp,
            "confidence": detection.confidence,
            "bbox": dict(zip(("x1", "y1", "x2", "y2"), detection.bbox, strict=True)),
        }

    def calibrate(self, historical, sam_detections, *, frame_sequence, timestamp):
        """Confirm by same-source mutual geometry; keep current geometry untouched."""
        historical = tuple(historical)
        sam_detections = tuple(
            detection for detection in sam_detections if detection.confidence >= 0.5
        )[: self.max_objects]
        used = set()
        confirmed = uncertain = 0
        for observed in historical:
            ranked = sorted(
                (
                    (overlap(observed.detection.bbox, d.bbox), i)
                    for i, d in enumerate(sam_detections)
                ),
                reverse=True,
            )
            if not ranked or ranked[0][0] < 0.45:
                continue  # Absence from SAM is not proof of a detector error.
            best, index = ranked[0]
            reverse = sorted(
                (
                    overlap(other.detection.bbox, sam_detections[index].bbox)
                    for other in historical
                    if other.object_id != observed.object_id
                ),
                reverse=True,
            )
            unambiguous = (
                len(ranked) == 1 or best - ranked[1][0] >= self.ambiguity_margin
            ) and (not reverse or best - reverse[0] >= self.ambiguity_margin)
            track = self.tracks.get(observed.object_id)
            if track is None or track.timestamp < timestamp:
                continue
            # Never roll a newer calibration backward.
            if track.calibration and track.calibration["timestamp_seconds"] > timestamp:
                continue
            status = "confirmed" if unambiguous else "uncertain"
            track.calibration = self._calibration(
                status, frame_sequence, timestamp, sam_detections[index]
            )
            if unambiguous:
                used.add(index)
                confirmed += 1
            else:
                uncertain += 1
        discovered = 0
        for index, detection in enumerate(sam_detections):
            # SAM-only detections are historical candidates, not present boxes.
            # Ambiguous overlaps are not counted as newly discovered people.
            if index in used or any(
                overlap(detection.bbox, h.detection.bbox) > 0.2 for h in historical
            ):
                continue
            if self.last_timestamp - timestamp <= self.ttl_seconds:
                self.seeds.append(CalibrationSeed(detection, frame_sequence, timestamp))
                discovered += 1
        self.seeds = self.seeds[-self.max_objects :]
        return {
            "confirmed": confirmed,
            "uncertain": uncertain,
            "discovered": discovered,
        }
