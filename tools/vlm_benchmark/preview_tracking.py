"""Conservative, CPU-only tracking for the benchmark preview (not detections).

Seeds are VLM boxes on their original evidence image. Delayed seeds are replayed
through a bounded history; response-local IDs never become person identities.
No prediction is emitted after uncertainty, a discontinuity or an expired seed.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

WIDTH = 320
HISTORY_SECONDS = 8.0
MAX_GAP = 0.30
MAX_PEOPLE = 12


def gray_frame(jpeg):
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if frame is None or min(frame.shape) < 8:
        raise ValueError("invalid tracking image")
    height = max(8, round(frame.shape[0] * WIDTH / frame.shape[1]))
    if height > 640:
        raise ValueError("unsupported tracking aspect ratio")
    return cv2.resize(frame, (WIDTH, height), interpolation=cv2.INTER_AREA)


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union else 0.0


def patch(frame, box):
    x1, y1, x2, y2 = np.rint(box).astype(int)
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return None
    # A side-clipped person can leave only background features in the crop.
    # Do not seed/continue that identity until a fresh interior localization.
    if x1 < 4 or x2 > frame.shape[1] - 4:
        return None
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None
    # Compare the interior, reducing contamination by surrounding background.
    dx, dy = max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)
    return cv2.resize(frame[y1 + dy : y2 - dy, x1 + dx : x2 - dx], (24, 32))


@dataclass
class Track:
    local_id: str
    box: np.ndarray
    points: np.ndarray
    reference: np.ndarray


class PreviewTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque(maxlen=128)
        self.tracks = []
        self.anchor_id = None
        self.anchor_time = -math.inf
        self.epoch = None
        self.status = "waiting_for_grounding"
        self.reason = None
        self.lost = 0
        self.replayed = 0

    def _seeds(self, frame, anchor):
        tracks = []
        h, w = frame.shape
        for index, obj in enumerate(anchor["objects"][:MAX_PEOPLE]):
            if obj.get("label") != "person":
                continue
            b = obj.get("bbox") or {}
            values = [b.get(k) for k in ("x1", "y1", "x2", "y2")]
            if not all(
                type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                for v in values
            ):
                continue
            box = np.array(values) * (w, h, w, h)
            ref = patch(frame, box)
            if ref is None or ref.std() < 8:
                continue
            mask = np.zeros_like(frame)
            x1, y1, x2, y2 = np.rint(box).astype(int)
            dx, dy = max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)
            mask[y1 + dy : y2 - dy, x1 + dx : x2 - dx] = 255
            points = cv2.goodFeaturesToTrack(frame, 40, 0.025, 3, mask=mask)
            if points is not None and len(points) >= 6:
                tracks.append(Track(f"{anchor['id']}:{index}", box, points, ref))
        return self._collisions(tracks)

    def _collisions(self, tracks):
        # Ambiguous overlaps are hidden, never resolved by choosing a nearby box.
        return [
            t
            for i, t in enumerate(tracks)
            if not any(
                i != j and iou(t.box, other.box) > 0.25
                for j, other in enumerate(tracks)
            )
        ]

    def _step(self, before, after, tracks):
        result = []
        options = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02),
        )
        for t in tracks:
            forward, ok, _ = cv2.calcOpticalFlowPyrLK(
                before, after, t.points, None, **options
            )
            if forward is None:
                continue
            backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(
                after, before, forward, None, **options
            )
            if backward is None:
                continue
            valid = (ok.ravel() == 1) & (back_ok.ravel() == 1)
            valid &= np.linalg.norm(backward - t.points, axis=2).ravel() < 1.0
            if valid.sum() < max(6, math.ceil(len(t.points) * 0.65)):
                continue
            old, new = t.points[valid, 0], forward[valid, 0]
            delta = new - old
            displacement = np.median(delta, axis=0)
            consistent = np.linalg.norm(delta - displacement, axis=1) < 2.5
            if consistent.sum() < max(6, math.ceil(len(old) * 0.75)):
                continue
            old, new = old[consistent], new[consistent]
            if np.linalg.norm(displacement) > 20:
                continue
            box = t.box + np.tile(displacement, 2)
            box = np.clip(
                box,
                (0, 0, 0, 0),
                (after.shape[1], after.shape[0], after.shape[1], after.shape[0]),
            )
            candidate = patch(after, box)
            if candidate is None or candidate.std() < 8:
                continue
            similarity = float(
                cv2.matchTemplate(candidate, t.reference, cv2.TM_CCOEFF_NORMED)[0, 0]
            )
            if not math.isfinite(similarity) or similarity < 0.55:
                continue
            result.append(Track(t.local_id, box, new.reshape(-1, 1, 2), t.reference))
        return self._collisions(result)

    def advance(self, frame, timestamp, *, epoch=0, anchor=None):
        """Caller holds lock. All timestamps are source-media time, including loops."""
        started = time.perf_counter()
        self.replayed = 0
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("invalid tracking time")
        if self.history:
            previous_time, previous_frame = self.history[-1]
            if (
                epoch != self.epoch
                or frame.shape != previous_frame.shape
                or not 0 < timestamp - previous_time <= MAX_GAP
            ):
                self.history.clear()
                self.tracks = []
                self.status, self.reason = "lost", "source_discontinuity"
        self.epoch = epoch
        if self.history and self.tracks:
            old_count = len(self.tracks)
            self.tracks = self._step(self.history[-1][1], frame, self.tracks)
            self.lost += old_count - len(self.tracks)
            if not self.tracks:
                self.status, self.reason = "lost", "appearance_or_motion_ambiguous"
        self.history.append((timestamp, frame))
        while self.history and timestamp - self.history[0][0] > HISTORY_SECONDS:
            self.history.popleft()
        if (
            anchor
            and anchor.get("epoch", epoch) == epoch
            and anchor["id"] != self.anchor_id
            and self.anchor_time < anchor["time"] <= timestamp
        ):
            self.anchor_id, self.anchor_time = anchor["id"], anchor["time"]
            self.tracks = []
            self.status, self.reason = "lost", "evidence_outside_history"
            if self.history[0][0] - MAX_GAP <= anchor["time"]:
                source = gray_frame(anchor["jpeg"])
                if source.shape == frame.shape:
                    tracks = self._seeds(source, anchor)
                    seed_count = len(tracks)
                    previous = anchor["time"]
                    for at, following in self.history:
                        if at <= previous + 1e-6:
                            continue
                        if (
                            at - previous > MAX_GAP
                            or time.perf_counter() - started > 0.12
                        ):
                            tracks = []
                            self.reason = "replay_gap_or_budget"
                            break
                        tracks = self._step(source, following, tracks)
                        source, previous = following, at
                        self.replayed += 1
                        if not tracks:
                            break
                    self.tracks = tracks
                    self.lost += seed_count - len(tracks)
                    self.reason = None if tracks else "unreliable_seed_or_tracking"
                    self.status = "tracking" if tracks else "lost"
                    if not anchor["objects"]:
                        self.status, self.reason = "no_person_boxes", None
        if timestamp - self.anchor_time > HISTORY_SECONDS and self.tracks:
            self.tracks = []
            self.status, self.reason = "lost", "grounding_expired"
        h, w = frame.shape
        return {
            "source_time": timestamp,
            "anchor_id": self.anchor_id,
            "anchor_age": timestamp - self.anchor_time
            if math.isfinite(self.anchor_time)
            else None,
            "status": self.status,
            "reason": self.reason,
            "objects": [
                {
                    "track_id": t.local_id,
                    "bbox": dict(
                        zip(
                            ("x1", "y1", "x2", "y2"),
                            (t.box / (w, h, w, h)).tolist(),
                            strict=True,
                        )
                    ),
                }
                for t in self.tracks
            ],
            "tracking_seconds": time.perf_counter() - started,
            "replayed_frames": self.replayed,
            "lost_tracks": self.lost,
            "history_frames": len(self.history),
        }
