"""Conservative, session-stable selection of a side-by-side camera view."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from .video_to_stream import LiveFrame, crop_frame

CROP_MODES = frozenset({"auto", "none", "left_half", "right_half"})


def stereo_pair_evidence(bgr: Any) -> bool | None:
    """Require distributed reciprocal matches and near-horizontal geometry.

    None means insufficient visual evidence, never permission to crop. Aspect
    ratio alone cannot distinguish a panorama from side-by-side camera views.
    """
    height, width = bgr.shape[:2]
    if width < 160 or height < 80 or width % 2 or width < height:
        return None
    half = width // 2
    scale = min(1.0, 480 / half, 320 / height)
    size = (max(1, round(half * scale)), max(1, round(height * scale)))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    left = cv2.resize(gray[:, :half], size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(gray[:, half:], size, interpolation=cv2.INTER_AREA)
    if min(float(left.std()), float(right.std())) < 10:
        return None
    for view in (left, right):
        distribution = np.bincount((view.ravel() // 16), minlength=16) / view.size
        probabilities = distribution[distribution > 0]
        if float(-(probabilities * np.log2(probabilities)).sum()) < 1.2:
            return None
    detector = cv2.ORB_create(nfeatures=900, edgeThreshold=16, fastThreshold=12)
    points_l, descriptors_l = detector.detectAndCompute(left, None)
    points_r, descriptors_r = detector.detectAndCompute(right, None)
    if descriptors_l is None or descriptors_r is None:
        return None
    if min(len(points_l), len(points_r)) < 60:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def distinctive(source: Any, target: Any) -> dict[int, int]:
        return {
            pair[0].queryIdx: pair[0].trainIdx
            for pair in matcher.knnMatch(source, target, k=2)
            if len(pair) == 2
            and pair[0].distance < 0.7 * pair[1].distance
            and pair[0].distance <= 48
        }

    forward = distinctive(descriptors_l, descriptors_r)
    reverse = distinctive(descriptors_r, descriptors_l)
    pairs = [(a, b) for a, b in forward.items() if reverse.get(b) == a]
    if len(pairs) < max(32, 0.10 * min(len(points_l), len(points_r))):
        return False
    source = np.float32([points_l[a].pt for a, _ in pairs])
    target = np.float32([points_r[b].pt for _, b in pairs])
    displacement = (target - source) / np.float32(size)
    # Zero-disparity copies/symmetry alone do not establish two camera views.
    if float(np.median(np.abs(target[:, 0] - source[:, 0]))) < 1.5:
        return False
    horizontal = (
        (np.abs(displacement[:, 1]) <= 0.025)
        & (np.abs(displacement[:, 0]) <= 0.22)
    )
    if float(horizontal.mean()) < 0.85:
        return False
    source, target = source[horizontal], target[horizontal]
    transform, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=1000, confidence=0.99,
    )
    if transform is None or inliers is None:
        return False
    mask = inliers.ravel().astype(bool)
    if int(mask.sum()) < 28 or float(mask.mean()) < 0.6:
        return False
    rotation = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
    zoom = math.hypot(transform[0, 0], transform[1, 0])
    if not (0.92 <= zoom <= 1.08 and abs(rotation) <= 4):
        return False
    if abs(transform[1, 2]) > size[1] * 0.035:
        return False
    # A shared logo, person or repeated corner is not a whole duplicate view.
    for coordinates in (source[mask], target[mask]):
        normalized = coordinates / np.float32(size)
        span = np.percentile(normalized, 95, axis=0) - np.percentile(
            normalized, 5, axis=0,
        )
        cells = np.clip((normalized * 3).astype(int), 0, 2)
        coverage = len({(int(x), int(y)) for x, y in cells})
        if min(span) < 0.45 or coverage < 6:
            return False
    return True


class FrameCropSelector:
    """Probe briefly before publishing frames; lock one layout for the session.

    Only the newest probing frame is retained. No full/half coordinate change
    can enter the tracking, evidence or delayed-overlay history mid-session.
    """

    def __init__(self, mode: str) -> None:
        if mode not in CROP_MODES:
            raise ValueError("frame crop mode must be auto, none, left_half, or right_half")
        self.mode = mode
        self.effective: str | None = None if mode == "auto" else mode
        self.reason = "probing" if mode == "auto" else "manual"
        self.probes = 0

    def status(self) -> dict[str, Any]:
        return {
            "requested": self.mode,
            "effective": self.effective,
            "reason": self.reason,
            "probed_frames": self.probes,
        }

    def frames(
        self, frames: Iterator[LiveFrame], *, stopped: Callable[[], bool],
    ) -> Iterator[LiveFrame]:
        first_time: float | None = None
        last_probe = -math.inf
        last_frame: LiveFrame | None = None
        positive_run = 0
        observed = 0
        for frame in frames:
            if stopped():
                return
            if self.effective is None:
                last_frame = frame
                observed += 1
                if first_time is None:
                    first_time = frame.timestamp_seconds
                if frame.timestamp_seconds - last_probe >= 0.12:
                    last_probe = frame.timestamp_seconds
                    self.probes += 1
                    try:
                        evidence = stereo_pair_evidence(frame.bgr)
                    except (cv2.error, ValueError, TypeError):
                        evidence = None
                    positive_run = positive_run + 1 if evidence is True else 0
                    if positive_run >= 3:
                        self.effective, self.reason = "left_half", "stereo_confirmed"
                if self.effective is None and (
                    frame.timestamp_seconds - first_time >= 1.0 or observed >= 20
                ):
                    self.effective, self.reason = "none", "stereo_not_confirmed"
                if self.effective is None:
                    continue
            last_frame = None
            yield replace(frame, bgr=crop_frame(frame.bgr, mode=self.effective))
        if last_frame is not None and not stopped():
            self.effective, self.reason = "none", "stereo_not_confirmed"
            yield last_frame
