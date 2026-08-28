"""Compose a smooth delayed video stream from sparse grounded results.

The source video keeps its display cadence.  Analysis results are keyframes on
that same media timeline; boxes between two keyframes are linearly interpolated
after compatible objects have been associated.  The compositor deliberately
delays video output so it can use both bounding-box endpoints.  When a new
endpoint is unavailable, the last analyzed state is latched onto every display
frame until a later keyframe explicitly updates or clears it.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import AnalysisResult, GroundedObject, NormalizedBBox
from .result_bus import FramePacket

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OverlaySelection:
    """One analysis state selected for a buffered display frame."""

    analysis: AnalysisResult | None
    mode: str
    left_message_id: str | None = None
    right_message_id: str | None = None


@dataclass(frozen=True, slots=True)
class _BufferedFrame:
    packet: FramePacket
    ready_at: float
    generation: int


def select_overlay(
    results: Sequence[AnalysisResult],
    media_timestamp: float,
    *,
    max_gap_seconds: float,
    frame_sequence: int | None = None,
) -> OverlaySelection:
    """Select or interpolate analysis state for one media timestamp.

    Results may arrive out of order, but callers must provide them sorted by
    ``(timestamp_seconds, source_frame_index)``.  Event state changes only on a
    keyframe; interpolation moves compatible boxes between two positive
    keyframes and never invents an object before its first detection.  Once a
    state exists it remains authoritative until the next keyframe, so inference
    jitter cannot turn intermediate display frames into unannotated flashes.
    """

    if not results:
        return OverlaySelection(None, "none")
    keys = [
        (item.frame.timestamp_seconds, item.frame.source_frame_index)
        for item in results
    ]
    query_sequence = math.inf if frame_sequence is None else int(frame_sequence)
    position = bisect_right(keys, (float(media_timestamp), query_sequence)) - 1
    if position < 0:
        return OverlaySelection(None, "none")

    left = results[position]
    left_timestamp = left.frame.timestamp_seconds
    age = max(0.0, float(media_timestamp) - left_timestamp)
    exact_sequence = (
        frame_sequence is None
        or left.frame.source_frame_index == int(frame_sequence)
    )
    if abs(float(media_timestamp) - left_timestamp) <= 1e-6 and exact_sequence:
        return OverlaySelection(left, "exact", left.message_id)

    right = results[position + 1] if position + 1 < len(results) else None
    if right is None:
        return OverlaySelection(left, "held", left.message_id)

    right_timestamp = right.frame.timestamp_seconds
    gap = right_timestamp - left_timestamp
    if gap <= 0 or gap > max_gap_seconds + 1e-9:
        # A long or invalid gap is unsafe for motion interpolation, but it is
        # not a reason to blink the last known bbox off. The right keyframe will
        # update or clear the state when its timestamp is reached.
        return OverlaySelection(left, "held", left.message_id)

    alpha = max(0.0, min(1.0, age / gap))
    interpolated = _interpolate_analysis(left, right, alpha)
    return OverlaySelection(
        interpolated,
        "interpolated",
        left.message_id,
        right.message_id,
    )


class DelayedOverlayComposer:
    """Thread-safe delayed full-cadence JPEG compositor."""

    def __init__(
        self,
        *,
        delay_seconds: float,
        max_gap_seconds: float,
        output_fps: float,
        render: Callable[[bytes, AnalysisResult], bytes],
        publish: Callable[[FramePacket], Any],
        enrich: Callable[[bytes, float, int], bytes] | None = None,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        if output_fps <= 0:
            raise ValueError("output_fps must be positive")
        self.delay_seconds = float(delay_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self.output_fps = float(output_fps)
        self._render = render
        self._publish = publish
        self._enrich = enrich
        self._condition = threading.Condition()
        self._frames: deque[_BufferedFrame] = deque()
        self._results: list[AnalysisResult] = []
        self._generation = 0
        self._closed = False
        self._thread: threading.Thread | None = None
        self._stream_session_id: str | None = None
        self._last_output_timestamp: float | None = None
        self._max_buffered_frames = max(
            8,
            math.ceil((self.delay_seconds + 2.0) * self.output_fps),
        )
        self._stats: dict[str, Any] = {
            "state": "idle",
            "output_fps": self.output_fps,
            "buffer_delay_seconds": self.delay_seconds,
            "bbox_max_gap_seconds": self.max_gap_seconds,
            "frames_buffered": 0,
            "frames_published": 0,
            "frames_dropped": 0,
            "late_results": 0,
            "publish_errors": 0,
            "enrichment_render_errors": 0,
            "annotation_mode": "none",
            "frame_sequence": None,
            "media_timestamp": None,
            "left_message_id": None,
            "right_message_id": None,
        }

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("delayed overlay composer is closed")
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name="delayed-overlay",
                daemon=True,
            )
            self._thread.start()

    def reset(self) -> None:
        """Discard one session without letting an in-flight render leak across it."""

        with self._condition:
            self._generation += 1
            self._frames.clear()
            self._results.clear()
            self._stream_session_id = None
            self._last_output_timestamp = None
            self._stats.update(
                {
                    "state": "buffering",
                    "frames_buffered": 0,
                    "frames_published": 0,
                    "frames_dropped": 0,
                    "late_results": 0,
                    "publish_errors": 0,
                    "enrichment_render_errors": 0,
                    "annotation_mode": "none",
                    "frame_sequence": None,
                    "media_timestamp": None,
                    "left_message_id": None,
                    "right_message_id": None,
                }
            )
            self._condition.notify_all()

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout_seconds))

    def submit_frame(self, packet: FramePacket) -> None:
        self.start()
        with self._condition:
            generation = self._generation
            self._frames.append(
                _BufferedFrame(
                    packet=packet,
                    ready_at=time.monotonic() + self.delay_seconds,
                    generation=generation,
                )
            )
            while len(self._frames) > self._max_buffered_frames:
                self._frames.popleft()
                self._stats["frames_dropped"] += 1
            self._stats["frames_buffered"] = len(self._frames)
            if self._stats["frames_published"] == 0:
                self._stats["state"] = "buffering"
            self._condition.notify_all()

    def submit_result(self, result: AnalysisResult) -> bool:
        self.start()
        with self._condition:
            if self._stream_session_id is None:
                self._stream_session_id = result.stream_session_id
            elif result.stream_session_id != self._stream_session_id:
                return False
            if self._last_output_timestamp is not None and (
                result.frame.timestamp_seconds <= self._last_output_timestamp + 1e-9
            ):
                self._stats["late_results"] += 1
                # Even a result for an already displayed frame remains useful
                # as the newest authoritative state for future frames. Record
                # the lateness, but never create a gap by discarding it.

            key = _analysis_key(result)
            keys = [_analysis_key(item) for item in self._results]
            index = bisect_right(keys, key)
            if index > 0 and keys[index - 1] == key:
                self._results[index - 1] = result
            else:
                self._results.insert(index, result)
            self._condition.notify_all()
            return True

    def stats(self) -> dict[str, Any]:
        with self._condition:
            payload = dict(self._stats)
            payload["frames_buffered"] = len(self._frames)
            return payload

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._closed and not self._frames:
                    self._condition.wait()
                if self._closed:
                    return
                buffered = self._frames[0]
                remaining = buffered.ready_at - time.monotonic()
                if remaining > 0:
                    self._condition.wait(remaining)
                    continue
                buffered = self._frames.popleft()
                results = tuple(self._results)
                self._stats["frames_buffered"] = len(self._frames)

            selection = select_overlay(
                results,
                buffered.packet.media_timestamp,
                max_gap_seconds=self.max_gap_seconds,
                frame_sequence=buffered.packet.frame_sequence,
            )
            rendered = buffered.packet.jpeg
            if self._enrich is not None:
                try:
                    rendered = self._enrich(
                        rendered,
                        buffered.packet.media_timestamp,
                        buffered.packet.frame_sequence,
                    )
                except Exception:
                    LOGGER.warning("could not render delayed frame enrichment")
                    rendered = buffered.packet.jpeg
                    with self._condition:
                        if (
                            buffered.generation == self._generation
                            and not self._closed
                        ):
                            self._stats["enrichment_render_errors"] += 1

            enriched = rendered
            mode = selection.mode
            if selection.analysis is not None:
                try:
                    rendered = self._render(rendered, selection.analysis)
                except Exception:
                    LOGGER.warning("could not render delayed overlay frame")
                    mode = "none"
                    rendered = enriched

            output = replace(
                buffered.packet,
                jpeg=rendered,
                analysis_message_id=selection.left_message_id,
                next_analysis_message_id=selection.right_message_id,
                annotation_mode=mode,
                stream_delay_seconds=self.delay_seconds,
            )
            with self._condition:
                if buffered.generation != self._generation or self._closed:
                    continue
                try:
                    # Keep reset and publish mutually exclusive so a completed
                    # reset cannot be followed by an old-generation frame.
                    self._publish(output)
                except Exception:
                    LOGGER.warning("could not publish delayed overlay frame")
                    self._stats["publish_errors"] += 1
                    self._stats["frames_dropped"] += 1
                    continue
                self._last_output_timestamp = output.media_timestamp
                self._stats.update(
                    {
                        "state": "running",
                        "frames_published": self._stats["frames_published"] + 1,
                        "annotation_mode": mode,
                        "frame_sequence": output.frame_sequence,
                        "media_timestamp": round(output.media_timestamp, 3),
                        "left_message_id": selection.left_message_id,
                        "right_message_id": selection.right_message_id,
                    }
                )
                self._prune_results_locked(output.media_timestamp)

    def _prune_results_locked(self, media_timestamp: float) -> None:
        cutoff = media_timestamp - self.max_gap_seconds
        while (
            len(self._results) > 2
            and self._results[1].frame.timestamp_seconds < cutoff
        ):
            self._results.pop(0)


def _analysis_key(result: AnalysisResult) -> tuple[float, int]:
    return result.frame.timestamp_seconds, result.frame.source_frame_index


def _interpolate_analysis(
    left: AnalysisResult,
    right: AnalysisResult,
    alpha: float,
) -> AnalysisResult:
    right_roles = _roles_by_object(right)
    left_roles = _roles_by_object(left)
    unused_right = set(range(len(right.objects)))
    objects: list[GroundedObject] = []
    for item in left.objects:
        candidates: list[tuple[float, int]] = []
        for index in unused_right:
            candidate = right.objects[index]
            if not _compatible_objects(
                item,
                candidate,
                left_roles.get(item.object_id, frozenset()),
                right_roles.get(candidate.object_id, frozenset()),
                analysis_kind=left.analysis_kind,
            ):
                continue
            distance = _center_distance(item.bbox, candidate.bbox)
            overlap = _bbox_iou(item.bbox, candidate.bbox)
            if not _plausible_box_motion(item.bbox, candidate.bbox, distance, overlap):
                continue
            score = overlap - (0.2 * distance)
            candidates.append((score, index))
        candidates.sort(reverse=True)
        best_index: int | None = candidates[0][1] if candidates else None
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.05:
            # Do not force a track identity through an ambiguous crossing.
            best_index = None
        if best_index is None:
            objects.append(item)
            continue
        unused_right.remove(best_index)
        candidate = right.objects[best_index]
        confidence = item.confidence
        if confidence is not None and candidate.confidence is not None:
            confidence = _lerp(confidence, candidate.confidence, alpha)
        objects.append(
            replace(
                item,
                bbox=_lerp_bbox(item.bbox, candidate.bbox, alpha),
                confidence=confidence,
            )
        )
    return replace(left, objects=tuple(objects))


def _roles_by_object(result: AnalysisResult) -> dict[str, frozenset[str]]:
    roles: dict[str, set[str]] = {}
    for judgment in result.judgments:
        if not judgment.detected and result.analysis_kind != "face_recognition":
            continue
        for object_id in judgment.object_ids:
            roles.setdefault(object_id, set()).add(judgment.event_label.lower())
    return {key: frozenset(value) for key, value in roles.items()}


def _compatible_objects(
    left: GroundedObject,
    right: GroundedObject,
    left_roles: frozenset[str],
    right_roles: frozenset[str],
    *,
    analysis_kind: str,
) -> bool:
    if _normalized_label(left.label) != _normalized_label(right.label):
        return False
    if left_roles and right_roles and left_roles.isdisjoint(right_roles):
        return False
    if analysis_kind == "face_recognition":
        left_status = str(left.attributes.get("status", ""))
        right_status = str(right.attributes.get("status", ""))
        if left_status and right_status and left_status != right_status:
            return False
        left_name = str(left.attributes.get("name", ""))
        right_name = str(right.attributes.get("name", ""))
        if left_name and right_name and left_name != right_name:
            return False
    return True


def _normalized_label(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _lerp_bbox(
    left: NormalizedBBox,
    right: NormalizedBBox,
    alpha: float,
) -> NormalizedBBox:
    return NormalizedBBox(
        _lerp(left.x1, right.x1, alpha),
        _lerp(left.y1, right.y1, alpha),
        _lerp(left.x2, right.x2, alpha),
        _lerp(left.y2, right.y2, alpha),
    )


def _lerp(left: float, right: float, alpha: float) -> float:
    return float(left) + ((float(right) - float(left)) * float(alpha))


def _bbox_iou(left: NormalizedBBox, right: NormalizedBBox) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    left_area = (left.x2 - left.x1) * (left.y2 - left.y1)
    right_area = (right.x2 - right.x1) * (right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(left: NormalizedBBox, right: NormalizedBBox) -> float:
    left_x = (left.x1 + left.x2) / 2.0
    left_y = (left.y1 + left.y2) / 2.0
    right_x = (right.x1 + right.x2) / 2.0
    right_y = (right.y1 + right.y2) / 2.0
    return math.hypot(left_x - right_x, left_y - right_y)


def _plausible_box_motion(
    left: NormalizedBBox,
    right: NormalizedBBox,
    distance: float,
    overlap: float,
) -> bool:
    left_width, left_height = left.x2 - left.x1, left.y2 - left.y1
    right_width, right_height = right.x2 - right.x1, right.y2 - right.y1
    left_area = left_width * left_height
    right_area = right_width * right_height
    area_ratio = right_area / left_area
    aspect_ratio = (right_width / right_height) / (left_width / left_height)
    if not 0.25 <= area_ratio <= 4.0:
        return False
    if not 0.33 <= aspect_ratio <= 3.0:
        return False
    diagonal = max(
        math.hypot(left_width, left_height),
        math.hypot(right_width, right_height),
    )
    allowed_distance = min(0.3, max(0.08, 2.5 * diagonal))
    return overlap > 0 or distance <= allowed_distance
