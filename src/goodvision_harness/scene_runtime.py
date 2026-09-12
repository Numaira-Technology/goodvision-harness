"""Experimental bridge from source observations to a persistent scene.

Live integration runs in shadow mode: it prepares bounded temporal inspections
without changing the existing alarm feed or invoking another model. An explicit
consumer can take an inspection and return a verdict via complete_inspection.
All methods are CPU-only; model calls must run outside the live callback lock.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from threading import RLock
from typing import Any

from .contracts import AnalysisResult, NormalizedBBox
from .rich_labeling import RichLabelSnapshot
from .scene_evidence import (
    EvidenceFrame,
    InspectionPlanner,
    InspectionRequest,
    SceneEvidenceBuffer,
)
from .scene_state import SceneState

_EVENTS = frozenset(("fight", "smoking", "fall", "weapon", "stranger"))
_PERSON_LABELS = frozenset(
    (
        "person",
        "pedestrian",
        "known_person",
        "unknown_person",
        "unusable_face",
        "face",
        "known_face",
        "unknown_face",
        "man",
        "woman",
        "smoker",
        "fighter",
    )
)


@dataclass(frozen=True, slots=True)
class TemporalInspection:
    """An immutable source interval; images are internal, never in scene JSON."""

    stream_session_id: str
    source_id: str
    request: InspectionRequest
    frames: tuple[EvidenceFrame, ...]
    subject_boxes: tuple[tuple[int, str, NormalizedBBox], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.request.as_dict(),
            "stream_session_id": self.stream_session_id,
            "source_id": self.source_id,
            "state": "prepared",
            "frames": [
                {
                    "frame_sequence": frame.frame_sequence,
                    "timestamp_seconds": frame.timestamp_seconds,
                }
                for frame in self.frames
            ],
        }


@dataclass(frozen=True, slots=True)
class TemporalVerdict:
    event_label: str
    subject_ids: tuple[str, ...]
    detected: bool | None
    confidence: float | None = None


def _subjects(attributes: Mapping[str, Any]) -> tuple[str, ...]:
    values = attributes.get("subject_ids", (attributes.get("subject_id"),))
    if not isinstance(values, (tuple, list)) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        return ()
    return tuple(sorted(set(values)))


class SceneRuntime:
    """One explicitly scoped camera session, with bounded observation memory."""

    def __init__(self, stream_session_id: str, source_id: str) -> None:
        self.scene = SceneState(stream_session_id, source_id)
        self.evidence = SceneEvidenceBuffer()
        self.planner = InspectionPlanner()
        self.stream_session_id = stream_session_id
        self.source_id = source_id
        self._lock = RLock()
        self._now = 0.0
        self._last_buffered = -math.inf
        self._closed = False
        self._jobs: OrderedDict[str, TemporalInspection] = OrderedDict()
        self._claimed: set[str] = set()
        self._pending_results: OrderedDict[str, AnalysisResult] = OrderedDict()
        self._accepted: OrderedDict[tuple[str, str], tuple[tuple[str, ...], ...]] = (
            OrderedDict()
        )
        self._processed: OrderedDict[tuple, None] = OrderedDict()
        self._forgotten_identities: set[str] = set()
        self._identities_disabled = False
        self._stats = {
            "association_revisions_ignored": 0,
            "prepared_inspections": 0,
            "completed_inspections": 0,
            "failed_inspections": 0,
            "accepted_temporal_verdicts": 0,
            "rejected_temporal_verdicts": 0,
        }

    def observe_frame(
        self,
        jpeg_bytes: bytes,
        timestamp_seconds: float,
        frame_sequence: int,
        *,
        force: bool = False,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self.scene.advance(timestamp_seconds)
            self._now = max(self._now, timestamp_seconds)
            # Keep several seconds of source evidence even on a 15+ FPS feed.
            if force or timestamp_seconds - self._last_buffered >= 0.25:
                if self.evidence.add(
                    stream_session_id=self.stream_session_id,
                    frame_sequence=frame_sequence,
                    timestamp_seconds=timestamp_seconds,
                    jpeg_bytes=jpeg_bytes,
                ):
                    self._last_buffered = max(self._last_buffered, timestamp_seconds)
            self._plan()

    def observe_tracks(self, snapshot: RichLabelSnapshot) -> None:
        """Accept registry-canonical subjects, never raw detector track IDs."""
        with self._lock:
            if self._closed or snapshot.stream_session_id != self.stream_session_id:
                return
            for item in snapshot.objects:
                if item.label.lower() not in _PERSON_LABELS:
                    continue
                state = item.observation_state or item.attributes.get(
                    "observation_state",
                    "observed",
                )
                self.scene.observe_person(
                    item.track_id,
                    timestamp_seconds=snapshot.timestamp_seconds,
                    frame_sequence=snapshot.frame_sequence,
                    bbox=item.bbox,
                    observation_state=state,
                )
            self._now = max(self._now, snapshot.timestamp_seconds)
            self.scene.advance(self._now)
            self._retry_results()
            self._plan()

    def observe_result(self, result: AnalysisResult) -> None:
        with self._lock:
            if self._closed or (
                result.stream_session_id != self.stream_session_id
                or result.source_id != self.source_id
            ):
                return
            at, sequence = (
                result.frame.timestamp_seconds,
                result.frame.source_frame_index,
            )
            for item in result.objects:
                ids = _subjects(item.attributes)
                if len(ids) != 1 or item.label.lower() not in _PERSON_LABELS:
                    continue
                state = item.attributes.get("face_observation_state")
                if state is None:
                    state = item.attributes.get("observation_state", "observed")
                    if (
                        item.attributes.get("identity_source") == "track_history"
                        and item.attributes.get("observation_status") is None
                    ):
                        state = "predicted"
                identity = item.attributes.get("subject_identity", item.attributes)
                if not isinstance(identity, Mapping):
                    identity = {}
                if self._identities_disabled or any(
                    identity.get(key) in self._forgotten_identities
                    for key in ("person_id", "matched_memory_id")
                ):
                    identity = None
                self.scene.observe_person(
                    ids[0],
                    timestamp_seconds=at,
                    frame_sequence=sequence,
                    bbox=item.bbox,
                    identity=identity,
                    observation_state=state,
                )
            self._now = max(self._now, at)
            self.scene.advance(self._now)
            self._pending_results[result.message_id] = result
            self._pending_results.move_to_end(result.message_id)
            while len(self._pending_results) > 128:
                self._pending_results.popitem(last=False)
            self._retry_results()
            self._plan()

    def _retry_results(self) -> None:
        ordered = sorted(
            self._pending_results.values(),
            key=lambda result: (
                result.frame.timestamp_seconds,
                result.frame.source_frame_index,
            ),
        )
        # Keep late negatives available until their source observations arrive.
        snapshot = self.scene.snapshot(include_history=False)
        event_groups = {
            label: {
                tuple(event["subject_ids"])
                for event in snapshot["events"]
                if event["event_label"] == label
            }
            for label in _EVENTS
        }
        for result in ordered:
            if self._now - result.frame.timestamp_seconds > 90.0:
                self._pending_results.pop(result.message_id, None)
                continue
            objects = {item.object_id: item for item in result.objects}
            at = result.frame.timestamp_seconds
            sequence = result.frame.source_frame_index
            for judgment in result.judgments:
                if judgment.event_label not in _EVENTS:
                    continue
                groups = (
                    {
                        _subjects(objects[object_id].attributes)
                        for object_id in judgment.object_ids
                    }
                    if judgment.detected or judgment.object_ids
                    else set(event_groups[judgment.event_label])
                )
                groups.discard(())
                if judgment.event_label == "stranger" and not judgment.detected:
                    known = {
                        sid
                        for item in result.objects
                        if item.attributes.get("status") == "known"
                        and item.attributes.get("face_observation_state") == "observed"
                        for sid in _subjects(item.attributes)
                    }
                    groups = {ids for ids in groups if set(ids) <= known}
                if judgment.detected and not groups:
                    # Same-message source-face correction may supply owners later.
                    continue
                for group_index, ids in enumerate(sorted(groups)):
                    # Preserve object/group identity for multiple simultaneous events.
                    processed_key = (result.message_id, judgment.event_label, ids)
                    if processed_key in self._processed:
                        continue
                    # Keep the evidence identity stable even if an association
                    # revision changes owners after the bridge cache expires.
                    evidence_id = sha256(
                        repr(
                            (
                                result.message_id,
                                judgment.event_label,
                                group_index,
                            )
                        ).encode()
                    ).hexdigest()
                    key = (result.message_id, judgment.event_label)
                    owners = tuple(sorted(groups))
                    accepted_owners = self._accepted.get(key)
                    if (
                        judgment.detected
                        and accepted_owners is not None
                        and accepted_owners != owners
                    ):
                        self._stats["association_revisions_ignored"] += 1
                        continue
                    accepted = self.scene.observe_event(
                        judgment.event_label,
                        ids,
                        timestamp_seconds=at,
                        frame_sequence=sequence,
                        detected=judgment.detected,
                        confidence=judgment.confidence,
                        evidence_id=evidence_id,
                    )
                    if accepted:
                        event_groups[judgment.event_label].add(ids)
                        if judgment.detected:
                            self._accepted[key] = owners
                        self._processed[processed_key] = None
                        while len(self._accepted) > 512:
                            self._accepted.popitem(last=False)
                        while len(self._processed) > 2048:
                            self._processed.popitem(last=False)

    def _plan(self) -> None:
        for request_id, job in tuple(self._jobs.items()):
            if self._now - job.request.end_seconds >= 20.0:
                self._jobs.pop(request_id)
                self._claimed.discard(request_id)
                self.planner.complete(
                    request_id, success=False, timestamp_seconds=self._now
                )
        if len(self._jobs) >= 4 or len(self.evidence) < 2:
            return
        planning_scene = self.scene.snapshot(include_history=False)
        planning_scene["events"] = [
            event
            for event in planning_scene["events"]
            if event["event_label"] != "stranger"
        ]
        for request in self.planner.plan(planning_scene, timestamp_seconds=self._now):
            if not request.event_labels:
                request = replace(
                    request, event_labels=("fight", "smoking", "fall", "weapon")
                )
            frames = self.evidence.select(
                stream_session_id=self.stream_session_id,
                start_seconds=request.start_seconds,
                end_seconds=request.end_seconds,
                limit=48,
            )
            histories = self.scene.snapshot()["entities"]
            boxes = {
                (record["frame_sequence"], entity["subject_id"]): NormalizedBBox(
                    *(record["bbox"][key] for key in ("x1", "y1", "x2", "y2"))
                )
                for entity in histories
                for record in entity["history"]
                if request.start_seconds
                <= record["timestamp_seconds"]
                <= request.end_seconds
            }
            if request.subject_ids:
                frames = tuple(
                    frame
                    for frame in frames
                    if all(
                        (frame.frame_sequence, sid) in boxes
                        for sid in request.subject_ids
                    )
                )
            if len(frames) > 4:
                frames = tuple(
                    frames[round(i * (len(frames) - 1) / 3)] for i in range(4)
                )
            retained = sum(
                len(frame.jpeg_bytes)
                for job in self._jobs.values()
                for frame in job.frames
            )
            if (
                len(frames) < 2
                or len(self._jobs) >= 4
                or retained + sum(len(frame.jpeg_bytes) for frame in frames)
                > 16 * 1024 * 1024
            ):
                self.planner.complete(
                    request.request_id, success=False, timestamp_seconds=self._now
                )
                continue
            self._jobs[request.request_id] = TemporalInspection(
                self.stream_session_id,
                self.source_id,
                request,
                frames,
                tuple(
                    (frame.frame_sequence, sid, box)
                    for frame in frames
                    for (sequence, sid), box in boxes.items()
                    if frame.frame_sequence == sequence
                ),
            )
            self._stats["prepared_inspections"] += 1

    def take_inspections(self) -> tuple[TemporalInspection, ...]:
        """Explicit worker seam; claiming never calls a model on the live thread."""
        with self._lock:
            jobs = tuple(
                job for key, job in self._jobs.items() if key not in self._claimed
            )
            self._claimed.update(job.request.request_id for job in jobs)
            return jobs

    def complete_inspection(
        self,
        request_id: str,
        verdicts: Sequence[TemporalVerdict],
        *,
        success: bool = True,
        completion_counts: dict[str, int] | None = None,
    ) -> bool:
        with self._lock:
            job = self._jobs.get(request_id)
            if self._closed or job is None or request_id not in self._claimed:
                return False
            if (
                type(success) is not bool
                or not isinstance(verdicts, (tuple, list))
                or len(verdicts) > 128
            ):
                raise ValueError("invalid temporal completion")
            allowed = {sid for _, sid, _ in job.subject_boxes}
            if job.request.subject_ids:
                allowed.intersection_update(job.request.subject_ids)
            seen: set[tuple[str, tuple[str, ...]]] = set()
            for verdict in verdicts:
                if (
                    not isinstance(verdict, TemporalVerdict)
                    or not isinstance(verdict.subject_ids, (tuple, list))
                    or not verdict.subject_ids
                    or len(verdict.subject_ids) > 128
                    or any(
                        not isinstance(sid, str) or not sid or len(sid) > 256
                        for sid in verdict.subject_ids
                    )
                    or not set(verdict.subject_ids) <= allowed
                    or verdict.event_label not in job.request.event_labels
                    or type(verdict.detected) not in (bool, type(None))
                    or (
                        verdict.confidence is not None
                        and (
                            type(verdict.confidence) not in (int, float)
                            or not 0 <= verdict.confidence <= 1
                        )
                    )
                ):
                    raise ValueError("temporal verdict does not match its inspection")
                key = (verdict.event_label, tuple(sorted(set(verdict.subject_ids))))
                if key in seen:
                    raise ValueError("duplicate event verdict in temporal inspection")
                seen.add(key)
            self._jobs.pop(request_id)
            self._claimed.discard(request_id)
            accepted_count = 0
            if success:
                # Use the actual last included source image, never completion time.
                newest = job.frames[-1]
                for index, verdict in enumerate(verdicts):
                    accepted_count += self.scene.observe_event(
                        verdict.event_label,
                        verdict.subject_ids,
                        timestamp_seconds=newest.timestamp_seconds,
                        frame_sequence=newest.frame_sequence,
                        detected=verdict.detected,
                        confidence=verdict.confidence,
                        evidence_id=f"inspection:{request_id}:{index}",
                    )
            self.planner.complete(
                request_id, success=success, timestamp_seconds=self._now
            )
            self._stats[
                "completed_inspections" if success else "failed_inspections"
            ] += 1
            rejected_count = len(verdicts) - accepted_count if success else 0
            self._stats["accepted_temporal_verdicts"] += accepted_count
            self._stats["rejected_temporal_verdicts"] += rejected_count
            if completion_counts is not None:
                completion_counts.update(
                    accepted_verdicts=accepted_count,
                    rejected_verdicts=rejected_count,
                )
            return True

    def close(self) -> None:
        """Stop observations and release images; loss is not an event clear."""
        with self._lock:
            self._closed = True
            self._jobs.clear()
            self._claimed.clear()
            self._pending_results.clear()
            self.evidence.reset()

    def forget_identity(self, person_id: str) -> None:
        with self._lock:
            self.scene.forget_identity(person_id)
            if len(self._forgotten_identities) >= 1024:
                self._identities_disabled = True
            else:
                self._forgotten_identities.add(person_id)
            self._pending_results.clear()

    def allow_identity(self, person_id: str) -> None:
        with self._lock:
            self._forgotten_identities.discard(person_id)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "mode": "shadow",
                "state": "stopped" if self._closed else "running",
                "automatic_temporal_inference": False,
                "scene": self.scene.snapshot(),
                "inspections": [
                    job.as_dict()
                    | {
                        "state": "claimed" if key in self._claimed else "prepared",
                    }
                    for key, job in self._jobs.items()
                ],
                "stats": dict(self._stats),
            }
