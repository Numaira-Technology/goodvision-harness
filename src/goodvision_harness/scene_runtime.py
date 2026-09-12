"""Bounded continuous scene state shared by shadow and active live pipelines.

Prepare temporal inspections from real source observations. A separate worker
claims these immutable intervals and completes them without running model calls
under video callback locks. Shadow mode also accepts legacy safety observations;
active mode obtains safety verdicts exclusively through temporal inspections.
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
from .event_sensitivity import normalize_event_sensitivities
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
    event_sensitivities: tuple[tuple[str, str], ...] = ()

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


@dataclass(frozen=True, slots=True)
class _AcceptedEvidence:
    subject_ids: tuple[str, ...]
    detected: bool | None
    confidence: float | None
    revision: int
    timestamp: float
    sequence: int


def _subjects(attributes: Mapping[str, Any]) -> tuple[str, ...]:
    values = attributes.get("subject_ids", (attributes.get("subject_id"),))
    if not isinstance(values, (tuple, list)) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        return ()
    return tuple(sorted(set(values)))


class SceneRuntime:
    """One explicitly scoped camera session, with bounded observation memory."""

    def __init__(
        self,
        stream_session_id: str,
        source_id: str,
        *,
        mode: str = "shadow",
        event_labels: tuple[str, ...] = ("fight", "smoking", "fall", "weapon"),
        event_sensitivities: Mapping[str, str] | None = None,
    ) -> None:
        if mode not in {"shadow", "active"}:
            raise ValueError("scene mode must be shadow or active")
        if not set(event_labels) <= _EVENTS - {"stranger"}:
            raise ValueError("unsupported temporal event label")
        self.mode = mode
        self.event_labels = tuple(dict.fromkeys(event_labels))
        self.event_sensitivities = tuple(
            normalize_event_sensitivities(event_sensitivities, event_labels).items()
        )
        self.automatic_temporal_inference = False
        self.scene = SceneState(stream_session_id, source_id)
        self.evidence = SceneEvidenceBuffer()
        self.planner = InspectionPlanner(
            full_scene_interval_seconds=5.0 if mode == "active" else 15.0,
            refresh_seconds=3.0 if mode == "active" else 8.0,
        )
        self.stream_session_id = stream_session_id
        self.source_id = source_id
        self._lock = RLock()
        self._now = 0.0
        self._last_buffered = -math.inf
        self._closed = False
        self._jobs: OrderedDict[str, TemporalInspection] = OrderedDict()
        self._latest_inspection: TemporalInspection | None = None
        self._claimed: set[str] = set()
        self._pending_results: OrderedDict[str, AnalysisResult] = OrderedDict()
        self._accepted: OrderedDict[tuple[str, str, str], _AcceptedEvidence] = (
            OrderedDict()
        )
        self._processed: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self._forgotten_identities: set[str] = set()
        self._identities_disabled = False
        self._stats = {
            "association_revisions_ignored": 0,
            "association_revisions_applied": 0,
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
        # Stable object slots survive changed owner sets and changed object order.
        accepted_slots: dict[tuple[str, str], set[str]] = {}
        accepted_sources: dict[str, tuple[float, int]] = {}
        for (message_id, label, slot), accepted in self._accepted.items():
            accepted_slots.setdefault((message_id, label), set()).add(slot)
            accepted_sources[message_id] = (accepted.timestamp, accepted.sequence)
        snapshot = self.scene.snapshot(include_history=False)
        event_groups = {
            label: {
                tuple(event["subject_ids"])
                for event in snapshot["events"]
                if event["event_label"] == label and event["phase"] != "retracted"
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
            source = self._processed.get(
                result.message_id, accepted_sources.get(result.message_id)
            )
            if source is not None and source != (at, sequence):
                self._stats["association_revisions_ignored"] += 1
                continue
            judgments = {
                judgment.event_label: judgment
                for judgment in result.judgments
                if judgment.event_label in _EVENTS
            }
            labels = set(judgments) | {
                label
                for message_id, label in accepted_slots
                if message_id == result.message_id
            }
            for label in sorted(labels):
                if self.mode == "active" and label != "stranger":
                    continue
                judgment = judgments.get(label)
                desired: dict[
                    str, tuple[tuple[str, ...], bool | None, float | None]
                ] = {}
                if judgment is not None:
                    if judgment.object_ids:
                        for object_id in judgment.object_ids:
                            item = objects.get(object_id)
                            ids = _subjects(item.attributes) if item is not None else ()
                            desired[f"object:{object_id}"] = (
                                ids,
                                judgment.detected if ids else None,
                                judgment.confidence if ids else None,
                            )
                    elif not judgment.detected:
                        # A whole-frame negative applies only where its source
                        # people were truly observed, enforced by SceneState.
                        for ids in event_groups[label]:
                            slot = "group:" + sha256(repr(ids).encode()).hexdigest()
                            desired[slot] = (ids, False, judgment.confidence)
                    if label == "stranger" and not judgment.detected:
                        known = {
                            sid
                            for item in result.objects
                            if item.attributes.get("status") == "known"
                            and item.attributes.get("face_observation_state")
                            == "observed"
                            for sid in _subjects(item.attributes)
                        }
                        desired = {
                            slot: value
                            for slot, value in desired.items()
                            if set(value[0]) <= known
                        }
                accepted_key = (result.message_id, label)
                slots = set(desired) | accepted_slots.get(accepted_key, set())
                changed = False
                for slot in sorted(slots):
                    key = (*accepted_key, slot)
                    previous = self._accepted.get(key)
                    ids, detected, confidence = desired.get(slot, ((), None, None))
                    if previous is None and not ids:
                        continue  # An unassigned first result has contributed nothing.
                    if previous is not None and (
                        previous.subject_ids,
                        previous.detected,
                        previous.confidence,
                    ) == (ids, detected, confidence):
                        continue
                    revision = 0 if previous is None else previous.revision + 1
                    accepted = self.scene.revise_event_evidence(
                        label,
                        ids,
                        timestamp_seconds=at,
                        frame_sequence=sequence,
                        detected=detected,
                        confidence=confidence,
                        evidence_id=sha256(repr(key).encode()).hexdigest(),
                        revision=revision,
                    )
                    if accepted:
                        self._accepted[key] = _AcceptedEvidence(
                            ids,
                            detected,
                            confidence,
                            revision,
                            at,
                            sequence,
                        )
                        self._accepted.move_to_end(key)
                        self._processed[result.message_id] = (at, sequence)
                        self._processed.move_to_end(result.message_id)
                        accepted_slots.setdefault(accepted_key, set()).add(slot)
                        if previous is not None:
                            self._stats["association_revisions_applied"] += 1
                        changed = True
                        while len(self._accepted) > 2048:
                            self._accepted.popitem(last=False)
                        while len(self._processed) > 512:
                            self._processed.popitem(last=False)
                if changed:
                    event_groups[label] = {
                        tuple(event["subject_ids"])
                        for event in self.scene.snapshot(include_history=False)[
                            "events"
                        ]
                        if event["event_label"] == label
                        and event["phase"] != "retracted"
                    }

    def _plan(self) -> None:
        if not self.event_labels:
            return
        for request_id, job in tuple(self._jobs.items()):
            if self._now - job.request.end_seconds >= 20.0:
                self._jobs.pop(request_id)
                self._claimed.discard(request_id)
                self.scene.unpin_source_observations(request_id)
                self.planner.complete(
                    request_id, success=False, timestamp_seconds=self._now
                )
        if len(self._jobs) >= 4 or len(self.evidence) < 2:
            return
        if self.mode == "active":
            recent = self.evidence.select(
                stream_session_id=self.stream_session_id,
                start_seconds=max(0.0, self._now - 3.0),
                end_seconds=self._now,
                limit=48,
            )
            if (
                len(recent) < 2
                or recent[-1].timestamp_seconds - recent[0].timestamp_seconds < 0.75
            ):
                return
        planning_scene = self.scene.snapshot(include_history=False)
        planning_scene["events"] = [
            event
            for event in planning_scene["events"]
            if event["event_label"] in self.event_labels
        ]
        for request in self.planner.plan(planning_scene, timestamp_seconds=self._now):
            request = replace(
                request,
                event_labels=tuple(
                    label
                    for label in (request.event_labels or self.event_labels)
                    if label in self.event_labels
                ),
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
            else:
                observed_sequences = {sequence for sequence, _ in boxes}
                frames = tuple(
                    frame
                    for frame in frames
                    if frame.frame_sequence in observed_sequences
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
            job = TemporalInspection(
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
                self.event_sensitivities,
            )
            frame_times = {
                frame.frame_sequence: frame.timestamp_seconds for frame in frames
            }
            if not self.scene.pin_source_observations(
                request.request_id,
                tuple(
                    (sid, frame_times[sequence], sequence, box)
                    for sequence, sid, box in job.subject_boxes
                ),
            ):
                self.planner.complete(
                    request.request_id, success=False, timestamp_seconds=self._now
                )
                continue
            self._jobs[request.request_id] = job
            self._stats["prepared_inspections"] += 1

    def inspection_active(self, request_id: str) -> bool:
        with self._lock:
            return not self._closed and request_id in self._jobs

    def take_inspections(self, limit: int = 4) -> tuple[TemporalInspection, ...]:
        """Explicit worker seam; claiming never calls a model on the live thread."""
        if type(limit) is not int or not 1 <= limit <= 4:
            raise ValueError("inspection limit must be within 1..4")
        with self._lock:
            jobs = tuple(
                job for key, job in self._jobs.items() if key not in self._claimed
            )[:limit]
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
            try:
                if success:
                    # The pinned real source frames remain grounded even when
                    # new tracking observations have rolled past the history limit.
                    source_people: dict[int, set[str]] = {}
                    for sequence, subject_id, _bbox in job.subject_boxes:
                        source_people.setdefault(sequence, set()).add(subject_id)
                    for index, verdict in enumerate(verdicts):
                        candidates = (
                            job.frames[-1:] if job.request.subject_ids else job.frames
                        )
                        newest = next(
                            (
                                frame
                                for frame in reversed(candidates)
                                if set(verdict.subject_ids)
                                <= source_people.get(frame.frame_sequence, set())
                            ),
                            None,
                        )
                        if newest is None:
                            # Separate appearances are not a joint interaction.
                            continue
                        accepted_count += self.scene.observe_event(
                            verdict.event_label,
                            verdict.subject_ids,
                            timestamp_seconds=newest.timestamp_seconds,
                            frame_sequence=newest.frame_sequence,
                            detected=verdict.detected,
                            confidence=verdict.confidence,
                            evidence_id=f"inspection:{request_id}:{index}",
                        )
            finally:
                self.scene.unpin_source_observations(request_id)
            if success:
                self._latest_inspection = job
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

    def latest_inspection(self) -> TemporalInspection | None:
        """Return the last successful immutable interval for evidence display."""
        with self._lock:
            return self._latest_inspection

    def close(self) -> None:
        """Stop observations and release images; loss is not an event clear."""
        with self._lock:
            self._closed = True
            for request_id in self._jobs:
                self.scene.unpin_source_observations(request_id)
            self._jobs.clear()
            self._latest_inspection = None
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
                "mode": self.mode,
                "state": "stopped" if self._closed else "running",
                "automatic_temporal_inference": self.automatic_temporal_inference
                and not self._closed,
                "event_labels": list(self.event_labels),
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
