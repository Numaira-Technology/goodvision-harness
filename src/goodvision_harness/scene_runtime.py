"""Bounded continuous scene state shared by shadow and active live pipelines.

Prepare temporal inspections from real source observations. A separate worker
claims these immutable intervals and completes them without running model calls
under video callback locks. Shadow mode also accepts legacy safety observations;
active mode obtains safety verdicts exclusively through temporal inspections.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from threading import RLock
from typing import Any
from uuid import uuid4

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
from .video_tracking import BACKEND as _NATIVE_VIDEO_BACKEND

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
class _TrackingInterval:
    request: InspectionRequest
    source_frames: tuple[tuple[int, float], ...]
    subject_boxes: tuple[tuple[int, str, NormalizedBBox], ...]


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
        tracking_mode: str = "immediate",
        tracking_history_seconds: float = 12.0,
        native_memory_scope: str | None = None,
    ) -> None:
        if mode not in {"shadow", "active"}:
            raise ValueError("scene mode must be shadow or active")
        if not set(event_labels) <= _EVENTS - {"stranger"}:
            raise ValueError("unsupported temporal event label")
        if tracking_mode not in {"immediate", "batch"}:
            raise ValueError("tracking mode must be immediate or batch")
        if native_memory_scope not in {None, "batch", "session"}:
            raise ValueError("native memory scope must be batch or session")
        if tracking_mode == "immediate" and native_memory_scope == "batch":
            raise ValueError("immediate tracking cannot have batch native memory")
        if (
            type(tracking_history_seconds) not in (int, float)
            or not math.isfinite(tracking_history_seconds)
            or not 3 <= tracking_history_seconds <= 90
        ):
            raise ValueError("tracking history must be within 3..90 source seconds")
        self.mode = mode
        self.tracking_mode = tracking_mode
        self.native_memory_scope = native_memory_scope or (
            "batch" if tracking_mode == "batch" else "session"
        )
        self.tracking_history_seconds = float(tracking_history_seconds)
        self.event_labels = tuple(dict.fromkeys(event_labels))
        self.event_sensitivities = tuple(
            normalize_event_sensitivities(event_sensitivities, event_labels).items()
        )
        self.automatic_temporal_inference = False
        self.scene = SceneState(
            stream_session_id,
            source_id,
            person_history_size=512 if tracking_mode == "batch" else None,
            person_history_seconds=(
                self.tracking_history_seconds if tracking_mode == "batch" else None
            ),
        )
        self.evidence = (
            SceneEvidenceBuffer(
                max_frames=512,
                max_bytes=64 * 1024 * 1024,
                history_seconds=self.tracking_history_seconds,
            )
            if tracking_mode == "batch"
            else SceneEvidenceBuffer()
        )
        self.planner = InspectionPlanner(
            full_scene_interval_seconds=5.0 if mode == "active" else 15.0,
            refresh_seconds=3.0 if mode == "active" else 8.0,
        )
        self.stream_session_id = stream_session_id
        self.source_id = source_id
        self._lock = RLock()
        self._now = 0.0
        self._tracking_completed_timestamp: float | None = None
        self._tracking_completed_sequence: int | None = None
        self._tracking_frames: OrderedDict[int, float] = OrderedDict()
        self._tracking_rejections: dict[str, int] = {}
        self._tracking_last_rejection: str | None = None
        self._tracking_intervals: OrderedDict[str, _TrackingInterval] = OrderedDict()
        self._tracking_interval_rejections: dict[str, int] = {}
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
            "completed_tracking_batches": 0,
            "completed_tracking_frames": 0,
            "rejected_tracking_batches": 0,
            "queued_tracking_intervals": 0,
            "rejected_tracking_intervals": 0,
        }

    def observe_frame(
        self,
        jpeg_bytes: bytes,
        timestamp_seconds: float,
        frame_sequence: int,
        *,
        force: bool = False,
        retain_source: bool = True,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self.scene.advance(timestamp_seconds)
            self._now = max(self._now, timestamp_seconds)
            self.evidence.advance(self._now)
            # Keep several seconds of source evidence even on a 15+ FPS feed.
            if retain_source and (
                self.tracking_mode == "batch"
                or force
                or timestamp_seconds - self._last_buffered >= 0.25
            ):
                if self.evidence.add(
                    stream_session_id=self.stream_session_id,
                    frame_sequence=frame_sequence,
                    timestamp_seconds=timestamp_seconds,
                    jpeg_bytes=jpeg_bytes,
                ):
                    self._last_buffered = max(self._last_buffered, timestamp_seconds)
            self._expire_inspections()
            self._prune_tracking_frames()
            if self.tracking_mode == "immediate":
                self._plan()
            else:
                self._drain_tracking_intervals()

    def observe_tracks(self, snapshot: RichLabelSnapshot) -> None:
        """Accept registry-canonical subjects, never raw detector track IDs."""
        with self._lock:
            if self._closed or snapshot.stream_session_id != self.stream_session_id:
                return
            if self.tracking_mode == "batch" and snapshot.timestamp_seconds > self._now:
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
            if self.tracking_mode == "immediate":
                self._now = max(self._now, snapshot.timestamp_seconds)
            self.scene.advance(self._now)
            self._retry_results()
            if self.tracking_mode == "immediate":
                self._plan()

    def _reject_tracking_batch(self, reason: str) -> bool:
        self._stats["rejected_tracking_batches"] += 1
        self._tracking_rejections[reason] = self._tracking_rejections.get(reason, 0) + 1
        self._tracking_last_rejection = reason
        return False

    def _prune_tracking_frames(self) -> None:
        while self._tracking_frames and (
            next(iter(self._tracking_frames.values()))
            < self._now - self.tracking_history_seconds
            or len(self._tracking_frames) > 512
        ):
            self._tracking_frames.popitem(last=False)

    def _tracking_source_error(
        self,
        snapshots: Sequence[RichLabelSnapshot],
        timestamp: float,
        sequence: int,
    ) -> str | None:
        if self._closed:
            return "closed"
        if self.tracking_mode != "batch":
            return "wrong_tracking_mode"
        if (
            not isinstance(snapshots, (tuple, list))
            or not 1 <= len(snapshots) <= 512
            or any(not isinstance(item, RichLabelSnapshot) for item in snapshots)
            or type(timestamp) not in (int, float)
            or not math.isfinite(timestamp)
            or timestamp < 0
            or type(sequence) is not int
            or sequence < 0
        ):
            return "invalid_batch"
        if any(item.stream_session_id != self.stream_session_id for item in snapshots):
            return "different_session"
        if (timestamp, sequence) != (
            snapshots[-1].timestamp_seconds,
            snapshots[-1].frame_sequence,
        ):
            return "incomplete_watermark"
        previous_time = self._tracking_completed_timestamp
        previous_sequence = self._tracking_completed_sequence
        for snapshot in snapshots:
            if (
                previous_time is not None and snapshot.timestamp_seconds < previous_time
            ) or (
                previous_sequence is not None
                and snapshot.frame_sequence <= previous_sequence
            ):
                return "out_of_order"
            previous_time = snapshot.timestamp_seconds
            previous_sequence = snapshot.frame_sequence
            if self._now - snapshot.timestamp_seconds > self.tracking_history_seconds:
                return "evidence_expired"
            if (
                self.evidence.source_frame(
                    stream_session_id=self.stream_session_id,
                    frame_sequence=snapshot.frame_sequence,
                    timestamp_seconds=snapshot.timestamp_seconds,
                )
                is None
            ):
                return "source_frame_missing"
        return None

    def observe_tracking_batch(
        self,
        snapshots: Sequence[RichLabelSnapshot],
        *,
        completed_timestamp_seconds: float,
        completed_frame_sequence: int,
    ) -> bool:
        """Commit a complete, ordered native-video result against captured originals.

        Sampling may skip source sequence numbers. A failed batch does not move
        the tracking watermark, refresh capture time, or generate event negatives.
        """
        with self._lock:
            error = self._tracking_source_error(
                snapshots, completed_timestamp_seconds, completed_frame_sequence
            )
            if error is not None:
                return self._reject_tracking_batch(error)

            # Validate all object data and historical source ordering before any
            # mutation, including a face observation that arrived ahead of SAM.
            people = {
                entity["subject_id"]: entity
                for entity in self.scene.snapshot()["entities"]
            }
            new_subjects: set[str] = set()
            observed_subjects: set[str] = set()
            for snapshot in snapshots:
                for item in snapshot.objects:
                    if item.label.lower() not in _PERSON_LABELS:
                        continue
                    state = item.observation_state or item.attributes.get(
                        "observation_state", "observed"
                    )
                    native = item.attributes.get("native_track")
                    if (
                        not item.track_id.strip()
                        or item.track_id != item.track_id.strip()
                        or len(item.track_id) > 256
                        or not isinstance(state, str)
                        or state
                        not in {"observed", "predicted", "held", "occluded", "lost"}
                        or (
                            native is not None
                            and (
                                not isinstance(native, Mapping)
                                or any(
                                    key in native
                                    and (
                                        not isinstance(native[key], str)
                                        or not native[key].strip()
                                        or len(native[key]) > 256
                                    )
                                    for key in ("backend", "epoch", "object_id")
                                )
                            )
                        )
                    ):
                        return self._reject_tracking_batch("invalid_observation")
                    if state != "observed":
                        continue
                    observed_subjects.add(item.track_id)
                    if item.track_id not in people:
                        new_subjects.add(item.track_id)
                    elif any(
                        (
                            record["frame_sequence"] < snapshot.frame_sequence
                            and record["timestamp_seconds"] > snapshot.timestamp_seconds
                        )
                        or (
                            record["frame_sequence"] > snapshot.frame_sequence
                            and record["timestamp_seconds"] < snapshot.timestamp_seconds
                        )
                        or (
                            record["frame_sequence"] == snapshot.frame_sequence
                            and record["timestamp_seconds"]
                            != snapshot.timestamp_seconds
                        )
                        for record in people[item.track_id]["history"]
                    ):
                        return self._reject_tracking_batch(
                            "inconsistent_source_history"
                        )
            if len(people) + len(new_subjects) > self.scene.max_entities and not (
                self.scene.reserve_person_capacity(tuple(observed_subjects))
            ):
                return self._reject_tracking_batch("subject_capacity")
            for snapshot in snapshots:
                for item in snapshot.objects:
                    if item.label.lower() not in _PERSON_LABELS:
                        continue
                    self.scene.observe_person(
                        item.track_id,
                        timestamp_seconds=snapshot.timestamp_seconds,
                        frame_sequence=snapshot.frame_sequence,
                        bbox=item.bbox,
                        observation_state=item.observation_state
                        or item.attributes.get("observation_state", "observed"),
                        allow_historical=True,
                        native_track=item.attributes.get("native_track"),
                    )
                self._tracking_frames[snapshot.frame_sequence] = (
                    snapshot.timestamp_seconds
                )
            self._tracking_completed_timestamp = float(completed_timestamp_seconds)
            self._tracking_completed_sequence = completed_frame_sequence
            self._stats["completed_tracking_batches"] += 1
            self._stats["completed_tracking_frames"] += len(snapshots)
            self._prune_tracking_frames()
            self.scene.advance(self._now)
            self._retry_results()
            self._enqueue_tracking_intervals(snapshots)
            self._plan()
            return True

    def _reject_tracking_interval(self, reason: str) -> None:
        self._stats["rejected_tracking_intervals"] += 1
        self._tracking_interval_rejections[reason] = (
            self._tracking_interval_rejections.get(reason, 0) + 1
        )

    def commit_native_tracking_batch(
        self,
        raw_snapshots: Sequence[RichLabelSnapshot],
        *,
        completed_timestamp_seconds: float,
        completed_frame_sequence: int,
        canonicalize: Callable[
            [Sequence[RichLabelSnapshot]], Sequence[RichLabelSnapshot]
        ],
    ) -> bool:
        """Validate source retention before a trusted registry allocates identities.

        The callback runs under the scene lock and must only map native tracks to
        canonical IDs, preserving source frames, observations, and geometry. This
        prevents capture from evicting evidence between admission and commit.
        """
        with self._lock:
            error = self._tracking_source_error(
                raw_snapshots, completed_timestamp_seconds, completed_frame_sequence
            )
            if error is not None:
                return self._reject_tracking_batch(error)
            native_keys: set[tuple[str, str, str]] = set()
            native_frames = []
            for snapshot in raw_snapshots:
                observations = {}
                for item in snapshot.objects:
                    native = item.attributes.get("native_track")
                    state = item.observation_state or item.attributes.get(
                        "observation_state", "observed"
                    )
                    seen_at = item.last_observed_timestamp_seconds
                    if seen_at is None:
                        seen_at = item.attributes.get(
                            "last_observed_timestamp_seconds",
                            snapshot.timestamp_seconds,
                        )
                    seen_sequence = item.last_observed_frame_sequence
                    if seen_sequence is None:
                        seen_sequence = item.attributes.get(
                            "last_observed_frame_sequence", snapshot.frame_sequence
                        )
                    if (
                        item.label.lower() not in _PERSON_LABELS
                        or state != "observed"
                        or type(seen_at) not in (int, float)
                        or seen_at != snapshot.timestamp_seconds
                        or type(seen_sequence) is not int
                        or seen_sequence != snapshot.frame_sequence
                        or not isinstance(native, Mapping)
                        or set(native) != {"backend", "epoch", "object_id"}
                        or native.get("backend") != _NATIVE_VIDEO_BACKEND
                        or any(
                            not isinstance(native.get(key), str)
                            or not native[key].strip()
                            or native[key] != native[key].strip()
                            or len(native[key]) > 256
                            for key in ("backend", "epoch", "object_id")
                        )
                    ):
                        return self._reject_tracking_batch("invalid_native_observation")
                    key = tuple(
                        native[name] for name in ("backend", "epoch", "object_id")
                    )
                    if key in observations:
                        return self._reject_tracking_batch(
                            "duplicate_native_observation"
                        )
                    observations[key] = item.bbox
                    native_keys.add(key)
                native_frames.append(observations)
            if len({key[:2] for key in native_keys}) > 1:
                return self._reject_tracking_batch("mixed_native_epochs")
            # Persistent native sessions reuse keys across commits. Reserve only
            # unseen keys, before the callback can mutate canonical identities.
            if not self.scene.can_reserve_native_person_slots(native_keys):
                return self._reject_tracking_batch("subject_capacity")
            canonical = canonicalize(raw_snapshots)
            if not isinstance(canonical, (tuple, list)) or len(canonical) != len(
                raw_snapshots
            ):
                raise ValueError("canonical mapping must preserve all source frames")
            native_subjects = {}
            for raw, mapped, expected in zip(
                raw_snapshots, canonical, native_frames, strict=True
            ):
                if not isinstance(mapped, RichLabelSnapshot) or (
                    mapped.stream_session_id,
                    mapped.timestamp_seconds,
                    mapped.frame_sequence,
                ) != (raw.stream_session_id, raw.timestamp_seconds, raw.frame_sequence):
                    raise ValueError("canonical mapping changed a source frame")
                actual = {}
                for item in mapped.objects:
                    native = item.attributes.get("native_track")
                    if not isinstance(native, Mapping):
                        raise ValueError("canonical mapping removed native provenance")
                    key = tuple(
                        native.get(name) for name in ("backend", "epoch", "object_id")
                    )
                    if (
                        key in actual
                        or expected.get(key) != item.bbox
                        or (
                            item.observation_state
                            or item.attributes.get("observation_state", "observed")
                        )
                        != "observed"
                        or native_subjects.get(key, item.track_id) != item.track_id
                    ):
                        raise ValueError(
                            "canonical mapping changed a source observation"
                        )
                    actual[key] = item.bbox
                    native_subjects[key] = item.track_id
                if actual != expected:
                    raise ValueError("canonical mapping omitted a source observation")
            if len(set(native_subjects.values())) != len(native_subjects):
                raise ValueError("canonical mapping merged distinct native people")
            return self.observe_tracking_batch(
                canonical,
                completed_timestamp_seconds=completed_timestamp_seconds,
                completed_frame_sequence=completed_frame_sequence,
            )

    def _enqueue_tracking_intervals(
        self, snapshots: Sequence[RichLabelSnapshot]
    ) -> None:
        if not self.event_labels:
            return
        # Native IDs are scoped to this completed batch. Independent, disjoint
        # source windows let an event collect the usual two observations without
        # falsely joining people across a tracker epoch or counting one frame twice.
        midpoint = len(snapshots) // 2
        windows = (snapshots,)
        if (
            len(snapshots) >= 8
            and snapshots[midpoint - 1].timestamp_seconds
            - snapshots[0].timestamp_seconds
            >= 0.75
            and snapshots[-1].timestamp_seconds - snapshots[midpoint].timestamp_seconds
            >= 0.75
        ):
            windows = (snapshots[:midpoint], snapshots[midpoint:])
        self._drain_tracking_intervals()
        for window in windows:
            observed = [
                snapshot
                for snapshot in window
                if any(
                    item.label.lower() in _PERSON_LABELS
                    and (
                        item.observation_state
                        or item.attributes.get("observation_state", "observed")
                    )
                    == "observed"
                    for item in snapshot.objects
                )
            ]
            if (
                len(observed) < 2
                or observed[-1].timestamp_seconds - observed[0].timestamp_seconds < 0.75
            ):
                self._reject_tracking_interval("insufficient_observed_sources")
                continue
            if len(self._tracking_intervals) >= 4:
                self._reject_tracking_interval("queue_capacity")
                continue
            if len(observed) > 4:
                observed = [
                    observed[round(index * (len(observed) - 1) / 3)]
                    for index in range(4)
                ]
            request = InspectionRequest(
                f"video-inspection-{uuid4().hex}",
                (),
                "full_scene_scan",
                self.event_labels,
                window[0].timestamp_seconds,
                window[-1].timestamp_seconds,
            )
            self._tracking_intervals[request.request_id] = _TrackingInterval(
                request,
                tuple(
                    (item.frame_sequence, item.timestamp_seconds) for item in observed
                ),
                tuple(
                    (snapshot.frame_sequence, item.track_id, item.bbox)
                    for snapshot in observed
                    for item in snapshot.objects
                    if item.label.lower() in _PERSON_LABELS
                    and (
                        item.observation_state
                        or item.attributes.get("observation_state", "observed")
                    )
                    == "observed"
                ),
            )
            self._stats["queued_tracking_intervals"] += 1
            self._drain_tracking_intervals()

    def _drain_tracking_intervals(self) -> None:
        if self._closed or self.tracking_mode != "batch":
            return
        blocked = False
        for request_id, interval in tuple(self._tracking_intervals.items()):
            frames = tuple(
                self.evidence.source_frame(
                    stream_session_id=self.stream_session_id,
                    frame_sequence=sequence,
                    timestamp_seconds=timestamp,
                )
                for sequence, timestamp in interval.source_frames
            )
            if any(frame is None for frame in frames):
                self._tracking_intervals.pop(request_id)
                self._reject_tracking_interval("source_frame_missing_or_expired")
                continue
            interval_bytes = sum(len(frame.jpeg_bytes) for frame in frames)
            if interval_bytes > 16 * 1024 * 1024:
                self._tracking_intervals.pop(request_id)
                self._reject_tracking_interval("image_budget")
                continue
            retained_bytes = sum(
                len(frame.jpeg_bytes)
                for job in self._jobs.values()
                for frame in job.frames
            )
            if (
                blocked
                or len(self._jobs) >= 4
                or retained_bytes + interval_bytes > 16 * 1024 * 1024
            ):
                # Preserve FIFO under pressure. Queued intervals hold source keys,
                # not JPEG copies, and are revalidated after a slot becomes free.
                blocked = True
                continue
            frame_times = dict(interval.source_frames)
            if not self.scene.pin_source_observations(
                request_id,
                tuple(
                    (sid, frame_times[sequence], sequence, bbox)
                    for sequence, sid, bbox in interval.subject_boxes
                ),
            ):
                self._tracking_intervals.pop(request_id)
                self._reject_tracking_interval("source_observation_expired")
                continue
            self._jobs[request_id] = TemporalInspection(
                self.stream_session_id,
                self.source_id,
                interval.request,
                frames,
                interval.subject_boxes,
                self.event_sensitivities,
            )
            self._tracking_intervals.pop(request_id)
            self._stats["prepared_inspections"] += 1

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
            if self.tracking_mode == "batch" and at > self._now:
                return
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
            if self.tracking_mode == "immediate":
                self._now = max(self._now, at)
            self.scene.advance(self._now)
            self._pending_results[result.message_id] = result
            self._pending_results.move_to_end(result.message_id)
            while len(self._pending_results) > 128:
                self._pending_results.popitem(last=False)
            self._retry_results()
            if self.tracking_mode == "immediate":
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

    def _planning_timestamp(self) -> float:
        if self.tracking_mode == "batch":
            return self._tracking_completed_timestamp or 0.0
        return self._now

    def _expire_inspections(self) -> None:
        for request_id, job in tuple(self._jobs.items()):
            if self._now - job.request.end_seconds >= 20.0:
                self._jobs.pop(request_id)
                self._claimed.discard(request_id)
                self.scene.unpin_source_observations(request_id)
                self.planner.complete(
                    request_id,
                    success=False,
                    timestamp_seconds=self._planning_timestamp(),
                )

    def _plan(self) -> None:
        if self.tracking_mode == "batch":
            self._drain_tracking_intervals()
            return
        if not self.event_labels or (
            self.tracking_mode == "batch" and self._tracking_completed_timestamp is None
        ):
            return
        self._expire_inspections()
        planning_timestamp = self._planning_timestamp()
        if len(self._jobs) >= 4 or len(self.evidence) < 2:
            return
        if self.mode == "active":
            recent = self.evidence.select(
                stream_session_id=self.stream_session_id,
                start_seconds=max(0.0, planning_timestamp - 3.0),
                end_seconds=planning_timestamp,
                limit=48,
                frame_sequences=(
                    set(self._tracking_frames)
                    if self.tracking_mode == "batch"
                    else None
                ),
            )
            if (
                len(recent) < 2
                or recent[-1].timestamp_seconds - recent[0].timestamp_seconds < 0.75
            ):
                return
        planning_scene = (
            self.scene.snapshot_at(planning_timestamp)
            if self.tracking_mode == "batch"
            else self.scene.snapshot(include_history=False)
        )
        planning_scene["events"] = [
            event
            for event in planning_scene["events"]
            if event["event_label"] in self.event_labels
        ]
        for request in self.planner.plan(
            planning_scene, timestamp_seconds=planning_timestamp
        ):
            request = replace(
                request,
                event_labels=tuple(
                    label
                    for label in (request.event_labels or self.event_labels)
                    if label in self.event_labels
                ),
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
                and (
                    self.tracking_mode != "batch"
                    or self._tracking_frames.get(record["frame_sequence"])
                    == record["timestamp_seconds"]
                )
            }
            sequences = {sequence for sequence, _ in boxes}
            if request.subject_ids:
                sequences = {
                    sequence
                    for sequence in sequences
                    if all((sequence, sid) in boxes for sid in request.subject_ids)
                }
            frames = self.evidence.select(
                stream_session_id=self.stream_session_id,
                start_seconds=request.start_seconds,
                end_seconds=request.end_seconds,
                limit=4,
                frame_sequences=sequences,
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
                    request.request_id,
                    success=False,
                    timestamp_seconds=planning_timestamp,
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
                    request.request_id,
                    success=False,
                    timestamp_seconds=planning_timestamp,
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
                request_id,
                success=success,
                timestamp_seconds=self._planning_timestamp(),
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
            self._drain_tracking_intervals()
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
            self._tracking_frames.clear()
            self._tracking_intervals.clear()

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

    def has_pending_inspections(self) -> bool:
        """Read pending work without copying retained source trajectories."""
        with self._lock:
            return bool(self._jobs or self._tracking_intervals)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "mode": self.mode,
                "tracking_mode": self.tracking_mode,
                "memory_scope": self.native_memory_scope,
                "capture_timestamp_seconds": self._now,
                "tracking_completed_timestamp_seconds": (
                    self._tracking_completed_timestamp
                ),
                "tracking_completed_frame_sequence": self._tracking_completed_sequence,
                "tracking_lag_seconds": (
                    None
                    if self._tracking_completed_timestamp is None
                    else max(0.0, self._now - self._tracking_completed_timestamp)
                ),
                "tracking_rejections": dict(self._tracking_rejections),
                "tracking_last_rejection": self._tracking_last_rejection,
                "tracking_interval_rejections": dict(
                    self._tracking_interval_rejections
                ),
                "pending_tracking_intervals": [
                    interval.request.as_dict()
                    for interval in self._tracking_intervals.values()
                ],
                "evidence_buffer": {
                    "frames": len(self.evidence),
                    "bytes": self.evidence.byte_count,
                    "max_frames": self.evidence.max_frames,
                    "max_bytes": self.evidence.max_bytes,
                    "history_seconds": self.evidence.history_seconds,
                },
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
