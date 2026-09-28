"""Bounded continuous scene state for the active live pipeline.

Prepare temporal inspections from real source observations. A separate worker
claims these immutable intervals and completes them without running model calls
under video callback locks. Safety verdicts come exclusively from temporal
inspections; face observations supply identities and stranger judgments.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from threading import RLock
from time import monotonic
from typing import Any
from uuid import uuid4

from .contracts import AnalysisResult, NormalizedBBox
from .event_sensitivity import normalize_event_sensitivities
from .rich_labeling import RichLabelSnapshot
from .scene_alerts import apply_alert_policy
from .scene_evidence import (
    EvidenceFrame,
    InspectionPlanner,
    InspectionRequest,
    SceneEvidenceBuffer,
)
from .scene_state import SceneState
from .video_tracking import BACKEND as _NATIVE_VIDEO_BACKEND
from .video_tracking import VideoFrame

_EVENTS = frozenset((
    "fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter", "stranger",
))
_INSPECTION_TIMEOUT_SECONDS = 20.0
_TEMPORAL_MIN_SPAN_SECONDS = 0.75
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
    evidence_after_frame_sequence: int | None = None
    grounding_required: bool = False

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
class _TemporalSource:
    frame: EvidenceFrame
    subject_boxes: tuple[tuple[int, str, NormalizedBBox], ...]
    epoch: tuple[tuple[str, str], ...]

    @property
    def pin_id(self) -> str:
        return f"temporal-source-{self.frame.frame_sequence}"


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
        mode: str = "active",
        event_labels: tuple[str, ...] = (
            "fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter",
        ),
        event_sensitivities: Mapping[str, str] | None = None,
        tracking_mode: str = "immediate",
        tracking_history_seconds: float = 12.0,
        native_memory_scope: str | None = None,
        temporal_observation_fps: float | None = None,
        max_pending_inspections: int = 4,
        inspection_image_budget_bytes: int = 16 * 1024 * 1024,
        stranger_enabled: bool = False,
        independent_temporal_inference: bool = False,
    ) -> None:
        if mode != "active":
            raise ValueError("scene mode must be active")
        if not set(event_labels) <= _EVENTS - {"stranger"}:
            raise ValueError("unsupported temporal event label")
        if tracking_mode not in {"immediate", "batch"}:
            raise ValueError("tracking mode must be immediate or batch")
        if native_memory_scope not in {None, "batch", "session"}:
            raise ValueError("native memory scope must be batch or session")
        if tracking_mode == "immediate" and native_memory_scope == "batch":
            raise ValueError("immediate tracking cannot have batch native memory")
        if temporal_observation_fps is not None and (
            type(temporal_observation_fps) not in (int, float)
            or not math.isfinite(temporal_observation_fps)
            or not 0 < temporal_observation_fps <= 3
            or tracking_mode != "batch"
            or native_memory_scope != "session"
        ):
            raise ValueError(
                "temporal cadence requires session tracking and 0 < fps <= 3"
            )
        if (
            type(max_pending_inspections) is not int
            or not 1 <= max_pending_inspections <= 32
        ):
            raise ValueError("pending inspection capacity must be within 1..32")
        if (
            type(inspection_image_budget_bytes) is not int
            or not 1 <= inspection_image_budget_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("inspection image budget must be within 1..64 MiB")
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
        self.temporal_observation_fps = temporal_observation_fps
        self.max_pending_inspections = max_pending_inspections
        self.inspection_image_budget_bytes = inspection_image_budget_bytes
        self.event_labels = tuple(dict.fromkeys(event_labels))
        self.stranger_enabled = stranger_enabled
        self._identity_reviews: dict[str, tuple[str, float]] = {}
        self.event_sensitivities = tuple(
            normalize_event_sensitivities(event_sensitivities, event_labels).items()
        )
        self.automatic_temporal_inference = False
        self.independent_temporal_inference = independent_temporal_inference
        self.scene = SceneState(
            stream_session_id,
            source_id,
            person_history_size=512 if tracking_mode == "batch" else None,
            person_history_seconds=(
                self.tracking_history_seconds if tracking_mode == "batch" else None
            ),
            max_source_pin_jobs=(
                64 if temporal_observation_fps is not None else max_pending_inspections
            ),
            retain_pinned_sources=temporal_observation_fps is not None,
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
            full_scene_interval_seconds=5.0,
            refresh_seconds=3.0,
        )
        self.stream_session_id = stream_session_id
        self.source_id = source_id
        self._lock = RLock()
        self._now = 0.0
        self._tracking_completed_timestamp: float | None = None
        self._tracking_completed_sequence: int | None = None
        self._tracking_sources_finished = False
        self._tracking_frames: OrderedDict[int, float] = OrderedDict()
        self._tracking_rejections: dict[str, int] = {}
        self._tracking_last_rejection: str | None = None
        self._tracking_intervals: OrderedDict[str, _TrackingInterval] = OrderedDict()
        self._tracking_interval_rejections: dict[str, int] = {}
        self._temporal_sources: OrderedDict[int, _TemporalSource] = OrderedDict()
        self._temporal_next_endpoint: float | None = None
        self._temporal_last_endpoint_sequence = -1
        self._temporal_finished = False
        self._retained_tracking_sources: dict[int, EvidenceFrame] = {}
        self._last_buffered = -math.inf
        self._closed = False
        self._jobs: OrderedDict[str, TemporalInspection] = OrderedDict()
        self._latest_inspection: TemporalInspection | None = None
        self._claimed: set[str] = set()
        self._inspection_deadlines: dict[str, float] = {}
        self._pending_results: OrderedDict[str, AnalysisResult] = OrderedDict()
        self._accepted: OrderedDict[tuple[str, str, str], _AcceptedEvidence] = (
            OrderedDict()
        )
        self._processed: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self._forgotten_identities: set[str] = set()
        self._identities_disabled = False
        self._subject_aliases: OrderedDict[str, str] = OrderedDict()
        self._stats = {
            "association_revisions_ignored": 0,
            "association_revisions_applied": 0,
            "prepared_inspections": 0,
            "completed_inspections": 0,
            "failed_inspections": 0,
            "expired_inspections": 0,
            "accepted_temporal_verdicts": 0,
            "rejected_temporal_verdicts": 0,
            "completed_tracking_batches": 0,
            "completed_tracking_frames": 0,
            "rejected_tracking_batches": 0,
            "queued_tracking_intervals": 0,
            "rejected_tracking_intervals": 0,
        }

    def _canonical_subject(self, subject_id: str) -> str:
        return self._subject_aliases.get(subject_id, subject_id)

    def _canonical_result(self, result: AnalysisResult) -> AnalysisResult:
        objects = []
        for item in result.objects:
            attributes = dict(item.attributes)
            ids = _subjects(attributes)
            if ids:
                ids = tuple(sorted({self._canonical_subject(sid) for sid in ids}))
                attributes["subject_ids"] = list(ids)
                if len(ids) == 1:
                    attributes.update(subject_id=ids[0], person_track_id=ids[0])
            objects.append(replace(item, attributes=attributes))
        return replace(result, objects=tuple(objects))

    def apply_subject_merges(self, merges: Sequence[tuple[str, str]]) -> int:
        """Apply only registry-proven source associations, within this session.

        Claimed model jobs stay immutable. Their response IDs are checked against
        the original request before trusted aliases are applied at completion.
        """
        return self._apply_subject_merges(merges, reviewed_identity=False)

    def apply_identity_merges(self, merges: Sequence[tuple[str, str]]) -> int:
        """Consolidate face-reviewed people while retaining native source tracks."""
        return self._apply_subject_merges(merges, reviewed_identity=True)

    def _apply_subject_merges(
        self, merges: Sequence[tuple[str, str]], *, reviewed_identity: bool
    ) -> int:
        if not isinstance(merges, (tuple, list)) or len(merges) > 4096:
            raise ValueError("subject merges must be bounded")
        if any(
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(
                not isinstance(sid, str) or not sid or len(sid) > 256 for sid in pair
            )
            for pair in merges
        ):
            raise ValueError("invalid subject merge")
        with self._lock:
            if self._closed:
                return 0
            count = 0
            for old_id, new_id in merges:
                old_id, new_id = (
                    self._canonical_subject(old_id),
                    self._canonical_subject(new_id),
                )
                if old_id == new_id:
                    continue
                # Registry never merges distinct native objects. Keep the same
                # guard at the scene boundary even for incorrectly wired callers.
                entities = {
                    item["subject_id"]: item
                    for item in self.scene.snapshot(include_history=False)["entities"]
                }
                if (
                    not reviewed_identity
                    and entities.get(old_id, {}).get("native_track")
                ):
                    continue
                applied = self.scene.merge_subject(
                    old_id, new_id, reviewed_identity=reviewed_identity
                )
                if old_id in entities and not applied:
                    continue
                self._subject_aliases = OrderedDict(
                    (key, new_id if value == old_id else value)
                    for key, value in self._subject_aliases.items()
                )
                self._subject_aliases[old_id] = new_id
                count += 1
            while len(self._subject_aliases) > 4096:
                self._subject_aliases.popitem(last=False)
            self._accepted = OrderedDict(
                (
                    key,
                    replace(
                        value,
                        subject_ids=tuple(
                            sorted(
                                {
                                    self._canonical_subject(sid)
                                    for sid in value.subject_ids
                                }
                            )
                        ),
                    ),
                )
                for key, value in self._accepted.items()
            )
            self._pending_results = OrderedDict(
                (key, self._canonical_result(value))
                for key, value in self._pending_results.items()
            )
            for key, interval in tuple(self._tracking_intervals.items()):
                self._tracking_intervals[key] = replace(
                    interval,
                    request=replace(
                        interval.request,
                        subject_ids=tuple(
                            sorted(
                                {
                                    self._canonical_subject(sid)
                                    for sid in interval.request.subject_ids
                                }
                            )
                        ),
                    ),
                    subject_boxes=tuple(
                        (sequence, self._canonical_subject(sid), box)
                        for sequence, sid, box in interval.subject_boxes
                    ),
                )
            for sequence, source in tuple(self._temporal_sources.items()):
                self._temporal_sources[sequence] = replace(
                    source, subject_boxes=tuple(
                        (frame, self._canonical_subject(sid), box)
                        for frame, sid, box in source.subject_boxes
                    ),
                )
            self._stats["subjects_merged"] = (
                self._stats.get("subjects_merged", 0) + count
            )
            if reviewed_identity:
                self._stats["identity_subjects_merged"] = (
                    self._stats.get("identity_subjects_merged", 0) + count
                )
            return count

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
                    self._canonical_subject(item.track_id),
                    timestamp_seconds=snapshot.timestamp_seconds,
                    frame_sequence=snapshot.frame_sequence,
                    bbox=item.bbox,
                    observation_state=state,
                    tracking_confirmation=item.attributes.get("tracking_confirmation"),
                    mask_contours=item.attributes.get("mask_contours"),
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

    @property
    def tracking_last_rejection(self) -> str | None:
        with self._lock:
            return self._tracking_last_rejection

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
                self._tracking_source_frame(
                    snapshot.frame_sequence, snapshot.timestamp_seconds,
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

            # Native track provenance was validated before this step. A reviewed
            # identity may span multiple sequential detector tracks; only the
            # scene ownership is canonicalized, never their source metadata.
            snapshots = tuple(
                replace(snapshot, objects=tuple(
                    replace(item, track_id=self._canonical_subject(item.track_id))
                    for item in snapshot.objects
                ))
                for snapshot in snapshots
            )
            if any(
                len({item.track_id for item in snapshot.objects})
                != len(snapshot.objects) for snapshot in snapshots
            ):
                return self._reject_tracking_batch("coexisting_reviewed_subjects")

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
                        or item.attributes.get("tracking_confirmation", "confirmed")
                        not in ("confirmed", "provisional")
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
            if (not self.independent_temporal_inference
                    and self.temporal_observation_fps is not None and self.event_labels):
                incoming = self._temporal_source_records(snapshots)
                if not self._temporal_sources_fit(incoming):
                    # Nothing from this batch has been committed. The owner must
                    # retain its validated SAM output and retry after capacity frees.
                    return self._reject_tracking_batch("inspection_capacity")
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
                        tracking_confirmation=item.attributes.get(
                            "tracking_confirmation"
                        ),
                        mask_contours=item.attributes.get("mask_contours"),
                    )
                self.scene.observe_tracking_presence(
                    tuple(
                        item.track_id for item in snapshot.objects
                        if item.label.lower() in _PERSON_LABELS
                        and (item.observation_state or item.attributes.get(
                            "observation_state", "observed",
                        )) == "observed"
                    ),
                    timestamp_seconds=snapshot.timestamp_seconds,
                    frame_sequence=snapshot.frame_sequence,
                )
                self._tracking_frames[snapshot.frame_sequence] = (
                    snapshot.timestamp_seconds
                )
            self._tracking_completed_timestamp = float(completed_timestamp_seconds)
            self._tracking_completed_sequence = completed_frame_sequence
            self._stats["completed_tracking_batches"] += 1
            self._stats["completed_tracking_frames"] += len(snapshots)
            self._tracking_last_rejection = None
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

    def _tracking_source_frame(
        self, sequence: int, timestamp: float,
    ) -> EvidenceFrame | None:
        frame = self.evidence.source_frame(
            stream_session_id=self.stream_session_id, frame_sequence=sequence,
            timestamp_seconds=timestamp,
        )
        if frame is None:
            frame = self._retained_tracking_sources.get(sequence)
        if frame is None or frame.timestamp_seconds != timestamp:
            return None
        return frame

    def commit_native_tracking_batch(
        self,
        raw_snapshots: Sequence[RichLabelSnapshot],
        *,
        completed_timestamp_seconds: float,
        completed_frame_sequence: int,
        canonicalize: Callable[
            [Sequence[RichLabelSnapshot]], Sequence[RichLabelSnapshot]
        ],
        retained_sources: Sequence[VideoFrame] | None = None,
    ) -> bool:
        """Use the coordinator's exact unacknowledged sources for this commit only."""
        with self._lock:
            retained = {}
            if retained_sources is not None:
                if (
                    not isinstance(retained_sources, (tuple, list))
                    or len(retained_sources) > 512
                    or not isinstance(raw_snapshots, (tuple, list))
                    or any(not isinstance(item, RichLabelSnapshot)
                           or item.stream_session_id != self.stream_session_id
                           for item in raw_snapshots)
                ):
                    raise ValueError("retained sources must belong to this batch")
                expected = {
                    item.frame_sequence: item.timestamp_seconds
                    for item in raw_snapshots
                }
                for source in retained_sources:
                    if (
                        not isinstance(source, VideoFrame)
                        or source.frame_sequence in retained
                        or expected.get(source.frame_sequence)
                        != source.timestamp_seconds
                    ):
                        raise ValueError("retained source does not match batch")
                    existing = self.evidence.source_frame(
                        stream_session_id=self.stream_session_id,
                        frame_sequence=source.frame_sequence,
                        timestamp_seconds=source.timestamp_seconds,
                    )
                    if (
                        existing is not None
                        and existing.jpeg_bytes != source.jpeg_bytes
                    ):
                        raise ValueError("retained source changed original image")
                    retained[source.frame_sequence] = EvidenceFrame(
                        self.stream_session_id, source.frame_sequence,
                        source.timestamp_seconds, source.jpeg_bytes,
                    )
            self._retained_tracking_sources = retained
            try:
                return self._commit_native_tracking_batch(
                    raw_snapshots,
                    completed_timestamp_seconds=completed_timestamp_seconds,
                    completed_frame_sequence=completed_frame_sequence,
                    canonicalize=canonicalize,
                )
            finally:
                self._retained_tracking_sources = {}

    def _commit_native_tracking_batch(
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
            if (
                self.temporal_observation_fps is not None and self.event_labels
                and not self._temporal_sources_fit(
                    self._temporal_source_records(raw_snapshots)
                )
            ):
                return self._reject_tracking_batch("inspection_capacity")
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
                        or native.get("backend")
                        not in {_NATIVE_VIDEO_BACKEND, "person_hybrid", "goodvision_contour"}
                        or item.attributes.get("tracking_confirmation", "confirmed")
                        not in ("confirmed", "provisional")
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
                    observations[key] = (
                        item.bbox,
                        item.attributes.get("tracking_confirmation", "confirmed"),
                    )
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
                        or expected.get(key)
                        != (
                            item.bbox,
                            item.attributes.get("tracking_confirmation", "confirmed"),
                        )
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
                    actual[key] = (
                        item.bbox,
                        item.attributes.get("tracking_confirmation", "confirmed"),
                    )
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
        if self.independent_temporal_inference or not self.event_labels:
            return
        if self.temporal_observation_fps is not None:
            for source in self._temporal_source_records(snapshots):
                if not self.scene.pin_source_observations(
                    source.pin_id,
                    tuple((sid, source.frame.timestamp_seconds, sequence, bbox)
                          for sequence, sid, bbox in source.subject_boxes),
                ):
                    raise RuntimeError("temporal source could not be retained")
                self._temporal_sources[source.frame.frame_sequence] = source
            self._drain_temporal_windows()
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
                    and item.attributes.get("tracking_confirmation", "confirmed")
                    == "confirmed"
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
                    and item.attributes.get("tracking_confirmation", "confirmed")
                    == "confirmed"
                    and (
                        item.observation_state
                        or item.attributes.get("observation_state", "observed")
                    )
                    == "observed"
                ),
            )
            self._stats["queued_tracking_intervals"] += 1
            self._drain_tracking_intervals()

    def _temporal_source_records(
        self, snapshots: Sequence[RichLabelSnapshot],
    ) -> tuple[_TemporalSource, ...]:
        records = []
        for snapshot in snapshots:
            observed = tuple(
                item for item in snapshot.objects
                if item.label.lower() in _PERSON_LABELS
                and item.attributes.get("tracking_confirmation", "confirmed")
                == "confirmed"
                and (item.observation_state or item.attributes.get(
                    "observation_state", "observed",
                )) == "observed"
            )
            if not observed:
                continue
            frame = self._tracking_source_frame(
                snapshot.frame_sequence, snapshot.timestamp_seconds,
            )
            if frame is None:
                raise RuntimeError("validated temporal source is missing")
            records.append(_TemporalSource(
                frame,
                tuple((snapshot.frame_sequence, item.track_id, item.bbox)
                      for item in observed),
                tuple(sorted({
                    (str(native.get("backend", "")), str(native.get("epoch", "")))
                    for item in observed
                    if isinstance(
                        native := item.attributes.get("native_track"), Mapping
                    )
                })),
            ))
        return tuple(records)

    def _temporal_sources_fit(self, incoming: Sequence[_TemporalSource]) -> bool:
        # Strong JPEG references and real-observation pins survive the ordinary
        # source ring's pruning. Count unique frames, not shared context copies.
        sources = dict(self._temporal_sources)
        sources.update((source.frame.frame_sequence, source) for source in incoming)
        if len(sources) > 64 - self.max_pending_inspections:
            return False
        frames = {
            source.frame.frame_sequence: source.frame for source in sources.values()
        }
        frames.update(
            (frame.frame_sequence, frame)
            for job in self._jobs.values() for frame in job.frames
        )
        return sum(len(frame.jpeg_bytes) for frame in frames.values()) <= (
            self.inspection_image_budget_bytes
        )

    def _release_temporal_sources(self, sequences: Sequence[int]) -> None:
        for sequence in sequences:
            source = self._temporal_sources.pop(sequence)
            self.scene.unpin_source_observations(source.pin_id)

    def _drain_temporal_windows(self) -> None:
        if self._closed or self.temporal_observation_fps is None:
            return
        interval = 1.0 / self.temporal_observation_fps
        context_seconds = max(1.0, interval)
        while self._temporal_sources and len(self._jobs) < self.max_pending_inspections:
            sources = tuple(self._temporal_sources.values())
            epoch = sources[0].epoch
            boundary = next(
                (index for index, source in enumerate(sources)
                 if source.epoch != epoch),
                len(sources),
            )
            segment = sources[:boundary]
            final_segment = boundary < len(sources) or self._temporal_finished
            if self._temporal_next_endpoint is None:
                self._temporal_next_endpoint = (
                    segment[0].frame.timestamp_seconds + _TEMPORAL_MIN_SPAN_SECONDS
                )
            selected = ()
            for index, source in enumerate(segment):
                if source.frame.frame_sequence <= self._temporal_last_endpoint_sequence:
                    continue
                if source.frame.timestamp_seconds + 1e-9 < self._temporal_next_endpoint:
                    if not (final_segment and index == len(segment) - 1):
                        continue
                prefix = segment[:index + 1]
                oldest = source.frame.timestamp_seconds - context_seconds
                start = next(
                    position for position, candidate in enumerate(prefix)
                    if candidate.frame.timestamp_seconds >= oldest - 1e-9
                )
                # Sampling is quantized: at 3 FPS, four real frames often span
                # 1.008 seconds, not exactly one second. Keep the closest real
                # predecessor when clipping at the nominal boundary would leave
                # less than the required span. Never stretch over an arbitrary
                # source outage or invent a frame at the boundary.
                if start and (
                    source.frame.timestamp_seconds
                    - prefix[start].frame.timestamp_seconds
                    < _TEMPORAL_MIN_SPAN_SECONDS - 1e-9
                ) and (
                    prefix[start - 1].frame.timestamp_seconds
                    >= oldest - _TEMPORAL_MIN_SPAN_SECONDS - 1e-9
                ):
                    start -= 1
                context = prefix[start:]
                if len(context) < 2 or (
                    context[-1].frame.timestamp_seconds
                    - context[0].frame.timestamp_seconds
                    < _TEMPORAL_MIN_SPAN_SECONDS - 1e-9
                ):
                    continue
                selected = context if len(context) <= 4 else tuple(
                    context[round(position * (len(context) - 1) / 3)]
                    for position in range(4)
                )
                break
            if not selected:
                if final_segment:
                    if any(
                        source.frame.frame_sequence
                        > self._temporal_last_endpoint_sequence
                        for source in segment
                    ):
                        self._reject_tracking_interval("insufficient_epoch_sources")
                    self._release_temporal_sources(tuple(
                        source.frame.frame_sequence for source in segment
                    ))
                    self._temporal_next_endpoint = None
                    continue
                return
            frames = tuple(source.frame for source in selected)
            request = InspectionRequest(
                f"video-inspection-{uuid4().hex}", (), "full_scene_scan",
                self.event_labels, frames[0].timestamp_seconds,
                frames[-1].timestamp_seconds,
            )
            boxes = tuple(box for source in selected for box in source.subject_boxes)
            times = {frame.frame_sequence: frame.timestamp_seconds for frame in frames}
            if not self.scene.pin_source_observations(
                request.request_id,
                tuple((sid, times[sequence], sequence, bbox)
                      for sequence, sid, bbox in boxes),
            ):
                # No cursor movement or release: capacity/provenance failure
                # cannot quietly consume this accepted source window.
                self._tracking_last_rejection = "inspection_pin_capacity"
                return
            self._jobs[request.request_id] = TemporalInspection(
                self.stream_session_id, self.source_id, request, frames, boxes,
                self.event_sensitivities, self._temporal_last_endpoint_sequence,
            )
            self._stats["queued_tracking_intervals"] += 1
            self._stats["prepared_inspections"] += 1
            self._temporal_last_endpoint_sequence = frames[-1].frame_sequence
            while self._temporal_next_endpoint <= frames[-1].timestamp_seconds + 1e-9:
                self._temporal_next_endpoint += interval
            # Keep enough prior true frames for the next temporal context. Jobs
            # own their images/pins independently of this rolling source window.
            oldest = frames[-1].timestamp_seconds - context_seconds
            predecessor = next((
                sequence
                for sequence, source in reversed(self._temporal_sources.items())
                if oldest - _TEMPORAL_MIN_SPAN_SECONDS - 1e-9
                <= source.frame.timestamp_seconds < oldest - 1e-9
            ), None)
            self._release_temporal_sources(tuple(
                sequence for sequence, source in self._temporal_sources.items()
                if source.frame.timestamp_seconds < oldest - 1e-9
                and sequence != predecessor
            ))

    def finish_inspection_sources(self) -> None:
        """Flush the last valid partial cadence interval without fabricating frames."""
        with self._lock:
            self._temporal_finished = True
            self._drain_tracking_intervals()

    def _drain_tracking_intervals(self) -> None:
        if self._closed or self.tracking_mode != "batch":
            return
        if self.temporal_observation_fps is not None:
            self._drain_temporal_windows()
            return
        blocked = False
        for request_id, interval in tuple(self._tracking_intervals.items()):
            frames = tuple(
                self._tracking_source_frame(sequence, timestamp)
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
            self._inspection_deadlines[request_id] = (
                monotonic() + _INSPECTION_TIMEOUT_SECONDS
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
            result = self._canonical_result(result)
            if any(item.event_label == "stranger" for item in result.judgments):
                self.stranger_enabled = True
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
                review = self._identity_reviews.get(ids[0])
                if (item.attributes.get("face_observation_state") == "observed"
                        and (review is None or at > review[1]
                             or item.attributes.get("status") == "unusable")):
                    self._identity_reviews.pop(ids[0], None)
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
        known_subjects = {
            item["subject_id"] for item in snapshot["entities"]
            if item.get("identity", {}).get("status") == "known"
        }
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
                if label != "stranger":
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
                            detected = judgment.detected if ids else None
                            if detected is True and (
                                item.attributes.get("face_observation_state")
                                != "observed"
                                or item.attributes.get("identity_source")
                                != "face_match"
                                or item.attributes.get("status") != "unknown"
                                or set(ids) & known_subjects
                            ):
                                detected = None
                            desired[f"object:{object_id}"] = (
                                ids,
                                detected,
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
        execution_now = monotonic() if self.tracking_mode == "batch" else None
        for request_id, job in tuple(self._jobs.items()):
            expired = (
                request_id in self._inspection_deadlines
                and execution_now >= self._inspection_deadlines[request_id]
                if self.tracking_mode == "batch"
                else self._now - job.request.end_seconds >= _INSPECTION_TIMEOUT_SECONDS
            )
            if expired:
                self._jobs.pop(request_id)
                self._claimed.discard(request_id)
                self._inspection_deadlines.pop(request_id, None)
                self._stats["expired_inspections"] += 1
                self.scene.unpin_source_observations(request_id)
                self.planner.complete(
                    request_id,
                    success=False,
                    timestamp_seconds=self._planning_timestamp(),
                )

    def _plan(self) -> None:
        if self.independent_temporal_inference:
            return
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
        planning_scene["entities"] = [
            entity
            for entity in planning_scene["entities"]
            if entity.get("tracking_confirmation", "confirmed") == "confirmed"
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
                and record.get("tracking_confirmation", "confirmed") == "confirmed"
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
                frame_sequences=(
                    None if not request.subject_ids
                    and "ground_litter" in request.event_labels else sequences
                ),
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
            self._expire_inspections()
            return not self._closed and request_id in self._jobs

    def take_inspections(self, limit: int = 4) -> tuple[TemporalInspection, ...]:
        """Explicit worker seam; claiming never calls a model on the live thread."""
        if type(limit) is not int or not 1 <= limit <= 4:
            raise ValueError("inspection limit must be within 1..4")
        with self._lock:
            if self._closed:
                return ()
            self._expire_inspections()
            if self.tracking_mode == "batch":
                self._drain_tracking_intervals()
            jobs = tuple(
                job for key, job in self._jobs.items() if key not in self._claimed
            )[:limit]
            self._claimed.update(job.request.request_id for job in jobs)
            if self.tracking_mode == "batch":
                # Arrived source age includes native hotstart/network delay.
                # Give a newly claimed immutable job its own bounded model
                # budget without moving capture time or its evidence timestamps.
                deadline = monotonic() + _INSPECTION_TIMEOUT_SECONDS
                for job in jobs:
                    self._inspection_deadlines[job.request.request_id] = deadline
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
            self._expire_inspections()
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
                    or (not verdict.subject_ids
                        and verdict.event_label != "ground_litter")
                    or (verdict.event_label == "ground_litter"
                        and bool(verdict.subject_ids))
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
            self._inspection_deadlines.pop(request_id, None)
            accepted_count = 0
            try:
                if success:
                    # The pinned real source frames remain grounded even when
                    # new tracking observations have rolled past the history limit.
                    source_people: dict[int, set[str]] = {}
                    for sequence, subject_id, _bbox in job.subject_boxes:
                        source_people.setdefault(sequence, set()).add(subject_id)
                    canonical_verdicts: dict[tuple[str, tuple[str, ...]], list] = {}
                    for verdict in verdicts:
                        key = (
                            verdict.event_label,
                            tuple(
                                sorted(
                                    {
                                        self._canonical_subject(sid)
                                        for sid in verdict.subject_ids
                                    }
                                )
                            ),
                        )
                        canonical_verdicts.setdefault(key, []).append(verdict)
                    completed_groups = set()
                    for index, verdict in enumerate(verdicts):
                        candidates = (
                            job.frames[-1:] if job.request.subject_ids else job.frames
                        )
                        if job.evidence_after_frame_sequence is not None:
                            candidates = tuple(
                                frame for frame in candidates
                                if frame.frame_sequence
                                > job.evidence_after_frame_sequence
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
                        participants = tuple(
                            sorted(
                                {
                                    self._canonical_subject(sid)
                                    for sid in verdict.subject_ids
                                }
                            )
                        )
                        if (
                            verdict.event_label == "fight"
                            and len(participants) < 2
                            and len(set(verdict.subject_ids)) >= 2
                        ):
                            continue
                        group = (verdict.event_label, participants)
                        if group in completed_groups:
                            continue
                        completed_groups.add(group)
                        matching = canonical_verdicts[group]
                        detected = verdict.detected
                        confidence = verdict.confidence
                        if any(other.detected is not detected for other in matching):
                            detected, confidence = None, None
                        accepted_count += self.scene.observe_event(
                            verdict.event_label,
                            participants,
                            timestamp_seconds=newest.timestamp_seconds,
                            frame_sequence=newest.frame_sequence,
                            detected=detected,
                            confidence=confidence,
                            evidence_id=f"inspection:{request_id}:{index}",
                            allow_historical=True,
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

    def finish_tracking_sources(self):
        """No new tracking result can arrive after the coordinator has drained."""
        with self._lock:
            self._tracking_sources_finished = True

    def unavailable_display_sources(
        self, sources: Sequence[tuple[int, float]]
    ) -> tuple[int, ...]:
        """Identify arrived sources that can no longer produce an exact display.

        Capture retains the JPEG before submitting its display stamp. A missing
        JPEG has therefore been rejected/evicted; a missing completed stamp behind
        the native watermark has been skipped. Neither requires waiting at EOF.
        """
        if not isinstance(sources, (tuple, list)) or len(sources) > 512:
            raise ValueError("display source query must be bounded")
        for source in sources:
            if not isinstance(source, (tuple, list)) or len(source) != 2:
                raise ValueError("invalid display source stamp")
            sequence, timestamp = source
            if (
                type(sequence) is not int
                or sequence < 0
                or type(timestamp) not in (int, float)
                or not math.isfinite(timestamp)
                or timestamp < 0
            ):
                raise ValueError("invalid display source stamp")
        with self._lock:
            return tuple(
                sequence
                for sequence, timestamp in sources
                if self._closed
                or self.evidence.source_frame(
                    stream_session_id=self.stream_session_id,
                    frame_sequence=sequence,
                    timestamp_seconds=timestamp,
                )
                is None
                or (
                    (self._tracking_sources_finished or (
                        self._tracking_completed_sequence is not None
                        and sequence <= self._tracking_completed_sequence))
                    and self._tracking_frames.get(sequence) != timestamp
                )
            )

    def tracked_display_frame(
        self,
        *,
        before_timestamp_seconds: float,
        after_frame_sequence: int | None = None,
    ) -> tuple[EvidenceFrame, dict[str, Any]] | None:
        """Select a retained completed source JPEG and its exact-source view.

        This read does not advance any clock. The caller controls monotonic
        presentation with after_frame_sequence; missing/expired evidence is not
        replaced by a nearby capture frame or held native geometry.
        """
        if (
            type(before_timestamp_seconds) not in (int, float)
            or not math.isfinite(before_timestamp_seconds)
            or before_timestamp_seconds < 0
        ):
            raise ValueError("display source timestamp must be finite and nonnegative")
        if after_frame_sequence is not None and (
            type(after_frame_sequence) is not int or after_frame_sequence < -1
        ):
            raise ValueError("display source sequence must be an integer >= -1")
        with self._lock:
            if self._closed or self._tracking_completed_timestamp is None:
                return None
            ceiling = min(
                before_timestamp_seconds, self._now, self._tracking_completed_timestamp
            )
            for sequence, timestamp in reversed(self._tracking_frames.items()):
                if timestamp > ceiling or (
                    after_frame_sequence is not None
                    and sequence <= after_frame_sequence
                ):
                    continue
                frame = self.evidence.source_frame(
                    stream_session_id=self.stream_session_id,
                    frame_sequence=sequence,
                    timestamp_seconds=timestamp,
                )
                if frame is None:
                    continue
                view = self.display_snapshot(timestamp, frame_sequence=sequence)
                return frame, view
            return None

    def display_snapshot(
        self, timestamp_seconds: float, *, frame_sequence: int | None = None,
        source_entities: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Pair source-aligned geometry with the latest accepted event state.

        Presentation must surface late verdicts immediately. The separate event
        clock deliberately does not change the source time of boxes or masks;
        evidence/planning keep using SceneState.snapshot_at unchanged.
        """
        metadata = getattr(self, "display_metadata", None)
        if metadata is not None:
            view = metadata.display_snapshot(timestamp_seconds)
            view["state"] = "stopped" if self._closed else "running"
            return view
        with self._lock:
            view = self.scene.snapshot_at(timestamp_seconds)
            current = self._alert_snapshot(include_history=False)
            if source_entities is not None:
                # A displayed source can outlive geometry history. Keep its
                # positions, but follow registry-proven identity associations.
                identities = {
                    entity["subject_id"]: entity["identity"]
                    for entity in current["entities"]
                }
                retained = {}
                for entity in sorted(source_entities, key=lambda item: (
                    bool(item.get("native_track")), bool(item.get("mask_contours")),
                    item["subject_id"] == self._canonical_subject(item["subject_id"]),
                )):
                    subject = self._canonical_subject(entity["subject_id"])
                    retained[subject] = {
                        **entity, "subject_id": subject, "entity_id": subject,
                        "identity": identities.get(subject, {}),
                    }
                view["entities"] = list(retained.values())
            view["events"] = current["events"]
            identity_states = {item["subject_id"]: item for item in current["entities"]}
            for entity in view["entities"]:
                state = identity_states.get(entity["subject_id"], {})
                for key in ("identity_pending", "identity_review_state"):
                    entity[key] = state.get(key)
            view["event_state_timestamp_seconds"] = current["timestamp_seconds"]
            if frame_sequence is not None:
                exact_people = set()
                for entity in view["entities"]:
                    if (
                        entity["frame_sequence"] == frame_sequence
                        and entity["last_observed_timestamp_seconds"]
                        == timestamp_seconds
                    ):
                        exact_people.add(entity["subject_id"])
                    else:
                        entity["observation_state"] = "occluded"
                for event in view["events"]:
                    if not set(event["subject_ids"]) <= exact_people:
                        event["observation_state"] = "uncertain"
            view["capture_timestamp_seconds"] = self._now
            view["tracking_completed_timestamp_seconds"] = (
                self._tracking_completed_timestamp
            )
            view["state"] = "stopped" if self._closed else "running"
            return view

    def close(self) -> None:
        """Stop observations and release images; loss is not an event clear."""
        with self._lock:
            self.scene.finish_review()
            self._closed = True
            for request_id in self._jobs:
                self.scene.unpin_source_observations(request_id)
            self._release_temporal_sources(tuple(self._temporal_sources))
            self._jobs.clear()
            self._latest_inspection = None
            self._claimed.clear()
            self._inspection_deadlines.clear()
            self._pending_results.clear()
            self.evidence.reset()
            self._tracking_frames.clear()
            self._tracking_intervals.clear()

    def forget_identity(self, person_id: str) -> None:
        with self._lock:
            self.scene.forget_identity(person_id)
            self._identity_reviews.clear()
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

    def apply_identity_review(self, states: Mapping[str, str]) -> None:
        with self._lock:
            self._identity_reviews = {
                self._canonical_subject(sid): (state, self._now)
                for sid, state in states.items()
                if state in {"known", "unknown", "pending", "unresolved"}
            }

    def _alert_snapshot(self, *, include_history: bool) -> dict:
        return apply_alert_policy(
            self.scene.snapshot(include_history=include_history),
            stranger_enabled=self.stranger_enabled,
            identity_reviews=self._identity_reviews,
        )

    def snapshot(self, *, include_history: bool = True) -> dict[str, Any]:
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
                "tracking_sources_finished": self._tracking_sources_finished,
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
                "inspection_expiry_clock": (
                    "monotonic_execution"
                    if self.tracking_mode == "batch"
                    else "source_time"
                ),
                "inspection_timeout_seconds": _INSPECTION_TIMEOUT_SECONDS,
                "independent_temporal_inference": self.independent_temporal_inference,
                "temporal_observation_fps": self.temporal_observation_fps,
                "max_pending_inspections": self.max_pending_inspections,
                "retained_temporal_sources": len(self._temporal_sources),
                "pending_temporal_endpoint": self._temporal_next_endpoint,
                "temporal_last_endpoint_sequence": (
                    self._temporal_last_endpoint_sequence
                ),
                "inspection_capacity_blocked": self._tracking_last_rejection in {
                    "inspection_capacity", "inspection_pin_capacity",
                },
                "event_labels": list(self.event_labels),
                "scene": self._alert_snapshot(include_history=include_history),
                "inspections": [
                    job.as_dict()
                    | {
                        "state": "claimed" if key in self._claimed else "prepared",
                    }
                    for key, job in self._jobs.items()
                ],
                "stats": dict(self._stats),
            }
