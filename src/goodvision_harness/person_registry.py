"""Session-local subject association shared by faces, actors and GPU detections.

Every operation is bounded, in-memory and non-blocking with respect to inference.
Held display boxes are never new observations. Public IDs belong to this registry,
not to a detector or to response-local model actor IDs.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import RLock
from uuid import uuid4

from .contracts import AnalysisResult, NormalizedBBox
from .face_tracking import FaceIdentityTracker
from .grounding import ModelActor, ModelObject, ObservationBundle
from .identity_review import IdentityReview
from .rich_labeling import RichLabelObject, RichLabelSnapshot

_PERSON_LABELS = frozenset(
    {
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
    }
)
_SOURCE_FACE_MAX_GAP_SECONDS = 0.6
_NATIVE_BACKENDS = frozenset({"sam3.1_video", "person_hybrid"})


def _native_metadata(value: object) -> tuple[str, str, str] | None:
    if not isinstance(value, Mapping) or set(value) != {
        "backend",
        "epoch",
        "object_id",
    }:
        return None
    parts = tuple(value[key] for key in ("backend", "epoch", "object_id"))
    if (
        any(
            not isinstance(part, str)
            or not part.strip()
            or len(part) > 256
            or part != part.strip()
            for part in parts
        )
        or parts[0] not in _NATIVE_BACKENDS
    ):
        return None
    return parts


@dataclass(frozen=True, slots=True)
class _SourceFace:
    subject_id: str
    bbox: NormalizedBBox
    person_id: str | None = None


@dataclass(frozen=True, slots=True)
class _FaceSnapshot:
    timestamp: float
    sequence: int
    faces: tuple[_SourceFace, ...]


@dataclass(slots=True)
class _Subject:
    subject_id: str
    bbox: NormalizedBBox
    seen_at: float
    kind: str = "body"
    vx: float = 0.0
    vy: float = 0.0
    identity: dict[str, object] = field(default_factory=dict)
    identity_at: float = -1.0
    identity_embeddings: tuple[tuple[float, ...], ...] = ()
    memory_candidate: str | None = None
    memory_confirmations: int = 0
    memory_frame: tuple[float, int] = (-1.0, -1)
    association_uncertain: bool = False
    native_track: tuple[str, str, str] | None = None


class PersonRegistry:
    """A single source/session's short-lived subject and identity history."""

    def __init__(
        self,
        stream_session_id: str = "",
        source_id: str = "",
        *,
        max_gap_seconds: float = 3.0,
        max_native_tracks: int = 4096,
    ) -> None:
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive and finite")
        if type(max_native_tracks) is not int or not 1 <= max_native_tracks <= 4096:
            raise ValueError("max_native_tracks must be within 1..4096")
        self.stream_session_id = stream_session_id
        self.source_id = source_id
        self.max_gap_seconds = max_gap_seconds
        self._namespace = uuid4().hex[:12]
        self._counter = 0
        self._subjects: dict[str, _Subject] = {}
        self._body_aliases: dict[str, str] = {}
        self._face_aliases: dict[str, str] = {}
        self._history: deque[RichLabelSnapshot] = deque(maxlen=1024)
        self._face_history: deque[_FaceSnapshot] = deque(maxlen=1024)
        self._pending: OrderedDict[str, tuple[AnalysisResult, ObservationBundle]] = (
            OrderedDict()
        )
        self._reassociating = False
        self._gpu_watermark = (-1.0, -1)
        self._actor_watermark = (-1.0, -1)
        self._latest_time = -1.0
        self._face_tracker = FaceIdentityTracker()
        self._removed_people: set[str] = set()
        self._native_epoch: tuple[str, str] | None = None
        self._native_tracks: dict[tuple[str, str, str, str, str], str] = {}
        self._max_native_tracks = max_native_tracks
        self._native_rejections: Counter[str] = Counter()
        self._subject_merges: OrderedDict[str, str] = OrderedDict()
        self._subject_aliases: OrderedDict[str, str] = OrderedDict()
        self._identity_review = IdentityReview()
        self._lock = RLock()

    def review_pending_identities(
        self, *, accept_merge: Callable[[str, str], bool] | None = None
    ) -> int:
        """Run private face comparisons on the service's background review worker."""
        with self._lock:
            return self._identity_review.review(accept_merge)

    def drain_identity_merges(self) -> tuple[tuple[str, str], ...]:
        with self._lock:
            return self._identity_review.drain()

    def identity_review_stats(self) -> dict:
        with self._lock:
            return self._identity_review.stats()

    def drain_subject_merges(self) -> tuple[tuple[str, str], ...]:
        """Consume trusted same-source face-to-native associations for scene state."""
        with self._lock:
            result = tuple(self._subject_merges.items())
            self._subject_merges.clear()
            return result

    def canonical_subject_id(self, subject_id: str) -> str:
        with self._lock:
            return self._identity_review.canonical(
                self._subject_aliases.get(subject_id, subject_id)
            )

    def _exact_source_face_matches(
        self, snapshot: RichLabelSnapshot, faces: tuple[_SourceFace, ...] | None = None
    ) -> dict[str, str]:
        """Require the same real frame and a unique match in both directions."""
        faces = (
            faces
            if faces is not None
            else next(
                (
                    item.faces
                    for item in self._face_history
                    if (item.timestamp, item.sequence)
                    == (snapshot.timestamp_seconds, snapshot.frame_sequence)
                ),
                (),
            )
        )
        bodies = [
            item
            for item in snapshot.objects
            if item.label.casefold() in {"person", "pedestrian"}
            and _native_metadata(item.attributes.get("native_track")) is not None
            and item.attributes.get("tracking_confirmation", "confirmed") == "confirmed"
            and (
                item.observation_state
                or item.attributes.get("observation_state", "observed")
            )
            == "observed"
        ]
        proposals = {}
        counts: Counter[str] = Counter()
        for body in bodies:
            candidates = [
                face
                for face in faces
                if _face_body_score(face.bbox, body.bbox) is not None
            ]
            counts.update(face.subject_id for face in candidates)
            if len(candidates) == 1:
                source_face = candidates[0]
                metadata = _native_metadata(body.attributes["native_track"])
                native_id = self._native_tracks.get(
                    (self.stream_session_id, self.source_id, *metadata)
                )
                current = self._subjects.get(native_id)
                if (
                    source_face.person_id
                    and current is not None
                    and current.identity.get("person_id")
                    and source_face.person_id != current.identity["person_id"]
                ):
                    self._native_rejections["face_identity_conflict"] += 1
                    continue
                proposals[body.track_id] = source_face.subject_id
        return {body: face for body, face in proposals.items() if counts[face] == 1}

    def _merge_face_subject(self, old_id: str, native_id: str) -> None:
        old, current = self._subjects.get(old_id), self._subjects.get(native_id)
        if (
            old is None
            and old_id
            and old_id not in self._subject_aliases
            and current is not None
            and current.native_track is not None
        ):
            # The exact source face is retained even after its live tracker
            # expires. Scene state may still hold its events and identity.
            old = _Subject(old_id, current.bbox, current.seen_at, "face")
            self._subjects[old_id] = old
        if (
            old is None
            or current is None
            or old_id == native_id
            or old.native_track is not None
            or old.kind != "face"
            or current.native_track is None
            or old.association_uncertain
        ):
            return
        # Conflicting established gallery identities require fresh identity
        # resolution; geometry alone cannot decide which identity was wrong.
        if (
            old.identity.get("person_id")
            and current.identity.get("person_id")
            and old.identity["person_id"] != current.identity["person_id"]
        ):
            self._native_rejections["face_identity_conflict"] += 1
            return
        if old.identity and old.identity_at >= current.identity_at:
            current.identity = dict(old.identity)
            current.identity_at = old.identity_at
            current.identity_embeddings = old.identity_embeddings
        self._subjects.pop(old_id)
        self._face_aliases = {
            key: native_id if value == old_id else value
            for key, value in self._face_aliases.items()
        }
        self._body_aliases = {
            key: native_id if value == old_id else value
            for key, value in self._body_aliases.items()
        }
        self._face_tracker.rebind_body(old_id, native_id)
        self._face_history = deque(
            (
                replace(
                    frame,
                    faces=tuple(
                        replace(face, subject_id=native_id)
                        if face.subject_id == old_id
                        else face
                        for face in frame.faces
                    ),
                )
                for frame in self._face_history
            ),
            maxlen=1024,
        )
        self._subject_merges[old_id] = native_id
        self._subject_aliases[old_id] = native_id
        self._identity_review.rebind(old_id, native_id)
        while len(self._subject_aliases) > 4096:
            self._subject_aliases.popitem(last=False)
        # Service drains after each result/batch. Bound abandoned consumers too.
        while len(self._subject_merges) > 4096:
            self._subject_merges.popitem(last=False)

    def _associate_source_faces(self, snapshot: RichLabelSnapshot) -> None:
        for native_id, face_id in self._exact_source_face_matches(snapshot).items():
            self._merge_face_subject(face_id, native_id)

    def begin_native_epoch(self, backend: str, epoch: str) -> bool:
        """Activate a host-validated video batch, including an empty batch.

        The host serializes activation and rejects obsolete batch callbacks. Native
        snapshots cannot switch epochs themselves. Reset discards active mappings,
        never transfers their identities, and retains immutable source history.
        """
        metadata = _native_metadata(
            {"backend": backend, "epoch": epoch, "object_id": "epoch-validation"}
        )
        if metadata is None:
            return False
        with self._lock:
            if self._native_epoch == (backend, epoch):
                return True
            retired = set(self._native_tracks.values())
            self._face_tracker.forget_bodies(retired)
            for subject_id in retired:
                self._subjects.pop(subject_id, None)
            self._body_aliases = {
                key: value
                for key, value in self._body_aliases.items()
                if value not in retired
            }
            self._face_aliases = {
                key: value
                for key, value in self._face_aliases.items()
                if value not in retired
            }
            self._face_history = deque(
                (
                    replace(
                        frame,
                        faces=tuple(
                            face
                            for face in frame.faces
                            if face.subject_id not in retired
                        ),
                    )
                    for frame in self._face_history
                ),
                maxlen=1024,
            )
            self._native_tracks.clear()
            self._native_epoch = backend, epoch
            return True

    def native_tracking_stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "backend": self._native_epoch[0] if self._native_epoch else None,
                "epoch": self._native_epoch[1] if self._native_epoch else None,
                "tracks": len(self._native_tracks),
                "capacity": self._max_native_tracks,
                "rejections": dict(self._native_rejections),
            }

    def forget_person(self, person_id: str) -> None:
        """Invalidate an enrollment immediately, including in-flight results."""
        with self._lock:
            self._removed_people.add(person_id)
            self._clear_person(person_id)

    def allow_person(self, person_id: str) -> None:
        """Permit a successfully re-enrolled ID to establish identity again."""
        with self._lock:
            self._removed_people.discard(person_id)

    def _clear_person(self, person_id: str) -> None:
        self._face_tracker.forget_person(person_id)
        self._identity_review.forget_person(person_id)
        for subject in self._subjects.values():
            if subject.identity.get("person_id") == person_id:
                subject.identity = {}
                subject.identity_at = -1.0
                subject.identity_embeddings = ()

    def _new_subject(
        self, bbox: NormalizedBBox, timestamp: float, kind: str = "body"
    ) -> _Subject:
        self._counter += 1
        subject = _Subject(
            f"subject-{self._namespace}-{self._counter}", bbox, timestamp, kind
        )
        self._subjects[subject.subject_id] = subject
        return subject

    def _expire(self, timestamp: float) -> None:
        self._latest_time = max(self._latest_time, timestamp)
        self._subjects = {
            key: value
            for key, value in self._subjects.items()
            if value.native_track is not None
            or timestamp - value.seen_at <= self.max_gap_seconds
        }
        self._body_aliases = {
            key: value
            for key, value in self._body_aliases.items()
            if value in self._subjects
        }
        self._face_aliases = {
            key: value
            for key, value in self._face_aliases.items()
            if value in self._subjects
        }

    def _replace_subject_aliases(
        self,
        old_id: str,
        new_id: str,
        timestamp: float,
        *,
        frame_sequence: int | None = None,
    ) -> None:
        self._body_aliases = {
            key: new_id if value == old_id else value
            for key, value in self._body_aliases.items()
        }
        self._face_aliases = {
            key: new_id if value == old_id else value
            for key, value in self._face_aliases.items()
        }
        self._face_tracker.rebind_body(old_id, new_id)
        self._history = deque(
            (
                replace(
                    snapshot,
                    objects=tuple(
                        replace(item, track_id=new_id)
                        if item.track_id == old_id
                        else item
                        for item in snapshot.objects
                    ),
                )
                if snapshot.timestamp_seconds == timestamp
                and (
                    frame_sequence is None or snapshot.frame_sequence == frame_sequence
                )
                else snapshot
                for snapshot in self._history
            ),
            maxlen=1024,
        )

    def observe_snapshot(self, snapshot: RichLabelSnapshot) -> None:
        with self._lock:
            if (
                self.stream_session_id
                and snapshot.stream_session_id != self.stream_session_id
            ):
                return
            if not self.stream_session_id:
                self.stream_session_id = snapshot.stream_session_id
            key = (snapshot.timestamp_seconds, snapshot.frame_sequence)
            if key <= self._gpu_watermark:
                return
            native = {}
            native_keys = set()
            for item in snapshot.objects:
                if "native_track" not in item.attributes:
                    continue
                metadata = _native_metadata(item.attributes["native_track"])
                if metadata is None or metadata in native_keys:
                    self._native_rejections["invalid_snapshot"] += 1
                    return
                if (
                    self._native_epoch is not None
                    and metadata[:2] != self._native_epoch
                ):
                    self._native_rejections["inactive_epoch"] += 1
                    return
                native[item.track_id] = metadata
                native_keys.add(metadata)
            if len({metadata[:2] for metadata in native.values()}) > 1:
                self._native_rejections["invalid_snapshot"] += 1
                return
            if (native or self._native_epoch is not None) and (
                key[0] <= self._gpu_watermark[0] or key[1] <= self._gpu_watermark[1]
            ):
                self._native_rejections["source_order"] += 1
                return
            if native and self._native_epoch is None:
                backend, epoch, _ = next(iter(native.values()))
                self.begin_native_epoch(backend, epoch)
            self._gpu_watermark = key
            observations = []
            native_objects = []
            source_faces = self._exact_source_face_matches(snapshot)
            for item in snapshot.objects:
                if item.label.casefold() not in {"person", "pedestrian"}:
                    continue
                state = getattr(item, "observation_state", None) or item.attributes.get(
                    "observation_state", "observed"
                )
                seen = getattr(item, "last_observed_timestamp_seconds", None)
                if seen is None:
                    seen = item.attributes.get(
                        "last_observed_timestamp_seconds", snapshot.timestamp_seconds
                    )
                if state != "observed" or type(seen) not in (float, int):
                    continue
                if (
                    not math.isfinite(seen)
                    or abs(float(seen) - snapshot.timestamp_seconds) > 0.001
                ):
                    continue
                if item.track_id in native:
                    if abs(float(seen) - snapshot.timestamp_seconds) > 1e-6:
                        self._native_rejections["invalid_observation"] += 1
                        continue
                    sequence = item.last_observed_frame_sequence
                    if sequence is None:
                        sequence = item.attributes.get(
                            "last_observed_frame_sequence", snapshot.frame_sequence
                        )
                    if type(sequence) is not int or sequence != snapshot.frame_sequence:
                        self._native_rejections["invalid_observation"] += 1
                        continue
                    metadata = native[item.track_id]
                    native_key = self.stream_session_id, self.source_id, *metadata
                    subject_id = self._native_tracks.get(native_key)
                    subject = self._subjects.get(subject_id)
                    if subject is None:
                        if len(self._native_tracks) >= self._max_native_tracks:
                            self._native_rejections["capacity"] += 1
                            continue
                        face_subject = self._subjects.get(
                            source_faces.get(item.track_id, "")
                        )
                        source_face_id = source_faces.get(item.track_id)
                        if (
                            face_subject is None
                            and source_face_id
                            and source_face_id not in self._subject_aliases
                        ):
                            # A real source face can outlive the short live face
                            # tracker. Reuse its public ID without restoring any
                            # expired gallery identity or guessing continuity.
                            face_subject = _Subject(
                                source_face_id,
                                item.bbox,
                                snapshot.timestamp_seconds,
                                "face",
                            )
                            self._subjects[source_face_id] = face_subject
                        subject = (
                            face_subject
                            if face_subject is not None
                            and face_subject.native_track is None
                            and face_subject.kind == "face"
                            and not face_subject.association_uncertain
                            else self._new_subject(
                                item.bbox, snapshot.timestamp_seconds
                            )
                        )
                        subject.kind = "body"
                        subject.native_track = metadata
                        self._native_tracks[native_key] = subject.subject_id
                    subject.bbox = item.bbox
                    subject.seen_at = snapshot.timestamp_seconds
                    native_objects.append(
                        replace(
                            item,
                            track_id=subject.subject_id,
                            label="person",
                            attributes={
                                **item.attributes,
                                "observation_state": "observed",
                                "last_observed_timestamp_seconds": seen,
                                "last_observed_frame_sequence": sequence,
                            },
                        )
                    )
                    continue
                observations.append((item.track_id, item.bbox, item.confidence or 0.0))
            assignments = self._observe_bodies(
                observations, snapshot.timestamp_seconds, worker=True
            )
            source_objects = {item.track_id: item for item in snapshot.objects}
            objects = tuple(native_objects) + tuple(
                replace(
                    source_objects[key],
                    track_id=assignments[key],
                    label="person",
                    bbox=box,
                    confidence=confidence,
                    attributes={
                        **source_objects[key].attributes,
                        "observation_state": "observed",
                        "last_observed_timestamp_seconds": snapshot.timestamp_seconds,
                    },
                )
                for key, box, confidence in observations
            )
            self._record_snapshot(
                snapshot.timestamp_seconds, snapshot.frame_sequence, objects
            )
            self._associate_source_faces(
                replace(snapshot, objects=tuple(native_objects))
            )

    def _observe_bodies(
        self,
        observations: Sequence[tuple[str, NormalizedBBox, float]],
        timestamp: float,
        *,
        worker: bool = False,
        excluded_subject_ids: set[str] | None = None,
    ) -> dict[str, str]:
        self._expire(timestamp)
        candidates = tuple(
            item
            for item in self._subjects.values()
            if item.subject_id not in (excluded_subject_ids or ())
            and item.native_track is None
        )
        assignments: dict[str, str] = {}
        proposals: dict[str, _Subject | None] = {}
        disputed: set[str] = set()
        uncertain_observations: set[str] = set()
        uncertain_candidates: dict[str, set[str]] = {}
        for local_id, box, _confidence in observations:
            scores: list[tuple[float, _Subject]] = []
            for subject in candidates:
                if (
                    timestamp < subject.seen_at
                    or timestamp - subject.seen_at > self.max_gap_seconds
                ):
                    continue
                score = (
                    _body_score(subject, box, timestamp)
                    if subject.kind == "body"
                    else _face_body_score(subject.bbox, box)
                )
                if score is not None:
                    if subject.association_uncertain:
                        uncertain_candidates.setdefault(subject.subject_id, set()).add(
                            local_id
                        )
                        continue
                    scores.append((score, subject))
            scores.sort(key=lambda item: item[0], reverse=True)
            proposals[local_id] = (
                scores[0][1]
                if scores and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.15)
                else None
            )
            if scores and proposals[local_id] is None:
                uncertain_observations.add(local_id)
                disputed.update(item.subject_id for _, item in scores[:2])
        counts = Counter(
            item.subject_id for item in proposals.values() if item is not None
        )
        disputed.update(key for key, count in counts.items() if count > 1)
        # Carry uncertainty across a close crossing while multiple current
        # people are reachable from one reset track. Simply forgetting the
        # previous dispute after one frame lets sparse observations exchange
        # identities once both new tracks have zero velocity again.
        for possible in uncertain_candidates.values():
            if len(possible) > 1:
                uncertain_observations.update(possible)
        disputed.update(
            proposals[local_id].subject_id
            for local_id in uncertain_observations
            if proposals[local_id] is not None
        )
        if disputed:
            self._face_tracker.forget_bodies(disputed)
            for subject_id in disputed:
                self._subjects.pop(subject_id, None)
        for local_id, box, _confidence in observations:
            subject = proposals[local_id]
            if local_id in uncertain_observations:
                subject = None
            if subject is not None and subject.subject_id in disputed:
                uncertain_observations.add(local_id)
                subject = None
            if subject is None:
                subject = self._new_subject(box, timestamp)
                # A crossing can reset both tracks while their boxes still
                # overlap. Their new IDs lack a reliable direction: reusing
                # those IDs by next-frame IoU would swap the two identities.
                # Keep this observation addressable in history, but require
                # a fresh association after the ambiguous observation.
                subject.association_uncertain = local_id in uncertain_observations
            else:
                gap = timestamp - subject.seen_at
                if gap > 0 and subject.kind == "body":
                    previous, current = _center(subject.bbox), _center(box)
                    subject.vx = (
                        0.35 * subject.vx + 0.65 * (current[0] - previous[0]) / gap
                    )
                    subject.vy = (
                        0.35 * subject.vy + 0.65 * (current[1] - previous[1]) / gap
                    )
                subject.bbox = box
                subject.seen_at = timestamp
                subject.kind = "body"
            assignments[local_id] = subject.subject_id
            if worker:
                self._body_aliases[local_id] = subject.subject_id
        return assignments

    def _record_snapshot(
        self,
        timestamp: float,
        sequence: int,
        objects: tuple[RichLabelObject, ...],
    ) -> None:
        snapshot = RichLabelSnapshot(
            self.stream_session_id or self._namespace, sequence, timestamp, objects
        )
        for observed in objects:
            state = observed.observation_state or observed.attributes.get(
                "observation_state", "observed"
            )
            if state == "observed":
                self._identity_review.observe(
                    observed.track_id, timestamp, sequence=sequence,
                    bbox=observed.bbox,
                    native=_native_metadata(observed.attributes.get("native_track"))
                    is not None,
                )
        # Same-frame sources may fill in missing observations, without duplicating IDs.
        existing = next(
            (
                item
                for item in self._history
                if (
                    item.timestamp_seconds == timestamp
                    and item.frame_sequence == sequence
                )
            ),
            None,
        )
        if existing is not None:
            merged = {item.track_id: item for item in existing.objects}
            merged.update({item.track_id: item for item in objects})
            self._history.remove(existing)
            snapshot = replace(snapshot, objects=tuple(merged.values()))
        records = sorted(
            (*self._history, snapshot),
            key=lambda item: (item.timestamp_seconds, item.frame_sequence),
        )
        cutoff = max(self._latest_time, timestamp) - 90.0
        self._history = deque(
            (item for item in records[-1024:] if item.timestamp_seconds >= cutoff),
            maxlen=1024,
        )

    def snapshot_at(
        self, timestamp: float, frame_sequence: int | None = None
    ) -> RichLabelSnapshot | None:
        with self._lock:
            return max(
                (
                    item
                    for item in self._history
                    if (
                        item.timestamp_seconds <= timestamp
                        and (
                            frame_sequence is None
                            or item.frame_sequence <= frame_sequence
                        )
                    )
                ),
                key=lambda item: (item.timestamp_seconds, item.frame_sequence),
                default=None,
            )

    def history(self) -> tuple[RichLabelSnapshot, ...]:
        with self._lock:
            return tuple(self._history)

    def _record_faces(
        self, timestamp: float, sequence: int, faces: tuple[ModelObject, ...]
    ) -> None:
        # Empty observations are meaningful: never borrow a disappeared person's
        # older face to identify an actor in a later, explicitly empty frame.
        snapshot = _FaceSnapshot(
            timestamp,
            sequence,
            tuple(
                _SourceFace(
                    str(item.attributes["subject_id"]),
                    item.bbox,
                    str(item.attributes["person_id"])
                    if item.attributes.get("status") == "known"
                    and item.attributes.get("person_id")
                    else None,
                )
                for item in faces
            ),
        )
        records = sorted(
            (
                *(
                    item
                    for item in self._face_history
                    if (item.timestamp, item.sequence) != (timestamp, sequence)
                ),
                snapshot,
            ),
            key=lambda item: (item.timestamp, item.sequence),
        )
        cutoff = max(self._latest_time, timestamp) - 90.0
        self._face_history = deque(
            (item for item in records[-1024:] if item.timestamp >= cutoff),
            maxlen=1024,
        )

    def _source_face_actors(
        self, actors: Sequence[ModelActor], timestamp: float, sequence: int
    ) -> dict[str, str]:
        snapshot = max(
            (
                item
                for item in self._face_history
                if item.timestamp <= timestamp and item.sequence <= sequence
            ),
            key=lambda item: (item.timestamp, item.sequence),
            default=None,
        )
        if (
            snapshot is None
            or timestamp - snapshot.timestamp > _SOURCE_FACE_MAX_GAP_SECONDS
        ):
            return {}
        proposals = {}
        candidate_counts: Counter[str] = Counter()
        for actor in actors:
            candidates = [
                face.subject_id
                for face in snapshot.faces
                if _face_body_score(face.bbox, actor.bbox) is not None
            ]
            candidate_counts.update(candidates)
            if len(candidates) == 1:
                proposals[actor.actor_id] = candidates[0]
        return {
            actor_id: subject_id
            for actor_id, subject_id in proposals.items()
            if candidate_counts[subject_id] == 1
        }

    def resolve_faces(
        self,
        objects: Sequence[ModelObject],
        *,
        timestamp: float,
        frame_sequence: int,
        people: Mapping[str, str],
        snapshot: RichLabelSnapshot | None = None,
    ) -> tuple[ModelObject, ...]:
        with self._lock:
            people = {
                key: name
                for key, name in people.items()
                if key not in self._removed_people
            }
            # Also handle direct runtime users without an enrollment callback,
            # even when this frame contains no faces at all.
            for person_id in {
                str(subject.identity["person_id"])
                for subject in self._subjects.values()
                if "person_id" in subject.identity
                and subject.identity["person_id"] not in people
            }:
                self._clear_person(person_id)
            if snapshot is not None:
                self.observe_snapshot(snapshot)
            canonical = self.snapshot_at(timestamp, frame_sequence)
            # A displayed tentative box is not trusted face/body identity evidence.
            # Reset held body links too, so a newly uncertain track cannot inherit
            # an earlier face association merely because its ID is unchanged.
            provisional_ids = (
                {
                    item.track_id
                    for item in canonical.objects
                    if item.attributes.get("tracking_confirmation", "confirmed")
                    != "confirmed"
                }
                if canonical is not None
                else set()
            )
            self._face_tracker.forget_bodies(provisional_ids)
            face_snapshot = (
                replace(
                    canonical,
                    objects=tuple(
                        item
                        for item in canonical.objects
                        if item.track_id not in provisional_ids
                    ),
                )
                if canonical is not None
                else None
            )
            resolved = self._face_tracker.resolve(
                objects,
                timestamp=timestamp,
                frame_sequence=frame_sequence,
                people=people,
                snapshot=face_snapshot,
                known_bodies={
                    key: {
                        "person_id": subject.identity["person_id"],
                        "known_at": subject.identity_at,
                        "embeddings": subject.identity_embeddings,
                    }
                    for key, subject in self._subjects.items()
                    if subject.identity.get("person_id") in people
                    and subject.identity_at <= timestamp
                    and key not in provisional_ids
                },
            )
            if canonical is not None and (
                canonical.timestamp_seconds,
                canonical.frame_sequence,
            ) == (timestamp, frame_sequence):
                source_faces = tuple(
                    _SourceFace(
                        self._face_aliases.get(
                            str(item.attributes.get("face_track_id")), ""
                        ),
                        item.bbox,
                        str(item.attributes["person_id"])
                        if item.attributes.get("status") == "known"
                        and item.attributes.get("person_id")
                        else None,
                    )
                    for item in resolved
                    if any(item.bbox == raw.bbox for raw in objects)
                )
                for native_id, old_id in self._exact_source_face_matches(
                    canonical, source_faces
                ).items():
                    self._merge_face_subject(old_id, native_id)
            result = []
            for item in resolved:
                attributes = dict(item.attributes)
                attributes["face_observation_state"] = (
                    "observed"
                    if any(item.bbox == raw.bbox for raw in objects)
                    else "held"
                )
                face_id = str(attributes.get("face_track_id", ""))
                # A delayed face must not inherit future identity state.
                if not face_id:
                    result.append(replace(item, attributes=attributes))
                    continue
                body_id = str(attributes.get("person_track_id", ""))
                subject = self._subjects.get(body_id) or self._subjects.get(
                    self._face_aliases.get(face_id, "")
                )
                if subject is None:
                    subject = self._new_subject(item.bbox, timestamp, "face")
                if attributes.pop("identity_reset", False):
                    if subject.native_track is not None:
                        # A face correction changes identity evidence, not the
                        # authoritative video object's public subject identifier.
                        subject.identity = {}
                        subject.identity_at = -1.0
                        subject.identity_embeddings = ()
                    else:
                        old_id = subject.subject_id
                        subject = self._new_subject(
                            subject.bbox, timestamp, subject.kind
                        )
                        self._subjects.pop(old_id, None)
                        self._replace_subject_aliases(
                            old_id, subject.subject_id, timestamp
                        )
                self._face_aliases[face_id] = subject.subject_id
                if (
                    subject.native_track is None
                    and item.attributes.get("identity_source") != "track_history"
                ):
                    subject.seen_at = max(subject.seen_at, timestamp)
                if subject.kind == "face":
                    subject.bbox = item.bbox
                    # A real weak face is still a spatial observation.
                    if item.attributes.get("observation_status") is not None:
                        subject.seen_at = max(subject.seen_at, timestamp)
                if (
                    attributes.get("status") == "known"
                    and timestamp >= subject.identity_at
                ):
                    subject.identity = {
                        key: attributes[key]
                        for key in ("status", "person_id", "name", "identity_source")
                        if key in attributes
                    }
                    if attributes.get("identity_source") != "track_history":
                        subject.identity_at = timestamp
                    if (
                        item.identity_quality
                        and item.identity_embedding
                        and attributes.get("identity_source") == "face_match"
                    ):
                        subject.identity_embeddings = (
                            *subject.identity_embeddings[-4:],
                            item.identity_embedding,
                        )
                elif subject.identity.get("person_id") not in people:
                    subject.identity = {}
                attributes.update(
                    subject_id=subject.subject_id,
                    subject_ids=[subject.subject_id],
                    person_track_id=subject.subject_id,
                )
                result.append(replace(item, attributes=attributes))
            self._expire(timestamp)
            self._record_faces(
                timestamp,
                frame_sequence,
                tuple(
                    item
                    for item in result
                    if item.attributes.get("subject_id")
                    and item.attributes.get("face_observation_state") == "observed"
                ),
            )
            source_body = self.snapshot_at(timestamp, frame_sequence)
            if source_body is not None and (
                source_body.timestamp_seconds,
                source_body.frame_sequence,
            ) == (timestamp, frame_sequence):
                self._associate_source_faces(source_body)
                result = [self._canonical_face(item) for item in result]
            for item in result:
                if item.attributes.get("face_observation_state") == "observed":
                    self._identity_review.observe_face(item, timestamp, frame_sequence)
            return tuple(self._canonical_face(item) for item in result)

    def _canonical_face(self, item: ModelObject) -> ModelObject:
        attributes = dict(item.attributes)
        for key in ("subject_id", "person_track_id"):
            value = attributes.get(key)
            if isinstance(value, str):
                attributes[key] = self._identity_review.canonical(
                    self._subject_aliases.get(value, value)
                )
        if isinstance(attributes.get("subject_ids"), (list, tuple)):
            attributes["subject_ids"] = list(
                dict.fromkeys(
                    self._identity_review.canonical(
                        self._subject_aliases.get(value, value)
                    )
                    for value in attributes["subject_ids"]
                )
            )
        return replace(item, attributes=attributes)

    def prepare_analysis(
        self, result: AnalysisResult, bundle: ObservationBundle | None = None
    ) -> AnalysisResult:
        if (
            self.stream_session_id
            and result.stream_session_id != self.stream_session_id
        ) or (self.source_id and result.source_id != self.source_id):
            return result
        bundle = bundle or ObservationBundle()
        timestamp = result.frame.timestamp_seconds
        sequence = result.frame.source_frame_index
        with self._lock:
            if not self.stream_session_id:
                self.stream_session_id = result.stream_session_id
            if not self.source_id:
                self.source_id = result.source_id
            actors = _unique_actors(bundle.actors)
            face_actor_map = self._source_face_actors(actors, timestamp, sequence)
            source_snapshot = self.snapshot_at(timestamp, sequence)
            native_actor_map = {}
            if source_snapshot is not None and (
                source_snapshot.timestamp_seconds,
                source_snapshot.frame_sequence,
            ) == (timestamp, sequence):
                for actor in actors:
                    face_id = face_actor_map.get(actor.actor_id)
                    old_body = _unique_body(actor.bbox, source_snapshot.objects)
                    if old_body is not None and "native_track" in old_body.attributes:
                        native_actor_map[actor.actor_id] = old_body.track_id
                        continue
                    if (
                        face_id is not None
                        and old_body is not None
                        and old_body.track_id != face_id
                    ):
                        # Keep the next face from jumping back to the provisional
                        # body ID after this event receives its source-face owner.
                        self._subjects.pop(old_body.track_id, None)
                        self._replace_subject_aliases(
                            old_body.track_id,
                            face_id,
                            timestamp,
                            frame_sequence=sequence,
                        )
            actor_map: dict[str, str] = dict(face_actor_map)
            actor_map.update(native_actor_map)
            latest_body_time = max(
                (
                    item.seen_at
                    for item in self._subjects.values()
                    if item.kind == "body"
                ),
                default=-1.0,
            )
            if (
                timestamp,
                sequence,
            ) > self._actor_watermark and timestamp >= latest_body_time:
                self._actor_watermark = (timestamp, sequence)
                actor_map.update(
                    self._observe_bodies(
                        [
                            (item.actor_id, item.bbox, item.confidence)
                            for item in actors
                            if item.actor_id not in actor_map
                        ],
                        timestamp,
                        excluded_subject_ids=set(face_actor_map.values())
                        | {
                            item.subject_id
                            for item in self._subjects.values()
                            if item.kind == "face"
                        },
                    )
                )
                body_objects = tuple(
                    RichLabelObject(
                        actor_map[item.actor_id],
                        "person",
                        item.bbox,
                        item.confidence,
                        {
                            "observation_state": "observed",
                            "last_observed_timestamp_seconds": timestamp,
                        },
                    )
                    for item in actors
                    if item.actor_id not in native_actor_map
                )
                if actors:
                    self._record_snapshot(timestamp, sequence, body_objects)
            else:
                snapshot = self.snapshot_at(timestamp, sequence)
                if (
                    snapshot is not None
                    and timestamp - snapshot.timestamp_seconds <= self.max_gap_seconds
                ):
                    for actor in actors:
                        if actor.actor_id in actor_map:
                            continue
                        match = _unique_body(actor.bbox, snapshot.objects)
                        if match is not None:
                            actor_map[actor.actor_id] = match.track_id
            faces = {
                str(item.attributes.get("face_track_id")): item
                for item in bundle.face_observations
                if item.attributes.get("face_track_id")
            }
            actor_boxes = {item.actor_id: item.bbox for item in actors}
            for actor in actors:
                subject = self._subjects.get(actor_map.get(actor.actor_id, ""))
                if (
                    subject is None
                    or subject.native_track is not None
                    or subject.seen_at > timestamp
                    or subject.identity_at > timestamp
                    or subject.memory_frame >= (timestamp, sequence)
                    or subject.identity.get("person_id") is not None
                ):
                    continue
                subject.memory_frame = (timestamp, sequence)
                if actor.matched_memory_id is None or actor.confidence < 0.8:
                    subject.memory_candidate = None
                    subject.memory_confirmations = 0
                    continue
                previous = subject.identity.get("matched_memory_id")
                if previous is not None and previous != actor.matched_memory_id:
                    if subject.memory_candidate != actor.matched_memory_id:
                        subject.memory_candidate = actor.matched_memory_id
                        subject.memory_confirmations = 0
                    subject.memory_confirmations += 1
                    if subject.memory_confirmations < 3:
                        continue
                    old_id = subject.subject_id
                    subject = self._new_subject(subject.bbox, timestamp)
                    self._subjects.pop(old_id, None)
                    actor_map[actor.actor_id] = subject.subject_id
                    subject.memory_frame = (timestamp, sequence)
                    self._replace_subject_aliases(old_id, subject.subject_id, timestamp)
                else:
                    subject.memory_candidate = None
                    subject.memory_confirmations = 0
                subject.identity = {
                    "status": "known",
                    "matched_memory_id": actor.matched_memory_id,
                    "identity_source": "face_reference_match",
                }
                subject.identity_at = timestamp
            snapshot = self.snapshot_at(timestamp, sequence)
            historical_bodies = (
                snapshot.objects
                if snapshot is not None
                and timestamp - snapshot.timestamp_seconds <= self.max_gap_seconds
                else ()
            )
            records = []
            for index, item in enumerate(result.objects):
                attributes = dict(item.attributes)
                if attributes.get("person_id") in self._removed_people:
                    for key in ("person_id", "name", "identity_source"):
                        attributes.pop(key, None)
                    attributes.update(status="unusable", reason="identity_removed")
                    item = replace(item, label="unusable_face")
                identity = attributes.get("subject_identity")
                if (
                    isinstance(identity, Mapping)
                    and identity.get("person_id") in self._removed_people
                ):
                    attributes.pop("subject_identity", None)
                ids = []
                for actor_id in (
                    bundle.object_actor_ids[index]
                    if index < len(bundle.object_actor_ids)
                    else ()
                ):
                    if actor_id in actor_map and _plausible_owner(
                        item.bbox, actor_boxes[actor_id]
                    ):
                        ids.append(actor_map[actor_id])
                face = faces.get(str(attributes.get("face_track_id", "")))
                if ids and any(
                    actor_id in face_actor_map
                    for actor_id in (
                        bundle.object_actor_ids[index]
                        if index < len(bundle.object_actor_ids)
                        else ()
                    )
                ):
                    # A provisional body ID may already be public/archived. A
                    # source-face correction replaces it, never adds a second
                    # owner to the same event.
                    for key in (
                        "subject_id",
                        "subject_ids",
                        "person_track_id",
                        "subject_identity",
                    ):
                        attributes.pop(key, None)
                for source in (attributes, face.attributes if face is not None else {}):
                    subject_id = source.get("subject_id")
                    if isinstance(subject_id, str) and subject_id.startswith(
                        f"subject-{self._namespace}-"
                    ):
                        ids.append(subject_id)
                if not ids and item.label.casefold() in _PERSON_LABELS:
                    match = _unique_body(item.bbox, historical_bodies, face=True)
                    if match is not None:
                        ids.append(match.track_id)
                ids = list(dict.fromkeys(ids))
                if ids:
                    attributes["subject_ids"] = ids
                    if len(ids) == 1:
                        attributes.update(subject_id=ids[0], person_track_id=ids[0])
                        subject = self._subjects.get(ids[0])
                        if (
                            subject is not None
                            and subject.seen_at <= timestamp
                            and subject.identity
                        ):
                            attributes["subject_identity"] = dict(subject.identity)
                records.append(replace(item, attributes=attributes))
            for judgment in result.judgments:
                if judgment.event_label != "fight" or not judgment.detected:
                    continue
                remaining = {
                    index
                    for index, item in enumerate(records)
                    if item.object_id in judgment.object_ids
                }
                while remaining:
                    component = {remaining.pop()}
                    frontier = list(component)
                    while frontier:
                        current_index = frontier.pop()
                        neighbors = {
                            index
                            for index in remaining
                            if _spatial_neighbors(
                                records[current_index].bbox, records[index].bbox
                            )
                        }
                        component.update(neighbors)
                        remaining.difference_update(neighbors)
                        frontier.extend(neighbors)
                    participants = sorted(
                        {
                            subject_id
                            for index in component
                            for subject_id in records[index].attributes.get(
                                "subject_ids", ()
                            )
                        }
                    )
                    if len(participants) > 1:
                        for index in component:
                            attributes = dict(records[index].attributes)
                            attributes["subject_ids"] = participants
                            attributes.pop("subject_identity", None)
                            records[index] = replace(
                                records[index], attributes=attributes
                            )
            prepared = replace(result, objects=tuple(records))
            if not self._reassociating:
                if _has_unassociated_positive(prepared) or _needs_face_association(
                    prepared, bundle, face_actor_map
                ):
                    self._pending[result.message_id] = (prepared, bundle)
                    self._pending.move_to_end(result.message_id)
                else:
                    self._pending.pop(result.message_id, None)
                while len(self._pending) > 128:
                    self._pending.popitem(last=False)
            return prepared

    def drain_reassociated_results(self) -> tuple[AnalysisResult, ...]:
        """Bounded background corrections; callers must not republish live results."""
        with self._lock:
            updates = []
            self._reassociating = True
            try:
                for message_id, (previous, bundle) in list(self._pending.items())[:32]:
                    if self._latest_time - previous.frame.timestamp_seconds > 90.0:
                        self._pending.pop(message_id, None)
                        continue
                    current = self.prepare_analysis(previous, bundle)
                    if any(
                        left.attributes.get("subject_ids")
                        != right.attributes.get("subject_ids")
                        for left, right in zip(
                            previous.objects, current.objects, strict=True
                        )
                    ):
                        updates.append(current)
                    face_actor_map = self._source_face_actors(
                        _unique_actors(bundle.actors),
                        current.frame.timestamp_seconds,
                        current.frame.source_frame_index,
                    )
                    if _has_unassociated_positive(current) or _needs_face_association(
                        current, bundle, face_actor_map
                    ):
                        self._pending[message_id] = (current, bundle)
                        self._pending.move_to_end(message_id)
                    else:
                        self._pending.pop(message_id, None)
            finally:
                self._reassociating = False
            return tuple(updates)


def _has_unassociated_positive(result: AnalysisResult) -> bool:
    positive_ids = {
        object_id
        for judgment in result.judgments
        if judgment.detected
        for object_id in judgment.object_ids
    }
    return any(
        item.object_id in positive_ids and not item.attributes.get("subject_ids")
        for item in result.objects
    )


def _needs_face_association(
    result: AnalysisResult,
    bundle: ObservationBundle,
    face_actor_map: Mapping[str, str],
) -> bool:
    positive_ids = {
        object_id
        for judgment in result.judgments
        if judgment.detected
        for object_id in judgment.object_ids
    }
    return any(
        item.object_id in positive_ids
        and index < len(bundle.object_actor_ids)
        and any(
            actor_id not in face_actor_map
            for actor_id in bundle.object_actor_ids[index]
        )
        for index, item in enumerate(result.objects)
    )


def _unique_actors(actors: Sequence[ModelActor]) -> tuple[ModelActor, ...]:
    result = []
    for actor in sorted(actors, key=lambda item: item.confidence, reverse=True):
        if actor.confidence < 0.6 or any(
            _iou(actor.bbox, previous.bbox) >= 0.7 for previous in result
        ):
            continue
        result.append(actor)
        if len(result) == 8:
            break
    return tuple(result)


def _center(box: NormalizedBBox) -> tuple[float, float]:
    return (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2


def _area(box: NormalizedBBox) -> float:
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def _iou(left: NormalizedBBox, right: NormalizedBBox) -> float:
    intersection = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
        0.0, min(left.y2, right.y2) - max(left.y1, right.y1)
    )
    return intersection / (_area(left) + _area(right) - intersection)


def _body_score(
    subject: _Subject, box: NormalizedBBox, timestamp: float
) -> float | None:
    ratio = _area(box) / _area(subject.bbox)
    if not 0.4 <= ratio <= 2.5:
        return None
    gap = timestamp - subject.seen_at
    previous, current = _center(subject.bbox), _center(box)
    distance = math.hypot(
        current[0] - previous[0] - subject.vx * gap,
        current[1] - previous[1] - subject.vy * gap,
    )
    scale = max(
        math.hypot(box.x2 - box.x1, box.y2 - box.y1),
        math.hypot(
            subject.bbox.x2 - subject.bbox.x1, subject.bbox.y2 - subject.bbox.y1
        ),
    )
    relative = distance / scale
    overlap = _iou(subject.bbox, box)
    if overlap < 0.2 and relative > 0.5:
        return None
    return 0.7 * overlap + 0.3 * max(0.0, 1.0 - relative)


def _unique_body(
    box: NormalizedBBox, bodies: Sequence[RichLabelObject], *, face: bool = False
) -> RichLabelObject | None:
    scores = []
    for body in bodies:
        if body.attributes.get("tracking_confirmation", "confirmed") != "confirmed":
            continue
        score = _iou(box, body.bbox)
        if face:
            intersection = max(
                0.0, min(box.x2, body.bbox.x2) - max(box.x1, body.bbox.x1)
            ) * max(0.0, min(box.y2, body.bbox.y2) - max(box.y1, body.bbox.y1))
            relative_y = (_center(box)[1] - body.bbox.y1) / (
                body.bbox.y2 - body.bbox.y1
            )
            if intersection / _area(box) >= 0.9 and 0 <= relative_y <= 0.6:
                score = max(score, 0.9)
        if score >= 0.2:
            scores.append((score, body))
    scores.sort(key=lambda item: item[0], reverse=True)
    return (
        scores[0][1]
        if scores and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.15)
        else None
    )


def _face_body_score(face: NormalizedBBox, body: NormalizedBBox) -> float | None:
    intersection = max(0.0, min(face.x2, body.x2) - max(face.x1, body.x1)) * max(
        0.0, min(face.y2, body.y2) - max(face.y1, body.y1)
    )
    relative_y = (_center(face)[1] - body.y1) / (body.y2 - body.y1)
    return 0.9 if intersection / _area(face) >= 0.9 and 0 <= relative_y <= 0.6 else None


def _plausible_owner(target: NormalizedBBox, body: NormalizedBBox) -> bool:
    x, y = _center(target)
    dx, dy = (body.x2 - body.x1) * 0.15, (body.y2 - body.y1) * 0.15
    return body.x1 - dx <= x <= body.x2 + dx and body.y1 - dy <= y <= body.y2 + dy


def _spatial_neighbors(left: NormalizedBBox, right: NormalizedBBox) -> bool:
    """Keep physically separate fights separate even within one frame judgment."""
    dx = max(0.0, left.x1 - right.x2, right.x1 - left.x2)
    dy = max(0.0, left.y1 - right.y2, right.y1 - left.y2)
    return (
        dx <= min(left.x2 - left.x1, right.x2 - right.x1) * 0.5
        and dy <= min(left.y2 - left.y1, right.y2 - right.y1) * 0.25
    )
