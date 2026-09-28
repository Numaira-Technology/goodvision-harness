"""Bounded, session-local people and event episodes built from source observations.

The registry owns person association. This module never merges people by geometry,
and a tracker prediction is never evidence that a person or event was observed.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, OrderedDict, deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import RLock
from uuid import uuid4

from .contracts import NormalizedBBox, json_mapping_as_dict
from .rich_labeling import validate_mask_contours

_EVENT_LABELS = frozenset({"fight", "smoking", "fall", "weapon", "stranger"})
_IDENTITY_KEYS = ("status", "person_id", "name", "identity_source", "matched_memory_id")
_NATIVE_TRACK_KEYS = ("backend", "epoch", "object_id")
_MAX_CONTOUR_HISTORY_POINTS = 131072


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be nonempty text of at most 256 characters")
    return value.strip()


def _integer(value: int, name: str, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def _time(value: float, name: str = "timestamp_seconds") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass
class _Person:
    subject_id: str
    bbox: NormalizedBBox
    first_seen: float
    last_seen: float
    frame_sequence: int
    history: deque
    identity: dict = field(default_factory=dict)
    identity_source: tuple[float, int] = (-1.0, -1)
    native_track: dict = field(default_factory=dict)
    tracking_confirmation: str | None = None


@dataclass
class _Event:
    event_label: str
    subject_ids: tuple[str, ...]
    history: deque
    event_id: str | None = None
    phase: str = "ended"
    first_seen: float = -1.0
    last_seen: float = -1.0
    frame_sequence: int = -1
    positives: int = 0
    negatives: int = 0
    uncertain: bool = False
    confidence: float | None = None
    last_positive_timestamp: float | None = None
    confirmed_since_timestamp: float | None = None
    ended_since_timestamp: float | None = None
    missing_observations: int = 0
    ended_reason: str | None = None


@dataclass
class _Evidence:
    event_label: str
    evidence_id: str
    subject_ids: tuple[str, ...]
    timestamp: float
    sequence: int
    detected: bool | None
    confidence: float | None
    revision: int = 0
    episode_id: str | None = None


@dataclass
class _Ledger:
    seed: _Event
    records: dict[tuple[str, str], _Evidence] = field(default_factory=dict)
    floor: tuple[float, int] = (-1.0, -1)
    watermark: tuple[float, int] = (-1.0, -1)


class SceneState:
    """Store real person observations and source-ordered event evidence.

    Methods return False for stale, duplicate, ungrounded or capacity-rejected
    observations; malformed arguments raise ValueError. Snapshot copies are safe
    for callers to mutate. All retained collections have explicit size limits.
    """

    def __init__(
        self,
        stream_session_id: str,
        source_id: str,
        *,
        max_entities: int = 128,
        history_size: int = 64,
        occlusion_seconds: float = 3.0,
        retention_seconds: float = 90.0,
        confirm_observations: int = 2,
        clear_observations: int = 2,
        absence_observations: int = 10,
        person_history_size: int | None = None,
        person_history_seconds: float | None = None,
        max_contour_history_points: int = _MAX_CONTOUR_HISTORY_POINTS,
        max_source_pin_jobs: int = 4,
        retain_pinned_sources: bool = False,
    ) -> None:
        self.stream_session_id = _text(stream_session_id, "stream_session_id")
        self.source_id = _text(source_id, "source_id")
        self.max_entities = _integer(max_entities, "max_entities", 1, 4096)
        self.max_source_pin_jobs = _integer(
            max_source_pin_jobs, "max_source_pin_jobs", 1, 64,
        )
        if type(retain_pinned_sources) is not bool:
            raise ValueError("retain_pinned_sources must be a boolean")
        self.retain_pinned_sources = retain_pinned_sources
        self.max_contour_history_points = _integer(
            max_contour_history_points, "max_contour_history_points",
            1, _MAX_CONTOUR_HISTORY_POINTS,
        )
        self.history_size = _integer(history_size, "history_size", 1, 1024)
        self.person_history_size = _integer(
            history_size if person_history_size is None else person_history_size,
            "person_history_size",
            1,
            1024,
        )
        self.person_history_seconds = (
            None
            if person_history_seconds is None
            else _time(person_history_seconds, "person_history_seconds")
        )
        if self.person_history_seconds == 0:
            raise ValueError("person_history_seconds must be positive")
        self.occlusion_seconds = _time(occlusion_seconds, "occlusion_seconds")
        self.retention_seconds = _time(retention_seconds, "retention_seconds")
        if (
            self.occlusion_seconds <= 0
            or self.retention_seconds <= self.occlusion_seconds
        ):
            raise ValueError("retention_seconds must exceed positive occlusion_seconds")
        self.confirm_observations = _integer(
            confirm_observations, "confirm_observations", 1
        )
        self.clear_observations = _integer(clear_observations, "clear_observations", 1)
        self.absence_observations = _integer(
            absence_observations, "absence_observations", 1,
        )
        self.max_events = min(self.max_entities * len(_EVENT_LABELS), 2048)
        self._people: dict[str, _Person] = {}
        # Masks are presentation-only: keep at most 128 maximum-size outlines,
        # or thousands of usual small outlines, independently of bbox history.
        self._contour_points = 0
        self._contour_serial = 0
        self._contour_records: dict[int, tuple[dict, int, int]] = {}
        self._contour_order: list[tuple[float, int, int, int]] = []
        self._events: dict[tuple[str, tuple[str, ...]], _Event] = {}
        self._evidence: OrderedDict[tuple[str, str], tuple[str, ...]] = OrderedDict()
        self._ledgers: dict[tuple[str, tuple[str, ...]], _Ledger] = {}
        self._record_groups: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
        self._source_pins: dict[
            str, tuple[tuple[str, float, int, NormalizedBBox], ...]
        ] = {}
        # Completed tracking facts share the event replay timeline. Pins retain
        # the prefix needed by an already admitted, late model judgment.
        self._tracking_presence: OrderedDict[int, tuple[float, frozenset[str]]] = (
            OrderedDict()
        )
        self._presence_watermark = (-1.0, -1)
        self._presence_floor = (-1.0, -1)
        self._transitions: deque = deque(maxlen=self.history_size)
        self._counters: Counter = Counter()
        self._timestamp = 0.0
        self._namespace = uuid4().hex[:12]
        self._episode = 0
        self._transition_sequence = 0
        self._review_complete = False
        self._lock = RLock()

    def _reject(self, reason: str) -> bool:
        self._counters[reason] += 1
        return False

    @staticmethod
    def _same_gallery_identity(left: _Person, right: _Person) -> bool:
        return bool(
            left.identity.get("status") == right.identity.get("status") == "known"
            and left.identity.get("person_id")
            and left.identity.get("person_id") == right.identity.get("person_id")
            and left.identity.get("identity_source") in {"face_match", "track_history"}
            and right.identity.get("identity_source") in {"face_match", "track_history"}
        )

    @staticmethod
    def _gallery_merge_conflict(left: _Person, right: _Person) -> bool:
        """Same gallery ID is evidence; simultaneous distinct observations veto it.

        Face and body observations of one person naturally overlap in time. A
        face contained by its body must not be confused with two people simply
        because their first/last observation envelopes overlap.
        """
        sources = {
            (item["timestamp_seconds"], item["frame_sequence"]): item
            for item in left.history
        }
        for item in right.history:
            other = sources.get((item["timestamp_seconds"], item["frame_sequence"]))
            if other is None:
                continue
            a, b = other.get("native_track"), item.get("native_track")
            if bool(a) == bool(b):
                if a and a != b:
                    return True
                if a:
                    continue
                left_box, right_box = other["bbox"], item["bbox"]
                intersection = max(
                    0,
                    min(left_box["x2"], right_box["x2"])
                    - max(left_box["x1"], right_box["x1"]),
                ) * max(
                    0,
                    min(left_box["y2"], right_box["y2"])
                    - max(left_box["y1"], right_box["y1"]),
                )
                union = (
                    sum(
                        (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                        for box in (left_box, right_box)
                    )
                    - intersection
                )
                if intersection < 0.8 * union:
                    return True
                continue
            body, face = (other, item) if a else (item, other)
            body, face = body["bbox"], face["bbox"]
            intersection = max(
                0, min(body["x2"], face["x2"]) - max(body["x1"], face["x1"])
            ) * max(0, min(body["y2"], face["y2"]) - max(body["y1"], face["y1"]))
            area = (face["x2"] - face["x1"]) * (face["y2"] - face["y1"])
            if intersection < 0.9 * area:
                return True
        return False

    def merge_subject(
        self, previous_id: str, current_id: str, *, reviewed_identity: bool = False
    ) -> bool:
        """Apply a registry-proven association; never infer one from geometry.

        Source times and evidence IDs do not change. Replaying combined ledgers
        counts each source frame once, including when both branches saw it.
        """
        previous_id = _text(previous_id, "previous_id")
        current_id = _text(current_id, "current_id")
        if type(reviewed_identity) is not bool:
            raise ValueError("reviewed_identity must be a boolean")
        with self._lock:
            if previous_id == current_id:
                return False
            previous = self._people.get(previous_id)
            current = self._people.get(current_id)
            if previous is None:
                return False
            if (
                not reviewed_identity
                and previous.native_track
                and (current is None or previous.native_track != current.native_track)
            ):
                return self._reject("native_subject_merge_rejected")
            if reviewed_identity and current is not None:
                same_gallery = self._same_gallery_identity(previous, current)
                overlapping = max(previous.first_seen, current.first_seen) <= min(
                    previous.last_seen, current.last_seen
                )
                if (
                    (overlapping and not same_gallery)
                    or (
                        same_gallery and self._gallery_merge_conflict(previous, current)
                    )
                    or (
                        previous.identity.get("person_id")
                        and current.identity.get("person_id")
                        and previous.identity["person_id"]
                        != current.identity["person_id"]
                    )
                ):
                    return self._reject("conflicting_identity_merge_rejected")
            if current is None:
                previous.subject_id = current_id
                self._people[current_id] = previous
            else:
                # A newly reviewed face fragment carries identity, not a new
                # full-body location. Keep the native person's geometry intact.
                native_person = (
                    current
                    if current.native_track
                    else previous
                    if previous.native_track
                    else None
                )
                geometry = (
                    (
                        native_person.bbox,
                        native_person.last_seen,
                        native_person.frame_sequence,
                        native_person.tracking_confirmation,
                        native_person.native_track,
                    )
                    if native_person is not None
                    and bool(previous.native_track) != bool(current.native_track)
                    else None
                )
                old_records = [*previous.history, *current.history]
                records = {}
                for person in (previous, current):
                    for record in person.history:
                        if geometry is not None and not record.get("native_track"):
                            record["identity_only"] = True
                        key = (record["timestamp_seconds"], record["frame_sequence"])
                        # Native full-body geometry wins over a same-frame face.
                        if key not in records or record.get("native_track"):
                            records[key] = record
                current.history = deque(
                    (records[key] for key in sorted(records)),
                    maxlen=self.person_history_size,
                )
                retained = {id(record) for record in current.history}
                for record in old_records:
                    if id(record) not in retained:
                        self._remove_contour_record(record)
                current.first_seen = min(current.first_seen, previous.first_seen)
                if (previous.last_seen, previous.frame_sequence) > (
                    current.last_seen,
                    current.frame_sequence,
                ):
                    current.bbox = previous.bbox
                    current.last_seen = previous.last_seen
                    current.frame_sequence = previous.frame_sequence
                    current.tracking_confirmation = previous.tracking_confirmation
                    current.native_track = previous.native_track
                if geometry is not None:
                    (
                        current.bbox,
                        current.last_seen,
                        current.frame_sequence,
                        current.tracking_confirmation,
                        current.native_track,
                    ) = geometry
                self._update_identity(
                    current, previous.identity, *previous.identity_source
                )
                self._prune_person_history(current)
            del self._people[previous_id]
            self._prune_contour_history()

            def subjects(values):
                return tuple(
                    sorted(
                        {current_id if sid == previous_id else sid for sid in values}
                    )
                )

            self._source_pins = {
                key: tuple(
                    (current_id if sid == previous_id else sid, at, seq, box)
                    for sid, at, seq, box in values
                )
                for key, values in self._source_pins.items()
            }
            self._tracking_presence = OrderedDict(
                (sequence, (at, frozenset(subjects(present))))
                for sequence, (at, present) in self._tracking_presence.items()
            )
            self._evidence = OrderedDict(
                (key, subjects(values)) for key, values in self._evidence.items()
            )
            for transition in self._transitions:
                transition["subject_ids"] = list(subjects(transition["subject_ids"]))
            for old_key in tuple(self._ledgers):
                if previous_id not in old_key[1]:
                    continue
                old = self._ledgers.pop(old_key)
                prior = self._events.pop(old_key, None)
                ids = subjects(old_key[1])
                key = (old_key[0], ids)
                collapsed_fight = old_key[0] == "fight" and len(ids) < 2
                old.seed = replace(
                    old.seed, subject_ids=ids,
                    missing_observations=(
                        0 if current is not None else old.seed.missing_observations
                    ),
                )
                if collapsed_fight:
                    old.seed = replace(
                        old.seed, phase="retracted", positives=0, uncertain=True
                    )
                for record in old.records.values():
                    record.subject_ids = subjects(record.subject_ids)
                    if collapsed_fight:
                        record.detected = None
                        record.confidence = None
                existing = self._ledgers.get(key)
                if existing is None:
                    self._ledgers[key] = old
                else:
                    # A compacted prefix cannot be expanded again. Use the seed
                    # at the later source floor; never add the two vote counts.
                    seed_ledger = max((old, existing), key=lambda item: item.floor)
                    floor = seed_ledger.floor
                    records = {**existing.records, **old.records}
                    self._ledgers[key] = _Ledger(
                        replace(
                            seed_ledger.seed, subject_ids=ids,
                            missing_observations=0,
                        ),
                        {
                            evidence_key: record
                            for evidence_key, record in records.items()
                            if record.sequence > floor[1]
                            and record.timestamp >= floor[0]
                        },
                        floor,
                        max(old.watermark, existing.watermark),
                    )
                for evidence_key in (
                    *old.records,
                    *(existing.records if existing else ()),
                ):
                    self._record_groups.pop(evidence_key, None)
                for evidence_key in self._ledgers[key].records:
                    self._record_groups[evidence_key] = key
                event, _ = self._replay(key)
                if collapsed_fight and prior is not None and prior.event_id:
                    event = replace(
                        event,
                        event_id=prior.event_id,
                        phase="retracted",
                        uncertain=True,
                    )
                    self._transition(event, "retract")
                self._events[key] = event
                self._prune_ledger(key)
            self._counters["subjects_merged"] += 1
            return True

    def advance(self, timestamp_seconds: float) -> None:
        """Advance source time; occlusion never generates negative evidence."""
        timestamp = _time(timestamp_seconds)
        with self._lock:
            self._timestamp = max(self._timestamp, timestamp)
            pinned_people = {
                subject for observations in self._source_pins.values()
                for subject, _at, _sequence, _bbox in observations
            }
            expired = [
                key
                for key, person in self._people.items()
                if self._timestamp - person.last_seen > self.retention_seconds
                and (not self.retain_pinned_sources or key not in pinned_people)
            ]
            for key in expired:
                for record in self._people[key].history:
                    self._remove_contour_record(record)
                del self._people[key]
                self._counters["entities_expired"] += 1
            if self.person_history_seconds is not None:
                for person in self._people.values():
                    self._prune_person_history(person)
            self._prune_contour_history()
            if expired:
                self._source_pins = {
                    key: tuple(item for item in observations if item[0] in self._people)
                    for key, observations in self._source_pins.items()
                }
            for key, event in list(self._events.items()):
                if (
                    event.phase in {"ended", "retracted"}
                    and self._timestamp - event.last_seen > self.retention_seconds
                ) or any(subject not in self._people for subject in event.subject_ids):
                    self._discard_group(key)
                    self._counters["events_expired"] += 1
            self._prune_tracking_presence()

    def observe_tracking_presence(
        self, subject_ids: tuple[str, ...], *, timestamp_seconds: float,
        frame_sequence: int,
    ) -> bool:
        """Accept one complete SAM result, including a successful empty frame.

        Capture ticks, display interpolation and failed requests never call this.
        Presence is not a VLM negative; both facts use the same event lifecycle.
        """
        at = _time(timestamp_seconds)
        sequence = _integer(frame_sequence, "frame_sequence", 0)
        if (not isinstance(subject_ids, (tuple, list))
                or len(subject_ids) > self.max_entities):
            raise ValueError("tracking presence must contain bounded subject IDs")
        present = frozenset(_text(sid, "subject_id") for sid in subject_ids)
        with self._lock:
            if (sequence <= self._presence_watermark[1]
                    or at < self._presence_watermark[0]):
                return self._reject("stale_tracking_presence_ignored")
            if not present <= self._people.keys():
                return self._reject("unknown_tracking_presence_ignored")
            self._timestamp = max(self._timestamp, at)
            self._tracking_presence[sequence] = (at, present)
            self._presence_watermark = (at, sequence)
            for event in self._events.values():
                prior = event.phase
                self._fold_presence(event, at, sequence, present)
                if prior != "ended" and event.phase == "ended":
                    self._transition(event, "end")
            self._prune_tracking_presence()
            return True

    def _fold_presence(self, event, at, sequence, present) -> None:
        if (event.event_label == "stranger" or event.event_id is None
                or event.phase in {"ended", "retracted"}
                or sequence < event.frame_sequence):
            return
        if present.intersection(event.subject_ids):
            event.missing_observations = 0
        else:
            event.missing_observations += 1
            if event.missing_observations >= self.absence_observations:
                event.phase = "ended"
                event.ended_reason = "all_subjects_absent"
                event.ended_since_timestamp = at
                event.last_seen = at
                event.frame_sequence = sequence

    def _prune_tracking_presence(self) -> None:
        # The size target may be exceeded while a bounded in-flight inspection
        # pins an older source. Its existing timeout releases that dependency.
        pinned = min(
            (seq for values in self._source_pins.values()
             for _sid, _at, seq, _box in values), default=math.inf,
        )
        prefix = []
        for sequence, (at, _present) in self._tracking_presence.items():
            if sequence >= pinned or not (
                len(self._tracking_presence) - len(prefix) > self.person_history_size
                or at < self._timestamp - self.retention_seconds
            ):
                break
            prefix.append((at, sequence))
        if not prefix:
            return
        floor = prefix[-1]
        for key, ledger in self._ledgers.items():
            if key[0] == "stranger" or ledger.floor[1] >= floor[1]:
                continue
            seed, _ = self._replay(key, through_sequence=floor[1])
            ledger.seed = replace(seed, history=deque(maxlen=0))
            ledger.floor = floor
            for evidence_key, record in tuple(ledger.records.items()):
                if record.sequence <= floor[1]:
                    del ledger.records[evidence_key]
                    self._record_groups.pop(evidence_key, None)
        for _at, sequence in prefix:
            del self._tracking_presence[sequence]
        self._presence_floor = floor

    def _discard_group(self, key: tuple[str, tuple[str, ...]]) -> None:
        ledger = self._ledgers.pop(key, None)
        if ledger is not None:
            for evidence_key in ledger.records:
                self._record_groups.pop(evidence_key, None)
        self._events.pop(key, None)

    def reserve_person_capacity(self, subject_ids: tuple[str, ...]) -> bool:
        """Retire expired unpinned batch tracks only when new people need room.

        Active event episodes retain their usual lifetime. Immediate tracking
        keeps its existing retention policy; no eviction occurs without a
        separately configured source-trajectory horizon.
        """
        subjects = {_text(subject, "subject_id") for subject in subject_ids}
        with self._lock:
            needed = len(set(self._people) | subjects) - self.max_entities
            if needed <= 0:
                return True
            if self.person_history_seconds is None:
                return False
            protected = {
                sid
                for observations in self._source_pins.values()
                for sid, _at, _sequence, _bbox in observations
            } | {
                sid
                for event in self._events.values()
                if event.phase in {"suspected", "confirmed"}
                for sid in event.subject_ids
            }
            eligible = sorted(
                (
                    person
                    for person in self._people.values()
                    if person.subject_id not in subjects | protected
                    and self._timestamp - person.last_seen > self.person_history_seconds
                ),
                key=lambda person: (person.last_seen, person.subject_id),
            )
            if len(eligible) < needed:
                return False
            retired = {person.subject_id for person in eligible[:needed]}
            for subject in retired:
                for record in self._people[subject].history:
                    self._remove_contour_record(record)
                del self._people[subject]
            self._prune_contour_history()
            for key, event in list(self._events.items()):
                if retired.intersection(event.subject_ids):
                    self._discard_group(key)
            self._counters["expired_batch_entities_retired"] += len(retired)
            return True

    def can_reserve_native_person_slots(self, keys: set[tuple[str, str, str]]) -> bool:
        """A continuing native identity does not need a new scene entity slot."""
        with self._lock:
            existing = {
                tuple(
                    person.native_track.get(name)
                    for name in ("backend", "epoch", "object_id")
                )
                for person in self._people.values()
                if person.native_track
            }
            return self.can_reserve_person_slots(len(keys - existing))

    def can_reserve_person_slots(self, new_people: int) -> bool:
        """Read-only native-batch preflight, before registry identity allocation."""
        new_people = _integer(new_people, "new_people", 0)
        with self._lock:
            needed = len(self._people) + new_people - self.max_entities
            if needed <= 0:
                return True
            if self.person_history_seconds is None:
                return False
            protected = {
                sid
                for observations in self._source_pins.values()
                for sid, _at, _sequence, _bbox in observations
            } | {
                sid
                for event in self._events.values()
                if event.phase in {"suspected", "confirmed"}
                for sid in event.subject_ids
            }
            return (
                sum(
                    person.subject_id not in protected
                    and self._timestamp - person.last_seen > self.person_history_seconds
                    for person in self._people.values()
                )
                >= needed
            )

    def observe_person(
        self,
        subject_id: str,
        *,
        timestamp_seconds: float,
        frame_sequence: int,
        bbox: NormalizedBBox,
        identity: Mapping | None = None,
        observation_state: str = "observed",
        allow_historical: bool = False,
        native_track: Mapping | None = None,
        tracking_confirmation: str | None = None,
        mask_contours: Mapping | None = None,
    ) -> bool:
        subject_id = _text(subject_id, "subject_id")
        timestamp = _time(timestamp_seconds)
        sequence = _integer(frame_sequence, "frame_sequence", 0)
        if not isinstance(bbox, NormalizedBBox):
            raise ValueError("bbox must be a NormalizedBBox")
        if not isinstance(observation_state, str) or observation_state not in {
            "observed",
            "predicted",
            "held",
            "occluded",
            "lost",
        }:
            raise ValueError("unsupported observation_state")
        if identity is not None and not isinstance(identity, Mapping):
            raise ValueError("identity must be a mapping")
        safe_contours = None
        if mask_contours is not None:
            validate_mask_contours(mask_contours)
            if (
                mask_contours["source_frame_sequence"] != sequence
                or mask_contours["source_timestamp_seconds"] != timestamp
                or observation_state != "observed"
            ):
                raise ValueError("mask_contours must match the observed source frame")
            safe_contours = json_mapping_as_dict(mask_contours)
        if type(allow_historical) is not bool:
            raise ValueError("allow_historical must be a boolean")
        if native_track is not None and not isinstance(native_track, Mapping):
            raise ValueError("native_track must be a mapping")
        if tracking_confirmation is not None and (
            not isinstance(tracking_confirmation, str)
            or tracking_confirmation not in {"provisional", "confirmed"}
        ):
            raise ValueError("unsupported tracking_confirmation")
        safe_native_track = {
            key: _text(native_track[key], f"native_track.{key}")
            for key in _NATIVE_TRACK_KEYS
            if native_track is not None and key in native_track
        }
        safe_identity = {
            key: _text(identity[key], f"identity.{key}")
            for key in _IDENTITY_KEYS
            if identity is not None and key in identity and identity[key] is not None
        }
        if identity is not None and identity.get("identity_reset") is True:
            safe_identity["identity_reset"] = True
        with self._lock:
            self.advance(timestamp)
            if observation_state != "observed":
                return self._reject("nonobserved_people_ignored")
            if self._timestamp - timestamp > self.retention_seconds:
                return self._reject("stale_people_ignored")
            if (
                allow_historical
                and self.person_history_seconds is not None
                and self._timestamp - timestamp > self.person_history_seconds
            ):
                return self._reject("expired_person_history_ignored")
            person = self._people.get(subject_id)
            if person is not None:
                if (
                    person.native_track
                    and not safe_native_track
                    and safe_identity.get("identity_source")
                    in {"face_match", "track_history"}
                ):
                    # A canonical person's face result can arrive after the
                    # body's result. Refresh its name/identity without moving
                    # the body to a face rectangle or fabricating a body frame.
                    # Unknown observations still need their true source record
                    # for stranger votes; identity confidence is not geometry.
                    if not any(
                        record["frame_sequence"] == sequence
                        and record["timestamp_seconds"] == timestamp
                        for record in person.history
                    ) and not self._insert_historical_person(
                        person, timestamp, sequence, bbox, {},
                        identity_only=True,
                    ):
                        return False
                    self._update_identity(person, safe_identity, timestamp, sequence)
                    self._counters["native_identity_updates"] += 1
                    return True
                if timestamp < person.last_seen or sequence < person.frame_sequence:
                    if allow_historical:
                        return self._insert_historical_person(
                            person,
                            timestamp,
                            sequence,
                            bbox,
                            safe_native_track,
                            tracking_confirmation,
                            safe_contours,
                        )
                    if self._has_source_observation(subject_id, timestamp, sequence):
                        if self._update_identity(
                            person, safe_identity, timestamp, sequence
                        ):
                            self._counters["historical_identity_updates"] += 1
                            return True
                    return self._reject("stale_people_ignored")
                same_frame = sequence == person.frame_sequence
                if same_frame and timestamp != person.last_seen:
                    return self._reject("stale_people_ignored")
            else:
                if len(self._people) >= self.max_entities:
                    return self._reject("entity_capacity_rejected")
                person = _Person(
                    subject_id,
                    bbox,
                    timestamp,
                    timestamp,
                    sequence,
                    deque(maxlen=self.person_history_size),
                )
                self._people[subject_id] = person
                self._counters["entities_created"] += 1
                same_frame = False
            self._update_identity(person, safe_identity, timestamp, sequence)
            if safe_native_track:
                person.native_track = safe_native_track
            if tracking_confirmation is not None:
                person.tracking_confirmation = tracking_confirmation
            if same_frame:
                if allow_historical:
                    person.bbox = bbox
                    self._insert_historical_person(
                        person,
                        timestamp,
                        sequence,
                        bbox,
                        safe_native_track,
                        tracking_confirmation,
                        safe_contours,
                    )
                elif tracking_confirmation is not None:
                    for record in person.history:
                        if record["frame_sequence"] == sequence:
                            record["tracking_confirmation"] = tracking_confirmation
                if safe_contours is not None:
                    person.bbox = bbox
                    for record in person.history:
                        if record["frame_sequence"] == sequence:
                            record["bbox"] = bbox.as_dict()
                            record["mask_contours"] = safe_contours
                            self._retain_contour_record(record)
                self._prune_contour_history()
                self._counters["same_frame_person_updates"] += 1
                return True
            person.bbox, person.last_seen, person.frame_sequence = (
                bbox,
                timestamp,
                sequence,
            )
            if len(person.history) == person.history.maxlen:
                self._remove_contour_record(person.history[0])
            person.history.append(
                {
                    "timestamp_seconds": timestamp,
                    "frame_sequence": sequence,
                    "bbox": bbox.as_dict(),
                    **({"mask_contours": safe_contours} if safe_contours else {}),
                    **(
                        {"native_track": safe_native_track} if safe_native_track else {}
                    ),
                    **(
                        {"tracking_confirmation": tracking_confirmation}
                        if tracking_confirmation is not None
                        else {}
                    ),
                }
            )
            self._retain_contour_record(person.history[-1])
            self._prune_person_history(person)
            self._prune_contour_history()
            self._counters["person_observations"] += 1
            return True

    def _prune_person_history(self, person: _Person) -> None:
        if self.person_history_seconds is not None:
            while person.history and (
                person.history[0]["timestamp_seconds"]
                < self._timestamp - self.person_history_seconds
            ):
                self._remove_contour_record(person.history.popleft())

    def _remove_contour_record(self, record: dict) -> None:
        prior = self._contour_records.pop(id(record), None)
        if prior is not None:
            self._contour_points -= prior[1]

    def _retain_contour_record(self, record: dict) -> None:
        self._remove_contour_record(record)
        contours = record.get("mask_contours")
        if contours is None:
            return
        points = sum(len(polygon) for polygon in contours["polygons"])
        self._contour_points += points
        self._contour_serial += 1
        serial = self._contour_serial
        self._contour_records[id(record)] = (record, points, serial)
        heapq.heappush(self._contour_order, (
            record["timestamp_seconds"], record["frame_sequence"], serial, id(record),
        ))

    def _prune_contour_history(self) -> None:
        """Drop oldest source masks only; never remove observation/event evidence."""
        while self._contour_points > self.max_contour_history_points:
            _timestamp, _sequence, serial, key = heapq.heappop(self._contour_order)
            entry = self._contour_records.get(key)
            if entry is None or entry[2] != serial:
                continue
            record, points, _serial = entry
            record.pop("mask_contours", None)
            self._remove_contour_record(record)
            self._counters["contour_history_masks_evicted"] += 1
            self._counters["contour_history_points_evicted"] += points
        # Same-frame replacement and normal history expiration leave lazy heap
        # entries. Compact them so auxiliary storage has its own strict bound.
        if len(self._contour_order) > 2 * len(self._contour_records) + 64:
            self._contour_order = [
                (record["timestamp_seconds"], record["frame_sequence"], serial, key)
                for key, (record, _points, serial) in self._contour_records.items()
            ]
            heapq.heapify(self._contour_order)

    def _insert_historical_person(
        self,
        person: _Person,
        timestamp: float,
        sequence: int,
        bbox: NormalizedBBox,
        native_track: dict,
        tracking_confirmation: str | None = None,
        mask_contours: dict | None = None,
        *,
        identity_only: bool = False,
    ) -> bool:
        """Add completed source geometry without rewinding present-day location."""
        if any(
            (
                record["frame_sequence"] < sequence
                and record["timestamp_seconds"] > timestamp
            )
            or (
                record["frame_sequence"] > sequence
                and record["timestamp_seconds"] < timestamp
            )
            or (
                record["frame_sequence"] == sequence
                and record["timestamp_seconds"] != timestamp
            )
            for record in person.history
        ):
            return self._reject("inconsistent_person_history_ignored")
        previous = next(
            (item for item in person.history if item["frame_sequence"] == sequence), {}
        )
        confirmation = (
            tracking_confirmation
            if tracking_confirmation is not None
            else (previous.get("tracking_confirmation"))
        )
        record = {
            "timestamp_seconds": timestamp,
            "frame_sequence": sequence,
            "bbox": bbox.as_dict(),
            **({"identity_only": True} if identity_only else {}),
            **({"mask_contours": mask_contours} if mask_contours else {}),
            **({"native_track": native_track} if native_track else {}),
            **(
                {"tracking_confirmation": confirmation}
                if confirmation is not None
                else {}
            ),
        }
        if native_track:
            person.native_track = native_track
        records = [
            item for item in person.history if item["frame_sequence"] != sequence
        ]
        records.append(record)
        records.sort(
            key=lambda item: (item["timestamp_seconds"], item["frame_sequence"])
        )
        retained = deque(records, maxlen=self.person_history_size)
        retained_ids = {id(item) for item in retained}
        for old_record in person.history:
            if id(old_record) not in retained_ids:
                self._remove_contour_record(old_record)
        person.history = retained
        if id(record) in retained_ids:
            self._retain_contour_record(record)
        person.first_seen = min(person.first_seen, timestamp)
        self._prune_person_history(person)
        self._prune_contour_history()
        self._counters["historical_person_observations"] += 1
        return record in person.history

    @staticmethod
    def _update_identity(
        person: _Person,
        identity: dict,
        timestamp: float,
        sequence: int,
    ) -> bool:
        # Face inference arrives behind tracking. Its own source watermark protects
        # established identity without rewinding the body's location or visibility.
        if (
            not identity
            or timestamp < person.identity_source[0]
            or sequence < person.identity_source[1]
        ):
            return False
        if identity.get("identity_reset"):
            person.identity.clear()
            identity = {
                key: value for key, value in identity.items() if key != "identity_reset"
            }
        if (
            identity.get("status") != "known"
            and person.identity.get("status") == "known"
        ) or (
            identity.get("status") not in {"known", "unknown"}
            and person.identity.get("status") == "unknown"
        ):
            return False
        if any(
            key in identity and identity[key] != person.identity.get(key)
            for key in ("person_id", "matched_memory_id")
        ):
            person.identity.clear()
        person.identity.update(identity)
        person.identity_source = (timestamp, sequence)
        return True

    def forget_identity(self, person_id: str) -> None:
        """Immediately remove a deleted gallery identity from retained entities."""
        person_id = _text(person_id, "person_id")
        with self._lock:
            for person in self._people.values():
                if person_id in (
                    person.identity.get("person_id"),
                    person.identity.get("matched_memory_id"),
                ):
                    person.identity.clear()
                    self._counters["identities_forgotten"] += 1

    def _has_source_observation(
        self, subject: str, timestamp: float, sequence: int
    ) -> bool:
        if subject not in self._people:
            return False
        records = [
            item
            for item in self._people[subject].history
            if (
                item["frame_sequence"] == sequence
                and abs(item["timestamp_seconds"] - timestamp) <= 0.6
            )
        ]
        # If a retained observation was explicitly made uncertain, an older pin
        # must not bypass that correction. Pins were confirmed when admitted.
        if records:
            return any(
                item.get("tracking_confirmation", "confirmed") == "confirmed"
                for item in records
            )
        return any(
            sid == subject and frame == sequence and abs(at - timestamp) <= 0.6
            for observations in self._source_pins.values()
            for sid, at, frame, _bbox in observations
        )

    def pin_source_observations(
        self,
        pin_id: str,
        observations: tuple[tuple[str, float, int, NormalizedBBox], ...],
    ) -> bool:
        """Retain an inspection's real source observations while inference runs.

        Pins must exactly match current source history (with 1us transport tolerance)
        and cannot create people, refresh visibility, or replace current geometry.
        At most four inspections, each four frames and 128 people, can be pinned.
        """
        pin_id = _text(pin_id, "pin_id")
        if not isinstance(observations, (tuple, list)) or len(observations) > 512:
            raise ValueError("source pins must contain at most 512 observations")
        checked = []
        seen = set()
        for observation in observations:
            if not isinstance(observation, (tuple, list)) or len(observation) != 4:
                raise ValueError(
                    "source pin observations require person, time, frame, bbox"
                )
            subject, at, sequence, bbox = observation
            subject = _text(subject, "subject_id")
            at, sequence = _time(at), _integer(sequence, "frame_sequence", 0)
            if not isinstance(bbox, NormalizedBBox) or (subject, sequence) in seen:
                raise ValueError("source pins require distinct real source boxes")
            seen.add((subject, sequence))
            checked.append((subject, at, sequence, bbox))
        if (
            len({item[2] for item in checked}) > 4
            or len({item[0] for item in checked}) > 128
        ):
            raise ValueError("source pins exceed four frames or 128 people")
        values = tuple(checked)
        with self._lock:
            if pin_id in self._source_pins:
                if self._source_pins[pin_id] == values:
                    return True
                return self._reject("changed_source_pin_ignored")
            if len(self._source_pins) >= self.max_source_pin_jobs:
                return self._reject("source_pin_capacity_rejected")
            for subject, at, sequence, bbox in values:
                person = self._people.get(subject)
                in_history = person is not None and any(
                    item["frame_sequence"] == sequence
                    and abs(item["timestamp_seconds"] - at) <= 1e-6
                    and item["bbox"] == bbox.as_dict()
                    and item.get("tracking_confirmation", "confirmed") == "confirmed"
                    for item in person.history
                )
                in_retained_source = any(
                    sid == subject and seq == sequence and abs(source_at - at) <= 1e-6
                    and source_bbox == bbox
                    for observations in self._source_pins.values()
                    for sid, source_at, seq, source_bbox in observations
                )
                if person is None or not (in_history or in_retained_source):
                    return self._reject("ungrounded_source_pin_ignored")
            self._source_pins[pin_id] = values
            self._counters["source_pins_created"] += 1
            return True

    def unpin_source_observations(self, pin_id: str) -> None:
        pin_id = _text(pin_id, "pin_id")
        with self._lock:
            if self._source_pins.pop(pin_id, None) is not None:
                self._counters["source_pins_released"] += 1
                self._prune_tracking_presence()

    def _transition(self, event: _Event, kind: str) -> None:
        self._transition_sequence += 1
        self._transitions.append(
            {
                "transition_id": (
                    f"transition-{self._namespace}-{self._transition_sequence}"
                ),
                "type": kind,
                "event_id": event.event_id,
                "event_label": event.event_label,
                "subject_ids": list(event.subject_ids),
                "phase": event.phase,
                "timestamp_seconds": event.last_seen,
                "frame_sequence": event.frame_sequence,
                "recorded_timestamp_seconds": self._timestamp,
            }
        )
        self._counters[f"event_{kind}s"] += 1

    def observe_event(
        self,
        event_label: str,
        subject_ids: tuple[str, ...],
        *,
        timestamp_seconds: float,
        frame_sequence: int,
        detected: bool | None,
        evidence_id: str,
        confidence: float | None = None,
        allow_historical: bool = False,
    ) -> bool:
        """Accept one immutable judgment grounded in one real source frame.

        Temporal workers may opt into source-ordered replay for late completions.
        One interval verdict contributes one judgment at its grounded endpoint,
        not one invented vote per image in the interval. Duplicate source frames
        never add votes, and later clear evidence still follows late positives.
        """
        if type(allow_historical) is not bool:
            raise ValueError("allow_historical must be a boolean")
        return self._submit_event(
            event_label,
            subject_ids,
            timestamp_seconds=timestamp_seconds,
            frame_sequence=frame_sequence,
            detected=detected,
            evidence_id=evidence_id,
            confidence=confidence,
            revision=0,
            revisable=False,
            allow_historical=allow_historical,
        )

    def revise_event_evidence(
        self,
        event_label: str,
        subject_ids: tuple[str, ...],
        *,
        timestamp_seconds: float,
        frame_sequence: int,
        detected: bool | None,
        evidence_id: str,
        confidence: float | None = None,
        revision: int = 0,
    ) -> bool:
        """Replay retained source evidence after an explicit, versioned correction.

        Initial evidence uses revision 0. Higher revisions must preserve its source
        frame and timestamp. Empty participants with detected=None explicitly withdraw
        a retained observation. Pruned evidence cannot be revised or resurrected.
        """
        return self._submit_event(
            event_label,
            subject_ids,
            timestamp_seconds=timestamp_seconds,
            frame_sequence=frame_sequence,
            detected=detected,
            evidence_id=evidence_id,
            confidence=confidence,
            revision=revision,
            revisable=True,
        )

    def _submit_event(
        self,
        event_label: str,
        subject_ids: tuple[str, ...],
        *,
        timestamp_seconds: float,
        frame_sequence: int,
        detected: bool | None,
        evidence_id: str,
        confidence: float | None,
        revision: int,
        revisable: bool,
        allow_historical: bool = False,
    ) -> bool:
        if not isinstance(event_label, str) or event_label not in _EVENT_LABELS:
            raise ValueError("unsupported event_label")
        revision = _integer(revision, "revision", 0)
        withdrawal = revisable and revision > 0 and detected is None
        if (
            not isinstance(subject_ids, (tuple, list))
            or not (0 if withdrawal else 1) <= len(subject_ids) <= self.max_entities
        ):
            raise ValueError(
                "subject_ids must be bounded and nonempty except withdrawals"
            )
        participants = tuple(
            sorted({_text(item, "subject_id") for item in subject_ids})
        )
        timestamp = _time(timestamp_seconds)
        sequence = _integer(frame_sequence, "frame_sequence", 0)
        evidence_id = _text(evidence_id, "evidence_id")
        if detected is not None and type(detected) is not bool:
            raise ValueError("detected must be true, false or None")
        if confidence is not None:
            confidence = _time(confidence, "confidence")
            if confidence > 1:
                raise ValueError("confidence must be between 0 and 1")
        key, evidence_key = (event_label, participants), (event_label, evidence_id)
        with self._lock:
            old_key = self._record_groups.get(evidence_key)
            old_ledger = self._ledgers.get(old_key)
            old = old_ledger.records[evidence_key] if old_ledger else None
            if not revisable and (old is not None or evidence_key in self._evidence):
                prior_participants = (
                    old.subject_ids if old is not None else self._evidence[evidence_key]
                )
                reason = (
                    "duplicate_evidence_ignored"
                    if prior_participants == participants
                    else "reassigned_evidence_ignored"
                )
                return self._reject(reason)
            if revisable:
                if old is None and (revision > 0 or evidence_key in self._evidence):
                    return self._reject("unretained_revision_ignored")
                if old is not None:
                    if revision <= old.revision:
                        return self._reject("duplicate_revision_ignored")
                    if (timestamp, sequence) != (old.timestamp, old.sequence):
                        return self._reject("retimed_revision_ignored")
            if self._timestamp - timestamp > self.retention_seconds:
                return self._reject("stale_evidence_ignored")
            if event_label != "stranger" and sequence <= self._presence_floor[1]:
                return self._reject("pruned_source_revision_ignored")
            if any(subject not in self._people for subject in participants):
                return self._reject("unknown_participants_ignored")
            if not participants:
                assert old_key is not None  # An initial withdrawal is invalid above.
                key = old_key
            event = self._events.get(key)
            if (
                not revisable
                and not allow_historical
                and event is not None
                and (timestamp < event.last_seen or sequence <= event.frame_sequence)
            ):
                return self._reject("stale_evidence_ignored")
            if not all(
                self._has_source_observation(subject, timestamp, sequence)
                for subject in participants
            ):
                return self._reject("ungrounded_evidence_ignored")
            ledger = self._ledgers.get(key)
            if ledger is not None and (
                timestamp < ledger.floor[0] or sequence <= ledger.floor[1]
            ):
                return self._reject("pruned_source_revision_ignored")
            if key not in self._events and len(self._events) >= self.max_events:
                ended = next(
                    (
                        item
                        for item, value in self._events.items()
                        if value.phase in {"ended", "retracted"} and item != old_key
                    ),
                    None,
                )
                if ended is None:
                    return self._reject("event_capacity_rejected")
                self._discard_group(ended)
                self._counters["ended_events_evicted"] += 1
            # All rejection checks precede mutations of the old contribution.
            affected = tuple(
                dict.fromkeys(item for item in (old_key, key) if item is not None)
            )
            previous = {item: self._events.get(item) for item in affected}
            previous_ids = {item: self._episode_ids(item) for item in affected}
            if ledger is None:
                seed = _Event(event_label, key[1], deque(maxlen=self.history_size))
                ledger = _Ledger(
                    seed, floor=self._presence_floor
                    if event_label != "stranger" else (-1.0, -1),
                )
                self._ledgers[key] = ledger
            if old_ledger is not None:
                del old_ledger.records[evidence_key]
            record = _Evidence(
                event_label,
                evidence_id,
                participants,
                timestamp,
                sequence,
                detected,
                confidence,
                revision,
                old.episode_id if old and old.subject_ids == participants else None,
            )
            ledger.records[evidence_key] = record
            ledger.watermark = (
                max(ledger.watermark[0], timestamp),
                max(ledger.watermark[1], sequence),
            )
            self._record_groups[evidence_key] = key
            self._evidence[evidence_key] = participants
            self._evidence.move_to_end(evidence_key)
            while len(self._evidence) > self.max_events * self.history_size:
                self._evidence.popitem(last=False)
            self._timestamp = max(self._timestamp, timestamp)
            for affected_key in affected:
                current, retained_ids = self._replay(affected_key)
                prior = previous[affected_key]
                for removed_id in sorted(previous_ids[affected_key] - retained_ids):
                    removed = replace(
                        prior or current, event_id=removed_id, phase="retracted"
                    )
                    self._transition(removed, "retract")
                if current.event_id is None and prior is not None and prior.event_id:
                    current = replace(
                        current,
                        event_id=prior.event_id,
                        phase="retracted",
                        first_seen=prior.first_seen,
                        confirmed_since_timestamp=prior.confirmed_since_timestamp,
                        ended_since_timestamp=prior.ended_since_timestamp,
                        last_positive_timestamp=prior.last_positive_timestamp,
                    )
                self._events[affected_key] = current
                if current.event_id and current.phase != "retracted":
                    kind = "update"
                    if (
                        prior is None
                        or prior.event_id != current.event_id
                        or prior.phase == "retracted"
                    ):
                        kind = "start"
                    elif prior.phase != "ended" and current.phase == "ended":
                        kind = "end"
                    self._transition(current, kind)
                self._prune_ledger(affected_key)
            self._counters["event_observations"] += 1
            if old is not None:
                self._counters["evidence_revisions"] += 1
            return True

    def _episode_ids(self, key: tuple[str, tuple[str, ...]]) -> set[str]:
        ledger = self._ledgers.get(key)
        if ledger is None:
            return set()
        ids = {
            record.episode_id for record in ledger.records.values() if record.episode_id
        }
        if ledger.seed.event_id:
            ids.add(ledger.seed.event_id)
        return ids

    def _source_batches(self, ledger: _Ledger) -> list[list[_Evidence]]:
        batches: dict[int, list[_Evidence]] = {}
        for record in sorted(
            ledger.records.values(), key=lambda item: (item.sequence, item.evidence_id)
        ):
            batches.setdefault(record.sequence, []).append(record)
        return list(batches.values())

    def _fold_source(
        self,
        event: _Event,
        records: list[_Evidence],
        used_ids: set[str],
        preferred: str | None = None,
    ) -> _Event:
        active = [
            record for record in records if record.subject_ids == event.subject_ids
        ]
        if not active:
            return event
        # Multiple IDs attributed to one frame are one observation, never two votes.
        detected = (
            True
            if any(record.detected is True for record in active)
            else False
            if all(record.detected is False for record in active)
            else None
        )
        timestamp = max(record.timestamp for record in active)
        if detected is True and event.phase in {"ended", "retracted"}:
            candidates = [record.episode_id for record in active] + [preferred]
            episode_id = next(
                (value for value in candidates if value and value not in used_ids), None
            )
            if episode_id is None:
                self._episode += 1
                episode_id = f"event-{self._namespace}-{self._episode}"
            event = _Event(
                event.event_label,
                event.subject_ids,
                deque(maxlen=self.history_size),
                event_id=episode_id,
                phase="suspected",
                first_seen=timestamp,
            )
            used_ids.add(episode_id)
        event.last_seen = max(event.last_seen, timestamp)
        event.frame_sequence = records[0].sequence
        confidences = [
            record.confidence
            for record in active
            if record.detected is detected and record.confidence is not None
        ]
        event.confidence = max(confidences) if confidences else None
        event.uncertain = detected is None
        if detected is True:
            event.last_positive_timestamp = timestamp
            event.positives += 1
            event.negatives = 0
            if event.positives >= self.confirm_observations:
                event.phase = "confirmed"
                if event.confirmed_since_timestamp is None:
                    event.confirmed_since_timestamp = timestamp
        elif detected is False:
            event.negatives += 1
            event.positives = 0
            if event.event_id and event.negatives >= self.clear_observations:
                event.phase = "ended"
                if event.ended_since_timestamp is None:
                    event.ended_since_timestamp = timestamp
        for record in active:
            record.episode_id = event.event_id
        event.history.append(
            {
                "timestamp_seconds": timestamp,
                "frame_sequence": event.frame_sequence,
                "evidence_id": active[0].evidence_id,
                "evidence_ids": [record.evidence_id for record in active],
                "revision": max(record.revision for record in active),
                "detected": detected,
                "confidence": event.confidence,
                "phase": event.phase,
            }
        )
        return event

    def _replay(
        self, key: tuple[str, tuple[str, ...]], *, through_sequence: int | None = None,
    ) -> tuple[_Event, set[str]]:
        ledger = self._ledgers[key]
        event = replace(ledger.seed, history=deque(maxlen=self.history_size))
        used_ids = {event.event_id} if event.event_id else set()
        batches = [
            batch for batch in self._source_batches(ledger)
            if through_sequence is None or batch[0].sequence <= through_sequence
        ]
        judgments = {batch[0].sequence: (index, batch)
                     for index, batch in enumerate(batches)}
        presence = {
            sequence: value for sequence, value in self._tracking_presence.items()
            if sequence > ledger.floor[1]
            and (through_sequence is None or sequence <= through_sequence)
        }
        for sequence in sorted(presence.keys() | judgments.keys()):
            if sequence in presence:
                at, present = presence[sequence]
                self._fold_presence(event, at, sequence, present)
            if sequence not in judgments:
                continue
            index, records = judgments[sequence]
            preferred = next(
                (
                    record.episode_id
                    for batch in batches[index:]
                    for record in batch
                    if record.subject_ids == key[1]
                    and record.detected is True
                    and record.episode_id
                    and record.episode_id not in used_ids
                ),
                None,
            )
            event = self._fold_source(event, records, used_ids, preferred)
        if through_sequence is None:
            event.last_seen = max(event.last_seen, ledger.watermark[0])
            event.frame_sequence = max(event.frame_sequence, ledger.watermark[1])
        return event, used_ids

    def _prune_ledger(self, key: tuple[str, tuple[str, ...]]) -> None:
        ledger = self._ledgers[key]
        while len(ledger.records) > self.history_size:
            # Prune whole source frames so duplicate IDs cannot cross the replay seed.
            batch = self._source_batches(ledger)[0]
            seed, _ = self._replay(key, through_sequence=batch[0].sequence)
            ledger.seed = replace(seed, history=deque(maxlen=0))
            ledger.floor = (
                max(record.timestamp for record in batch),
                batch[0].sequence,
            )
            for record in batch:
                evidence_key = (record.event_label, record.evidence_id)
                del ledger.records[evidence_key]
                self._record_groups.pop(evidence_key, None)
                self._counters["evidence_records_pruned"] += 1

    def finish_review(self) -> None:
        """Finish work without inventing evidence that an event stopped occurring."""
        with self._lock:
            self._review_complete = True

    def snapshot(self, *, include_history: bool = True) -> dict:
        """Copy scene state; internal scheduling can omit evidence histories."""
        if type(include_history) is not bool:
            raise ValueError("include_history must be a boolean")
        with self._lock:
            visible = {
                key
                for key, person in self._people.items()
                if self._timestamp - person.last_seen <= self.occlusion_seconds
            }
            positions = {
                person.subject_id: next(
                    (
                        item
                        for item in reversed(person.history)
                        if not item.get("identity_only")
                    ),
                    {},
                )
                for person in self._people.values()
            }
            entities = [
                {
                    "entity_id": person.subject_id,
                    "subject_id": person.subject_id,
                    "label": "person",
                    "bbox": person.bbox.as_dict(),
                    **(
                        {"mask_contours": positions[person.subject_id]["mask_contours"]}
                        if "mask_contours" in positions[person.subject_id]
                        else {}
                    ),
                    "identity": person.identity,
                    **(
                        {"tracking_confirmation": person.tracking_confirmation}
                        if person.tracking_confirmation is not None
                        else {}
                    ),
                    **(
                        {"native_track": person.native_track}
                        if person.native_track
                        else {}
                    ),
                    "observation_state": "observed"
                    if person.subject_id in visible
                    else "occluded",
                    "first_observed_timestamp_seconds": person.first_seen,
                    "last_observed_timestamp_seconds": person.last_seen,
                    "frame_sequence": person.frame_sequence,
                    **({"history": list(person.history)} if include_history else {}),
                }
                for person in self._people.values()
            ]
            events = [
                {
                    "event_id": event.event_id,
                    "event_label": event.event_label,
                    "subject_ids": list(event.subject_ids),
                    "phase": event.phase,
                    **(
                        {
                            "review_state": "complete",
                            "review_outcome": "unconfirmed"
                            if event.phase == "suspected" else event.phase,
                        }
                        if self._review_complete else {}
                    ),
                    "observation_state": "uncertain"
                    if event.uncertain
                    or any(subject not in visible for subject in event.subject_ids)
                    else "observed",
                    "first_evidence_timestamp_seconds": event.first_seen,
                    "last_evidence_timestamp_seconds": event.last_seen,
                    "last_positive_timestamp_seconds": event.last_positive_timestamp,
                    "confirmed_since_timestamp_seconds": (
                        event.confirmed_since_timestamp
                    ),
                    "ended_since_timestamp_seconds": event.ended_since_timestamp,
                    **({"ended_reason": event.ended_reason}
                       if event.ended_reason is not None else {}),
                    "frame_sequence": event.frame_sequence,
                    "confidence": event.confidence,
                    "positive_observations": event.positives,
                    "negative_observations": event.negatives,
                    **({"history": list(event.history)} if include_history else {}),
                }
                for event in self._events.values()
                if event.event_id is not None
            ]
            return deepcopy(
                {
                    "schema_version": "1",
                    "stream_session_id": self.stream_session_id,
                    "source_id": self.source_id,
                    "timestamp_seconds": self._timestamp,
                    "entities": entities,
                    "events": events,
                    "transitions": list(self._transitions) if include_history else [],
                    "source_pins": {
                        "jobs": len(self._source_pins),
                        "observations": sum(
                            len(items) for items in self._source_pins.values()
                        ),
                    },
                    "counters": dict(self._counters),
                }
            )

    def snapshot_at(
        self, timestamp_seconds: float, *, include_history: bool = False
    ) -> dict:
        """Project retained observations without aging the presentation clock backward.

        This bounded source view is for planning, not event-history replay. Events
        whose latest evidence is newer than the requested watermark are omitted.
        """
        timestamp = _time(timestamp_seconds)
        with self._lock:
            snapshot = self.snapshot(include_history=include_history)
            snapshot["timestamp_seconds"] = timestamp
            entities = []
            visible = set()
            for person in self._people.values():
                history = [
                    record
                    for record in person.history
                    if record["timestamp_seconds"] <= timestamp
                    and not record.get("identity_only")
                ]
                if not history:
                    continue
                latest = history[-1]
                observed = (
                    timestamp - latest["timestamp_seconds"] <= self.occlusion_seconds
                )
                if observed:
                    visible.add(person.subject_id)
                entities.append(
                    {
                        "entity_id": person.subject_id,
                        "subject_id": person.subject_id,
                        "label": "person",
                        "bbox": latest["bbox"],
                        **(
                            {"mask_contours": latest["mask_contours"]}
                            if "mask_contours" in latest else {}
                        ),
                        **(
                            {"tracking_confirmation": latest["tracking_confirmation"]}
                            if "tracking_confirmation" in latest
                            else {}
                        ),
                        "identity": person.identity
                        if person.identity_source[0] <= timestamp
                        else {},
                        **(
                            {"native_track": latest["native_track"]}
                            if "native_track" in latest
                            else {}
                        ),
                        "observation_state": "observed" if observed else "occluded",
                        "first_observed_timestamp_seconds": person.first_seen,
                        "last_observed_timestamp_seconds": latest["timestamp_seconds"],
                        "frame_sequence": latest["frame_sequence"],
                        **({"history": history} if include_history else {}),
                    }
                )
            snapshot["entities"] = entities
            snapshot["events"] = [
                event
                for event in snapshot["events"]
                if event["last_evidence_timestamp_seconds"] <= timestamp
            ]
            for event in snapshot["events"]:
                key = (event["event_label"], tuple(event["subject_ids"]))
                event["observation_state"] = (
                    "uncertain"
                    if self._events[key].uncertain
                    or any(subject not in visible for subject in event["subject_ids"])
                    else "observed"
                )
            snapshot["transitions"] = [
                item
                for item in snapshot["transitions"]
                if item.get("timestamp_seconds", timestamp) <= timestamp
            ]
            return deepcopy(snapshot)
