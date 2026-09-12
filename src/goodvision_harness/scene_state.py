"""Bounded, session-local people and event episodes built from source observations.

The registry owns person association. This module never merges people by geometry,
and a tracker prediction is never evidence that a person or event was observed.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict, deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import RLock
from uuid import uuid4

from .contracts import NormalizedBBox

_EVENT_LABELS = frozenset({"fight", "smoking", "fall", "weapon", "stranger"})
_IDENTITY_KEYS = ("status", "person_id", "name", "identity_source", "matched_memory_id")


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
    ) -> None:
        self.stream_session_id = _text(stream_session_id, "stream_session_id")
        self.source_id = _text(source_id, "source_id")
        self.max_entities = _integer(max_entities, "max_entities", 1, 4096)
        self.history_size = _integer(history_size, "history_size", 1, 1024)
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
        self.max_events = min(self.max_entities * len(_EVENT_LABELS), 2048)
        self._people: dict[str, _Person] = {}
        self._events: dict[tuple[str, tuple[str, ...]], _Event] = {}
        self._evidence: OrderedDict[tuple[str, str], tuple[str, ...]] = OrderedDict()
        self._ledgers: dict[tuple[str, tuple[str, ...]], _Ledger] = {}
        self._record_groups: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}
        self._source_pins: dict[
            str, tuple[tuple[str, float, int, NormalizedBBox], ...]
        ] = {}
        self._transitions: deque = deque(maxlen=self.history_size)
        self._counters: Counter = Counter()
        self._timestamp = 0.0
        self._namespace = uuid4().hex[:12]
        self._episode = 0
        self._transition_sequence = 0
        self._lock = RLock()

    def _reject(self, reason: str) -> bool:
        self._counters[reason] += 1
        return False

    def advance(self, timestamp_seconds: float) -> None:
        """Advance source time; occlusion never generates negative evidence."""
        timestamp = _time(timestamp_seconds)
        with self._lock:
            self._timestamp = max(self._timestamp, timestamp)
            expired = [
                key
                for key, person in self._people.items()
                if self._timestamp - person.last_seen > self.retention_seconds
            ]
            for key in expired:
                del self._people[key]
                self._counters["entities_expired"] += 1
            if expired:
                self._source_pins = {
                    key: tuple(item for item in observations if item[0] in self._people)
                    for key, observations in self._source_pins.items()
                }
            for key, event in list(self._events.items()):
                if self._timestamp - event.last_seen > self.retention_seconds or any(
                    subject not in self._people for subject in event.subject_ids
                ):
                    self._discard_group(key)
                    self._counters["events_expired"] += 1

    def _discard_group(self, key: tuple[str, tuple[str, ...]]) -> None:
        ledger = self._ledgers.pop(key, None)
        if ledger is not None:
            for evidence_key in ledger.records:
                self._record_groups.pop(evidence_key, None)
        self._events.pop(key, None)

    def observe_person(
        self,
        subject_id: str,
        *,
        timestamp_seconds: float,
        frame_sequence: int,
        bbox: NormalizedBBox,
        identity: Mapping | None = None,
        observation_state: str = "observed",
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
        safe_identity = {
            key: _text(identity[key], f"identity.{key}")
            for key in _IDENTITY_KEYS
            if identity is not None and key in identity and identity[key] is not None
        }
        with self._lock:
            self.advance(timestamp)
            if observation_state != "observed":
                return self._reject("nonobserved_people_ignored")
            if self._timestamp - timestamp > self.retention_seconds:
                return self._reject("stale_people_ignored")
            person = self._people.get(subject_id)
            if person is not None:
                if timestamp < person.last_seen or sequence < person.frame_sequence:
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
                    deque(maxlen=self.history_size),
                )
                self._people[subject_id] = person
                self._counters["entities_created"] += 1
                same_frame = False
            self._update_identity(person, safe_identity, timestamp, sequence)
            if same_frame:
                self._counters["same_frame_person_updates"] += 1
                return True
            person.bbox, person.last_seen, person.frame_sequence = (
                bbox,
                timestamp,
                sequence,
            )
            person.history.append(
                {
                    "timestamp_seconds": timestamp,
                    "frame_sequence": sequence,
                    "bbox": bbox.as_dict(),
                }
            )
            self._counters["person_observations"] += 1
            return True

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
        if identity.get("status") != "known" and person.identity:
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
        return any(
            item["frame_sequence"] == sequence
            and abs(item["timestamp_seconds"] - timestamp) <= 0.6
            for item in self._people[subject].history
        ) or any(
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
            if len(self._source_pins) >= 4:
                return self._reject("source_pin_capacity_rejected")
            for subject, at, sequence, bbox in values:
                person = self._people.get(subject)
                if person is None or not any(
                    item["frame_sequence"] == sequence
                    and abs(item["timestamp_seconds"] - at) <= 1e-6
                    and item["bbox"] == bbox.as_dict()
                    for item in person.history
                ):
                    return self._reject("ungrounded_source_pin_ignored")
            self._source_pins[pin_id] = values
            self._counters["source_pins_created"] += 1
            return True

    def unpin_source_observations(self, pin_id: str) -> None:
        pin_id = _text(pin_id, "pin_id")
        with self._lock:
            if self._source_pins.pop(pin_id, None) is not None:
                self._counters["source_pins_released"] += 1

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
    ) -> bool:
        """Accept a new monotonic source observation; never implicitly revise it."""
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
            if any(subject not in self._people for subject in participants):
                return self._reject("unknown_participants_ignored")
            if not participants:
                assert old_key is not None  # An initial withdrawal is invalid above.
                key = old_key
            event = self._events.get(key)
            if (
                not revisable
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
                ledger = _Ledger(seed)
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
            event.positives += 1
            event.negatives = 0
            if event.positives >= self.confirm_observations:
                event.phase = "confirmed"
        elif detected is False:
            event.negatives += 1
            event.positives = 0
            if event.event_id and event.negatives >= self.clear_observations:
                event.phase = "ended"
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

    def _replay(self, key: tuple[str, tuple[str, ...]]) -> tuple[_Event, set[str]]:
        ledger = self._ledgers[key]
        event = replace(ledger.seed, history=deque(maxlen=self.history_size))
        used_ids = {event.event_id} if event.event_id else set()
        batches = self._source_batches(ledger)
        for index, records in enumerate(batches):
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
        event.last_seen = max(event.last_seen, ledger.watermark[0])
        event.frame_sequence = max(event.frame_sequence, ledger.watermark[1])
        return event, used_ids

    def _prune_ledger(self, key: tuple[str, tuple[str, ...]]) -> None:
        ledger = self._ledgers[key]
        while len(ledger.records) > self.history_size:
            # Prune whole source frames so duplicate IDs cannot cross the replay seed.
            batch = self._source_batches(ledger)[0]
            used_ids = {ledger.seed.event_id} if ledger.seed.event_id else set()
            seed = self._fold_source(ledger.seed, batch, used_ids)
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
            entities = [
                {
                    "entity_id": person.subject_id,
                    "subject_id": person.subject_id,
                    "label": "person",
                    "bbox": person.bbox.as_dict(),
                    "identity": person.identity,
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
                    "observation_state": "uncertain"
                    if event.uncertain
                    or any(subject not in visible for subject in event.subject_ids)
                    else "observed",
                    "first_evidence_timestamp_seconds": event.first_seen,
                    "last_evidence_timestamp_seconds": event.last_seen,
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
