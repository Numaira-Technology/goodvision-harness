"""Display behavior state on the latest face observation of the same person.

Event results describe their source frame, even when inference finishes much
later.  This display-only timeline replays those results in source order and
keeps each person's behavior state independent of both other people and the
face detector's faster updates.  Public results and their evidence stay intact.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .contracts import AnalysisResult, GroundedObject
from .grounding import SAFETY_EVENT_LABELS

_FrameKey = tuple[float, int]
_ResultKey = tuple[float, int, int, str]
_StateKey = tuple[str, str]
_FACE_STATUSES = frozenset(("known", "unknown", "unusable"))
_FACE_LABELS = frozenset(("known person", "unknown person", "unusable face"))
CLEAR_AFTER_NEGATIVES = 10


@dataclass(frozen=True, slots=True)
class PersonOverlayAnalysis:
    """Face annotations for display, over an unchanged source result."""

    source: AnalysisResult
    objects: tuple[GroundedObject, ...]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)


@dataclass(frozen=True, slots=True)
class _EventState:
    key: _ResultKey
    detected: bool
    negative_streak: int = 0


def is_face_result(result: AnalysisResult) -> bool:
    """An omitted or failed face branch is never an empty face observation."""

    return any(item.event_label == "stranger" for item in result.judgments)


def _frame_key(result: AnalysisResult) -> _FrameKey:
    return result.frame.timestamp_seconds, result.frame.source_frame_index


def _result_key(result: AnalysisResult) -> _ResultKey:
    return (*_frame_key(result), result.sequence, result.message_id)


def _faces(result: AnalysisResult) -> tuple[GroundedObject, ...]:
    return tuple(
        item for item in result.objects
        if item.attributes.get("status") in _FACE_STATUSES
        or item.label.casefold().replace("_", " ").strip() in _FACE_LABELS
    )


def _subject_ids(item: GroundedObject) -> tuple[str, ...]:
    value = item.attributes.get("subject_ids")
    if value is None:
        value = (item.attributes.get("subject_id"),)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not isinstance(entry, str) or not entry.strip() for entry in value)
    ):
        return ()
    return tuple(dict.fromkeys(value))


def _face_subject(item: GroundedObject) -> str | None:
    ids = _subject_ids(item)
    return ids[0] if len(ids) == 1 else None


def _observed_face(item: GroundedObject) -> bool:
    state = item.attributes.get("face_observation_state")
    if state is not None:
        return state == "observed"
    return not (
        item.attributes.get("identity_source") == "track_history"
        and item.attributes.get("observation_status") is None
    )


class PersonEventOverlay:
    """Bounded source-time history and person-specific behavior state.

    Callers serialize access. A source-time clear affects only people actually
    observed in a nearby face result. Losing a face suppresses its box, without
    treating absence as a behavioral clear. Events whose source face result has
    not arrived are reconsidered when that result arrives.
    """

    def __init__(
        self,
        *,
        source_match_seconds: float = 0.6,
        face_hold_seconds: float = 3.0,
        history_seconds: float = 90.0,
        max_history: int = 1024,
        clear_after_negatives: int = CLEAR_AFTER_NEGATIVES,
    ) -> None:
        for name, value in (
            ("source_match_seconds", source_match_seconds),
            ("face_hold_seconds", face_hold_seconds),
            ("history_seconds", history_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if max_history < 2:
            raise ValueError("max_history must be at least two")
        if type(clear_after_negatives) is not int or clear_after_negatives < 1:
            raise ValueError("clear_after_negatives must be a positive integer")
        self.source_match_seconds = float(source_match_seconds)
        self.face_hold_seconds = float(face_hold_seconds)
        self.history_seconds = float(history_seconds)
        self.max_history = int(max_history)
        self.clear_after_negatives = clear_after_negatives
        self._retired_sessions: deque[str] = deque(maxlen=16)
        self._session_id: str | None = None
        self._source_id: str | None = None
        self._results: list[AnalysisResult] = []
        self._base_state: dict[_StateKey, _EventState] = {}
        self._base_face: AnalysisResult | None = None
        self._folded_through: _ResultKey | None = None
        self._has_face_results = False

    @property
    def has_face_results(self) -> bool:
        return self._has_face_results

    def reset(self) -> None:
        """Discard annotations and prevent an old session from reappearing."""

        if self._session_id is not None:
            self._retired_sessions.append(self._session_id)
        self._session_id = None
        self._source_id = None
        self._results.clear()
        self._base_state.clear()
        self._base_face = None
        self._folded_through = None
        self._has_face_results = False

    def submit_result(self, result: AnalysisResult) -> None:
        if result.stream_session_id in self._retired_sessions:
            return
        if self._session_id != result.stream_session_id:
            self.reset()
            self._session_id = result.stream_session_id
            self._source_id = result.source_id
        if result.source_id != self._source_id:
            return
        key = _result_key(result)
        # Once folded, source history cannot be revised safely. In particular,
        # a very late positive must not bypass an already folded newer clear.
        if self._folded_through is not None and key[:2] <= self._folded_through[:2]:
            return
        self._results = [
            item for item in self._results if item.message_id != result.message_id
        ]
        position = bisect_right([_result_key(item) for item in self._results], key)
        self._results.insert(position, result)
        self._has_face_results |= is_face_result(result)
        self._prune()

    def select(
        self, media_timestamp: float, frame_sequence: int | None = None
    ) -> PersonOverlayAnalysis | None:
        query = (
            float(media_timestamp),
            math.inf if frame_sequence is None else frame_sequence,
        )
        if self._folded_through is not None and query < self._folded_through[:2]:
            return None
        results = [item for item in self._results if _frame_key(item) <= query]
        faces = self._face_history(results)
        if not faces:
            return None
        source = faces[-1]
        if (
            _frame_key(source) > query
            or media_timestamp - source.frame.timestamp_seconds
            > self.face_hold_seconds + 1e-9
        ):
            return None
        state = dict(self._base_state)
        self._apply_results(state, results, faces)
        objects = []
        for item in _faces(source):
            subject_id = _face_subject(item)
            active = tuple(
                label for label in SAFETY_EVENT_LABELS
                if subject_id is not None
                and (event := state.get((subject_id, label))) is not None
                and event.detected
            )
            objects.append(replace(
                item, attributes={**item.attributes, "active_events": active}
            ))
        return PersonOverlayAnalysis(source, tuple(objects))

    def _face_history(
        self, results: Sequence[AnalysisResult]
    ) -> list[AnalysisResult]:
        return sorted(
            (
                *(() if self._base_face is None else (self._base_face,)),
                *(item for item in results if is_face_result(item)),
            ),
            key=_result_key,
        )

    def _apply_results(
        self,
        state: dict[_StateKey, _EventState],
        results: Sequence[AnalysisResult],
        faces: Sequence[AnalysisResult],
    ) -> None:
        face_keys = [_frame_key(item) for item in faces]
        # A repeated frame or corrected message is one observation per event,
        # even when independent branches give it different message IDs.
        observations = {}
        for result in results:
            for judgment in result.judgments:
                if judgment.event_label in SAFETY_EVENT_LABELS:
                    observations[(*_frame_key(result), judgment.event_label)] = (
                        result, judgment,
                    )
        for result, judgment in observations.values():
            position = bisect_right(face_keys, _frame_key(result)) - 1
            if position < 0:
                continue
            face = faces[position]
            if (
                result.frame.timestamp_seconds - face.frame.timestamp_seconds
                > self.source_match_seconds + 1e-9
            ):
                continue
            visible = {
                subject_id for item in _faces(face)
                if _observed_face(item)
                and (subject_id := _face_subject(item)) is not None
            }
            objects = {item.object_id: item for item in result.objects}
            key = _result_key(result)
            owners: set[str] = set()
            fully_associated = True
            if judgment.detected:
                for object_id in judgment.object_ids:
                    item = objects.get(object_id)
                    ids = _subject_ids(item) if item is not None else ()
                    if not ids:
                        fully_associated = False
                    owners.update(ids)
                # A temporary body identity is not yet proof that an
                # observed face belongs to somebody else. Keep its event
                # pending without clearing visible people's older state.
                fully_associated &= owners <= visible
            affected = owners | (visible if fully_associated else set())
            for subject_id in affected:
                state_key = (subject_id, judgment.event_label)
                previous = state.get(state_key)
                if previous is None or previous.key <= key:
                    positive = subject_id in owners
                    streak = 0 if positive else min(
                        self.clear_after_negatives,
                        (previous.negative_streak if previous else 0) + 1,
                    )
                    active = positive or bool(
                        previous and previous.detected
                        and streak < self.clear_after_negatives
                    )
                    state[state_key] = _EventState(key, active, streak)

    def _prune(self) -> None:
        if not self._results:
            return
        cutoff = self._results[-1].frame.timestamp_seconds - self.history_seconds
        count = max(0, len(self._results) - self.max_history)
        while (
            count < len(self._results)
            and self._results[count].frame.timestamp_seconds < cutoff
        ):
            count += 1
        if not count:
            return
        # Fold complete frames so a duplicate observation cannot straddle the
        # compacted state and retained history and count twice.
        while (
            count < len(self._results)
            and _frame_key(self._results[count - 1]) == _frame_key(self._results[count])
        ):
            count += 1
        removed = self._results[:count]
        # Use the full face history: a same-source face branch may sort after
        # the behavior branch being folded and is still a valid association.
        self._apply_results(
            self._base_state, removed, self._face_history(self._results)
        )
        for result in removed:
            if is_face_result(result):
                self._base_face = result
        self._folded_through = _result_key(removed[-1])
        del self._results[:count]
        # Stable IDs are session-local and never reused. Cap dormant subjects as
        # well as raw history, so a long-running camera has bounded memory. The
        # folded source watermark rejects old observations for evicted IDs.
        recent_subjects: dict[str, _ResultKey] = {}
        for (subject_id, _), event in self._base_state.items():
            recent_subjects[subject_id] = max(
                event.key, recent_subjects.get(subject_id, event.key)
            )
        if len(recent_subjects) > self.max_history:
            retained = set(sorted(
                recent_subjects, key=recent_subjects.__getitem__, reverse=True
            )[:self.max_history])
            self._base_state = {
                key: value for key, value in self._base_state.items()
                if key[0] in retained
            }
