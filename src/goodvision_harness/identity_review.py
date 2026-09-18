"""Bounded session-local face review, independent of native detector track IDs."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .contracts import NormalizedBBox
from .grounding import ModelObject


@dataclass
class _Profile:
    samples: OrderedDict = field(default_factory=OrderedDict)
    person_id: str | None = None
    conflicting_identity: bool = False
    gallery_confirmed: bool = False
    reviewed: bool = False
    ambiguous: bool = False
    usable: bool = False
    latest_source: tuple[float, int] = (-1.0, -1)


class _FaceScoreBlock:
    """Compare one embedding dimension in bulk; update just a merged row."""

    def __init__(
        self, profiles: dict[str, _Profile], dimension: int, *, eager: bool
    ) -> None:
        self.indices = {subject: index for index, subject in enumerate(profiles)}
        self.samples = np.zeros((len(profiles), 4, dimension), dtype=np.float32)
        self.sources = np.full((len(profiles), 4), -1, dtype=np.int32)
        self.sample_ids = {}
        for subject, profile in profiles.items():
            index = self.indices[subject]
            for sample_index, sample in enumerate(profile.samples.values()):
                self.samples[index, sample_index] = sample
                source = index * 4 + sample_index
                self.sources[index, sample_index] = source
                self.sample_ids[id(sample)] = source
        flat = self.samples.reshape(-1, dimension)
        self.similarities = flat @ flat.T if eager else None
        self.rows: dict[int, np.ndarray] = {}
        self.computed: set[int] = set()
        self.scores = np.full((len(profiles), len(profiles)), np.nan, dtype=np.float32)
        if eager:
            similarities = self.similarities.copy()
            valid = self.sources.reshape(-1) >= 0
            similarities[~valid, :] = -np.inf
            similarities[:, ~valid] = -np.inf
            similarities = similarities.reshape(len(profiles), 4, len(profiles), 4)
            forward = np.sort(similarities.max(axis=3), axis=1)[:, -2, :]
            self.scores = np.minimum(forward, forward.T)
            self.computed.update(self.indices.values())

    def update(self, subject: str, profile: _Profile) -> None:
        index = self.indices[subject]
        self.sources[index] = -1
        for sample_index, sample in enumerate(profile.samples.values()):
            self.sources[index, sample_index] = self.sample_ids[id(sample)]
        self._compute_row(index)

    def _compute_row(self, index: int) -> None:
        sources = self.sources[index]
        targets = self.sources.reshape(-1)
        if self.similarities is not None:
            original = self.similarities[np.maximum(sources, 0)]
        else:
            missing = [
                source for source in sources
                if source >= 0 and source not in self.rows
            ]
            if missing:
                flat = self.samples.reshape(-1, self.samples.shape[-1])
                self.rows.update(zip(missing, flat[missing] @ flat.T, strict=True))
            original = np.stack([
                self.rows[source] if source >= 0 else np.zeros(len(targets), np.float32)
                for source in sources
            ])
        similarities = original[:, np.maximum(targets, 0)]
        similarities[sources < 0, :] = -np.inf
        similarities[:, targets < 0] = -np.inf
        similarities = similarities.reshape(4, len(self.indices), 4)
        forward = np.sort(similarities.max(axis=2), axis=0)[-2]
        backward = np.sort(similarities.max(axis=0), axis=1)[:, -2]
        self.scores[index] = np.minimum(forward, backward)
        self.scores[:, index] = self.scores[index]
        self.computed.add(index)

    def score(self, left: str, right: str) -> float:
        left_index, right_index = self.indices[left], self.indices[right]
        if left_index not in self.computed and right_index not in self.computed:
            self._compute_row(left_index)
        return float(self.scores[left_index, right_index])


class _ReviewScores:
    def __init__(
        self, profiles: OrderedDict[str, _Profile], dirty: set[str] | None = None
    ) -> None:
        groups: dict[int, dict[str, _Profile]] = {}
        for subject, profile in profiles.items():
            dimensions = {sample.size for sample in profile.samples.values()}
            if len(profile.samples) >= 2 and len(dimensions) == 1:
                dimension = dimensions.pop()
                groups.setdefault(dimension, {})[subject] = profile
        self.blocks = {}
        for dimension, group in groups.items():
            block = _FaceScoreBlock(
                group, dimension,
                eager=dirty is None
                or len(set(group) & dirty) >= max(16, len(group) // 4),
            )
            self.blocks.update(dict.fromkeys(group, block))

    def score(self, left: str, right: str) -> float | None:
        block = self.blocks.get(left)
        if block is None or block is not self.blocks.get(right):
            return None
        return block.score(left, right)

    def update(self, subject: str, profile: _Profile) -> None:
        self.blocks[subject].update(subject, profile)


class IdentityReview:
    """Retain private embeddings; require independent, unambiguous face agreement.

    Callers serialize access. Native source mappings are never rewritten: output
    pairs describe reviewed person identities, not detector object equivalence.
    """

    def __init__(self, *, capacity: int = 512) -> None:
        self.capacity = capacity
        self.profiles: OrderedDict[str, _Profile] = OrderedDict()
        self.intervals: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self.aliases: OrderedDict[str, str] = OrderedDict()
        self.pending: set[str] = set()
        self.merges: list[tuple[str, str]] = []
        self.counters: Counter = Counter()
        self.native_subjects: set[str] = set()
        self.sources: OrderedDict = OrderedDict()
        self.conflicts: set[tuple[str, str]] = set()
        self.conflict_capacity_exceeded = False

    def canonical(self, subject: str) -> str:
        return self.aliases.get(subject, subject)

    def observe(
        self, subject: str, timestamp: float, *, sequence: int | None = None,
        bbox: NormalizedBBox | None = None, native: bool = False,
        face: bool = False,
    ) -> None:
        subject = self.canonical(subject)
        first, last = self.intervals.get(subject, (timestamp, timestamp))
        self.intervals[subject] = (min(first, timestamp), max(last, timestamp))
        self.intervals.move_to_end(subject)
        if native:
            self.native_subjects.add(subject)
        if sequence is not None and bbox is not None:
            source = (timestamp, sequence)
            observations = self.sources.setdefault(source, {})
            kind = "face" if face else "native" if native else "body"
            for (other, other_kind), other_box in observations.items():
                if subject == other:
                    continue
                intersection = max(
                    0, min(bbox.x2, other_box.x2) - max(bbox.x1, other_box.x1)
                ) * max(
                    0, min(bbox.y2, other_box.y2) - max(bbox.y1, other_box.y1)
                )
                area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
                other_area = (other_box.x2 - other_box.x1) * (
                    other_box.y2 - other_box.y1
                )
                if kind == other_kind == "native":
                    conflict = True
                elif kind == other_kind == "face":
                    conflict = intersection / (area + other_area - intersection) < 0.8
                elif "face" in {kind, other_kind}:
                    face_area = area if kind == "face" else other_area
                    conflict = intersection / face_area < 0.9
                else:
                    conflict = False
                if conflict:
                    if len(self.conflicts) < 16384:
                        self.conflicts.add(tuple(sorted((subject, other))))
                    else:
                        self.conflict_capacity_exceeded = True
            if len(observations) < 128 or (subject, kind) in observations:
                observations[(subject, kind)] = bbox
            else:
                self.conflict_capacity_exceeded = True
            self.sources.move_to_end(source)
            while len(self.sources) > 2048:
                self.sources.popitem(last=False)
        while len(self.intervals) > 4096:
            evicted, _ = self.intervals.popitem(last=False)
            self.profiles.pop(evicted, None)
            self.pending.discard(evicted)
            self.native_subjects.discard(evicted)

    def observe_face(self, item: ModelObject, timestamp: float, sequence: int) -> None:
        subject = item.attributes.get("subject_id")
        if not isinstance(subject, str) or not subject:
            return
        observed = item.attributes.get("face_observation_state") == "observed"
        self.observe(
            subject, timestamp, sequence=sequence if observed else None,
            bbox=item.bbox if observed else None, face=True,
        )
        person_id = item.attributes.get("person_id")
        identity_source = item.attributes.get("identity_source")
        known = (
            observed and item.attributes.get("status") == "known"
            and isinstance(person_id, str) and bool(person_id.strip())
            and identity_source in {"face_match", "track_history"}
        )
        subject = self.canonical(subject)
        profile = self.profiles.get(subject)
        source = (timestamp, sequence)
        if observed and profile is not None and source > profile.latest_source:
            profile.latest_source = source
            # Only a validated sample below can make this new face usable.
            profile.usable = False
        if known and identity_source == "track_history":
            self._remember_known(subject, person_id)
        if (
            not item.identity_embedding
            or not observed
            or item.attributes.get("identity_source") != "face_match"
            or item.attributes.get("status") not in {"known", "unknown"}
        ):
            return
        sample = np.asarray(item.identity_embedding, dtype=np.float32)
        if sample.ndim != 1 or not 2 <= sample.size <= 4096:
            return
        norm = float(np.linalg.norm(sample))
        if not np.isfinite(sample).all() or not np.isfinite(norm) or norm <= 0:
            return
        subject = self.canonical(subject)
        if not item.identity_quality and not known:
            return
        profile = self.profiles.setdefault(subject, _Profile())
        if known:
            # Gallery classification already passed its own quality, score and
            # ambiguity checks. The stricter re-identification quality flag is
            # for comparing strangers, not for re-proving an enrollment ID.
            self._remember_known(subject, person_id, confirmed=True)
        if not item.identity_quality:
            return
        source = (timestamp, sequence)
        if source in profile.samples:
            return
        profile.samples[source] = sample / norm
        profile.reviewed = False
        if source >= profile.latest_source:
            profile.latest_source, profile.usable = source, True
        while len(profile.samples) > 4:
            profile.samples.popitem(last=False)
        self.profiles.move_to_end(subject)
        self.pending.add(subject)
        self._trim_profiles()

    def _remember_known(
        self, subject: str, person_id: str, *, confirmed: bool = False
    ) -> None:
        profile = self.profiles.setdefault(subject, _Profile())
        if profile.person_id and profile.person_id != person_id:
            profile.conflicting_identity = True
        profile.person_id = person_id
        profile.gallery_confirmed |= confirmed
        profile.reviewed = False
        self.profiles.move_to_end(subject)
        self.pending.add(subject)
        self._trim_profiles()

    def _trim_profiles(self) -> None:
        while len(self.profiles) > self.capacity:
            evicted, _ = self.profiles.popitem(last=False)
            self.pending.discard(evicted)

    def rebind(self, old: str, new: str) -> None:
        """Follow an independently proven same-frame face-to-body association."""
        old, new = self.canonical(old), self.canonical(new)
        if old == new:
            return
        old_interval = self.intervals.pop(old, None)
        if old_interval:
            self.observe(new, old_interval[0])
            self.observe(new, old_interval[1])
        profile = self.profiles.pop(old, None)
        if profile:
            current = self.profiles.setdefault(new, _Profile())
            current.conflicting_identity |= profile.conflicting_identity or bool(
                current.person_id and profile.person_id
                and current.person_id != profile.person_id
            )
            current.person_id = current.person_id or profile.person_id
            current.gallery_confirmed |= profile.gallery_confirmed
            current.reviewed = False
            current.ambiguous |= profile.ambiguous
            if profile.latest_source > current.latest_source:
                current.latest_source, current.usable = (
                    profile.latest_source, profile.usable,
                )
            current.samples = OrderedDict(sorted({
                **profile.samples, **current.samples,
            }.items())[-4:])
            self.pending.add(new)
        self.pending.discard(old)
        if old in self.native_subjects:
            self.native_subjects.discard(old)
            self.native_subjects.add(new)
        self.conflicts = {
            tuple(sorted(new if value == old else value for value in pair))
            for pair in self.conflicts if set(pair) != {old, new}
        }
        for source, observations in self.sources.items():
            self.sources[source] = {
                (new if subject == old else subject, kind): bbox
                for (subject, kind), bbox in observations.items()
            }
        self.aliases = OrderedDict(
            (key, new if value == old else value) for key, value in self.aliases.items()
        )
        self.aliases[old] = new
        while len(self.aliases) > 4096:
            self.aliases.popitem(last=False)

    def forget_person(self, person_id: str) -> None:
        for subject, profile in tuple(self.profiles.items()):
            if profile.person_id == person_id:
                del self.profiles[subject]
                self.pending.discard(subject)

    def _score(
        self, left: str, right: str, *, scores: _ReviewScores | None = None
    ) -> float | None:
        a, b = self.profiles[left], self.profiles[right]
        if (
            len(a.samples) < 2 or len(b.samples) < 2
            or a.conflicting_identity or b.conflicting_identity
            or (a.person_id and b.person_id and a.person_id != b.person_id)
        ):
            return None
        first, second = self.intervals.get(left), self.intervals.get(right)
        if first is None or second is None:
            return None
        # Two observed trajectories that overlap cannot be collapsed by facial
        # resemblance. This also excludes two faces present in the same frame.
        if max(first[0], second[0]) <= min(first[1], second[1]):
            return None
        if scores is not None:
            return scores.score(left, right)
        a_samples, b_samples = list(a.samples.values()), list(b.samples.values())
        dimensions = {sample.size for sample in (*a_samples, *b_samples)}
        if len(dimensions) != 1:
            return None
        similarities = np.stack(a_samples) @ np.stack(b_samples).T
        # At least two distinct source frames on *each* trajectory must agree;
        # one particularly favorable photo cannot establish the whole merge.
        return float(min(
            np.sort(similarities.max(axis=1))[-2],
            np.sort(similarities.max(axis=0))[-2],
        ))

    def review(self, accept_merge: Callable[[str, str], bool] | None = None) -> int:
        dirty = sorted(self.pending, key=lambda sid: self.intervals.get(sid, (0, 0)))
        self.pending.clear()
        if not dirty:
            return 0
        merged = self._review_known(dirty, accept_merge)
        for subject in dirty:
            profile = self.profiles.get(self.canonical(subject))
            if profile is not None:
                profile.ambiguous = False
        scores = _ReviewScores(self.profiles, set(map(self.canonical, dirty)))
        for subject in dirty:
            subject = self.canonical(subject)
            if subject not in self.profiles:
                continue
            self.counters["reviewed_subjects"] += 1
            candidates = []
            for other in self.profiles:
                if subject == other:
                    continue
                score = self._score(subject, other, scores=scores)
                if score is not None:
                    candidates.append((score, other))
            candidates.sort(key=lambda value: (
                -value[0], self.intervals[value[1]][0], value[1],
            ))
            if not candidates or candidates[0][0] < 0.65:
                continue
            best_score, other = candidates[0]
            # Several fragments can all represent one person. Treat them as a
            # group only if close alternatives also agree with the other end of
            # this proposed merge. Check both directions to avoid stealing an
            # ambiguous person's identity through its less ambiguous neighbor.
            ambiguous = False
            for third in self.profiles:
                if third in {subject, other}:
                    continue
                left_score = self._score(subject, third, scores=scores)
                right_score = self._score(other, third, scores=scores)
                if (
                    left_score is not None and best_score - left_score < 0.08
                    and (right_score is None or right_score < 0.65)
                ) or (
                    right_score is not None and best_score - right_score < 0.08
                    and (left_score is None or left_score < 0.65)
                ):
                    ambiguous = True
                    break
            if ambiguous:
                self.profiles[subject].ambiguous = True
                self.profiles[other].ambiguous = True
                self.counters["ambiguous_matches"] += 1
                continue
            new, old = sorted(
                (subject, other), key=lambda sid: (self.intervals[sid][0], sid)
            )
            if accept_merge is not None and not accept_merge(old, new):
                self.profiles[subject].ambiguous = True
                self.profiles[other].ambiguous = True
                self.counters["rejected_merges"] += 1
                continue
            self.rebind(old, new)
            scores.update(new, self.profiles[new])
            self.merges.append((old, new))
            self.counters["merged_subjects"] += 1
            merged += 1
        # Rebinding may mark a profile dirty; this pass already evaluated all
        # source samples. A subsequent real observation will schedule it again.
        self.pending.clear()
        for subject in dirty:
            profile = self.profiles.get(self.canonical(subject))
            if profile is not None:
                profile.reviewed = True
        return merged

    def states(self) -> dict[str, str]:
        """Public review outcomes, without exposing face samples or scores."""
        return {
            subject: (
                "pending" if not profile.reviewed else
                "unresolved" if profile.conflicting_identity or profile.ambiguous else
                "known" if profile.person_id and profile.gallery_confirmed else
                "unresolved" if not profile.usable else
                "unknown" if not profile.person_id and len(profile.samples) >= 2 else
                "unresolved"
            )
            for subject, profile in self.profiles.items()
        }

    def _review_known(self, dirty, accept_merge) -> int:
        """Consolidate confirmed enrollment IDs without requiring two new photos."""
        if self.conflict_capacity_exceeded:
            return 0
        merged = 0

        def rank(subject):
            return (
                subject not in self.native_subjects,
                self.intervals[subject][0], subject,
            )

        for subject in dirty:
            subject = self.canonical(subject)
            profile = self.profiles.get(subject)
            if profile is None or not profile.person_id or profile.conflicting_identity:
                continue
            for other in sorted(tuple(self.profiles), key=rank):
                subject = self.canonical(subject)
                if subject == other or other not in self.profiles:
                    continue
                profile, candidate = self.profiles[subject], self.profiles[other]
                if (
                    candidate.person_id != profile.person_id
                    or candidate.conflicting_identity
                    or not (profile.gallery_confirmed or candidate.gallery_confirmed)
                    or tuple(sorted((subject, other))) in self.conflicts
                ):
                    continue
                new, old = sorted((subject, other), key=rank)
                if accept_merge is not None and not accept_merge(old, new):
                    self.counters["rejected_merges"] += 1
                    continue
                self.rebind(old, new)
                subject = new
                self.merges.append((old, new))
                self.counters["merged_subjects"] += 1
                self.counters["known_subjects_merged"] += 1
                merged += 1
        return merged

    def drain(self) -> tuple[tuple[str, str], ...]:
        result = tuple(self.merges)
        self.merges.clear()
        return result

    def stats(self) -> dict:
        return {
            "pending_subjects": len(self.pending),
            "retained_subjects": len(self.profiles),
            "reviewed_subjects": self.counters["reviewed_subjects"],
            "merged_subjects": self.counters["merged_subjects"],
            "known_subjects_merged": self.counters["known_subjects_merged"],
            "ambiguous_matches": self.counters["ambiguous_matches"],
            "rejected_merges": self.counters["rejected_merges"],
        }
