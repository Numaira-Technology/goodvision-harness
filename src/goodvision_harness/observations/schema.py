"""Model-independent observations; source time is distinct from execution time."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..rich_labeling import RichLabelSnapshot


def text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be nonempty text of at most 256 characters")


def seconds(value: float, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class Observation:
    """One source-anchored perception result, including empty successful frames.

    Track IDs are local to the model/session until PersonRegistry canonicalizes
    them. Inference duration never supplies source ordering or evidence votes.
    """

    source_id: str
    snapshot: RichLabelSnapshot
    model: str
    model_version: str | None = None
    inference_seconds: float | None = None

    def __post_init__(self) -> None:
        text(self.source_id, "source_id")
        text(self.model, "model")
        if not isinstance(self.snapshot, RichLabelSnapshot):
            raise TypeError("snapshot must be a RichLabelSnapshot")
        if self.model_version is not None:
            text(self.model_version, "model_version")
        if self.inference_seconds is not None:
            seconds(self.inference_seconds, "inference_seconds")

    def as_dict(self) -> dict:
        return {
            "schema_version": "1",
            "kind": "perception",
            "source_id": self.source_id,
            "model": self.model,
            "model_version": self.model_version,
            "inference_seconds": self.inference_seconds,
            **self.snapshot.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class EventObservation:
    """A judgment, not an alert. Unknown and failure never mean clear.

    Actors must already be canonical and observed at this exact source frame.
    Revisions use the same evidence ID and original source coordinates.
    """

    source_id: str
    stream_session_id: str
    frame_sequence: int
    timestamp_seconds: float
    event_label: str
    subject_ids: tuple[str, ...]
    detected: bool | None
    evidence_id: str
    model: str
    model_version: str | None = None
    confidence: float | None = None
    inference_seconds: float | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "stream_session_id",
            "event_label",
            "evidence_id",
            "model",
        ):
            text(getattr(self, name), name)
        for name in ("frame_sequence", "revision"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        seconds(self.timestamp_seconds, "timestamp_seconds")
        if (
            not isinstance(self.subject_ids, tuple)
            or len(self.subject_ids) > 128
            or len(set(self.subject_ids)) != len(self.subject_ids)
        ):
            raise ValueError("subject_ids must be a bounded unique tuple")
        for subject in self.subject_ids:
            text(subject, "subject_id")
        if not self.subject_ids and not (self.revision > 0 and self.detected is None):
            raise ValueError("only an explicit revision may withdraw participants")
        if self.detected is not None and type(self.detected) is not bool:
            raise ValueError("detected must be true, false or null")
        if self.confidence is not None:
            seconds(self.confidence, "confidence")
            if self.confidence > 1:
                raise ValueError("confidence must be within 0..1")
        if self.model_version is not None:
            text(self.model_version, "model_version")
        if self.inference_seconds is not None:
            seconds(self.inference_seconds, "inference_seconds")

    def as_dict(self) -> dict:
        return {
            "type": "event_observation",
            **asdict(self),
            "subject_ids": list(self.subject_ids),
        }
