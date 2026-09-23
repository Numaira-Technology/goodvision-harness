"""Normalize existing harness results without inventing observation evidence."""

from ..rich_labeling import RichLabelObject, RichLabelSnapshot
from .schema import Observation


def is_observed(item: RichLabelObject, snapshot: RichLabelSnapshot) -> bool:
    state = item.observation_state or item.attributes.get("observation_state")
    return state in (None, "observed") and (
        item.last_observed_frame_sequence in (None, snapshot.frame_sequence)
        and item.last_observed_timestamp_seconds in (None, snapshot.timestamp_seconds)
    )


def from_snapshot(
    snapshot: RichLabelSnapshot,
    *,
    source_id: str,
    model: str,
    model_version: str | None = None,
    inference_seconds: float | None = None,
) -> Observation:
    return Observation(source_id, snapshot, model, model_version, inference_seconds)
