"""The new observation boundary uses the existing production event ledger."""

import io
import json
from dataclasses import replace

import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.evaluation.replay import replay
from goodvision_harness.events.state_machine import SceneState
from goodvision_harness.observations.schema import EventObservation
from goodvision_harness.perception.settings import SupervisionSettings


def evidence(sequence, detected, **kwargs):
    return EventObservation(
        "camera",
        "session",
        sequence,
        float(sequence),
        "smoking",
        ("a",),
        detected,
        f"e-{sequence}",
        "test-model",
        **kwargs,
    )


def test_late_completion_does_not_reopen_a_cleared_event():
    state = SceneState("session", "camera")
    for sequence in range(4):
        state.observe_person(
            "a",
            timestamp_seconds=float(sequence),
            frame_sequence=sequence,
            bbox=NormalizedBBox(0.1, 0.1, 0.4, 0.7),
        )
    for sequence in (1, 2, 3, 0):
        assert state.observe_observation(evidence(sequence, sequence < 2))
    (event,) = state.snapshot()["events"]
    assert event["phase"] == "ended"
    assert event["ended_since_timestamp_seconds"] == 3
    assert not state.observe_observation(evidence(0, True))
    assert not state.observe_observation(
        replace(evidence(3, True), source_id="another")
    )


def test_explicit_revision_withdraws_evidence_unknown_does_not_clear():
    state = SceneState("session", "camera")
    for sequence in range(3):
        state.observe_person(
            "a",
            timestamp_seconds=float(sequence),
            frame_sequence=sequence,
            bbox=NormalizedBBox(0.1, 0.1, 0.4, 0.7),
        )
        assert state.observe_observation(
            evidence(sequence, True if sequence < 2 else None), revisable=True
        )
    assert state.snapshot()["events"][0]["phase"] == "confirmed"
    assert state.observe_observation(
        replace(evidence(0, None), revision=1, subject_ids=()), revisable=True
    )
    assert state.snapshot()["events"][0]["phase"] != "confirmed"


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": "false"},
        {"confidence_threshold": float("nan")},
        {"nms_threshold": 1.1},
        {"max_tracks": True},
        {"smoothing_length": 0},
        {"unknown": 1},
        {"zones": [{"name": "bad", "points": [[0, 0], [1, 1]]}]},
    ],
)
def test_invalid_settings_are_rejected_at_load(settings):
    with pytest.raises(ValueError):
        SupervisionSettings.from_mapping(settings)


def test_unified_event_observation_replays_late_evidence_and_revision():
    rows = [{"type": "session", "source_id": "camera", "stream_session_id": "session"}]
    for sequence in range(4):
        rows.append(
            {
                "type": "person",
                "subject_id": "a",
                "timestamp_seconds": sequence,
                "frame_sequence": sequence,
                "bbox": [0.1, 0.1, 0.4, 0.7],
            }
        )
    rows.extend(evidence(sequence, sequence < 2).as_dict() for sequence in (1, 2, 3, 0))
    source = io.BytesIO("\n".join(json.dumps(row) for row in rows).encode())
    snapshots = list(replay(source))
    assert snapshots[-1]["scene"]["events"][0]["phase"] == "ended"
    assert all(row["accepted"] for row in snapshots)
