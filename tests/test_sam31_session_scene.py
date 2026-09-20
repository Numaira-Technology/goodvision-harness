"""Persistent native outputs must reuse scene capacity and retain source evidence."""
import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.scene_state import SceneState
from goodvision_harness.video_tracking import BACKEND


def observation(sequence, *, native_id="7", epoch="persistent"):
    return RichLabelSnapshot(
        "s", sequence, sequence / 4,
        (RichLabelObject(
            "native-" + native_id, "person", NormalizedBBox(.1, .1, .4, .9),
            attributes={"native_track": {
                "backend": BACKEND, "epoch": epoch, "object_id": native_id,
            }},
            observation_state="observed",
            last_observed_timestamp_seconds=sequence / 4,
            last_observed_frame_sequence=sequence,
        ),),
    )


def test_same_native_person_reuses_full_scene_capacity_before_mapping():
    runtime = SceneRuntime(
        "s", "camera", mode="active", tracking_mode="batch",
        native_memory_scope="session", event_labels=(),
    )
    runtime.scene = SceneState("s", "camera", max_entities=1, person_history_seconds=12)
    registry = PersonRegistry("s", "camera")
    mapped_calls = []

    def canonicalize(frames):
        mapped_calls.append(tuple(frame.frame_sequence for frame in frames))
        registry.begin_native_epoch(BACKEND, "persistent")
        mapped = []
        for frame in frames:
            registry.observe_snapshot(frame)
            mapped.append(
                registry.snapshot_at(frame.timestamp_seconds, frame.frame_sequence)
            )
        return tuple(mapped)

    for sequence in range(5):
        runtime.observe_frame(b"source", sequence / 4, sequence, force=True)
    for start in (0, 2):
        frames = (observation(start), observation(start + 1))
        assert runtime.commit_native_tracking_batch(
            frames, completed_frame_sequence=start + 1,
            completed_timestamp_seconds=(start + 1) / 4, canonicalize=canonicalize,
        )
    state = runtime.snapshot()
    assert state["memory_scope"] == "session"
    assert state["capture_timestamp_seconds"] == 1
    assert state["tracking_completed_timestamp_seconds"] == .75
    assert len(state["scene"]["entities"]) == 1
    assert state["scene"]["counters"]["entities_created"] == 1
    assert mapped_calls == [(0, 1), (2, 3)]
    # A different native person cannot take an occupied, still-observed slot.
    assert not runtime.commit_native_tracking_batch(
        (observation(4, native_id="8"),), completed_frame_sequence=4,
        completed_timestamp_seconds=1, canonicalize=canonicalize,
    )
    assert mapped_calls == [(0, 1), (2, 3)]
    assert runtime.snapshot()["tracking_last_rejection"] == "subject_capacity"


def test_native_capacity_does_not_merge_equal_ids_from_different_epochs():
    state = SceneState("s", "camera", max_entities=1, person_history_seconds=12)
    native = {"backend": BACKEND, "epoch": "old", "object_id": "7"}
    state.observe_person(
        "known-subject", timestamp_seconds=0, frame_sequence=0,
        bbox=NormalizedBBox(.1, .1, .4, .9), native_track=native,
    )
    assert state.can_reserve_native_person_slots({(BACKEND, "old", "7")})
    assert not state.can_reserve_native_person_slots({(BACKEND, "new", "7")})


@pytest.mark.parametrize("scope", ["unlimited", "", "camera"])
def test_native_memory_scope_must_be_explicit_supported_value(scope):
    with pytest.raises(ValueError, match="native memory scope"):
        SceneRuntime("s", "camera", tracking_mode="batch", native_memory_scope=scope)
