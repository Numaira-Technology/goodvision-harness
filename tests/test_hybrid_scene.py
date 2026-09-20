"""Tentative A/B observations remain displayable but are not identity/event proof."""

from dataclasses import replace

import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.grounding import ModelObject
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.scene_state import SceneState

BODY = NormalizedBBox(0.1, 0.05, 0.5, 0.95)
FACE = NormalizedBBox(0.2, 0.1, 0.3, 0.25)


def tracked(sequence, confirmation="provisional", backend="person_hybrid"):
    return RichLabelSnapshot(
        "camera",
        sequence,
        sequence * 0.5,
        (
            RichLabelObject(
                "worker-0",
                "person",
                BODY,
                0.9,
                {
                    "native_track": {
                        "backend": backend,
                        "epoch": "epoch",
                        "object_id": "0",
                    },
                    "tracking_confirmation": confirmation,
                },
                observation_state="observed",
                last_observed_frame_sequence=sequence,
                last_observed_timestamp_seconds=sequence * 0.5,
            ),
        ),
    )


def face(registry, sequence):
    return registry.resolve_faces(
        (ModelObject("unknown_person", FACE, 0.9, {"status": "unknown"}),),
        timestamp=sequence * 0.5,
        frame_sequence=sequence,
        people={},
    )[0].attributes["subject_id"]


@pytest.mark.parametrize("backend", ["person_hybrid", "sam3.1_video", "goodvision_contour"])
@pytest.mark.parametrize("face_first", [True, False])
def test_provisional_person_cannot_merge_with_face_then_confirmed_same_id_can(
    backend, face_first
):
    registry = PersonRegistry("camera", "source")
    if face_first:
        face_id = face(registry, 0)
    registry.observe_snapshot(tracked(0, backend=backend))
    body_id = registry.snapshot_at(0.0, 0).objects[0].track_id
    if not face_first:
        face_id = face(registry, 0)
    assert body_id != face_id
    assert registry.drain_subject_merges() == ()
    registry.observe_snapshot(tracked(1, "confirmed", backend))
    assert registry.snapshot_at(0.5, 1).objects[0].track_id == body_id
    assert face(registry, 1) == body_id
    assert registry.canonical_subject_id(face_id) == body_id


def test_backend_change_requires_explicit_epoch_activation():
    registry = PersonRegistry("camera", "source")
    registry.observe_snapshot(tracked(0, "confirmed", "sam3.1_video"))
    registry.observe_snapshot(tracked(1, "confirmed", "person_hybrid"))
    assert registry.snapshot_at(0.5, 1).frame_sequence == 0
    assert registry.begin_native_epoch("person_hybrid", "hybrid-epoch")
    source = tracked(2, "confirmed", "person_hybrid")
    obj = source.objects[0]
    source = replace(
        source,
        objects=(
            replace(
                obj,
                attributes={
                    **obj.attributes,
                    "native_track": {
                        **obj.attributes["native_track"],
                        "epoch": "hybrid-epoch",
                    },
                },
            ),
        ),
    )
    registry.observe_snapshot(source)
    assert registry.snapshot_at(1.0, 2).frame_sequence == 2
    assert not registry.begin_native_epoch("fake-person-backend", "x")


def test_scene_preserves_provisional_history_and_excludes_it_from_events_and_pins():
    scene = SceneState("camera", "source")
    assert scene.observe_person(
        "p",
        timestamp_seconds=0.0,
        frame_sequence=0,
        bbox=BODY,
        tracking_confirmation="provisional",
    )
    entity = scene.snapshot()["entities"][0]
    assert entity["observation_state"] == "observed"
    assert entity["tracking_confirmation"] == "provisional"
    assert entity["history"][0]["tracking_confirmation"] == "provisional"
    assert (
        scene.snapshot_at(0.0)["entities"][0]["tracking_confirmation"] == "provisional"
    )
    assert not scene.pin_source_observations("job", (("p", 0.0, 0, BODY),))
    assert not scene.observe_event(
        "fall",
        ("p",),
        timestamp_seconds=0.0,
        frame_sequence=0,
        detected=True,
        evidence_id="first",
    )
    assert scene.observe_person(
        "p",
        timestamp_seconds=0.5,
        frame_sequence=1,
        bbox=BODY,
        tracking_confirmation="confirmed",
    )
    assert scene.pin_source_observations("job", (("p", 0.5, 1, BODY),))
    assert scene.observe_event(
        "fall",
        ("p",),
        timestamp_seconds=0.5,
        frame_sequence=1,
        detected=True,
        evidence_id="second",
    )
    assert scene.snapshot()["entities"][0]["tracking_confirmation"] == "confirmed"
    # Confirmation at frame 1 does not rewrite frame 0's proof status.
    assert (
        scene.snapshot_at(0.0)["entities"][0]["tracking_confirmation"] == "provisional"
    )
    assert not scene._has_source_observation("p", 0.0, 0)


def test_same_source_downgrade_cannot_use_previous_confirmed_pin():
    scene = SceneState("camera", "source")
    scene.observe_person(
        "p",
        timestamp_seconds=0.0,
        frame_sequence=0,
        bbox=BODY,
        tracking_confirmation="confirmed",
    )
    assert scene.pin_source_observations("job", (("p", 0.0, 0, BODY),))
    scene.observe_person(
        "p",
        timestamp_seconds=0.0,
        frame_sequence=0,
        bbox=BODY,
        tracking_confirmation="provisional",
        allow_historical=True,
    )
    assert not scene._has_source_observation("p", 0.0, 0)


def test_same_frame_direct_update_keeps_confirmation_consistent_with_history():
    scene = SceneState("camera", "source")
    scene.observe_person(
        "p",
        timestamp_seconds=0.0,
        frame_sequence=0,
        bbox=BODY,
        tracking_confirmation="confirmed",
    )
    scene.observe_person(
        "p",
        timestamp_seconds=0.0,
        frame_sequence=0,
        bbox=BODY,
        tracking_confirmation="provisional",
    )
    assert scene.snapshot()["entities"][0]["tracking_confirmation"] == "provisional"
    assert (
        scene.snapshot_at(0.0)["entities"][0]["tracking_confirmation"] == "provisional"
    )
    assert not scene._has_source_observation("p", 0.0, 0)


def test_historical_insert_does_not_change_current_confirmation():
    scene = SceneState("camera", "source")
    scene.observe_person(
        "p",
        timestamp_seconds=1.0,
        frame_sequence=2,
        bbox=BODY,
        tracking_confirmation="confirmed",
    )
    scene.observe_person(
        "p",
        timestamp_seconds=0.0,
        frame_sequence=0,
        bbox=BODY,
        tracking_confirmation="provisional",
        allow_historical=True,
    )
    assert scene.snapshot()["entities"][0]["tracking_confirmation"] == "confirmed"
    assert (
        scene.snapshot_at(0.0)["entities"][0]["tracking_confirmation"] == "provisional"
    )


@pytest.mark.parametrize("backend", ["person_hybrid", "sam3.1_video"])
def test_runtime_displays_provisional_boxes_but_schedules_only_confirmed_sources(
    backend,
):
    runtime = SceneRuntime(
        "camera", "source", tracking_mode="batch", mode="active", event_labels=("fall",)
    )
    sources = tuple(tracked(i, backend=backend) for i in range(4))
    for snapshot in sources:
        runtime.observe_frame(
            b"retained-source", snapshot.timestamp_seconds, snapshot.frame_sequence
        )
    assert runtime.commit_native_tracking_batch(
        sources,
        completed_timestamp_seconds=1.5,
        completed_frame_sequence=3,
        canonicalize=lambda frames: frames,
    )
    assert runtime._jobs == {}
    source, view = runtime.tracked_display_frame(before_timestamp_seconds=1.5)
    assert source.frame_sequence == 3
    assert view["entities"][0]["bbox"] == BODY.as_dict()
    assert view["entities"][0]["tracking_confirmation"] == "provisional"
    confirmed = tuple(tracked(i, "confirmed", backend) for i in range(4, 8))
    for snapshot in confirmed:
        runtime.observe_frame(
            b"retained-source", snapshot.timestamp_seconds, snapshot.frame_sequence
        )
    assert runtime.commit_native_tracking_batch(
        confirmed,
        completed_timestamp_seconds=3.5,
        completed_frame_sequence=7,
        canonicalize=lambda frames: frames,
    )
    assert runtime._jobs
    assert all(
        sequence >= 4
        for job in runtime._jobs.values()
        for sequence, _subject, _bbox in job.subject_boxes
    )


def test_canonical_mapping_cannot_promote_provisional_observation():
    runtime = SceneRuntime(
        "camera", "source", tracking_mode="batch", mode="active", event_labels=()
    )
    source = tracked(0)
    runtime.observe_frame(b"retained-source", 0.0, 0)

    def corrupt(frames):
        obj = frames[0].objects[0]
        return (
            replace(
                frames[0],
                objects=(
                    replace(
                        obj,
                        attributes={
                            **obj.attributes,
                            "tracking_confirmation": "confirmed",
                        },
                    ),
                ),
            ),
        )

    with pytest.raises(ValueError, match="changed a source observation"):
        runtime.commit_native_tracking_batch(
            (source,),
            completed_timestamp_seconds=0.0,
            completed_frame_sequence=0,
            canonicalize=corrupt,
        )
