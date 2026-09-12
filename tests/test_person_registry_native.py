"""Native video IDs remain authoritative without implying face identity."""

from dataclasses import replace

import pytest

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.grounding import ModelActor, ModelObject, ObservationBundle
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot

BACKEND = "sam3.1_video"
LEFT = NormalizedBBox(0.1, 0.05, 0.45, 0.95)
RIGHT = NormalizedBBox(0.55, 0.05, 0.9, 0.95)
MIDDLE = NormalizedBBox(0.325, 0.05, 0.675, 0.95)
FACE = NormalizedBBox(0.2, 0.1, 0.3, 0.2)


def snapshot(at, objects=(("1", LEFT),), *, epoch="epoch-a", state="observed"):
    return RichLabelSnapshot(
        "session",
        int(at * 10),
        at,
        tuple(
            RichLabelObject(
                f"wire-{index}",
                "person",
                box,
                0.9,
                {
                    "native_track": {
                        "backend": BACKEND,
                        "epoch": epoch,
                        "object_id": object_id,
                    },
                    "mask_ref": "source-mask",
                },
                observation_state=state,
                last_observed_timestamp_seconds=at if state == "observed" else 0,
                last_observed_frame_sequence=int(at * 10) if state == "observed" else 0,
            )
            for index, (object_id, box) in enumerate(objects)
        ),
    )


def subjects(registry):
    return {
        item.attributes["native_track"]["object_id"]: item.track_id
        for item in registry.history()[-1].objects
        if "native_track" in item.attributes
    }


def face(status="known", person_id="alice"):
    attrs = {"status": status}
    if status == "known":
        attrs.update(person_id=person_id, name=person_id)
    return ModelObject(
        "known_person" if status == "known" else "unknown_person", FACE, 0.9, attrs
    )


def resolve(registry, at, status="known", person_id="alice"):
    return registry.resolve_faces(
        [face(status, person_id)],
        timestamp=at,
        frame_sequence=int(at * 10),
        people={"alice": "Alice", "bob": "Bob"},
    )[0]


def analysis(at, boxes=(LEFT,), *, memories=()):
    actors = tuple(
        ModelActor(
            str(i), box, 0.9, matched_memory_id=memories[i] if memories else None
        )
        for i, box in enumerate(boxes)
    )
    return AnalysisResult(
        str(at),
        "camera",
        "session",
        int(at * 10) + 1,
        "now",
        FrameReference(int(at * 10), at, "now", 640, 480),
        (
            EventJudgment(
                "smoking", True, 0.9, "test", tuple(a.actor_id for a in actors)
            ),
        ),
        tuple(GroundedObject(a.actor_id, "person", a.bbox) for a in actors),
        "test",
        0,
    ), ObservationBundle(actors, tuple((a.actor_id,) for a in actors))


def test_native_ids_survive_crossing_and_wire_id_reordering():
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(0, (("1", LEFT), ("2", RIGHT))))
    initial = subjects(registry)
    for at, boxes in (
        (1, (("1", MIDDLE), ("2", MIDDLE))),
        (2, (("2", LEFT), ("1", RIGHT))),
    ):
        registry.observe_snapshot(snapshot(at, boxes))
        assert subjects(registry) == initial
    assert len(set(initial.values())) == 2


def test_long_occlusion_preserves_native_id_and_identity_without_fake_observations():
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(0))
    initial = subjects(registry)["1"]
    assert resolve(registry, 0).attributes["subject_id"] == initial
    registry.observe_snapshot(snapshot(10, (), state="observed"))
    assert registry.snapshot_at(10, 100).objects == ()
    assert registry._subjects[initial].seen_at == 0
    registry.observe_snapshot(snapshot(11, state="predicted"))
    assert registry.snapshot_at(11, 110).objects == ()
    assert registry._subjects[initial].seen_at == 0
    registry.observe_snapshot(snapshot(12))
    assert subjects(registry)["1"] == initial
    observed = resolve(registry, 12, "unknown")
    assert observed.attributes["person_id"] == "alice"
    assert observed.attributes["subject_id"] == initial


def test_epoch_switch_same_native_integer_does_not_inherit_face_identity():
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(0))
    old = subjects(registry)["1"]
    resolve(registry, 0)
    assert registry.begin_native_epoch(BACKEND, "epoch-b")
    registry.observe_snapshot(snapshot(1, epoch="epoch-b"))
    new = subjects(registry)["1"]
    assert new != old
    assert old not in registry._subjects
    assert "person_id" not in resolve(registry, 1, "unknown").attributes
    assert not registry._subjects[new].identity
    assert registry.history()[0].objects[0].track_id == old


def test_epoch_activation_is_explicit_and_late_old_epoch_cannot_advance_watermark():
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(0))
    registry.observe_snapshot(snapshot(1, epoch="epoch-b"))
    assert registry.history()[-1].timestamp_seconds == 0
    assert registry.begin_native_epoch(BACKEND, "epoch-b")
    registry.observe_snapshot(snapshot(1, epoch="epoch-b"))
    current = subjects(registry)
    registry.observe_snapshot(snapshot(50, epoch="epoch-a"))
    assert registry.history()[-1].timestamp_seconds == 1
    registry.observe_snapshot(snapshot(2, epoch="epoch-b"))
    assert subjects(registry) == current
    assert registry.native_tracking_stats()["rejections"]["inactive_epoch"] == 2


def test_empty_batch_still_activates_epoch_and_records_exact_source_frame():
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(0))
    old = subjects(registry)["1"]
    registry.begin_native_epoch(BACKEND, "empty")
    registry.observe_snapshot(snapshot(1, (), epoch="empty"))
    current = registry.snapshot_at(1, 10)
    assert (current.timestamp_seconds, current.frame_sequence, current.objects) == (
        1,
        10,
        (),
    )
    assert old not in registry._subjects
    assert registry.native_tracking_stats()["tracks"] == 0
    registry.observe_snapshot(snapshot(2, epoch="epoch-a"))
    assert registry.history()[-1] == current


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {},
        {"backend": BACKEND},
        {"backend": "other", "epoch": "e", "object_id": "1"},
        {"backend": BACKEND, "epoch": "", "object_id": "1"},
        {"backend": BACKEND, "epoch": "e", "object_id": 1},
        {"backend": BACKEND, "epoch": "e", "object_id": " 1"},
        {"backend": BACKEND, "epoch": "e", "object_id": "1", "extra": True},
        {"backend": BACKEND, "epoch": "e", "object_id": "1" * 257},
    ],
)
def test_malformed_native_metadata_never_falls_back_to_geometric_identity(metadata):
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(
        RichLabelSnapshot("session", 0, 0, (RichLabelObject("old", "person", LEFT),))
    )
    previous = registry.history()
    good = snapshot(1)
    bad = replace(
        good, objects=(replace(good.objects[0], attributes={"native_track": metadata}),)
    )
    registry.observe_snapshot(bad)
    assert registry.history() == previous
    assert registry.native_tracking_stats()["tracks"] == 0
    registry.observe_snapshot(good)
    assert subjects(registry)["1"] != previous[0].objects[0].track_id


def test_duplicate_native_objects_and_mixed_epochs_are_rejected_atomically():
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(0, (("1", LEFT), ("1", RIGHT))))
    assert not registry.history()
    original = snapshot(0, (("1", LEFT), ("2", RIGHT)))
    other = snapshot(0, (("2", RIGHT),), epoch="different").objects[0]
    mixed = replace(
        original, objects=(original.objects[0], replace(other, track_id="other"))
    )
    registry.observe_snapshot(mixed)
    assert not registry.history()
    assert not registry._subjects


def test_native_capacity_rejects_newcomers_without_evicting_or_reidentifying_people():
    registry = PersonRegistry("session", "camera", max_native_tracks=2)
    registry.observe_snapshot(snapshot(0, (("1", LEFT), ("2", RIGHT))))
    initial = subjects(registry)
    registry.observe_snapshot(snapshot(1, (("3", MIDDLE), ("1", RIGHT))))
    assert subjects(registry) == {"1": initial["1"]}
    assert registry.native_tracking_stats()["tracks"] == 2
    assert registry.native_tracking_stats()["rejections"]["capacity"] == 1
    registry.observe_snapshot(snapshot(10, (("2", LEFT),)))
    assert subjects(registry)["2"] == initial["2"]
    registry.begin_native_epoch(BACKEND, "epoch-b")
    registry.observe_snapshot(snapshot(11, (("3", MIDDLE),), epoch="epoch-b"))
    assert registry.native_tracking_stats()["tracks"] == 1


def test_predictions_cannot_create_native_people_or_consume_identity_capacity():
    registry = PersonRegistry("session", "camera", max_native_tracks=1)
    registry.observe_snapshot(snapshot(0, state="predicted"))
    assert not registry._subjects
    assert registry.native_tracking_stats()["tracks"] == 0
    registry.observe_snapshot(snapshot(1, (("2", RIGHT),)))
    assert set(subjects(registry)) == {"2"}


def test_canonical_snapshot_preserves_real_source_and_native_provenance():
    registry = PersonRegistry("session", "camera")
    raw = snapshot(2)
    registry.observe_snapshot(raw)
    canonical = registry.snapshot_at(2, 20).objects[0]
    assert canonical.track_id != raw.objects[0].track_id
    assert (
        canonical.attributes["native_track"]
        == raw.objects[0].attributes["native_track"]
    )
    assert canonical.attributes["mask_ref"] == "source-mask"
    assert (
        canonical.observation_state,
        canonical.last_observed_timestamp_seconds,
        canonical.last_observed_frame_sequence,
    ) == ("observed", 2, 20)
    registry.observe_snapshot(
        replace(raw, objects=(replace(raw.objects[0], bbox=RIGHT),))
    )
    assert registry.history()[-1].objects[0] == canonical


def test_native_observation_with_mismatched_attribute_source_is_not_accepted():
    registry = PersonRegistry("session", "camera")
    obj = RichLabelObject(
        "wire",
        "person",
        LEFT,
        attributes={
            "native_track": {"backend": BACKEND, "epoch": "e", "object_id": "1"},
            "observation_state": "observed",
            "last_observed_timestamp_seconds": 0,
            "last_observed_frame_sequence": 999,
        },
    )
    registry.observe_snapshot(RichLabelSnapshot("session", 0, 0, (obj,)))
    assert not registry._subjects
    assert registry.history()[-1].objects == ()


def test_legacy_actors_can_reference_native_source_but_cannot_rewrite_its_geometry():
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(0, (("1", LEFT), ("2", RIGHT))))
    initial = subjects(registry)
    canonical = registry.history()[-1]
    prepared = registry.prepare_analysis(*analysis(0, (LEFT, RIGHT)))
    assert [obj.attributes["subject_id"] for obj in prepared.objects] == [
        initial["1"],
        initial["2"],
    ]
    assert registry.history()[-1] == canonical
    registry.prepare_analysis(*analysis(1, (MIDDLE, MIDDLE)))
    assert set(initial.values()) <= registry._subjects.keys()
    assert registry._subjects[initial["1"]].bbox == LEFT
    assert registry._subjects[initial["2"]].bbox == RIGHT
    registry.observe_snapshot(snapshot(2, (("1", RIGHT), ("2", LEFT))))
    assert subjects(registry) == initial


def test_native_subject_ids_are_not_reset_by_vlm_memory_matches():
    registry = PersonRegistry("session", "camera")
    initial = None
    for at in range(5):
        registry.observe_snapshot(snapshot(at))
        initial = initial or subjects(registry)["1"]
        prepared = registry.prepare_analysis(
            *analysis(at, memories=("alice" if at == 0 else "bob",))
        )
        assert prepared.objects[0].attributes["subject_id"] == initial
        assert not registry._subjects[initial].identity


def test_native_face_identity_deletion_and_reenrollment_preserve_track_id():
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(0))
    initial = subjects(registry)["1"]
    resolve(registry, 0)
    registry.forget_person("alice")
    registry.observe_snapshot(snapshot(1))
    stale = resolve(registry, 1)
    assert stale.attributes["status"] == "unusable"
    assert not registry._subjects[initial].identity
    registry.allow_person("alice")
    registry.observe_snapshot(snapshot(2))
    accepted = resolve(registry, 2)
    assert accepted.attributes["person_id"] == "alice"
    assert accepted.attributes["subject_id"] == initial


def test_native_face_identity_correction_changes_identity_not_video_subject_id():
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(0))
    initial = subjects(registry)["1"]
    resolve(registry, 0)
    for at in (1, 2, 3):
        registry.observe_snapshot(snapshot(at))
        corrected = resolve(registry, at, person_id="bob")
        assert corrected.attributes["subject_id"] == initial
    assert corrected.attributes["person_id"] == "bob"
    registry.observe_snapshot(snapshot(4))
    assert subjects(registry)["1"] == initial


def test_native_session_isolation_and_older_source_callbacks():
    first = PersonRegistry("session", "camera")
    second = PersonRegistry("session", "other-camera")
    first.observe_snapshot(snapshot(2))
    second.observe_snapshot(snapshot(2))
    assert subjects(first) != subjects(second)
    before = first.history()
    first.observe_snapshot(replace(snapshot(3), stream_session_id="old-session"))
    first.observe_snapshot(snapshot(1))
    assert first.history() == before
    assert next(iter(first._native_tracks))[0:2] == ("session", "camera")


def test_native_timestamp_cannot_retime_or_rewind_the_source_frame_sequence():
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(2))
    previous = registry.history()
    retimed = snapshot(3)
    obj = replace(retimed.objects[0], last_observed_frame_sequence=20)
    registry.observe_snapshot(replace(retimed, frame_sequence=20, objects=(obj,)))
    assert registry.history() == previous
    registry.observe_snapshot(RichLabelSnapshot("session", 19, 4, ()))
    assert registry.history() == previous
    registry.observe_snapshot(snapshot(3))
    assert registry.history()[-1].frame_sequence == 30


@pytest.mark.parametrize("capacity", [0, 4097, True, 1.5])
def test_invalid_native_capacity_is_rejected(capacity):
    with pytest.raises(ValueError):
        PersonRegistry(max_native_tracks=capacity)


@pytest.mark.parametrize(
    "backend,epoch",
    [("other", "e"), (BACKEND, ""), (BACKEND, None), (BACKEND, "x" * 257)],
)
def test_invalid_native_epoch_cannot_reset_active_identity(backend, epoch):
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(snapshot(0))
    initial = dict(registry._native_tracks)
    assert not registry.begin_native_epoch(backend, epoch)
    assert registry._native_tracks == initial
