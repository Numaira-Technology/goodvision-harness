from __future__ import annotations

from dataclasses import replace

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

BODY = NormalizedBBox(0.1, 0.05, 0.45, 0.95)
FACE = NormalizedBBox(0.2, 0.1, 0.3, 0.2)
ITEM = NormalizedBBox(0.25, 0.21, 0.27, 0.23)
OTHER = NormalizedBBox(0.6, 0.05, 0.95, 0.95)


def _result(
    timestamp: float,
    *,
    detected: bool = True,
    label: str = "cigarette",
    box: NormalizedBBox = ITEM,
) -> AnalysisResult:
    return AnalysisResult(
        "message-" + str(timestamp),
        "camera",
        "session",
        int(timestamp * 10) + 1,
        "2026-01-01T00:00:00Z",
        FrameReference(int(timestamp * 10), timestamp, "now", 640, 480),
        (EventJudgment("smoking", detected, 0.9, "Test", ("o1",) if detected else ()),),
        (GroundedObject("o1", label, box),) if detected else (),
        "test",
        0.1,
    )


def _bundle(box: NormalizedBBox = BODY, actor_id: str = "a1") -> ObservationBundle:
    return ObservationBundle((ModelActor(actor_id, box, 0.9),), ((actor_id,),))


def _snapshot(
    timestamp: float,
    box: NormalizedBBox = BODY,
    *,
    track_id: str = "P0001",
    predicted: bool = False,
) -> RichLabelSnapshot:
    return RichLabelSnapshot(
        "session",
        int(timestamp * 10),
        timestamp,
        (
            RichLabelObject(
                track_id,
                "person",
                box,
                0.9,
                observation_state="predicted" if predicted else "observed",
                last_observed_timestamp_seconds=0.0 if predicted else timestamp,
                last_observed_frame_sequence=0 if predicted else int(timestamp * 10),
            ),
        ),
    )


def _face(status: str, *, person_id: str = "alice") -> ModelObject:
    attributes = {"status": status}
    if status == "known":
        attributes.update(person_id=person_id, name=person_id)
    return ModelObject(
        "known_person" if status == "known" else "unknown_person", FACE, 0.9, attributes
    )


def test_negative_frames_preserve_subject_without_holding_event_positive() -> None:
    registry = PersonRegistry("session", "camera")
    first = registry.prepare_analysis(_result(0), _bundle())
    subject_id = first.objects[0].attributes["subject_id"]
    negative = registry.prepare_analysis(
        _result(1, detected=False), _bundle(actor_id="new-id")
    )
    assert negative.judgments[0].detected is False
    assert negative.objects == ()
    again = registry.prepare_analysis(_result(2), _bundle(actor_id="third-id"))
    assert again.objects[0].attributes["subject_id"] == subject_id


def test_long_absence_allocates_new_subject_and_session_namespaces_do_not_leak() -> (
    None
):
    registry = PersonRegistry("session", "camera")
    first = registry.prepare_analysis(_result(0), _bundle())
    later = registry.prepare_analysis(_result(3.1), _bundle())
    other = PersonRegistry("session", "camera").prepare_analysis(_result(0), _bundle())
    assert (
        len(
            {
                first.objects[0].attributes["subject_id"],
                later.objects[0].attributes["subject_id"],
                other.objects[0].attributes["subject_id"],
            }
        )
        == 3
    )
    wrong = replace(_result(4), stream_session_id="other-session")
    assert registry.prepare_analysis(wrong, _bundle()) is wrong


def test_gpu_upgrade_keeps_subject_and_predictions_do_not_refresh_it() -> None:
    registry = PersonRegistry("session", "camera")
    first = registry.prepare_analysis(_result(0), _bundle())
    subject_id = first.objects[0].attributes["subject_id"]
    registry.observe_snapshot(_snapshot(0.2))
    assert registry.history()[-1].objects[0].track_id == subject_id
    registry.observe_snapshot(_snapshot(2, predicted=True))
    registry.observe_snapshot(_snapshot(3.3))
    assert registry.history()[-1].objects[0].track_id != subject_id


def test_known_face_subject_upgrades_to_body_without_new_public_id() -> None:
    registry = PersonRegistry("session", "camera")
    first = registry.resolve_faces(
        [_face("known")], timestamp=0, frame_sequence=0, people={"alice": "Alice"}
    )[0]
    result = registry.prepare_analysis(_result(0), _bundle())
    assert result.objects[0].attributes["subject_id"] == first.attributes["subject_id"]
    assert result.objects[0].attributes["subject_identity"]["person_id"] == "alice"


def test_deleted_person_is_removed_on_body_only_frames_and_stale_face_completion() -> (
    None
):
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(_snapshot(0))
    face = registry.resolve_faces(
        [_face("known")], timestamp=0, frame_sequence=0, people={"alice": "Alice"}
    )[0]
    first = registry.prepare_analysis(_result(0), _bundle())
    subject_id = first.objects[0].attributes["subject_id"]
    assert first.objects[0].attributes["subject_identity"]["person_id"] == "alice"
    registry.forget_person("alice")
    registry.observe_snapshot(_snapshot(1))
    current = registry.prepare_analysis(_result(1), _bundle())
    assert current.objects[0].attributes["subject_id"] == subject_id
    assert "subject_identity" not in current.objects[0].attributes

    # A worker that read the enrollment before deletion cannot resurrect it.
    stale = registry.resolve_faces(
        [face], timestamp=2, frame_sequence=20, people={"alice": "Alice"}
    )[0]
    assert stale.attributes["status"] == "unusable"
    assert "person_id" not in stale.attributes
    registry.observe_snapshot(_snapshot(2))
    current = registry.prepare_analysis(_result(2), _bundle())
    assert "subject_identity" not in current.objects[0].attributes

    # A result already prepared before deletion may still be enqueued late.
    assert (
        "subject_identity" not in registry.prepare_analysis(first).objects[0].attributes
    )
    stale_result = replace(
        _result(2),
        objects=(GroundedObject("o1", face.label, FACE, attributes=face.attributes),),
    )
    sanitized = registry.prepare_analysis(stale_result)
    assert "person_id" not in sanitized.objects[0].attributes
    assert sanitized.objects[0].attributes["status"] == "unusable"

    registry.allow_person("alice")
    reenrolled = registry.resolve_faces(
        [_face("known")], timestamp=3, frame_sequence=30, people={"alice": "New Alice"}
    )[0]
    assert reenrolled.attributes["status"] == "known"
    assert reenrolled.attributes["person_id"] == "alice"


def test_empty_face_observations_clear_removed_enrollment_without_callback() -> None:
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(_snapshot(0))
    registry.resolve_faces(
        [_face("known")], timestamp=0, frame_sequence=0, people={"alice": "Alice"}
    )
    registry.observe_snapshot(_snapshot(1))
    registry.resolve_faces([], timestamp=1, frame_sequence=10, people={})
    current = registry.prepare_analysis(_result(1), _bundle())
    assert "subject_identity" not in current.objects[0].attributes


def test_known_identity_survives_twenty_seconds_of_continuous_side_views() -> None:
    registry = PersonRegistry("session", "camera")
    subject_id = None
    for index in range(51):
        timestamp = index * 0.4
        registry.observe_snapshot(_snapshot(timestamp))
        face = registry.resolve_faces(
            [_face("known" if index == 0 else "unknown")],
            timestamp=timestamp,
            frame_sequence=index * 4,
            people={"alice": "Alice"},
        )[0]
        assert face.attributes["status"] == "known"
        assert face.attributes["person_id"] == "alice"
        subject_id = subject_id or face.attributes["subject_id"]
        assert face.attributes["subject_id"] == subject_id


def test_a_single_conflicting_identity_does_not_replace_known_person() -> None:
    registry = PersonRegistry("session", "camera")
    people = {"alice": "Alice", "bob": "Bob"}
    first = registry.resolve_faces(
        [_face("known")], timestamp=0, frame_sequence=0, people=people
    )[0]
    conflict = registry.resolve_faces(
        [_face("known", person_id="bob")],
        timestamp=0.4,
        frame_sequence=4,
        people=people,
    )[0]
    assert conflict.attributes["person_id"] == "alice"
    assert conflict.attributes["subject_id"] == first.attributes["subject_id"]
    for index in (2, 3):
        conflict = registry.resolve_faces(
            [_face("known", person_id="bob")],
            timestamp=index * 0.4,
            frame_sequence=index * 4,
            people=people,
        )[0]
    assert conflict.attributes["person_id"] == "bob"
    assert conflict.attributes["subject_id"] != first.attributes["subject_id"]


def test_no_owner_is_invented_from_nearby_person_or_invalid_actor_reference() -> None:
    registry = PersonRegistry("session", "camera")
    bundle = ObservationBundle((ModelActor("a1", BODY),), (("missing",),))
    result = registry.prepare_analysis(_result(0), bundle)
    assert "subject_id" not in result.objects[0].attributes
    wrong_owner = registry.prepare_analysis(_result(1), _bundle(OTHER))
    assert "subject_id" not in wrong_owner.objects[0].attributes


def test_late_actor_frame_does_not_inherit_future_body_observation() -> None:
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(_snapshot(2))
    result = registry.prepare_analysis(_result(1), _bundle())
    assert "subject_id" not in result.objects[0].attributes
    assert registry.snapshot_at(1) is None


def test_model_actors_work_when_a_newer_face_frame_completed_first() -> None:
    registry = PersonRegistry("session", "camera")
    registry.resolve_faces(
        [_face("known")], timestamp=2, frame_sequence=20, people={"alice": "Alice"}
    )
    result = registry.prepare_analysis(_result(1), _bundle())
    assert "subject_id" in result.objects[0].attributes
    assert registry.snapshot_at(1) is not None


def test_late_actor_binds_source_face_without_moving_latest_face() -> None:
    registry = PersonRegistry("session", "camera")
    people = {"alice": "Alice"}
    source_face = registry.resolve_faces(
        [_face("known")], timestamp=1, frame_sequence=10, people=people
    )[0]
    latest_box = NormalizedBBox(0.23, 0.1, 0.33, 0.2)
    latest_face = registry.resolve_faces(
        [replace(_face("known"), bbox=latest_box)],
        timestamp=2,
        frame_sequence=20,
        people=people,
    )[0]
    current = registry.prepare_analysis(_result(1), _bundle())
    subject_id = source_face.attributes["subject_id"]
    assert latest_face.attributes["subject_id"] == subject_id
    assert current.objects[0].attributes["subject_ids"] == (subject_id,)
    assert current.objects[0].bbox == ITEM
    assert registry._subjects[subject_id].bbox == latest_box
    assert registry._subjects[subject_id].seen_at == 2


def test_late_actor_cannot_bind_new_person_at_old_person_position() -> None:
    registry = PersonRegistry("session", "camera")
    people = {"alice": "Alice", "bob": "Bob"}
    old_face = registry.resolve_faces(
        [_face("known")], timestamp=1, frame_sequence=10, people=people
    )[0]
    new_face = registry.resolve_faces(
        [_face("known", person_id="bob")],
        timestamp=5,
        frame_sequence=50,
        people=people,
    )[0]
    current = registry.prepare_analysis(_result(1), _bundle())
    assert old_face.attributes["subject_id"] != new_face.attributes["subject_id"]
    assert (
        current.objects[0].attributes["subject_id"]
        == old_face.attributes["subject_id"]
    )
    assert "subject_identity" not in current.objects[0].attributes


def test_source_actor_uses_frame_sequence_and_recent_face_limit() -> None:
    for face_timestamp, face_sequence, source_timestamp in (
        (1.0, 11, 1.0),  # Same timestamp, but a future source frame.
        (0.0, 0, 0.6001),  # The last face is too old for source association.
    ):
        registry = PersonRegistry("session", "camera")
        face = registry.resolve_faces(
            [_face("known")],
            timestamp=face_timestamp,
            frame_sequence=face_sequence,
            people={"alice": "Alice"},
        )[0]
        current = registry.prepare_analysis(_result(source_timestamp), _bundle())
        assert (
            current.objects[0].attributes.get("subject_id")
            != face.attributes["subject_id"]
        )


def test_source_actor_rejects_two_faces_inside_one_body() -> None:
    registry = PersonRegistry("session", "camera")
    second_face = replace(
        _face("known", person_id="bob"),
        bbox=NormalizedBBox(0.31, 0.1, 0.41, 0.2),
    )
    faces = registry.resolve_faces(
        [_face("known"), second_face],
        timestamp=1,
        frame_sequence=10,
        people={"alice": "Alice", "bob": "Bob"},
    )
    current = registry.prepare_analysis(_result(1), _bundle())
    assert not set(current.objects[0].attributes.get("subject_ids", ())).intersection(
        face.attributes["subject_id"] for face in faces
    )


def test_one_source_face_cannot_be_claimed_by_two_actors() -> None:
    registry = PersonRegistry("session", "camera")
    face = registry.resolve_faces(
        [_face("known")], timestamp=1, frame_sequence=10, people={"alice": "Alice"}
    )[0]
    bundle = ObservationBundle(
        (
            ModelActor("a1", BODY),
            ModelActor("a2", NormalizedBBox(0.05, 0.01, 0.6, 0.9)),
        ),
        (("a1", "a2"),),
    )
    current = registry.prepare_analysis(_result(1), bundle)
    assert face.attributes["subject_id"] not in current.objects[0].attributes.get(
        "subject_ids", ()
    )


def test_face_history_covers_delayed_results_and_expires_within_90_seconds() -> None:
    registry = PersonRegistry("session", "camera")
    original_id = None
    for index in range(1000):
        face = registry.resolve_faces(
            [_face("known")],
            timestamp=index / 7.5,
            frame_sequence=index,
            people={"alice": "Alice"},
        )[0]
        if index == 413:
            original_id = face.attributes["subject_id"]
    assert len(registry._face_history) <= 1024
    assert registry._face_history[0].timestamp >= 999 / 7.5 - 90
    current = registry.prepare_analysis(_result(55.1), _bundle())
    assert current.objects[0].attributes["subject_id"] == original_id


def test_held_face_cannot_supply_source_actor_or_observed_clear() -> None:
    registry = PersonRegistry("session", "camera")
    face = registry.resolve_faces(
        [_face("unknown")], timestamp=0, frame_sequence=0, people={}
    )[0]
    held = registry.resolve_faces([], timestamp=0.5, frame_sequence=5, people={})[0]
    assert face.attributes["face_observation_state"] == "observed"
    assert held.attributes["face_observation_state"] == "held"
    current = registry.prepare_analysis(_result(0.5), _bundle())
    assert current.objects[0].attributes["subject_id"] != face.attributes["subject_id"]


def test_later_source_face_corrects_pending_event_with_provisional_body_id() -> None:
    registry = PersonRegistry("session", "camera")
    # The event was provisionally associated before the face branch completed.
    previous = registry.prepare_analysis(_result(1), _bundle())
    old_id = previous.objects[0].attributes["subject_id"]
    # The nearest preceding face finishes later. It must not inherit a future
    # body snapshot, and supplies its own source-time face ID for the event.
    face = registry.resolve_faces(
        [_face("known")], timestamp=0.5, frame_sequence=5, people={"alice": "Alice"}
    )[0]
    assert face.attributes["subject_id"] != old_id
    updates = registry.drain_reassociated_results()
    assert len(updates) == 1
    assert updates[0].objects[0].attributes["subject_ids"] == (
        face.attributes["subject_id"],
    )
    assert (
        updates[0].objects[0].attributes["subject_id"]
        == face.attributes["subject_id"]
    )
    assert updates[0].message_id == previous.message_id
    assert updates[0].judgments == previous.judgments
    assert updates[0].frame == previous.frame
    assert registry.drain_reassociated_results() == ()
    following = registry.resolve_faces(
        [_face("known")], timestamp=1.5, frame_sequence=15, people={"alice": "Alice"}
    )[0]
    assert following.attributes["subject_id"] == face.attributes["subject_id"]


def test_multiple_people_keep_distinct_subjects_when_actor_order_changes() -> None:
    registry = PersonRegistry("session", "camera")
    for timestamp, actors in (
        (0, (ModelActor("left", BODY), ModelActor("right", OTHER))),
        (1, (ModelActor("right", OTHER), ModelActor("left", BODY))),
    ):
        result = registry.prepare_analysis(
            _result(timestamp), ObservationBundle(actors, (("left",),))
        )
        subject_id = result.objects[0].attributes["subject_id"]
        if timestamp == 0:
            original = subject_id
        else:
            assert subject_id == original
        assert len({item.track_id for item in registry.history()[-1].objects}) == 2


def test_history_covers_65_second_late_results_and_remains_bounded() -> None:
    registry = PersonRegistry("session", "camera")
    for index in range(901):
        registry.observe_snapshot(_snapshot(index / 7.5))
    history = registry.history()
    assert len(history) <= 1024
    assert history[0].timestamp_seconds >= 30
    past = registry.snapshot_at(55)
    assert past is not None and 54.8 <= past.timestamp_seconds <= 55
    late = registry.prepare_analysis(_result(55), _bundle())
    assert late.objects[0].attributes["subject_id"] == past.objects[0].track_id
    assert registry.snapshot_at(20) is None


def test_background_association_correction_preserves_message_and_verdict() -> None:
    registry = PersonRegistry("session", "camera")
    original = _result(1, label="person", box=BODY)
    immediate = registry.prepare_analysis(original)
    assert "subject_id" not in immediate.objects[0].attributes
    assert registry.drain_reassociated_results() == ()
    registry.observe_snapshot(_snapshot(1))
    updates = registry.drain_reassociated_results()
    assert len(updates) == 1
    assert updates[0].message_id == original.message_id
    assert updates[0].frame == original.frame
    assert updates[0].judgments == original.judgments
    assert "subject_id" in updates[0].objects[0].attributes
    assert registry.drain_reassociated_results() == ()


def test_future_identity_is_not_attached_to_an_older_subject_event() -> None:
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(_snapshot(1))
    registry.observe_snapshot(_snapshot(2))
    registry.resolve_faces(
        [_face("known")], timestamp=2, frame_sequence=20, people={"alice": "Alice"}
    )
    late = registry.prepare_analysis(_result(1), _bundle())
    assert "subject_id" in late.objects[0].attributes
    assert "subject_identity" not in late.objects[0].attributes


def test_older_face_result_keeps_its_own_box_and_cannot_replace_future_identity() -> (
    None
):
    registry = PersonRegistry("session", "camera")
    people = {"alice": "Alice", "bob": "Bob"}
    registry.observe_snapshot(_snapshot(1))
    registry.observe_snapshot(_snapshot(2))
    current = registry.resolve_faces(
        [_face("known", person_id="bob")], timestamp=2, frame_sequence=20, people=people
    )[0]
    old = registry.resolve_faces(
        [_face("known")], timestamp=1, frame_sequence=10, people=people
    )[0]
    assert old.bbox == FACE
    assert old.attributes["person_id"] == "alice"
    assert "subject_id" not in old.attributes
    following = registry.resolve_faces(
        [_face("unknown")], timestamp=2.4, frame_sequence=24, people=people
    )[0]
    assert following.attributes["person_id"] == "bob"
    assert following.attributes["subject_id"] == current.attributes["subject_id"]


def test_crossing_ambiguity_does_not_inherit_either_previous_identity() -> None:
    registry = PersonRegistry("session", "camera")
    left = NormalizedBBox(0.15, 0.05, 0.5, 0.95)
    right = NormalizedBBox(0.5, 0.05, 0.85, 0.95)
    first = ObservationBundle((ModelActor("left", left), ModelActor("right", right)))
    registry.prepare_analysis(_result(0, detected=False), first)
    old_ids = {item.track_id for item in registry.history()[-1].objects}
    middle = NormalizedBBox(0.325, 0.05, 0.675, 0.95)
    registry.prepare_analysis(
        _result(0.5, detected=False), ObservationBundle((ModelActor("middle", middle),))
    )
    new_ids = {item.track_id for item in registry.history()[-1].objects}
    assert not new_ids.intersection(old_ids)


def test_body_observations_keep_known_identity_between_sparse_face_requests() -> None:
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(_snapshot(0))
    first = registry.resolve_faces(
        [_face("known")], timestamp=0, frame_sequence=0, people={"alice": "Alice"}
    )[0]
    for index in range(1, 151):
        registry.observe_snapshot(_snapshot(index / 7.5))
    # No face inference completed during the intervening twenty seconds.
    current = registry.resolve_faces(
        [_face("unknown")], timestamp=20, frame_sequence=200, people={"alice": "Alice"}
    )[0]
    assert current.attributes["status"] == "known"
    assert current.attributes["person_id"] == "alice"
    assert current.attributes["subject_id"] == first.attributes["subject_id"]


def test_fight_targets_share_one_participant_group_without_changing_boxes() -> None:
    registry = PersonRegistry("session", "camera")
    original = replace(
        _result(0),
        judgments=(EventJudgment("fight", True, 0.9, "Fight", ("left", "right")),),
        objects=(
            GroundedObject("left", "person", BODY),
            GroundedObject("right", "person", OTHER),
        ),
    )
    bundle = ObservationBundle(
        (ModelActor("a1", BODY), ModelActor("a2", OTHER)),
        (("a1",), ("a2",)),
    )
    current = registry.prepare_analysis(original, bundle)
    left, right = current.objects
    assert left.bbox == BODY and right.bbox == OTHER
    assert len(left.attributes["subject_ids"]) == 2
    assert left.attributes["subject_ids"] == right.attributes["subject_ids"]
    assert left.attributes["subject_id"] != right.attributes["subject_id"]


def test_two_spatially_separate_fights_do_not_share_a_group() -> None:
    registry = PersonRegistry("session", "camera")
    boxes = (
        NormalizedBBox(0.02, 0.1, 0.12, 0.9),
        NormalizedBBox(0.13, 0.1, 0.23, 0.9),
        NormalizedBBox(0.70, 0.1, 0.80, 0.9),
        NormalizedBBox(0.81, 0.1, 0.91, 0.9),
    )
    ids = tuple(f"o{index}" for index in range(4))
    original = replace(
        _result(0),
        judgments=(EventJudgment("fight", True, 0.9, "Two fights", ids),),
        objects=tuple(
            GroundedObject(object_id, "person", box)
            for object_id, box in zip(ids, boxes, strict=True)
        ),
    )
    bundle = ObservationBundle(
        tuple(
            ModelActor(object_id, box)
            for object_id, box in zip(ids, boxes, strict=True)
        ),
        tuple((object_id,) for object_id in ids),
    )
    current = registry.prepare_analysis(original, bundle)
    left = current.objects[0].attributes["subject_ids"]
    right = current.objects[2].attributes["subject_ids"]
    assert len(left) == len(right) == 2
    assert current.objects[1].attributes["subject_ids"] == left
    assert current.objects[3].attributes["subject_ids"] == right
    assert not set(left).intersection(right)


def test_reprocessing_the_same_memory_actor_does_not_advance_conflict() -> None:
    registry = PersonRegistry("session", "camera")
    first = ObservationBundle((ModelActor("a1", BODY, 0.9, "alice"),), (("a1",),))
    initial = registry.prepare_analysis(_result(0), first)
    conflict = ObservationBundle((ModelActor("a1", BODY, 0.9, "bob"),), (("a1",),))
    for _ in range(5):
        current = registry.prepare_analysis(_result(1), conflict)
    assert (
        current.objects[0].attributes["subject_id"]
        == initial.objects[0].attributes["subject_id"]
    )
    assert (
        current.objects[0].attributes["subject_identity"]["matched_memory_id"]
        == "alice"
    )


def test_memory_identity_change_requires_consecutive_uninterrupted_confirmation() -> (
    None
):
    registry = PersonRegistry("session", "camera")
    previous_id = None
    for timestamp, identity in enumerate(
        ("alice", "bob", "alice", "bob", None, "bob", "bob")
    ):
        bundle = ObservationBundle((ModelActor("a1", BODY, 0.9, identity),), (("a1",),))
        current = registry.prepare_analysis(_result(timestamp), bundle)
        subject_id = current.objects[0].attributes["subject_id"]
        previous_id = previous_id or subject_id
        assert subject_id == previous_id
        assert (
            current.objects[0].attributes["subject_identity"]["matched_memory_id"]
            == "alice"
        )
    current = registry.prepare_analysis(
        _result(7), ObservationBundle((ModelActor("a1", BODY, 0.9, "bob"),), (("a1",),))
    )
    assert current.objects[0].attributes["subject_id"] != previous_id
    assert (
        current.objects[0].attributes["subject_identity"]["matched_memory_id"] == "bob"
    )
