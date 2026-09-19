"""Identity continuity across late association, face churn and event consumers."""

from dataclasses import replace

import pytest
from test_event_archive import result as archive_result
from test_scene_runtime import result as scene_result
from test_scene_runtime import tracks
from test_stranger_identity_correction import known
from test_subject_fusion import native

from goodvision_harness.contracts import EventJudgment, GroundedObject, NormalizedBBox
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.grounding import ModelObject, ObservationBundle
from goodvision_harness.person_identity import stranger_evidence
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.scene_runtime import SceneRuntime

BODY = NormalizedBBox(0.1, 0.05, 0.9, 0.95)
FACE = NormalizedBBox(0.2, 0.1, 0.3, 0.25)
FAR = NormalizedBBox(0.6, 0.1, 0.7, 0.25)
PEOPLE = {"frank": "Frank", "visitor": "Visitor"}


def face(status, box=FACE, *, person="frank", quality=False):
    attributes = {
        "status": status,
        "threshold": 0.4,
        "similarity": 0.9 if status == "known" else 0.1,
    }
    if status == "known":
        attributes.update(person_id=person, name=PEOPLE[person])
    return ModelObject(
        {
            "known": "known_person",
            "unknown": "unknown_person",
            "unusable": "unusable_face",
        }[status],
        box,
        0.9,
        attributes,
        identity_embedding=((1.0, 0.0) if status == "known" else (0.0, 1.0))
        if quality or status == "known"
        else (),
        identity_quality=quality,
    )


def resolve(registry, at, *objects):
    return registry.resolve_faces(
        objects, timestamp=at, frame_sequence=int(at * 10), people=PEOPLE
    )


def test_late_body_merge_updates_existing_face_identity_and_survives_churn():
    registry = PersonRegistry("s", "camera")
    registry.observe_snapshot(native(0, boxes=(BODY,)))
    first = resolve(registry, 0, face("known"))[0]
    sid = first.attributes["subject_id"]
    for at in (1, 2, 3):
        registry.observe_snapshot(native(at, boxes=(BODY,)))
    fragment = resolve(registry, 4, face("unusable", FAR))[0]
    assert fragment.attributes["subject_id"] != sid
    registry.observe_snapshot(native(4, boxes=(BODY,)))
    assert (fragment.attributes["subject_id"], sid) in registry.drain_subject_merges()
    later = resolve(registry, 4.4, face("unknown", FAR))[0]
    assert later.attributes["subject_id"] == sid
    assert later.attributes["status"] == "known"
    assert later.attributes["person_id"] == "frank"
    assert later.attributes["observation_status"] == "unknown"
    assert later.attributes["identity_confirmed_at"] == 0
    assert stranger_evidence(later.attributes) is False
    # Both the existing and replacement face track borrow the body identity.
    registry.observe_snapshot(native(9, boxes=(BODY,)))
    replacement = resolve(registry, 9, face("unknown"))[0]
    assert replacement.attributes["status"] == "known"
    assert replacement.attributes["identity_confirmed_at"] == 0
    registry.review_pending_identities()
    assert registry.identity_review_states()[sid] == "known"


def test_epoch_reset_drops_face_only_track_that_was_later_merged():
    registry = PersonRegistry("s", "camera")
    first = resolve(registry, 0, face("known"))[0]
    registry.observe_snapshot(native(0, boxes=(BODY,)))
    registry.begin_native_epoch("sam3.1_video", "replacement")
    registry.observe_snapshot(native(1, boxes=(BODY,), epoch="replacement"))
    current = resolve(registry, 1, face("unknown"))[0]
    assert current.attributes["subject_id"] != first.attributes["subject_id"]
    assert current.attributes["status"] == "unknown"


def test_native_person_identity_can_be_reset_by_real_strong_mismatch():
    registry = PersonRegistry("s", "camera")
    registry.observe_snapshot(native(0, boxes=(BODY,)))
    first = resolve(registry, 0, face("known", quality=True))[0]
    for index in range(1, 7):
        at = index * 0.5
        registry.observe_snapshot(native(at, boxes=(BODY,)))
        current = resolve(registry, at, face("unknown", quality=True))[0]
        assert current.attributes["subject_id"] == first.attributes["subject_id"]
        assert current.attributes["status"] == ("known" if index < 5 else "unknown")
        if index == 5:
            assert current.attributes["identity_reset"] is True


def test_late_analysis_does_not_get_future_subject_identity():
    registry = PersonRegistry("s", "camera")
    registry.observe_snapshot(native(1, boxes=(BODY,)))
    previous = resolve(registry, 1, face("unknown"))[0]
    resolve(registry, 2, face("known"))
    obj = GroundedObject("f", "unknown_person", FACE, attributes=previous.attributes)
    value = scene_result(1, detected=False, label="stranger", objects=(obj,))
    value = replace(
        value, judgments=(EventJudgment("stranger", True, 0.9, "test", ("f",)),)
    )
    prepared = registry.prepare_analysis(value, ObservationBundle())
    assert prepared.objects[0].attributes["status"] == "unknown"
    assert "subject_identity" not in prepared.objects[0].attributes
    assert prepared.judgments[0].detected


def test_epoch_change_does_not_inherit_old_person_identity():
    registry = PersonRegistry("s", "camera")
    registry.observe_snapshot(native(0, boxes=(BODY,)))
    previous = resolve(registry, 0, face("known"))[0]
    registry.begin_native_epoch("sam3.1_video", "new-epoch")
    registry.observe_snapshot(native(1, boxes=(BODY,), epoch="new-epoch"))
    current = resolve(registry, 1, face("unknown"))[0]
    assert current.attributes["subject_id"] != previous.attributes["subject_id"]
    assert current.attributes["status"] == "unknown"


@pytest.mark.parametrize("held_first", [True, False])
def test_scene_mixed_frame_keeps_known_person_and_other_stranger_separate(
    held_first,
):
    runtime = SceneRuntime("s", "camera")
    for at in (1, 2):
        tracks(runtime, at, ("a", "b"))
        known_face = GroundedObject(
            "known",
            "known_person",
            FACE,
            0.9,
            attributes={
                "subject_id": "a",
                "status": "known",
                "person_id": "frank",
                "name": "Frank",
                "identity_source": "face_match",
                "face_observation_state": "observed",
            },
        )
        held = GroundedObject(
            "held",
            "unknown_person",
            FAR,
            0.9,
            attributes={
                "subject_id": "a",
                "status": "unknown",
                "identity_source": "track_history",
                "face_observation_state": "held",
            },
        )
        objects = (held, known_face) if held_first else (known_face, held)
        value = scene_result(at, (("b",),), label="stranger", objects=objects)
        value = replace(
            value,
            judgments=(replace(value.judgments[0], object_ids=("held", "event-0")),),
        )
        runtime.observe_result(value)
    events = runtime.scene.snapshot()["events"]
    assert any(e["subject_ids"] == ["b"] and e["phase"] == "confirmed" for e in events)
    assert not any(
        e["subject_ids"] == ["a"] and e["phase"] == "confirmed" for e in events
    )


def test_scene_explicit_identity_reset_accepts_new_stranger():
    runtime = SceneRuntime("s", "camera")
    tracks(runtime, 1)
    runtime.scene.observe_person(
        "a",
        timestamp_seconds=1,
        frame_sequence=10,
        bbox=BODY,
        identity={"status": "known", "person_id": "frank"},
    )
    for at in (2, 3):
        tracks(runtime, at)
        value = scene_result(at, (("a",),), label="stranger")
        value = replace(
            value, objects=(replace(value.objects[0], label="unknown_person"),)
        )
        if at == 2:
            value = replace(
                value,
                objects=(
                    replace(
                        value.objects[0],
                        attributes={
                            **value.objects[0].attributes,
                            "identity_reset": True,
                        },
                    ),
                ),
            )
        runtime.observe_result(value)
    view = runtime.scene.snapshot()
    assert view["entities"][0]["identity"]["status"] == "unknown"
    assert view["events"][0]["phase"] == "confirmed"


def test_archive_ignores_held_votes_and_keeps_future_unknown_identity(tmp_path):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0)
    try:
        archive.submit_result(known(1))
        held = archive_result(2, "held", subject="other", label="stranger")
        held["objects"][0]["attributes"].update(
            status="unknown",
            identity_source="track_history",
            face_observation_state="held",
        )
        archive.submit_result(held)
        future = archive_result(3, "future", subject="face-fragment", label="stranger")
        future["objects"][0]["attributes"].update(
            status="unknown",
            identity_source="face_match",
            face_observation_state="observed",
            identity_reset=True,
        )
        archive.submit_result(future)
        archive.reconcile_subjects("session", {})
        assert archive.flush()
        assert not archive.list_events(subject_id="other")["events"]
        event = archive.list_events(message_id="future")["events"][0]
        assert event["end_reason"] != "retracted"
        assert event["identity_state"] == "unknown"
        assert event["person_name"] is None
    finally:
        archive.close()


def test_archive_registry_continuity_corrects_merged_fragment_after_last_gallery_match(
    tmp_path,
):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0)
    try:
        archive.submit_result(known(57.803, subject="body"))
        archive.submit_result(
            archive_result(62.049, "fragment", subject="fragment", label="stranger")
        )
        archive.reconcile_subjects("session", {"fragment": "body"})
        continuity = known(63, subject="body", source="track_history")
        continuity["objects"][0]["attributes"]["identity_confirmed_at"] = 57.803
        archive.submit_result(continuity)
        assert archive.flush()
        event = archive.list_events(message_id="fragment")["events"][0]
        assert event["end_reason"] == "retracted"
        assert event["person_name"] == "Frank"
    finally:
        archive.close()


def test_archive_revised_held_observation_cannot_reappear_after_restart(tmp_path):
    settings = EventArchiveSettings(directory=tmp_path)
    archive = EventArchive(settings, clock=lambda: 0)
    try:
        value = archive_result(1, "sample", subject="person", label="stranger")
        value["objects"][0]["attributes"].update(
            status="unknown",
            identity_source="face_match",
            face_observation_state="observed",
        )
        archive.submit_result(value)
        assert archive.flush()
        event_id = archive.list_events()["events"][0]["event_id"]
        value["objects"][0]["attributes"].update(
            identity_source="track_history", face_observation_state="held"
        )
        archive.submit_result(value)
        assert archive.flush()
        assert archive.get_event(event_id)["end_reason"] == "retracted"
    finally:
        archive.close()
    reopened = EventArchive(settings, clock=lambda: 0)
    try:
        reopened.reconcile_subjects("session", {})
        assert reopened.flush()
        assert reopened.get_event(event_id)["end_reason"] == "retracted"
    finally:
        reopened.close()
