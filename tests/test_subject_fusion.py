"""A delayed native track and local face require source evidence to share a person."""

import pytest

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.grounding import ModelObject
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime, TemporalVerdict
from goodvision_harness.scene_state import SceneState

BODY = NormalizedBBox(0.1, 0.05, 0.5, 0.95)
FACE = NormalizedBBox(0.2, 0.1, 0.3, 0.25)
OTHER_FACE = NormalizedBBox(0.32, 0.1, 0.42, 0.25)
NATIVE = {"backend": "sam3.1_video", "epoch": "epoch-a", "object_id": "0"}


def native(at, *, boxes=(BODY,), epoch="epoch-a", sequence=None):
    seq = int(at * 10) if sequence is None else sequence
    return RichLabelSnapshot(
        "s",
        seq,
        at,
        tuple(
            RichLabelObject(
                str(index),
                "person",
                box,
                0.9,
                {"native_track": {**NATIVE, "epoch": epoch, "object_id": str(index)}},
                observation_state="observed",
                last_observed_timestamp_seconds=at,
                last_observed_frame_sequence=seq,
            )
            for index, box in enumerate(boxes)
        ),
    )


def face(registry, at, *, boxes=(FACE,), person=None, sequence=None):
    attrs = (
        {"status": "unknown"}
        if person is None
        else {
            "status": "known",
            "person_id": person,
            "name": person,
        }
    )
    return registry.resolve_faces(
        tuple(
            ModelObject(
                "unknown_person" if person is None else "known_person", box, 0.9, attrs
            )
            for box in boxes
        ),
        timestamp=at,
        frame_sequence=int(at * 10) if sequence is None else sequence,
        people={"alice": "Alice", "bob": "Bob"},
    )


def subject(registry, at):
    return registry.snapshot_at(at, int(at * 10)).objects[0].track_id


def observation(at, sid, *, label="stranger", detected=True):
    return AnalysisResult(
        f"{label}-{at}-{sid}",
        "camera",
        "s",
        int(at * 10) + 1,
        "now",
        FrameReference(int(at * 10), at, "now", 640, 480),
        (EventJudgment(label, detected, 0.9, "test", ("person",)),),
        (
            GroundedObject(
                "person",
                "person",
                FACE,
                attributes={
                    "subject_ids": [sid],
                    "status": "unknown",
                    "identity_source": "face_match",
                    "face_observation_state": "observed",
                },
            ),
        ),
        "test",
        0.1,
    )


def test_late_native_claims_exact_source_face_without_duplicate_or_future_identity():
    registry = PersonRegistry("s", "camera")
    first = face(registry, 0)[0].attributes["subject_id"]
    for at in range(1, 21):
        assert face(registry, at, person="alice")[0].attributes["subject_id"] == first
    registry.observe_snapshot(native(0))
    assert subject(registry, 0) == first
    assert len(registry._subjects) == 1
    assert registry.drain_subject_merges() == ()
    assert registry._subjects[first].identity["person_id"] == "alice"
    # Source geometry is from frame zero; later face identity is not exported
    # into that historical native source snapshot.
    item = registry.snapshot_at(0, 0).objects[0]
    assert item.bbox == BODY
    assert "subject_identity" not in item.attributes
    assert face(registry, 21, person="alice")[0].attributes["subject_id"] == first


@pytest.mark.parametrize(
    "face_at, face_seq, body_at, body_seq",
    [
        (0, 0, 0.1, 1),
        (0, 0, 0, 1),
        (0.1, 1, 0, 0),
    ],
)
def test_nearby_or_future_face_is_not_exact_native_association(
    face_at, face_seq, body_at, body_seq
):
    registry = PersonRegistry("s", "camera")
    old = face(registry, face_at, sequence=face_seq)[0].attributes["subject_id"]
    registry.observe_snapshot(native(body_at, sequence=body_seq))
    new = registry.snapshot_at(body_at, body_seq).objects[0].track_id
    assert new != old
    assert registry.drain_subject_merges() == ()


@pytest.mark.parametrize(
    "faces,bodies",
    [
        ((FACE, OTHER_FACE), (BODY,)),
        ((FACE,), (BODY, BODY)),
    ],
)
def test_ambiguous_crossing_or_two_faces_never_merge(faces, bodies):
    registry = PersonRegistry("s", "camera")
    old = {item.attributes["subject_id"] for item in face(registry, 0, boxes=faces)}
    registry.observe_snapshot(native(0, boxes=bodies))
    new = {item.track_id for item in registry.snapshot_at(0, 0).objects}
    assert old.isdisjoint(new)
    assert len(new) == len(bodies)
    assert registry.drain_subject_merges() == ()


def test_held_face_is_not_same_source_association():
    registry = PersonRegistry("s", "camera")
    old = face(registry, 0)[0].attributes["subject_id"]
    held = face(registry, 1, boxes=())
    assert held[0].attributes["face_observation_state"] == "held"
    registry.observe_snapshot(native(1))
    assert subject(registry, 1) != old
    assert registry.drain_subject_merges() == ()


def test_expired_live_face_can_still_be_claimed_from_its_retained_exact_source():
    registry = PersonRegistry("s", "camera")
    old = face(registry, 0)[0].attributes["subject_id"]
    face(registry, 20, boxes=())
    assert old not in registry._subjects
    registry.observe_snapshot(native(0))
    assert subject(registry, 0) == old
    assert registry.drain_subject_merges() == ()


def test_expired_face_event_owner_is_merged_when_existing_native_catches_up():
    registry = PersonRegistry("s", "camera")
    registry.observe_snapshot(native(0))
    canonical = subject(registry, 0)
    old = face(registry, 2)[0].attributes["subject_id"]
    face(registry, 20, boxes=())
    assert old not in registry._subjects
    registry.observe_snapshot(native(2))
    assert registry.drain_subject_merges() == ((old, canonical),)


def test_existing_native_late_match_merges_face_and_keeps_identity_and_event_history():
    registry = PersonRegistry("s", "camera")
    runtime = SceneRuntime("s", "camera", mode="active", tracking_mode="batch")
    registry.observe_snapshot(native(0))
    canonical = subject(registry, 0)
    runtime.scene.observe_person(
        canonical, timestamp_seconds=0, frame_sequence=0, bbox=BODY, native_track=NATIVE
    )
    # Face analysis is ahead of the native body watermark, so it initially
    # creates a provisional face entity and two real stranger observations.
    old = face(registry, 2)[0].attributes["subject_id"]
    assert old != canonical
    for at in (2, 3):
        if at != 2:
            face(registry, at)
        runtime.observe_frame(b"source", at, int(at * 10))
        runtime.observe_result(observation(at, old))
    before = runtime.scene.snapshot()["events"][0]
    assert before["phase"] == "confirmed"
    registry.observe_snapshot(native(2))
    merges = registry.drain_subject_merges()
    assert merges == ((old, canonical),)
    assert runtime.apply_subject_merges(merges) == 1
    scene = runtime.scene.snapshot()
    assert [item["subject_id"] for item in scene["entities"]] == [canonical]
    after = scene["events"][0]
    assert after["event_id"] == before["event_id"]
    assert after["phase"] == "confirmed"
    assert after["subject_ids"] == [canonical]
    assert after["history"] == before["history"]
    assert scene["entities"][0]["last_observed_timestamp_seconds"] == 3
    # Delayed old messages and current face aliases cannot recreate the entity.
    runtime.observe_result(observation(3, old))
    assert len(runtime.scene.snapshot()["entities"]) == 1
    assert face(registry, 4)[0].attributes["subject_id"] == canonical


def test_exact_current_body_rebinds_preexisting_face_alias_in_both_arrival_orders():
    registry = PersonRegistry("s", "camera")
    registry.observe_snapshot(native(0))
    canonical = subject(registry, 0)
    old = face(registry, 2)[0].attributes["subject_id"]
    # No face at this source time yet. Native arrives first this time.
    registry.observe_snapshot(native(3))
    assert face(registry, 3)[0].attributes["subject_id"] == canonical
    assert registry.drain_subject_merges() == ((old, canonical),)
    assert len(registry._subjects) == 1


@pytest.mark.parametrize("expire_live_face", [False, True])
def test_different_known_people_are_not_merged_by_body_geometry(expire_live_face):
    registry = PersonRegistry("s", "camera")
    registry.observe_snapshot(native(0))
    canonical = subject(registry, 0)
    face(registry, 0, person="alice")
    # An expired face track is a new independent face, not an Alice correction.
    old = face(registry, 5, person="bob")[0].attributes["subject_id"]
    assert old != canonical
    if expire_live_face:
        face(registry, 20, boxes=())
        assert old not in registry._subjects
    registry.observe_snapshot(native(5))
    assert registry.drain_subject_merges() == ()
    assert registry._subjects[canonical].identity["person_id"] == "alice"
    if not expire_live_face:
        assert registry._subjects[old].identity["person_id"] == "bob"


def test_new_epoch_does_not_reuse_retired_face_native_subject():
    registry = PersonRegistry("s", "camera")
    old = face(registry, 0, person="alice")[0].attributes["subject_id"]
    registry.observe_snapshot(native(0))
    assert subject(registry, 0) == old
    assert registry.begin_native_epoch("sam3.1_video", "epoch-b")
    registry.observe_snapshot(native(1, epoch="epoch-b"))
    assert subject(registry, 1) != old
    assert registry.drain_subject_merges() == ()


def test_merged_duplicate_source_events_still_need_two_independent_frames():
    state = SceneState("s", "camera")
    for sid in ("face", "native"):
        state.observe_person(sid, timestamp_seconds=1, frame_sequence=10, bbox=BODY)
        state.observe_event(
            "smoking",
            (sid,),
            timestamp_seconds=1,
            frame_sequence=10,
            detected=True,
            evidence_id=sid,
        )
    assert state.merge_subject("face", "native")
    event = state.snapshot()["events"][0]
    assert event["phase"] == "suspected"
    assert event["positive_observations"] == 1
    assert set(event["history"][0]["evidence_ids"]) == {"face", "native"}
    state.observe_person("native", timestamp_seconds=2, frame_sequence=20, bbox=BODY)
    state.observe_event(
        "smoking",
        ("native",),
        timestamp_seconds=2,
        frame_sequence=20,
        detected=True,
        evidence_id="next",
    )
    assert state.snapshot()["events"][0]["phase"] == "confirmed"


def test_false_two_person_fight_is_retracted_when_its_two_owners_are_one_person():
    state = SceneState("s", "camera")
    for at in (1, 2):
        for sid in ("face", "native"):
            state.observe_person(
                sid, timestamp_seconds=at, frame_sequence=at, bbox=BODY
            )
        state.observe_event(
            "fight",
            ("face", "native"),
            timestamp_seconds=at,
            frame_sequence=at,
            detected=True,
            evidence_id=str(at),
        )
    before = state.snapshot()["events"][0]
    assert before["phase"] == "confirmed"
    state.merge_subject("face", "native")
    event = state.snapshot()["events"][0]
    assert event["event_id"] == before["event_id"]
    assert event["phase"] == "retracted"


def test_inflight_old_subject_verdict_remains_source_grounded_after_merge():
    runtime = SceneRuntime("s", "camera", mode="active")
    for at in (0, 1):
        runtime.observe_tracks(
            RichLabelSnapshot(
                "s", int(at * 10), at, (RichLabelObject("face", "person", BODY),)
            )
        )
        runtime.observe_frame(b"exact source", at, int(at * 10))
    job = runtime.take_inspections(limit=1)[0]
    assert {sid for _, sid, _ in job.subject_boxes} == {"face"}
    runtime.apply_subject_merges((("face", "native"),))
    # Alias IDs do not expand the IDs the external model was allowed to return.
    with pytest.raises(ValueError):
        runtime.complete_inspection(
            job.request.request_id, (TemporalVerdict("smoking", ("native",), True),)
        )
    assert runtime.complete_inspection(
        job.request.request_id, (TemporalVerdict("smoking", ("face",), True),)
    )
    event = runtime.scene.snapshot()["events"][0]
    assert event["subject_ids"] == ["native"]
    assert event["phase"] == "suspected"
    assert event["last_evidence_timestamp_seconds"] == job.frames[-1].timestamp_seconds
    assert len(runtime.scene.snapshot()["entities"]) == 1


def test_native_objects_cannot_be_merged_by_scene_alias_api():
    runtime = SceneRuntime("s", "camera")
    for index, sid in enumerate(("a", "b")):
        runtime.scene.observe_person(
            sid,
            timestamp_seconds=1,
            frame_sequence=1,
            bbox=BODY,
            native_track={**NATIVE, "object_id": str(index)},
        )
    assert runtime.apply_subject_merges((("a", "b"),)) == 0
    assert len(runtime.scene.snapshot()["entities"]) == 2


@pytest.mark.parametrize("second_detected", [True, False])
def test_inflight_duplicate_owners_do_not_add_two_votes_or_hide_disagreement(
    second_detected,
):
    runtime = SceneRuntime("s", "camera", mode="active")
    for at in (0, 1):
        runtime.observe_tracks(
            RichLabelSnapshot(
                "s",
                int(at * 10),
                at,
                tuple(
                    RichLabelObject(sid, "person", BODY) for sid in ("face", "native")
                ),
            )
        )
        runtime.observe_frame(b"source", at, int(at * 10))
    job = runtime.take_inspections(limit=1)[0]
    runtime.apply_subject_merges((("face", "native"),))
    assert runtime.complete_inspection(
        job.request.request_id,
        (
            TemporalVerdict("smoking", ("face",), True),
            TemporalVerdict("smoking", ("native",), second_detected),
        ),
    )
    events = runtime.scene.snapshot()["events"]
    if second_detected:
        assert events[0]["phase"] == "suspected"
        assert events[0]["positive_observations"] == 1
    else:
        assert events == []


def test_service_callbacks_merge_before_publishing_the_completed_native_source():
    from threading import RLock
    from types import SimpleNamespace

    from goodvision_harness.service import HarnessRuntime
    from goodvision_harness.video_tracking import VideoTrackingBatch

    registry = PersonRegistry("s", "camera")
    scene = SceneRuntime(
        "s", "camera", mode="active", tracking_mode="batch", event_labels=()
    )
    service = HarnessRuntime.__new__(HarnessRuntime)
    service._state_lock = RLock()
    service._session = SimpleNamespace(
        stream_session_id="s", person_registry=registry, archive=None
    )
    service._scene = scene
    service._scene_worker = None
    service._scene_errors = 0
    service._scene_is_active = lambda: True
    published = []
    service._on_scene_update = lambda: published.append(scene.snapshot())
    scene.observe_frame(b"first", 0, 0)
    service._on_tracking_batch(
        VideoTrackingBatch("s", "first", "epoch-a", (native(0),), 0, 0)
    )
    canonical = subject(registry, 0)
    old = face(registry, 2)[0].attributes["subject_id"]
    scene.observe_frame(b"second", 2, 20)
    service._on_timeline_result(observation(2, old))
    assert len(published[-1]["scene"]["entities"]) == 2
    old_event = published[-1]["scene"]["events"][0]["event_id"]
    service._on_tracking_batch(
        VideoTrackingBatch("s", "second", "epoch-a", (native(2),), 20, 2)
    )
    service._on_timeline_result(observation(2, old))
    final = published[-1]["scene"]
    assert [item["subject_id"] for item in final["entities"]] == [canonical]
    assert final["events"][0]["event_id"] == old_event
    assert final["events"][0]["subject_ids"] == [canonical]
    assert final["events"][0]["positive_observations"] == 1
    assert service._scene_errors == 0
