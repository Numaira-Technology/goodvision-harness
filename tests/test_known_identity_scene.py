"""Known gallery identities consolidate face fragments without changing tracks."""

import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.scene_state import SceneState

BODY = NormalizedBBox(0.328125, 0.304688, 0.734375, 1.0)
FACE = NormalizedBBox(0.368828, 0.486057, 0.704638, 1.0)
FRANK = {
    "status": "known",
    "person_id": "frank",
    "name": "Frank",
    "identity_source": "face_match",
}
NATIVE = {"backend": "sam3.1_video", "epoch": "epoch", "object_id": "0"}


def observe(scene, subject, at, *, native=None, identity=None, bbox=FACE):
    assert scene.observe_person(
        subject,
        timestamp_seconds=at,
        frame_sequence=int(at * 10),
        bbox=bbox,
        identity=FRANK if identity is None else identity,
        native_track=native,
    )


def test_gallery_review_merges_overlapping_face_fragments_and_preserves_body():
    runtime = SceneRuntime("s", "camera", event_labels=("smoking",))
    scene = runtime.scene
    observe(scene, "body", 0, native=NATIVE, bbox=BODY)
    observe(scene, "face-fragment", 2)
    assert scene.observe_event(
        "smoking",
        ("face-fragment",),
        timestamp_seconds=2,
        frame_sequence=20,
        detected=True,
        evidence_id="first-smoking",
    )
    observe(scene, "body", 3, native=NATIVE, bbox=BODY)
    observe(scene, "face-fragment", 4)
    assert runtime.apply_identity_merges((("face-fragment", "body"),)) == 1
    view = scene.snapshot()
    assert len(view["entities"]) == 1
    entity = view["entities"][0]
    assert entity["native_track"] == NATIVE
    assert entity["bbox"] == BODY.as_dict()
    assert entity["last_observed_timestamp_seconds"] == 3
    assert entity["frame_sequence"] == 30
    assert view["events"][0]["subject_ids"] == ["body"]
    # Further same-gallery face results must not restore a separate entity or
    # overwrite the native body's location with the face box.
    observe(scene, "body", 5, identity={**FRANK, "identity_source": "track_history"})
    assert scene.snapshot()["entities"][0]["bbox"] == BODY.as_dict()
    assert scene.snapshot()["entities"][0]["last_observed_timestamp_seconds"] == 3
    assert scene.snapshot_at(4)["entities"][0]["bbox"] == BODY.as_dict()
    assert scene.snapshot_at(4)["entities"][0]["native_track"] == NATIVE


def test_same_source_known_face_inside_body_is_not_a_second_person():
    scene = SceneState("s", "camera")
    observe(scene, "body", 2, native=NATIVE, bbox=BODY)
    observe(scene, "face", 2)
    assert scene.merge_subject("face", "body", reviewed_identity=True)
    entity = scene.snapshot()["entities"][0]
    assert entity["bbox"] == BODY.as_dict()
    assert entity["native_track"] == NATIVE
    assert len(entity["history"]) == 1


@pytest.mark.parametrize("case", ["two_faces", "two_bodies", "disjoint_face"])
def test_same_gallery_id_cannot_override_simultaneous_conflicting_observations(case):
    scene = SceneState("s", "camera")
    observe(scene, "a", 2, native=None if case == "two_faces" else NATIVE, bbox=BODY)
    observe(
        scene,
        "b",
        2,
        native={**NATIVE, "object_id": "1"} if case == "two_bodies" else None,
        bbox=NormalizedBBox(0.01, 0.1, 0.1, 0.3),
    )
    assert not scene.merge_subject("b", "a", reviewed_identity=True)
    assert len(scene.snapshot()["entities"]) == 2


def test_matching_names_with_different_gallery_ids_never_merge():
    scene = SceneState("s", "camera")
    observe(scene, "a", 1)
    observe(scene, "b", 2, identity={**FRANK, "person_id": "another-frank"})
    assert not scene.merge_subject("b", "a", reviewed_identity=True)


def test_unknown_faces_still_require_nonoverlapping_reviewed_trajectories():
    scene = SceneState("s", "camera")
    unknown = {"status": "unknown", "identity_source": "face_match"}
    observe(scene, "a", 1, identity=unknown)
    observe(scene, "b", 2, identity=unknown)
    observe(scene, "a", 3, identity=unknown)
    assert not scene.merge_subject("b", "a", reviewed_identity=True)
