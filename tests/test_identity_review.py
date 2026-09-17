"""Background face review merges people, retaining real source provenance."""

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.face_recognition import (
    FaceRecognitionRuntime,
    FaceRecognizerConfig,
)
from goodvision_harness.grounding import ModelObject
from goodvision_harness.identity_review import IdentityReview, _ReviewScores
from goodvision_harness.identity_review_worker import IdentityReviewWorker
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.service import HarnessRuntime
from goodvision_harness.sliding_window import FrameWindow, SampledFrame

BODY = NormalizedBBox(0.1, 0.05, 0.45, 0.95)
FACE = NormalizedBBox(0.2, 0.1, 0.3, 0.2)


def face(subject="a", *, embedding=(1.0, 0.0, 0.0), **attributes):
    return ModelObject(
        "unknown_person", FACE, 0.95,
        {
            "subject_id": subject, "status": "unknown",
            "identity_source": "face_match", "face_observation_state": "observed",
            **attributes,
        },
        identity_embedding=embedding, identity_quality=True,
    )


def samples(review, subject, start, **kwargs):
    for at in (start, start + 0.5):
        review.observe_face(face(subject, **kwargs), at, int(at * 10))


def test_repeated_unknown_faces_merge_across_expired_live_tracks():
    registry = PersonRegistry("s", "camera")
    ids = []
    for at in (0, 0.5, 10, 10.5):
        observed = registry.resolve_faces(
            [face()], timestamp=at, frame_sequence=int(at * 10), people={},
        )[0]
        ids.append(observed.attributes["subject_id"])
        if at == 10:
            assert registry.review_pending_identities() == 0
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]
    assert registry.review_pending_identities() == 1
    assert registry.drain_identity_merges() == ((ids[2], ids[0]),)
    assert registry.drain_identity_merges() == ()
    assert registry.identity_review_stats()["pending_subjects"] == 0


@pytest.mark.parametrize("overrides", [
    {"identity_quality": False},
    {"identity_embedding": ()},
    {"identity_embedding": (float("nan"), 0.0, 1.0)},
])
def test_unusable_private_face_features_do_not_establish_a_merge(overrides):
    review = IdentityReview()
    samples(review, "a", 0)
    for at in (10, 11):
        review.observe_face(replace(face("b"), **overrides), at, int(at * 10))
    assert review.review() == 0
    assert review.drain() == ()


@pytest.mark.parametrize("attributes", [
    {"face_observation_state": "held"},
    {"identity_source": "track_history"},
    {"status": "unusable"},
])
def test_held_or_inherited_identity_does_not_vote(attributes):
    review = IdentityReview()
    samples(review, "a", 0)
    samples(review, "b", 10, **attributes)
    assert review.review() == 0


def test_duplicate_source_frame_cannot_supply_two_identity_votes():
    review = IdentityReview()
    samples(review, "a", 0)
    for _ in range(3):
        review.observe_face(face("b"), 10, 100)
    assert review.review() == 0


def test_one_favorable_sample_is_insufficient_on_either_side():
    review = IdentityReview()
    samples(review, "a", 0)
    review.observe_face(face("b"), 10, 100)
    review.observe_face(face("b", embedding=(0.0, 1.0, 0.0)), 11, 110)
    assert review.review() == 0


def test_coobserved_bodies_cannot_merge_even_with_identical_face_features():
    review = IdentityReview()
    samples(review, "a", 0)
    samples(review, "b", 10)
    review.observe("a", 12)  # Body remains visible while the other face appears.
    assert review.review() == 0


def test_gallery_conflicts_and_ambiguous_candidates_are_rejected():
    review = IdentityReview()
    samples(review, "a", 0, status="known", person_id="alice")
    samples(review, "b", 10, status="known", person_id="bob")
    assert review.review() == 0
    review = IdentityReview()
    samples(review, "a", 0)
    samples(review, "b", 10, embedding=(0.75, 0.6614378, 0))
    samples(review, "c", 20, embedding=(0.75, -0.6614378, 0))
    assert review.review() == 0
    assert review.stats()["ambiguous_matches"] == 3


def test_multiple_matching_fragments_finish_in_one_background_review():
    review = IdentityReview()
    for sid, at in (("a", 0), ("b", 10), ("c", 20)):
        samples(review, sid, at)
    assert review.review() == 2
    assert review.drain() == (("b", "a"), ("c", "a"))


def test_gallery_identity_merges_single_face_fragments_into_native_person():
    review = IdentityReview()
    review.observe("body", 0, sequence=0, bbox=BODY, native=True)
    review.observe("body", 100, sequence=1000, bbox=BODY, native=True)
    review.observe_face(
        replace(
            face("body", status="known", person_id="frank"), identity_quality=False,
        ),
        0, 0,
    )
    for index in range(18):
        review.observe_face(
            replace(
                face(f"face-{index}", status="known", person_id="frank"),
                identity_quality=False,
            ),
            46 + index, 460 + index * 10,
        )
    assert review.review() == 18
    assert {new for _old, new in review.drain()} == {"body"}
    assert len(review.profiles) == 1
    assert review.stats()["known_subjects_merged"] == 18


def test_gallery_identity_prefers_native_person_even_if_face_was_seen_first():
    review = IdentityReview()
    review.observe_face(face("first-face", status="known", person_id="frank"), 0, 0)
    review.observe("body", 1, sequence=10, bbox=BODY, native=True)
    review.observe_face(face("body", status="known", person_id="frank"), 1, 10)
    assert review.review() == 1
    assert review.drain() == (("first-face", "body"),)


def test_inherited_known_fragment_requires_direct_gallery_confirmation():
    review = IdentityReview()
    for subject, at in (("a", 0), ("b", 1)):
        review.observe_face(face(
            subject, status="known", person_id="frank", identity_source="track_history",
        ), at, int(at * 10))
    assert review.review() == 0
    review.observe_face(face("a", status="known", person_id="frank"), 2, 20)
    assert review.review() == 1
    assert review.canonical("b") == "a"


def test_same_display_name_does_not_merge_different_gallery_people():
    review = IdentityReview()
    review.observe_face(face("a", status="known", person_id="a", name="Frank"), 0, 0)
    review.observe_face(face("b", status="known", person_id="b", name="Frank"), 1, 10)
    assert review.review() == 0


def test_invalid_gallery_embedding_cannot_establish_known_merge():
    review = IdentityReview()
    review.observe_face(face("a", status="known", person_id="frank"), 0, 0)
    review.observe_face(replace(
        face("b", status="known", person_id="frank"), identity_embedding=(),
    ), 1, 10)
    assert review.review() == 0


@pytest.mark.parametrize("kind", ["faces", "native_bodies", "face_outside_body"])
def test_same_gallery_identity_preserves_distinct_coobserved_people(kind):
    review = IdentityReview()
    other_box = NormalizedBBox(0.7, 0.1, 0.8, 0.2)
    if kind == "native_bodies":
        review.observe("a", 0, sequence=0, bbox=BODY, native=True)
        review.observe("b", 0, sequence=0, bbox=BODY, native=True)
    elif kind == "face_outside_body":
        review.observe("a", 0, sequence=0, bbox=BODY, native=True)
    review.observe_face(face("a", status="known", person_id="frank"), 0, 0)
    review.observe_face(replace(
        face("b", status="known", person_id="frank"),
        bbox=FACE if kind == "native_bodies" else other_box,
    ), 0, 0)
    assert review.review() == 0
    assert review.drain() == ()


def test_known_identity_conflict_survives_source_frame_eviction_and_rebind():
    review = IdentityReview()
    review.observe_face(face("a", status="known", person_id="frank"), 0, 0)
    review.observe_face(replace(
        face("b", status="known", person_id="frank"),
        bbox=NormalizedBBox(0.7, 0.1, 0.8, 0.2),
    ), 0, 0)
    for index in range(1, 2050):
        review.observe("a", index, sequence=index, bbox=BODY)
    review.rebind("a", "canonical-a")
    assert len(review.sources) == 2048
    assert review.review() == 0


def test_known_registry_review_canonicalizes_later_faces_without_rewriting_native_id():
    registry = PersonRegistry("s", "camera")
    known = face(status="known", person_id="frank", name="Frank")
    first = registry.resolve_faces(
        [known], timestamp=0, frame_sequence=0, people={"frank": "Frank"},
    )[0]
    registry.observe_snapshot(native_snapshot(10, "native"))
    native_id = registry.history()[-1].objects[0].track_id
    second = registry.resolve_faces(
        [known], timestamp=10, frame_sequence=100, people={"frank": "Frank"},
    )[0]
    assert first.attributes["subject_id"] != second.attributes["subject_id"]
    assert registry.review_pending_identities() == 1
    assert registry.canonical_subject_id(first.attributes["subject_id"]) == native_id
    registry.observe_snapshot(native_snapshot(11, "native"))
    assert registry.history()[-1].objects[0].track_id == native_id
    third = registry.resolve_faces(
        [known], timestamp=11, frame_sequence=110, people={"frank": "Frank"},
    )[0]
    assert third.attributes["subject_id"] == native_id


@pytest.mark.parametrize("dirty", [None, set()])
def test_batched_face_scores_equal_pairwise_scores_before_and_after_rebind(dirty):
    review = IdentityReview()
    rng = np.random.default_rng(42)
    for index in range(16):
        subject = f"p{index}"
        dimension = 7 if index < 12 else 3
        for sample_index in range(2 + index % 3):
            at = index * 10 + sample_index
            review.observe_face(
                face(subject, embedding=tuple(rng.normal(size=dimension))),
                at, int(at * 10),
            )
    scores = _ReviewScores(review.profiles, dirty)

    def check():
        for left in review.profiles:
            for right in review.profiles:
                expected = review._score(left, right)
                actual = review._score(left, right, scores=scores)
                if expected is None:
                    assert actual is None
                else:
                    assert actual == pytest.approx(expected, abs=1e-6)

    check()
    review.rebind("p1", "p0")
    scores.update("p0", review.profiles["p0"])
    check()


def test_scene_rejected_merge_does_not_mutate_registry_or_archive_queue():
    review = IdentityReview()
    samples(review, "a", 0)
    samples(review, "b", 10)
    assert review.review(lambda old, new: False) == 0
    assert set(review.profiles) == {"a", "b"}
    assert review.canonical("b") == "b"
    assert review.drain() == ()


def test_same_frame_face_to_body_rebind_preserves_private_review_samples():
    review = IdentityReview()
    samples(review, "face-only", 0)
    review.rebind("face-only", "native-body")
    samples(review, "later-body", 10)
    assert review.review() == 1
    assert review.drain() == (("later-body", "native-body"),)
    assert review.canonical("face-only") == "native-body"


def test_review_is_bounded_and_session_local():
    first, second = IdentityReview(capacity=3), IdentityReview()
    for index in range(6):
        samples(first, f"p{index}", index * 10)
    assert len(first.profiles) == 3
    assert len(first.pending) == 3
    samples(second, "other-session", 100)
    assert second.review() == 0


def native_snapshot(at, native_id):
    sequence = int(at * 10)
    return RichLabelSnapshot("s", sequence, at, (
        RichLabelObject(
            "wire-id", "person", BODY, 0.95,
            {"native_track": {
                "backend": "sam3.1_video", "epoch": "e1", "object_id": native_id,
            }},
            observation_state="observed",
            last_observed_timestamp_seconds=at,
            last_observed_frame_sequence=sequence,
        ),
    ))


def stranger_result(at, observation):
    return AnalysisResult(
        f"face-{at}", "camera", "s", int(at * 10) + 1, "now",
        FrameReference(int(at * 10), at, "now", 640, 480),
        (EventJudgment("stranger", True, 0.95, "face observation", ("face",)),),
        (GroundedObject("face", "unknown_person", observation.bbox,
                        attributes=dict(observation.attributes)),),
        "test", 0,
    )


def test_native_review_consolidates_events_and_continues_with_immutable_track_ids():
    registry = PersonRegistry("s", "camera")
    runtime = SceneRuntime("s", "camera", tracking_mode="batch", event_labels=())

    def commit(at, native_id):
        runtime.observe_frame(b"original-source", at, int(at * 10))
        raw = native_snapshot(at, native_id)

        def canonicalize(snapshots):
            for snapshot in snapshots:
                registry.observe_snapshot(snapshot)
            return tuple(registry.snapshot_at(
                snapshot.timestamp_seconds, snapshot.frame_sequence,
            ) for snapshot in snapshots)

        assert runtime.commit_native_tracking_batch(
            (raw,), completed_timestamp_seconds=at,
            completed_frame_sequence=int(at * 10), canonicalize=canonicalize,
        )
        return registry.history()[-1].objects[0].track_id

    ids = []
    for at, native_id in ((0, "1"), (0.5, "1"), (5, "2"), (5.5, "2")):
        ids.append(commit(at, native_id))
        observation = registry.resolve_faces(
            [face()], timestamp=at, frame_sequence=int(at * 10), people={},
        )[0]
        runtime.observe_result(stranger_result(at, observation))
    assert ids[0] != ids[2]
    assert len(runtime.snapshot()["scene"]["events"]) == 2
    assert registry.review_pending_identities(
        accept_merge=lambda old, new: runtime.apply_identity_merges(((old, new),)) > 0,
    ) == 1
    assert registry.drain_identity_merges() == ((ids[2], ids[0]),)
    assert runtime.snapshot()["stats"]["identity_subjects_merged"] == 1
    assert commit(6, "2") == ids[2]  # Native registry ID remains immutable.
    view = runtime.snapshot()["scene"]
    assert len(view["entities"]) == len(view["events"]) == 1
    assert view["entities"][0]["subject_id"] == ids[0]
    assert view["entities"][0]["native_track"]["object_id"] == "2"
    assert view["events"][0]["subject_ids"] == [ids[0]]
    assert view["events"][0]["positive_observations"] == 4
    assert view["events"][0]["phase"] == "confirmed"
    history = view["entities"][0]["history"]
    assert {record["native_track"]["object_id"] for record in history} == {"1", "2"}


def test_unknown_face_upgrades_unusable_identity_and_close_finishes_review():
    runtime = SceneRuntime("s", "camera", event_labels=())
    for at, status in ((0, "unusable"), (1, "unknown")):
        runtime.scene.observe_person(
            "a", timestamp_seconds=at, frame_sequence=int(at * 10), bbox=BODY,
            identity={"status": status, "identity_source": "face_match"},
        )
    assert runtime.scene.snapshot()["entities"][0]["identity"]["status"] == "unknown"
    runtime.scene.observe_event(
        "stranger", ("a",), timestamp_seconds=1, frame_sequence=10,
        detected=True, evidence_id="one-source",
    )
    runtime.close()
    event = runtime.snapshot()["scene"]["events"][0]
    assert event["review_state"] == "complete"
    assert event["review_outcome"] == "unconfirmed"
    assert event["phase"] == "suspected"
    assert event["negative_observations"] == 0


def test_face_pipeline_background_service_review_rekeys_archived_evidence(tmp_path):
    registry = PersonRegistry("s", "camera")
    scene = SceneRuntime("s", "camera", event_labels=())
    recognizer = FaceRecognitionRuntime(FaceRecognizerConfig(
        gallery_path=tmp_path / "faces.json", model_root=tmp_path / "models",
        provider="CPUExecutionProvider",
    ))
    detected_face = SimpleNamespace(
        bbox=np.asarray((75, 20, 175, 120), dtype=np.float32),
        normed_embedding=np.asarray((1, 0, 0), dtype=np.float32),
        det_score=0.99,
        kps=np.asarray(((105, 50), (145, 50), (125, 80), (108, 105), (142, 105))),
    )
    recognizer._model = SimpleNamespace(get=lambda image: [detected_face])
    encoded_ok, encoded = cv2.imencode(".jpg", np.zeros((180, 320, 3), np.uint8))
    assert encoded_ok
    jpeg = encoded.tobytes()

    def encode_clip(manifest, destination, **kwargs):
        destination.write_bytes(b"encoded-evidence")
        return {"duration_seconds": 3, "fps": 5, "partial": False}

    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path / "events"),
        encoder=encode_clip, clock=lambda: 0,
    )
    host = HarnessRuntime(None)
    session = SimpleNamespace(
        stream_session_id="s", person_registry=registry, archive=archive,
    )
    host._session, host._scene = session, scene
    worker = IdentityReviewWorker(lambda: host._review_session_identities(0, session))
    host._identity_worker = worker
    try:
        assert archive.start_session("s", "camera")
        subjects = []
        for at in (0, 0.5, 10, 10.5):
            sequence = int(at * 10)
            sampled = SampledFrame(
                sample_index=sequence, source_frame_index=sequence,
                timestamp_seconds=at, jpeg_bytes=jpeg, width=320, height=180,
            )
            response = recognizer._classify_window(
                FrameWindow(index=sequence, frames=(sampled,)), tracker=registry,
            )
            assert response.error is None
            observation = response.classification.objects[0]
            assert observation.identity_quality
            assert observation.attributes["status"] == "unknown"
            subjects.append(observation.attributes["subject_id"])
            analysis = stranger_result(at, observation)
            scene.observe_result(analysis)
            assert archive.submit_frame(jpeg, at, sequence, "s")
            assert archive.submit_result(analysis)
        assert archive.flush()
        assert subjects[0] != subjects[2]
        assert len(scene.snapshot()["scene"]["entities"]) == 2
        assert len(archive.list_events()["events"]) == 2
        host._wake_identity_review()
        assert worker.wait_until_idle(3)
        assert worker.stats()["state"] == "idle"
        assert archive.flush()
        snapshot = host.scene_snapshot()
        assert snapshot["identity_review"]["merged_subjects"] == 1
        assert len(snapshot["scene"]["entities"]) == 1
        assert len(snapshot["scene"]["events"]) == 1
        events = archive.list_events()["events"]
        assert len(events) == 1
        assert events[0]["subject_id"] == subjects[0]
        assert archive._subject_aliases[("s", subjects[2])] == subjects[0]
        assert "identity_embedding" not in repr(snapshot)
    finally:
        worker.close(timeout_seconds=3)
        host._session = None
        host.stop()
        archive.close()


def test_known_frank_fragments_consolidate_through_background_service(tmp_path):
    registry = PersonRegistry("s", "camera")
    scene = SceneRuntime("s", "camera", tracking_mode="batch", event_labels=())
    known = replace(
        face(status="known", person_id="frank", name="Frank"),
        identity_quality=False,
    )

    def encode_clip(manifest, destination, **kwargs):
        destination.write_bytes(b"encoded-evidence")
        return {"duration_seconds": 3, "fps": 5, "partial": False}

    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path / "events"), encoder=encode_clip,
    )
    host = HarnessRuntime(None)
    session = SimpleNamespace(
        stream_session_id="s", person_registry=registry, archive=archive,
    )
    host._session, host._scene = session, scene
    worker = IdentityReviewWorker(lambda: host._review_session_identities(0, session))
    host._identity_worker = worker
    try:
        assert archive.start_session("s", "camera")
        subjects = []
        for index in range(19):
            at = index * 4
            raw = native_snapshot(at, "native-frank")
            scene.observe_frame(b"original-source", at, at * 10)

            def canonicalize(snapshots):
                for snapshot in snapshots:
                    registry.observe_snapshot(snapshot)
                return tuple(registry.snapshot_at(
                    snapshot.timestamp_seconds, snapshot.frame_sequence,
                ) for snapshot in snapshots)

            assert scene.commit_native_tracking_batch(
                (raw,), completed_timestamp_seconds=at,
                completed_frame_sequence=at * 10, canonicalize=canonicalize,
            )
            # Later detections are low in the body, as in the reported run;
            # they cannot use the normal exact-source head/body association.
            observation = registry.resolve_faces(
                [known if index == 0 else replace(
                    known, bbox=NormalizedBBox(0.2, 0.7, 0.3, 0.9),
                )],
                timestamp=at, frame_sequence=at * 10, people={"frank": "Frank"},
            )[0]
            subjects.append(observation.attributes["subject_id"])
            analysis = stranger_result(at, observation)
            analysis = replace(
                analysis,
                judgments=(replace(analysis.judgments[0], detected=False),),
                objects=(replace(analysis.objects[0], label="known_person"),),
            )
            scene.observe_result(analysis)
            if index == 1:
                assert scene.scene.observe_event(
                    "smoking", (subjects[0],), timestamp_seconds=at,
                    frame_sequence=at * 10, detected=True, evidence_id="smoking",
                )
        assert len(set(subjects)) == 19
        assert len(scene.snapshot()["scene"]["entities"]) == 19
        host._wake_identity_review()
        assert worker.wait_until_idle(3)
        assert archive.flush()
        snapshot = host.scene_snapshot()
        assert snapshot["identity_review"]["known_subjects_merged"] == 18, (
            snapshot["identity_review"], snapshot["scene"]["entities"],
            registry._identity_review.conflicts,
        )
        entity, = snapshot["scene"]["entities"]
        assert entity["subject_id"] == subjects[0]
        assert entity["identity"]["name"] == "Frank"
        assert entity["bbox"] == BODY.as_dict()
        assert entity["native_track"]["object_id"] == "native-frank"
        event, = snapshot["scene"]["events"]
        assert event["event_label"] == "smoking"
        assert event["subject_ids"] == [subjects[0]]
        assert all(
            registry.canonical_subject_id(sid) == subjects[0] for sid in subjects
        )
        assert all(
            archive._subject_aliases[("s", sid)] == subjects[0] for sid in subjects[1:]
        )
    finally:
        worker.close(timeout_seconds=3)
        host._session = None
        host.stop()
        archive.close()
