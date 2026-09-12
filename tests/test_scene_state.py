"""Lifecycle and hostile-input checks for the persistent scene core."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.scene_state import SceneState

BOX = NormalizedBBox(0.1, 0.1, 0.4, 0.9)


def person(scene, timestamp, sequence, subject="a", **kwargs):
    return scene.observe_person(
        subject,
        timestamp_seconds=timestamp,
        frame_sequence=sequence,
        bbox=BOX,
        **kwargs,
    )


def evidence(
    scene,
    timestamp,
    sequence,
    *,
    detected=True,
    subjects=("a",),
    label="smoking",
    evidence_id=None,
):
    return scene.observe_event(
        label,
        subjects,
        timestamp_seconds=timestamp,
        frame_sequence=sequence,
        detected=detected,
        evidence_id=evidence_id or f"e-{sequence}",
        confidence=0.9,
    )


def test_episode_survives_occlusion_and_requires_observed_negative_evidence():
    scene = SceneState("session", "camera")
    person(scene, 1, 1, identity={"status": "known", "person_id": "alice"})
    assert evidence(scene, 1, 1)
    assert scene.snapshot()["events"][0]["phase"] == "suspected"
    person(scene, 2, 2)
    assert evidence(scene, 2, 2)
    original_id = scene.snapshot()["events"][0]["event_id"]
    scene.advance(7)
    lost = scene.snapshot()
    assert lost["entities"][0]["observation_state"] == "occluded"
    assert lost["entities"][0]["identity"]["person_id"] == "alice"
    assert lost["events"][0]["phase"] == "confirmed"
    assert lost["events"][0]["observation_state"] == "uncertain"
    assert not evidence(scene, 7, 7, detected=False)
    person(scene, 8, 8)
    assert evidence(scene, 8, 8, detected=False)
    assert scene.snapshot()["events"][0]["phase"] == "confirmed"
    person(scene, 9, 9)
    assert evidence(scene, 9, 9, detected=False)
    assert scene.snapshot()["events"][0]["phase"] == "ended"
    assert not evidence(scene, 2, 2, evidence_id="late-positive")
    assert scene.snapshot()["events"][0]["event_id"] == original_id
    person(scene, 10, 10)
    evidence(scene, 10, 10)
    restarted = scene.snapshot()["events"][0]
    assert restarted["phase"] == "suspected"
    assert restarted["event_id"] != original_id
    assert [item["type"] for item in scene.snapshot()["transitions"]] == [
        "start",
        "update",
        "update",
        "end",
        "start",
    ]


def test_duplicate_evidence_and_same_source_do_not_confirm_or_transfer_event():
    scene = SceneState("session", "camera")
    person(scene, 1, 1)
    person(scene, 1, 1, "b")
    evidence(scene, 1, 1, evidence_id="original")
    assert not evidence(scene, 1, 1, evidence_id="another-model-same-frame")
    person(scene, 2, 2)
    assert not evidence(scene, 2, 2, evidence_id="original")
    assert not evidence(scene, 1, 1, subjects=("b",), evidence_id="original")
    events = scene.snapshot()["events"]
    assert len(events) == 1
    assert events[0]["positive_observations"] == 1
    assert events[0]["phase"] == "suspected"
    assert scene.snapshot()["counters"]["reassigned_evidence_ignored"] == 1


def test_predictions_do_not_create_people_or_extend_visibility_or_clear_events():
    scene = SceneState("session", "camera")
    assert not person(scene, 0, 0, observation_state="predicted")
    assert not scene.snapshot()["entities"]
    person(scene, 1, 1)
    evidence(scene, 1, 1)
    assert not person(scene, 5, 5, observation_state="held")
    state = scene.snapshot()
    assert state["entities"][0]["last_observed_timestamp_seconds"] == 1
    assert state["entities"][0]["observation_state"] == "occluded"
    assert not evidence(scene, 5, 5, detected=False)
    assert scene.snapshot()["events"][0]["phase"] == "suspected"


def test_participant_sets_are_canonical_and_separate_from_single_person_events():
    scene = SceneState("session", "camera")
    for sequence in (1, 2):
        for subject in ("a", "b", "c"):
            person(scene, sequence, sequence, subject)
        order = ("b", "a") if sequence == 1 else ("a", "b")
        evidence(scene, sequence, sequence, subjects=order, label="fight")
    evidence(scene, 2, 2, subjects=("c",), evidence_id="c-smoking")
    events = scene.snapshot()["events"]
    assert [(event["subject_ids"], event["phase"]) for event in events] == [
        (["a", "b"], "confirmed"),
        (["c"], "suspected"),
    ]
    person(scene, 3, 3, "a")
    assert not evidence(scene, 3, 3, subjects=("a", "b"), label="fight", detected=False)
    assert scene.snapshot()["events"][0]["negative_observations"] == 0


def test_late_negative_can_use_matching_history_but_cannot_regress_newer_result():
    scene = SceneState("session", "camera")
    for sequence in range(1, 6):
        person(scene, sequence, sequence)
    assert evidence(scene, 1, 1)
    assert evidence(scene, 3, 3)
    assert not evidence(scene, 2, 2, detected=False)
    assert evidence(scene, 4, 4, detected=False)
    assert scene.snapshot()["events"][0]["phase"] == "confirmed"
    # Matching frame number alone cannot substitute a different source timestamp.
    assert not evidence(scene, 5.61, 5, detected=False)
    assert evidence(scene, 5, 5, detected=False)
    assert scene.snapshot()["events"][0]["phase"] == "ended"


def test_known_identity_is_public_only_and_forget_is_immediate():
    scene = SceneState("session", "camera")
    person(
        scene,
        1,
        1,
        identity={
            "status": "known",
            "person_id": "alice",
            "name": "Alice",
            "embedding": [1.0, 2.0],
            "image": "sensitive-face-data",
        },
    )
    person(scene, 2, 2, identity={"status": "unknown"})
    state = scene.snapshot()
    assert state["entities"][0]["identity"] == {
        "status": "known",
        "person_id": "alice",
        "name": "Alice",
    }
    state["entities"][0]["identity"]["name"] = "mutated externally"
    assert scene.snapshot()["entities"][0]["identity"]["name"] == "Alice"
    scene.forget_identity("alice")
    assert not scene.snapshot()["entities"][0]["identity"]
    json.dumps(scene.snapshot(), allow_nan=False)


def test_duplicate_person_source_allows_identity_enrichment_but_not_face_bbox():
    scene = SceneState("session", "camera")
    person(scene, 1, 1)
    face = NormalizedBBox(0.2, 0.2, 0.3, 0.3)
    assert scene.observe_person(
        "a",
        timestamp_seconds=1,
        frame_sequence=1,
        bbox=face,
        identity={"status": "known", "person_id": "alice"},
    )
    stored = scene.snapshot()["entities"][0]
    assert stored["bbox"] == BOX.as_dict()
    assert stored["identity"]["person_id"] == "alice"
    assert len(stored["history"]) == 1


def test_changed_identity_does_not_mix_old_name_and_new_person_id():
    scene = SceneState("session", "camera")
    person(
        scene, 1, 1, identity={"status": "known", "person_id": "alice", "name": "Alice"}
    )
    person(scene, 2, 2, identity={"status": "known", "person_id": "bob", "name": None})
    assert scene.snapshot()["entities"][0]["identity"] == {
        "status": "known",
        "person_id": "bob",
    }


def test_delayed_identity_enriches_known_source_history_without_rewinding_person():
    scene = SceneState("session", "camera")
    person(scene, 1, 1)
    moved_box = NormalizedBBox(0.5, 0.1, 0.8, 0.9)
    scene.observe_person("a", timestamp_seconds=2, frame_sequence=2, bbox=moved_box)
    assert person(scene, 1, 1, identity={"status": "known", "person_id": "alice"})
    current = scene.snapshot()["entities"][0]
    assert current["identity"]["person_id"] == "alice"
    assert current["bbox"] == moved_box.as_dict()
    assert current["last_observed_timestamp_seconds"] == 2
    assert current["frame_sequence"] == 2
    assert len(current["history"]) == 2
    assert person(scene, 2, 2, identity={"status": "known", "person_id": "bob"})
    assert not person(scene, 1, 1, identity={"status": "known", "person_id": "alice"})
    assert scene.snapshot()["entities"][0]["identity"]["person_id"] == "bob"


def test_delayed_identity_requires_retained_matching_source_and_cannot_refresh_loss():
    scene = SceneState("session", "camera", history_size=2)
    person(scene, 1, 1)
    person(scene, 2, 2)
    person(scene, 3, 3)
    assert not person(scene, 1, 1, identity={"status": "known", "person_id": "old"})
    scene.advance(8)
    assert person(scene, 2, 2, identity={"status": "known", "person_id": "alice"})
    entity = scene.snapshot()["entities"][0]
    assert entity["last_observed_timestamp_seconds"] == 3
    assert entity["observation_state"] == "occluded"
    assert not person(scene, 2, 2, identity={"status": "unknown"})
    assert scene.snapshot()["entities"][0]["identity"]["person_id"] == "alice"


def test_lightweight_snapshot_omits_histories_but_keeps_safe_current_state():
    scene = SceneState("session", "camera")
    person(scene, 1, 1, identity={"status": "known", "person_id": "alice"})
    evidence(scene, 1, 1)
    compact = scene.snapshot(include_history=False)
    assert not compact["transitions"]
    assert "history" not in compact["entities"][0]
    assert "history" not in compact["events"][0]
    assert compact["events"][0]["phase"] == "suspected"
    compact["entities"][0]["identity"]["person_id"] = "changed"
    assert scene.snapshot()["entities"][0]["identity"]["person_id"] == "alice"
    assert scene.snapshot()["transitions"]


def test_unknown_event_participants_cannot_advance_time_and_expire_scene():
    scene = SceneState("session", "camera")
    person(scene, 1, 1)
    evidence(scene, 1, 1)
    assert not evidence(scene, 1000, 1000, subjects=("not-observed",))
    assert scene.snapshot()["timestamp_seconds"] == 1
    assert scene.snapshot()["events"][0]["phase"] == "suspected"


def test_histories_entities_events_and_evidence_cache_remain_bounded():
    scene = SceneState("session", "camera", max_entities=4, history_size=3)
    for subject in "abcd":
        assert person(scene, 0, 0, subject)
    assert not person(scene, 0, 0, "e")
    # 5 labels * 4 entities fills the active-event budget with distinct keys.
    for label in ("smoking", "fight", "fall", "weapon", "stranger"):
        for subject in "abcd":
            assert evidence(
                scene,
                0,
                0,
                label=label,
                subjects=(subject,),
                evidence_id=f"{label}-{subject}",
            )
    assert not evidence(
        scene, 0, 0, label="fight", subjects=("a", "b"), evidence_id="additional-group"
    )
    for sequence in range(1, 40):
        person(scene, sequence, sequence)
        evidence(scene, sequence, sequence)
    state = scene.snapshot()
    assert len(state["entities"]) <= 4
    assert len(state["events"]) == 20
    assert len(state["transitions"]) == 3
    assert all(
        len(item["history"]) <= 3 for item in state["entities"] + state["events"]
    )
    assert len(scene._evidence) <= scene.max_events * scene.history_size
    scene.advance(200)
    assert not scene.snapshot()["events"]
    assert not scene.snapshot()["entities"]
    assert not any(item["type"] == "end" for item in scene.snapshot()["transitions"])


def test_no_event_negative_still_blocks_older_positive():
    scene = SceneState("session", "camera")
    person(scene, 1, 1)
    person(scene, 2, 2)
    assert evidence(scene, 2, 2, detected=False)
    assert not evidence(scene, 1, 1)
    assert not scene.snapshot()["events"]


def test_concurrent_duplicate_replies_count_once():
    scene = SceneState("session", "camera")
    person(scene, 1, 1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(lambda _: evidence(scene, 1, 1), range(30)))
    assert sum(accepted) == 1
    assert scene.snapshot()["events"][0]["phase"] == "suspected"


@pytest.mark.parametrize(
    "options",
    [
        {"max_entities": True},
        {"max_entities": 0},
        {"history_size": 1.5},
        {"history_size": 0},
        {"occlusion_seconds": float("nan")},
        {"occlusion_seconds": 0},
        {"retention_seconds": 2},
        {"clear_observations": True},
        {"confirm_observations": 0},
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        SceneState("session", "camera", **options)


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -1, True, "1"])
def test_invalid_timestamps_are_rejected(timestamp):
    scene = SceneState("session", "camera")
    with pytest.raises(ValueError):
        person(scene, timestamp, 1)
    with pytest.raises(ValueError):
        scene.advance(timestamp)


@pytest.mark.parametrize(
    "extra",
    [
        {"frame_sequence": True},
        {"frame_sequence": 1.5},
        {"frame_sequence": -1},
        {"confidence": float("nan")},
        {"confidence": 1.1},
        {"confidence": True},
        {"subject_ids": ()},
        {"subject_ids": "a"},
        {"subject_ids": ("",)},
        {"event_label": "unknown"},
        {"detected": 1},
        {"evidence_id": ""},
    ],
)
def test_invalid_event_arguments_are_rejected(extra):
    scene = SceneState("session", "camera")
    values = dict(
        event_label="smoking",
        subject_ids=("a",),
        timestamp_seconds=1,
        frame_sequence=1,
        detected=True,
        evidence_id="e",
        confidence=0.9,
    )
    values.update(extra)
    with pytest.raises(ValueError):
        scene.observe_event(**values)
