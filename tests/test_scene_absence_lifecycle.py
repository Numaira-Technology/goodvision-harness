"""Successful tracking-source presence closes safety episodes independently of VLM."""

import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.scene_state import SceneState

BOX = NormalizedBBox(0.1, 0.1, 0.4, 0.9)


def source(scene, sequence, subjects=()):
    """One completed SAM source, including an authoritative empty result."""
    timestamp = sequence / 10
    for subject in subjects:
        assert scene.observe_person(
            subject,
            timestamp_seconds=timestamp,
            frame_sequence=sequence,
            bbox=BOX,
        )
    assert scene.observe_tracking_presence(
        subjects, timestamp_seconds=timestamp, frame_sequence=sequence,
    )


def verdict(scene, sequence, *, label="smoking", subjects=("a",), detected=True,
            evidence_id=None):
    return scene.observe_event(
        label,
        subjects,
        timestamp_seconds=sequence / 10,
        frame_sequence=sequence,
        detected=detected,
        evidence_id=evidence_id or f"{label}-{sequence}",
        confidence=0.9,
        allow_historical=True,
    )


def event(scene, *, label="smoking", subjects=("a",)):
    matches = [
        item for item in scene.snapshot(include_history=False)["events"]
        if item["event_label"] == label
        and item["subject_ids"] == sorted(subjects)
    ]
    assert len(matches) == 1
    return matches[0]


def confirm(scene, *, label="smoking", subjects=("a",)):
    for sequence in (1, 2):
        source(scene, sequence, subjects)
        assert verdict(scene, sequence, label=label, subjects=subjects)
    result = event(scene, label=label, subjects=subjects)
    assert result["phase"] == "confirmed"
    return result["event_id"]


@pytest.mark.parametrize("label", ["smoking", "fight", "fall", "weapon"])
def test_tenth_successful_absent_source_ends_each_safety_label_without_vlm_clear(label):
    scene = SceneState("run", "camera")
    subjects = ("a", "b") if label == "fight" else ("a",)
    original = confirm(scene, label=label, subjects=subjects)
    for sequence in range(3, 12):
        source(scene, sequence)
    assert event(scene, label=label, subjects=subjects)["phase"] == "confirmed"

    source(scene, 12)
    ended = event(scene, label=label, subjects=subjects)
    assert ended["event_id"] == original
    assert ended["phase"] == "ended"
    assert ended["negative_observations"] == 0
    assert ended["last_positive_timestamp_seconds"] == pytest.approx(0.2)
    assert ended["ended_since_timestamp_seconds"] == pytest.approx(1.2)
    assert ended["ended_reason"] == "all_subjects_absent"
    for sequence in range(13, 17):
        source(scene, sequence)
    assert event(scene, label=label, subjects=subjects)[
        "ended_since_timestamp_seconds"
    ] == pytest.approx(1.2)
    ends = [
        item for item in scene.snapshot()["transitions"]
        if item["event_id"] == original and item["type"] == "end"
    ]
    assert len(ends) == 1
    assert ends[0]["frame_sequence"] == 12


def test_absence_threshold_is_independent_of_clear_observations():
    scene = SceneState("run", "camera", absence_observations=3)
    assert scene.clear_observations == 2
    confirm(scene)
    source(scene, 3)
    source(scene, 4)
    assert event(scene)["phase"] == "confirmed"
    source(scene, 5)
    assert event(scene)["phase"] == "ended"
    assert event(scene)["negative_observations"] == 0


def test_any_group_member_reappearance_resets_the_whole_absence_streak():
    scene = SceneState("run", "camera")
    subjects = ("a", "b")
    confirm(scene, label="fight", subjects=subjects)
    for start, returning in ((3, "a"), (13, "b")):
        for sequence in range(start, start + 9):
            source(scene, sequence)
        source(scene, start + 9, (returning,))
        assert event(scene, label="fight", subjects=subjects)["phase"] == "confirmed"
    for sequence in range(23, 32):
        source(scene, sequence)
    assert event(scene, label="fight", subjects=subjects)["phase"] == "confirmed"
    source(scene, 32)
    assert event(scene, label="fight", subjects=subjects)["phase"] == "ended"


def test_alternating_visible_group_members_never_counts_as_all_people_absent():
    scene = SceneState("run", "camera")
    subjects = ("a", "b")
    confirm(scene, label="fight", subjects=subjects)
    for sequence in range(3, 24):
        source(scene, sequence, ("a" if sequence % 2 else "b",))
    assert event(scene, label="fight", subjects=subjects)["phase"] == "confirmed"
    assert event(scene, label="fight", subjects=subjects)["negative_observations"] == 0


def test_unrelated_visible_person_does_not_reset_event_participants_absence():
    scene = SceneState("run", "camera")
    confirm(scene)
    for sequence in range(3, 13):
        source(scene, sequence, ("someone-else",))
    assert event(scene)["phase"] == "ended"


def test_source_gaps_count_processed_frames_instead_of_missing_sequence_numbers():
    scene = SceneState("run", "camera")
    confirm(scene)
    for sequence in range(10, 100, 10):
        source(scene, sequence)
    assert event(scene)["phase"] == "confirmed"
    source(scene, 100)
    assert event(scene)["phase"] == "ended"
    assert event(scene)["ended_since_timestamp_seconds"] == pytest.approx(10.0)


def test_duplicate_out_of_order_and_backward_time_sources_do_not_add_absence():
    scene = SceneState("run", "camera")
    confirm(scene)
    for sequence in range(3, 11):
        source(scene, sequence)
    for timestamp, sequence in ((1.0, 10), (0.9, 9), (0.4, 11)):
        assert not scene.observe_tracking_presence(
            (), timestamp_seconds=timestamp, frame_sequence=sequence,
        )
    assert not scene.observe_tracking_presence(
        ("a",), timestamp_seconds=1.0, frame_sequence=10,
    )
    source(scene, 11)
    assert event(scene)["phase"] == "confirmed"
    source(scene, 12)
    assert event(scene)["phase"] == "ended"


def test_clock_advance_and_reading_display_state_are_not_successful_sam_frames():
    scene = SceneState("run", "camera")
    original = confirm(scene)
    for timestamp in range(1, 20):
        scene.advance(timestamp)
        scene.snapshot()
    current = event(scene)
    assert current["event_id"] == original
    assert current["phase"] == "confirmed"
    assert current["negative_observations"] == 0
    assert current["ended_since_timestamp_seconds"] is None


def test_absence_neither_creates_events_nor_changes_stranger_lifecycle():
    empty = SceneState("empty-run", "camera")
    for sequence in range(1, 15):
        source(empty, sequence)
    assert empty.snapshot()["events"] == []

    scene = SceneState("run", "camera")
    confirm(scene, label="stranger")
    for sequence in range(3, 23):
        source(scene, sequence)
    current = event(scene, label="stranger")
    assert current["phase"] == "confirmed"
    assert current["negative_observations"] == 0


def test_clear_and_absence_are_independent_or_conditions_not_added_together():
    scene = SceneState("run", "camera")
    confirm(scene)
    source(scene, 3, ("a",))
    assert verdict(scene, 3, detected=False)
    for sequence in range(4, 13):
        source(scene, sequence)
    assert event(scene)["phase"] == "confirmed"
    assert event(scene)["negative_observations"] == 1
    source(scene, 13)
    assert event(scene)["phase"] == "ended"
    assert event(scene)["negative_observations"] == 1


def test_late_first_positive_replays_already_completed_absence_before_publishing():
    scene = SceneState("run", "camera")
    source(scene, 1, ("a",))
    source(scene, 2, ("a",))
    for sequence in range(3, 13):
        source(scene, sequence)
    assert scene.snapshot()["events"] == []
    assert verdict(scene, 1)
    ended = event(scene)
    assert ended["phase"] == "ended"
    assert ended["ended_since_timestamp_seconds"] == pytest.approx(1.2)
    assert verdict(scene, 2)
    updated = event(scene)
    assert updated["event_id"] == ended["event_id"]
    assert updated["phase"] == "ended"
    assert updated["negative_observations"] == 0
    assert updated["last_positive_timestamp_seconds"] == pytest.approx(0.2)
    assert updated["ended_since_timestamp_seconds"] == pytest.approx(1.2)


def test_late_confirmation_and_duplicate_source_verdict_do_not_revive_episode():
    scene = SceneState("run", "camera")
    source(scene, 1, ("a",))
    assert verdict(scene, 1)
    original = event(scene)["event_id"]
    source(scene, 2, ("a",))
    for sequence in range(3, 13):
        source(scene, sequence)
    assert event(scene)["phase"] == "ended"
    assert verdict(scene, 2)
    assert verdict(scene, 1, evidence_id="different-request-same-source")
    current = event(scene)
    assert current["event_id"] == original
    assert current["phase"] == "ended"
    assert current["positive_observations"] == 2
    assert current["ended_since_timestamp_seconds"] == pytest.approx(1.2)


def test_reappearance_needs_new_positive_episode_and_two_new_confirmation_sources():
    scene = SceneState("run", "camera")
    original = confirm(scene)
    for sequence in range(3, 13):
        source(scene, sequence)
    source(scene, 13, ("a",))
    assert event(scene)["phase"] == "ended"
    assert verdict(scene, 13)
    restarted = event(scene)
    assert restarted["event_id"] != original
    assert restarted["phase"] == "suspected"
    assert restarted["positive_observations"] == 1
    assert restarted["ended_since_timestamp_seconds"] is None
    source(scene, 14, ("a",))
    assert verdict(scene, 14)
    assert event(scene)["phase"] == "confirmed"
    assert verdict(scene, 2, evidence_id="old-request-late-after-reappearance")
    current = event(scene)
    assert current["event_id"] == restarted["event_id"]
    assert current["phase"] == "confirmed"
    assert current["positive_observations"] == 2


def test_vlm_ledger_pruning_preserves_absence_termination_during_late_replay():
    scene = SceneState("run", "camera", history_size=2, person_history_size=64)
    original = confirm(scene)
    for sequence in range(3, 7):
        source(scene, sequence, ("a",))
        assert verdict(scene, sequence)
    for sequence in range(7, 17):
        source(scene, sequence)
    assert event(scene)["phase"] == "ended"
    assert verdict(scene, 6, evidence_id="retained-source-second-request")
    current = event(scene)
    assert current["event_id"] == original
    assert current["phase"] == "ended"
    assert current["ended_since_timestamp_seconds"] == pytest.approx(1.6)
    assert current["negative_observations"] == 0


def test_presence_pruning_preserves_seed_and_rejects_old_first_event_for_new_label():
    scene = SceneState("run", "camera", person_history_size=4)
    original = confirm(scene)
    for sequence in range(3, 25):
        source(scene, sequence)
    current = event(scene)
    assert current["event_id"] == original
    assert current["phase"] == "ended"
    assert current["ended_since_timestamp_seconds"] == pytest.approx(1.2)
    # Person geometry remains grounded: only the missing presence prefix makes
    # accepting a new, old-source event unsafe after the replay floor advances.
    person = next(
        item for item in scene.snapshot()["entities"] if item["subject_id"] == "a"
    )
    assert any(item["frame_sequence"] == 1 for item in person["history"])
    assert not verdict(scene, 1, label="weapon")
    assert not verdict(scene, 1, evidence_id="pruned-old-smoking-request")
    assert event(scene)["phase"] == "ended"


def test_pinned_pending_vlm_keeps_presence_prefix_until_its_late_verdict_completes():
    scene = SceneState("run", "camera", person_history_size=3,
                       retain_pinned_sources=True)
    source(scene, 1, ("a",))
    source(scene, 2, ("a",))
    assert scene.pin_source_observations(
        "accepted-vlm", (("a", 0.1, 1, BOX), ("a", 0.2, 2, BOX)),
    )
    for sequence in range(3, 21):
        source(scene, sequence)
    assert verdict(scene, 1)
    assert verdict(scene, 2)
    current = event(scene)
    assert current["phase"] == "ended"
    assert current["ended_since_timestamp_seconds"] == pytest.approx(1.2)
    assert current["negative_observations"] == 0
    scene.unpin_source_observations("accepted-vlm")
    source(scene, 21)
    assert scene.snapshot()["source_pins"]["jobs"] == 0
    assert not verdict(scene, 1, label="weapon")
    assert event(scene)["phase"] == "ended"


@pytest.mark.parametrize("person_history_size", [4, 64])
def test_subject_merge_does_not_revive_absence_ended_episode(person_history_size):
    scene = SceneState("run", "camera", person_history_size=person_history_size)
    original = confirm(scene)
    for sequence in range(3, 17):
        source(scene, sequence)
    assert scene.merge_subject("a", "canonical", reviewed_identity=True)
    ended = event(scene, subjects=("canonical",))
    assert ended["event_id"] == original
    assert ended["phase"] == "ended"
    assert ended["ended_since_timestamp_seconds"] == pytest.approx(1.2)
    source(scene, 17, ("canonical",))
    assert verdict(scene, 17, subjects=("canonical",))
    restarted = event(scene, subjects=("canonical",))
    assert restarted["event_id"] != original
    assert restarted["phase"] == "suspected"


def test_subject_merge_preserves_observed_presence_when_replaying_current_episode():
    scene = SceneState("run", "camera")
    original = confirm(scene)
    for sequence in range(3, 12):
        source(scene, sequence, ("a",))
    for sequence in range(12, 21):
        source(scene, sequence)
    assert scene.merge_subject("a", "canonical", reviewed_identity=True)
    current = event(scene, subjects=("canonical",))
    assert current["event_id"] == original
    assert current["phase"] == "confirmed"
    source(scene, 21, ("canonical",))
    for sequence in range(22, 31):
        source(scene, sequence)
    assert event(scene, subjects=("canonical",))["phase"] == "confirmed"
    source(scene, 31)
    assert event(scene, subjects=("canonical",))["phase"] == "ended"


@pytest.mark.parametrize("person_history_size", [4, 64])
def test_merging_two_existing_ended_ledgers_does_not_reopen_either_episode(
    person_history_size,
):
    scene = SceneState("run", "camera", person_history_size=person_history_size)
    for sequence in (1, 2):
        source(scene, sequence, ("a", "b"))
        for subject in ("a", "b"):
            assert verdict(
                scene, sequence, subjects=(subject,),
                evidence_id=f"{subject}-{sequence}",
            )
    original_ids = {
        event(scene, subjects=(subject,))["event_id"] for subject in ("a", "b")
    }
    for sequence in range(3, 17):
        source(scene, sequence)
    assert all(item["phase"] == "ended" for item in scene.snapshot()["events"])
    assert scene.merge_subject("a", "b")
    current = event(scene, subjects=("b",))
    assert current["phase"] == "ended"
    assert current["ended_since_timestamp_seconds"] == pytest.approx(1.2)
    assert current["negative_observations"] == 0
    source(scene, 17, ("b",))
    assert verdict(scene, 17, subjects=("b",))
    restarted = event(scene, subjects=("b",))
    assert restarted["phase"] == "suspected"
    assert restarted["event_id"] not in original_ids
