from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.face_tracking import FaceIdentityTracker
from goodvision_harness.grounding import ModelObject
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot

PEOPLE = {"alice": "Alice", "bob": "Bob"}
FACE = NormalizedBBox(0.20, 0.10, 0.30, 0.20)
FAR_FACE = NormalizedBBox(0.60, 0.10, 0.70, 0.20)
BODY = NormalizedBBox(0.10, 0.05, 0.80, 0.95)


def _face(
    status: str,
    bbox: NormalizedBBox = FACE,
    *,
    person_id: str = "alice",
) -> ModelObject:
    attributes: dict[str, object] = {"status": status}
    if status == "known":
        attributes.update(person_id=person_id, name=PEOPLE[person_id])
    elif status == "unusable":
        attributes["reason"] = "face_too_small"
    return ModelObject(
        label={
            "known": "known_person",
            "unknown": "unknown_person",
            "unusable": "unusable_face",
        }[status],
        bbox=bbox,
        confidence=0.9,
        attributes=attributes,
    )


def _snapshot(
    timestamp: float,
    sequence: int,
    bodies: Sequence[tuple[str, NormalizedBBox]] = (("P0001", BODY),),
) -> RichLabelSnapshot:
    return RichLabelSnapshot(
        stream_session_id="test-session",
        timestamp_seconds=timestamp,
        frame_sequence=sequence,
        objects=tuple(
            RichLabelObject(track_id=track_id, label="person", bbox=bbox)
            for track_id, bbox in bodies
        ),
    )


def _resolve(
    tracker: FaceIdentityTracker,
    objects: Sequence[ModelObject],
    timestamp: float,
    sequence: int,
    *,
    people: Mapping[str, str] = PEOPLE,
    snapshot: RichLabelSnapshot | None = None,
) -> tuple[ModelObject, ...]:
    return tracker.resolve(
        objects,
        timestamp=timestamp,
        frame_sequence=sequence,
        people=people,
        snapshot=snapshot,
    )


def test_known_identity_survives_poor_face_without_body_tracking() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known")], 0.0, 0)

    observed = _resolve(tracker, [_face("unusable")], 0.4, 1)[0]

    assert observed.label == "known_person"
    assert observed.attributes["status"] == "known"
    assert observed.attributes["person_id"] == "alice"
    assert observed.attributes["identity_source"] == "track_history"
    assert observed.attributes["identity_age_seconds"] == pytest.approx(0.4)


def test_repeated_observed_faces_preserve_known_identity_without_time_expiry() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known")], 0.0, 0)
    for sequence in range(1, 21):
        observed = _resolve(
            tracker,
            [_face("unusable")],
            sequence * 0.4,
            sequence,
        )[0]
        assert observed.attributes["status"] == "known"

    expired = _resolve(tracker, [_face("unusable")], 8.4, 21)[0]

    assert expired.attributes["status"] == "known"
    assert expired.attributes["person_id"] == "alice"


def test_body_continuity_allows_face_to_move_after_brief_back_view() -> None:
    tracker = FaceIdentityTracker()
    _resolve(
        tracker,
        [_face("known")],
        0.0,
        0,
        snapshot=_snapshot(0.0, 0),
    )
    for sequence in range(1, 6):
        timestamp = sequence * 0.4
        assert (
            _resolve(
                tracker,
                [],
                timestamp,
                sequence,
                snapshot=_snapshot(timestamp, sequence),
            )
            == ()
        )

    observed = _resolve(
        tracker,
        [_face("unusable", FAR_FACE)],
        2.4,
        6,
        snapshot=_snapshot(2.4, 6),
    )[0]

    assert observed.attributes["status"] == "known"
    assert observed.attributes["person_id"] == "alice"
    assert observed.attributes["person_track_id"] == "P0001"


def test_no_body_and_long_face_absence_require_new_identity_evidence() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known")], 0.0, 0)

    observed = _resolve(tracker, [_face("unusable")], 3.1, 1)[0]

    assert observed.attributes["status"] == "unusable"
    assert "person_id" not in observed.attributes


def test_brief_body_disappearance_preserves_identity_when_track_returns() -> None:
    tracker = FaceIdentityTracker()
    _resolve(
        tracker,
        [_face("known")],
        0.0,
        0,
        snapshot=_snapshot(0.0, 0),
    )
    _resolve(tracker, [], 0.4, 1, snapshot=_snapshot(0.4, 1, ()))

    observed = _resolve(
        tracker,
        [_face("unusable")],
        0.8,
        2,
        snapshot=_snapshot(0.8, 2),
    )[0]

    assert observed.attributes["status"] == "known"
    assert observed.attributes["person_id"] == "alice"


def test_changed_body_track_does_not_inherit_overlapping_face_identity() -> None:
    tracker = FaceIdentityTracker()
    _resolve(
        tracker,
        [_face("known")],
        0.0,
        0,
        snapshot=_snapshot(0.0, 0),
    )

    observed = _resolve(
        tracker,
        [_face("unusable")],
        0.4,
        1,
        snapshot=_snapshot(0.4, 1, (("P0002", BODY),)),
    )[0]

    assert observed.attributes["status"] == "unusable"
    assert "person_id" not in observed.attributes


def test_same_body_id_with_discontinuous_geometry_does_not_inherit() -> None:
    tracker = FaceIdentityTracker()
    left_body = NormalizedBBox(0.10, 0.05, 0.40, 0.95)
    right_body = NormalizedBBox(0.55, 0.05, 0.90, 0.95)
    _resolve(
        tracker,
        [_face("known")],
        0.0,
        0,
        snapshot=_snapshot(0.0, 0, (("P0001", left_body),)),
    )

    observed = _resolve(
        tracker,
        [_face("unusable", FAR_FACE)],
        0.4,
        1,
        snapshot=_snapshot(0.4, 1, (("P0001", right_body),)),
    )[0]

    assert observed.attributes["status"] == "unusable"
    assert "person_id" not in observed.attributes


@pytest.mark.parametrize(
    ("snapshot_timestamp", "snapshot_sequence"),
    [(0.0, 0), (1.1, 1), (1.0, 2)],
)
def test_stale_or_future_body_snapshot_cannot_supply_continuity(
    snapshot_timestamp: float,
    snapshot_sequence: int,
) -> None:
    tracker = FaceIdentityTracker()
    _resolve(
        tracker,
        [_face("known")],
        0.0,
        0,
        snapshot=_snapshot(0.0, 0),
    )

    observed = _resolve(
        tracker,
        [_face("unusable", FAR_FACE)],
        1.0,
        1,
        snapshot=_snapshot(snapshot_timestamp, snapshot_sequence),
    )[0]

    assert observed.attributes["status"] == "unusable"
    assert "person_id" not in observed.attributes


def test_overlapping_bodies_make_face_identity_association_ambiguous() -> None:
    tracker = FaceIdentityTracker()
    _resolve(
        tracker,
        [_face("known")],
        0.0,
        0,
        snapshot=_snapshot(0.0, 0),
    )

    observed = _resolve(
        tracker,
        [_face("unusable")],
        0.4,
        1,
        snapshot=_snapshot(0.4, 1, (("P0001", BODY), ("P0002", BODY))),
    )[0]

    assert observed.attributes["status"] == "unusable"
    assert "person_id" not in observed.attributes


def test_two_faces_cannot_inherit_the_same_known_identity() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known")], 0.0, 0)
    nearby_face = NormalizedBBox(0.21, 0.10, 0.31, 0.20)

    observed = _resolve(
        tracker,
        [_face("unusable"), _face("unusable", nearby_face)],
        0.4,
        1,
    )

    assert len(observed) == 2
    assert all(item.attributes["status"] == "unusable" for item in observed)
    assert all("person_id" not in item.attributes for item in observed)


def test_distinct_people_retain_independent_identity_histories() -> None:
    tracker = FaceIdentityTracker()
    _resolve(
        tracker,
        [_face("known"), _face("known", FAR_FACE, person_id="bob")],
        0.0,
        0,
    )

    observed = _resolve(
        tracker,
        [_face("unusable", FAR_FACE), _face("unusable")],
        0.4,
        1,
    )

    assert [item.attributes["person_id"] for item in observed] == ["bob", "alice"]


def test_single_conflicting_match_does_not_replace_known_identity() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known")], 0.0, 0)

    replacement = _resolve(
        tracker,
        [_face("known", person_id="bob")],
        0.4,
        1,
    )[0]
    retained = _resolve(tracker, [_face("unusable")], 0.8, 2)[0]

    assert replacement.attributes["person_id"] == "alice"
    assert replacement.attributes["identity_source"] == "track_history"
    assert retained.attributes["person_id"] == "alice"


def test_clear_unknown_is_reported_on_the_first_observation() -> None:
    tracker = FaceIdentityTracker()

    observed = [
        _resolve(tracker, [_face("unknown")], sequence * 0.4, sequence)[0]
        for sequence in range(1, 4)
    ]

    assert [item.attributes["status"] for item in observed] == [
        "unknown",
        "unknown",
        "unknown",
    ]
    assert [item.attributes["unknown_confirmations"] for item in observed] == [1, 2, 3]
    assert observed[-1].label == "unknown_person"
    assert "person_id" not in observed[-1].attributes
    after_unknown = _resolve(tracker, [_face("unusable")], 1.6, 4)[0]
    assert after_unknown.attributes["status"] == "unknown"
    assert after_unknown.attributes["identity_source"] == "track_history"
    assert after_unknown.attributes["identity_age_seconds"] == pytest.approx(0.4)


def test_poor_observation_does_not_delay_the_next_unknown_frame() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("unknown")], 0.0, 0)
    _resolve(tracker, [_face("unknown")], 0.4, 1)
    _resolve(tracker, [_face("unusable")], 0.8, 2)

    observed = _resolve(tracker, [_face("unknown")], 1.2, 3)[0]

    assert observed.attributes["status"] == "unknown"
    assert observed.attributes["unknown_confirmations"] == 1


@pytest.mark.parametrize("with_body", [False, True])
def test_unknown_returns_immediately_after_a_missing_face(with_body: bool) -> None:
    tracker = FaceIdentityTracker()
    for sequence in range(2):
        timestamp = sequence * 0.4
        _resolve(
            tracker,
            [_face("unknown")],
            timestamp,
            sequence,
            snapshot=_snapshot(timestamp, sequence) if with_body else None,
        )
    _resolve(
        tracker,
        [],
        0.8,
        2,
        snapshot=_snapshot(0.8, 2) if with_body else None,
    )

    observed = _resolve(
        tracker,
        [_face("unknown")],
        1.2,
        3,
        snapshot=_snapshot(1.2, 3) if with_body else None,
    )[0]

    assert observed.attributes["status"] == "unknown"
    assert observed.attributes["unknown_confirmations"] == 1


def test_new_face_is_reported_immediately_with_its_own_track() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("unknown")], 0.0, 0)
    _resolve(tracker, [_face("unknown")], 0.4, 1)

    observed = _resolve(tracker, [_face("unknown", FAR_FACE)], 0.8, 2)[0]

    assert observed.attributes["status"] == "unknown"
    assert observed.attributes["unknown_confirmations"] == 1
    assert observed.bbox == FAR_FACE


def test_sparse_unknown_observations_do_not_need_consecutive_frames() -> None:
    tracker = FaceIdentityTracker()
    for sequence, timestamp in ((0, 0.0), (11, 4.5), (30, 12.0)):
        observed = _resolve(tracker, [_face("unknown")], timestamp, sequence)[0]
        assert observed.attributes["status"] == "unknown"
        assert observed.attributes["unknown_confirmations"] == 1
        assert observed.bbox == FACE


@pytest.mark.parametrize(("timestamp", "sequence"), [(0.0, 2), (0.8, 0), (0.4, 1)])
def test_late_or_duplicate_frames_do_not_mutate_current_identity(
    timestamp: float,
    sequence: int,
) -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known", person_id="bob")], 0.4, 1)

    late = _resolve(tracker, [_face("known")], timestamp, sequence)[0]
    current = _resolve(tracker, [_face("unusable")], 1.2, 3)[0]

    assert late.attributes["person_id"] == "alice"
    assert current.attributes["person_id"] == "bob"


def test_late_unknown_does_not_advance_confirmation_count() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("unknown")], 0.0, 0)
    _resolve(tracker, [_face("unknown")], 0.4, 1)

    late = _resolve(tracker, [_face("unknown")], 0.0, 0)[0]
    current = _resolve(tracker, [_face("unknown")], 0.8, 2)[0]

    assert late.attributes["status"] == "unusable"
    assert late.attributes["reason"] == "out_of_order_frame"
    assert current.attributes["unknown_confirmations"] == 3
    assert current.attributes["status"] == "unknown"


def test_deleting_gallery_identity_immediately_invalidates_history() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known")], 0.0, 0)

    observed = _resolve(
        tracker,
        [_face("unusable")],
        0.4,
        1,
        people={"bob": "Bob"},
    )[0]

    assert observed.attributes["status"] == "unusable"
    assert "person_id" not in observed.attributes


def test_retained_identity_uses_current_gallery_name() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known")], 0.0, 0)

    observed = _resolve(
        tracker,
        [_face("unusable")],
        0.4,
        1,
        people={"alice": "Updated name"},
    )[0]

    assert observed.attributes["name"] == "Updated name"


def test_new_session_tracker_does_not_inherit_body_id_or_identity() -> None:
    previous = FaceIdentityTracker()
    _resolve(
        previous,
        [_face("known")],
        0.0,
        0,
        snapshot=_snapshot(0.0, 0),
    )

    current = FaceIdentityTracker()
    observed = _resolve(
        current,
        [_face("unusable")],
        0.4,
        1,
        snapshot=_snapshot(0.4, 1),
    )[0]

    assert observed.attributes["status"] == "unusable"
    assert "person_id" not in observed.attributes


def _confirmed_stranger(
    tracker: FaceIdentityTracker,
    *,
    with_body: bool = False,
) -> ModelObject:
    for sequence in range(3):
        timestamp = sequence * 0.4
        observed = _resolve(
            tracker,
            [_face("unknown")],
            timestamp,
            sequence,
            snapshot=_snapshot(timestamp, sequence) if with_body else None,
        )[0]
    assert observed.attributes["status"] == "unknown"
    return observed


def test_confirmed_stranger_survives_poor_face_with_same_track() -> None:
    tracker = FaceIdentityTracker()
    confirmed = _confirmed_stranger(tracker)

    retained = _resolve(tracker, [_face("unusable")], 1.2, 3)[0]
    reliable_again = _resolve(tracker, [_face("unknown")], 1.6, 4)[0]

    for observed in (retained, reliable_again):
        assert observed.label == "unknown_person"
        assert observed.attributes["status"] == "unknown"
        assert (
            observed.attributes["face_track_id"]
            == (confirmed.attributes["face_track_id"])
        )
    assert retained.attributes["identity_source"] == "track_history"
    assert retained.attributes["identity_age_seconds"] == pytest.approx(0.4)


def test_repeated_poor_faces_do_not_extend_confirmed_stranger_lifetime() -> None:
    tracker = FaceIdentityTracker()
    _confirmed_stranger(tracker)

    for sequence in range(3, 10):
        timestamp = sequence * 0.4
        observed = _resolve(tracker, [_face("unusable")], timestamp, sequence)[0]
        assert observed.attributes["status"] == "unknown"
        assert observed.attributes["identity_age_seconds"] == pytest.approx(
            timestamp - 0.8
        )

    expired = _resolve(tracker, [_face("unusable")], 4.0, 10)[0]
    assert expired.attributes["status"] == "unusable"
    returning = _resolve(tracker, [_face("unknown")], 4.4, 11)[0]
    assert returning.attributes["status"] == "unknown"
    assert returning.attributes["unknown_confirmations"] == 1


def test_missing_face_without_body_expires_after_three_seconds() -> None:
    tracker = FaceIdentityTracker()
    confirmed = _confirmed_stranger(tracker)

    for sequence in range(3, 6):
        timestamp = sequence * 0.4
        observed = _resolve(tracker, [], timestamp, sequence)
        assert len(observed) == 1
        assert observed[0].bbox == FACE
        assert (
            observed[0].attributes["face_track_id"]
            == (confirmed.attributes["face_track_id"])
        )
        assert observed[0].attributes["identity_source"] == "track_history"
    assert _resolve(tracker, [], 4.0, 10) == ()


def test_missing_face_follows_body_and_expires_without_new_evidence() -> None:
    tracker = FaceIdentityTracker()
    _confirmed_stranger(tracker, with_body=True)
    moved_body = NormalizedBBox(0.15, 0.075, 0.85, 0.975)

    for sequence in range(3, 10):
        timestamp = sequence * 0.4
        observed = _resolve(
            tracker,
            [],
            timestamp,
            sequence,
            snapshot=_snapshot(timestamp, sequence, (("P0001", moved_body),)),
        )
        assert len(observed) == 1
        assert observed[0].label == "unknown_person"
        assert observed[0].attributes["person_track_id"] == "P0001"
        assert observed[0].attributes["identity_age_seconds"] == pytest.approx(
            timestamp - 0.8
        )
        assert observed[0].bbox.x1 == pytest.approx(FACE.x1 + 0.05)
        assert observed[0].bbox.y1 == pytest.approx(FACE.y1 + 0.025)
        assert observed[0].bbox.x2 == pytest.approx(FACE.x2 + 0.05)
        assert observed[0].bbox.y2 == pytest.approx(FACE.y2 + 0.025)
    assert (
        _resolve(
            tracker,
            [],
            4.0,
            10,
            snapshot=_snapshot(4.0, 10, (("P0001", moved_body),)),
        )
        == ()
    )


@pytest.mark.parametrize("with_body", [False, True])
def test_single_frame_stranger_keeps_annotation_for_brief_missing_face(
    with_body: bool,
) -> None:
    tracker = FaceIdentityTracker()
    _resolve(
        tracker, [_face("unknown")], 0.0, 0,
        snapshot=_snapshot(0.0, 0) if with_body else None,
    )

    held = _resolve(
        tracker,
        [],
        0.8,
        2,
        snapshot=_snapshot(0.8, 2) if with_body else None,
    )
    assert len(held) == 1
    assert held[0].attributes["status"] == "unknown"
    assert held[0].attributes["identity_source"] == "track_history"


@pytest.mark.parametrize(
    "bodies",
    [
        (("P0002", BODY),),
        (("P0001", NormalizedBBox(0.85, 0.05, 0.99, 0.95)),),
        (("P0001", BODY), ("P0002", BODY)),
    ],
    ids=["new-id", "jumped", "ambiguous"],
)
def test_confirmed_stranger_history_is_dropped_when_body_continuity_breaks(
    bodies: Sequence[tuple[str, NormalizedBBox]],
) -> None:
    tracker = FaceIdentityTracker()
    _confirmed_stranger(tracker, with_body=True)

    assert (
        _resolve(
            tracker,
            [],
            1.2,
            3,
            snapshot=_snapshot(1.2, 3, bodies),
        )
        == ()
    )
    returning = _resolve(
        tracker,
        [_face("unknown")],
        1.6,
        4,
        snapshot=_snapshot(1.6, 4),
    )[0]
    assert returning.attributes["status"] == "unknown"
    assert returning.attributes["unknown_confirmations"] == 1


def test_known_match_immediately_clears_retained_unknown_status() -> None:
    tracker = FaceIdentityTracker()
    _confirmed_stranger(tracker)

    known = _resolve(tracker, [_face("known")], 1.2, 3)[0]
    retained = _resolve(tracker, [_face("unusable")], 1.6, 4)[0]

    for observed in (known, retained):
        assert observed.attributes["status"] == "known"
        assert observed.attributes["person_id"] == "alice"


def test_strangers_keep_independent_track_ids_when_face_order_changes() -> None:
    tracker = FaceIdentityTracker()
    for sequence in range(3):
        confirmed = _resolve(
            tracker,
            [_face("unknown"), _face("unknown", FAR_FACE)],
            sequence * 0.4,
            sequence,
        )
    first_id, second_id = [str(item.attributes["face_track_id"]) for item in confirmed]
    assert first_id != second_id

    observed = _resolve(
        tracker,
        [_face("unusable", FAR_FACE), _face("unusable")],
        1.2,
        3,
    )

    assert [item.attributes["face_track_id"] for item in observed] == [
        second_id,
        first_id,
    ]


def test_ambiguous_faces_do_not_duplicate_retained_stranger() -> None:
    tracker = FaceIdentityTracker()
    _confirmed_stranger(tracker)
    nearby_face = NormalizedBBox(0.21, 0.10, 0.31, 0.20)

    observed = _resolve(
        tracker,
        [_face("unusable"), _face("unusable", nearby_face)],
        1.2,
        3,
    )

    assert len(observed) == 2
    assert all(item.attributes["status"] == "unusable" for item in observed)


def test_only_sustained_strict_quality_mismatches_can_reset_known_identity() -> None:
    tracker = FaceIdentityTracker()
    initial = replace(
        _face("known"), identity_embedding=(1.0, 0.0), identity_quality=True
    )
    _resolve(tracker, [initial], 0, 0)
    mismatch = replace(
        _face("unknown"),
        identity_embedding=(0.0, 1.0),
        identity_quality=True,
        attributes={"status": "unknown", "threshold": 0.4, "similarity": 0.0},
    )
    for sequence in range(1, 5):
        current = _resolve(tracker, [mismatch], sequence * 0.5, sequence)[0]
        assert current.attributes["status"] == "known"
    reset = _resolve(tracker, [mismatch], 2.5, 5)[0]
    assert reset.attributes["status"] == "unknown"
    assert reset.attributes["identity_reset"] is True
    for sequence in (6, 7):
        current = _resolve(tracker, [mismatch], sequence * 0.5, sequence)[0]
    assert current.attributes["status"] == "unknown"


def test_ordinary_side_angle_mismatches_do_not_count_as_identity_conflicts() -> None:
    tracker = FaceIdentityTracker()
    initial = replace(
        _face("known"), identity_embedding=(1.0, 0.0), identity_quality=True
    )
    _resolve(tracker, [initial], 0, 0)
    mismatch = replace(
        _face("unknown"),
        identity_embedding=(0.0, 1.0),
        identity_quality=False,
        attributes={"status": "unknown", "threshold": 0.4, "similarity": 0.0},
    )
    for sequence in range(1, 26):
        current = _resolve(tracker, [mismatch], sequence * 0.4, sequence)[0]
        assert current.attributes["person_id"] == "alice"


def test_unusable_observation_breaks_conflicting_identity_confirmation() -> None:
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known")], 0, 0)
    _resolve(tracker, [_face("known", person_id="bob")], 0.4, 1)
    _resolve(tracker, [_face("unusable")], 0.8, 2)
    for sequence in (3, 4):
        current = _resolve(
            tracker, [_face("known", person_id="bob")], sequence * 0.4, sequence
        )[0]
        assert current.attributes["person_id"] == "alice"


@pytest.mark.parametrize("scale", [0.6, 1.6, 2.0])
def test_abrupt_face_scale_change_cannot_inherit_identity_without_body(scale) -> None:
    tracker = FaceIdentityTracker()
    first = _resolve(tracker, [_face("known")], 0, 0)[0]
    half = 0.05 * scale
    changed = NormalizedBBox(0.25 - half, 0.15 - half, 0.25 + half, 0.15 + half)
    current = _resolve(tracker, [_face("unknown", changed)], 0.4, 1)[0]
    assert current.attributes["status"] == "unknown"
    assert "person_id" not in current.attributes
    assert current.attributes["face_track_id"] != first.attributes["face_track_id"]


@pytest.mark.parametrize(
    "control_box",
    [
        (0.3433422, 0.2153531, 0.6547684, 0.7004456),
        (0.3448998, 0.2208400, 0.6564789, 0.6983201),
        (0.3479972, 0.2239379, 0.6646481, 0.6964636),
        (0.3482744, 0.2292196, 0.6557488, 0.6945011),
        (0.3445552, 0.2296647, 0.6603707, 0.6959354),
        (0.3461619, 0.2356177, 0.6686591, 0.6946286),
    ],
)
def test_real_sample_hard_cut_does_not_inherit_face_only_identity(control_box) -> None:
    # Actual detector boxes: OpenCV extra David video at 5s and face/100032540_1.
    # Original/320/160/96/64/32 inputs vary the detector's normalized box.
    # The different person overlaps enough to pass the previous IoU-only
    # check. This is a hard-cut stress case, not an accuracy benchmark.
    david = NormalizedBBox(0.4823766, 0.2744443, 0.6549603, 0.6114651)
    other = NormalizedBBox(*control_box)
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known", david)], 0, 0)
    current = _resolve(tracker, [_face("unknown", other)], 1, 1)[0]
    assert current.attributes["status"] == "unknown"
    assert "person_id" not in current.attributes


def test_gradual_approach_and_brief_occlusion_preserve_face_only_identity() -> None:
    tracker = FaceIdentityTracker()
    first = _resolve(tracker, [_face("known")], 0, 0)[0]
    for index in range(1, 9):
        half = 0.05 * 1.09**index
        box = NormalizedBBox(0.25 - half, 0.15 - half, 0.25 + half, 0.15 + half)
        current = _resolve(tracker, [_face("unusable", box)], index * 0.2, index)[0]
        assert current.attributes["person_id"] == "alice"
        assert current.attributes["face_track_id"] == first.attributes["face_track_id"]
    assert _resolve(tracker, [], 2, 9) == ()
    current = _resolve(tracker, [_face("unusable", box)], 2.5, 10)[0]
    assert current.attributes["person_id"] == "alice"


def test_position_jump_breaks_face_identity_but_shared_body_handles_scale_change() -> (
    None
):
    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known")], 0, 0)
    moved = _resolve(tracker, [_face("unusable", FAR_FACE)], 0.4, 1)[0]
    assert "person_id" not in moved.attributes

    tracker = FaceIdentityTracker()
    _resolve(tracker, [_face("known")], 0, 0, snapshot=_snapshot(0, 0))
    large = NormalizedBBox(0.15, 0.05, 0.35, 0.25)
    current = _resolve(
        tracker,
        [_face("unusable", large)],
        0.4,
        1,
        snapshot=_snapshot(0.4, 1),
    )[0]
    assert current.attributes["person_id"] == "alice"
