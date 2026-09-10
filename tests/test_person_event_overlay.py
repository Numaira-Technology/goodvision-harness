from __future__ import annotations

from dataclasses import replace

import pytest

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.person_event_overlay import (
    PersonEventOverlay,
    PersonOverlayAnalysis,
    is_face_result,
)


def _face(
    subject: str | None,
    *,
    x: float = 0.1,
    status: str = "known",
    attributes: dict | None = None,
) -> GroundedObject:
    return GroundedObject(
        f"face-{subject}", f"{status}_person", NormalizedBBox(x, 0.1, x + 0.1, 0.3),
        0.95,
        {
            "status": status,
            **({"subject_id": subject} if subject is not None else {}),
            **(attributes or {}),
        },
    )


def _result(
    at: float,
    *,
    faces: tuple[GroundedObject, ...] | None = None,
    events: dict[str, tuple[tuple[str, ...], ...] | None] | None = None,
    message: str | None = None,
    session: str = "session-1",
    source: str = "camera",
    sequence: int | None = None,
    frame_sequence: int | None = None,
) -> AnalysisResult:
    """None event value is negative; each positive tuple is one object's owners."""

    objects = list(faces or ())
    judgments = []
    for label, groups in (events or {}).items():
        object_ids = []
        for index, subjects in enumerate(groups or ()):
            object_id = f"{label}-{index}"
            object_ids.append(object_id)
            objects.append(GroundedObject(
                object_id, "person", NormalizedBBox(0.1, 0.1, 0.9, 0.9), 0.9,
                {"subject_ids": subjects} if subjects else {},
            ))
        judgments.append(EventJudgment(
            label, groups is not None, 0.9, f"{label} observation", tuple(object_ids)
        ))
    if faces is not None:
        unknown = tuple(
            item.object_id for item in faces
            if item.attributes.get("status") == "unknown"
        )
        judgments.append(EventJudgment(
            "stranger", bool(unknown), 0.9, "Face observation", unknown
        ))
    return AnalysisResult(
        message_id=message or f"result-{at}", source_id=source,
        stream_session_id=session, sequence=sequence or int(at * 100) + 1,
        emitted_at="2026-09-10T09:00:00Z",
        frame=FrameReference(
            int(at * 100) if frame_sequence is None else frame_sequence,
            at, "2026-09-10T09:00:00Z", 320, 180,
        ),
        judgments=tuple(judgments), objects=tuple(objects), model_name="model",
        model_latency_seconds=1.0, analysis_kind="selected_features",
        missing_event_labels=() if faces is not None else ("stranger",),
    )


def _events(
    analysis: PersonOverlayAnalysis | None,
) -> dict[str | None, tuple[str, ...]]:
    assert analysis is not None
    return {
        item.attributes.get("subject_id"): tuple(item.attributes["active_events"])
        for item in analysis.objects
    }


def test_events_follow_latest_face_without_changing_source_evidence():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    original = _result(
        10, faces=(_face("alice"), _face("bob", x=0.6)),
        events={"smoking": (("alice",),)},
    )
    latest = _result(12, faces=(_face("alice", x=0.4), _face("bob", x=0.7)))
    original_json, latest_json = original.as_dict(), latest.as_dict()
    overlay.submit_result(original)
    overlay.submit_result(latest)

    selected = overlay.select(12.1)
    assert _events(selected) == {"alice": ("smoking",), "bob": ()}
    assert selected.objects[0].bbox == latest.objects[0].bbox
    assert selected.source is latest
    assert selected.message_id == latest.message_id
    assert selected.as_dict() == latest_json
    assert original.as_dict() == original_json
    assert latest.as_dict() == latest_json


def test_clear_only_affects_people_visible_at_the_clear_source_frame():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(
        10, faces=(_face("alice"), _face("bob", x=0.6)),
        events={"smoking": (("alice",), ("bob",))},
    ))
    overlay.submit_result(_result(
        11, faces=(_face("alice"),), events={"smoking": None}
    ))
    overlay.submit_result(_result(12, faces=(_face("alice"), _face("bob", x=0.6))))
    assert _events(overlay.select(12)) == {"alice": (), "bob": ("smoking",)}


def test_persons_events_clear_independently():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(
        10, faces=(_face("alice"),),
        events={"smoking": (("alice",),), "weapon": (("alice",),)},
    ))
    overlay.submit_result(_result(
        11, faces=(_face("alice"),), events={"smoking": None}
    ))
    assert _events(overlay.select(11)) == {"alice": ("weapon",)}
    overlay.submit_result(_result(
        12, faces=(_face("alice"),), events={"weapon": None}
    ))
    assert _events(overlay.select(12)) == {"alice": ()}


def test_fight_activates_all_associated_participants_and_leaves_others_visible():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(
        10, faces=(_face("alice"), _face("bob", x=0.4), _face("carol", x=0.7)),
        events={"fight": (("alice", "bob"),)},
    ))
    assert _events(overlay.select(10)) == {
        "alice": ("fight",), "bob": ("fight",), "carol": (),
    }


@pytest.mark.parametrize("ambiguous", [False, True])
def test_positive_only_clears_other_people_when_every_object_is_associated(ambiguous):
    overlay = PersonEventOverlay(clear_after_negatives=1)
    faces = (_face("alice"), _face("bob", x=0.6))
    overlay.submit_result(_result(
        10, faces=faces, events={"smoking": (("alice",),)}
    ))
    overlay.submit_result(_result(
        11, faces=faces,
        events={"smoking": (("bob",), ()) if ambiguous else (("bob",),)},
    ))
    assert _events(overlay.select(11)) == {
        "alice": ("smoking",) if ambiguous else (), "bob": ("smoking",),
    }


def test_delayed_event_uses_its_historical_subject_on_the_latest_face_frame():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(10, faces=(_face("alice"),)))
    overlay.submit_result(_result(20, faces=(_face("alice", x=0.7), _face("bob"))))
    overlay.submit_result(_result(
        10, events={"smoking": (("alice",),)}, message="late-behavior"
    ))
    assert _events(overlay.select(20)) == {"alice": ("smoking",), "bob": ()}


def test_source_face_arriving_after_event_reconsiders_pending_event():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(12, faces=(_face("alice", x=0.7),)))
    overlay.submit_result(_result(
        10, events={"smoking": (("alice",),)}, message="behavior"
    ))
    assert _events(overlay.select(12)) == {"alice": ()}
    overlay.submit_result(_result(10, faces=(_face("alice"),), message="source-face"))
    assert _events(overlay.select(12)) == {"alice": ("smoking",)}


def test_temporary_body_owner_never_clears_a_visible_face_before_reassociation():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    faces = (_face("alice"), _face("bob", x=0.6))
    overlay.submit_result(_result(
        10, faces=faces, events={"smoking": (("alice",),)}
    ))
    overlay.submit_result(_result(11, faces=faces, message="faces-11"))
    overlay.submit_result(_result(
        11, events={"smoking": (("temporary-body",),)}, message="behavior-11"
    ))
    assert _events(overlay.select(11)) == {"alice": ("smoking",), "bob": ()}
    overlay.submit_result(_result(
        11, events={"smoking": (("bob",),)}, message="behavior-11"
    ))
    assert _events(overlay.select(11)) == {"alice": (), "bob": ("smoking",)}


@pytest.mark.parametrize("face_age, active", [(0.0, True), (0.6, True), (0.61, False)])
def test_only_recent_previous_source_face_can_anchor_an_event(face_age, active):
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(10 - face_age, faces=(_face("alice"),)))
    overlay.submit_result(_result(10, events={"smoking": (("alice",),)}))
    assert _events(overlay.select(10)) == {"alice": ("smoking",) if active else ()}


def test_future_face_never_supplies_source_visibility_for_a_clear():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(
        10, faces=(_face("alice"),), events={"smoking": (("alice",),)}
    ))
    overlay.submit_result(_result(12.1, faces=(_face("alice"),)))
    overlay.submit_result(_result(12, events={"smoking": None}))
    assert _events(overlay.select(12.1)) == {"alice": ("smoking",)}


def test_out_of_order_positive_cannot_override_newer_clear():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(10, faces=(_face("alice"),)))
    overlay.submit_result(_result(
        12, faces=(_face("alice"),), events={"smoking": None}
    ))
    overlay.submit_result(_result(
        10, events={"smoking": (("alice",),)}, message="late-positive"
    ))
    assert _events(overlay.select(12)) == {"alice": ()}
    assert _events(overlay.select(10)) == {"alice": ("smoking",)}


@pytest.mark.parametrize("face_first", [False, True])
def test_same_source_independent_branches_and_message_correction(face_first):
    overlay = PersonEventOverlay(clear_after_negatives=1)
    face = _result(10, faces=(_face("alice"),), message="face")
    event = _result(10, events={"smoking": (("alice",),)}, message="behavior")
    for result in ((face, event) if face_first else (event, face)):
        overlay.submit_result(result)
    assert _events(overlay.select(10)) == {"alice": ("smoking",)}
    overlay.submit_result(_result(10, events={"smoking": None}, message="behavior"))
    assert _events(overlay.select(10)) == {"alice": ()}
    assert len(overlay._results) == 2


def test_same_frame_result_sequence_orders_conflicting_verdicts():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(10, faces=(_face("alice"),), message="face"))
    overlay.submit_result(_result(
        10, events={"smoking": None}, message="newer", sequence=1002
    ))
    overlay.submit_result(_result(
        10, events={"smoking": (("alice",),)}, message="older", sequence=1001
    ))
    assert _events(overlay.select(10)) == {"alice": ()}


def test_unknown_person_returns_to_unknown_baseline_after_clear():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    face = _face("visitor", status="unknown")
    overlay.submit_result(_result(
        10, faces=(face,), events={"smoking": (("visitor",),)}
    ))
    assert _events(overlay.select(10)) == {"visitor": ("smoking",)}
    overlay.submit_result(_result(11, faces=(face,), events={"smoking": None}))
    selected = overlay.select(11)
    assert _events(selected) == {"visitor": ()}
    assert selected.objects[0].attributes["status"] == "unknown"


def test_face_without_subject_never_inherits_a_nearby_persons_event():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(
        10, faces=(_face("alice"),), events={"smoking": (("alice",),)}
    ))
    overlay.submit_result(_result(11, faces=(_face(None), _face("bob", x=0.6))))
    assert _events(overlay.select(11)) == {None: (), "bob": ()}


def test_empty_face_result_stops_drawing_but_does_not_clear_person_state():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(
        10, faces=(_face("alice"),), events={"smoking": (("alice",),)}
    ))
    overlay.submit_result(_result(11, faces=(), events={"smoking": None}))
    assert overlay.has_face_results
    assert overlay.select(11).objects == ()
    overlay.submit_result(_result(12, faces=(_face("alice"),)))
    assert _events(overlay.select(12)) == {"alice": ("smoking",)}


def test_missing_face_branch_does_not_replace_face_observation_or_clear_event():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    face = _result(
        10, faces=(_face("alice"),), events={"smoking": (("alice",),)}
    )
    overlay.submit_result(face)
    other = _result(11, events={"fall": None})
    assert not is_face_result(other)
    overlay.submit_result(other)
    assert overlay.select(11).source is face
    assert _events(overlay.select(11)) == {"alice": ("smoking",)}


@pytest.mark.parametrize("attributes, clear", [
    ({"face_observation_state": "held", "observation_status": "known"}, False),
    ({"face_observation_state": "observed", "identity_source": "track_history"}, True),
    ({"identity_source": "track_history"}, False),
    ({"identity_source": "track_history", "observation_status": "unusable"}, True),
])
def test_predicted_faces_show_existing_events_but_cannot_supply_clear_visibility(
    attributes, clear
):
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(
        10, faces=(_face("alice"),), events={"smoking": (("alice",),)}
    ))
    overlay.submit_result(_result(
        11, faces=(_face("alice", attributes=attributes),), events={"smoking": None}
    ))
    assert _events(overlay.select(11)) == {"alice": () if clear else ("smoking",)}


def test_face_display_expires_and_never_uses_a_future_source_frame():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(10, faces=(_face("alice"),), frame_sequence=10))
    assert overlay.select(9.9) is None
    assert overlay.select(10, frame_sequence=9) is None
    assert overlay.select(10, frame_sequence=10) is not None
    assert overlay.select(13) is not None
    assert overlay.select(13.01) is None
    assert overlay.has_face_results


def test_new_session_and_reset_drop_annotations_and_ignore_retired_sessions():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    old = _result(
        10, faces=(_face("alice"),), events={"smoking": (("alice",),)}
    )
    overlay.submit_result(old)
    overlay.submit_result(_result(0, faces=(_face("alice"),), session="session-2"))
    overlay.submit_result(old)
    assert _events(overlay.select(0)) == {"alice": ()}
    overlay.reset()
    assert not overlay.has_face_results
    assert overlay.select(0) is None
    overlay.submit_result(old)
    assert overlay.select(10) is None
    overlay.submit_result(_result(0, faces=(_face("alice"),), session="session-3"))
    assert _events(overlay.select(0)) == {"alice": ()}


def test_another_camera_with_the_same_session_id_is_ignored():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    overlay.submit_result(_result(10, faces=(_face("alice"),)))
    overlay.submit_result(_result(
        10, faces=(_face("alice"),), events={"smoking": (("alice",),)},
        source="another-camera",
    ))
    assert _events(overlay.select(10)) == {"alice": ()}


@pytest.mark.parametrize("limit", ["age", "count"])
def test_pruning_preserves_person_state_and_clear_watermarks(limit):
    overlay = PersonEventOverlay(
        clear_after_negatives=1,
        history_seconds=3 if limit == "age" else 90,
        max_history=3 if limit == "count" else 1024,
    )
    positive = _result(
        0, faces=(_face("alice"),), events={"smoking": (("alice",),)}
    )
    overlay.submit_result(positive)
    for at in range(1, 15):
        overlay.submit_result(_result(at, faces=(_face("alice"),)))
    assert _events(overlay.select(14)) == {"alice": ("smoking",)}
    overlay.submit_result(_result(
        15, faces=(_face("alice"),), events={"smoking": None}
    ))
    for at in range(16, 30):
        overlay.submit_result(_result(at, faces=(_face("alice"),)))
    overlay.submit_result(replace(positive, message_id="late-old-positive"))
    assert _events(overlay.select(29)) == {"alice": ()}
    assert len(overlay._results) <= overlay.max_history
    assert overlay.select(0) is None


def test_pruning_retains_the_previous_face_anchor_for_nearby_events():
    overlay = PersonEventOverlay(max_history=2)
    overlay.submit_result(_result(10, faces=(_face("alice"),)))
    overlay.submit_result(_result(10.1, events={"smoking": (("alice",),)}))
    overlay.submit_result(_result(11, faces=(_face("alice", x=0.7),)))
    assert _events(overlay.select(11)) == {"alice": ("smoking",)}


def test_history_and_dormant_subject_state_are_bounded():
    overlay = PersonEventOverlay(max_history=4)
    for at in range(100):
        subject = f"visitor-{at}"
        overlay.submit_result(_result(
            at, faces=(_face(subject),), events={"smoking": ((subject,),)}
        ))
    assert len(overlay._results) == 4
    assert len({subject for subject, _ in overlay._base_state}) <= 4
    assert _events(overlay.select(99)) == {"visitor-99": ("smoking",)}


def test_unknown_person_label_is_supported_when_status_attribute_is_absent():
    overlay = PersonEventOverlay(clear_after_negatives=1)
    face = replace(
        _face("visitor", status="unknown"), attributes={"subject_id": "visitor"}
    )
    result = _result(10, faces=(face,))
    assert is_face_result(result)
    overlay.submit_result(result)
    assert _events(overlay.select(10)) == {"visitor": ()}


@pytest.mark.parametrize("kwargs", [
    {"source_match_seconds": -1}, {"face_hold_seconds": float("inf")},
    {"history_seconds": float("nan")}, {"max_history": 1},
])
def test_invalid_history_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PersonEventOverlay(**kwargs)


def test_default_clear_requires_ten_visible_negatives_per_person_and_event():
    overlay = PersonEventOverlay()
    overlay.submit_result(_result(
        0, faces=(_face("alice"), _face("bob", x=0.6)),
        events={"smoking": (("alice",), ("bob",)), "weapon": (("alice",),)},
    ))
    for at in range(1, 10):
        overlay.submit_result(_result(
            at, faces=(_face("alice"),), events={"smoking": None},
        ))
        assert _events(overlay.select(at)) == {"alice": ("smoking", "weapon")}
    overlay.submit_result(_result(
        10, faces=(_face("alice"),), events={"smoking": None},
    ))
    overlay.submit_result(_result(11, faces=(_face("alice"), _face("bob", x=0.6))))
    assert _events(overlay.select(11)) == {
        "alice": ("weapon",), "bob": ("smoking",),
    }


@pytest.mark.parametrize("pruned", [False, True])
def test_positive_resets_streak_and_pruning_preserves_partial_confirmation(pruned):
    overlay = PersonEventOverlay(max_history=3 if pruned else 1024)
    for at in range(20):
        positive = at in (0, 9)
        overlay.submit_result(_result(
            at, faces=(_face("alice"),),
            events={"smoking": (("alice",),) if positive else None},
        ))
        assert _events(overlay.select(at)) == {
            "alice": ("smoking",) if at < 19 else (),
        }


def test_duplicates_missing_capabilities_and_unobserved_faces_do_not_count_clear():
    overlay = PersonEventOverlay()
    overlay.submit_result(_result(
        0, faces=(_face("alice"),), events={"smoking": (("alice",),)},
    ))
    for at in range(1, 10):
        negative = _result(at, faces=(_face("alice"),), events={"smoking": None})
        overlay.submit_result(negative)
        overlay.submit_result(negative)
        overlay.submit_result(replace(negative, message_id=f"replay-{at}"))
    overlay.submit_result(_result(10, faces=(_face("alice"),)))
    overlay.submit_result(_result(
        11, faces=(_face("alice", attributes={"face_observation_state": "held"}),),
        events={"smoking": None},
    ))
    assert _events(overlay.select(11)) == {"alice": ("smoking",)}
    overlay.submit_result(_result(
        12, faces=(_face("alice"),), events={"smoking": None},
    ))
    assert _events(overlay.select(12)) == {"alice": ()}


def test_late_positive_with_ten_newer_source_negatives_never_revives_live_face():
    overlay = PersonEventOverlay()
    overlay.submit_result(_result(0, faces=(_face("alice"),)))
    for at in reversed(range(1, 11)):
        overlay.submit_result(_result(
            at, faces=(_face("alice"),), events={"smoking": None},
        ))
    overlay.submit_result(_result(
        0, events={"smoking": (("alice",),)}, message="late-positive",
    ))
    assert _events(overlay.select(9)) == {"alice": ("smoking",)}
    assert _events(overlay.select(10)) == {"alice": ()}


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_clear_confirmation_configuration_is_rejected(value):
    with pytest.raises(ValueError, match="clear_after_negatives"):
        PersonEventOverlay(clear_after_negatives=value)
