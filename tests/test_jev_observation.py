import math

import pytest

from goodvision_harness.jev_observation import (
    assemble_response,
    build_jev_questions,
    observation_subjects,
)


def answer(choice="absent"):
    return {
        "type": "choice",
        "choice": choice,
        "confidence": 1.0,
        "probabilities": {
            k: float(k == choice) for k in ("present", "absent", "uncertain")
        },
    }


def test_person_boxes_are_copied_without_inventing_unknown_geometry():
    subjects = observation_subjects(
        "p1 at (10, 20, 300, 900): standing.\np2 at unknown: obscured."
    )
    assert subjects == [
        {"subject_id": "p1", "bbox": [10, 20, 300, 900]},
        {"subject_id": "p2", "bbox": None},
    ]
    questions = build_jev_questions(("fight",), subjects)
    assert set(questions) == {"p1_fight", "p2_fight"}
    assert "p1" in questions["p1_fight"]["instructions"]
    result = assemble_response(
        subjects,
        ("fight",),
        {"p1_fight": answer("present"), "p2_fight": answer("uncertain")},
    )
    assert result["subjects"] == subjects
    assert [v["detected"] for v in result["verdicts"]] == [True, None]


@pytest.mark.parametrize(
    "text",
    [
        "A man in the room.",
        "p1 at (0,0,0,10): invalid.",
        "p1 at (0,0,1001,10): invalid.",
        "p1 at (0,0,10,10): a.\np1 at (0,0,20,20): b.",
    ],
)
def test_bad_grounding_is_not_silently_completed(text):
    with pytest.raises(ValueError):
        observation_subjects(text)


def test_empty_scene_preserves_verdicts_without_fake_subject():
    subjects = observation_subjects("No visible people.")
    result = assemble_response(subjects, ("fall",), {"scene_fall": answer()})
    assert result["subjects"] == []
    assert result["verdicts"][0]["subject_ids"] == []
    assert result["verdicts"][0]["detected"] is False


def test_missing_answers_cannot_become_negatives():
    with pytest.raises(ValueError, match="coverage"):
        assemble_response([], ("fall",), {})


@pytest.mark.parametrize("bad", [math.nan, -0.1, 1.1, True, None])
def test_invalid_confidence_is_rejected(bad):
    value = answer()
    value["confidence"] = bad
    with pytest.raises(ValueError):
        assemble_response([], ("fall",), {"scene_fall": value})


def test_smoking_standard_and_high_share_production_rules():
    standard = build_jev_questions(("smoking",), [], {"smoking": "standard"})
    high = build_jev_questions(("smoking",), [], {"smoking": "high"})
    assert (
        "hand near the mouth alone is insufficient"
        in standard["scene_smoking"]["instructions"]
    )
    assert standard != high


def test_inline_people_are_not_silently_dropped():
    subjects = observation_subjects(
        "p1 at (0,0,50,90): walks. p2 at (70,10,90,80): sits."
    )
    assert len(subjects) == 2


def test_partially_malformed_people_are_rejected():
    with pytest.raises(ValueError, match="malformed"):
        observation_subjects("p1 at (0,0,50,90): walks. p2 at [70,10,90,80]: sits.")
