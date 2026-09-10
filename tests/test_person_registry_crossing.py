"""Crossing safety with ideal boxes; these tests do not measure a detector."""

import json
from pathlib import Path

import pytest

from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.grounding import ModelActor, ObservationBundle
from goodvision_harness.person_registry import PersonRegistry


def _observe(registry, timestamp, boxes):
    actors = tuple(ModelActor(str(i), box) for i, box in enumerate(boxes))
    result = AnalysisResult(
        str(timestamp),
        "camera",
        "session",
        int(timestamp * 25) + 1,
        "now",
        FrameReference(int(timestamp * 25), timestamp, "now", 384, 288),
        (EventJudgment("test", False, 1, "Geometry control"),),
        tuple(GroundedObject(actor.actor_id, "person", actor.bbox) for actor in actors),
        "test",
        0,
    )
    result = registry.prepare_analysis(
        result,
        ObservationBundle(actors, tuple((actor.actor_id,) for actor in actors)),
    )
    return [item.attributes["subject_id"] for item in result.objects]


@pytest.mark.parametrize("step", [5, 10, 20, 40])
def test_public_crossing_geometry_never_inherits_a_different_person(step):
    data = json.loads(
        (Path(__file__).parent / "fixtures/caviar_crossing_geometry.json").read_text()
    )
    registry = PersonRegistry("session", "camera")
    owners = {}
    for index in range(200, 401, step):
        people = data["frames"][str(index)]
        subjects = _observe(
            registry,
            index / 25,
            [NormalizedBBox(*person["bbox"]) for person in people],
        )
        for subject, person in zip(subjects, people, strict=True):
            assert owners.setdefault(subject, person["gt_id"]) == person["gt_id"]


@pytest.mark.parametrize("interval", [0.2, 0.4, 1.6])
def test_multiple_people_moving_together_keep_independent_subjects(interval):
    registry = PersonRegistry("session", "camera")
    initial = None
    for index in range(12):
        # No crossing or conflict: quarantining every new person would fail.
        offset = index * 0.006
        boxes = [
            NormalizedBBox(left + offset, 0.1, left + offset + 0.1, 0.8)
            for left in (0.05, 0.30, 0.55)
        ]
        current = _observe(registry, index * interval, boxes)
        initial = initial or current
        assert current == initial
        assert len(set(current)) == 3
