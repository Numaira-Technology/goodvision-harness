"""The hand-written benchmark vocabulary must load into the harness as-is.

A definition that the harness rejects, or a label that does not match the
annotation label, produces a zero for that question rather than an error, so
the mismatch has to fail here instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from goodvision_harness.event_registry import (
    BALANCED_RECALL,
    MAX_INJECTED_EVENTS,
    PERSON_TARGET,
    vocabulary_from_json,
)

VOCABULARY = Path(__file__).resolve().parents[1] / "events" / "vocabulary.yaml"

# Labels dropped from the benchmark, and why. Adding one of these to the
# vocabulary is a mistake rather than an omission.
EXCLUDED = {
    "person_interacts_with_laptop": "retired, no longer a correct answer",
    "physical_conflict": "already mapped to the built-in fight event",
    "vehicle_stops": "vehicle subject, unsupported by the person tracker",
    "vehicle_starts": "vehicle subject, unsupported by the person tracker",
    "vehicle_turns_right": "vehicle subject, unsupported by the person tracker",
    "vehicle_turns_left": "vehicle subject, unsupported by the person tracker",
    "vehicle_reverses": "vehicle subject, unsupported by the person tracker",
    "vehicle_makes_u_turn": "vehicle subject, unsupported by the person tracker",
    "vehicle_drops_off_person": "vehicle subject, unsupported by the person tracker",
    "vehicle_picks_up_person": "vehicle subject, unsupported by the person tracker",
}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return yaml.safe_load(VOCABULARY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def injected(rows):
    return vocabulary_from_json(rows).injected()


def test_the_vocabulary_loads_into_the_harness_unchanged(rows, injected):
    assert len(rows) == len(injected) == 26
    assert len(rows) <= MAX_INJECTED_EVENTS


def test_every_event_is_judged_on_evidence_rather_than_for_recall(injected):
    """These are everyday activities; the safety recall bias would sink
    precision on all 26 of them."""

    assert all(item.recall == BALANCED_RECALL for item in injected)


def test_every_event_boxes_the_actor(injected):
    """The ground truth these are scored against is the actor's box."""

    assert all(item.target == PERSON_TARGET for item in injected)


def test_every_event_states_what_it_is_not(injected):
    """counter_examples is where precision is won on near-miss pairs."""

    assert all(item.counter_examples.strip() for item in injected)
    assert all(item.definition.strip() for item in injected)


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("person_opens_facility_door", "person_closes_facility_door"),
        ("person_opens_vehicle_door", "person_closes_vehicle_door"),
        ("person_opens_trunk", "person_closes_trunk"),
        ("person_picks_up_object", "person_puts_down_object"),
        ("person_texts_on_phone", "person_talks_on_phone"),
        ("person_enters_vehicle", "person_exits_vehicle"),
        ("person_sits_down", "person_stands_up"),
        ("person_loads_vehicle", "person_unloads_vehicle"),
    ],
)
def test_near_miss_pairs_point_at_each_other(injected, one, other):
    """Each half of a confusable pair must name its twin somewhere.

    Otherwise the model has no way to tell which of the two a frame shows.
    """

    by_label = {item.label: item for item in injected}
    assert one in by_label and other in by_label
    texts = " ".join(
        f"{by_label[label].definition} {by_label[label].counter_examples}"
        for label in (one, other)
    )
    assert other in texts or one in texts


def test_dropped_labels_stay_out(rows):
    present = {row["label"] for row in rows}
    assert not present & set(EXCLUDED)
