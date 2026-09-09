"""Keep the offline GT scorer honest about assignment and expected track expiry."""

from tools.low_resolution_tracking_eval import one_to_one_matches, score_continuity


def test_assignment_maximizes_cardinality_before_overlap():
    actors = [{"bbox": [0, 0, 0.8, 1]}, {"bbox": [0, 0, 0.45, 1]}]
    truth = [{"bbox": [0, 0, 0.5, 1]}, {"bbox": [0.5, 0, 1, 1]}]
    matches = one_to_one_matches(actors, truth, 0.29)
    assert {(a, g) for a, g, _ in matches} == {(0, 1), (1, 0)}


def _frame(timestamp, assignments):
    return {
        "timestamp": timestamp,
        "truth": [{"gt_id": "a"}, {"gt_id": "b"}],
        "matches": [
            {"gt_id": gt, "subject_id": subject} for gt, subject in assignments.items()
        ],
    }


def test_switch_and_wrong_inheritance_exclude_normal_expiry():
    report = score_continuity(
        [
            _frame(0, {"a": "s1", "b": "s2"}),
            _frame(1.6, {"a": "s2", "b": "s1"}),
            _frame(8, {"a": "s3", "b": "s4"}),
        ]
    )
    counts = report["counts"]
    assert counts["subject_switches"] == 2
    assert counts["wrong_inheritances"] == 2
    assert counts["reappearances_after_expiry"] == 2
    assert counts["evaluated_continuity_links"] == 2
    assert counts["gt_match_rate"] == 1


def test_missing_actors_reduce_coverage_without_inventing_switches():
    counts = score_continuity(
        [
            _frame(0, {"a": "s1", "b": "s2"}),
            _frame(1.6, {}),
            _frame(3.2, {"a": "s3", "b": "s4"}),
        ]
    )["counts"]
    assert counts["missed_gt_observations"] == 2
    assert counts["subject_switches"] == 0
    assert counts["continuity_rate_on_matched_links"] is None
    assert counts["gt_match_rate"] == 4 / 6


def test_localized_actors_without_subjects_are_not_continuity_successes():
    counts = score_continuity(
        [
            _frame(0, {"a": None, "b": None}),
            _frame(1.6, {"a": None, "b": None}),
        ]
    )["counts"]
    assert counts["gt_match_rate"] == 1
    assert counts["unassociated_gt_observations"] == 4
    assert counts["associated_gt_observations"] == 0
    assert counts["evaluated_continuity_links"] == 0
    assert counts["continuity_rate_on_matched_links"] is None
