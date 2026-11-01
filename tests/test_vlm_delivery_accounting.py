from copy import deepcopy

import pytest

from tools.vlm_benchmark.accounting import delivery_counts, reconcile_retention
from tools.vlm_multistream_benchmark import report_lane, summarize


def row(i, *, attempted=True, error=None, historical=False, complete=True):
    return {
        "request_id": str(i),
        "vlm_lane": "information",
        "state": "failed" if error else "completed",
        "analysis_complete": complete and not error,
        "error_code": error,
        "historical_only": historical,
        "judgments": [],
        "timing": {
            "http_request_attempted": attempted,
            "inspection_attempted": attempted is not False,
            "source_to_publish_seconds": 4 if historical else 1,
            "model_seconds": 1,
        },
    }


def test_history_failures_and_missing_records_keep_separate_denominators():
    rows = [
        row(1),
        row(2, historical=True),
        row(3, error="timeout", historical=True),
        row(4, attempted=False, error="admission_timeout"),
        row(5, complete=False),
        row(6, attempted=None, error="connection"),
    ]
    result = summarize(rows, {}, 8, expected=10, deadline=3)
    counts = result["delivery_accounting"]
    assert counts["http_attempted"] == 4
    assert counts["http_attempted_complete"] == 2
    assert counts["http_attempted_without_complete"] == 2
    assert counts["http_not_attempted_records"] == 1
    assert counts["http_attempt_unknown_records"] == 1
    assert counts["planned_without_record"] == 4
    assert counts["complete_results"] == 2
    assert counts["historical_reports"] == 2
    assert (
        counts["historical_complete_results"]
        == counts["historical_failed_reports"]
        == 1
    )
    assert counts["nonhistorical_complete_results"] == 1
    assert result["on_time_valid_results"] == 1
    assert result["on_time_coverage"] == 0.1
    assert result["historical_only_results"] == 1


def test_legacy_request_bytes_do_not_invent_a_confirmed_send():
    old = row(1)
    old["timing"] = {"http_timing": {"request_bytes": 100}}
    counts = delivery_counts([old], 2)
    assert counts["http_attempted"] == 0
    assert counts["http_attempt_unknown_records"] == 1
    assert counts["planned_without_record"] == 1


def test_retention_reconciles_each_camera_and_detects_offsetting_loss_duplicate():
    rows = [[row(1)], [row(2)]]
    worker = dict(
        submitted=1, finished=1, published=1, inspection_attempted=1, http_attempted=1
    )
    stats = [{"request_accounting": {"events": worker}} for _ in range(2)]
    assert reconcile_retention(rows, stats, ["events"], report_lane)["valid"] is True
    corrupted = [[], rows[1] * 2]
    assert (
        reconcile_retention(corrupted, stats, ["events"], report_lane)["valid"] is False
    )


@pytest.mark.parametrize(
    "counter", ["submitted", "finished", "published", "http_attempted"]
)
def test_retention_detects_unfinished_unpublished_and_lost_attempts(counter):
    worker = dict(
        submitted=1, finished=1, published=1, inspection_attempted=1, http_attempted=1
    )
    worker[counter] += 1
    assert (
        reconcile_retention(
            [[row(1)]],
            [{"request_accounting": {"events": worker}}],
            ["events"],
            report_lane,
        )["valid"]
        is False
    )


def test_retention_marks_old_missing_instrumentation_unknown_and_discards_invalid():
    assert (
        reconcile_retention([[row(1)]], [{}], ["events"], report_lane)["valid"] is None
    )
    stats = [{"late_information_reports_discarded": 1}]
    assert (
        reconcile_retention([[row(1)]], stats, ["events"], report_lane)["valid"]
        is False
    )


def test_same_window_id_in_two_lanes_is_not_a_duplicate_but_same_lane_is():
    a = row(1)
    b = deepcopy(a)
    b["vlm_lane"] = "grounding"
    worker = dict(
        submitted=1, finished=1, published=1, inspection_attempted=1, http_attempted=1
    )
    stats = [{"request_accounting": {"events": worker, "grounding": worker}}]
    assert (
        reconcile_retention([[a, b]], stats, ["events", "grounding"], report_lane)[
            "valid"
        ]
        is True
    )
    doubled = {k: v * 2 for k, v in worker.items()}
    assert (
        reconcile_retention(
            [[a, a]],
            [{"request_accounting": {"events": doubled}}],
            ["events"],
            report_lane,
        )["valid"]
        is False
    )
