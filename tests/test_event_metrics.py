"""Hand-counted event cases distinguish alerts, duplicates, exposure and timing."""

import json
from dataclasses import replace

import pytest

from goodvision_harness.evaluation.cli import main
from goodvision_harness.evaluation.event_metrics import EventEpisode, evaluate_events


def episode(identifier, start=0.0, end=10.0, alert=None, **kwargs):
    values = dict(
        source_id="camera",
        stream_session_id="session",
        event_label="smoking",
        subject_ids=("a",),
        start_seconds=start,
        end_seconds=end,
        alert_seconds=alert,
    )
    values.update(kwargs)
    return EventEpisode(identifier, **values)


def test_precision_recall_duplicates_and_wall_clock_detection_latency():
    targets = [episode("one"), episode("two", 20.0, 30.0)]
    predictions = [
        episode("hit", 1.0, 12.0, 4.0),
        episode("duplicate", 2.0, 11.0, 5.0),
        episode("false", 40.0, 45.0, 41.0),
    ]
    result = evaluate_events(predictions, targets, camera_hours=2)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 2
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == 0.5
    assert result["duplicate_alerts"] == 1
    assert result["false_alerts_per_camera_hour"] == 1
    assert result["time_to_detect_median_seconds"] == 4
    assert result["end_delay_p95_seconds"] == 2
    assert result["missed_event_duration_seconds"] == 11
    assert (
        evaluate_events(list(reversed(predictions)), targets, camera_hours=2) == result
    )


@pytest.mark.parametrize(
    "change",
    [
        {"source_id": "other"},
        {"stream_session_id": "other"},
        {"event_label": "fall"},
        {"subject_ids": ("b",)},
    ],
)
def test_matching_respects_source_session_event_and_actors(change):
    prediction = episode("prediction", alert=1.0, **change)
    assert (
        evaluate_events([prediction], [episode("truth")], camera_hours=1)["recall"] == 0
    )


def test_matching_maximizes_event_count_instead_of_greedy_overlap():
    targets = [episode("first", 0.0, 10.0), episode("second", 10.0, 20.0)]
    predictions = [episode("wide", 0.0, 20.0, 1.0), episode("narrow", 0.0, 10.0, 2.0)]
    assert evaluate_events(predictions, targets, camera_hours=1)["recall"] == 1


def test_empty_corpus_is_not_fabricated_perfect_accuracy_and_exposure_is_required():
    result = evaluate_events([], [], camera_hours=1)
    assert result["precision"] is None and result["recall"] is None
    assert result["time_to_detect_p95_seconds"] is None
    assert result["false_alerts_per_camera_hour"] == 0
    with pytest.raises(ValueError, match="exposure"):
        evaluate_events([], [], camera_hours=0)
    with pytest.raises(ValueError, match="alert times"):
        evaluate_events([episode("missing")], [], camera_hours=1)
    with pytest.raises(ValueError, match="unique"):
        evaluate_events(
            [], [episode("duplicate"), episode("duplicate")], camera_hours=1
        )
    assert (
        evaluate_events(
            [replace(episode("pred", alert=1), subject_ids=("other",))],
            [episode("truth")],
            camera_hours=1,
            match_actors=False,
        )["recall"]
        == 1
    )


def test_event_evaluation_cli_preserves_existing_output(tmp_path):
    from dataclasses import asdict

    predictions, targets, report = (
        tmp_path / name for name in ("p.json", "t.json", "r.json")
    )
    predictions.write_text(json.dumps([asdict(episode("prediction", alert=1.0))]))
    targets.write_text(json.dumps([asdict(episode("truth"))]))
    args = [
        str(predictions),
        str(targets),
        "--camera-hours",
        "1",
        "--output",
        str(report),
    ]
    assert main(args) == 0
    assert json.loads(report.read_text())["recall"] == 1
    original = report.read_bytes()
    assert main(args) == 2
    assert report.read_bytes() == original
