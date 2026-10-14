"""Tests for the offline CLI-output -> capture converter (video A path)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.extract import event_recognition, temporal_event_end  # noqa: E402
from bench.from_cli import capture_from_cli, episodes_from_vision_results  # noqa: E402
from bench.schema import Capture  # noqa: E402


def _vr(idx, label, detected, start, end):
    return {
        "type": "vision_result", "event_label": label, "detected": detected,
        "window_index": idx, "window_start_seconds": start, "window_end_seconds": end,
    }


def test_single_positive_run_gives_one_episode():
    records = [
        _vr(0, "fight", False, 0.0, 0.4),
        _vr(1, "fight", True, 0.4, 0.8),
        _vr(2, "fight", True, 0.8, 1.2),
        _vr(3, "fight", False, 1.2, 1.6),
    ]
    episodes = episodes_from_vision_results(records, "fight")
    assert len(episodes) == 1
    assert episodes[0]["first_evidence"] == 0.4
    assert episodes[0]["ended_since"] == 1.6  # window after the run ends


def test_two_runs_give_two_episodes():
    records = [
        _vr(0, "fight", True, 0.0, 0.4),
        _vr(1, "fight", False, 0.4, 0.8),
        _vr(2, "fight", True, 0.8, 1.2),
        _vr(3, "fight", False, 1.2, 1.6),
    ]
    episodes = episodes_from_vision_results(records, "fight")
    assert len(episodes) == 2


def test_positive_to_eof_uses_last_positive_end():
    records = [_vr(0, "fight", True, 0.0, 0.4), _vr(1, "fight", True, 0.4, 0.8)]
    episodes = episodes_from_vision_results(records, "fight")
    assert episodes[0]["ended_since"] == 0.8


def test_ignores_other_labels():
    records = [_vr(0, "smoking", True, 0.0, 0.4), _vr(1, "fight", False, 0.4, 0.8)]
    assert episodes_from_vision_results(records, "fight") == []


def test_end_to_end_capture_answers_q1_q4():
    records = [
        _vr(0, "fight", False, 0.0, 0.4),
        _vr(1, "fight", True, 15.0, 15.4),
        _vr(2, "fight", True, 17.6, 18.0),
        _vr(3, "fight", False, 18.0, 18.4),
    ]
    cap = Capture.from_json(capture_from_cli("vidA", records, ("fight",)))
    assert event_recognition(cap, "physical_conflict") == "yes"
    assert temporal_event_end(cap, "physical_conflict", 1) == 18.4


def test_no_positive_reads_no():
    records = [_vr(0, "fight", False, 0.0, 0.4)]
    cap = Capture.from_json(capture_from_cli("vidA", records, ("fight",)))
    assert event_recognition(cap, "physical_conflict") == "no"
