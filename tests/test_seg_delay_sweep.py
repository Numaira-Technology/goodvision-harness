"""Delay scan acceptance and isolation without GPU or paid API calls."""

import argparse
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools.seg_pipeline.service import benchmark_config, display_delay
from tools.seg_pipeline.sweep import parse_delays, run_sweep


@pytest.mark.parametrize("value", ["", "nan", "inf", "-0.1", "8.1"])
def test_delay_rejects_unrenderable_values(value):
    with pytest.raises(ValueError):
        display_delay(value)


@pytest.mark.parametrize("value", ["", "1,0.5", "0.5,0.5", "0.5,nan"])
def test_scan_requires_finite_unique_ascending_delays(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_delays(value)


@pytest.mark.parametrize("delay", [0.5, 1, 1.5, 2])
def test_delay_is_the_only_changed_main_setting(tmp_path, delay):
    job = {
        "output": str(tmp_path),
        "endpoint": "http://127.0.0.1:8092",
        "face_model_root": str(tmp_path / "models"),
        "allow_face_download": False,
    }
    original = benchmark_config(job)
    candidate = benchmark_config(job | {"display_delay_seconds": delay})
    assert candidate.display.sam_live_delay_seconds == delay
    assert replace(candidate, display=original.display) == original
    assert replace(candidate.display, sam_live_delay_seconds=0) == original.display


def args_for(directory):
    return SimpleNamespace(
        batch_dir=directory,
        model="yolov8-n",
        expected_models=["yolov8-n", "yolov8-s"],
        delays=[0.5, 1, 1.5, 2],
        repeats=2,
        endpoint="http://127.0.0.1:8092",
        allow_face_download=False,
    )


def fake_runner(coverages, calls, *, fail=False):
    def run(argv):
        from pathlib import Path

        delay = float(argv[argv.index("--display-delay") + 1])
        calls.append(delay)
        directory = Path(argv[argv.index("--output") + 1])
        directory.mkdir()
        trials = [
            {
                "repeat": index + 1,
                "model": "yolov8-n",
                "valid": not fail,
                "configured_delay_seconds": delay,
                "coverage": coverage,
                "coverage_pass": not fail and coverage >= 0.7,
            }
            for index, coverage in enumerate(coverages[delay])
        ]
        (directory / "summary.json").write_text(json.dumps({"trials": trials}))
        (directory / "report.html").write_text("TEST DOUBLE, NOT A GPU RESULT")
        if fail:
            raise SystemExit(1)

    return run


def test_scan_requires_each_repeat_to_pass_and_stops_at_first_passing_slot(tmp_path):
    calls = []
    # Mean at 0.5s is above 70%, but the second repetition misses the threshold.
    result = run_sweep(
        args_for(tmp_path),
        fake_runner({0.5: [0.85, 0.65], 1: [0.70, 0.75]}, calls),
    )
    assert calls == [0.5, 1]
    assert result["status"] == "passed"
    assert result["lowest_tested_passing_delay_seconds"] == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert list(summary["models"]) == ["yolov8-n"]
    assert "1 / 2" in (tmp_path / "report.html").read_text()
    assert not (tmp_path / "yolov8-n/delay-1.5s").exists()


def test_no_passing_slot_does_not_claim_model_can_never_pass(tmp_path):
    calls = []
    args = args_for(tmp_path)
    result = run_sweep(
        args, fake_runner({delay: [0.6, 0.69] for delay in args.delays}, calls)
    )
    assert calls == args.delays
    assert result["status"] == "not_reached"
    assert result["lowest_tested_passing_delay_seconds"] is None


def test_invalid_run_stops_and_preserves_failure_evidence(tmp_path):
    calls = []
    with pytest.raises(SystemExit):
        run_sweep(
            args_for(tmp_path), fake_runner({0.5: [0.95, 0.95]}, calls, fail=True)
        )
    assert calls == [0.5]
    summary = json.loads((tmp_path / "summary.json").read_text())
    result = summary["models"]["yolov8-n"]
    assert result["status"] == "failed"
    assert result["lowest_tested_passing_delay_seconds"] is None
    assert len(result["steps"][0]["summary"]["trials"]) == 2


def test_existing_results_cannot_be_overwritten(tmp_path):
    args = args_for(tmp_path)
    run_sweep(args, fake_runner({0.5: [0.8, 0.8]}, []))
    with pytest.raises(FileExistsError):
        run_sweep(args, fake_runner({0.5: [0.8, 0.8]}, []))
