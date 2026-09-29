import json

import pytest

from tools.seg_pipeline.astra_agreement import discover_trials, score_trial
from tools.seg_pipeline.count_accuracy import evaluate_trial


def test_micro_count_f1_excludes_uncertain_and_accepts_zero_delay_job(tmp_path):
    (tmp_path / "job.json").write_text(
        json.dumps({"video_sha256": "video", "model": "sample", "delay": 0})
    )
    rows = []
    for sequence, displayed_count in enumerate((3, 3, 1, 0)):
        rows.extend(
            [
                {"kind": "capture", "sequence": sequence, "video_pts": sequence},
                {
                    "kind": "present", "sequence": sequence,
                 "subjects": [{"id": i} for i in range(displayed_count)]},
            ]
        )
    (tmp_path / "frames.jsonl").write_text("\n".join(map(json.dumps, rows)))
    reference = {
        "video_sha256": "video",
        "samples": [
            {"timestamp_seconds": i, "person_count": count, "status": status,
             "possible_counts": [count, count + 1]}
            for i, (count, status) in enumerate(
                ((4, "clear"), (2, "clear"), (1, "clear"), (2, "uncertain"))
            )
        ],
    }

    result = score_trial(
        tmp_path, reference,
        {"model": "sample", "configured_delay_seconds": 0, "repeat": 1, "coverage": 0.75},
        "0-second batch",
    )

    assert result["f1"] == pytest.approx(6 / 7)
    assert result["count_overlap"] == 6
    assert result["missed_people"] == 1
    assert result["extra_people"] == 1
    assert result["undercount_frames"] == 1
    assert result["overcount_frames"] == 1
    assert result["matched_clear_samples"] == result["clear_samples"] == 3
    assert [row["count_overlap"] for row in result["clear_details"]] == [3, 2, 1]
    assert [row["difference"] for row in result["clear_details"]] == [-1, 1, 0]
    assert sum(row["missed_people"] for row in result["clear_details"]) == result["missed_people"]
    assert sum(row["extra_people"] for row in result["clear_details"]) == result["extra_people"]
    assert result["uncertain"][0]["possible_counts"] == [2, 3]
    assert result["uncertain"][0]["displayed_count"] == 0


def test_count_accuracy_uses_effective_delay_when_override_is_null(tmp_path):
    (tmp_path / "job.json").write_text(
        json.dumps(
            {
                "video_sha256": "video",
                "model": "sample",
                "display_delay_seconds": None,
                "delay": 0,
            }
        )
    )
    (tmp_path / "frames.jsonl").write_text(
        "\n".join(
            map(
                json.dumps,
                [
                    {"kind": "capture", "sequence": 0, "video_pts": 0},
                    {"kind": "present", "sequence": 0, "subjects": []},
                ],
            )
        )
    )
    reference = {
        "video_sha256": "video",
        "samples": [
            {"timestamp_seconds": 0, "person_count": 0, "status": "clear"}
        ],
    }

    assert evaluate_trial(tmp_path, reference)["delay_seconds"] == 0


def test_score_trial_rejects_incomplete_clear_alignment(tmp_path):
    (tmp_path / "job.json").write_text(
        json.dumps({"video_sha256": "video", "model": "sample", "delay": 0})
    )
    (tmp_path / "frames.jsonl").write_text(
        "\n".join(
            map(
                json.dumps,
                [
                    {"kind": "capture", "sequence": 0, "video_pts": 0},
                    {"kind": "present", "sequence": 0, "subjects": [{}]},
                ],
            )
        )
    )
    reference = {
        "video_sha256": "video",
        "samples": [
            {"timestamp_seconds": 0, "person_count": 1, "status": "clear"},
            {"timestamp_seconds": 1, "person_count": 4, "status": "clear"},
        ],
    }
    trial = {
        "model": "sample",
        "configured_delay_seconds": 0,
        "repeat": 1,
        "coverage": 1,
    }

    with pytest.raises(ValueError, match="Clear samples failed alignment"):
        score_trial(tmp_path, reference, trial, "0-second batch")


def test_discover_trials_rejects_nonzero_combined_report_trial(tmp_path):
    combined = tmp_path / "combined" / "report.html"
    combined.parent.mkdir()
    combined.write_text('<a href="../run/report.html">run</a>')
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.html").write_text("")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "valid": True,
                        "configured_delay_seconds": 0.5,
                        "repeat": 1,
                    }
                ]
            }
        )
    )
    sweep_summary = tmp_path / "sweep" / "model" / "delay-0.5s" / "summary.json"
    sweep_summary.parent.mkdir(parents=True)
    sweep_summary.write_text(json.dumps({"trials": []}))

    with pytest.raises(ValueError, match="nonzero-delay trial"):
        list(discover_trials(combined, tmp_path / "sweep"))
