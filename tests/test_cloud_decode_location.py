import argparse
import json

import pytest

from tools.cloud_cpu_tests.decode_location import compare


def summary(location, coverage):
    return {
        "location": location,
        "video_sha256": "same-video",
        "delay_seconds": 1.0,
        "video_duration_seconds": 56.0,
        "browser_process_cpu_percent": {"p95": 40.0},
        "lane": {
            "valid": True,
            "invariant": {"host": "same-cloud", "gpu": "same-gpu"},
            "coverage": coverage,
            "strict_coverage": coverage,
            "presentation_fps": 15.0,
            "cloud_source_to_result_seconds": {"p95": 0.3},
            "source_to_present_seconds": {"p95": 1.2},
            "cloud_process_cpu_percent": {"p95": 30.0},
        },
    }


def test_decode_comparison_requires_matching_video_and_keeps_conditions_distinct(
    tmp_path,
):
    mac, cloud = tmp_path / "mac", tmp_path / "cloud"
    mac.mkdir()
    cloud.mkdir()
    (mac / "summary.json").write_text(json.dumps(summary("mac", 0.75)))
    (cloud / "summary.json").write_text(json.dumps(summary("cloud", 0.9)))
    args = argparse.Namespace(
        mac=[mac], cloud=[cloud], output=tmp_path / "comparison.md"
    )

    compare(args)

    report = args.output.read_text()
    assert "75.0%" in report
    assert "90.0%" in report
    assert "未将预览画面传回 Mac" in report

    changed = summary("cloud", 0.9)
    changed["video_sha256"] = "other-video"
    (cloud / "summary.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Different video_sha256"):
        compare(args)
