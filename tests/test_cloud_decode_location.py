import argparse
import json

import pytest

from tools.cloud_cpu_tests.decode_location import compare, vlm_outcome


def summary(location, coverage):
    return {
        "location": location,
        "video_sha256": "same-video",
        "delay_seconds": 1.0,
        "video_duration_seconds": 56.0,
        "browser_process_cpu_percent": {"p95": 40.0},
        "browser_protocol": {"upload_window": 4, "analysis_features": ["fight"]},
        "test_source_sha256": {"decode_location.py": "same", "evaluate.cjs": "same"},
        "vlm": {"states": {"completed": 2}, "valid": True, "problems": []},
        "valid": True,
        "problems": [],
        "lane": {
            "valid": True,
            "invariant": {"host": "same-cloud", "gpu": "same-gpu"},
            "coverage": coverage,
            "strict_coverage": coverage,
            "presentation_fps": 15.0,
            "analysis_backpressure_warnings": 0,
            "cloud_source_to_result_seconds": {"p95": 0.3},
            "source_to_present_seconds": {"p95": 1.2},
            "cloud_process_cpu_percent": {"p95": 30.0},
        },
    }


def test_decode_comparison_requires_matching_video_and_keeps_conditions_distinct(
    tmp_path,
):
    mac = [tmp_path / "mac-1", tmp_path / "mac-2"]
    cloud = [tmp_path / "cloud-1", tmp_path / "cloud-2"]
    for path, location, coverage in [
        (mac[0], "mac", 0.75),
        (mac[1], "mac", 0.7),
        (cloud[0], "cloud", 0.9),
        (cloud[1], "cloud", 0.85),
    ]:
        path.mkdir()
        (path / "summary.json").write_text(json.dumps(summary(location, coverage)))
    args = argparse.Namespace(
        mac=mac, cloud=cloud, output=tmp_path / "comparison.md"
    )

    compare(args)

    report = args.output.read_text()
    assert "75.0%" in report and "70.0%" in report
    assert "90.0%" in report and "85.0%" in report
    assert "未将预览画面传回 Mac" in report

    changed = summary("cloud", 0.9)
    changed["video_sha256"] = "other-video"
    (cloud[0] / "summary.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Different video_sha256"):
        compare(args)

    changed = summary("cloud", 0.9)
    changed["browser_protocol"]["upload_window"] = 1
    (cloud[0] / "summary.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Browser analysis settings changed"):
        compare(args)

    changed = summary("cloud", 0.9)
    changed["valid"] = False
    changed["problems"] = ["VLM requests did not all complete successfully"]
    (cloud[0] / "summary.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Invalid browser or VLM trial"):
        compare(args)

    changed = summary("cloud", 0.9)
    changed["test_source_sha256"]["decode_location.py"] = "different"
    (cloud[0] / "summary.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Test script source changed"):
        compare(args)


def test_vlm_failure_invalidates_decode_trial(tmp_path):
    path = tmp_path / "vlm.jsonl"
    payloads = [
        {"type": "vlm_report", "stream_session_id": "current", "state": "completed"},
        {
            "type": "vlm_report",
            "stream_session_id": "current",
            "state": "failed",
            "error_code": "http_error",
        },
        {"type": "vlm_report", "stream_session_id": "other", "state": "failed"},
    ]
    path.write_text("\n".join(json.dumps({"payload": row}) for row in payloads))
    result = vlm_outcome(path, {"current"}, expected=True)
    assert result["states"] == {"completed": 1, "failed": 1}
    assert not result["valid"]
    missing = vlm_outcome(tmp_path / "missing.jsonl", {"current"}, expected=True)
    assert missing["valid"] is False
