import json

import pytest

from tools.seg_pipeline.count_accuracy import (
    evaluate_trial,
    validate_reference,
    write_review,
)


def test_alignment_uses_video_pts_and_excludes_uncertain_and_missing(tmp_path):
    (tmp_path / "job.json").write_text(
        json.dumps({"video_sha256": "abc", "model": "test", "display_delay_seconds": 1})
    )
    rows = [
        {"kind": "capture", "sequence": 1, "pts": 0, "video_pts": 1.02},
        {"kind": "present", "sequence": 1, "pts": 0, "subjects": []},
        {"kind": "capture", "sequence": 2, "pts": 1, "video_pts": 2.02},
        {
            "kind": "present",
            "sequence": 2,
            "pts": 1,
            "subjects": [{"track_id": "a"}, {"track_id": "a"}],
        },
    ]
    (tmp_path / "frames.jsonl").write_text("\n".join(map(json.dumps, rows)))
    reference = {
        "video_sha256": "abc",
        "samples": [
            {"timestamp_seconds": 1, "person_count": 1, "status": "clear"},
            {"timestamp_seconds": 2, "person_count": 2, "status": "clear"},
            {"timestamp_seconds": 3, "person_count": 0, "status": "clear"},
            {"timestamp_seconds": 4, "person_count": 3, "status": "uncertain"},
        ],
    }
    result = evaluate_trial(tmp_path, reference)
    assert result["eligible"] == 3
    assert result["matched"] == 2
    assert result["exact_count_accuracy"] == 0.5
    assert result["mae"] == 0.5
    assert result["samples"][2]["predicted"] is None
    assert not result["complete_alignment"]
    reference["video_sha256"] = "different"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        evaluate_trial(tmp_path, reference)


def test_review_rejects_invalid_counts_and_images_outside_frames(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    secret = tmp_path / "private.jpg"
    secret.write_bytes(b"private data")
    reference = {
        "samples": [
            {
                "timestamp_seconds": 0,
                "person_count": 1,
                "status": "clear",
                "image": "../private.jpg",
                "note": "",
            }
        ]
    }
    with pytest.raises(ValueError, match="Invalid review image path"):
        write_review(reference, frames, tmp_path / "review.html")
    reference["samples"][0]["person_count"] = True
    with pytest.raises(ValueError, match="nonnegative integer"):
        validate_reference(reference)
    reference["samples"][0]["person_count"] = 1
    reference["samples"][0]["status"] = 'clear" onmouseover="alert(1)'
    with pytest.raises(ValueError, match="clear or uncertain"):
        validate_reference(reference)
