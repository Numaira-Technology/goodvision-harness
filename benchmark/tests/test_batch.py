"""Unit tests for the batch planner and manifest classification (no GPU/infra)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.manifest import classify, fetch_rows, fps_by_video  # noqa: E402
from bench.run_batch import build_script, plan_video  # noqa: E402
from bench.schema import Question  # noqa: E402


def _q(**kw):
    base = {"qa_id": "q", "capability": "counting", "question": "?", "video_id": "v"}
    base.update(kw)
    return Question.from_json(base)


def test_classify_buckets():
    assert classify(_q(capability="counting")) == "gpu_tracks"
    assert classify(_q(capability="location_tracking")) == "gpu_tracks"
    assert classify(_q(capability="event_recognition",
                       target_label="physical_conflict")) == "cli_fight"
    assert classify(_q(capability="event_recognition",
                       target_label="meva:person_texts_on_phone")) == "no_detector"
    # MOT count-over-time temporal is computed from tracks
    assert classify(_q(capability="temporal_questions", target_label="person",
                       question="全画面第一次恰好出现5名可见有效行人是在第几秒？")) == "gpu_tracks"
    # fight-end temporal
    assert classify(_q(capability="temporal_questions", target_label="physical_conflict",
                       question="When does the first physical conflict end?")) == "cli_fight"
    # meva event-duration temporal -> no detector
    assert classify(_q(capability="temporal_questions", target_label="meva:person_talks_to_person",
                       question="'person talks to person' begins at 16.3 seconds...")) == "no_detector"


def test_fps_by_video_from_clip():
    q = _q(video_id="s", public_context={"clip": {"end_frame_number": 420, "end_sec": 14.0}})
    assert fps_by_video([q]) == {"s": 30.0}


def test_fetch_rows_seq_gets_fps_file_does_not():
    media = [
        {"video_id": "seq", "source_dataset": "mot17",
         "media": {"media_type": "image_sequence", "path_or_name": "d/img1/", "sha256": "x"}},
        {"video_id": "f", "source_dataset": "ntu_cctv_fights",
         "media": {"media_type": "file", "path_or_name": "a.mpeg", "drive_file_id": "gid", "sha256": "y"}},
    ]
    rows = {r["video_id"]: r for r in fetch_rows(media, {"seq": 14.0})}
    assert rows["seq"]["assemble_fps"] == 14.0
    assert rows["f"]["assemble_fps"] == ""
    assert rows["f"]["drive_file_id"] == "gid"


def test_plan_video_image_sequence_assembles_then_captures():
    row = {"video_id": "vid1", "source_dataset": "mot17",
           "media": {"media_type": "image_sequence", "path_or_name": "mot17/x/img1/"}}
    lines = plan_video(row, 30.0, "/data/root", "http://h:7900", "caps")
    joined = "\n".join(lines)
    assert "assemble_frames.sh '/data/root/mot17/x/img1/'" in joined
    assert "30" in joined
    assert "--video-id vid1 --out caps/vid1.json" in joined


def test_plan_video_file_feeds_directly():
    row = {"video_id": "v2", "source_dataset": "ntu_cctv_fights",
           "media": {"media_type": "file", "path_or_name": "a.mpeg"}}
    lines = plan_video(row, None, "/data/root", "http://h:7900", "caps")
    joined = "\n".join(lines)
    assert "assemble_frames" not in joined
    assert "'/data/root/a.mpeg' bench" in joined


def test_build_script_has_one_capture_per_video():
    media = [
        {"video_id": "a", "source_dataset": "ntu_cctv_fights",
         "media": {"media_type": "file", "path_or_name": "a.mpeg"}},
        {"video_id": "b", "source_dataset": "mot17",
         "media": {"media_type": "image_sequence", "path_or_name": "b/img1/"}},
    ]
    script = build_script(media, {"b": 30.0}, "/root", "http://h:7900", "caps")
    assert script.startswith("#!/usr/bin/env bash")
    assert script.count("--video-id a ") == 1
    assert script.count("--video-id b ") == 1
