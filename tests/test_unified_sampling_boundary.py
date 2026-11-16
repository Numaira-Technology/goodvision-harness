"""Offline tail-boundary regressions; no model calls or private media needed."""
import base64
import io
import json
import shutil
import subprocess

import pytest
from PIL import Image

from tools.unified_benchmark import preflight
from tools.unified_benchmark.contract import CURRENT_VERSION, VERSION, file_hash, rules_for, sample_times


def test_grid_caps_tail_but_keeps_exact_last_frame_and_legacy():
    assert sample_times(13.003688, last_frame_time=12.937) == list(map(float, range(13)))
    assert sample_times(2.1, last_frame_time=2.0) == [0., 1., 2.]
    assert sample_times(2., last_frame_time=2.0) == [0., 1.]
    assert sample_times(.02, last_frame_time=0.) == [0.]
    assert sample_times(13.003688)[-1] == 13.
    assert "last decoded" not in rules_for(VERSION)["sampling"]
    assert "last decoded" in rules_for(CURRENT_VERSION)["sampling"]


@pytest.mark.parametrize("last", [-.1, float("nan"), float("inf"), True, "2"])
def test_invalid_last_timestamp_fails(last):
    with pytest.raises(ValueError):
        sample_times(3, last_frame_time=last)


def timestamp_result(frames, start="5.0"):
    return subprocess.CompletedProcess([], 0, json.dumps({"format": {"start_time": start},
        "frames": [{"best_effort_timestamp_time": t} for t in frames]}), "")


def test_timestamp_probe_uses_container_origin_and_max_pts(monkeypatch):
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: timestamp_result(["5.0", "6.0", "5.5"]))
    assert preflight.probe_last_frame_time("unused.mp4") == 1.


@pytest.mark.parametrize("frames,start", [([], "0"), (["nan"], "0"), (["1"], "nan"), (["1"], "2"), (["N/A"], "0")])
def test_missing_or_invalid_timestamps_fail_closed(monkeypatch, frames, start):
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: timestamp_result(frames, start))
    with pytest.raises(ValueError):
        preflight.probe_last_frame_time("unused.mp4")


@pytest.fixture
def media_setup(tmp_path, monkeypatch):
    media = tmp_path / "media"; (media / "test").mkdir(parents=True)
    video = media / "test/sample.mp4"; video.write_bytes(b"synthetic media")
    monkeypatch.setattr(preflight.run_eval, "probe_duration", lambda p: 2.003688)
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, b"", b""))
    monkeypatch.setattr(preflight, "probe_last_frame_time", lambda p: 1.937)
    buf = io.BytesIO(); Image.new("RGB", (720, 480), (12, 34, 56)).save(buf, format="JPEG")
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    calls = []
    def frame(p, t):
        calls.append(t)
        return uri if t < 2 else None
    monkeypatch.setattr(preflight.run_eval, "frame_uri", frame)
    return media, {"filename": video.name, "sha256": file_hash(video)}, tmp_path / "bundle", calls


def test_inspection_records_only_excluded_tail(media_setup):
    media, row, out, calls = media_setup
    result = preflight.inspect_video("test", row, out, media, CURRENT_VERSION)
    assert calls == [0., 1.]
    assert result["sampling_audit"] == {"last_decoded_frame_time": 1.937,
        "container_grid_times": [0., 1., 2.], "retained_times": [0., 1.], "excluded_tail_times": [2.]}
    assert [f["time"] for f in result["frames"]] == calls
    assert all(file_hash(out / f["path"]) == f["sha256"] for f in result["frames"])


def test_in_range_extraction_failure_is_not_skipped(media_setup, monkeypatch):
    media, row, out, _ = media_setup
    original = preflight.run_eval.frame_uri
    monkeypatch.setattr(preflight.run_eval, "frame_uri", lambda p, t: None if t == 1 else original(p, t))
    with pytest.raises(ValueError, match="planned frame unavailable"):
        preflight.inspect_video("test", row, out, media, CURRENT_VERSION)


def test_legacy_keeps_original_tail_failure(media_setup, monkeypatch):
    media, row, out, calls = media_setup
    def forbidden(p):
        pytest.fail("legacy boundary must not use the new cap")
    monkeypatch.setattr(preflight, "probe_last_frame_time", forbidden)
    with pytest.raises(ValueError, match="planned frame unavailable"):
        preflight.inspect_video("test", row, out, media, VERSION)
    assert calls == [0., 1., 2.]


def test_cap_does_not_bypass_two_frame_minimum(media_setup, monkeypatch):
    media, row, out, _ = media_setup
    monkeypatch.setattr(preflight, "probe_last_frame_time", lambda p: .937)
    with pytest.raises(ValueError, match="too short"):
        preflight.inspect_video("test", row, out, media, CURRENT_VERSION)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")
def test_real_media_with_audio_tail(tmp_path):
    """Container time can exceed the video without corrupt data or failed decoding."""
    media = tmp_path / "media"; (media / "test").mkdir(parents=True)
    silent = tmp_path / "silent.mp4"; video = media / "test/tail.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=blue:s=64x48:r=15", "-frames:v", "30", "-c:v", "mpeg4", str(silent)], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(silent), "-f", "lavfi", "-i",
        "anullsrc=r=48000:cl=mono", "-t", "2.01", "-c:v", "copy", "-c:a", "aac", str(video)], check=True)
    assert preflight.run_eval.probe_duration(video) > 2
    result = preflight.inspect_video("test", {"filename": video.name, "sha256": file_hash(video)},
                                     tmp_path / "bundle", media, CURRENT_VERSION)
    assert result["sampling_audit"]["excluded_tail_times"] == [2.]
    assert [f["time"] for f in result["frames"]] == [0., 1.]
