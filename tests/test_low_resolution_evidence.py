"""Real low-resolution replay/encoding regressions; no model accuracy claims."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.video_evidence import EvidenceFrame, FrameRing, atomic_json
from tools.low_resolution_evidence_eval import SIZES, Variant, evaluate_variant

HAS_ENCODER = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not HAS_ENCODER, reason="real local video encoder required")
@pytest.mark.parametrize(("width", "height"), SIZES)
@pytest.mark.parametrize("fps", (1, 5))
@pytest.mark.parametrize("scenario", ("bridge", "split"))
def test_low_resolution_replay_preserves_episode_count_and_full_clip(
    tmp_path: Path,
    width: int,
    height: int,
    fps: int,
    scenario: str,
) -> None:
    record = evaluate_variant(
        tmp_path / "replay", Variant(width, height, fps, scenario)
    )
    assert record["passed"], record
    assert record["event_count"] == (2 if scenario == "split" else 1)
    assert record["video_count"] == record["event_count"]
    assert all(clip["decoded_frames"] == 15 for clip in record["clips"])
    assert all(clip["duration_seconds"] == 3.0 for clip in record["clips"])


@pytest.mark.skipif(not HAS_ENCODER, reason="real local video encoder required")
@pytest.mark.parametrize("fps", (1, 5))
@pytest.mark.parametrize("scenario", ("short", "gap"))
def test_short_or_missing_source_is_explicitly_partial(
    tmp_path: Path,
    fps: int,
    scenario: str,
) -> None:
    record = evaluate_variant(tmp_path / "replay", Variant(64, 36, fps, scenario))
    assert record["passed"], record
    assert all(
        clip["partial"] and clip["duration_seconds"] < 3 for clip in record["clips"]
    )


def test_successor_frame_proves_low_fps_tail_without_entering_output_window() -> None:
    ring = FrameRing(90, 1024)
    for index in range(4):
        ring.append(EvidenceFrame(b"jpeg", float(index), index))
    selected, reason = ring.select(0, 3, 1, 5)
    assert [frame.timestamp_seconds for frame in selected] == [0, 1, 2, 3]
    assert reason is None


@pytest.mark.parametrize("timestamps", ((0, 1, 2), (0, 1, 2, 4)))
def test_unknown_or_disconnected_tail_cannot_gain_full_coverage(timestamps) -> None:
    ring = FrameRing(90, 1024)
    for index, timestamp in enumerate(timestamps):
        ring.append(EvidenceFrame(b"jpeg", timestamp, index))
    selected, reason = ring.select(0, 3, 1, 5)
    assert selected[-1].timestamp_seconds == 2
    assert reason in {"source_gap", "post_roll_unavailable"}


def test_atomic_manifest_does_not_expand_valid_windows_path_past_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    padding = max(1, 218 - len(str(root)) - 1)
    directory = root / ("x" * padding)
    path = directory / "manifest.json"
    # The former .manifest.json.<uuid>.tmp would exceed the Windows 260 limit.
    assert len(str(directory / (".manifest.json." + "a" * 32 + ".tmp"))) >= 260
    assert len(str(path)) < 260
    atomic_json(path, {"durable": True})
    assert json.loads(path.read_text()) == {"durable": True}
    assert list(directory.iterdir()) == [path]


@pytest.mark.skipif(not HAS_ENCODER, reason="real local video encoder required")
def test_high_resolution_then_tiny_odd_source_share_archive_without_cross_session_media(
    tmp_path: Path,
) -> None:
    archive = EventArchive(EventArchiveSettings(directory=tmp_path / "shared"))
    try:
        high = evaluate_variant(
            tmp_path / "high", Variant(1920, 1080, 5, "short"), shared_archive=archive
        )
        tiny = evaluate_variant(
            tmp_path / "tiny", Variant(17, 9, 1, "bridge"), shared_archive=archive
        )
        assert high["passed"], high
        assert tiny["passed"], tiny
        assert (high["clips"][0]["width"], high["clips"][0]["height"]) == (720, 404)
        assert (tiny["clips"][0]["width"], tiny["clips"][0]["height"]) == (16, 8)
        assert (
            high["event_summaries"][0]["event_id"]
            != tiny["event_summaries"][0]["event_id"]
        )
        assert archive.list_events()["count"] == 2
    finally:
        archive.close()


@pytest.mark.skipif(not HAS_ENCODER, reason="real local video encoder required")
@pytest.mark.parametrize("fps", (1, 5))
def test_low_resolution_cache_pressure_reports_bounded_outcome(
    tmp_path: Path, fps: int
) -> None:
    record = evaluate_variant(
        tmp_path / "replay", Variant(96, 54, fps, "cache_pressure", 8 * 1024)
    )
    assert record["passed"], record
    assert record["max_observed_cache_bytes"] <= 8 * 1024
    assert record["archive_status"]["dropped"] > 0
