"""Replay tests exercise the real temporal adapter using a fake HTTP endpoint."""

import asyncio
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from goodvision_harness.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_stress", ROOT / "tools/replay_stress.py"
)
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (64, 48))
    assert writer.isOpened()
    for _ in range(16):
        writer.write(np.full((48, 64, 3), 80, np.uint8))
    writer.release()
    return replay.Clip("test", path, (replay.LabelledEvent("smoking", 0.5, 3.75),))


def config():
    config = load_config(ROOT / "config.yaml")
    return replace(config, runtime=replace(config.runtime, scene_max_in_flight=2))


def run(clip, **kwargs):
    return replay.replay_clip(
        clip,
        config(),
        synthetic_tracks=True,
        realtime=False,
        sample_fps=4,
        drain_seconds=3,
        profile=kwargs.pop("profile", replay.SimulationProfile(latency_ms=0)),
        **kwargs,
    )


def test_replay_uses_temporal_schema_actual_images_and_confirmation(clip):
    report = run(clip)
    assert report["drain_complete"]
    assert report["events"][0]["confirmed"]
    assert report["events"][0]["first_alert_delay_seconds"] is None
    assert report["sam_validated"] is False
    assert report["model_mode"] == "oracle_http"
    assert report["event_sensitivities"]["smoking"] == "high"
    assert not report["diagnostics"]["failures"]
    assert report["calls"]
    assert all(2 <= len(c["source_timestamps"]) <= 4 for c in report["calls"])
    row = report["diagnostics"]["records"][0]
    assert row["frames"] and row["frames"][0]["width"] == 64
    assert row["response_outcome"] == "validated"
    assert row["evidence_outcome"] == "accepted"


@pytest.mark.parametrize(
    "field,category",
    [
        ("auth_rate", "auth"),
        ("timeout_rate", "timeout"),
        ("parse_error_rate", "parse"),
    ],
)
def test_faults_never_become_negative_evidence(clip, field, category):
    report = run(clip, profile=replay.SimulationProfile(latency_ms=0, **{field: 1}))
    assert report["diagnostics"]["failures"][category] > 0
    assert not report["events"][0]["observed"]
    assert (
        report["diagnostics"]["unanalyzed_gaps"]["smoking"]["current_seconds"] == 3.75
    )
    assert all(r["result"] == "unavailable" for r in report["diagnostics"]["records"])


def test_timeline_compression_changes_sampling_and_scales_truth(clip):
    normal, fast = run(clip), run(clip, speed=2)
    assert normal["sampled_frames"] == 16
    assert fast["sampled_frames"] == 8
    assert fast["events"][0]["start"] == 0.25
    assert fast["events"][0]["end"] == 1.875
    assert max(max(c["source_timestamps"]) for c in fast["calls"]) <= 1.875


def test_hard_request_budget_is_shared_and_thread_safe(clip):
    budget = replay.RequestBudget(2)
    report = run(clip, budget=budget)
    assert len(report["calls"]) == budget.used == 2
    assert report["diagnostics"]["failures"]["budget_exhausted"] > 0
    budget = replay.RequestBudget(7)
    with ThreadPoolExecutor(8) as pool:
        assert sum(pool.map(lambda _: budget.take(), range(50))) == 7


def test_slow_responses_can_finish_out_of_order_without_losing_evidence(
    clip, monkeypatch
):
    simulate = replay.ReplayInspector._simulate
    second_done = Event()

    async def delay_first(self, request):
        with self.call_lock:
            first = self.calls[0]["request_id"] == self._replay_job.request.request_id
        if first:
            deadline = time.monotonic() + 2
            while not second_done.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert second_done.is_set(), "two inspection slots must run concurrently"
            await asyncio.sleep(0.05)
        response = await simulate(self, request)
        if not first:
            second_done.set()
        return response

    # Force a known ordering within this run; source endpoints can differ
    # between unpaced runs as the producer and worker overlap.
    monkeypatch.setattr(replay.ReplayInspector, "_simulate", delay_first)
    report = run(
        clip,
        profile=replay.SimulationProfile(
            latency_ms=0,
            slow_rate=1,
            slow_ms=20,
        ),
    )
    assert (
        report["calls"][1]["finished_monotonic"]
        < report["calls"][0]["finished_monotonic"]
    )
    assert report["drain_complete"]
    assert report["events"][0]["confirmed"]
    assert not report["diagnostics"]["failures"]
    assert report["simulation_profile"]["slow_rate"] == 1


def test_manifest_requires_explicit_reviewed_labels(clip, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": [{"path": str(clip.path)}]}))
    with pytest.raises(ValueError, match="reviewed events"):
        replay.load_manifest(manifest)


def test_incomplete_drain_waits_for_worker_shutdown_before_returning(clip, monkeypatch):
    states = []
    close = replay.SceneInspectionWorker.close

    def observed_close(self, *args, **kwargs):
        close(self, *args, **kwargs)
        states.append(self.stats()["state"])

    monkeypatch.setattr(replay.SceneInspectionWorker, "close", observed_close)
    with pytest.raises(TimeoutError, match="did not drain"):
        replay.replay_clip(
            clip,
            config(),
            synthetic_tracks=True,
            realtime=False,
            sample_fps=4,
            drain_seconds=0.01,
            profile=replay.SimulationProfile(latency_ms=1200),
        )
    assert states[-1] == "stopped"


def test_changed_source_is_rejected_before_report(clip, monkeypatch):
    original = replay._sha256
    source_reads = 0

    def changed(path):
        nonlocal source_reads
        if path == clip.path:
            source_reads += 1
            return "before" if source_reads == 1 else "after"
        return original(path)

    monkeypatch.setattr(replay, "_sha256", changed)
    with pytest.raises(RuntimeError, match="input changed"):
        run(clip)


def test_real_budget_preflight_fails_before_replay_or_output(
    clip, tmp_path, monkeypatch
):
    tracks = tmp_path / "tracks.jsonl"
    tracks.write_text(
        json.dumps(
            {
                "stream_session_id": "old",
                "frame_sequence": 0,
                "timestamp_seconds": 0,
                "objects": [],
            }
        )
        + "\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "path": str(clip.path),
                        "tracks": str(tracks),
                        "events": [],
                    }
                ]
            }
        )
    )
    output = tmp_path / "report.json"

    def forbidden(*args, **kwargs):
        pytest.fail("preflight must reject before any replay")

    monkeypatch.setattr(replay, "replay_clip", forbidden)
    with pytest.raises(SystemExit) as error:
        replay.main(
            [
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--real",
                "--max-requests",
                "1",
            ]
        )
    assert error.value.code == 2
    assert not output.exists()


def test_cli_reports_multiple_clips_and_speeds_with_one_budget(clip, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clips": [
                    {"id": name, "path": str(clip.path), "events": []}
                    for name in ("first", "second")
                ]
            }
        )
    )
    output = tmp_path / "report.json"
    args = [
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--synthetic-tracks",
        "--no-realtime",
        "--speeds",
        "1,2",
        "--max-requests",
        "2",
        "--latency-ms",
        "0",
    ]
    assert replay.main(args) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    assert len(report["runs"]) == 4
    assert report["request_budget"] == {"limit": 2, "used": 2}
    assert sum(len(r["calls"]) for r in report["runs"]) == 2
    assert all(r["drain_complete"] for r in report["runs"])
    assert not replay._partial_output_path(output).exists()
    assert not replay._reservation_path(output).exists()
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        replay.main(args)
    assert output.read_bytes() == before


def test_cli_failure_before_first_run_leaves_no_empty_report(
    clip, tmp_path, monkeypatch
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clips": [{"path": str(clip.path), "events": []}],
            }
        )
    )
    output = tmp_path / "report.json"

    def fail(*args, **kwargs):
        raise RuntimeError("injected replay failure")

    monkeypatch.setattr(replay, "replay_clip", fail)
    with pytest.raises(RuntimeError, match="injected replay failure"):
        replay.main(
            [
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--synthetic-tracks",
                "--no-realtime",
            ]
        )
    assert not output.exists()
    assert not replay._partial_output_path(output).exists()
    assert not replay._reservation_path(output).exists()
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_cli_preserves_completed_runs_when_a_later_run_fails(
    clip, tmp_path, monkeypatch
):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "clips": [
                    {"id": name, "path": str(clip.path), "events": []}
                    for name in ("first", "second")
                ]
            }
        )
    )
    output = tmp_path / "report.json"
    completed = {"clip_id": "first", "drain_complete": True}
    calls = 0

    def complete_then_fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return completed
        raise RuntimeError("injected later failure")

    monkeypatch.setattr(replay, "replay_clip", complete_then_fail)
    with pytest.raises(RuntimeError, match="injected later failure"):
        replay.main(
            [
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--synthetic-tracks",
                "--no-realtime",
            ]
        )

    checkpoint = replay._partial_output_path(output)
    report = json.loads(checkpoint.read_text())
    assert not output.exists()
    assert report["status"] == "failed"
    assert report["runs"] == [completed]
    assert report["failure"] == {"error_type": "RuntimeError"}
    assert not replay._reservation_path(output).exists()


def test_cli_reserves_output_before_model_requests(clip, tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": [{"path": str(clip.path), "events": []}]}))
    output = tmp_path / "report.json"

    def forbidden(*args, **kwargs):
        pytest.fail("a competing writer must be rejected before replay")

    monkeypatch.setattr(replay, "replay_clip", forbidden)
    with replay._reserve_output(output):
        with pytest.raises(FileExistsError):
            replay.main(
                [
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--synthetic-tracks",
                    "--no-realtime",
                ]
            )
    assert not replay._reservation_path(output).exists()


def test_missing_recorded_tracks_are_not_synthetic_negatives(clip, tmp_path):
    tracks = tmp_path / "tracks.jsonl"
    tracks.write_text(
        json.dumps(
            {
                "stream_session_id": "old",
                "frame_sequence": 0,
                "timestamp_seconds": 0,
                "objects": [],
            }
        )
        + "\n"
    )
    report = replay.replay_clip(
        replace(clip, tracks=tracks),
        config(),
        realtime=False,
        sample_fps=4,
        profile=replay.SimulationProfile(latency_ms=0),
        drain_seconds=3,
    )
    assert report["missing_track_frames"] == 15
    assert report["calls"] == []
    assert report["perception_source"] == "recorded_tracks"


def test_real_mode_refuses_synthetic_inputs_before_any_model_calls(clip):
    with pytest.raises(ValueError, match="recorded tracks"):
        run(clip, real=True)


def test_nonfinite_inputs_rejected():
    with pytest.raises(ValueError):
        replay.LabelledEvent("smoking", 0, float("inf"))
    with pytest.raises(ValueError):
        replay.SimulationProfile(latency_ms=float("nan"))
    assert replay.parse_sensitivities(["smoking=standard"])["smoking"] == "standard"
