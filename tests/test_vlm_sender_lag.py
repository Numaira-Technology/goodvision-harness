"""Deterministic sender stalls exercise the actual benchmark input scheduler."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_vlm_independent import ROOT
from test_vlm_multistream_benchmark import report

from goodvision_harness.config import load_config
from tools import vlm_multistream_benchmark as benchmark
from tools.vlm_benchmark.modes import usable


@pytest.mark.parametrize(
    "stall, valid, skipped, maximum_lag",
    [
        (0.0, True, (), 0.0),
        (0.5, True, (), 0.1),
        (0.65, False, (), 0.25),
        (0.8, False, (), 0.4),
        (0.81, False, (1,), 0.41),
        (1.7, False, (1, 2, 3), 1.3),
    ],
)
def test_real_sender_preserves_source_clock_and_exposes_stalled_input(
    monkeypatch,
    tmp_path,
    stall,
    valid,
    skipped,
    maximum_lag,
):
    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            assert seconds >= 0
            self.now += seconds

    clock = Clock()
    offers = []
    branches = []

    class Branch:
        # Isolate model/runtime work only. The production release loop, lag
        # decisions, collection, denominator and disk reports remain real.
        def __init__(self, config, **_kwargs):
            self.results = self
            self.finished = False
            self.closes = 0
            branches.append(self)

        def start(self):
            pass

        def health(self):
            return {"state": "running"}

        def offer(self, frame, source_pts, sequence):
            offers.append((sequence, source_pts, frame, clock.now))
            if sequence == 0:
                clock.sleep(stall)
            return True

        def finish_source(self):
            self.finished = True

        def stats(self):
            return {
                "state": "drained" if self.finished else "running",
                "scheduled_observation_windows": 1,
            }

        def records_after(self, cursor):
            if cursor or not self.finished:
                return []
            sequence, source_pts, _frame, _at = offers[-1]
            payload = {
                **report(),
                "vlm_lane": "information",
                "window_id": "final-diagnostic-window",
                "frame": {
                    "source_frame_index": sequence,
                    "timestamp_seconds": source_pts,
                },
            }
            return [SimpleNamespace(cursor=1, payload=payload)]

        def close(self, **_kwargs):
            self.closes += 1

    monkeypatch.setattr(benchmark, "time", clock)
    monkeypatch.setattr(benchmark, "IndependentVLMBranch", Branch)
    args = SimpleNamespace(
        real=False,
        duration=3,
        observation_fps=1.25,
        output_mode="events",
        simulated_latency=0,
        local_concurrency=4,
        global_limit=0,
        per_stream_limit=4,
        request_rate=40,
        queue_timeout=1,
        deadline=3,
    )
    frames = [b"first-fixture-frame", b"second-fixture-frame"]
    result = benchmark.run_stage(
        args,
        load_config(ROOT / "config.yaml"),
        1,
        [frames],
        tmp_path,
        1,
    )

    expected = [tick for tick in range(8) if tick not in skipped]
    assert [sequence for sequence, *_ in offers] == expected
    assert [source_pts for _, source_pts, *_ in offers] == pytest.approx(
        [tick / 2.5 for tick in expected]
    )
    assert [frame for _, _, frame, _ in offers] == [
        frames[tick % len(frames)] for tick in expected
    ]
    # Kept frames retain their original release times. In the long stall case
    # old ticks 1/2/3 are never offered together as accelerated catch-up work.
    assert [at for *_, at in offers] == pytest.approx(
        [0.0 if tick == 0 else max(stall, tick / 2.5) for tick in expected]
    )
    assert result["sender_missed_samples"] == len(skipped)
    assert result["sender_missed_samples_by_stream"] == [len(skipped)]
    assert result["offer_rejected_by_stream"] == [0]
    assert result["input_release_lag_seconds"]["max"] == pytest.approx(maximum_lag)
    assert result["input_timing_valid"] is valid
    assert usable(result) is valid
    assert result["all_drained"]
    assert result["result_records_missing"] == 0
    assert result["per_stream"][0]["expected_observations"] == 3
    assert result["per_stream"][0]["complete_valid_results"] == 1
    assert result["per_stream"][0]["without_complete_result"] == 2
    assert branches[0].finished and branches[0].closes == 2

    stage = tmp_path / "streams-1-repeat-1"
    assert json.loads((stage / "summary.json").read_text()) == result
    rows = [
        json.loads(line) for line in (stage / "reports.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1 and rows[0]["camera"] == 1
    assert rows[0]["report"]["frame"] == {
        "source_frame_index": 7,
        "timestamp_seconds": 2.8,
    }
    assert (stage / "status.jsonl").read_text().strip()
