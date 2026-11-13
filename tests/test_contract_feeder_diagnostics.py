"""Contract fixture wait measurements cannot be treated as production drops."""

from types import SimpleNamespace

import benchmark_runtime_fixtures as fixtures
import pytest


@pytest.mark.parametrize("fail_offer", [False, True])
def test_fixture_diagnostics_preserve_group_waits_without_inventing_samples(
    monkeypatch, fail_offer,
):
    clock = [100.0]
    monkeypatch.setattr(fixtures, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    delivered = []

    class Branch:
        config = SimpleNamespace(stream_session_id="synthetic")

        def offer(self, frame, timestamp, tick):
            if fail_offer:
                raise RuntimeError("synthetic offer failure")
            delivered.append((frame, timestamp, tick))
            return True

    class Stop:
        def wait(self, delay):
            clock[0] += delay + 0.6
            return False

        def is_set(self):
            return False

    feeder = fixtures.ContractSourceFeeder(
        [Branch(), Branch()], [[b"a", b"b"]],
        [(0, [(0, 0), (1, 0)]), (0.4, [(0, 1), (1, 1)])], 100.0, 2.5,
    )
    feeder.stop = Stop()
    feeder._run()
    report = feeder.diagnostics()
    assert report["scope"] == "contract fixture release-group waits only"
    assert report["release_groups_planned"] == 2
    assert report["release_groups_measured"] == (1 if fail_offer else 2)
    assert report["release_group_excess_wait_seconds"] == pytest.approx(
        [0.6] if fail_offer else [0.6, 0.6]
    )
    assert report["planned_sample_count"] == 4
    assert report["sample_measurements_available"] is False
    assert report["missed"] == 0 and report["missed_by_stream"] == [0, 0]
    assert report["offer_rejected"] == [0, 0]
    assert report["feeder_error_type"] == ("RuntimeError" if fail_offer else None)
    assert "samples" not in report and "derived_drop_count" not in report
    assert len(delivered) == (0 if fail_offer else 4)


@pytest.mark.parametrize("excess_wait, timing_valid", [(0.0, True), (0.4, False)])
def test_contract_delivery_preserves_real_timing_rejection(
    tmp_path, monkeypatch, excess_wait, timing_valid,
):
    """Complete adapter input must not fabricate benchmark timing acceptance."""
    import json
    from threading import Event

    from test_vlm_multistream_benchmark import report

    from goodvision_harness.config import load_config
    from tools import vlm_multistream_benchmark as benchmark
    from tools.vlm_benchmark import source_feeder
    from tools.vlm_benchmark.modes import usable

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, delay):
            self.now += delay

    clock = Clock()

    class ClockEvent(Event):
        def wait(self, timeout=None):
            if not self.is_set() and timeout is not None:
                clock.sleep(timeout + excess_wait)
            return self.is_set()

    offers, branches = [], []

    class Branch:
        def __init__(self, config, **kwargs):
            self.config, self.results, self.finished = config, self, False
            branches.append(self)

        def start(self):
            pass

        def health(self):
            return {"state": "running"}

        def offer(self, frame, timestamp, tick):
            offers.append((frame, timestamp, tick))
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
            payload = {
                **report(), "vlm_lane": "information", "window_id": "fixture",
                "frame": {"source_frame_index": 7, "timestamp_seconds": 2.8},
            }
            return [SimpleNamespace(cursor=1, payload=payload)]

        def close(self, **kwargs):
            pass

    monkeypatch.setattr(benchmark, "time", clock)
    monkeypatch.setattr(fixtures, "time", clock)
    monkeypatch.setattr(source_feeder, "Event", ClockEvent)
    monkeypatch.setattr(benchmark, "TimedSourceFeeder", fixtures.ContractSourceFeeder)
    monkeypatch.setattr(benchmark, "IndependentVLMBranch", Branch)
    args = SimpleNamespace(
        real=False, duration=3, observation_fps=1.25, output_mode="events",
        simulated_latency=0, local_concurrency=4, global_limit=0,
        per_stream_limit=4, request_rate=40, queue_timeout=1, deadline=3,
    )
    diagnostic = {}
    result = benchmark.run_stage(
        args, load_config(fixtures.ROOT / "config.yaml"), 1, [[b"a", b"b"]],
        tmp_path, 1, source_diagnostics=diagnostic,
    )
    assert offers == [(b"a" if tick % 2 == 0 else b"b", tick / 2.5, tick)
                      for tick in range(8)]
    assert diagnostic["planned_sample_count"] == 8
    assert diagnostic["release_groups_measured"] == (
        diagnostic["release_groups_planned"]
    ) == 8
    assert diagnostic["release_group_excess_wait_seconds"] == pytest.approx(
        [excess_wait] * 8
    )
    assert diagnostic["sample_measurements_available"] is False
    assert result["sender_missed_samples"] == 0
    assert result["sender_missed_samples_by_stream"] == [0]
    assert result["offer_rejected_by_stream"] == [0]
    assert result["input_release_lag_seconds"]["max"] == pytest.approx(excess_wait)
    assert result["input_timing_valid"] is timing_valid
    assert result["result_records_missing"] == 0
    assert all(lane["complete_valid_results"] == 1 for lane in result["lanes"].values())
    assert usable(result) is timing_valid
    assert result["all_drained"] and branches[0].finished
    saved = json.loads((tmp_path / "streams-1-repeat-1/summary.json").read_text())
    assert saved == result
