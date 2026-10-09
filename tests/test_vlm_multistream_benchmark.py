from itertools import pairwise
from types import SimpleNamespace

import pytest
from test_vlm_independent import ROOT, jpeg

from goodvision_harness.config import load_config
from tools.vlm_multistream_benchmark import release_schedule, run_stage, summarize


def report(*, seconds=1, complete=True, error=None, uncertain=False):
    return {
        "state": "failed" if error else "completed",
        "analysis_complete": complete,
        "error_code": error,
        "judgments": [{"observed_detected": None if uncertain else False}],
        "timing": {"source_to_publish_seconds": seconds, "model_seconds": seconds},
    }


def test_missing_late_incomplete_and_failed_work_stays_in_denominator():
    rows = [
        report(),
        report(seconds=4),
        report(complete=False),
        report(error="admission_timeout"),
    ]
    result = summarize(rows, {}, 8, expected=10, deadline=3)
    assert result["complete_valid_results"] == 2
    assert result["on_time_valid_results"] == 1
    assert result["on_time_coverage"] == 0.1
    assert result["errors"] == {"admission_timeout": 1}


def test_uncertain_is_visible_and_grounding_cannot_inflate_event_throughput():
    rows = [report(uncertain=True), {**report(), "vlm_lane": "grounding"}]
    result = summarize(rows, {}, 8, expected=10, deadline=3)
    assert result["complete_valid_results"] == 1
    assert result["uncertain_judgment_fraction"] == 1
    assert result["on_time_coverage"] == 0.1


def test_validation_reason_is_summarized_separately_from_admission_failures():
    malformed = report(error="invalid_response")
    malformed["timing"]["validation_reason"] = "invalid_json_syntax"
    result = summarize([malformed, report(error="admission_timeout")], {}, 8, 10, 3)
    assert result["validation_reason_counts"] == {"invalid_json_syntax": 1}


def test_successful_unwrapping_cannot_count_malformed_output_as_valid():
    valid, malformed = report(), report(error="invalid_response")
    for row in (valid, malformed):
        row["timing"]["response_diagnostics"] = {"content_wrapper": "code_fence"}
    result = summarize([valid, malformed], {}, 8, 10, 3)
    assert result["accepted_wrapper_counts"] == {"code_fence": 1}
    assert result["complete_valid_results"] == 1
    assert result["errors"] == {"invalid_response": 1}
    assert result["on_time_coverage"] == 0.1


@pytest.mark.parametrize("count", [2, 8, 12])
def test_staggered_sources_keep_every_frame_and_identical_local_clock(count):
    offsets, releases = release_schedule(count, 30, 2.5, 1.25, "staggered")
    by_camera = [[] for _ in range(count)]
    for at, actions in releases:
        for camera, tick in actions:
            by_camera[camera].append((tick, at - offsets[camera]))
    for values in by_camera:
        assert [t for t, at in values] == list(range(75))
        assert [at for t, at in values] == pytest.approx(
            [t / 2.5 for t in range(75)], abs=1e-8
        )
    assert offsets[-1] < 1 / 1.25
    assert releases[-1][0] == pytest.approx(29.6 + offsets[-1])
    phases = [offsets[c] + 0.8 for c in range(count)]
    assert [b - a for a, b in pairwise(phases)] == pytest.approx(
        [0.8 / count] * (count - 1)
    )


def test_aligned_sources_keep_previous_rotating_send_order():
    offsets, releases = release_schedule(8, 30, 2.5, 1.25, "aligned")
    assert offsets == [0] * 8
    assert len(releases) == 75
    for tick, (at, actions) in enumerate(releases):
        assert at == pytest.approx(tick / 2.5)
        assert actions == [((tick + offset) % 8, tick) for offset in range(8)]


@pytest.mark.parametrize(
    ("observation_fps", "expected"), [(1.25, 3), (2, 5), (2.5, 6), (3, 7)]
)
def test_simulated_cadence_delivers_every_expected_window(
    tmp_path, observation_fps, expected
):
    args = SimpleNamespace(
        real=False,
        duration=3,
        simulated_latency=0,
        observation_fps=observation_fps,
        output_mode="events",
        local_concurrency=4,
        global_limit=16,
        per_stream_limit=2,
        request_rate=20,
        queue_timeout=1,
        deadline=3,
    )
    result = run_stage(
        args, load_config(ROOT / "config.yaml"), 1, [[jpeg()]], tmp_path, 1
    )
    assert result["input_timing_valid"]
    assert result["all_drained"]
    assert result["offer_rejected_by_stream"] == [0]
    assert result["result_records_missing"] == 0
    assert result["shared_admission"]["grants"] == expected
    stream = result["per_stream"][0]
    assert stream["expected_observations"] == expected
    assert stream["scheduled_observations"] == expected
    assert stream["complete_valid_results"] == expected
    assert stream["on_time_coverage"] == 1
    assert stream["errors"] == {}


def test_phase_statistics_include_observed_failures_and_do_not_invent_zeros():
    good, bad = report(), report(error="timeout")
    good["timing"]["http_timing"] = {
        "send_request_body_seconds": 0.2,
        "response_read_seconds": 1.0,
    }
    bad["timing"]["http_timing"] = {"send_request_body_seconds": 0.3}
    result = summarize([good, bad], {}, 8, 10, 3)["request_phase_metrics"]
    assert result["http_timing.send_request_body_seconds"]["count"] == 2
    assert result["http_timing.response_read_seconds"]["count"] == 1


@pytest.mark.parametrize("finish_after, drained", [(17.0, True), (25.0, False)])
def test_drain_collects_last_timeout_without_extending_request_or_waiting_forever(
    tmp_path, monkeypatch, finish_after, drained
):
    from tools import vlm_multistream_benchmark as benchmark

    class Clock:
        now = 1000.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    configs = []

    class Branch:
        def __init__(self, config, **kwargs):
            configs.append(config)
            self.finished = None
            self.results = self

        def start(self):
            pass

        def health(self):
            return {"state": "running"}

        def offer(self, *args):
            return True

        def finish_source(self):
            self.finished = clock.now

        def done(self):
            return (
                self.finished is not None
                and clock.now - self.finished >= finish_after
            )

        def stats(self):
            return {"state": "drained" if self.done() else "running"}

        def records_after(self, cursor):
            if cursor or not self.done():
                return []
            payload = report(seconds=finish_after, error="timeout", complete=False)
            payload["vlm_lane"] = "combined"
            return [SimpleNamespace(cursor=1, payload=payload)]

        def close(self, **kwargs):
            pass

    monkeypatch.setattr(benchmark, "time", clock)
    monkeypatch.setattr(benchmark, "IndependentVLMBranch", Branch)
    monkeypatch.setattr(benchmark, "environment_api_key", lambda settings: "fake-key")
    base = load_config(ROOT / "config.yaml")
    from dataclasses import replace

    base = replace(base, runtime=replace(base.runtime, scene_timeout_seconds=15))
    args = SimpleNamespace(
        real=True, duration=3, observation_fps=1.25, output_mode="compact",
        local_concurrency=4, global_limit=0, per_stream_limit=4,
        request_rate=40, queue_timeout=1, deadline=3,
    )
    result = run_stage(args, base, 1, [[jpeg()]], tmp_path, 1)
    assert configs[0].model_settings.timeout_seconds == 15
    assert result["all_drained"] is drained
    assert result["drain_timeout_seconds"] == 20
    assert result["drain_elapsed_seconds"] == pytest.approx(
        min(finish_after, 20), abs=0.03
    )
    lane = result["lanes"]["grounding"]
    assert lane["expected_observations"] == 3
    assert lane["complete_valid_results"] == 0
    assert lane["errors"] == ({"timeout": 1} if drained else {})
