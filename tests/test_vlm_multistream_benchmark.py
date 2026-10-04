from itertools import pairwise

import pytest

from tools.vlm_multistream_benchmark import release_schedule, summarize


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
