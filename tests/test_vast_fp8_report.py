"""FP8 report retains usage coverage and vLLM telemetry without inflating success."""

import json

import pytest
from test_tracked_decisions_comparison import write_run

from tools.tracked_grounding.report import analyze
from tools.vast_vlm import fp8_adapter
from tools.vlm_multistream_benchmark import LABELS


@pytest.mark.parametrize(
    "version",
    [
        "vast-qwen35-fp8-yolo26s-v1",
        "vast-qwen35-fp8-yolo26s-v2-main",
        "vast-qwen35-fp8-yolo26s-v3-isolated",
    ],
)
def test_six_fp8_rounds_keep_fixed_denominators_tokens_and_server_timings(
    tmp_path, version
):
    summary = write_run(tmp_path, decisions=False)
    summary.update(fp8_adapter.plan(), state="completed")
    summary["version"] = version
    for job in summary["jobs"]:
        job["result"].update(input_schedule_span_seconds=55.8, drain_elapsed_seconds=4)
        job["telemetry"] = {"server_phases": {"queue": {"available": False}}}
        path = tmp_path / job["raw_reports"]
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        for row in rows:
            if row["report"]["analysis_complete"]:
                row["report"]["timing"]["http_timing"] = {
                    "prompt_tokens": 2400,
                    "completion_tokens": 80,
                    "http_elapsed_seconds": 4,
                    "stream_first_content_seconds": 3,
                }
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    reference = {
        "policy": summary["event_policy"],
        "lookup": {(0,): "w"},
        "refs": {"w": {"judgments": {label: {"detected": False} for label in LABELS}}},
    }
    result = analyze(tmp_path, reference)
    assert len(result["trials"]) == 6
    for trial in result["trials"]:
        assert trial["telemetry"]["server_phases"]["queue"]["available"] is False
        for lane in trial["lanes"].values():
            assert lane["sent_fraction"] == 2 / 828
            assert lane["complete_fraction"] == 1 / 828
            assert lane["tokens"]["completion_tokens"]["coverage"] == 0.5
            assert lane["tokens"]["api_completion_tps"]["p50"] == 20
            assert lane["tokens"]["visible_generation_tps_estimate"]["count"] == 0
            assert lane["astra"]["binary_comparable"] == 6
