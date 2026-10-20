"""Spawn twelve real benchmark workers with local mock VLM/YOLO transports."""

import json
from functools import partial

import httpx
import pytest
from test_tracked_grounding import answer, jpeg
from test_vast_vlm import stream

from tools.tracked_grounding.detector import DetectorBroker
from tools.vast_vlm import benchmark, fp8_adapter, fp8_sweep
from tools.vlm_benchmark import modes
from tools.vlm_multistream_benchmark import report_lane, run_stage


def offline_inspector(config, *, broker_url):
    assert config.api_key == ""

    def respond(request):
        if request.method == "HEAD":
            return httpx.Response(200)
        assert str(request.url) == "http://127.0.0.1:8000/v1/chat/completions"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        assert "provider" not in payload and "reasoning" not in payload
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        result = (
            answer()
            if payload["max_tokens"] == 1600
            else dict.fromkeys(config.labels, False)
        )
        return httpx.Response(
            200, content=stream(result), headers={"content-type": "text/event-stream"}
        )

    benchmark.create_vlm_http_client = lambda **kw: httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    )
    return fp8_adapter.inspector(config, broker_url=broker_url)


@pytest.mark.parametrize("mode", ["events", "compact", "parallel"])
@pytest.mark.parametrize("sequences", [None, 16, 24, 32])
def test_twelve_sources_fp8_and_person_reports(tmp_path, monkeypatch, mode, sequences):
    monkeypatch.setenv("OPENROUTER_API_KEY", "wrong-cloud-key")
    monkeypatch.setenv("LOCAL_INFERENCE_API_KEY", "must-not-forward")
    args = (
        fp8_adapter.arguments(video="output.mp4", real=True)
        if sequences is None
        else fp8_sweep.arguments(
            fp8_sweep.Experiment(sequences), video="output.mp4", real=True
        )
    )
    base = modes.configure(args)
    # Offline fixture shortens media duration; formal load is tested separately.
    args.duration, args.output_mode, args.deadline = 1.6, mode, 3
    # This checks offline routing/accounting, not the host journal throughput.
    # Keep production source freshness and the real shared gate unchanged.
    args.queue_timeout = 3
    args.admission_storage = "workspace"
    calls = []

    def detector(row, frame):
        calls.append((row["stream_session_id"], row["frame_sequence"]))
        return [
            {"track_id": "P0001", "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9}}
        ]

    with DetectorBroker("http://127.0.0.1:1", tmp_path, transport=detector) as broker:
        result = run_stage(
            args,
            base,
            12,
            [[jpeg()] * 4],
            tmp_path,
            1,
            inspector_factory=partial(offline_inspector, broker_url=broker.endpoint),
            source_hook=broker.submit,
        )
        broker.drain()
    assert result["all_drained"] and not result["result_records_missing"]
    assert result["result_retention"]["valid"]
    assert len(calls) == len(set(calls)) == 48
    for lane in result["lanes"].values():
        assert lane["expected_observations"] == lane["complete_valid_results"] == 12, (
            lane["errors"]
        )
    records = [
        json.loads(s)["report"]
        for s in (tmp_path / "streams-12-repeat-1/reports.jsonl")
        .read_text()
        .splitlines()
    ]
    for row in records:
        assert len(row["judgments"]) == 6
        assert row["timing"]["http_timing"]["completion_tokens"] == 80
        if report_lane(row) == "grounding":
            assert row["subject_id_scope"] == "stream_session"
            assert row["objects"][0]["attributes"]["track_id"] == "P0001"
            assert (
                row["timing"]["source_to_publish_seconds"]
                >= row["timing"]["inspection_timing"]["detector_wait_seconds"]
            )
