"""Twelve spawned camera workers exercise the actual native adapter, offline."""

import json
from functools import partial

import httpx
import pytest
from benchmark_runtime_fixtures import (
    assert_admission_drained,
    configure_ephemeral_admission,
)
from test_decisions_benchmark import response_body
from test_tracked_grounding import jpeg

from tools.decisions_benchmark import adapter as native
from tools.decisions_benchmark.report import audit_coverage, audit_rows
from tools.tracked_decisions import adapter, suite
from tools.tracked_grounding.detector import DetectorBroker
from tools.vlm_benchmark import modes
from tools.vlm_multistream_benchmark import report_lane, run_stage


def offline_inspector(config, *, broker_url, audit_dir):
    # This factory runs inside each spawned process. Production transport is
    # replaced only here, so no API call or real credential is possible.
    assert config.api_key == "offline-native-key"
    configure_ephemeral_admission(config)

    def respond(request):
        if request.method == "HEAD":
            assert request.url.host == "api.openai.com"
            return httpx.Response(200)
        assert str(request.url) == native.p.ENDPOINT
        body = json.loads(request.content)
        sample = response_body()["answers"][0]
        answers = [{**sample, "name": q["name"]} for q in body["questions"]]
        return httpx.Response(
            200,
            json={
                "model": native.p.MODEL,
                "answers": answers,
                "usage": {"input_tokens": 1234, "output_tokens": 0},
            },
        )

    native.create_vlm_http_client = lambda **kw: httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    )
    return adapter.inspector(config, broker_url=broker_url, audit_dir=audit_dir)


@pytest.mark.parametrize("mode", ["events", "compact", "parallel"])
def test_twelve_sources_native_decisions_and_person_reports(
    tmp_path, monkeypatch, mode
):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-native-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "wrong-provider-key")
    args = suite.arguments()
    args.real = True
    args.duration, args.output_mode, args.deadline = 1.6, mode, 3
    # This checks offline routing/accounting, not the host journal throughput.
    # Keep production source freshness and the real shared gate unchanged.
    args.queue_timeout = 3
    args.admission_storage = "workspace"  # Portable mock, formal profile uses tmpfs.
    base = modes.configure(suite.arguments())
    calls = []

    def detector(row, frame):
        calls.append((row["stream_session_id"], row["frame_sequence"]))
        return [
            {
                "track_id": "P0001",
                "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9},
            }
        ]

    with DetectorBroker("http://127.0.0.1:1", tmp_path, transport=detector) as broker:
        result = run_stage(
            args,
            base,
            12,
            [[jpeg()] * 4],
            tmp_path,
            1,
            inspector_factory=partial(
                offline_inspector,
                broker_url=broker.endpoint,
                audit_dir=tmp_path / "api-audit",
            ),
            source_hook=broker.submit,
        )
        broker.drain()
    assert result["all_drained"] and not result["result_records_missing"]
    assert result["result_retention"]["valid"]
    assert len(calls) == len(set(calls)) == 48
    for lane in result["lanes"].values():
        assert lane["expected_observations"] == 12
        assert lane["complete_valid_results"] == 12, (
            lane["errors"],
            lane["request_phase_metrics"],
            result["shared_admission"],
        )
    assert_admission_drained(result, args, 12 * len(result["lanes"]))
    records = [
        json.loads(s)["report"]
        for s in (tmp_path / "streams-12-repeat-1/reports.jsonl")
        .read_text()
        .splitlines()
    ]
    assert audit_coverage(records, audit_rows(tmp_path))["valid"]
    for row in records:
        if report_lane(row) == "grounding":
            assert row["subject_id_scope"] == "stream_session"
            assert row["objects"][0]["attributes"]["track_id"] == "P0001"
            assert len(row["judgments"]) == 6
            assert len(row["observations"]) == 11
            assert (
                row["timing"]["source_to_publish_seconds"]
                >= row["timing"]["inspection_timing"]["detector_wait_seconds"]
            )
