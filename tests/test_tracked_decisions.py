"""No live APIs: native HTTP contract, track mapping, failures and export."""

import copy
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import replace

import httpx
import pytest
from test_decisions_benchmark import inspection, response_body

from goodvision_harness.model_api import ModelAPIError
from tools.benchmark_runtime.vlm_event_policy import event_policy_text
from tools.decisions_benchmark import adapter as decisions
from tools.decisions_benchmark import profile as base
from tools.tracked_decisions import adapter as a
from tools.tracked_decisions import suite
from tools.tracked_decisions.__main__ import main
from tools.tracked_grounding.detector import DetectorBroker
from tools.vlm_benchmark import modes


def people(n=2):
    return [
        {"track_id": f"P{i:04d}", "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9}}
        for i in range(n)
    ]


def reply_for(body):
    sample = response_body()["answers"][0]
    return {
        "model": base.MODEL,
        "usage": {"input_tokens": 1234, "output_tokens": 0},
        "answers": [
            {**copy.deepcopy(sample), "name": q["name"]} for q in body["questions"]
        ],
    }


@pytest.mark.parametrize("action", ["plan", "preflight", "run", "bundle", "compare"])
def test_offline_entrypoints(action, monkeypatch, capsys):
    monkeypatch.setattr(suite, "credential", lambda *a, **k: pytest.fail("read key"))
    monkeypatch.setattr(suite, "run", lambda *a, **k: pytest.fail("ran model"))
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: pytest.fail("network"))
    assert main([action]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not plan["execute"]
    assert plan["event_policy"]["sha256"] == base.POLICY_SHA256
    assert plan["request_budget"] == {
        "formal": 6624,
        "warmup": 1152,
        "preflight": 4,
        "total_upper": 7780,
    }
    assert plan["preview"].startswith("15 FPS")


@pytest.mark.parametrize(
    "key,value",
    [
        ("streams", [1]),
        ("modes", ["events"]),
        ("person_ids_only", False),
        ("observation_fps", 1),
        ("global_limit", 96),
        ("endpoint", "http://localhost:8000/v1/decisions"),
        ("parallel_scheduling", "independent"),
    ],
)
def test_frozen_load(key, value):
    args = suite.arguments()
    setattr(args, key, value)
    with pytest.raises(ValueError):
        modes.validate(args)


@pytest.mark.parametrize(
    "mode,n", [("events", 2), ("compact", 0), ("compact", 2), ("compact", 38)]
)
def test_original_images_one_http_call_and_person_mapping(
    tmp_path, monkeypatch, mode, n
):
    cfg, job = inspection()
    cfg = replace(cfg, output_mode=mode)
    requests = []

    def response(req):
        assert str(req.url) == base.ENDPOINT
        assert req.headers["authorization"] == "Bearer test-key"
        body = json.loads(req.content)
        requests.append(body)
        assert set(body) == {"model", "input", "questions"}
        assert body["model"] == base.MODEL
        content = body["input"][0]["content"]
        assert len([p for p in content if p["type"] == "input_image"]) == 4
        assert content[0]["text"].startswith(event_policy_text(base.LABELS))
        assert "Return ONLY JSON" not in content[0]["text"]
        assert len(body["questions"]) == (6 if mode == "events" else 5 * n + 6)
        for q in body["questions"]:
            assert q["choices"] == decisions.CHOICES
        reply = reply_for(body)
        reply["answers"].reverse()  # Explicit name mapping, not response order.
        return httpx.Response(200, json=reply, headers={"x-request-id": "req_test123"})

    monkeypatch.setattr(
        decisions,
        "create_vlm_http_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(response)),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "wrong-cloud-key")
    with DetectorBroker(
        "http://127.0.0.1:8092", tmp_path, transport=lambda *x: people(n)
    ) as broker:
        for f in job.frames:
            broker.submit(
                0,
                job.stream_session_id,
                f.frame_sequence,
                f.timestamp_seconds,
                f.jpeg_bytes,
            )
        obj = a.inspector(cfg, broker_url=broker.endpoint, audit_dir=tmp_path / "audit")
        try:
            verdicts = obj.inspect(job)
            assert len(verdicts) == (6 if mode == "events" else 5 * n + 6)
            assert all(v.detected is None for v in verdicts)
            if mode == "compact":
                assert {v.subject_ids[0] for v in verdicts if v.subject_ids} == {
                    p["track_id"] for p in people(n)
                }
                assert all(
                    v.track_session_id == job.stream_session_id for v in verdicts
                )
                assert all(
                    not v.subject_ids
                    for v in verdicts
                    if v.event_label == "ground_litter"
                )
            assert len(requests) == 1
            assert obj.last_http_timing["prompt_tokens"] == 1234
            assert obj.settings.timeout_seconds == 20
        finally:
            obj.close()
    audit = json.loads(next((tmp_path / "audit").glob("*.jsonl")).read_text())
    assert audit["complete"] and audit["headers"]["openai_request_id"] == "req_test123"
    assert "test-key" not in json.dumps(audit) and "data:image" not in json.dumps(audit)


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "duplicate",
        "unknown",
        "refusal",
        "probability",
        "model",
        "wrong-type",
    ],
)
def test_malformed_person_answer_never_complete(problem, tmp_path, monkeypatch):
    cfg, job = inspection()
    cfg = replace(cfg, output_mode="compact")
    requests = []

    def response(req):
        body = json.loads(req.content)
        requests.append(body)
        data = reply_for(body)
        if problem == "missing":
            data["answers"].pop()
        if problem == "duplicate":
            data["answers"][-1] = copy.deepcopy(data["answers"][0])
        if problem == "unknown":
            data["answers"][0]["name"] = "person_99_smoking"
        if problem == "refusal":
            data["answers"][0]["type"] = "refusal"
        if problem == "probability":
            data["answers"][0]["confidence"] = 2
        if problem == "model":
            data["model"] = "wrong-model"
        if problem == "wrong-type":
            data["answers"][0]["choice"] = False
        return httpx.Response(200, json=data)

    monkeypatch.setattr(
        decisions,
        "create_vlm_http_client",
        lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(response)),
    )
    with DetectorBroker(
        "http://127.0.0.1:8092", tmp_path, transport=lambda *x: people()
    ) as broker:
        for f in job.frames:
            broker.submit(
                0,
                job.stream_session_id,
                f.frame_sequence,
                f.timestamp_seconds,
                f.jpeg_bytes,
            )
        obj = a.inspector(cfg, broker_url=broker.endpoint, audit_dir=tmp_path / "audit")
        try:
            with pytest.raises(ModelAPIError):
                obj.inspect(job)
        finally:
            obj.close()
    assert len(requests) == 1
    assert not json.loads(next((tmp_path / "audit").glob("*.jsonl")).read_text())[
        "complete"
    ]


@pytest.mark.parametrize("failure", ["crowd", "detector"])
def test_detector_or_capacity_failure_never_sent(tmp_path, monkeypatch, failure):
    cfg, job = inspection()
    cfg = replace(cfg, output_mode="compact")

    def transport(*a):
        if failure == "detector":
            raise RuntimeError("detector unavailable")
        return people(39)

    monkeypatch.setattr(
        decisions,
        "create_vlm_http_client",
        lambda **kw: pytest.fail("must not call HTTP"),
    )
    with DetectorBroker(
        "http://127.0.0.1:8092", tmp_path, transport=transport
    ) as broker:
        for f in job.frames:
            broker.submit(
                0,
                job.stream_session_id,
                f.frame_sequence,
                f.timestamp_seconds,
                f.jpeg_bytes,
            )
        obj = a.inspector(cfg, broker_url=broker.endpoint)
        try:
            with pytest.raises((ValueError, ModelAPIError)):
                obj.inspect(job)
        finally:
            obj.close()


@pytest.mark.parametrize("status", [401, 402, 403])
def test_access_failure_stops_remaining_rounds(tmp_path, status):
    with pytest.raises(RuntimeError, match="stop remaining"):
        suite.check_stage(
            {},
            {"lanes": {"events": {"http_status_counts": {str(status): 1}}}},
            tmp_path,
        )


def test_transient_429_not_retried_but_quota_stops(tmp_path):
    result = {"lanes": {"events": {"http_status_counts": {"429": 1}}}}
    suite.check_stage({}, result, tmp_path)
    (tmp_path / "api-audit").mkdir()
    (tmp_path / "api-audit/a.jsonl").write_text(
        json.dumps({"error_code": "insufficient_quota"}) + "\n"
    )
    with pytest.raises(RuntimeError, match="quota"):
        suite.check_stage({}, result, tmp_path)


def test_source_bundle_standalone(tmp_path):
    from tools.decisions_benchmark.package import bundle

    target = tmp_path / "source.tar.gz"
    bundle(target, tracked=True)
    stage = tmp_path / "isolated"
    stage.mkdir()
    with tarfile.open(target) as archive:
        names = archive.getnames()
        assert not any(n.endswith((".env", ".pt", ".mp4")) for n in names)
        archive.extractall(stage, filter="data")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.tracked_decisions.suite import plan; "
            "from tools.tracked_decisions.comparison import compare; "
            "assert plan()['request_budget']['formal']==6624",
        ],
        cwd=stage,
        env=dict(os.environ, PYTHONPATH=str(stage / "src") + os.pathsep + str(stage)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (stage / "workers/detector_rich_label/tracker.py").exists()
