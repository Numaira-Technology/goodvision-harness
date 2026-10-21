"""Native adapter through real twelve-source scheduling, without GPU/model calls."""

import json
from functools import partial

import httpx
import pytest
from benchmark_runtime_fixtures import (
    assert_admission_drained,
    configure_ephemeral_admission,
)
from test_tracked_grounding import jpeg
from test_videojev_probe import responses

from tools.tracked_grounding.detector import DetectorBroker
from tools.videojev_benchmark import adapter, suite
from tools.videojev_probe.contract import MODEL, POLICY_SHA, REVISION
from tools.vlm_benchmark import modes
from tools.vlm_multistream_benchmark import drain_budget_seconds, report_lane, run_stage


def offline_inspector(config, *, broker_url):
    configure_ephemeral_admission(config)

    def respond(request):
        assert request.url.host == "127.0.0.1"
        assert "authorization" not in request.headers
        if request.method == "HEAD":
            return httpx.Response(200)
        body = json.loads(request.content)
        assert body["policy_sha256"] == POLICY_SHA and body["fps"] == 2.5
        assert len(body["images"]) in (2, 3, 4)
        assert all("bbox" not in q["id"] for q in body["questions"])
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "revision": REVISION,
                "request_id": body["request_id"],
                "raw": responses(body["questions"], 2),
                "timings": {"server_queue_seconds": 0.001},
            },
        )

    adapter.create_vlm_http_client = lambda **kw: httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    )
    return adapter.inspector(config, broker_url=broker_url)


def test_fixed_load_and_policy():
    assert callable(suite.factory) and callable(suite.inspector)
    args = suite.arguments()
    assert (args.streams, args.repeats, args.observation_fps, args.global_limit) == (
        [12],
        2,
        1.25,
        48,
    )
    assert args.http_timeout == 20 and args.person_ids_only
    assert suite.plan()["event_policy"]["sha256"] == POLICY_SHA
    args.real, args.output_mode = True, "events"
    assert drain_budget_seconds(args, modes.configure(args)) >= 20


@pytest.mark.parametrize("mode", ["events", "compact", "parallel"])
def test_twelve_camera_native_route(tmp_path, monkeypatch, mode):
    monkeypatch.delenv("GOODVISION_LOCAL_API_KEY", raising=False)
    args = suite.arguments()
    args.real = True
    args.duration, args.output_mode, args.deadline = 1.6, mode, 3
    # This checks offline routing/accounting, not the host journal throughput.
    # Keep production source freshness and the real shared gate unchanged.
    args.queue_timeout = 3
    args.admission_storage = "workspace"
    base = modes.configure(suite.arguments())
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
        assert lane["complete_valid_results"] == 12, (
            lane["errors"],
            lane["request_phase_metrics"],
            result["shared_admission"],
        )
    assert_admission_drained(result, args, 12 * len(result["lanes"]))
    rows = [
        json.loads(s)["report"]
        for s in (tmp_path / "streams-12-repeat-1/reports.jsonl")
        .read_text()
        .splitlines()
    ]
    for row in rows:
        assert row["timing"]["http_request_attempted"] is True
        assert row["timing"]["http_timing"]["response_bytes"] > 0
        assert row["timing"]["http_timing"]["http_elapsed_seconds"] >= 0
        assert len(row["judgments"]) == 6
        if report_lane(row) == "grounding":
            assert row["subject_id_scope"] == "stream_session"
            assert len(row["observations"]) == 11
            assert row["objects"][0]["attributes"]["track_id"] == "P0001"


def test_server_identity_validation_and_native_uncertainty(tmp_path):
    import base64

    from test_videojev_probe import window

    from tools.videojev_benchmark.server import Scorer
    from tools.videojev_probe.contract import questions

    class Runtime:
        def predict(self, images, fps, qs):
            assert len(images) == 3 and fps == 2.5
            return responses(qs, 2), {"decision_seconds": 0.01}

    scorer = Scorer(Runtime(), tmp_path / "audit.jsonl")
    body = dict(
        model=MODEL,
        revision=REVISION,
        policy_sha256=POLICY_SHA,
        fps=2.5,
        request_id="test",
        session="camera",
        lane="events",
        images=[base64.b64encode(jpeg()).decode()] * 3,
        questions=questions(window(), ppe_variant="focused"),
    )
    with pytest.raises(ValueError):
        scorer.submit({**body, "policy_sha256": "different"})
    response = scorer.submit(body).result(timeout=2)
    assert response["request_id"] == "test"
    assert all(r["answer_index"] == 2 for r in response["raw"])
    scorer.queue.join()
    assert json.loads((tmp_path / "audit.jsonl").read_text())["status"] == "completed"


def server_body():
    import base64

    from test_videojev_probe import window

    from tools.videojev_probe.contract import questions

    return dict(
        model=MODEL,
        revision=REVISION,
        policy_sha256=POLICY_SHA,
        fps=2.5,
        request_id="test",
        session="camera",
        lane="events",
        images=[base64.b64encode(jpeg()).decode()] * 3,
        questions=questions(window(), ppe_variant="focused"),
    )


def test_malformed_requests_cannot_kill_scorer(tmp_path):
    from copy import deepcopy

    from tools.videojev_benchmark.server import Scorer

    class Runtime:
        def predict(self, images, fps, qs):
            return responses(qs, 2), {}

    scorer = Scorer(Runtime(), tmp_path / "audit.jsonl")
    body = server_body()
    invalid = [None, [], {**body, "request_id": ""}, {**body, "images": [1, 2]}]
    for key in ("session", "lane"):
        invalid.append({k: v for k, v in body.items() if k != key})
    duplicate = deepcopy(body)
    duplicate["questions"][1] = duplicate["questions"][0]
    invalid.extend([duplicate, {**body, "questions": [None] * 6}])
    for request in invalid:
        with pytest.raises(ValueError):
            scorer.submit(request)
    assert scorer.submit(body).result(timeout=2)["request_id"] == "test"
    scorer.queue.join()
    assert scorer.healthy
    assert len((tmp_path / "audit.jsonl").read_text().splitlines()) == 1


def test_cancelled_queue_work_is_skipped_and_running_work_keeps_scorer_healthy(
    tmp_path,
):
    from threading import Event

    from tools.videojev_benchmark.server import Scorer

    entered, release = Event(), Event()
    calls = []

    class Runtime:
        def predict(self, images, fps, qs):
            calls.append(1)
            entered.set()
            assert release.wait(2)
            return responses(qs, 2), {}

    scorer = Scorer(Runtime(), tmp_path / "audit.jsonl")
    first = scorer.submit(server_body())
    try:
        assert entered.wait(2)
        assert not first.cancel()  # GPU work has been claimed atomically.
        second = scorer.submit({**server_body(), "request_id": "queued"})
        assert second.cancel()
    finally:
        release.set()
    assert first.result(timeout=2)["request_id"] == "test"
    scorer.queue.join()
    assert calls == [1] and scorer.healthy
    rows = [
        json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()
    ]
    assert [row["status"] for row in rows] == ["completed", "cancelled_before_gpu"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_unused_question_metadata_is_rejected_before_queueing(
    tmp_path, value
):
    from tools.videojev_benchmark.server import Scorer

    class Runtime:
        def predict(self, images, fps, qs):
            return responses(qs, 2), {}

    scorer = Scorer(Runtime(), tmp_path / "audit.jsonl")
    body = server_body()
    body["questions"][0]["metadata"] = {"unused": [value]}
    with pytest.raises(ValueError, match="finite JSON"):
        scorer.submit(body)
    assert scorer.queue.unfinished_tasks == 0
    assert scorer.submit(server_body()).result(timeout=2)["request_id"] == "test"
    scorer.queue.join()
    assert scorer.healthy


def test_nonfinite_runtime_metadata_fails_request_without_killing_worker(tmp_path):
    from tools.videojev_benchmark.server import Scorer

    class Runtime:
        calls = 0

        def predict(self, images, fps, qs):
            self.calls += 1
            return responses(qs, 2), {
                "decision_seconds": float("nan") if self.calls == 1 else 0.01
            }

    audit = tmp_path / "audit.jsonl"
    scorer = Scorer(Runtime(), audit)
    with pytest.raises(ValueError, match="JSON compliant"):
        scorer.submit(server_body()).result(timeout=2)
    assert scorer.submit(server_body()).result(timeout=2)["request_id"] == "test"
    scorer.queue.join()
    assert scorer.healthy
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["failed", "completed"]


def test_audit_write_failure_fails_request_and_health_without_losing_queue_task(
    tmp_path,
):
    from tools.videojev_benchmark.server import Scorer

    class Runtime:
        def predict(self, images, fps, qs):
            return responses(qs, 2), {}

    scorer = Scorer(Runtime(), tmp_path / "audit.jsonl")
    scorer.audit = tmp_path  # Opening an existing directory must fail on every OS.
    with pytest.raises(RuntimeError, match="audit write failed"):
        scorer.submit(server_body()).result(timeout=2)
    scorer.queue.join()
    assert not scorer.healthy
    with pytest.raises(RuntimeError, match="unhealthy"):
        scorer.submit(server_body())
