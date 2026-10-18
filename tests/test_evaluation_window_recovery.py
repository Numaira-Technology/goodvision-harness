"""Window-local recovery through real accounting, capture and answer boundaries.

Model transports, elapsed backoff clocks and noted detector boundaries are replaced.
No GPU, provider credentials or paid requests are used.
"""
import hashlib
import io
import json
import time
from functools import partial
from types import SimpleNamespace
from urllib.error import HTTPError

import httpx
import pytest
from PIL import Image

from goodvision_harness.evaluation import stream_ledger, summary
from goodvision_harness import model_api
from goodvision_harness.evaluation.answer_artifact import build_answer_artifact
from goodvision_harness.evaluation.evidence import EvidenceCapture, NATIVE_SKIP_POLICY
from goodvision_harness.evaluation.failures import failure_policy, native_report_failure_policy
from goodvision_harness.evaluation.model_ledger import LedgerBlocked, ModelLedger
from goodvision_harness.evaluation.native_stream import NativeCapture
from goodvision_harness.evaluation.profile import EvaluationEventProfile
from goodvision_harness.evaluation.recording import publish_once
from goodvision_harness.evaluation.stream_recording import StreamingRecorder
from goodvision_harness.model_api import ModelAPIError, ModelAPISettings
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_model import TemporalVLMInspector
from goodvision_harness.scene_runtime import TemporalInspection
from goodvision_harness.vlm_independent import IndependentVLMBranch, VLMProcessConfig
from goodvision_harness.vlm_independent_model import IndependentVerdict


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, delay):
        self.waits.append(delay)
        self.now += delay


def rejection(status=429, code="http_error", retry_after=0):
    error = ModelAPIError("controlled provider rejection", code=code, status_code=status)
    error.evaluation_diagnostic = {"http_status": status, "retry_after_seconds": retry_after}
    return error


def response(value, response_id="response-1"):
    return {"id": response_id, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(value)}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    seed.mkdir()
    files = {}
    for name in ("model-requests.jsonl", "model-responses.jsonl", "attempt-ledger.jsonl"):
        path = seed / name
        path.write_bytes(b"")
        files[name] = {"path": name, "sha256": hashlib.sha256(b"").hexdigest()}
    manifest = seed / "manifest.json"
    digest = publish_once(manifest, {"schema_version": "unified-model-ledger-seed-v1", "files": files})
    item = ModelLedger(tmp_path / "ledger", seed_manifest_path=manifest,
                       seed_manifest_sha256=digest, owner_lock_path=tmp_path / "owner.lock")
    monkeypatch.setattr(stream_ledger, "_ledger", item)
    factory = stream_ledger.LedgerInspectorFactory(str(item.root), str(manifest), digest,
                                                  str(tmp_path / "owner.lock"))
    monkeypatch.setattr(stream_ledger, "_identity", factory)
    yield item, factory
    item.close()


def rows(ledger, name):
    path = ledger.root / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def inspection(sequence):
    sink = io.BytesIO()
    Image.new("RGB", (16, 16), "gray").save(sink, format="JPEG")
    frames = tuple(EvidenceFrame("session", index, float(index), sink.getvalue())
                   for index in (sequence * 2, sequence * 2 + 1))
    request = InspectionRequest(f"request-{sequence}", (), "independent_source", ("smoking",),
                                frames[0].timestamp_seconds, frames[-1].timestamp_seconds)
    return TemporalInspection("session", "source", request, frames)


@pytest.mark.parametrize("exhausted", [False, True])
def test_429_backoff_preserves_attempts_and_next_window(ledger, monkeypatch, exhausted):
    item, factory = ledger
    clock = Clock()
    monkeypatch.setattr(stream_ledger, "time", clock)
    replies = iter([rejection(), rejection(retry_after=90),
                    rejection() if exhausted else response({"smoking": False}),
                    response({"smoking": True}, "response-2")])

    async def transport(self, payload, headers, *, timing):
        value = next(replies)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(TemporalVLMInspector, "_request", transport)
    settings = ModelAPISettings(provider="openrouter", max_retries=0)
    config = VLMProcessConfig("session", "source", 1, ("smoking",), settings,
                              "qwen/qwen3-vl-8b-instruct", api_key="test-only",
                              output_mode="events", allowed_event_labels=("smoking",),
                              scene_only_event_labels=())
    inspector = factory(config)
    try:
        if exhausted:
            with pytest.raises(ModelAPIError) as error:
                inspector.inspect(inspection(0))
            assert error.value.status_code == 429
        else:
            assert inspector.inspect(inspection(0))[0].detected is False
        assert clock.waits == ([60.0, 90.0, 60.0] if exhausted else [60.0, 90.0])
        assert inspector.last_rate_limit_backoff_seconds == sum(clock.waits)
        assert inspector.inspect(inspection(1))[0].detected is True
        requests = rows(item, "model-requests.jsonl")
        assert [row["attempt"] for row in requests] == [1, 2, 3, 1]
        assert len({row["request_sha256"] for row in requests[:3]}) == 1
        assert len(rows(item, "model-responses.jsonl")) == 4
        assert not item.in_flight and not item.blocked
    finally:
        inspector.close()


@pytest.mark.parametrize("code", ["http_error", "overloaded", "inspection_failed"])
def test_explicit_429_is_window_local_even_after_worker_error_normalization(code):
    policy = native_report_failure_policy({"state": "failed", "error_code": code,
                                           "timing": {"http_status": 429}})
    assert policy["scope"] == "window"
    assert policy["action"] == "preserve_and_continue"


@pytest.mark.parametrize("unknown", [False, True])
def test_reloaded_window_budget_does_not_dispatch_or_block_next_input(ledger, monkeypatch, unknown):
    item, factory = ledger
    monkeypatch.setattr(stream_ledger, "time", Clock())
    replies = iter(([rejection(None, "timeout")] if unknown else [rejection()] * 3)
                   + [response({"smoking": True})])

    async def transport(self, payload, headers, *, timing):
        value = next(replies)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(TemporalVLMInspector, "_request", transport)
    config = VLMProcessConfig("session", "source", 1, ("smoking",),
        ModelAPISettings(provider="openrouter", max_retries=0), "test-model", api_key="test-only",
        output_mode="events", allowed_event_labels=("smoking",), scene_only_event_labels=())
    inspector = factory(config)
    try:
        with pytest.raises(ModelAPIError):
            inspector.inspect(inspection(0))
    finally:
        inspector.close()
        item.close()
    # Reload the saved cumulative ledger instead of mutating counters in memory.
    restored = ModelLedger(factory.directory, seed_manifest_path=factory.seed_manifest,
        seed_manifest_sha256=factory.seed_sha256, owner_lock_path=factory.owner_lock)
    monkeypatch.setattr(stream_ledger, "_ledger", restored)
    inspector = factory(config)
    try:
        before = len(rows(restored, "model-requests.jsonl"))
        with pytest.raises(LedgerBlocked) as blocked:
            inspector.inspect(inspection(0))
        assert blocked.value.scope == "source"
        assert blocked.value.code == ("unresolved_same_input_request" if unknown
                                      else "same_input_attempts_exhausted")
        assert len(rows(restored, "model-requests.jsonl")) == before
        assert inspector.inspect(inspection(1))[0].detected is True
        assert len(rows(restored, "model-requests.jsonl")) == before + 1
    finally:
        inspector.close()
        restored.close()


@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_and_balance_rejections_remain_shared(status):
    assert native_report_failure_policy({"state": "failed", "error_code": "http_error",
        "timing": {"http_status": status}})["scope"] == "shared"
    assert failure_policy(rejection(status))["scope"] == "shared"


def failed_report(status=429, code="http_error"):
    return {"stream_session_id": "session", "source_id": "source", "request_id": "request-0",
            "vlm_lane": "information", "state": "failed", "error_code": code,
            "timing": {"http_status": status},
            "evidence": [{"frame_sequence": 0, "timestamp_seconds": 0.0}],
            "observations": [], "judgments": []}


def saved_evidence(status=429, code="http_error", finished=True, frames=True):
    capture = EvidenceCapture(video_id="clip", stream_session_id="session", source_id="source",
        source={"width": 16, "height": 16, "duration_seconds": 1},
        provenance={"coverage_policy": NATIVE_SKIP_POLICY, "scheduling_mode": "native_streaming"},
        frame_plan=[{"frame": 0, "t": 0.0, "jpeg_sha256": "picture"}],
        window_plan=[{"window_id": "window-0", "start": 0.0, "end": 0.0,
                      "event_labels": ["smoking"], "report_lanes": ["information"]}])
    if frames:
        capture.tracking_frame(frame=0, timestamp_seconds=0.0, subjects=[],
                               raw_reference="tracking.json", jpeg_sha256="picture")
    capture.independent_report(failed_report(status, code), window_id="window-0",
                               raw_reference="window.json", publication_cursor=1)
    return capture.seal(source_eof=finished, tracking_drained=True, events_drained=True,
                        fusion_drained=True)


def artifact(tmp_path, evidence, name="answer"):
    question = {"qa_id": name, "video_id": "clip", "question": "Is anyone smoking?",
                "question_form": "event_presence_binary", "target_label": "smoking"}
    path = tmp_path / f"{name}.json"
    digest = publish_once(path, build_answer_artifact(question, evidence))
    return path, digest


class AnswerTransport:
    settings = SimpleNamespace(max_retries=0)

    def __init__(self, replies):
        self.replies = iter(replies)

    def chat_completion(self, **kwargs):
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return value


def answer(path, digest, output, item, client):
    return summary.execute_summary(path, artifact_sha256=digest, output=output, client=client,
        ledger=item, model="qwen/qwen3-vl-8b-instruct", parameters={"max_tokens": 1024},
        max_input_bytes=100000)


@pytest.mark.parametrize(("status", "code"), [(429, "http_error"), (None, "invalid_response"),
    (None, "unresolved_same_input_request"), (None, "same_input_attempts_exhausted")])
def test_saved_window_gap_does_not_veto_final_answer(tmp_path, ledger, status, code):
    evidence = saved_evidence(status, code)
    original = json.dumps(evidence, sort_keys=True)
    path, digest = artifact(tmp_path, evidence)
    client = AnswerTransport([response({"status": "insufficient_evidence", "final_answer": None,
                                       "evidence_refs": []})])
    result = answer(path, digest, tmp_path / "output", ledger[0], client)
    assert result["status"] == "insufficient_evidence"
    assert len(rows(ledger[0], "model-requests.jsonl")) == 1
    assert evidence["failures"] and evidence["observation_manifest"]["complete"] is False
    assert json.dumps(evidence, sort_keys=True) == original


@pytest.mark.parametrize("case", ["auth", "unfinished", "no_observation", "shared"])
def test_unsafe_or_empty_evidence_is_not_sent(tmp_path, ledger, case):
    evidence = saved_evidence(status=401 if case == "auth" else 429,
                              finished=case != "unfinished", frames=case != "no_observation",
                              code="inspection_failed" if case == "shared" else "http_error")
    if case == "shared":
        evidence = saved_evidence(status=None, code="inspection_failed")
    path, digest = artifact(tmp_path, evidence)
    result = answer(path, digest, tmp_path / "output", ledger[0], AnswerTransport([]))
    assert result["status"] == "not_called"
    assert rows(ledger[0], "model-requests.jsonl") == []


def test_summary_429_does_not_require_batch_recovery():
    assert failure_policy(rejection())["scope"] == "source"
    assert failure_policy(LedgerBlocked("ledger_unavailable"))["scope"] == "shared"


@pytest.mark.parametrize("exhausted", [False, True])
def test_summary_429_waits_then_continues_without_restarting(tmp_path, ledger, monkeypatch, exhausted):
    clock = Clock()
    monkeypatch.setattr(summary, "time", clock)
    evidence = saved_evidence()
    path, digest = artifact(tmp_path, evidence)
    ok = response({"status": "insufficient_evidence", "final_answer": None, "evidence_refs": []})
    client = AnswerTransport([rejection(), rejection(), rejection() if exhausted else ok, ok])
    first = answer(path, digest, tmp_path / "first", ledger[0], client)
    assert first["status"] == ("request_failure" if exhausted else "insufficient_evidence")
    assert clock.waits == ([60.0, 60.0, 60.0] if exhausted else [60.0, 60.0])
    second_path, second_digest = artifact(tmp_path, evidence, "second")
    second = answer(second_path, second_digest, tmp_path / "second", ledger[0], client)
    assert second["status"] == "insufficient_evidence"
    assert [row["attempt"] for row in rows(ledger[0], "model-requests.jsonl")] == [1, 2, 3, 1]
    assert len(rows(ledger[0], "model-responses.jsonl")) == 4


def test_window_errors_do_not_poison_native_capture(tmp_path):
    profile = EvaluationEventProfile(("smoking",), (), (), "source", "profile")
    recorder = StreamingRecorder(tmp_path / "capture", video_id="clip", stream_session_id="session",
                                source_id="source", source={"width": 16, "height": 16},
                                provenance={}, profile=profile)
    capture = NativeCapture(recorder, profile=profile, source_fps=2, native_batches=False,
                            vlm_enabled=True)
    capture.tracker = SimpleNamespace(stats=lambda: {"state": "running", "failed": 0})
    capture.worker = SimpleNamespace(stats=lambda: {"state": "running", "failed_requests": 3})
    try:
        for report in (failed_report(), failed_report(None, "invalid_response")):
            capture.independent_report(report)
            capture.check_runtime()
        assert capture._shared_failure is False
        assert len(list((recorder.root / "records").glob("*-independent.json"))) == 2
    finally:
        capture.fusion.close()
        capture.scene.close()


@pytest.mark.parametrize("status", [401, 402, 403, None])
def test_summary_never_retries_non_429_or_unknown_outcome(tmp_path, ledger, monkeypatch, status):
    clock = Clock()
    monkeypatch.setattr(summary, "time", clock)
    path, digest = artifact(tmp_path, saved_evidence())
    client = AnswerTransport([rejection(status, "timeout" if status is None else "http_error")])
    if status is None:
        result = answer(path, digest, tmp_path / "first", ledger[0], client)
        assert result["status"] == "request_failure"
        # A new output directory cannot evade the existing unknown request block.
        again = answer(path, digest, tmp_path / "again", ledger[0], client)
        assert again["failure_policy"]["code"] == "unresolved_same_input_request"
        assert ledger[0].blocked
    else:
        with pytest.raises(ModelAPIError):
            answer(path, digest, tmp_path / "first", ledger[0], client)
    assert len(rows(ledger[0], "model-requests.jsonl")) == 1
    assert len(rows(ledger[0], "model-responses.jsonl")) == 1
    assert clock.waits == []


def test_long_retry_after_cools_down_without_retry_or_batch_stop(tmp_path, ledger, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(summary, "time", clock)
    path, digest = artifact(tmp_path, saved_evidence())
    result = answer(path, digest, tmp_path / "output", ledger[0],
                    AnswerTransport([rejection(retry_after=600)]))
    assert result["failure_policy"]["scope"] == "source"
    assert result["status"] == "request_failure"
    assert clock.waits == [600.0]
    assert result["rate_limit_backoff_seconds"] == 600.0
    assert result["model_seconds"] == 0.0
    assert len(rows(ledger[0], "model-requests.jsonl")) == 1


@pytest.mark.parametrize("pooled", [False, True])
def test_real_answer_client_preserves_provider_retry_after(tmp_path, ledger, monkeypatch, pooled):
    clock = Clock()
    monkeypatch.setattr(summary, "time", clock)
    settings = ModelAPISettings(provider="openrouter", max_retries=0)
    client_type = model_api.PooledModelAPIClient if pooled else model_api.ModelAPIClient
    client = client_type(settings, api_key="test-only")
    if pooled:
        client._http.close()
        client._http = httpx.Client(transport=httpx.MockTransport(lambda request:
            httpx.Response(429, headers={"Retry-After": "600"}, json={"error": "limited"})))
    else:
        def reject(request, **kwargs):
            raise HTTPError(request.full_url, 429, "limited", {"Retry-After": "600"}, io.BytesIO(b"{}"))
        monkeypatch.setattr(model_api, "urlopen", reject)
    path, digest = artifact(tmp_path, saved_evidence())
    try:
        result = answer(path, digest, tmp_path / "output", ledger[0], client)
        assert clock.waits == [600.0]
        assert result["failure_policy"]["scope"] == "source"
        assert len(rows(ledger[0], "model-requests.jsonl")) == 1
        assert rows(ledger[0], "model-responses.jsonl")[0]["error"]["retry_after_seconds"] == 600.0
    finally:
        if pooled:
            client.close()


class WindowTransport:
    """External inference substitute used by the real spawned worker."""
    def __init__(self, failure_kind):
        self.failure_kind = failure_kind

    def inspect(self, job):
        if job.frames[-1].timestamp_seconds <= 1.0:
            if self.failure_kind == "429":
                raise rejection()
            if self.failure_kind.startswith("shared_"):
                raise LedgerBlocked(self.failure_kind.removeprefix("shared_"))
            raise LedgerBlocked(self.failure_kind, scope="source")
        return (IndependentVerdict("smoking", (), False, None, (), ("scene_only",)),)

    def close(self):
        pass


def window_transport_factory(config, *, failure_kind):
    return WindowTransport(failure_kind)


@pytest.mark.parametrize("failure_kind", ["429", "same_input_attempts_exhausted",
    "unresolved_same_input_request", "shared_model_ledger_unavailable",
    "shared_same_input_attempts_exhausted"])
def test_real_native_worker_isolates_windows_but_stops_shared_failures(tmp_path, failure_kind):
    profile = EvaluationEventProfile(("smoking",), (), (), "source", "profile")
    recorder = StreamingRecorder(tmp_path / "capture", video_id="clip", stream_session_id="session",
                                source_id="source", source={"width": 16, "height": 16},
                                provenance={}, profile=profile)
    capture = NativeCapture(recorder, profile=profile, source_fps=2, native_batches=False,
                            vlm_enabled=True)
    capture.tracker = SimpleNamespace(stats=lambda: {"state": "running", "failed": 0})
    config = VLMProcessConfig("session", "source", 1, ("smoking",),
        ModelAPISettings(provider="openrouter", max_retries=0), "test-model",
        output_mode="events", max_in_flight=1)
    worker = IndependentVLMBranch(config, fusion=capture.fusion,
        inspector_factory=partial(window_transport_factory, failure_kind=failure_kind),
        report_observer=capture.independent_report,
        rejected_report_observer=recorder.rejected_report)
    capture.worker = worker
    worker.start()
    try:
        for sequence in range(4):
            frame = inspection(sequence).frames[0]
            worker.offer(frame.jpeg_bytes, float(sequence), sequence)
            if sequence in (1, 3):
                deadline = time.monotonic() + 10
                expected = 1 if sequence == 1 else 2
                while worker.stats().get("published_reports", 0) < expected:
                    if failure_kind.startswith("shared_") and worker.stats().get("state") == "failed":
                        break
                    if not failure_kind.startswith("shared_"):
                        capture.check_runtime()
                    assert time.monotonic() < deadline, "worker did not publish the next window"
                    time.sleep(.02)
                if failure_kind.startswith("shared_"):
                    with pytest.raises(RuntimeError, match="native VLM worker failed"):
                        capture.check_runtime()
                    assert capture._shared_failure
                    return
        worker.finish_source()
        assert worker.wait_until_idle(10)
        capture.check_runtime()
        reports = [json.loads(path.read_text())["payload"]
                   for path in sorted((recorder.root / "records").glob("*-independent.json"))]
        if failure_kind == "429":
            assert any(row["timing"].get("http_status") == 429 for row in reports)
        else:
            assert any(row["error_code"] == failure_kind for row in reports)
        assert any(row["state"] == "completed" for row in reports)
        assert not capture._shared_failure
        assert worker.stats()["failed_requests"] >= 1
        assert worker.stats()["succeeded_requests"] >= 1
    finally:
        worker.close(timeout_seconds=5)
        capture.fusion.close()
        capture.scene.close()
