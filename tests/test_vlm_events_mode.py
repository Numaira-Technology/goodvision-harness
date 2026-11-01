"""Scene-level event delivery remains usable without SAM or face analysis."""

import time
from dataclasses import replace
from threading import Event

import pytest
from fastapi.testclient import TestClient
from test_vlm_independent import ROOT, await_condition, config, jpeg

from goodvision_harness import service
from goodvision_harness.config import load_config
from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.vlm_admission import AdmissionSettings, SharedAdmission
from goodvision_harness.vlm_independent import IndependentVLMBranch
from goodvision_harness.vlm_independent_model import IndependentVerdict


class EventInspector:
    def inspect(self, job):
        at = job.frames[-1].timestamp_seconds
        time.sleep(0.7 if at < 1 else 0.01)
        return tuple(
            IndependentVerdict(label, (), at < 1, None, (), ("scene_only",))
            for label in job.request.event_labels
        )

    def close(self):
        pass


def event_factory(_config):
    return EventInspector()


class EventBranch(IndependentVLMBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inspector_factory=event_factory, **kwargs)


def test_late_positive_is_delivered_without_overwriting_newer_negative():
    branch = EventBranch(replace(config(), output_mode="events", labels=("fall",)))
    branch.start()
    try:
        for seq, at in enumerate([0, 0.4, 0.8]):
            branch.offer(jpeg(), at, seq)
        time.sleep(0.1)
        for seq, at in enumerate([1.2, 1.6], 3):
            branch.offer(jpeg(), at, seq)
        await_condition(lambda: len(branch.results.records_after(0)) == 2)
        rows = [record.payload for record in branch.results.records_after(0)]
        assert [r["frame"]["timestamp_seconds"] for r in rows] == [1.6, 0.8]
        assert rows[1]["judgments"][0]["detected"] is True
        assert rows[1]["historical_only"] is True
        latest = branch.results.latest().payload
        assert latest["judgments"][0]["detected"] is False
        assert latest["event_scope"] == "scene"
        assert latest["judgments"][0]["confirmation_state"] == "scene_observation"
        assert latest["judgments"][0]["localization_state"] == "not_requested"
        assert branch.wait_until_idle(3)
        assert branch.stats()["scheduled_observation_windows"] == 2
        assert branch.health()["last_result_at"] == rows[0]["emitted_at"]
    finally:
        branch.close(timeout_seconds=5)


class RateLimitedInspector:
    def inspect(self, _job):
        raise ModelAPIError(
            "private upstream response", code="http_error", status_code=429
        )

    def close(self):
        pass


def rate_limited_factory(_config):
    return RateLimitedInspector()


class MalformedInspector(RateLimitedInspector):
    def inspect(self, _job):
        error = ModelAPIError("private upstream response", code="invalid_response")
        error.validation_reason = "invalid_json_syntax"
        error.response_diagnostics = {
            "finish_reason": "stop",
            "content_kind": "text",
            "content_characters": 42,
            "private": "secret",
            "content_parts": "secret",
        }
        raise error


def malformed_factory(_config):
    return MalformedInspector()


class DiagnosedEventInspector(EventInspector):
    def inspect(self, job):
        self.last_http_timing = {
            "send_request_body_seconds": 0.1,
            "response_bytes": 100,
            "total_seconds": float("nan"),
            "prompt_tokens": "secret",
            "Authorization": "secret",
        }
        self.last_inspection_timing = {
            "request_prepare_seconds": 0.01,
            "private": "secret",
        }
        self.last_response_diagnostics = {
            "finish_reason": "stop",
            "content_wrapper": "code_fence",
            "private": "secret",
        }
        return super().inspect(job)


def diagnosed_factory(_config):
    return DiagnosedEventInspector()


def test_valid_reply_format_crosses_process_boundary_without_raw_text():
    branch = IndependentVLMBranch(
        replace(config(), output_mode="events"), inspector_factory=diagnosed_factory
    )
    branch.start()
    try:
        for seq, at in enumerate([0, 0.4, 0.8]):
            branch.offer(jpeg(), at, seq)
        assert branch.wait_until_idle(3)
        row = branch.results.latest().payload
        assert row["analysis_complete"]
        assert row["timing"]["response_diagnostics"] == {
            "finish_reason": "stop",
            "content_wrapper": "code_fence",
        }
        assert row["timing"]["http_timing"] == {
            "send_request_body_seconds": 0.1,
            "response_bytes": 100,
        }
        assert row["timing"]["inspection_timing"] == {"request_prepare_seconds": 0.01}
        assert "secret" not in str(row)
    finally:
        branch.close(timeout_seconds=5)


def test_malformed_reply_metadata_crosses_process_boundary_without_raw_text():
    branch = IndependentVLMBranch(
        replace(config(), output_mode="events"), inspector_factory=malformed_factory
    )
    branch.start()
    try:
        for seq, at in enumerate([0, 0.4, 0.8]):
            branch.offer(jpeg(), at, seq)
        assert branch.wait_until_idle(3)
        row = branch.results.latest().payload
        assert row["timing"]["response_diagnostics"] == {
            "finish_reason": "stop",
            "content_kind": "text",
            "content_characters": 42,
        }
        assert row["timing"]["validation_reason"] == "invalid_json_syntax"
        assert "secret" not in str(row) and "private upstream" not in str(row)
    finally:
        branch.close(timeout_seconds=5)


def test_rate_limit_status_is_counted_without_exposing_upstream_text():
    branch = IndependentVLMBranch(
        replace(config(), output_mode="events"),
        inspector_factory=rate_limited_factory,
    )
    branch.start()
    try:
        for seq, at in enumerate([0, 0.4, 0.8]):
            branch.offer(jpeg(), at, seq)
        assert branch.wait_until_idle(3)
        result = branch.results.latest().payload
        assert result["error_code"] == "http_error"
        assert result["timing"]["http_status"] == 429
        assert not result["analysis_complete"]
        assert branch.stats()["http_status_429"] == 1
        assert "private upstream" not in str(result)
    finally:
        branch.close(timeout_seconds=5)


def test_admission_timeout_is_not_reported_as_validated_model_response(tmp_path):
    admission = AdmissionSettings(
        str(tmp_path / "gate.sqlite"),
        global_limit=1,
        per_stream_limit=1,
        queue_timeout_seconds=0.05,
    )
    gate = SharedAdmission(admission)
    branch = EventBranch(
        replace(
            config(),
            output_mode="events",
            admission=admission,
            diagnostics_enabled=True,
        )
    )
    try:
        with gate.slot("holder", stop=Event(), source_deadline=time.monotonic() + 10):
            branch.start()
            for seq, at in enumerate([0, 0.4, 0.8]):
                branch.offer(jpeg(), at, seq)
            await_condition(lambda: branch.results.latest() is not None)
            assert branch.results.latest().payload["error_code"] == "admission_timeout"
            await_condition(lambda: bool(branch.diagnostics.snapshot(())["records"]))
            row = branch.diagnostics.snapshot(())["records"][0]
            assert row["response_outcome"] == "failed"
            assert row["result"] == "unavailable"
    finally:
        branch.close(timeout_seconds=5)


def test_events_configuration_and_admission_settings(monkeypatch, tmp_path):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    monkeypatch.setenv("SCENE_VLM_OUTPUT_MODE", "events")
    monkeypatch.setenv("VLM_ADMISSION_DATABASE", str(tmp_path / "shared.db"))
    monkeypatch.setenv("VLM_GLOBAL_MAX_IN_FLIGHT", "12")
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.runtime.scene_vlm_output_mode == "events"
    assert cfg.runtime.scene_global_max_in_flight == 12
    monkeypatch.setenv("SCENE_INDEPENDENT_VLM", "false")
    with pytest.raises(ValueError, match="events mode requires"):
        load_config(ROOT / "config.yaml")


def test_twelfth_camera_can_publish_event_only_results_without_tracking(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "12")
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "12")
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    monkeypatch.setattr(service, "IndependentVLMBranch", EventBranch)
    base = load_config(ROOT / "config.yaml")
    cfg = replace(
        base,
        runtime=replace(
            base.runtime, scene_mode="active", scene_vlm_output_mode="events"
        ),
        rich_labeling=replace(
            base.rich_labeling,
            enabled=False,
            tracking_enabled=False,
            processing_method=None,
        ),
        events=replace(base.events, directory=tmp_path / "events"),
        service=replace(base.service, webhook=replace(base.service.webhook, url=None)),
    )
    runtime = service.HarnessRuntime(cfg)
    runtime._aiclub_api_key = "TEST_KEY"
    try:
        with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
            assert runtime.tracking_camera_capacity() == 12
            runtime.start_camera(dry_run=False, features=["fall"])
            branch = runtime._vlm_branch
            assert branch.config.output_mode == "events"
            assert branch.fusion is None
            assert branch._display_fusion is None
            for seq, at in enumerate([0, 0.4, 0.8]):
                branch.offer(jpeg(), at, seq)
            await_condition(lambda: branch.results.latest() is not None)
            payload = client.get("/v1/vlm/results/latest").json()
            assert payload["analysis_complete"]
            assert payload["event_scope"] == "scene"
            assert payload["judgments"][0]["detected"] is True
            assert payload["objects"] == []
    finally:
        runtime.stop()
