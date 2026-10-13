"""Actual service admission/routes stay live under stalled SAM and scene work."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from threading import Event, RLock, Thread
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_vlm_independent import await_condition, jpeg, stub_factory

from goodvision_harness import service
from goodvision_harness.config import load_config
from goodvision_harness.contracts import FrameReference
from goodvision_harness.vlm_independent import IndependentVLMBranch

ROOT = Path(__file__).resolve().parents[1]


class LocalBranch(IndependentVLMBranch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, inspector_factory=stub_factory, **kwargs)


@pytest.mark.parametrize("state", ["stopped", "stopping", "drained", "failed"])
def test_vlm_health_distinguishes_stop_from_failure(state):
    branch = SimpleNamespace(health=lambda: {
        "state": state, "last_source_age_seconds": 0,
        "last_success_age_seconds": 0,
    })
    result = service.HarnessRuntime._vlm_health(None, branch)
    assert result["readiness_reason"] == (
        "event_analysis_failed" if state == "failed" else state
    )
    assert not result["running"] and not result["ready"]


def test_failed_tracking_drain_preserves_inflight_vlm_results(monkeypatch):
    from test_vlm_independent import config

    branch = IndependentVLMBranch(config(), inspector_factory=stub_factory)
    runtime = service.HarnessRuntime.__new__(service.HarnessRuntime)
    runtime._state_lock = RLock()
    runtime._state = {}
    runtime._generation = 1
    session = SimpleNamespace(stream_session_id=branch.config.stream_session_id)
    runtime._session = session
    runtime._scene_worker = branch
    control = SimpleNamespace(lock=RLock(), state="running", error=None)
    monkeypatch.setattr(runtime, "_archive_run", lambda _: control)
    finished = []

    def finish(_reason, error):
        # Real process output must be published before the failure closes it.
        finished.append((error, branch.health(), branch.results.latest()))
        branch.close()

    monkeypatch.setattr(runtime, "_finish_scene_end", finish)
    coordinator = SimpleNamespace(stats=lambda: {"state": "failed"})
    branch.start()
    try:
        await_condition(lambda: branch.health()["state"] == "running")
        assert branch.offer(jpeg(), 0, 0)
        assert branch.offer(jpeg(), .8, 12)
        # Stop immediately: the requests are still queued or in flight.
        runtime._drain_native_end(1, session, None, coordinator, "stopped")
        assert len(finished) == 1
        error, health, latest = finished[0]
        assert error == service.PUBLIC_TRACKING_ERROR
        assert health["state"] == "drained"
        assert health["succeeded_requests"] >= 1
        assert health["active_requests"] == 0
        assert latest is not None
        assert control.state == "failed"  # Tracking failure remains visible.
    finally:
        branch.close(timeout_seconds=6)


def test_camera_vlm_runs_and_api_streams_while_main_state_lock_is_held(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.delenv("SCENE_VLM_OUTPUT_MODE", raising=False)
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    monkeypatch.setattr(service, "IndependentVLMBranch", LocalBranch)
    base = load_config(ROOT / "config.yaml")
    cfg = replace(
        base,
        runtime=replace(base.runtime, scene_mode="active", scene_independent_vlm=True),
        rich_labeling=replace(
            base.rich_labeling, enabled=False, tracking_enabled=False, backend="paddle"
        ),
        events=replace(base.events, directory=tmp_path / "events"),
        service=replace(base.service, webhook=replace(base.service.webhook, url=None)),
    )
    runtime = service.HarnessRuntime(cfg)
    runtime._aiclub_api_key = "TEST_KEY"
    release, held = Event(), Event()
    blocker = None
    try:
        with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
            runtime.start_camera(dry_run=False, features=["smoking", "fall"])
            branch = runtime._vlm_branch
            assert branch is not None
            await_condition(lambda: branch.stats()["state"] == "running")
            session_id = runtime._session.stream_session_id
            generation = runtime._generation

            def lock_main():
                with runtime._state_lock:
                    held.set()
                    release.wait(15.0)

            blocker = Thread(target=lock_main)
            blocker.start()
            assert held.wait(2.0)
            durations = []
            for seq in range(0, 61, 6):
                start = time.monotonic()
                assert runtime.push_camera_frame(
                    jpeg(),
                    width=64,
                    height=48,
                    stream_session_id=session_id,
                    generation=generation,
                    crop_version=0,
                    source_sequence=seq,
                    media_timestamp=seq / 15.0,
                )
                durations.append(time.monotonic() - start)
                time.sleep(0.07)
            await_condition(lambda: branch.stats().get("published_reports", 0) >= 3)
            assert max(durations) < 1.0
            independent = client.get("/v1/vlm/results/latest")
            assert independent.status_code == 200
            payload = independent.json()
            assert payload["vlm_lane"] == "information"
            assert payload["analysis_kind"] == "vlm_event"
            removed = {"events", "missing_event_labels", "association", "confirmation"}
            assert removed.isdisjoint(payload)
            assert payload["judgments"][0]["observed_detected"] is True
            assert payload["judgments"][0]["detected"] is None
            overall = client.get("/v1/results/overall/latest")
            assert overall.status_code == 200
            assert removed.isdisjoint(overall.json())
            assert overall.json()["message_id"] == payload["message_id"]
            assert (
                client.get("/v1/results/smoking/latest").json()["analysis_kind"]
                == "vlm_event"
            )
            status = client.get("/v1/vlm/status").json()
            assert status["output_mode"] == "parallel"
            assert set(status["lanes"]) == {"events", "compact"}
            assert status["independent"]
            assert status["stream_session_id"] == payload["stream_session_id"]
            assert status["generation"] == payload["generation"]
            assert set(status["event_labels"]) == {"smoking", "fall"}
            assert set(status["event_labels"]) - {
                j["event_label"] for j in payload["judgments"]
            } == {"fall"}
            assert not payload["analysis_complete"]
            assert client.get("/v1/vlm/associations").json()["associations"] == []

            # Consume the actual SSE generator without TestClient buffering an
            # intentionally unending HTTP response.
            async def read_event(*, overall_only=False):
                request = SimpleNamespace(
                    headers={}, is_disconnected=lambda: asyncio.sleep(0, result=False)
                )
                response = service._result_events_response(
                    request,
                    runtime,
                    after=0,
                    result_bus=branch.results,
                    event_name=None if overall_only else "vlm_report",
                    overall_only=overall_only,
                )
                chunk = await anext(response.body_iterator)
                await response.body_iterator.aclose()
                return chunk

            chunk = asyncio.run(read_event())
            assert b"event: vlm_report" in chunk
            assert session_id.encode() in chunk
            data = next(
                line[6:] for line in chunk.splitlines() if line.startswith(b"data: ")
            )
            assert removed.isdisjoint(json.loads(data))
            overall_chunk = asyncio.run(read_event(overall_only=True))
            assert b"event: overall_description" in overall_chunk
            overall_data = next(
                line[6:] for line in overall_chunk.splitlines()
                if line.startswith(b"data: ")
            )
            assert removed.isdisjoint(json.loads(overall_data))
            assert (
                branch.stats()["camera_source_dispatch"].get(
                    "unsent_sources_superseded", 0
                )
                > 0
            )
            release.set()
            blocker.join(2.0)
            runtime._last_frame_monotonic = None
            runtime._last_scene_inference_monotonic = None
            assert runtime.readiness() == (True, "ready")

            # The old fused projector must not replace independent safety results.
            await_condition(lambda: branch.wait_until_idle(0.02))
            old_id = runtime.results.latest(feature="smoking").payload["message_id"]
            latest_id = runtime.results.latest().payload["message_id"]
            # Exercise a real stranger projection without loading a Face model.
            runtime._state["event_labels"] = ["smoking", "stranger"]
            runtime._scene_result_publisher.observe_reference(
                session_id, FrameReference(100, 10.0, "now", 64, 48)
            )
            runtime._publish_scene_result(
                {
                    "state": "running",
                    "scene": {
                        "stream_session_id": session_id,
                        "source_id": runtime.source_id,
                        "timestamp_seconds": 10.0,
                        "entities": [{
                            "subject_id": "person-a", "observation_state": "visible",
                            "last_observed_timestamp_seconds": 10.0,
                            "frame_sequence": 100,
                            "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9},
                        }],
                        "events": [{
                            "event_id": "stranger-event", "event_label": "stranger",
                            "subject_ids": ["person-a"], "phase": "confirmed",
                        }],
                    },
                }
            )
            assert (
                runtime.results.latest(feature="smoking").payload["message_id"]
                == old_id
            )
            assert runtime.results.latest().payload["message_id"] == latest_id
            assert runtime.results.latest(feature="stranger") is not None
    finally:
        release.set()
        if blocker is not None:
            blocker.join(2.0)
        runtime.stop()
        branch = runtime._vlm_branch
        if branch is not None:
            branch.close(timeout_seconds=6.0)


def test_default_configuration_enables_independent_process(monkeypatch):
    monkeypatch.delenv("SCENE_INDEPENDENT_VLM", raising=False)
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    assert load_config(ROOT / "config.yaml").runtime.scene_independent_vlm is True


@pytest.mark.parametrize("cancel_preparation", [False, True])
def test_mixed_start_prepares_resident_face_outside_lock_before_admission(
    tmp_path, monkeypatch, cancel_preparation
):
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_kwargs: "")
    monkeypatch.setattr(service, "IndependentVLMBranch", LocalBranch)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("optional model readiness must not gate the VLM")

    base = load_config(ROOT / "config.yaml")
    cfg = replace(
        base,
        runtime=replace(base.runtime, scene_mode="active", scene_independent_vlm=True),
        rich_labeling=replace(
            base.rich_labeling, enabled=False, tracking_enabled=False, backend="paddle"
        ),
        face=replace(base.face, gallery_path=tmp_path / "gallery.json"),
        events=replace(base.events, directory=tmp_path / "events"),
        service=replace(base.service, webhook=replace(base.service.webhook, url=None)),
    )
    runtime = service.HarnessRuntime(cfg)
    runtime._aiclub_api_key = "TEST_KEY"
    monkeypatch.setattr(runtime, "_require_processing_ready", forbidden)
    preparing, release = Event(), Event()
    slot_calls, errors = [], []

    def warm_slot(face, slot, _image):
        assert runtime._session is None and runtime._vlm_branch is None
        slot_calls.append((face, slot))
        preparing.set()
        assert release.wait(6)
        return []

    monkeypatch.setattr(service.FaceRecognitionRuntime, "_infer_slot", warm_slot)

    def start():
        try:
            runtime.start_camera(dry_run=False, features=["smoking", "stranger"])
        except Exception as error:
            errors.append(error)

    starter = Thread(target=start)
    starter.start()
    try:
        assert preparing.wait(3)
        # Public status/stop must remain usable during slow model loading.
        status = []
        reader = Thread(target=lambda: status.append(runtime.status()))
        reader.start()
        reader.join(1)
        assert not reader.is_alive() and status
        assert runtime._session is None and runtime._vlm_branch is None
        if cancel_preparation:
            assert runtime.stop_session()
        release.set()
        starter.join(6)
        assert not starter.is_alive()
        if cancel_preparation:
            assert len(errors) == 1 and str(errors[0]) == "session start cancelled"
            assert runtime._session is None and runtime._vlm_branch is None
            return
        assert not errors
        branch = runtime._vlm_branch
        assert branch is not None
        await_condition(lambda: branch.stats()["state"] == "running")
        resident = runtime._face_runtime
        assert resident.warmed
        assert len(slot_calls) == cfg.face.inference_instances
        assert all(face is resident for face, _ in slot_calls)
        assert runtime.stop_session()
        await_condition(lambda: runtime._session is None)
        runtime.start_camera(dry_run=False, features=["smoking", "stranger"])
        assert runtime._face_runtime is resident and resident.warmed
        assert len(slot_calls) == cfg.face.inference_instances
    finally:
        release.set()
        starter.join(6)
        runtime.stop()
