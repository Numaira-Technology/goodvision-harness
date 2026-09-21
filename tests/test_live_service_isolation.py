"""Live status and publication must not inherit durable tracking latency."""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from test_local_gpu_runtime import rig as rig
from test_vlm_independent import (
    config,
    job,
    jpeg,
    observe_person,
    report,
    tracked_fusion,
    verdict,
)

from goodvision_harness import service
from goodvision_harness.contracts import FrameReference
from goodvision_harness.result_bus import ResultBus
from goodvision_harness.vlm_independent import (
    IndependentEventState,
    IndependentVLMBranch,
)


def forbidden(*args, **kwargs):
    raise AssertionError("lightweight health must not call a detailed worker")


def test_scene_publication_keeps_vlm_latest_and_preserves_feature_history():
    bus = ResultBus()
    vlm = report(IndependentEventState(config()), 1.0, 15, (verdict(),))
    bus.publish(vlm)
    stranger = {
        "type": "analysis_result",
        "frame": {"timestamp_seconds": 20},
        "judgments": [{"event_label": "stranger", "detected": True}],
    }
    bus.publish(stranger, promote_latest=False)
    assert bus.latest().payload == vlm
    assert bus.latest(feature="stranger").payload == stranger
    assert len(bus.records_after(0)) == 2
    bus.reset()
    bus.publish(stranger)  # Legacy mode retains the original promotion behavior.
    assert bus.latest().payload == stranger


def test_tracking_owned_safety_confirmation_reaches_delivery_without_replacing_vlm(
    rig, monkeypatch,
):
    runtime = rig.make()
    cfg = replace(config(), source_id=runtime.source_id)
    state, fusion = IndependentEventState(cfg), tracked_fusion(cfg)
    runtime._state["event_labels"] = ["smoking"]
    runtime._scene_result_publisher.reset(cfg.stream_session_id)
    delivered = []
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_delivery", SimpleNamespace(submit=delivered.append))
        for n, detected in enumerate((True, True, False, False)):
            at, sequence = round(1.0 + n * 0.8, 1), 10 + n * 8
            raw = report(state, at, sequence, (verdict(detected),))
            runtime.results.publish(raw)
            observe_person(fusion, at, sequence)
            assert fusion._associate(raw, (), set())
            runtime._scene_result_publisher.observe_reference(
                cfg.stream_session_id,
                FrameReference(sequence, at, "2026-09-21T08:00:00Z", 64, 48),
            )
            runtime._publish_scene_result(fusion.scene.snapshot(include_history=False))
            assert runtime.results.latest().payload["message_id"] == raw["message_id"]
            result = runtime.results.latest(feature="smoking").payload
            assert result["analysis_kind"] == "scene_state"
            assert result["judgments"][0]["detected"] is (n in (1, 2))
            assert result["objects"][0]["object_id"] == "sam-person"
            assert delivered[-1]["message_id"] == result["message_id"]
    assert len(delivered) == 4


def test_viewer_sequences_are_independent_and_do_not_wait_for_state_lock(rig):
    runtime = rig.make()
    runtime._camera_display_metadata.reset("run")
    source = runtime._camera_display_metadata.source_identity()
    held, release = Event(), Event()

    def hold_state():
        with runtime._state_lock:
            held.set()
            release.wait(5)

    thread = Thread(target=hold_state)
    thread.start()
    try:
        assert held.wait(1)
        with ThreadPoolExecutor(1) as pool:
            first = {**source, "display_instance_id": "one", "report_sequence": 100}
            assert pool.submit(runtime.record_display_metrics, first).result(1)[
                "accepted"
            ]
            second = {**source, "display_instance_id": "two", "report_sequence": 1}
            assert pool.submit(runtime.record_display_metrics, second).result(1)[
                "accepted"
            ]
            assert not pool.submit(runtime.record_display_metrics, first).result(1)[
                "accepted"
            ]
    finally:
        release.set()
        thread.join(2)
    assert runtime.status(lightweight=True)["local_camera_display"] == second
    for index in range(100):
        runtime.record_display_metrics({**first, "display_instance_id": str(index)})
    assert len(runtime._display_reports) == 64
    runtime._camera_display_metadata.reset("next-run")
    assert runtime.status(lightweight=True)["local_camera_display"] is None
    assert not runtime.record_display_metrics(second)["accepted"]


@pytest.mark.parametrize("reset_during_admission", [False, True])
def test_disk_admission_releases_status_lock_but_orders_native_commit(
    rig,
    monkeypatch,
    reset_during_admission,
):
    runtime = rig.make()
    entered, release, commit_started = Event(), Event(), Event()
    order = []

    class Coordinator:
        def submit(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return True

        stats = staticmethod(forbidden)
        retained_sources = staticmethod(lambda batch: ())

    monkeypatch.setattr(service, "VideoSessionTrackingCoordinator", Coordinator)
    runtime._rich_labels = Coordinator()
    runtime._active_rich_session_id = "run"
    runtime._session = SimpleNamespace(
        stream_session_id="run", mode="stream", running=True
    )
    runtime._scene = SimpleNamespace(
        mode="active",
        event_labels=(),
        observe_frame=lambda *args, **kwargs: order.append(
            ("retain", kwargs["retain_source"])
        ),
    )
    runtime._last_scene_publish_monotonic = time.monotonic() + 100
    monkeypatch.setattr(
        runtime, "_on_tracking_batch", lambda *args, **kwargs: order.append("commit")
    )
    generation, revision = runtime._generation, runtime._rich_config_revision

    def commit():
        commit_started.set()
        runtime._on_processing_batch(revision, SimpleNamespace())

    try:
        with ThreadPoolExecutor(3) as pool:
            admission = pool.submit(
                runtime._on_display_frame, jpeg(), 1.0, 15, generation=generation
            )
            assert entered.wait(1)
            result = pool.submit(runtime.status, lightweight=True).result(1)
            assert result["source_frames_received"] == 1
            assert runtime.preview_frames.latest()[1].frame_sequence == 15
            callback = pool.submit(commit)
            assert commit_started.wait(1)
            assert not callback.done()
            if reset_during_admission:
                with runtime._state_lock:
                    runtime._generation += 1
                    runtime._rich_config_revision += 1
                    runtime._session = None
                    runtime.preview_frames.reset()
            release.set()
            admission.result(2)
            callback.result(2)
            assert order == (
                [] if reset_during_admission else [("retain", True), "commit"]
            )
            if reset_during_admission:
                assert runtime.preview_frames.latest() is None
    finally:
        release.set()
        runtime._session = runtime._rich_labels = runtime._scene = None


def test_failed_reports_do_not_refresh_success_health_or_call_fusion(rig):
    runtime = rig.make()
    branch = IndependentVLMBranch(config())
    # Drive the real parent reader with controlled reports, without a child/API.
    from queue import Queue

    branch._outgoing.close()
    branch._outgoing = Queue()
    branch._process = SimpleNamespace(pid=123, is_alive=lambda: True)
    branch.fusion = SimpleNamespace(stats=forbidden, offer=lambda *args: None)
    reader = Thread(target=branch._read)
    reader.start()
    try:
        state = IndependentEventState(config())
        first = report(state, 1.0, 15, (verdict(),))
        first["timing"]["evidence_received_monotonic"] = time.monotonic()
        branch._outgoing.put({"kind": "report", "report": first})
        from test_vlm_independent import await_condition

        await_condition(lambda: branch.health().get("published_reports", 0) == 1)
        previous = branch._last_success_monotonic
        failed = report(state, 2.0, 30, ())
        failed.update(
            state="failed", error_code="invalid_response", analysis_complete=False
        )
        failed["timing"]["evidence_received_monotonic"] = time.monotonic()
        branch._outgoing.put({"kind": "report", "report": failed})
        await_condition(lambda: branch.health().get("published_reports", 0) == 2)
        assert branch._last_success_monotonic == previous
        with branch._lock:
            branch._status["state"] = "running"
            branch._last_source_monotonic = time.monotonic()
            branch._last_success_monotonic -= (
                runtime.settings.ready_max_result_age_seconds + 1
            )
        health = runtime._vlm_health(branch)
        assert health["last_attempt_state"] == "failed"
        assert health["last_attempt_error"] == "invalid_response"
        assert health["readiness_reason"] == "analysis_stale"
        assert not health["ready"]
    finally:
        branch._stop.set()
        reader.join(1)
        branch._incoming.close()


def test_without_tracking_dual_vlm_uses_application_capacity(rig, monkeypatch):
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "2")
    runtime = rig.make()
    assert not runtime._tracking_requested
    assert runtime.tracking_camera_capacity() == 2
    runtime._tracking_requested = True
    assert runtime.tracking_camera_capacity() == 1


@pytest.mark.parametrize("target_fps", [15, 30])
def test_old_display_config_does_not_throttle_original_video(
    rig, monkeypatch, target_fps
):
    from test_video_connection import ControlledAdapter

    from goodvision_harness import video_connection

    monkeypatch.setenv("LOCAL_CAMERA_TARGET_FPS", str(target_fps))
    runtime = rig.make()
    runtime.config = replace(
        runtime.config, display=replace(runtime.config.display, fps=5)
    )
    options = []

    def adapter(url, **kwargs):
        options.append(kwargs)
        return ControlledAdapter(url, **kwargs)

    monkeypatch.setattr(video_connection, "NetworkStreamAdapter", adapter)
    runtime.connect_stream("rtsp://camera.test/live")
    assert options[0]["output_fps"] == target_fps
    assert runtime.config.display.fps == 5


def test_invalid_coordinate_response_retains_safe_diagnosis(monkeypatch):
    import json

    import httpx

    from goodvision_harness import scene_model
    from goodvision_harness.vlm_independent_model import IndependentVLMInspector

    response = {
        "subjects": [{"subject_id": "p1", "bbox": [100, 100, 500, 900]}],
        "verdicts": [
            {
                "event_label": "smoking",
                "subject_ids": ["p1"],
                "detected": True,
                "confidence": 0.9,
            }
        ],
    }
    monkeypatch.setattr(
        scene_model,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": json.dumps(response),
                                    "reasoning_content": "PRIVATE_MODEL_TEXT",
                                },
                            }
                        ],
                    },
                )
            ),
        ),
    )
    cfg = config()
    inspector = IndependentVLMInspector(
        cfg.model_settings, model="test", api_key="PRIVATE_KEY"
    )
    try:
        (observed,) = inspector.inspect(job(cfg, 1.0, 15))
        assert observed.detected is True
        assert not observed.subject_boxes
        assert observed.grounding_issues == ("invalid_subject_bbox",)
        assert "PRIVATE" not in str(observed)
    finally:
        inspector.close()
