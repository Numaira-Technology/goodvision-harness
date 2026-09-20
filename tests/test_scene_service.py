"""Exercise active scene lifecycle and rejection of the removed observer mode."""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from goodvision_harness import service
from goodvision_harness.config import load_config
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.privacy import PUBLIC_INSPECTION_ERROR, PUBLIC_TRACKING_ERROR
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]
BOX = NormalizedBBox(0.1, 0.1, 0.45, 0.95)


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    for name in (
        "SCENE_ENGINE_ENABLED",
        "SCENE_ENGINE_MODE",
        "RESULT_WEBHOOK_URL",
        "RICH_LABEL_ENABLED",
        "RICH_LABEL_TRACKING_ENABLED",
        "INFERENCE_PROVIDER",
        "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT",
        "LOCAL_CAMERA_DISPLAY_DELAY_SECONDS",
        "LOCAL_CAMERA_MOTION_WIDTH",
        "LOCAL_CAMERA_TARGET_FPS",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def runtime_factory(tmp_path, monkeypatch):
    instances = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("scene tests must not initialize a remote or face model")

    class ForbiddenFaceRuntime:
        def __init__(self, *_args, **_kwargs):
            forbidden()

    monkeypatch.setattr(service, "RichLabelHTTPClient", forbidden)
    monkeypatch.setattr(service, "FaceRecognitionRuntime", ForbiddenFaceRuntime)

    def make(*, enabled=True):
        config = load_config(ROOT / "config.yaml")
        config = replace(
            config,
            runtime=replace(
                config.runtime, scene_mode="active" if enabled else "disabled"
            ),
            rich_labeling=replace(
                config.rich_labeling,
                enabled=False,
                tracking_enabled=False,
            ),
            events=replace(config.events, directory=tmp_path / str(len(instances))),
            service=replace(
                config.service, webhook=replace(config.service.webhook, url=None)
            ),
        )
        runtime = HarnessRuntime(config)
        instances.append(runtime)
        return runtime

    yield make
    for runtime in instances:
        runtime.stop()


def _jpeg():
    ok, encoded = cv2.imencode(".jpg", np.zeros((160, 240, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_local_motion_assets_and_timing_config_are_delivered_without_models(
    runtime_factory,
):
    runtime = runtime_factory()
    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        settings = client.get("/v1/ui/info").json()["local_camera_display"]
        assert settings["target_fps"] == 15
        assert settings["motion_backend"] == "dense"
        assert settings["delay_seconds"] == 6
        assert settings["motion_width"] == 160
        assert settings["anchor_policy"] == "nearest_arrived"
        assert settings["allow_partial_boundaries"] is False
        assert settings["max_propagation_seconds"] == 1.5
        assert settings["max_history_seconds"] >= settings["delay_seconds"] + 4
        adapter = client.get("/assets/camera_display_dense.js")
        assert adapter.status_code == 200
        assert "createDenseMotionAdapter" in adapter.text
        vendor = client.get("/assets/opencv-4.13.0.js")
        assert vendor.status_code == 200
        assert hashlib.sha256(vendor.content).hexdigest() == (
            "63366510248adf3a7eddf3e793dd825404efb7df3749f4d6f8557c7fa4ca8aa0"
        )
        assert "immutable" in vendor.headers["cache-control"]


def test_display_working_resolution_override_does_not_change_original_output(
    runtime_factory, monkeypatch,
):
    monkeypatch.setenv("LOCAL_CAMERA_MOTION_WIDTH", "240")
    monkeypatch.setenv("LOCAL_CAMERA_DISPLAY_DELAY_SECONDS", "7")
    monkeypatch.setenv("LOCAL_CAMERA_TARGET_FPS", "18")
    runtime = runtime_factory()
    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        settings = client.get("/v1/ui/info").json()["local_camera_display"]
        assert settings["motion_width"] == 240
        assert settings["target_fps"] == 18
        assert settings["delay_seconds"] == 7
        assert settings["max_history_seconds"] >= 11
        assert settings["max_propagation_seconds"] == 1.5
        assert "max_image_side" not in settings


def _start(runtime):
    return runtime.start_camera(dry_run=True, features=["smoking"])


def _tracks(runtime, at=1.0, sequence=10):
    return RichLabelSnapshot(
        stream_session_id=runtime._session.stream_session_id,
        frame_sequence=sequence,
        timestamp_seconds=at,
        objects=(RichLabelObject("worker-person-1", "person", BOX, 0.99),),
    )


def _result(runtime, subject_id, *, at=1.0, sequence=10, detected=True):
    return AnalysisResult(
        message_id=f"scene-test-{sequence}",
        source_id=runtime.source_id,
        stream_session_id=runtime._session.stream_session_id,
        sequence=sequence,
        emitted_at="2026-09-12T00:00:00.000Z",
        frame=FrameReference(sequence, at, "2026-09-12T00:00:00.000Z", 240, 160),
        judgments=(
            EventJudgment(
                "smoking",
                detected,
                0.95,
                "Smoking observation.",
                ("cigarette",) if detected else (),
            ),
        ),
        objects=(
            GroundedObject(
                "cigarette",
                "cigarette",
                NormalizedBBox(0.2, 0.2, 0.23, 0.24),
                0.95,
                {"subject_ids": [subject_id]},
            ),
        )
        if detected
        else (),
        model_name="offline-test",
        model_latency_seconds=0.1,
    )


def test_scene_config_is_disabled_by_default_and_environment_can_enable(monkeypatch):
    assert load_config(ROOT / "config.yaml").runtime.scene_mode == "disabled"
    monkeypatch.setenv("SCENE_ENGINE_MODE", "active")
    assert load_config(ROOT / "config.yaml").runtime.scene_mode == "active"
    monkeypatch.setenv("SCENE_ENGINE_MODE", "disabled")
    monkeypatch.setenv("SCENE_ENGINE_ENABLED", "false")
    assert load_config(ROOT / "config.yaml").runtime.scene_mode == "disabled"


@pytest.mark.parametrize("mode", ["disabled", "active"])
def test_removed_legacy_enable_flag_never_enables_an_observer(monkeypatch, mode):
    monkeypatch.setenv("SCENE_ENGINE_MODE", mode)
    monkeypatch.setenv("SCENE_ENGINE_ENABLED", "true")
    with pytest.raises(ValueError, match=r"removed.*SCENE_ENGINE_MODE=active"):
        load_config(ROOT / "config.yaml")


@pytest.mark.parametrize("mode_source", ["environment", "yaml", "direct"])
def test_shadow_mode_is_rejected(tmp_path, monkeypatch, mode_source):
    config_path = ROOT / "config.yaml"
    if mode_source == "environment":
        monkeypatch.setenv("SCENE_ENGINE_MODE", "shadow")
    elif mode_source == "yaml":
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["runtime"]["scene_mode"] = "shadow"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="scene_mode must be disabled or active"):
        config = load_config(config_path)
        HarnessRuntime(
            replace(config, runtime=replace(config.runtime, scene_mode="shadow"))
        )


@pytest.mark.parametrize("legacy_enabled", [True, False])
def test_legacy_yaml_enable_flag_cannot_activate_scene(tmp_path, legacy_enabled):
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["runtime"]["scene_enabled"] = legacy_enabled
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    if legacy_enabled:
        with pytest.raises(ValueError, match="scene_enabled was removed"):
            load_config(path)
    else:
        assert load_config(path).runtime.scene_mode == "disabled"


@pytest.mark.parametrize("value", ["true", 1, None, [], {}])
def test_scene_config_requires_a_yaml_boolean(tmp_path, value):
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["runtime"]["scene_enabled"] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=r"scene_enabled.*true or false"):
        load_config(path)


def test_scene_environment_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("SCENE_ENGINE_ENABLED", "perhaps")
    with pytest.raises(ValueError, match=r"SCENE_ENGINE_ENABLED.*boolean"):
        load_config(ROOT / "config.yaml")


def test_disabled_scene_preserves_live_output_without_constructing_observer(
    runtime_factory,
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled scene must not allocate an observer")

    monkeypatch.setattr(service, "SceneRuntime", forbidden)
    runtime = runtime_factory(enabled=False)
    _start(runtime)
    runtime._on_display_frame(_jpeg(), 1, 10)
    assert runtime._scene is None
    assert runtime.preview_frames.latest() is not None
    assert runtime.scene_snapshot()["enabled"] is False
    assert runtime.scene_snapshot()["state"] == "disabled"


def wait_for_postprocessing(runtime):
    deadline = time.monotonic() + 5
    while runtime.status().get("scene_drain", {}).get("state") == "running":
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_real_dry_run_camera_feeds_scene_and_releases_evidence_on_stop(
    runtime_factory,
    monkeypatch,
):
    runtime = runtime_factory()
    assert runtime.scene_snapshot()["state"] == "idle"
    received = Event()
    original = runtime._on_display_frame

    def record(*args):
        original(*args)
        received.set()

    monkeypatch.setattr(runtime, "_on_display_frame", record)
    _start(runtime)
    scene = runtime._scene
    assert runtime.push_camera_frame(_jpeg(), width=240, height=160)
    assert received.wait(3), "dry-run camera source should reach the scene callback"
    assert scene.evidence.byte_count > 0
    snapshot = runtime.scene_snapshot()
    assert snapshot["state"] == "running"
    assert snapshot["mode"] == "active"
    assert snapshot["automatic_temporal_inference"] is False
    assert runtime.stop_session()
    wait_for_postprocessing(runtime)
    assert scene.evidence.byte_count == 0
    assert len(scene.evidence) == 0
    assert runtime.scene_snapshot()["state"] == "stopped"


def test_tracking_callback_exports_registry_subjects_and_ignores_old_sessions(
    runtime_factory,
):
    runtime = runtime_factory()
    _start(runtime)
    tracks = _tracks(runtime)
    runtime._on_tracking_snapshot(tracks)
    entities = runtime.scene_snapshot()["scene"]["entities"]
    assert len(entities) == 1
    canonical = runtime._session.person_registry.snapshot_at(1, 10)
    assert entities[0]["subject_id"] == canonical.objects[0].track_id
    assert entities[0]["subject_id"] != "worker-person-1"
    old = replace(tracks, stream_session_id="retired-session", timestamp_seconds=100)
    runtime._on_tracking_snapshot(old)
    assert runtime.scene_snapshot()["scene"]["entities"] == entities


def test_legacy_timeline_safety_results_do_not_create_scene_events(
    runtime_factory,
):
    runtime = runtime_factory()
    _start(runtime)
    registry = runtime._session.person_registry
    for sequence, at, detected in (
        (10, 1, True),
        (20, 2, True),
        (30, 3, False),
        (40, 4, False),
    ):
        # Simulate canonical tracking arriving through another branch without
        # first delivering a scene tracking callback.
        registry.observe_snapshot(_tracks(runtime, at, sequence))
        canonical = registry.snapshot_at(at, sequence)
        subject_id = canonical.objects[0].track_id
        runtime._on_timeline_result(
            _result(
                runtime,
                subject_id,
                at=at,
                sequence=sequence,
                detected=detected,
            )
        )
    events = runtime.scene_snapshot()["scene"]["events"]
    assert events == []
    assert runtime.scene_snapshot()["scene"]["entities"][0]["subject_id"] == subject_id


def test_legacy_timeline_owner_corrections_do_not_create_scene_events(runtime_factory):
    runtime = runtime_factory()
    _start(runtime)
    right_box = NormalizedBBox(0.6, 0.1, 0.9, 0.95)
    for sequence, at in ((10, 1), (20, 2)):
        runtime._on_tracking_snapshot(
            replace(
                _tracks(runtime, at, sequence),
                objects=(
                    RichLabelObject("worker-person-1", "person", BOX, 0.99),
                    RichLabelObject("worker-person-2", "person", right_box, 0.99),
                ),
            )
        )
        canonical = runtime._session.person_registry.snapshot_at(at, sequence)
        first_id, second_id = (item.track_id for item in canonical.objects)
        runtime._on_timeline_result(
            _result(runtime, first_id, at=at, sequence=sequence)
        )
    for sequence, at in ((10, 1), (20, 2)):
        runtime._on_timeline_result(
            _result(runtime, second_id, at=at, sequence=sequence)
        )
    events = {
        tuple(event["subject_ids"]): event
        for event in runtime.scene_snapshot()["scene"]["events"]
    }
    assert events == {}


def test_stale_generation_cannot_add_frames_or_people_to_new_scene(runtime_factory):
    runtime = runtime_factory()
    _start(runtime)
    old_generation = runtime._generation
    old_callback = runtime._session.callbacks.on_timeline_result
    old_result = _result(runtime, "old-subject")
    old_scene = runtime._scene
    session = runtime._session
    assert runtime.stop_session()
    session.join(3)
    runtime._end_drain_thread.join(3)
    _start(runtime)
    scene = runtime._scene
    before = runtime.scene_snapshot()
    old_callback(old_result)
    runtime._guarded_callback(
        old_generation, runtime._on_display_frame, _jpeg(), 99, 990
    )
    assert runtime.scene_snapshot() == before
    assert scene is not old_scene
    assert len(scene.evidence) == 0


def test_scene_endpoint_is_read_only_uncached_and_excludes_image_bytes(runtime_factory):
    runtime = runtime_factory()
    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        _start(runtime)
        runtime._on_display_frame(_jpeg(), 1, 10)
        runtime._on_tracking_snapshot(_tracks(runtime))
        assert runtime._identity_worker.wait_until_idle(2)
        before = runtime.scene_snapshot()
        response = client.get("/v1/experimental/scene")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == before
        assert runtime.scene_snapshot() == before
        assert "jpeg_bytes" not in response.text
        assert "base64" not in response.text
        assert client.post("/v1/experimental/scene", json={}).status_code == 405


def test_scene_frame_failure_does_not_drop_public_preview(runtime_factory, monkeypatch):
    runtime = runtime_factory()
    _start(runtime)

    def broken(*_args):
        raise RuntimeError("observer failed")

    monkeypatch.setattr(runtime._scene, "observe_frame", broken)
    runtime._on_display_frame(_jpeg(), 1, 10)
    assert runtime.preview_frames.latest()[1].frame_sequence == 10
    assert runtime.status()["source_frames_received"] == 1


def test_scene_result_failure_keeps_active_mode_out_of_legacy_timeline(
    runtime_factory, monkeypatch
):
    runtime = runtime_factory()
    _start(runtime)
    submitted = []
    monkeypatch.setattr(runtime._delayed_overlay, "submit_result", submitted.append)

    def broken(*_args):
        raise RuntimeError("observer failed")

    monkeypatch.setattr(runtime._scene, "observe_result", broken)
    result = _result(runtime, "subject-test")
    runtime._on_timeline_result(result)
    assert submitted == []


def test_scene_tracking_failure_keeps_active_mode_out_of_legacy_overlay(
    runtime_factory, monkeypatch
):
    runtime = runtime_factory()
    _start(runtime)
    submitted = []
    correction = _result(runtime, "subject-test")
    monkeypatch.setattr(runtime._delayed_overlay, "submit_result", submitted.append)
    monkeypatch.setattr(
        runtime._session.person_registry,
        "drain_reassociated_results",
        lambda: (correction,),
    )

    def broken(*_args):
        raise RuntimeError("observer failed")

    monkeypatch.setattr(runtime._scene, "observe_tracks", broken)
    runtime._on_tracking_snapshot(_tracks(runtime))
    assert submitted == []
    assert runtime._session.person_registry.history()


def test_failed_session_start_closes_and_clears_scene(runtime_factory, monkeypatch):
    runtime = runtime_factory()
    created = []
    original = service.SceneRuntime

    def make(*args, **kwargs):
        scene = original(*args, **kwargs)
        created.append(scene)
        return scene

    def fail(_session):
        raise RuntimeError("could not start test session")

    monkeypatch.setattr(service, "SceneRuntime", make)
    monkeypatch.setattr(service.StreamingSession, "start", fail)
    with pytest.raises(RuntimeError, match="could not start test session"):
        _start(runtime)
    assert len(created) == 1
    assert created[0].snapshot()["state"] == "stopped"
    assert runtime._scene is None
    assert runtime._session is None


def test_end_callback_releases_evidence_without_claiming_event_clear(runtime_factory):
    runtime = runtime_factory()
    _start(runtime)
    runtime._on_display_frame(_jpeg(), 1, 10)
    runtime._on_tracking_snapshot(_tracks(runtime))
    subject = runtime.scene_snapshot()["scene"]["entities"][0]["subject_id"]
    runtime._scene.scene.observe_event(
        "smoking", (subject,), timestamp_seconds=1, frame_sequence=10,
        detected=True, evidence_id="temporal-event",
    )
    old_scene = runtime._scene
    runtime._on_end("disconnected", None)
    runtime._end_drain_thread.join(3)
    assert not runtime._end_drain_thread.is_alive()
    wait_for_postprocessing(runtime)
    assert old_scene.evidence.byte_count == 0
    snapshot = runtime.scene_snapshot()
    assert snapshot["state"] == "stopped"
    assert snapshot["scene"]["events"][0]["phase"] == "suspected"


def test_gallery_delete_forgets_retained_scene_identity(runtime_factory, monkeypatch):
    runtime = runtime_factory()
    _start(runtime)
    runtime._scene.scene.observe_person(
        "subject-alice",
        timestamp_seconds=1,
        frame_sequence=10,
        bbox=BOX,
        identity={"status": "known", "person_id": "alice", "name": "Alice"},
    )
    monkeypatch.setattr(
        runtime,
        "_ensure_face_runtime",
        lambda: SimpleNamespace(delete_person=lambda _id: True),
    )
    forgotten = []
    original_forget = runtime._camera_display_metadata.forget_identity

    def forget(person_id):
        assert runtime._identity_control_lock._is_owned()
        assert runtime._scene.snapshot()["scene"]["entities"][0]["identity"] == {}
        forgotten.append(person_id)
        original_forget(person_id)

    monkeypatch.setattr(runtime._camera_display_metadata, "forget_identity", forget)
    assert runtime.delete_face("alice")
    entities = runtime.scene_snapshot()["scene"]["entities"]
    assert entities[0]["identity"] == {}
    assert forgotten == ["alice"]


def test_active_tracking_failure_is_visible_before_stop_without_echoing_details(
    runtime_factory,
):
    runtime = runtime_factory()
    _start(runtime)
    session = runtime._session
    previous, active = runtime._rich_labels, runtime._active_rich_session_id
    private = "https://private.example/?token=secret RemoteProtocolError raw text"
    stats = {"state": "failed", "last_error": private, "pending_frames": 3,
             "recovery_required": True}
    runtime._rich_labels = SimpleNamespace(stats=lambda: stats)
    runtime._active_rich_session_id = session.stream_session_id
    try:
        result = runtime.status()
        assert result["last_error"] == PUBLIC_TRACKING_ERROR
        assert result["rich_labeling"]["last_error"] == PUBLIC_TRACKING_ERROR
        assert result["readiness_reason"] == "person_tracking_failed"
        assert result["ready"] is False
        assert result["pipeline_state"] in {"starting", "running"}
        assert session.running
        assert result["rich_labeling"]["pending_frames"] == 3
        assert stats["last_error"] == private
        assert "secret" not in str(result)
        runtime._active_rich_session_id = "obsolete-run"
        assert runtime.status()["last_error"] is None
    finally:
        runtime._rich_labels, runtime._active_rich_session_id = previous, active


def test_only_terminal_inspection_worker_failure_sets_event_analysis_error(
    runtime_factory,
):
    runtime = runtime_factory()
    _start(runtime)
    previous = runtime._scene_worker
    stats = {"state": "running", "failed_requests": 5}
    runtime._scene_worker = SimpleNamespace(stats=lambda: stats)
    try:
        assert runtime.status()["last_error"] is None
        stats["state"] = "failed"
        result = runtime.status()
        assert result["last_error"] == PUBLIC_INSPECTION_ERROR
        assert result["readiness_reason"] == "event_analysis_failed"
        assert result["ready"] is False
        assert runtime._session.running
    finally:
        runtime._scene_worker = previous


@pytest.mark.parametrize("stage", ["tracking", "inspection"])
def test_online_drain_failure_uses_its_stage_not_exception_or_prior_generic_error(
    runtime_factory, monkeypatch, stage,
):
    runtime = runtime_factory()
    session = SimpleNamespace(stream_session_id="fault-run")
    control = SimpleNamespace(lock=RLock(), state="running", error=None)
    finished = []
    runtime._session = session
    monkeypatch.setattr(runtime, "_archive_run", lambda _session: control)
    monkeypatch.setattr(
        runtime, "_finish_scene_end", lambda reason, error: finished.append(error),
    )

    def fail():
        raise RuntimeError("https://private.example/?token=secret raw exception")

    coordinator = SimpleNamespace(
        stats=lambda: {"state": "failed" if stage == "tracking" else "running"},
        wait_until_idle=lambda **kwargs: True,
    )
    scene = SimpleNamespace(
        finish_tracking_sources=lambda: None,
        finish_inspection_sources=fail,
    )
    try:
        runtime._drain_native_end(
            runtime._generation, session, scene, coordinator, "stopped",
            "an unrelated generic request error",
        )
        assert finished == [
            PUBLIC_TRACKING_ERROR if stage == "tracking" else PUBLIC_INSPECTION_ERROR
        ]
        assert control.state == "failed"
        assert runtime._state["scene_drain"] == {"state": "failed"}
    finally:
        runtime._session = None


def test_successful_enrollment_allows_display_identity_only_after_scene_allow(
    runtime_factory, monkeypatch,
):
    runtime = runtime_factory()
    calls = []

    def enroll(person_id, name, image):
        calls.append("enroll")
        return SimpleNamespace(person_id=person_id, name=name, sample_count=1)

    def allow(person_id):
        assert runtime._identity_control_lock._is_owned()
        calls.append(("display", person_id))

    monkeypatch.setattr(runtime, "_ensure_face_runtime", lambda: SimpleNamespace(
        enroll=enroll, delete_person=lambda _id: False,
    ))
    monkeypatch.setattr(
        runtime, "_observe_scene",
        lambda action, person_id: calls.append((action, person_id)),
    )
    monkeypatch.setattr(runtime._camera_display_metadata, "allow_identity", allow)
    monkeypatch.setattr(
        runtime._camera_display_metadata, "forget_identity",
        lambda _id: pytest.fail("failed deletion must preserve display identity"),
    )
    enrolled = runtime.enroll_face("alice", "Alice", b"unused")
    assert enrolled["person"]["person_id"] == "alice"
    assert calls == ["enroll", ("allow_identity", "alice"), ("display", "alice")]
    assert runtime.delete_face("alice") is False
