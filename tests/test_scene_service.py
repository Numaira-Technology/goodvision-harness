"""Exercise the experimental scene observer through actual service lifecycle hooks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event
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
        "RESULT_WEBHOOK_URL",
        "RICH_LABEL_ENABLED",
        "RICH_LABEL_TRACKING_ENABLED",
        "INFERENCE_PROVIDER",
        "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def runtime_factory(tmp_path, monkeypatch):
    instances = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("scene tests must not initialize a remote or face model")

    monkeypatch.setattr(service, "RichLabelHTTPClient", forbidden)
    monkeypatch.setattr(service, "FaceRecognitionRuntime", forbidden)

    def make(*, enabled=True):
        config = load_config(ROOT / "config.yaml")
        config = replace(
            config,
            runtime=replace(config.runtime, scene_enabled=enabled),
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
    assert load_config(ROOT / "config.yaml").runtime.scene_enabled is False
    monkeypatch.setenv("SCENE_ENGINE_ENABLED", "true")
    assert load_config(ROOT / "config.yaml").runtime.scene_enabled is True
    monkeypatch.setenv("SCENE_ENGINE_ENABLED", "false")
    assert load_config(ROOT / "config.yaml").runtime.scene_enabled is False


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


def test_real_dry_run_camera_feeds_scene_and_releases_evidence_on_stop(
    runtime_factory,
    monkeypatch,
):
    runtime = runtime_factory()
    assert runtime.scene_snapshot()["state"] == "idle"
    received = Event()
    original = runtime._on_timeline_result

    def record(result):
        original(result)
        received.set()

    monkeypatch.setattr(runtime, "_on_timeline_result", record)
    _start(runtime)
    scene = runtime._scene
    assert runtime.push_camera_frame(_jpeg(), width=240, height=160)
    assert received.wait(3), "dry-run camera inference should reach the scene callback"
    assert scene.evidence.byte_count > 0
    snapshot = runtime.scene_snapshot()
    assert snapshot["state"] == "running"
    assert snapshot["mode"] == "shadow"
    assert snapshot["automatic_temporal_inference"] is False
    assert runtime.stop_session()
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


def test_timeline_receives_exact_registry_visibility_before_positive_and_negative(
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
    assert len(events) == 1
    assert events[0]["subject_ids"] == [subject_id]
    assert events[0]["phase"] == "ended"


def test_timeline_owner_correction_withdraws_old_person_event(runtime_factory):
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
    assert events[(first_id,)]["phase"] == "retracted"
    assert events[(second_id,)]["phase"] == "confirmed"
    assert runtime.scene_snapshot()["stats"]["association_revisions_applied"] == 2


def test_stale_generation_cannot_add_frames_or_people_to_new_scene(runtime_factory):
    runtime = runtime_factory()
    _start(runtime)
    old_generation = runtime._generation
    old_callback = runtime._session.callbacks.on_timeline_result
    old_result = _result(runtime, "old-subject")
    old_scene = runtime._scene
    assert runtime.stop_session()
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


def test_scene_result_failure_does_not_drop_existing_timeline(
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
    assert submitted == [result]


def test_scene_tracking_failure_does_not_drop_registry_corrections(
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
    assert submitted == [correction]
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
    runtime._on_timeline_result(_result(runtime, subject))
    runtime._on_end("disconnected", None)
    assert runtime._scene.evidence.byte_count == 0
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
    assert runtime.delete_face("alice")
    entities = runtime.scene_snapshot()["scene"]["entities"]
    assert entities[0]["identity"] == {}
