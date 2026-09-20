"""Continuous scene operation through real service/session/worker boundaries."""

from __future__ import annotations

import io
import time
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread, get_ident

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from goodvision_harness import pipeline, service
from goodvision_harness.config import load_config
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    NormalizedBBox,
)
from goodvision_harness.rich_labeling import (
    RichLabelCoordinator,
    RichLabelObject,
    RichLabelSnapshot,
)
from goodvision_harness.scene_capture import SceneCaptureClient
from goodvision_harness.scene_inspection import build_temporal_messages
from goodvision_harness.scene_runtime import TemporalVerdict
from goodvision_harness.service import HarnessRuntime

ROOT = Path(__file__).resolve().parents[1]
BOX = NormalizedBBox(0.1, 0.1, 0.6, 0.95)


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **kwargs: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    for name in (
        "SCENE_ENGINE_ENABLED",
        "SCENE_ENGINE_MODE",
        "SCENE_TIMEOUT_SECONDS",
        "SCENE_MAX_IMAGE_SIDE",
        "SCENE_MAX_TOKENS",
        "RESULT_WEBHOOK_URL",
        "RICH_LABEL_ENABLED",
        "RICH_LABEL_TRACKING_ENABLED",
        "INFERENCE_PROVIDER",
        "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)


def _jpeg():
    with Image.new("RGB", (240, 160), "gray") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG")
        return buffer.getvalue()


def _await(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("continuous service did not reach expected state")
        time.sleep(0.005)


def _await_scene_drain(runtime):
    # Finalization includes durable writes and an encoder with a 10-second
    # budget. These tests assert completion/order, not a three-second SLA.
    deadline = time.monotonic() + 15
    while True:
        status = runtime.status()
        drain = status["scene_drain"]
        if drain == {"state": "completed"}:
            return
        if drain == {"state": "failed"} or time.monotonic() >= deadline:
            pytest.fail(
                f"scene drain did not complete: {drain}; "
                f"postprocessing={status['postprocessing_runs']}"
            )
        time.sleep(0.01)


class OfflineTracker:
    def infer(self, _jpeg, *, stream_session_id, frame_sequence, timestamp_seconds):
        return RichLabelSnapshot(
            stream_session_id, frame_sequence, timestamp_seconds, ()
        )

    def close(self):
        pass


class Inspector:
    def __init__(self, *, blocked=False):
        self.jobs = []
        self.detected = True
        self.started = Event()
        self.release = Event()
        self.closed = Event()
        if not blocked:
            self.release.set()

    def inspect(self, job):
        self.jobs.append(job)
        self.started.set()
        assert self.release.wait(5)
        ids = job.request.subject_ids or tuple(
            dict.fromkeys(subject for _, subject, _ in job.subject_boxes)
        )
        return (
            tuple(
                TemporalVerdict(label, (ids[0],), self.detected, 0.9)
                for label in job.request.event_labels
            )
            if ids
            else ()
        )

    def close(self):
        self.closed.set()

    def stats(self):
        return {"requests": len(self.jobs)}


@pytest.fixture
def factory(tmp_path, monkeypatch):
    runtimes, inspectors = [], []

    def no_remote(*_args, **_kwargs):
        raise AssertionError("unit tests must not construct remote inference")

    monkeypatch.setattr(service, "TemporalVLMInspector", no_remote)
    monkeypatch.setattr(service, "RichLabelHTTPClient", no_remote)

    def make(*, tracker=False, inspector=None, scene_side=720):
        config = load_config(ROOT / "config.yaml")
        config = replace(
            config,
            runtime=replace(
                config.runtime,
                scene_mode="active",
                scene_max_image_side=scene_side,
                # This suite exercises tracker-gated temporal scene inspection.
                # Independent VLM admission has its own service regression suite.
                scene_independent_vlm=False,
            ),
            rich_labeling=replace(
                config.rich_labeling,
                enabled=False,
                tracking_enabled=False,
                backend="paddle",
            ),
            events=replace(
                config.events, directory=tmp_path / f"events-{len(runtimes)}"
            ),
            service=replace(
                config.service, webhook=replace(config.service.webhook, url=None)
            ),
        )
        coordinator = RichLabelCoordinator(OfflineTracker()) if tracker else None
        runtime = HarnessRuntime(config, rich_label_coordinator=coordinator)
        runtime._aiclub_api_key = "TEST_KEY"
        runtimes.append(runtime)
        if inspector is not None:
            inspectors.append(inspector)
            monkeypatch.setattr(
                service, "TemporalVLMInspector", lambda *_a, **_k: inspector
            )
        return runtime

    yield make
    for inspector in inspectors:
        inspector.release.set()
    for runtime in runtimes:
        runtime.stop()


def _observe(runtime, timestamp, sequence):
    runtime._on_tracking_snapshot(
        RichLabelSnapshot(
            runtime._session.stream_session_id,
            sequence,
            timestamp,
            (RichLabelObject("tracker-id", "person", BOX, 0.99),),
        )
    )
    runtime._on_display_frame(_jpeg(), timestamp, sequence)


def test_temporal_evidence_api_returns_exact_source_and_expires_on_reset(factory):
    runtime = factory(tracker=True, inspector=Inspector())
    with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
        assert client.get("/v1/scene/evidence").status_code == 404
        runtime.start_camera(dry_run=False, features=["smoking"])
        _observe(runtime, 0, 0)
        _observe(runtime, 1, 10)
        _await(lambda: runtime._scene.latest_inspection() is not None)
        _await(lambda: not runtime._scene.snapshot()["inspections"])
        response = client.get("/v1/scene/evidence")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        evidence = response.json()
        assert evidence["event_labels"] == ["smoking"]
        assert 2 <= len(evidence["frames"]) <= 4
        assert "base64" not in response.text and "jpeg_bytes" not in response.text
        for frame in evidence["frames"]:
            image = client.get(frame["url"])
            assert image.status_code == 200
            assert image.content == _jpeg()
            assert (
                float(image.headers["x-media-timestamp"]) == frame["timestamp_seconds"]
            )
        _await(lambda: runtime._scene_archive.stats()["stored_events"] > 0)
        history = client.get("/v1/scene/history").json()["events"]
        assert history and history[0]["event_label"] == "smoking"
        assert "identity" not in history[0]
        old_url = evidence["frames"][0]["url"]
        session = runtime._session
        assert runtime.stop_session()
        session.join(timeout=5)
        assert not session.running
        runtime._end_drain_thread.join(timeout=5)
        assert not runtime._end_drain_thread.is_alive()
        assert runtime.status()["pipeline_state"] == "idle"
        assert client.get("/v1/scene/evidence").status_code == 404
        assert client.get(old_url).status_code == 404
        runtime.start_camera(dry_run=True, features=["smoking"])
        assert client.get(old_url).status_code == 404


def test_safety_only_capture_avoids_legacy_build_and_classification_with_video(
    factory, monkeypatch
):
    consumed = Event()
    original_windows = pipeline.iter_sliding_windows

    def windows(*args, **kwargs):
        for window in original_windows(*args, **kwargs):
            yield window
            consumed.set()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("active capture must not run the per-frame safety model")

    monkeypatch.setattr(service, "build_selected_features_grounding_client", forbidden)
    monkeypatch.setattr(service, "build_grounding_client", forbidden)
    monkeypatch.setattr(SceneCaptureClient, "classify_window", forbidden)
    monkeypatch.setattr(pipeline, "iter_sliding_windows", windows)
    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking", "fall"])
    assert runtime.push_camera_frame(_jpeg(), width=240, height=160)
    assert consumed.wait(2), "the source window must actually traverse the session"
    _await(lambda: runtime.annotated_frames.latest() is not None)
    assert runtime.preview_frames.latest() is not None
    assert runtime._session.running
    assert runtime.status()["last_error"] is None
    assert runtime.results.latest() is None
    assert runtime.scene_snapshot()["mode"] == "active"
    assert runtime.scene_snapshot()["automatic_temporal_inference"] is False
    assert runtime.scene_results.latest() is not None


def test_mixed_active_session_keeps_raw_face_frames_without_ungated_results(
    factory, monkeypatch
):
    original = service.build_selected_features_grounding_client
    selected = []

    def build(*args, **kwargs):
        selected.append(kwargs["features"])
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "build_selected_features_grounding_client", build)
    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking", "stranger"])
    assert selected == [("stranger",)]
    assert runtime._session.client._safety_client is None
    assert runtime.push_camera_frame(_jpeg(), width=240, height=160)
    assert runtime._scene.event_labels == ("smoking",)
    _await(lambda: runtime.frames.latest() is not None)
    exact = runtime.frames.latest()[1]
    assert exact.analysis_message_id
    assert exact.annotation_mode == "none"
    assert runtime.results.latest() is None


def test_scene_confirmation_reaches_legacy_json_sse_and_webhook(factory, monkeypatch):
    from types import SimpleNamespace

    from test_branch_result_delivery import sse_records

    runtime = factory()
    delivered = []
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "2")
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "1")
    with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
        runtime.start_camera(dry_run=True, features=["smoking", "fight"])
        runtime._delivery = SimpleNamespace(
            submit=delivered.append, close=lambda **_kwargs: None,
        )
        for seq, detected in ((1, True), (2, True), (3, False), (4, False)):
            runtime._on_display_frame(_jpeg(), float(seq), seq)
            state = runtime._scene.scene
            state.observe_person(
                "person-a", timestamp_seconds=float(seq), frame_sequence=seq, bbox=BOX,
            )
            state.observe_event(
                "smoking", ("person-a",), timestamp_seconds=float(seq),
                frame_sequence=seq, detected=detected, evidence_id=f"e-{seq}",
                confidence=0.9,
            )
            runtime._on_scene_update()
            payload = client.get("/v1/results/smoking/latest").json()
            assert payload["judgments"][0]["detected"] is (seq in (2, 3))
            assert payload["analysis_kind"] == "scene_state"
            assert payload["missing_event_labels"] == ["fight"]
            assert payload["frame"]["source_frame_index"] == seq
            assert payload["frame"]["width"] == 240
            scoped = client.get(
                "/v1/fleet/cameras/camera01/results/smoking/latest"
            ).json()
            assert scoped == dict(payload, camera_id="camera01")
        streamed = sse_records(runtime, feature="smoking")
        assert [row[1]["judgments"][0]["detected"] for row in streamed] == [
            False, True, True, False,
        ]
        assert [service._feature_result_payload(row, "smoking") for row in delivered] == [
            row[1] for row in streamed
        ]
        assert client.get("/v1/results/fight/latest").status_code == 404
        before = runtime.results.latest().cursor
        runtime._on_scene_update()
        assert runtime.results.latest().cursor == before
        # A raw face/classifier callback cannot replace confirmed-state results.
        raw = AnalysisResult(
            "raw", runtime.source_id, runtime._session.stream_session_id, 99,
            "2026-09-19T01:00:00Z", FrameReference(
                4, 4, "2026-09-19T01:00:00Z", 240, 160,
            ), (EventJudgment("stranger", False, 0.9, "Raw observation."),),
            (), "standard", 0.01,
        )
        runtime._on_analysis_result(raw)
        assert runtime.results.latest().cursor == before
        assert client.get("/v1/results/stranger/latest").status_code == 404
        runtime.stop_session()
        runtime.start_camera(dry_run=True, features=["smoking"])
        assert client.get("/v1/results/smoking/latest").status_code == 404
        runtime._delivery = None


@pytest.mark.parametrize(
    "features,scene_side",
    [
        (["smoking"], 720),
        (["smoking", "stranger"], 720),
        (["smoking"], 1600),
        (["smoking", "stranger"], 1600),
    ],
)
def test_active_camera_admits_the_source_resolution_advertised_to_browser(
    factory, features, scene_side
):
    runtime = factory(scene_side=scene_side)
    info = runtime.ui_info()
    settings = info["multi"] if "stranger" in features else info
    side = settings["max_image_side"]
    assert side >= scene_side
    assert info["annotated_output"]["mode"] == "scene"
    assert info["annotated_output"]["buffer_delay_seconds"] == 0
    assert info["annotated_output"]["bbox_max_gap_seconds"] is None
    runtime.start_camera(dry_run=True, features=features)
    assert runtime._session.config.video.max_image_side == side
    height = side // 2
    with Image.new("RGB", (side, height), "gray") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG")
        assert runtime.push_camera_frame(buffer.getvalue(), width=side, height=height)
    _await(lambda: runtime.preview_frames.latest() is not None)
    with Image.open(io.BytesIO(runtime.preview_frames.latest()[1].jpeg)) as image:
        assert image.size == (side, height)
    # Session-specific source resolution must not change legacy model defaults.
    assert runtime.config.video.max_image_side == 320


def test_active_exact_face_frame_keeps_source_bytes_and_rejects_stale_or_foreign(
    factory, monkeypatch
):
    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking", "stranger"])

    def forbidden(*_args, **_kwargs):
        raise AssertionError(
            "current scene overlays must not alter historical evidence"
        )

    monkeypatch.setattr(runtime._delayed_overlay, "submit_frame", forbidden)
    monkeypatch.setattr(service, "_render_analysis_jpeg", forbidden)
    jpeg = _jpeg()
    result = AnalysisResult(
        message_id="exact-face-20",
        source_id=runtime.source_id,
        stream_session_id=runtime._session.stream_session_id,
        sequence=20,
        emitted_at="2026-09-12T00:00:00Z",
        frame=FrameReference(20, 2, "2026-09-12T00:00:00Z", 240, 160),
        judgments=(EventJudgment("stranger", False, 0.9, "Known person.", ()),),
        objects=(),
        model_name="offline-face",
        model_latency_seconds=0.1,
    )
    runtime._on_analysis_frame(result, jpeg)
    exact = runtime.frames.latest()
    assert exact[1].jpeg == jpeg
    assert exact[1].analysis_message_id == result.message_id
    assert exact[1].frame_sequence == 20 and exact[1].media_timestamp == 2
    assert exact[1].annotation_mode == "none"
    runtime._on_analysis_frame(
        replace(result, stream_session_id="retired-session"), jpeg
    )
    runtime._on_analysis_frame(
        replace(
            result,
            message_id="old-face",
            frame=replace(result.frame, timestamp_seconds=1, source_frame_index=10),
        ),
        jpeg,
    )
    assert runtime.frames.latest() == exact
    assert runtime.status()["analysis_frames_published"] == 1
    assert runtime.scene_snapshot()["scene"]["events"] == []
    assert runtime.results.latest() is None


def test_live_active_session_requires_a_tracker_before_start(factory):
    runtime = factory()
    with pytest.raises(RuntimeError, match="person tracking module"):
        runtime.start_camera(dry_run=False, features=["smoking"])
    assert runtime._session is None
    assert runtime._scene_worker is None


def test_actual_worker_confirms_canonical_person_event_and_unknown_does_not_clear(
    factory,
):
    inspector = Inspector()
    runtime = factory(tracker=True, inspector=inspector)
    runtime.start_camera(
        dry_run=False,
        features=["smoking"],
        event_sensitivities={"smoking": "standard", "fight": "high"},
    )
    _observe(runtime, 0, 0)
    _observe(runtime, 1, 10)
    _await(
        lambda: (
            runtime._scene_worker.stats()["succeeded_requests"] >= 1
            and not runtime._scene.snapshot()["inspections"]
        )
    )
    _observe(runtime, 4, 40)
    _await(
        lambda: any(
            event["phase"] == "confirmed"
            for event in runtime.scene_snapshot()["scene"]["events"]
        )
    )
    state = runtime.scene_snapshot()
    assert state["automatic_temporal_inference"] is True
    event = state["scene"]["events"][0]
    canonical = state["scene"]["entities"][0]["subject_id"]
    assert event["subject_ids"] == [canonical]
    assert canonical != "tracker-id"
    assert all(job.request.event_labels == ("smoking",) for job in inspector.jobs)
    assert all(
        dict(job.event_sensitivities) == {"smoking": "standard"}
        for job in inspector.jobs
    )
    assert "STANDARD" in str(build_temporal_messages(inspector.jobs[0]))
    inspector.detected = None
    before = runtime._scene_worker.stats()["succeeded_requests"]
    _observe(runtime, 7, 70)
    _await(lambda: runtime._scene_worker.stats()["succeeded_requests"] > before)
    event = runtime.scene_snapshot()["scene"]["events"][0]
    assert event["phase"] == "confirmed"
    assert event["negative_observations"] == 0
    _await(lambda: runtime.scene_results.latest() is not None)
    assert (
        runtime.scene_results.latest().payload["scene"]["events"][0]["phase"]
        == "confirmed"
    )
    _await(lambda: runtime.results.latest(feature="smoking") is not None)
    assert runtime.results.latest(feature="smoking").payload["judgments"][0][
        "detected"
    ] is True


def _fake_encoder(_manifest, destination, **_kwargs):
    destination.write_bytes(b"scene-evidence")
    return {"duration_seconds": 3.0, "fps": 5.0, "partial": False}


def test_temporal_events_keep_separate_videos_for_the_same_person(factory):
    runtime = factory(tracker=True, inspector=Inspector())
    archive = runtime.event_archive()
    archive._encoder = _fake_encoder
    with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
        runtime.start_camera(dry_run=False, features=["smoking", "fall"])
        _observe(runtime, 0, 0)
        _observe(runtime, 1, 10)
        _await(lambda: runtime._scene_worker.stats()["succeeded_requests"] >= 1)
        _observe(runtime, 4, 40)
        _await(lambda: runtime._scene_worker.stats()["succeeded_requests"] >= 2)
        _observe(runtime, 5, 50)
        _observe(runtime, 7, 70)
        _await(lambda: len([
            event for event in runtime.scene_snapshot()["scene"]["events"]
            if event["phase"] == "confirmed"
        ]) == 2)
        canonical = runtime.scene_snapshot()["scene"]["entities"][0]["subject_id"]
        assert runtime.stop_session()
        _await(lambda: runtime.status()["pipeline_state"] == "idle")
        _await_scene_drain(runtime)
        events = client.get("/v1/events").json()["events"]
        assert {event["event_label"] for event in events} == {"smoking", "fall"}
        assert len({event["event_id"] for event in events}) == 2
        assert all(event["subject_ids"] == [canonical] for event in events)
        assert all(event["finalized"] for event in events)
        for event in events:
            assert event["evidence"]["status"] == "ready"
            video = client.get(event["evidence"]["available"]["url"])
            assert video.status_code == 200
            assert video.content == b"scene-evidence"


def test_full_archive_queue_retains_distinct_episodes_until_stop_drain(
    factory, monkeypatch,
):
    runtime = factory()
    archive = runtime.event_archive()
    archive._encoder = _fake_encoder
    runtime.start_camera(dry_run=True, features=["smoking"])
    session = runtime._session
    allow, final_attempted = Event(), Event()
    submit = archive.submit_scene

    def temporarily_full(snapshot, **kwargs):
        if snapshot["state"] == "stopped":
            final_attempted.set()
        return allow.is_set() and submit(snapshot, **kwargs)

    monkeypatch.setattr(archive, "submit_scene", temporarily_full)
    try:
        for timestamp in range(1, 15):
            runtime._last_scene_publish_monotonic = time.monotonic()
            _observe(runtime, timestamp, timestamp * 10)
            subject = runtime.scene_snapshot()["scene"]["entities"][0]["subject_id"]
            assert runtime._scene.scene.observe_event(
                "smoking", (subject,), timestamp_seconds=timestamp,
                frame_sequence=timestamp * 10, detected=timestamp in {1, 2, 13, 14},
                confidence=0.9, evidence_id=f"temporal-{timestamp}",
            )
            # Several worker updates can be observed in the next snapshot.
            # Its terminal transition must finish the older retained episode.
            if timestamp in {2, 14}:
                runtime._on_scene_update()
        assert runtime.stop_session()
        assert final_attempted.wait(2)
        assert runtime.status()["pipeline_state"] == "idle"
        assert runtime.status()["postprocessing_runs"][-1]["state"] == "running"
        allow.set()
        _await(lambda: runtime.status()["pipeline_state"] == "idle")
        _await_scene_drain(runtime)
        events = archive.list_events(stream_session_id=session.stream_session_id)[
            "events"
        ]
        assert len(events) == 2
        assert {event["start_seconds"] for event in events} == {1, 13}
        assert len({event["event_id"] for event in events}) == 2
        first = next(event for event in events if event["start_seconds"] == 1)
        assert first["end_reason"] == "negative"
        assert first["end_seconds"] == 12
        assert all(event["subject_ids"] == [subject] for event in events)
        assert all(event["finalized"] for event in events)
        assert all(event["evidence"]["status"] == "ready" for event in events)
    finally:
        allow.set()


def test_stop_finishes_accepted_model_response_before_permitting_restart(factory):
    inspector = Inspector(blocked=True)
    runtime = factory(tracker=True, inspector=inspector)
    runtime.start_camera(dry_run=False, features=["smoking"])
    _observe(runtime, 0, 0)
    _observe(runtime, 1, 10)
    assert inspector.started.wait(2)
    old_scene = runtime._scene
    old_generation = runtime._generation
    assert runtime.stop_session()
    assert runtime.status()["pipeline_state"] == "draining"
    assert runtime._generation == old_generation
    assert runtime._scene is old_scene
    assert not runtime.push_camera_frame(_jpeg(), width=240, height=160)
    with pytest.raises(RuntimeError, match="already running"):
        runtime.start_camera(dry_run=True, features=["smoking"])
    inspector.release.set()
    _await(lambda: runtime.status()["pipeline_state"] == "idle")
    _await_scene_drain(runtime)
    assert not inspector.closed.is_set()  # Keep transport for the next run.
    assert old_scene.snapshot()["stats"]["completed_inspections"] >= 1
    assert old_scene.snapshot()["scene"]["events"]
    runtime.start_camera(dry_run=True, features=["smoking"])
    current_scene = runtime._scene
    assert current_scene.stream_session_id != old_scene.stream_session_id
    assert current_scene.snapshot()["scene"]["events"] == []
    assert current_scene.snapshot()["stats"]["completed_inspections"] == 0
    assert runtime.scene_results.latest() is None
    assert runtime.annotated_frames.latest() is None
    runtime.stop()
    assert inspector.closed.wait(2)


def test_temporal_transport_survives_runs_and_retires_on_credential_change(
    factory, monkeypatch
):
    runtime = factory(tracker=True)
    clients, arguments, calls = [], [], []

    class Transport(Inspector):
        def warmup(self):
            calls.append((self, "warmup", get_ident()))
            return True

        def inspect(self, job):
            calls.append((self, "inspect", get_ident()))
            return super().inspect(job)

        def close(self):
            calls.append((self, "close", get_ident()))
            super().close()

    def create(*args, **kwargs):
        client = Transport()
        clients.append(client)
        arguments.append(kwargs)
        calls.append((client, "create", get_ident()))
        return client

    monkeypatch.setattr(service, "TemporalVLMInspector", create)

    def run_once():
        runtime.start_camera(dry_run=False, features=["smoking"])
        _observe(runtime, 0, 0)
        _observe(runtime, 1, 10)
        _await(lambda: runtime._scene.snapshot()["stats"]["completed_inspections"] >= 1)
        assert runtime.stop_session()
        _await(lambda: runtime.status()["pipeline_state"] == "idle")
        _await_scene_drain(runtime)

    run_once()
    run_once()
    assert len(clients) == 1
    assert len(clients[0].jobs) >= 2
    assert not clients[0].closed.is_set()
    assert sum(kind == "warmup" for _, kind, _ in calls) == 1
    assert len({thread for _, _, thread in calls}) == 1

    runtime._aiclub_api_key = "REPLACEMENT_TEST_KEY"
    run_once()
    assert len(clients) == 2
    assert clients[0].closed.is_set()
    assert arguments[0]["api_key"] == "TEST_KEY"
    assert arguments[1]["api_key"] == "REPLACEMENT_TEST_KEY"
    assert not clients[1].closed.is_set()
    runtime.stop()
    assert clients[1].closed.wait(2)
    assert len({thread for _, _, thread in calls}) == 1


def test_stopped_run_evidence_does_not_block_or_mutate_the_next_run(
    factory, monkeypatch,
):
    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking"])
    session = runtime._session
    archive = session.archive
    entered, release = Event(), Event()
    original_wait = archive.wait_until_idle

    def gated_wait(*args, **kwargs):
        if kwargs.get("stream_session_id") == session.stream_session_id:
            entered.set()
            if not release.wait(kwargs.get("timeout_seconds", 0.1)):
                return False
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(archive, "wait_until_idle", gated_wait)
    try:
        assert runtime.stop_session()
        assert entered.wait(2)
        assert not session.running
        assert runtime.status()["pipeline_state"] == "idle"
        assert runtime._session is None
        assert runtime.status()["postprocessing_runs"][-1]["stage"] == "evidence"
        runtime.start_camera(dry_run=True, features=["smoking"])
        new_session, new_scene, generation = (
            runtime._session, runtime._scene, runtime._generation,
        )
        assert new_session.stream_session_id != session.stream_session_id
        _observe(runtime, 1, 10)
        before = runtime.scene_snapshot()["scene"]
        release.set()
        _await(lambda: runtime._run_postprocess[
            session.stream_session_id
        ].state == "finalized")
        assert runtime._session is new_session
        assert runtime._scene is new_scene
        assert runtime._generation == generation
        assert runtime.status()["pipeline_state"] == "running"
        assert runtime.status()["scene_drain"] is None
        assert runtime.scene_snapshot()["scene"] == before
    finally:
        release.set()


def test_active_configuration_environment_is_effective(monkeypatch):
    monkeypatch.setenv("SCENE_ENGINE_MODE", "active")
    monkeypatch.setenv("SCENE_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("SCENE_MAX_IMAGE_SIDE", "1024")
    monkeypatch.setenv("SCENE_MAX_TOKENS", "2048")
    config = load_config(ROOT / "config.yaml")
    assert config.runtime.scene_mode == "active"
    assert config.runtime.scene_timeout_seconds == 12
    assert config.runtime.scene_max_image_side == 1024
    assert config.runtime.scene_max_tokens == 2048


@pytest.mark.parametrize(
    "name,value",
    [
        ("SCENE_ENGINE_MODE", "automatic"),
        ("SCENE_TIMEOUT_SECONDS", "nan"),
        ("SCENE_TIMEOUT_SECONDS", "inf"),
        ("SCENE_TIMEOUT_SECONDS", "21"),
        ("SCENE_MAX_IMAGE_SIDE", "2049"),
        ("SCENE_MAX_TOKENS", "16385"),
    ],
)
def test_invalid_active_configuration_is_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises((ValueError, TypeError)):
        load_config(ROOT / "config.yaml")


def test_camera_source_ids_and_metrics_are_isolated_across_runs(factory):
    runtime = factory()
    with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
        runtime.start_camera(dry_run=True, features=["smoking"])
        sid = runtime._session.stream_session_id
        source = {
            "stream_session_id": sid, "generation": 8, "crop_version": 0,
            "source_sequence": 42, "media_timestamp": 1.25,
        }
        assert runtime.push_camera_frame(_jpeg(), width=240, height=160, **source)
        frame = runtime.preview_frames.latest()[1]
        assert (frame.frame_sequence, frame.media_timestamp) == (42, 1.25)
        # A retransmission after a lost ACK is idempotently accepted.
        before = runtime.status()["source_frames_received"]
        assert runtime.push_camera_frame(_jpeg(), width=240, height=160, **source)
        assert runtime.status()["source_frames_received"] == before
        metadata = client.get("/v1/video/display/metadata").json()
        assert metadata["generation"] == 8
        metrics = {
            "stream_session_id": sid, "generation": 8, "crop_version": 0,
            "report_sequence": 1, "captured_frames": 20,
            "distinct_presented_frames": 15, "window_presented_fps": 15.0,
            "contour_frame_ratio": 0.5,
            "complete_contour_frames": 3, "partial_contour_frames": 4,
            "complete_contour_frame_ratio": 0.2,
        }
        assert client.post("/v1/video/display/metrics", json=metrics).json()["accepted"]
        duplicate = client.post("/v1/video/display/metrics", json=metrics).json()
        assert not duplicate["accepted"]
        assert client.post("/v1/video/display/metrics", json={
            **metrics, "jpeg": "not-allowed",
        }).status_code == 400
        assert client.post("/v1/video/display/metrics", json={
            **metrics, "complete_contour_frame_ratio": 1.1,
        }).status_code == 400
        assert runtime.stop_session()
        _await(lambda: runtime.status()["pipeline_state"] == "idle")
        runtime.start_camera(dry_run=True, features=["smoking"])
        assert not runtime.push_camera_frame(
            _jpeg(), width=240, height=160,
            **{**source, "source_sequence": 43, "media_timestamp": 1.5},
        )
        assert not client.post("/v1/video/display/metrics", json={
            **metrics, "report_sequence": 2,
        }).json()["accepted"]
        assert runtime.status()["local_camera_display"] is None
        assert client.get("/assets/camera_display.js").status_code == 200


def test_camera_websocket_sends_geometry_without_waiting_for_another_upload(
    factory, monkeypatch, caplog,
):
    import asyncio
    import json
    import struct
    from types import SimpleNamespace

    runtime = factory()
    with TestClient(service.create_app(runtime_factory=lambda: runtime)) as client:
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["type"] == "hello"
            socket.send_json({"type": "start", "mode": "camera", "dry_run": True,
                              "features": ["smoking"]})
            started = socket.receive_json()
            sid = started["stream_session_id"]
            source = {"stream_session_id": sid, "generation": 8, "crop_version": 0,
                      "source_sequence": 42, "media_timestamp": 1.25,
                      "width": 240, "height": 160}
            encoded = json.dumps(source).encode()
            # A successful upload ACK must not call the full metadata snapshot.
            original = runtime._camera_display_metadata.snapshot
            def snapshot_off_loop():
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return original()
                raise AssertionError("metadata snapshot blocked the ASGI event loop")
            monkeypatch.setattr(runtime._camera_display_metadata, "snapshot", snapshot_off_loop)
            socket.send_bytes(b"GVC2" + struct.pack(">I", len(encoded)) + encoded + _jpeg())
            while True:
                message = socket.receive_json()
                if message["type"] == "camera_frame_ack":
                    assert message["accepted"] and message["source_sequence"] == 42
                    break
            runtime._camera_display_metadata.observe_batch(SimpleNamespace(
                stream_session_id=sid, predictor_epoch="test-epoch", frames=(
                    SimpleNamespace(frame_sequence=42, timestamp_seconds=1.25, objects=()),
                ),
            ))
            while True:
                message = socket.receive_json()
                if message["type"] == "camera_display_metadata" and message["anchors"]:
                    break
            assert message["anchors"][0]["source_sequence"] == 42
            assert message["anchors"][0]["subjects"] == []
            assert message["generation"] == 8
            with caplog.at_level("INFO", logger="uvicorn.error"):
                socket.send_json({"type": "camera_display_metrics",
                                  "stream_session_id": sid, "generation": 8,
                                  "crop_version": 0, "report_sequence": 1,
                                  "captured_frames": 1})
                _await(lambda: runtime._state["local_camera_display"] is not None)
            assert "local_camera_display_timing" in caplog.text
            socket.send_json({"type": "stop"})
            while socket.receive_json()["type"] != "ended":
                pass


def test_stop_cannot_join_the_drain_worker_before_its_thread_starts(
    factory, monkeypatch,
):
    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking"])
    starting, allow_start, stopped = Event(), Event(), Event()
    errors = []

    class DelayedDrainThread(Thread):
        def start(self):
            if self.name == "scene-source-end":
                starting.set()
                assert allow_start.wait(3)
            super().start()

    monkeypatch.setattr(service, "Thread", DelayedDrainThread)

    def stop_capture():
        try:
            assert runtime.stop_session()
        except Exception as exc:
            errors.append(exc)
        finally:
            stopped.set()

    stopper = Thread(target=stop_capture, daemon=True)
    try:
        stopper.start()
        assert starting.wait(2)
        # stop_session may time out its short wait, but it must not join the
        # drain thread while Thread.start() is still publishing OS ownership.
        assert stopped.wait(1)
        assert not errors
    finally:
        allow_start.set()
        stopper.join(timeout=3)
    _await(lambda: getattr(runtime, "_end_drain_thread", None) is not None)
    runtime._end_drain_thread.join(timeout=3)
    assert not runtime._end_drain_thread.is_alive()
    assert runtime.status()["pipeline_state"] == "idle"


def test_rejected_camera_sample_does_not_advance_source_receipt(factory, monkeypatch):
    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking"])
    session = runtime._session
    source = {
        "stream_session_id": session.stream_session_id, "generation": 1,
        "crop_version": 0, "source_sequence": 50, "media_timestamp": 2.0,
    }
    original = session.push_camera_frame
    monkeypatch.setattr(session, "push_camera_frame", lambda *_a, **_k: False)
    assert not runtime.push_camera_frame(_jpeg(), width=240, height=160, **source)
    assert runtime._camera_display_metadata.last_source == (-1, -1.0)
    assert runtime._camera_next_sample_at == 0
    monkeypatch.setattr(session, "push_camera_frame", original)
    assert runtime.push_camera_frame(_jpeg(), width=240, height=160, **source)
    assert runtime._camera_display_metadata.last_source == (50, 2.0)


def test_background_snapshot_refreshes_after_gallery_delete(factory, monkeypatch):
    from types import SimpleNamespace

    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking"])
    session, scene = runtime._session, runtime._scene
    entered, release = Event(), Event()
    original_wait = session.archive.wait_until_idle
    original_snapshot = scene.snapshot
    identity = {"person_id": "deleted-person", "name": "Old name"}

    def snapshot(*args, **kwargs):
        result = original_snapshot(*args, **kwargs)
        result["identity_probe"] = dict(identity)
        return result

    def gated_wait(*args, **kwargs):
        if kwargs.get("stream_session_id") == session.stream_session_id:
            entered.set()
            if not release.wait(kwargs.get("timeout_seconds", 0.1)):
                return False
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(scene, "snapshot", snapshot)
    monkeypatch.setattr(scene, "forget_identity", lambda _person: identity.clear())
    monkeypatch.setattr(session.archive, "wait_until_idle", gated_wait)
    monkeypatch.setattr(runtime, "_ensure_face_runtime", lambda: SimpleNamespace(
        delete_person=lambda _person: True,
    ))
    try:
        assert runtime.stop_session()
        assert entered.wait(2)
        assert runtime.delete_face("deleted-person")
        release.set()
        _await(lambda: runtime._run_postprocess[session.stream_session_id].state
               == "finalized")
        assert runtime.scene_snapshot()["identity_probe"] == {}
    finally:
        release.set()


def test_scene_archive_failure_cannot_finalize_run(factory, monkeypatch):
    runtime = factory()
    runtime.start_camera(dry_run=True, features=["smoking"])
    sid = runtime._session.stream_session_id
    monkeypatch.setattr(
        runtime._scene_archive, "failed_for_run", lambda run: run == sid,
    )
    assert runtime.stop_session()
    _await(lambda: runtime._run_postprocess[sid].state == "failed")
    assert runtime._run_postprocess[sid].error == "postprocess_failed"
    assert runtime.status()["scene_drain"] == {"state": "failed"}


def test_worker_start_failure_stops_already_started_source(factory, monkeypatch):
    runtime = factory(tracker=True, inspector=Inspector())
    started = []
    original_start = pipeline.StreamingSession.start

    def source_start(session):
        started.append(session)
        original_start(session)

    def fail_worker_start(_worker):
        raise RuntimeError("injected worker startup failure")

    monkeypatch.setattr(pipeline.StreamingSession, "start", source_start)
    monkeypatch.setattr(service.SceneInspectionWorker, "start", fail_worker_start)
    with pytest.raises(RuntimeError, match="injected worker"):
        runtime.start_camera(dry_run=False, features=["smoking"])
    assert len(started) == 1
    assert not started[0].running
    assert runtime._session is None
    assert runtime.status()["pipeline_state"] == "error"
    assert runtime._run_postprocess[started[0].stream_session_id].state == "failed"


def test_blocked_status_does_not_stall_display_metadata_event_loop():
    import asyncio
    from threading import Timer
    from types import SimpleNamespace

    import httpx

    from goodvision_harness.camera_display import CameraDisplayMetadata

    entered, release = Event(), Event()
    metadata = CameraDisplayMetadata()

    def slow_status():
        entered.set()
        release.wait(2)
        return {"pipeline_state": "running"}

    app = service.create_app()
    app.state.runtime = SimpleNamespace(
        status=slow_status, _camera_display_metadata=metadata,
    )

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            started = time.monotonic()
            blocked = asyncio.create_task(client.get("/v1/status"))
            try:
                assert await asyncio.to_thread(entered.wait, 1)
                response = await client.get("/v1/video/display/metadata")
                assert response.status_code == 200
                assert time.monotonic() - started < 1
                assert not blocked.done()
            finally:
                release.set()
                await blocked

    watchdog = Timer(1.5, release.set)
    watchdog.start()
    try:
        asyncio.run(check())
    finally:
        release.set()
        watchdog.cancel()
