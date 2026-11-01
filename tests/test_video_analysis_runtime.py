"""Preview and global analysis use the same source with separate lifecycles."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_local_gpu_runtime import rig as rig
from test_video_connection import ControlledAdapter, frame, wait_for

from goodvision_harness import video_connection
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.service import HarnessRuntime, MissingAPIKeyError
from goodvision_harness.video_to_stream import LiveFrame


@pytest.fixture
def connected_runtime(rig, monkeypatch):
    adapters = []

    def make_adapter(url, **kwargs):
        adapter = ControlledAdapter(url, **kwargs)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(video_connection, "NetworkStreamAdapter", make_adapter)
    runtime = rig.make()
    runtime.connect_stream("rtsp://private-user:private-password@camera.local/live")
    connection = runtime._video_connection
    adapter = adapters[0]
    # Pretend the preview was opened several minutes before analysis starts.
    adapter.items.put(frame(9000))
    wait_for(lambda: connection.latest_jpeg() is not None)
    return runtime, connection, adapter, adapters


def test_connect_only_needs_video_and_never_checks_model_or_key(rig, monkeypatch):
    runtime = rig.make()
    runtime._aiclub_api_key = None
    adapters = []

    def make_adapter(url, **kwargs):
        adapter = ControlledAdapter(url, **kwargs)
        adapters.append(adapter)
        return adapter

    def forbidden(*args, **kwargs):
        pytest.fail("connecting preview must not create or probe a model")

    monkeypatch.setattr(video_connection, "NetworkStreamAdapter", make_adapter)
    for name in ("_require_processing_ready", "_ensure_face_runtime", "start_stream"):
        monkeypatch.setattr(runtime, name, forbidden)
    for name in (
        "build_selected_features_grounding_client", "build_grounding_client",
        "create_tracking_client",
    ):
        monkeypatch.setattr(f"goodvision_harness.service.{name}", forbidden)
    result = runtime.connect_stream("rtsp://private-user:password@camera.local/live")
    assert result["connection_state"] == "connecting"
    assert result["analysis_state"] == "idle"
    adapters[0].items.put(frame(0))
    wait_for(lambda: runtime._video_connection.latest_jpeg() is not None)
    status = runtime.status()
    assert status["connected"] is True
    assert status["analysis_state"] == "idle"
    assert status["source_frames_received"] == 0
    assert runtime._session is None
    assert "private-user" not in str(status)
    assert "password" not in str(status)


def test_missing_analysis_key_does_not_disconnect_or_clear_preview(
    connected_runtime, monkeypatch,
):
    runtime, connection, adapter, _ = connected_runtime
    runtime.update_analysis_settings({"features": ["smoking"]})
    runtime._aiclub_api_key = None
    runtime.dry_run = False
    jpeg = connection.latest_jpeg()
    monkeypatch.setattr(
        runtime, "_require_processing_ready",
        lambda: pytest.fail("missing key must be checked before model readiness"),
    )
    with pytest.raises(MissingAPIKeyError):
        runtime.prepare_analysis()
    assert connection.running
    assert runtime.status()["connected"]
    assert connection.latest_jpeg() == jpeg
    assert not adapter.stop_requested.is_set()
    assert runtime._session is None


@pytest.mark.parametrize(
    ("scene_mode", "independent", "dry_run", "features"),
    [
        ("active", False, False, ["smoking"]),
        ("disabled", True, False, ["smoking"]),
        ("active", True, True, ["smoking"]),
        ("active", True, False, ["stranger"]),
    ],
)
def test_preflight_preserves_processing_readiness_outside_independent_events(
    connected_runtime, monkeypatch, scene_mode, independent, dry_run, features,
):
    runtime, connection, _adapter, _ = connected_runtime
    runtime.config = replace(
        runtime.config,
        runtime=replace(
            runtime.config.runtime,
            scene_mode=scene_mode,
            scene_independent_vlm=independent,
        ),
    )
    runtime._scene_enabled = scene_mode == "active"
    runtime.dry_run = dry_run
    runtime.update_analysis_settings({"features": features})

    def unavailable():
        raise RuntimeError("selected processing method is unavailable")

    monkeypatch.setattr(runtime, "_require_processing_ready", unavailable)
    with pytest.raises(RuntimeError, match="selected processing method"):
        runtime.prepare_analysis()
    assert connection.running
    assert runtime._session is None


def test_camera_analysis_choices_persist_independently_and_invalid_save_is_atomic(
    rig, tmp_path,
):
    first = rig.make()
    first.update_analysis_settings({
        "features": ["smoking", "weapon"],
        "event_sensitivities": {"smoking": "standard", "weapon": "high"},
    })
    second_store = DeploymentSettingsStore(tmp_path / "camera02" / "deployment.json")
    second = HarnessRuntime(rig.config, deployment_store=second_store)
    try:
        second.update_analysis_settings({
            "features": ["fall"], "event_sensitivities": {"fall": "high"},
        })
        persisted = rig.store.path.with_name("analysis-settings.json").read_bytes()
        with pytest.raises(ValueError, match="sensitivity"):
            first.update_analysis_settings({
                "features": ["smoking"], "event_sensitivities": {"smoking": "bad"},
            })
        assert (
            rig.store.path.with_name("analysis-settings.json").read_bytes() == persisted
        )
        assert second.analysis_settings()["features"] == ["fall"]
        restarted = rig.make()
        assert restarted.analysis_settings()["features"] == ["smoking", "weapon"]
        assert restarted.analysis_settings()["event_sensitivities"] == {
            "smoking": "standard", "weapon": "high",
        }
    finally:
        second.stop()


def test_analysis_starts_and_stops_on_existing_source_and_video_keeps_running(
    connected_runtime,
):
    runtime, connection, adapter, adapters = connected_runtime
    # This lifecycle test uses a two-second watchdog and a manually fed source.
    # Do not inherit a production playback buffer longer than that watchdog.
    runtime.update_deployment_settings({"dry_run": True, "buffer_delay_seconds": 0})
    runtime.update_analysis_settings({"features": ["smoking"]})
    ready = runtime.prepare_analysis()
    assert ready == {"prepared": True, "connection_id": connection.connection_id}
    runtime.start_connected_analysis()
    session = runtime._session
    assert session is not None
    assert session.video_stream._connection is connection
    assert len(adapters) == 1
    assert adapter.frames_calls == 1
    assert runtime.status()["source_frames_received"] == 0
    adapter.items.put(frame(9001))
    wait_for(lambda: runtime.status()["source_frames_received"] >= 1)
    assert runtime.status()["last_frame_sequence"] == 9001
    assert runtime.status()["last_media_timestamp"] == round(9001 / 15, 3)
    with pytest.raises(RuntimeError, match="stop"):
        runtime.disconnect_stream()
    with pytest.raises(RuntimeError, match="stop"):
        runtime.update_analysis_settings({"features": ["weapon"]})
    assert runtime.stop_session()
    session.join(timeout=2)
    wait_for(lambda: runtime._session is None)
    assert connection.running
    assert not adapter.stop_requested.is_set()
    assert runtime.status()["connected"]
    assert runtime.status()["analysis_state"] == "idle"
    first_sequence = connection._sequence
    adapter.items.put(frame(9010))
    wait_for(lambda: connection._sequence > first_sequence)
    runtime.start_connected_analysis()
    restarted = runtime._session
    assert restarted is not session
    assert restarted.video_stream._connection is connection
    assert len(adapters) == 1
    assert runtime.status()["source_frames_received"] == 0
    adapter.items.put(frame(9011))
    wait_for(lambda: runtime.status()["source_frames_received"] >= 1)
    assert runtime.status()["last_frame_sequence"] == 9011
    runtime.stop_session()
    restarted.join(timeout=2)
    wait_for(lambda: runtime._session is None)
    runtime.disconnect_stream()
    wait_for(lambda: not connection.running)
    assert runtime.status()["connection_state"] == "disconnected"
    assert connection.latest_jpeg() is None


def test_minutes_of_preview_do_not_delay_pipeline_sampling_or_change_frame_identity(
    rig, monkeypatch,
):
    config = replace(rig.config, video=replace(rig.config.video, fps=1.25))
    sources = [
        LiveFrame(index=9000, timestamp_seconds=600.0, bgr=frame(0).bgr),
        LiveFrame(index=9003, timestamp_seconds=600.2, bgr=frame(1).bgr),
        LiveFrame(index=9012, timestamp_seconds=600.8, bgr=frame(2).bgr),
    ]
    adapter = SimpleNamespace(realtime=True, frames=lambda: iter(sources))
    displayed = []
    archived = []
    archive = SimpleNamespace(submit_frame=lambda *args: archived.append(args))
    monkeypatch.setattr(
        "goodvision_harness.pipeline.NetworkStreamAdapter",
        lambda *args, **kwargs: pytest.fail("must reuse connected source"),
    )
    session = StreamingSession.for_stream(
        config, SimpleNamespace(settings=config.inference),
        SessionCallbacks(on_display_frame=lambda *args: displayed.append(args)),
        stream_url="rtsp://camera.local/live", video_stream=adapter, archive=archive,
    )
    sampled = list(session._sampled_frames())
    assert [item.source_frame_index for item in sampled] == [9000, 9012]
    assert [item.timestamp_seconds for item in sampled] == [600.0, 600.8]
    assert [item[1:3] for item in displayed] == [
        (600.0, 9000), (600.2, 9003), (600.8, 9012),
    ]
    assert [item[1:3] for item in archived] == [
        (600.0, 9000), (600.2, 9003), (600.8, 9012),
    ]


def test_connection_failure_during_analysis_start_never_opens_another_decoder(
    connected_runtime, monkeypatch,
):
    runtime, connection, _, adapters = connected_runtime
    runtime.update_deployment_settings({"dry_run": True})
    runtime.update_analysis_settings({"features": ["smoking"]})
    # Fail after preparation/initial admission, before the session's source is
    # attached. A legacy fallback must not silently create another connection.
    monkeypatch.setattr(runtime, "_ensure_delivery", lambda: connection.stop(timeout=1))
    with pytest.raises(RuntimeError):
        runtime.start_connected_analysis()
    assert len(adapters) == 1
    assert runtime._session is None
    assert not runtime.status()["connected"]
    assert runtime.status()["session_start_pending"] is False


def test_secondary_preview_does_not_depend_on_stale_local_tracking_settings(
    rig, monkeypatch,
):
    monkeypatch.setenv("HARNESS_FLEET_CAMERA_INDEX", "2")
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "16")
    runtime = rig.make()
    assert runtime.tracking_camera_capacity() == 16
    assert runtime.service_execution_settings()["tracking_mode"] == "legacy"
    adapter = ControlledAdapter("rtsp://camera02.local/live")
    monkeypatch.setattr(
        video_connection, "NetworkStreamAdapter", lambda *args, **kwargs: adapter,
    )
    monkeypatch.setattr(
        runtime, "_require_processing_ready",
        lambda: pytest.fail("preview must not depend on peer model settings"),
    )
    status = runtime.connect_stream("rtsp://camera02.local/live")
    assert status["connection_state"] == "connecting"
    adapter.items.put(frame(0))
    wait_for(lambda: runtime._video_connection.latest_jpeg() is not None)
    assert runtime.status()["connected"]
    assert runtime._session is None
    assert runtime.service_execution_settings()["tracking_mode"] == "legacy"
