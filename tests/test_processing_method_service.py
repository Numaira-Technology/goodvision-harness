"""Switch real service sessions while keeping perception plugins isolated."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from goodvision_harness import perception, service
from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettingsStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.delenv("PERCEPTION_METHOD", raising=False)
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        rich_labeling=replace(
            config.rich_labeling,
            enabled=True,
            backend="sam31_stream",
            processing_method="sam_live",
        ),
        runtime=replace(config.runtime, scene_mode="active"),
        events=replace(config.events, directory=tmp_path / "events"),
        service=replace(
            config.service, webhook=replace(config.service.webhook, url=None)
        ),
    )
    clients, coordinators, callbacks, runtimes = [], [], [], []
    control = SimpleNamespace(ready=True, probe=None, remote_busy=False)

    class Client:
        startup_frames = 1
        max_append_frames = 4

        def __init__(self, rich):
            self.method = rich.processing_method
            self.closed = False
            clients.append(self)

        def gpu_ready(self):
            return control.probe(self) if control.probe is not None else control.ready

        def ready_for_new_session(self):
            return self.gpu_ready() and not control.remote_busy

        def close_complete(self):
            return self.closed

        def close(self):
            self.closed = True

        def transport_stats(self):
            return {}

        def discard_session(self):
            pass

    def coordinator(client, rich, on_batch):
        callbacks.append(on_batch)
        value = perception.create_tracking_coordinator(client, rich, on_batch)
        coordinators.append(value)
        return value

    monkeypatch.setattr(service, "create_tracking_client", Client)
    monkeypatch.setattr(service, "create_tracking_coordinator", coordinator)
    store = DeploymentSettingsStore(tmp_path / "settings.json")

    def make(configuration=None):
        value = service.HarnessRuntime(configuration or config, deployment_store=store)
        runtimes.append(value)
        return value

    value = SimpleNamespace(
        make=make,
        config=config,
        store=store,
        clients=clients,
        coordinators=coordinators,
        callbacks=callbacks,
        control=control,
    )
    yield value
    for runtime in runtimes:
        runtime.stop()


def test_stop_save_start_and_restart_bind_only_the_selected_plugin(rig):
    runtime = rig.make()
    first = runtime.start_camera(dry_run=True, features=["smoking"])
    assert first["processing_method"] == "sam_live"
    first_coordinator = runtime._rich_labels
    with pytest.raises(RuntimeError, match="active session"):
        runtime.update_deployment_settings({"processing_method": "hybrid"})
    assert runtime.stop_session()
    payload = runtime.update_deployment_settings({"processing_method": "hybrid"})
    assert payload["processing_method"] == "hybrid"
    assert runtime._rich_labels is None
    assert first_coordinator.stats()["state"] == "closed"
    assert rig.store.load().processing_method == "hybrid"
    second = runtime.start_camera(dry_run=True, features=["smoking"])
    assert second["stream_session_id"] != first["stream_session_id"]
    assert second["processing_method"] == "hybrid"
    assert runtime._rich_labels is not first_coordinator
    assert rig.clients[-1].method == "hybrid"
    assert runtime.config.inference == rig.config.inference
    assert runtime.config.model_api == rig.config.model_api
    assert runtime.stop_session()
    runtime.update_deployment_settings(
        {"buffer_delay_seconds": 0.5, "autostart": False}
    )
    restored = rig.make()
    assert restored.config.rich_labeling.processing_method == "hybrid"
    assert restored.ui_info()["processing_method"] == "hybrid"
    assert restored.deployment_settings()["buffer_delay_seconds"] == 0.5


def test_old_plugin_callback_cannot_commit_to_a_new_route_even_with_current_session_id(
    rig,
):
    runtime = rig.make()
    runtime.start_camera(dry_run=True, features=["smoking"])
    old_callback = rig.callbacks[0]
    assert runtime.stop_session()
    runtime.update_deployment_settings({"processing_method": "hybrid"})
    new_session = runtime.start_camera(dry_run=True, features=["smoking"])
    before = runtime.scene_snapshot()
    old_callback(
        SimpleNamespace(
            stream_session_id=new_session["stream_session_id"],
            predictor_epoch="retired-route",
            frames=(),
            completed_timestamp_seconds=0,
            completed_frame_sequence=0,
        )
    )
    assert runtime.scene_snapshot() == before


def test_inflight_readiness_for_old_route_is_closed_and_cannot_install_after_switch(
    rig,
):
    rig.control.ready = False
    runtime = rig.make()
    entered, release = Event(), Event()

    def blocked(client):
        if client.method == "sam_live":
            entered.set()
            assert release.wait(3)
        return True

    rig.control.probe = blocked
    probe = Thread(target=runtime._try_enable_rich_labels)
    probe.start()
    try:
        assert entered.wait(1)
        runtime.update_deployment_settings({"processing_method": "hybrid"})
        release.set()
        probe.join(2)
        assert not probe.is_alive()
        assert runtime._rich_labels is None
        assert all(client.closed for client in rig.clients)
        assert runtime._try_enable_rich_labels()
        assert rig.clients[-1].method == "hybrid"
        assert runtime._rich_labels is not None
    finally:
        release.set()
        probe.join(3)


@pytest.mark.parametrize("state", ["starting", "running", "stopping", "draining"])
def test_route_api_rejects_unfinished_lifecycle_even_after_source_thread_exits(
    rig, state
):
    runtime = rig.make()
    with TestClient(service.create_app(lambda: runtime)) as client:
        runtime._update_state(pipeline_state=state)
        response = client.put(
            "/v1/deployment/settings", json={"processing_method": "hybrid"}
        )
        assert response.status_code == 409
        assert runtime._processing_method() == "sam_live"
        assert rig.store.load() is None
        runtime._update_state(pipeline_state="idle")


def test_route_change_waits_for_native_cleanup_and_does_not_mutate_early(
    rig, monkeypatch
):
    runtime = rig.make()
    coordinator = runtime._rich_labels
    monkeypatch.setattr(coordinator, "wait_until_idle", lambda **_: False)
    with pytest.raises(RuntimeError, match="active session"):
        runtime.update_deployment_settings({"processing_method": "hybrid"})
    assert rig.store.load() is None
    assert runtime._rich_labels is coordinator
    monkeypatch.setattr(coordinator, "wait_until_idle", lambda **_: True)
    runtime.update_deployment_settings({"processing_method": "hybrid"})
    assert runtime._rich_labels is None


@pytest.mark.parametrize("invalid", ["removed", "openrouter", None, True, []])
def test_invalid_mixed_route_update_keeps_disk_client_and_other_settings(rig, invalid):
    runtime = rig.make()
    runtime.update_deployment_settings({"processing_method": "sam_live"})
    before = rig.store.path.read_bytes()
    coordinator = runtime._rich_labels
    with pytest.raises(ValueError):
        runtime.update_deployment_settings(
            {
                "processing_method": invalid,
                "dry_run": True,
                "api_key": "test-new-key",
            }
        )
    assert rig.store.path.read_bytes() == before
    assert runtime._rich_labels is coordinator
    assert runtime.dry_run is False
    assert rig.store.load_api_key() is None


def test_invalid_other_field_rejects_valid_method_before_persisting_or_closing(rig):
    runtime = rig.make()
    original = runtime._rich_labels
    with pytest.raises(ValueError, match="control characters"):
        runtime.update_deployment_settings(
            {"processing_method": "hybrid", "api_key": "bad\nkey"}
        )
    assert rig.store.load() is None
    assert runtime._rich_labels is original
    assert not rig.clients[0].closed


def test_offline_methods_can_be_saved_but_start_fails_explicitly_before_capture(
    rig, monkeypatch
):
    rig.control.ready = False
    runtime = rig.make()
    monkeypatch.setattr(runtime, "_ensure_rich_readiness_probe", lambda: None)
    with TestClient(service.create_app(lambda: runtime)) as client:
        before = len(rig.clients)
        saved = client.put(
            "/v1/deployment/settings", json={"processing_method": "hybrid"}
        )
        assert saved.status_code == 200
        assert all(method["available"] for method in saved.json()["processing_methods"])
        assert len(rig.clients) == before, "saving a route does not probe GPU readiness"
        response = client.post(
            "/v1/sessions/start",
            json={
                "stream_url": "rtsp://fixture.test/live",
                "features": ["smoking"],
                "dry_run": True,
            },
        )
        assert response.status_code == 503
        assert "人物处理服务尚未就绪" in response.json()["detail"]
        assert runtime._session is None
        with client.websocket_connect("/ws") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "mode": "camera",
                    "features": ["smoking"],
                    "dry_run": True,
                }
            )
            message = ws.receive_json()
            while message.get("type") in {"status", "hello"}:
                message = ws.receive_json()
            assert message["type"] == "error"
            assert "人物处理服务尚未就绪" in message["message"]
        assert runtime._session is None


def test_previous_health_success_does_not_allow_start_after_worker_goes_offline(rig):
    runtime = rig.make()
    assert runtime._rich_labels is not None
    rig.control.ready = False
    with pytest.raises(service.ProcessingMethodUnavailableError):
        runtime.start_camera(dry_run=True, features=["smoking"])
    assert runtime._session is None
    rig.control.ready = True
    started = runtime.start_camera(dry_run=True, features=["smoking"])
    assert started["processing_method"] == "sam_live"


def test_switch_retains_local_cleanup_and_checks_global_remote_ownership_before_capture(
    rig, monkeypatch
):
    runtime = rig.make()
    old = runtime._rich_labels
    released = Event()
    monkeypatch.setattr(old, "close_complete", released.is_set)
    try:
        saved = runtime.update_deployment_settings({"processing_method": "hybrid"})
        assert saved["processing_method"] == "hybrid"
        assert saved["processing_cleanup"] == "closing"
        assert runtime._retiring_rich_labels is old
        assert runtime.status()["pipeline_state"] == "stopping"
        assert runtime.status()["processing_retiring_method"] == "sam_live"
        assert runtime.stop_session() is False
        assert runtime.status()["pipeline_state"] == "stopping"
        assert runtime.status()["processing_cleanup"] == "closing"
        with pytest.raises(RuntimeError, match="active session"):
            runtime.start_camera(dry_run=True, features=["smoking"])
        with pytest.raises(RuntimeError, match="active session"):
            runtime.update_deployment_settings({"processing_method": "sam_live"})
        released.set()
        runtime._rich_retirement_thread.join(2)
        assert runtime._retiring_rich_labels is None
        assert runtime.status()["processing_cleanup"] == "remote_check_pending"
        rig.control.remote_busy = True
        with pytest.raises(
            service.ProcessingMethodUnavailableError, match="旧会话仍在收尾"
        ):
            runtime.start_camera(dry_run=True, features=["smoking"])
        assert runtime._session is None
        rig.control.remote_busy = False
        assert (
            runtime.start_camera(dry_run=True, features=["smoking"])[
                "processing_method"
            ]
            == "hybrid"
        )
        assert runtime.status()["processing_cleanup"] is None
    finally:
        released.set()


def test_stopping_preserves_last_measured_method_statistics_for_comparison(rig):
    runtime = rig.make()
    runtime.start_camera(dry_run=True, features=["smoking"])
    coordinator = runtime._rich_labels
    with coordinator._condition:
        coordinator._stats.update(
            processed_frames=19, source_to_result_seconds=1.25, dropped_backpressure=3
        )
    assert runtime.stop_session()
    stats = runtime.status()["processing"]
    assert stats["method"] == "sam_live"
    assert stats["processed_frames"] == 19
    assert stats["source_to_result_seconds"] == 1.25
    assert stats["dropped_frames"] == 3
    runtime.start_camera(dry_run=True, features=["smoking"])
    assert runtime.status()["processing"]["processed_frames"] == 0
    assert runtime.status()["processing"]["source_to_result_seconds"] is None
