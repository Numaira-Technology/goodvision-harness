"""Execution locations persist independently and build isolated model routes."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import (
    DeploymentSettings,
    DeploymentSettingsStore,
)
from goodvision_harness.local_gpu import LocalGPUDeployment
from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.service_locations import (
    SERVICE_LOCATION_FIELDS,
    ServiceLocations,
    validate_service_location,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    for name in (
        "INFERENCE_EXECUTION_MODE",
        "TRACKING_EXECUTION_MODE",
        "INFERENCE_SERVER_URL",
        "INFERENCE_SERVER_MODEL",
        "TRACKING_SERVER_URL",
        "TRACKING_SERVER_CAPACITY",
        "LOCAL_INFERENCE_ENDPOINT",
        "LOCAL_INFERENCE_MODEL",
        "LOCAL_INFERENCE_MAX_TOKENS",
        "LOCAL_TRACKING_ENDPOINT",
        "LOCAL_TRACKING_FPS",
        "LOCAL_GPU_AVAILABLE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def base():
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    return replace(
        config,
        aiclub=ModelAPISettings(
            provider="openrouter",
            extra_headers={"Authorization": "cloud-secret"},
        ),
    )


@pytest.mark.parametrize(
    "inference,tracking",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_old_local_flags_select_compatible_modes(inference, tracking):
    locations = ServiceLocations.resolve(
        None,
        local_inference=inference,
        local_tracking=tracking,
    )
    assert locations.inference_mode == ("local" if inference else "cloud")
    assert locations.tracking_mode == ("local" if tracking else "legacy")


def test_server_environment_overrides_legacy_flags_and_needs_no_primary_gpu(
    monkeypatch,
    base,
):
    for name, value in {
        "INFERENCE_EXECUTION_MODE": "server",
        "TRACKING_EXECUTION_MODE": "server",
        "INFERENCE_SERVER_URL": "http://inference-machine:8000/",
        "INFERENCE_SERVER_MODEL": "customer-model",
        "TRACKING_SERVER_URL": "http://sam:8092",
        "TRACKING_SERVER_CAPACITY": "24",
    }.items():
        monkeypatch.setenv(name, value)
    locations = ServiceLocations.resolve(
        None, local_inference=True, local_tracking=True
    )
    gpu = LocalGPUDeployment()
    config = locations.configure(base, gpu, camera_index=16)
    assert (
        config.model_api.endpoint == "http://inference-machine:8000/v1/chat/completions"
    )
    assert config.inference.model == "customer-model"
    assert config.rich_labeling.endpoint == (
        "http://sam:8092/v1/cameras/camera-16/perception/sam_live/sessions"
    )
    assert locations.tracking_capacity(32, gpu) == 24


def test_persisted_locations_survive_restart_and_take_precedence_over_environment(
    tmp_path,
    monkeypatch,
):
    locations = ServiceLocations(
        inference_mode="cloud",
        tracking_mode="server",
        inference_server_url="",
        tracking_server_url="https://tracking.company:9443/",
        tracking_server_capacity=16,
    )
    store = DeploymentSettingsStore(tmp_path / "deployment.json")
    saved = DeploymentSettings(
        "rtsp://source/live",
        False,
        False,
        local_inference_enabled=False,
        local_tracking_enabled=False,
        **locations.public_status(),
    )
    store.save(saved)
    restored = store.load()
    assert restored == saved
    monkeypatch.setenv("TRACKING_EXECUTION_MODE", "legacy")
    monkeypatch.setenv("TRACKING_SERVER_URL", "http://wrong-server")
    monkeypatch.setenv("TRACKING_SERVER_CAPACITY", "1")
    assert (
        ServiceLocations.resolve(
            restored,
            local_inference=False,
            local_tracking=False,
        )
        == locations
    )
    assert set(locations.public_status()) == SERVICE_LOCATION_FIELDS


def test_old_saved_settings_do_not_require_new_fields(tmp_path):
    store = DeploymentSettingsStore(tmp_path / "deployment.json")
    original = DeploymentSettings("", False, False, local_tracking_enabled=True)
    store.save(original)
    assert store.load() == original
    assert (
        ServiceLocations.resolve(
            store.load(),
            local_inference=False,
            local_tracking=True,
        ).tracking_mode
        == "local"
    )


@pytest.mark.parametrize(
    "value,normalized",
    [
        ("http://192.168.1.8:8092", "http://192.168.1.8:8092"),
        ("https://GPU-SERVER:443/", "https://gpu-server:443"),
        ("http://[::1]:8092/", "http://[::1]:8092"),
        ("http://sam-server", "http://sam-server"),
        ("", ""),
    ],
)
def test_origin_validation_accepts_plain_hosts_addresses_and_optional_port(
    value, normalized
):
    assert validate_service_location("tracking_server_url", value) == normalized


@pytest.mark.parametrize(
    "value",
    [
        None,
        5,
        [],
        "ftp://host:8092",
        "http://user:secret@host:8092",
        "http://@host:8092",
        "http://host/v1/video/sessions",
        "http://host?token=secret",
        "http://host?",
        "http://host#fragment",
        "http://host#",
        "http://host:0",
        "http://host:65536",
        "http://host:port",
        "http://host:",
        "http://",
        "host:8092",
        "//host:8092",
        "http://host\\evil",
        "http://host\nevil",
        "http://host%2Fother",
        "http://host|other",
        "http://.",
        "http://[invalid-ipv6]:8092",
    ],
)
def test_service_origin_rejects_credentials_paths_and_malformed_authorities(value):
    with pytest.raises(ValueError):
        validate_service_location("inference_server_url", value)


@pytest.mark.parametrize("capacity", [0, -1, 1.5, "12", True, False, None])
def test_server_capacity_is_a_strict_positive_integer(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        ServiceLocations(tracking_server_capacity=capacity)


@pytest.mark.parametrize("role", ["inference", "tracking"])
def test_server_mode_requires_url_but_inactive_server_url_may_be_empty(role):
    with pytest.raises(ValueError, match="required"):
        ServiceLocations().updated({f"{role}_mode": "server"})
    configured = ServiceLocations().updated(
        {
            f"{role}_mode": "server",
            f"{role}_server_url": "http://server:8092",
        }
    )
    assert getattr(configured, f"{role}_server_url") == "http://server:8092"


@pytest.mark.parametrize(
    "mode,old,accepted",
    [
        ("local", True, True),
        ("server", False, True),
        ("cloud", False, True),
        ("server", True, False),
        ("cloud", True, False),
        ("local", False, False),
    ],
)
def test_updated_new_modes_and_legacy_flags_cannot_contradict(mode, old, accepted):
    initial = ServiceLocations(inference_server_url="http://model:8000")
    payload = {"inference_mode": mode, "local_inference_enabled": old}
    if accepted:
        assert initial.updated(payload).inference_mode == mode
    else:
        with pytest.raises(ValueError, match="contradicts"):
            initial.updated(payload)


def test_legacy_updates_translate_only_the_corresponding_mode():
    initial = ServiceLocations(
        inference_mode="server",
        inference_server_url="http://model:8000",
        tracking_mode="server",
        tracking_server_url="http://tracking:8092",
    )
    updated = initial.updated(
        {"local_inference_enabled": False, "stream_url": "ignored"}
    )
    assert updated.inference_mode == "cloud"
    assert updated.tracking_mode == "server"
    assert initial.inference_mode == "server"
    with pytest.raises(ValueError, match="boolean"):
        initial.updated({"local_tracking_enabled": "true"})


def test_cloud_inference_remains_unchanged_with_remote_self_hosted_tracking(base):
    locations = ServiceLocations(
        tracking_mode="server",
        tracking_server_url="http://gpu-tracker:8092",
    )
    config = locations.configure(base, LocalGPUDeployment())
    assert config.model_api == base.model_api
    assert config.inference == base.inference
    assert config.rich_labeling.endpoint == "http://gpu-tracker:8092/v1/video/sessions"
    assert config.rich_labeling.backend == "sam31_stream"
    assert config.rich_labeling.processing_method == "sam_live"
    assert config.rich_labeling.max_fps == 2
    assert config.runtime.scene_mode == "active"


@pytest.mark.parametrize("bad", [None, True, "2", 0, .49, 30.01, float("nan"), float("inf"), 10**500])
def test_tracking_sample_rate_rejects_invalid_console_values(bad):
    with pytest.raises(ValueError, match="tracking_sample_fps"):
        ServiceLocations().updated({"tracking_sample_fps": bad})


@pytest.mark.parametrize("mode", ["legacy", "local", "server"])
def test_sampling_rate_only_changes_tracking_for_all_execution_modes(base, mode):
    base = replace(base, rich_labeling=replace(base.rich_labeling, backend="sam31_stream"))
    locations = ServiceLocations(tracking_mode=mode, tracking_server_url="http://sam:8092",
                                 tracking_sample_fps=1.5)
    config = locations.configure(base, LocalGPUDeployment(available=True, gpu_count=1))
    assert config.rich_labeling.max_fps == 1.5
    assert config.display == base.display
    assert config.face == base.face
    assert config.video == base.video


def test_saved_sample_rate_overrides_environment_after_restart(tmp_path, monkeypatch):
    store = DeploymentSettingsStore(tmp_path / "deployment.json")
    store.save(DeploymentSettings("", False, False, tracking_sample_fps=2.5))
    monkeypatch.setenv("LOCAL_TRACKING_FPS", "4")
    resolved = ServiceLocations.resolve(store.load(), local_inference=False, local_tracking=False)
    assert resolved.tracking_sample_fps == 2.5
    assert ServiceLocations.resolve(None, local_inference=False, local_tracking=False).tracking_sample_fps == 4


def test_server_inference_is_a_fresh_route_without_cloud_headers_or_parameters(base):
    locations = ServiceLocations(
        inference_mode="server",
        inference_server_url="https://model:8000",
        inference_server_model="customer-vision",
    )
    config = locations.configure(base, LocalGPUDeployment())
    assert config.model_api.provider == "local"
    assert config.model_api.api_key_env == "LOCAL_INFERENCE_API_KEY"
    assert not config.model_api.extra_headers
    assert config.inference.request_parameters == {"temperature": 0, "max_tokens": 1600}
    assert config.runtime.scene_max_tokens == 1600
    assert config.rich_labeling == base.rich_labeling


def test_local_mode_still_requires_local_gpu_but_server_mode_does_not(base):
    for payload in ({"inference_mode": "local"}, {"tracking_mode": "local"}):
        with pytest.raises(ValueError, match="GPU"):
            ServiceLocations(**payload).configure(base, LocalGPUDeployment())
    server = ServiceLocations(tracking_mode="server", tracking_server_url="http://sam")
    server.configure(base, LocalGPUDeployment())


def test_tracking_capacity_distinguishes_legacy_local_and_server():
    gpu = LocalGPUDeployment(True, 1)
    assert ServiceLocations().tracking_capacity(24, gpu) == 1
    assert ServiceLocations(tracking_mode="local").tracking_capacity(24, gpu) == 24
    assert (
        ServiceLocations(tracking_mode="local").tracking_capacity(
            24, LocalGPUDeployment()
        )
        == 1
    )
    locations = ServiceLocations(
        tracking_mode="server",
        tracking_server_url="http://sam",
        tracking_server_capacity=100,
    )
    assert locations.tracking_capacity(24, LocalGPUDeployment()) == 100


def test_persistent_invalid_new_setting_is_rejected_without_overwriting_existing(
    tmp_path,
):
    store = DeploymentSettingsStore(tmp_path / "deployment.json")
    original = DeploymentSettings("", False, False)
    store.save(original)
    before = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.save(replace(original, tracking_server_url="http://user:password@sam"))
    assert store.path.read_bytes() == before


@pytest.mark.parametrize("capacity,index", [(1, 1), (16, 16), (100, 100)])
def test_server_tracking_readiness_uses_actual_perception_client_routes(
    base,
    capacity,
    index,
):
    from goodvision_harness.perception import PerceptionSessionHTTPClient
    from goodvision_harness.video_tracking_session import LIMIT_FIELDS

    config = ServiceLocations(
        tracking_mode="server",
        tracking_server_url="http://sam-machine:8092",
        tracking_server_capacity=capacity,
    ).configure(base, LocalGPUDeployment(), index)
    requested = []
    limits = dict.fromkeys(LIMIT_FIELDS, 1)
    limits.update(
        history_frames=32,
        startup_frames=1,
        max_append_frames=4,
        hotstart_delay=1,
        max_pending_frames=0,
    )

    def health(request):
        requested.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "enabled": True,
                "ready": True,
                "gpu_ready": True,
                "backend": "sam3.1_video",
                "memory_scope": "session",
                "native_sam_video_tracking": True,
                "sam3_commit": "worker-commit",
                "limits": limits,
                "error_code": None,
                "active_session": False,
            },
        )

    client = PerceptionSessionHTTPClient(
        config.rich_labeling.endpoint,
        method=config.rich_labeling.processing_method,
        transport=httpx.MockTransport(health),
    )
    try:
        assert client.ready_for_new_session()
        prefix = f"/cameras/camera-{index:02d}" if capacity > 1 else ""
        assert requested == [f"/health{prefix}/perception/sam_live"]
        assert client.endpoint == (
            f"http://sam-machine:8092/v1{prefix}/perception/sam_live/sessions"
        )
    finally:
        client.close()


def test_invalid_server_configuration_can_be_recovered_and_restarted(
    tmp_path,
    monkeypatch,
    base,
):
    from goodvision_harness.service import HarnessRuntime

    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("TRACKING_EXECUTION_MODE", "server")
    monkeypatch.setenv("TRACKING_SERVER_URL", "http://bad-server/unsupported/path")
    monkeypatch.setattr(
        "goodvision_harness.service.create_tracking_client",
        lambda _: SimpleNamespace(gpu_ready=lambda: False, close=lambda: None),
    )
    config = replace(
        base,
        rich_labeling=replace(
            base.rich_labeling,
            enabled=False,
            tracking_enabled=False,
            backend="sam31_stream",
            processing_method=None,
        ),
        runtime=replace(base.runtime, scene_mode="active"),
        events=replace(base.events, directory=tmp_path / "events"),
    )
    store = DeploymentSettingsStore(tmp_path / "deployment.json")
    runtime = HarnessRuntime(config, deployment_store=store)
    restarted = None
    try:
        assert runtime.config_error
        settings = runtime.update_deployment_settings(
            {
                "tracking_mode": "server",
                "tracking_server_url": "http://sam-machine:8092",
                "tracking_server_capacity": 16,
            }
        )
        assert runtime.config_error is None
        assert settings["tracking_server_url"] == "http://sam-machine:8092"
        assert settings["local_gpu"]["available"] is False
        restarted = HarnessRuntime(config, deployment_store=store)
        assert restarted.config_error is None
        assert restarted.config == runtime.config
    finally:
        runtime.stop()
        if restarted is not None:
            restarted.stop()
