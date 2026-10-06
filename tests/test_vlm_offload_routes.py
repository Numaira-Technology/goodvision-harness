"""Direct and delegated VLM routes remain independent of GPU tracking."""

from dataclasses import replace
from pathlib import Path

import pytest

from goodvision_harness.config import load_config
from goodvision_harness.local_gpu import LocalGPUDeployment
from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.model_routing import resolve_model_route
from goodvision_harness.service_locations import ServiceLocations


@pytest.fixture
def base(monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    for name in (
        "INFERENCE_PROVIDER",
        "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT",
        "HARNESS_INFERENCE_PROFILE",
        "LOCAL_INFERENCE_INSPECTION_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")


def test_goodvision_server_enables_vlm_only_analysis_without_a_primary_gpu(base):
    locations = ServiceLocations(
        inference_mode="server",
        inference_server_url="http://model-host:8000",
        inference_server_model="installed-vision",
        inference_server_protocol="goodvision",
    )
    configured = locations.configure(base, LocalGPUDeployment())
    assert configured.model_api.provider == "local"
    assert (
        configured.model_api.inspection_endpoint
        == "http://model-host:8000/v1/inspections"
    )
    assert configured.model_api.endpoint == "http://model-host:8000/v1/chat/completions"
    assert configured.model_api.extra_headers == {}
    assert configured.inference.model == "installed-vision"
    assert configured.runtime.scene_mode == "active"
    assert locations.tracking_mode == "legacy"
    assert configured.rich_labeling == base.rich_labeling
    direct = replace(locations, inference_server_protocol="compatible").configure(
        base,
        LocalGPUDeployment(),
    )
    assert direct.model_api.inspection_endpoint == ""
    assert direct.runtime.scene_mode == base.runtime.scene_mode


def test_managed_local_vlm_offloads_without_requiring_local_tracking(base, monkeypatch):
    monkeypatch.setenv(
        "LOCAL_INFERENCE_INSPECTION_ENDPOINT", "http://processor:8080/v1/inspections"
    )
    gpu = LocalGPUDeployment(available=True, gpu_count=1)
    configured = gpu.configure(base, inference=True, tracking=False)
    assert (
        configured.model_api.inspection_endpoint
        == "http://processor:8080/v1/inspections"
    )
    assert configured.runtime.scene_mode == "active"
    assert configured.rich_labeling == base.rich_labeling
    assert gpu.configure(base, inference=False, tracking=False) == base


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://user:secret@host/v1/inspections",
        "http://host:0/v1/inspections",
        "http://host:65536/v1/inspections",
        "http://host/v1/inspections?key=secret",
        "http://host/v1/inspections#fragment",
        "file:///tmp/model",
        "http://bad host/",
    ],
)
def test_cpu_companion_endpoint_rejects_credentials_and_invalid_routes(endpoint):
    with pytest.raises(ValueError, match="inspection_endpoint"):
        ModelAPISettings(provider="local", inspection_endpoint=endpoint)


def test_cpu_companion_route_requires_a_local_provider():
    with pytest.raises(ValueError, match="inspection_endpoint"):
        ModelAPISettings(
            provider="openrouter", inspection_endpoint="http://host/v1/inspections"
        )


def test_switching_to_cloud_discards_the_old_cpu_route(base, monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")
    settings, _, _ = resolve_model_route(
        {
            "inference_service": {
                "provider": "local",
                "endpoint": "http://local/v1/chat/completions",
                "inspection_endpoint": "http://local/v1/inspections",
            },
        },
        {},
    )
    assert settings.provider == "openrouter"
    assert settings.inspection_endpoint == ""


def test_mapping_retains_an_explicit_local_cpu_route():
    settings = ModelAPISettings.from_mapping(
        {
            "provider": "local",
            "inspection_endpoint": "http://processor/v1/inspections",
        }
    )
    assert settings.inspection_endpoint == "http://processor/v1/inspections"


@pytest.mark.parametrize("protocol", ["compatible", "goodvision"])
def test_protocol_persists_and_survives_runtime_restart(
    tmp_path,
    monkeypatch,
    base,
    protocol,
):
    from goodvision_harness.deployment_settings import DeploymentSettingsStore
    from goodvision_harness.service import HarnessRuntime

    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("INFERENCE_SERVER_PROTOCOL", "compatible")
    config = replace(base, events=replace(base.events, directory=tmp_path / "events"))
    store = DeploymentSettingsStore(tmp_path / "deployment.json")
    runtime = HarnessRuntime(config, deployment_store=store)
    restarted = None
    try:
        saved = runtime.update_deployment_settings(
            {
                "inference_mode": "server",
                "inference_server_url": "http://analysis:8094",
                "inference_server_model": "vision",
                "inference_server_protocol": protocol,
            }
        )
        assert saved["inference_server_protocol"] == protocol
        assert store.load().inference_server_protocol == protocol
        assert runtime.config.model_api.inspection_endpoint == (
            "http://analysis:8094/v1/inspections" if protocol == "goodvision" else ""
        )
        monkeypatch.setenv(
            "INFERENCE_SERVER_PROTOCOL",
            "goodvision" if protocol == "compatible" else "compatible",
        )
        restarted = HarnessRuntime(config, deployment_store=store)
        assert (
            restarted.service_execution_settings()["inference_server_protocol"]
            == protocol
        )
        assert restarted.config.model_api == runtime.config.model_api
        assert restarted.service_execution_settings()["tracking_mode"] == "legacy"
        before = store.path.read_bytes()
        with pytest.raises(ValueError, match="inference_server_protocol"):
            runtime.update_deployment_settings({"inference_server_protocol": "invalid"})
        assert store.path.read_bytes() == before
    finally:
        runtime.stop()
        if restarted is not None:
            restarted.stop()
