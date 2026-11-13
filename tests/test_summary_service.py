"""Summary connections round-trip independently of cloud routes and secrets."""

import json
import os
import stat
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from test_video_connection import wait_for
from test_workflow_runtime import configure, graph
from test_workflow_runtime import rig as rig

from goodvision_harness import workflow_summary
from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]
URL = "/v1/deployment/summary-service"
CUSTOM = {
    "mode": "custom",
    "endpoint": "https://gateway.example/custom/prefix/chat/completions",
    "model": "openai/vision-model",
    "request_parameters": {"temperature": None, "max_completion_tokens": 1024},
}


def test_settings_api_persists_connection_and_write_only_key(rig):
    runtime, _, _, _ = rig
    baseline = runtime.config
    with TestClient(create_app(lambda: runtime)) as client:
        assert client.get(URL).json()["mode"] == "inherit"
        response = client.put(URL, json={**CUSTOM, "api_key": "summary-secret"})
        assert response.status_code == 200
        saved = response.json()
        assert saved["api_key_present"] is True
        assert saved["model"] == CUSTOM["model"]
        assert saved["endpoint"] == CUSTOM["endpoint"]
        assert "summary-secret" not in response.text
        assert "summary-secret" not in client.get(URL).text
        assert client.get(URL).headers["cache-control"] == "no-store"
        assert runtime.config is baseline
        runtime.update_deployment_settings({"autostart": False})
        assert client.get(URL).json() == saved
        path = runtime._deployment_store.path.with_name("summary-service.json")
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        # A worker/restarted runtime in this installation sees the same route.
        store = DeploymentSettingsStore(path.with_name("camera02-settings.json"))
        restored = HarnessRuntime(
            load_config(ROOT / "config.yaml"), deployment_store=store
        )
        try:
            assert restored.workflow_summary_service() == saved
            assert restored._workflow_summary_route()[1] == "summary-secret"
        finally:
            restored.stop()
        assert client.put(URL, json={"model": "another-model"}).json()[
            "api_key_present"
        ]
        assert not client.put(URL, json={"clear_api_key": True}).json()[
            "api_key_present"
        ]
        client.put(URL, json={"api_key": "second-secret"})
        assert not client.put(
            URL, json={"endpoint": "https://next.example/chat"}
        ).json()["api_key_present"]
        assert client.put(URL, json={"mode": "inherit"}).status_code == 200
        assert runtime._workflow_summary_configuration() is baseline


@pytest.mark.parametrize("key", ["summary-key", ""])
def test_custom_route_is_used_for_real_test_and_live_summary_without_cloud_key(
    rig,
    monkeypatch,
    key,
):
    runtime, _, _, clock = rig
    monkeypatch.setenv("LOCAL_INFERENCE_API_KEY", "unrelated-local-secret")
    runtime._aiclub_api_key = None
    runtime.update_workflow_summary_service({**CUSTOM, "api_key": key})
    baseline = runtime.config
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "zh": "绿色画面。",
                                    "en": "A green scene.",
                                }
                            )
                        },
                    }
                ]
            },
        )

    monkeypatch.setattr(
        workflow_summary,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    runtime.dry_run = False
    configure(runtime, graph("summary"))
    assert runtime.prepare_analysis()["prepared"]
    result = runtime.test_workflow(graph("summary"))
    assert result["results"][0]["result"]["summary"] == "A green scene."
    runtime.start_connected_analysis()
    worker = runtime._workflow_summaries
    try:
        clock.advance(worker, 1)
        wait_for(lambda: worker.latest() is not None)
        assert worker.latest()["summary"] == "A green scene."
    finally:
        runtime.stop_session()
    assert len(requests) == 2
    for request in requests:
        assert str(request.url) == CUSTOM["endpoint"]
        assert request.headers.get("authorization") == (
            f"Bearer {key}" if key else None
        )
        payload = json.loads(request.content)
        assert payload["model"] == CUSTOM["model"]
        assert payload["max_completion_tokens"] == 1024
        assert "temperature" not in payload and "max_tokens" not in payload
        assert "provider" not in payload and "reasoning" not in payload
        assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )
    assert runtime.config.inference == baseline.inference
    assert runtime.config.model_api == baseline.model_api


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "custom", "endpoint": ""},
        {"mode": []},
        {"model": "bad\nmodel"},
        {"endpoint": "http://user:password@host/chat"},
        {"endpoint": "http://host:99999/chat"},
        {"endpoint": "ftp://host/chat"},
        {"endpoint": "https://host/chat?api_key=secret"},
        {"request_parameters": {"messages": []}},
        {"request_parameters": {"max_tokens": -1}},
        {"request_parameters": {"max_tokens": 512, "max_completion_tokens": 512}},
        {"api_key": "bad\nkey"},
        {"api_key": "key", "clear_api_key": True},
        {"clear_api_key": "true"},
        {"unknown": 1},
    ],
)
def test_invalid_settings_have_no_disk_side_effects(rig, payload):
    runtime, _, _, _ = rig
    runtime.update_workflow_summary_service({**CUSTOM, "api_key": "summary-key"})
    before = runtime._deployment_store.load_summary_service()
    with TestClient(create_app(lambda: runtime)) as client:
        assert client.put(URL, json=payload).status_code == 400
    assert runtime._deployment_store.load_summary_service() == before


def test_saving_settings_is_blocked_during_analysis(rig):
    runtime, _, _, _ = rig
    configure(runtime, graph("summary"))
    runtime.start_connected_analysis()
    with TestClient(create_app(lambda: runtime)) as client:
        assert client.put(URL, json=CUSTOM).status_code == 409
    assert runtime.workflow_summary_service()["mode"] == "inherit"


def test_custom_summary_service_does_not_change_event_test_route(rig, monkeypatch):
    runtime, _, _, _ = rig
    runtime.update_workflow_summary_service(CUSTOM)
    seen = []
    from goodvision_harness import grounding

    original = grounding.build_selected_features_grounding_client

    def build(inference, settings, **options):
        seen.append((inference, settings))
        return original(inference, settings, **options)

    monkeypatch.setattr(grounding, "build_selected_features_grounding_client", build)
    result = runtime.test_workflow(graph("events", "summary"))
    assert {item["type"] for item in result["results"]} == {"events", "summary"}
    assert seen[0] == (runtime.config.inference, runtime.config.model_api)
    assert (
        runtime._workflow_summary_configuration().model_api.endpoint
        == CUSTOM["endpoint"]
    )
