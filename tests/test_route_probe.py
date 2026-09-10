from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock

import httpx
import pytest
from fastapi.testclient import TestClient

import goodvision_harness.config as config_module
import goodvision_harness.service as service_module
from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.model_api import (
    ModelAPIClient, ModelAPIError, ModelAPISettings, PooledModelAPIClient,
)
from goodvision_harness.privacy import contains_private_implementation_detail
from goodvision_harness.route_probe import (
    MAX_PROBE_CONCURRENCY, ROUTE_CATALOG, ROUTES, ROUTES_BY_ID, choose_route,
    finish_result, probe_sample, probe_window,
)
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]


def results(**latencies):
    rows = []
    for route_id, seconds in latencies.items():
        row = {"id": route_id, "samples": [
            {"ok": True, "seconds": seconds, "error": None} for _ in range(3)
        ]}
        finish_result(row)
        rows.append(row)
    return rows


@pytest.mark.parametrize(("latencies", "selected", "reason"), [
    ({"A": 2.4, "B": 2.1, "C": 1.0}, "B", "preferred_below_threshold"),
    ({"A": 2.49999, "C": 1.0}, "A", "preferred_below_threshold"),
    ({"A": 2.5, "C": 1.0}, "C", "fastest_available"),
    ({"A": 3.0, "B": 2.6, "C": 1.2, "E": 1.1}, "E", "fastest_available"),
    ({"D": 2.0, "C": 1.0}, "D", "preferred_below_threshold"),
    ({"F": 2.4, "G": 2.0, "H": 2.1, "I": 0.7}, "G", "preferred_below_threshold"),
    ({"H": 2.5, "AA": 1.0, "AR": 1.5}, "AA", "fastest_available"),
    ({"J": 2.0, "AE": 1.8, "AH": 1.7, "I": 0.7}, "I", "fastest_available"),
    ({"A": 4.0, "B": 3.0}, "B", "fastest_available"),
    ({}, None, "no_available_route"),
])
def test_selection_rule(latencies, selected, reason):
    assert choose_route(results(**latencies)) == (selected, reason)


def test_partial_success_and_non_finite_latency_cannot_win():
    rows = results(A=0.1, B=1.0, C=2.0)
    rows[0]["samples"][1]["ok"] = False
    finish_result(rows[0])
    assert rows[0]["median_seconds"] is None
    rows[1]["median_seconds"] = float("nan")
    assert choose_route(rows)[0] == "C"


def completion():
    return {"choices": [{"message": {"content": json.dumps({"events": [
        {"event_label": label, "detected": False, "confidence": 0.9,
         "description": "未发现异常事件。 / No visible event.", "objects": []}
        for label in ("fight", "smoking", "fall", "weapon")
    ]})}}]}


class FakePool:
    assistant_text = staticmethod(ModelAPIClient.assistant_text)

    def __init__(self, settings, *, api_key=None):
        self.settings = settings
        self.calls = []
        self.closed = False

    def chat_completion(self, **payload):
        self.calls.append(payload)
        return completion()

    def close(self):
        self.closed = True


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", "customer-standard")
    monkeypatch.setenv("INFERENCE_API_KEY", "secret-for-test-only")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "false")
    monkeypatch.setenv("RICH_LABEL_ENABLED", "false")
    monkeypatch.delenv("RESULT_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(service_module, "PooledModelAPIClient", FakePool)
    config = load_config(ROOT / "config.delivery.yaml")
    store = DeploymentSettingsStore(tmp_path / "deployment-settings.json")
    value = HarnessRuntime(config, deployment_store=store)
    yield value
    value.stop()


def wait_for_probe(runtime):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        data = runtime.route_probe_status()
        if data["state"] != "running":
            return data
        time.sleep(0.01)
    pytest.fail("probe did not complete")


def fast_samples(monkeypatch):
    def sample(_transport, _settings, route, _index):
        return {"ok": True, "error": None,
                "seconds": {"D": 2.0, "C": 1.0}.get(route.id, 5.0)}
    monkeypatch.setattr(service_module, "probe_sample", sample)


def test_console_selection_is_private_persisted_and_used_by_live_requests(
    runtime, monkeypatch,
):
    fast_samples(monkeypatch)
    with TestClient(create_app(lambda: runtime)) as client:
        assert client.post("/v1/deployment/routes/check").status_code == 202
        data = wait_for_probe(runtime)
        assert data["state"] == "complete"
        assert data["selected_id"] == "B"
        assert data["selected_label"] == "线路 B"
        assert data["selection_persisted"] is True
        public = client.get("/v1/deployment/routes").json()
        assert len(public["results"]) == 8
        assert [row["id"] for row in public["results"]] == list("ABCDEFGH")
        assert [row["label"] for row in public["results"]] == [
            f"线路 {letter}" for letter in "ABCDEFGH"
        ]
        assert public["results"][0]["median_seconds"] == 1.0
        assert public["results"][1]["median_seconds"] == 2.0
        assert runtime._route_probe["results"][0]["id"] == "C"
        assert not contains_private_implementation_detail(public)
        assert "secret-for-test-only" not in json.dumps(public)
        for route in ROUTE_CATALOG:
            assert route.provider.split("/")[0] not in json.dumps(public).lower()
        assert runtime._route_transport.settings == runtime.config.model_api
        assert runtime._deployment_store.load_model_route() == "D"
        runtime.start_camera(features=("fight", "smoking", "fall", "weapon"))
        frame = probe_window().frames[0]
        assert runtime.push_camera_frame(frame.jpeg_bytes, width=320, height=180)
        assert runtime.results.records_after(0, timeout_seconds=5)
        payload = runtime._route_transport.calls[-1]
        assert payload["model"] == "qwen/qwen3.5-35b-a3b:nitro"
        assert payload["provider"] == {
            "sort": "throughput", "allow_fallbacks": True,
            "ignore": ["coreweave", "novita"],
        }
        assert client.post("/v1/deployment/routes/check").status_code == 409
    restored = HarnessRuntime(runtime.config, deployment_store=runtime._deployment_store)
    try:
        assert restored.route_probe_status()["selected_id"] == "B"
        assert restored._route_pool_enabled is True
    finally:
        restored.stop()


def test_check_blocks_duplicate_start_and_credential_change(runtime, monkeypatch):
    entered, release = Event(), Event()

    def blocked(*_args):
        entered.set()
        assert release.wait(5)
        return {"ok": False, "error": "request_failed", "seconds": 0.01}

    monkeypatch.setattr(service_module, "probe_sample", blocked)
    try:
        runtime.start_route_probe()
        assert entered.wait(2)
        assert runtime.route_probe_status()["state"] == "running"
        with pytest.raises(RuntimeError, match="already running"):
            runtime.start_route_probe()
        with pytest.raises(RuntimeError, match="route check"):
            runtime.start_camera(features=("smoking",))
        with pytest.raises(RuntimeError, match="route check"):
            runtime.update_deployment_settings({"api_key": "replacement"})
    finally:
        release.set()
    data = wait_for_probe(runtime)
    assert data["state"] == "failed"
    assert data["selected_id"] is None
    assert runtime._selected_route_id == "A"
    assert runtime._deployment_store.load_model_route() is None


def test_failed_save_retains_previous_selection(runtime, monkeypatch):
    fast_samples(monkeypatch)

    def failed(_route):
        raise OSError("internal vendor secret")

    monkeypatch.setattr(runtime._deployment_store, "save_model_route", failed)
    runtime.start_route_probe()
    data = wait_for_probe(runtime)
    assert data["state"] == "failed"
    assert data["selected_id"] is None
    assert runtime._selected_route_id == "A"
    assert "internal vendor secret" not in json.dumps(data)


def test_invalid_saved_route_is_not_applied(runtime):
    path = runtime._deployment_store.path.with_name("selected-route.json")
    path.write_text(json.dumps({"schema_version": 1, "route_id": "arbitrary/model"}))
    restored = HarnessRuntime(runtime.config, deployment_store=runtime._deployment_store)
    try:
        assert restored.route_probe_status()["selected_id"] is None
        assert restored._selected_route_id == "A"
        assert restored.route_probe_status()["error"] == "saved_route_unavailable"
    finally:
        restored.stop()


@pytest.mark.parametrize(("stored_id", "public_id"), [
    ("C", "A"), ("D", "B"), ("AA", "C"), ("AE", "D"),
    ("G", "E"), ("K", "F"), ("B", "G"), ("F", "H"),
])
def test_saved_route_keeps_number_and_base_model_while_enabling_nitro(
    runtime, stored_id, public_id,
):
    runtime._deployment_store.save_model_route(stored_id)
    saved_path = runtime._deployment_store.path.with_name("selected-route.json")
    saved = saved_path.read_bytes()
    restored = HarnessRuntime(runtime.config, deployment_store=runtime._deployment_store)
    try:
        status = restored.route_probe_status()
        assert status["selected_id"] == public_id
        assert status["selected_label"] == f"线路 {public_id}"
        assert status["selection_persisted"] is True
        route = ROUTES_BY_ID[stored_id]
        assert restored.config.inference.model == f"{route.model}:nitro"
        assert restored.config.inference.request_parameters["provider"] == {
            "sort": "throughput", "allow_fallbacks": True,
            "ignore": ["coreweave", "novita"],
        }
        assert saved_path.read_bytes() == saved
    finally:
        restored.stop()


@pytest.mark.parametrize("route_id", ["H", "R", "L", "S", "AB", "AG", "AK"])
def test_removed_provider_routes_cannot_be_selected_saved_restored_or_called(
    runtime, route_id,
):
    assert route_id not in ROUTES_BY_ID
    assert route_id not in {route.id for route in ROUTES}
    assert choose_route(results(**{route_id: 0.1, "C": 1.0}))[0] == "C"
    with pytest.raises(ValueError):
        runtime._deployment_store.save_model_route(route_id)
    path = runtime._deployment_store.path.with_name("selected-route.json")
    path.write_text(json.dumps({"schema_version": 1, "route_id": route_id}))
    restored = HarnessRuntime(runtime.config, deployment_store=runtime._deployment_store)
    try:
        assert restored.route_probe_status()["selected_id"] is None
        assert restored._selected_route_id == "A"
        assert restored.route_probe_status()["error"] == "saved_route_unavailable"
        assert restored.config.inference.request_parameters["provider"] == {
            "sort": "throughput", "allow_fallbacks": True,
            "ignore": ["coreweave", "novita"],
        }
    finally:
        restored.stop()
    route = next(route for route in ROUTE_CATALOG if route.id == route_id)
    pool = FakePool(runtime.config.model_api)
    with pytest.raises(ValueError, match="route is unavailable"):
        probe_sample(pool, runtime.config.inference, route, 1)
    assert not pool.calls


def test_large_catalog_limits_requests_and_completes_every_route(runtime, monkeypatch):
    lock, all_workers_started, release = Lock(), Event(), Event()
    active = peak = 0
    calls = {route.id: [] for route in ROUTES}

    def sample(_transport, _settings, route, trial):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            calls[route.id].append(trial)
            if active == MAX_PROBE_CONCURRENCY:
                all_workers_started.set()
        try:
            assert release.wait(5)
            return {"ok": True, "error": None, "seconds": 3.0}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(service_module, "probe_sample", sample)
    try:
        runtime.start_route_probe()
        assert all_workers_started.wait(2)
        data = runtime.route_probe_status()
        assert sum(row["state"] == "pending" for row in data["results"]) > 0
    finally:
        release.set()
    data = wait_for_probe(runtime)
    assert data["state"] == "complete"
    assert peak == MAX_PROBE_CONCURRENCY
    assert len(data["results"]) == len(ROUTES)
    assert all(indices == [0, 1, 2, 3] for indices in calls.values())
    public = json.dumps(data).lower()
    for route in ROUTES:
        assert route.model not in public
        assert route.provider.split("/")[0] not in public


def test_probe_uses_live_contract_and_rejects_incomplete_results(runtime):
    pool = FakePool(runtime.config.model_api)
    settings = replace(runtime.config.inference, request_parameters={
        "max_tokens": 1200, "models": ["bad/fallback"], "stream": True,
        "provider": {"only": ["alibaba"], "order": ["alibaba"],
                     "sort": "price", "allow_fallbacks": False},
        "reasoning": {"effort": "high"},
    })
    sample = probe_sample(pool, settings, ROUTES_BY_ID["C"], 1)
    assert sample["ok"]
    payload = pool.calls[0]
    assert payload["model"] == "qwen/qwen3-vl-8b-instruct:nitro"
    assert payload["provider"] == {
        "sort": "throughput", "allow_fallbacks": True,
        "ignore": ["coreweave", "novita"],
    }
    assert not {"models", "stream", "reasoning"}.intersection(payload)
    assert payload["messages"][0]["content"].startswith(settings.system_prompt)
    pool.chat_completion = lambda **_: {"choices": [{"message": {"content": "{}"}}]}
    assert probe_sample(pool, settings, ROUTES_BY_ID["C"], 1)["error"] == "invalid_result"


def test_pooled_transport_reuses_client_and_never_echoes_vendor_errors():
    pool = PooledModelAPIClient(
        ModelAPISettings(provider="openrouter", max_retries=0), api_key="secret",
    )
    pool._http.close()
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) > 1:
            return httpx.Response(401, json={"error": "secret upstream detail"})
        return httpx.Response(200, json=completion())

    pool._http = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        assert pool.chat_completion(model="test", messages=[{"role": "user", "content": "test"}])
        with pytest.raises(ModelAPIError, match="HTTP 401") as exc:
            pool.chat_completion(model="test", messages=[{"role": "user", "content": "test"}])
        assert "secret" not in str(exc.value)
        assert requests[0].headers["Authorization"] == "Bearer secret"
    finally:
        pool.close()
