"""Actionable inference errors survive transport, composition and status safely."""

import io
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from test_branch_result_delivery import observation, runtime_without_workers
from test_organizer import _apply, _branches

from goodvision_harness.config import load_config
from goodvision_harness.grounding import (
    AIClubGroundingClient,
    AIClubMultiGroundingClient,
    SelectedFeaturesGroundingClient,
)
from goodvision_harness.inference import QwenWindowClient
from goodvision_harness.model_api import (
    ModelAPIError,
    ModelAPISettings,
    PooledModelAPIClient,
    inference_error_detail,
)
from goodvision_harness.organizer import ResultOrganizer
from goodvision_harness.service import create_app
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


def window(index=1):
    image = io.BytesIO()
    Image.new("RGB", (24, 24)).save(image, format="JPEG")
    return FrameWindow(
        index, (SampledFrame(index, index, float(index), image.getvalue(), 24, 24),)
    )


@pytest.mark.parametrize("client_kind", ["binary", "single", "multi", "composite"])
def test_actual_http_failure_reaches_organizer_without_provider_body(client_kind):
    transport = PooledModelAPIClient(
        ModelAPISettings(provider="local"), api_key="TEST_SECRET"
    )
    transport._http.close()
    transport._http = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                401, text="PRIVATE_BODY TEST_SECRET https://private-host"
            )
        )
    )
    settings = replace(load_config().inference, event_label="smoking")
    try:
        if client_kind == "binary":
            client = QwenWindowClient(settings, transport)
        elif client_kind == "single":
            client = AIClubGroundingClient(settings, transport)
        else:
            client = AIClubMultiGroundingClient(
                settings, transport, event_labels=("smoking",)
            )
            if client_kind == "composite":
                client = SelectedFeaturesGroundingClient(
                    client,
                    None,
                    safety_event_labels=("smoking",),
                    include_stranger=False,
                )
        response = client.classify_window(window())
        organizer = ResultOrganizer("smoking")
        _, state = _apply(organizer, response)
        branch = "generic" if client_kind == "binary" else "safety"
        detail = state["branch_error_details"][branch]
        assert detail["code"] == "http_error"
        assert detail["http_status"] == 401
        assert "authentication" in detail["message"]
        assert "configured service" in detail["message"]
        assert all(
            private not in json.dumps(state)
            for private in ("PRIVATE_BODY", "TEST_SECRET", "private-host")
        )
    finally:
        transport.close()


@pytest.mark.parametrize(
    "status,word",
    [
        (400, "Invalid"),
        (401, "authentication"),
        (402, "credit"),
        (403, "denied"),
        (404, "not found"),
        (429, "rate limit"),
        (503, "unavailable"),
    ],
)
def test_http_status_has_an_actionable_safe_explanation(status, word):
    detail = inference_error_detail(
        ModelAPIError("SECRET raw body", code="http_error", status_code=status)
    )
    assert word in detail["message"]
    assert f"HTTP {status}" in detail["message"]
    assert "SECRET" not in json.dumps(detail)


def test_error_details_follow_branch_health_ordering_and_recovery():
    organizer = ResultOrganizer("smoking,stranger")
    response = replace(
        _branches(3, errors=(("safety", "http_error"),)),
        branch_error_details={
            "safety": {
                "code": "http_error",
                "http_status": 429,
                "message": "SECRET",
                "body": "SECRET",
            },
        },
    )
    _, state = _apply(organizer, response)
    assert state["branch_error_details"]["safety"]["http_status"] == 429
    assert "SECRET" not in json.dumps(state)
    # Other branches and late successes cannot hide this request failure.
    _, state = _apply(organizer, _branches(4, completed=("stranger",)))
    assert state["branch_error_details"]["safety"]["http_status"] == 429
    _, state = _apply(organizer, _branches(2, completed=("safety",)))
    assert state["branch_error_details"]["safety"]["http_status"] == 429
    _, state = _apply(organizer, _branches(5, completed=("safety",)))
    assert state["branch_error_details"] == {}
    _, state = _apply(organizer, response)
    assert state["branch_error_details"] == {}


@pytest.mark.parametrize(
    "path", ["latest", "overall/latest", "smoking/latest", "fall/latest"]
)
def test_polling_opt_in_returns_empty_without_changing_existing_clients(path):
    runtime, _ = runtime_without_workers()
    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        url = f"/v1/results/{path}"
        assert client.get(url).status_code == 404
        empty = client.get(url, params={"allow_empty": True})
        assert empty.status_code == 204
        assert empty.content == b""
        assert empty.headers["cache-control"] == "no-store"
        # Recovery uses the same URL and returns real cached output, not a stub.
        runtime.results.publish(observation("smoking", 1).as_dict())
        found = client.get(url, params={"allow_empty": True})
        if path == "fall/latest":
            assert found.status_code == 204
        else:
            assert found.status_code == 200
            assert found.json()["message_id"] == "smoking-1"


def test_summary_preserves_http_status_and_recovers_without_private_body(monkeypatch):
    from test_workflow_summary import jpeg_bytes, make_manager, snapshot

    from goodvision_harness import workflow_summary

    responses = iter(
        [
            httpx.Response(401, text="PRIVATE_BODY SECRET"),
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "A synthetic test scene."}}]},
            ),
        ]
    )
    monkeypatch.setattr(
        workflow_summary,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: next(responses)),
        ),
    )
    manager = make_manager(
        inference=None,
        model_settings=ModelAPISettings(provider="local"),
        model="fixture",
    )
    failed = manager.run_once("summary-a", snapshot(jpeg=jpeg_bytes()))
    assert failed["error"]["http_status"] == 401
    assert "authentication" in failed["error"]["message"]
    assert "SECRET" not in json.dumps(failed)
    recovered = manager.run_once("summary-a", snapshot(2, jpeg=jpeg_bytes()))
    assert recovered["error"] is None
    assert recovered["status"] == "completed"


def test_camera_scoped_empty_results_keep_204_and_camera_headers(monkeypatch):
    monkeypatch.setenv("HARNESS_FLEET_CAPACITY", "1")
    runtime, _ = runtime_without_workers()
    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        url = "/v1/fleet/cameras/camera01/results/smoking/latest"
        response = client.get(url, params={"allow_empty": True})
        assert response.status_code == 204
        assert response.content == b""
        assert response.headers["x-camera-id"] == "camera01"
        runtime.results.publish(observation("smoking", 1).as_dict())
        response = client.get(url, params={"allow_empty": True})
        assert response.status_code == 200
        assert response.json()["camera_id"] == "camera01"
