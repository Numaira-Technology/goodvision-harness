"""Delegated inspection keeps original evidence and application validation."""

import base64
import json
from dataclasses import replace

import httpx
import pytest
from test_scene_model import _job, _response

from goodvision_harness import scene_model
from goodvision_harness.model_api import ModelAPIError, ModelAPISettings
from goodvision_harness.scene_model import TemporalVLMInspector
from goodvision_harness.vlm_compact_model import (
    CompactGroundingInspector,
    EventJSONInspector,
)
from goodvision_harness.vlm_independent_model import IndependentVLMInspector
from goodvision_harness.vlm_lean_grounding import LeanGroundingInspector


@pytest.mark.parametrize(
    ("inspector_class", "kind", "content"),
    [
        (TemporalVLMInspector, "temporal", None),
        (
            IndependentVLMInspector,
            "full",
            {
                "subjects": [{"subject_id": "p1", "bbox": [0.1, 0.1, 0.6, 0.9]}],
                "verdicts": [
                    {
                        "event_label": "smoking",
                        "subject_ids": ["p1"],
                        "detected": True,
                        "confidence": 0.9,
                    }
                ],
            },
        ),
        (EventJSONInspector, "events", {"smoking": False}),
        (CompactGroundingInspector, "compact", {"no_events": 0.9}),
        (LeanGroundingInspector, "lean-v1", {
            "v": 1, "s": [],
            "e": {"smoking": [{"p": [], "d": False, "c": 0.9}]},
        }),
    ],
)
def test_all_lanes_send_originals_without_local_image_preparation(
    monkeypatch,
    inspector_class,
    kind,
    content,
):
    job = _job()
    seen = []
    completion = _response(None if content is None else json.dumps(content))

    def capture(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "version": 1,
                "request_id": job.request.request_id,
                "completion": completion,
                "timing": {
                    "request_prepare_seconds": 0.02,
                    "verdict_parse_seconds": 0.001,
                    "private": "secret",
                },
            },
        )

    monkeypatch.setattr(
        scene_model,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(capture),
            headers={"Authorization": "cloud-secret", "X-Private": "secret"},
            cookies={"cloud": "secret"},
            params={"cloud_key": "secret"},
            auth=("cloud", "secret"),
        ),
    )
    settings = ModelAPISettings(
        provider="local",
        endpoint="http://model/v1/chat/completions",
        inspection_endpoint="http://processor/v1/inspections",
    )
    inspector = inspector_class(settings, model="local-vision", api_key="local-key")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("application must not prepare images in offload mode")

    monkeypatch.setattr(inspector, "build_messages", forbidden)
    try:
        verdicts = inspector.inspect(job)
        assert verdicts[0].event_label == "smoking"
        assert inspector.stats()["total_tokens"] == 70
        timing = inspector.last_inspection_timing
        assert timing["server_request_prepare_seconds"] == 0.02
        assert "private" not in inspector.last_inspection_timing
    finally:
        inspector.close()
    request = seen[0]
    assert str(request.url) == settings.inspection_endpoint
    assert request.headers["authorization"] == "Bearer local-key"
    assert "x-private" not in request.headers and "cookie" not in request.headers
    payload = json.loads(request.content)
    assert payload["kind"] == kind
    assert "messages" not in payload
    assert "source_id" not in payload["job"]
    assert "private-camera" not in request.content.decode()
    assert "cloud-secret" not in str(request.headers)
    for frame, original in zip(payload["job"]["frames"], job.frames, strict=True):
        assert base64.b64decode(frame["jpeg_base64"]) == original.jpeg_bytes
        assert frame["timestamp_seconds"] == original.timestamp_seconds
        assert frame["frame_sequence"] == original.frame_sequence


@pytest.mark.parametrize("failure", ["wrong_job", "invalid_verdict", "http_error"])
def test_invalid_or_unreachable_companion_never_falls_back_to_another_route(
    monkeypatch,
    failure,
):
    seen = []

    def capture(request):
        seen.append(request)
        if failure == "http_error":
            return httpx.Response(503, text="private upstream details")
        response = _response()
        if failure == "invalid_verdict":
            response = _response(
                json.dumps(
                    [
                        {
                            "event_label": "weapon",
                            "subject_ids": ["person-1"],
                            "detected": True,
                            "confidence": 0.9,
                        }
                    ]
                )
            )
        return httpx.Response(
            200,
            json={
                "version": 1,
                "request_id": "wrong" if failure == "wrong_job" else "request",
                "completion": response,
            },
        )

    monkeypatch.setattr(
        scene_model,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(capture)),
    )
    inspector = TemporalVLMInspector(
        ModelAPISettings(
            provider="local",
            inspection_endpoint="http://processor/v1/inspections",
        ),
        model="local-vision",
    )
    try:
        with pytest.raises(ModelAPIError) as result:
            inspector.inspect(_job())
        assert result.value.code == (
            "http_error" if failure == "http_error" else "invalid_response"
        )
        assert inspector.stats()["failed_requests"] == 1
        assert len(seen) == 1
        assert str(seen[0].url) == inspector.settings.inspection_endpoint
        assert "private" not in str(result.value)
    finally:
        inspector.close()


def test_companion_deadline_cancels_http_and_releases_the_pool(monkeypatch):
    import asyncio

    cancelled = []

    async def slow(_request):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(
        scene_model,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(slow)),
    )
    inspector = TemporalVLMInspector(
        ModelAPISettings(
            provider="local",
            timeout_seconds=0.02,
            inspection_endpoint="http://processor/v1/inspections",
        ),
        model="local-vision",
    )
    try:
        with pytest.raises(ModelAPIError, match="deadline") as result:
            inspector.inspect(replace(_job(), grounding_required=True))
        assert result.value.code == "timeout"
        assert cancelled == [True]
    finally:
        inspector.close()


def test_synchronous_temporal_worker_delegates_pixel_validation(monkeypatch):
    from types import SimpleNamespace

    from goodvision_harness import scene_inspection
    from goodvision_harness.scene_runtime import TemporalVerdict

    def forbidden(_jpeg):
        raise AssertionError("pixel validation belongs to the companion")

    monkeypatch.setattr(scene_inspection, "_read_image", forbidden)
    completed = []

    def complete(request_id, verdicts, **kwargs):
        completed.append((request_id, verdicts, kwargs))
        return True

    runtime = SimpleNamespace(
        take_inspections=lambda: (_job(),), complete_inspection=complete
    )
    inspector = SimpleNamespace(
        settings=ModelAPISettings(
            provider="local", inspection_endpoint="http://processor/v1/inspections"
        ),
        inspect=lambda _: (TemporalVerdict("smoking", ("person-1",), False, 0.9),),
    )
    counts = scene_inspection.run_pending_inspections(runtime, inspector)
    assert counts.succeeded_requests == 1
    assert completed[0][2]["completion_counts"] == {}
