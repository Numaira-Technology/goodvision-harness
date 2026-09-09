"""Only the explicit synthetic fixture simulates GPU health for protocol tests."""

import httpx
from fastapi.testclient import TestClient

from goodvision_harness.rich_labeling import RichLabelHTTPClient
from tools import paddle_delivery_smoke as smoke
from workers.paddle_rich_label import app as worker


def _worker_transport(application_client):
    def request(request):
        response = application_client.request(
            request.method, request.url.path,
            headers=dict(request.headers), content=request.content,
        )
        return httpx.Response(response.status_code, json=response.json())

    return httpx.MockTransport(request)


def test_isolated_synthetic_fixture_exercises_real_harness_gpu_gate(monkeypatch):
    monkeypatch.setenv("RICH_DEVICE", "cpu")
    models = smoke.SyntheticModels(worker)
    engine = worker.RichLabelEngine(models)
    assert models.gpu_ready is False
    with TestClient(worker.create_app(engine)) as application_client:
        with RichLabelHTTPClient(
            "http://isolated-synthetic-worker/v1/labels",
            transport=_worker_transport(application_client),
        ) as client:
            assert client.gpu_ready() is True
            first = client.infer(
                smoke.make_jpeg(), stream_session_id="synthetic-smoke",
                frame_sequence=1, timestamp_seconds=0.1,
            )
            assert len(first.objects) == 2
            assert engine._attribute_idle.wait(2.0)
            second = client.infer(
                smoke.make_jpeg(), stream_session_id="synthetic-smoke",
                frame_sequence=2, timestamp_seconds=0.2,
            )
            person = next(item for item in second.objects if item.label == "person")
            assert person.attributes["synthetic_protocol_test"] is True
    assert "GPU readiness is simulated" in smoke.NOTICE


def test_synthetic_readiness_does_not_enable_production_cpu_worker(monkeypatch):
    monkeypatch.setenv("RICH_DEVICE", "cpu")
    synthetic = smoke.SyntheticModels(worker)
    synthetic.ensure_detector_loaded()
    assert synthetic.gpu_ready is True

    with TestClient(worker.create_app()) as application_client:
        with RichLabelHTTPClient(
            "http://production-worker/v1/labels",
            transport=_worker_transport(application_client),
        ) as client:
            assert client.gpu_ready() is False
        response = application_client.get("/health/ready")
        assert response.json() == {"ready": False, "gpu_ready": False}
        assert response.status_code == 503
        denied = application_client.post("/v1/labels", content=b"unused")
        assert denied.status_code == 503
