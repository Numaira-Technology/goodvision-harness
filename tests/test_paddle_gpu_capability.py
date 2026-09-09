"""Worker readiness requires real GPU execution, not a configured device name."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from workers.paddle_rich_label.app import PaddleRichModels, create_app


def test_production_cpu_worker_stays_live_without_loading_or_running_models(
    monkeypatch,
):
    monkeypatch.setenv("RICH_DEVICE", "cpu")
    monkeypatch.setitem(sys.modules, "paddle", None)
    monkeypatch.setitem(sys.modules, "paddlex", None)
    application = create_app()
    with TestClient(application) as client:
        assert client.get("/health/live").json() == {"live": True}
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json() == {"ready": False, "gpu_ready": False}
        assert client.post("/v1/labels", content=b"unused").status_code == 503
    models = application.state.engine.models
    assert models._detector is None
    assert models._attribute is None


def _install_gpu_runtime(monkeypatch, failure):
    calls = []
    monkeypatch.setenv("RICH_DEVICE", "gpu:2" if failure == "wrong_index" else "gpu:0")

    class Tensor:
        place = SimpleNamespace(is_gpu_place=lambda: failure != "cpu_fallback")

        def __add__(self, value):
            calls.append("kernel")
            assert value == 1.0
            if failure == "kernel":
                raise RuntimeError("private GPU driver failure")
            return self

        def numpy(self):
            return np.array([2.0])

    def to_tensor(values, *, place):
        calls.append("tensor")
        assert values == [1.0]
        assert place == ("cuda", 0)
        return Tensor()

    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(
        device=SimpleNamespace(
            is_compiled_with_cuda=lambda: failure != "not_compiled",
            cuda=SimpleNamespace(
                device_count=lambda: 0 if failure == "no_device" else 1,
            ),
        ),
        CUDAPlace=lambda index: ("cuda", index),
        to_tensor=to_tensor,
    ))

    def detect(self, image):
        calls.append("detector")
        assert image.shape == (640, 640, 3)
        self._detector = object()
        if failure == "detector":
            raise RuntimeError("private detector failure")
        return ()

    def attributes(self, images):
        calls.append("attributes")
        assert len(images) == 1
        self._attribute = object()
        if failure == "attributes":
            raise RuntimeError("private attribute failure")
        return ({},) if failure == "empty_attributes" else ({"orientation": "front"},)

    monkeypatch.setattr(PaddleRichModels, "detect", detect)
    monkeypatch.setattr(PaddleRichModels, "attributes", attributes)
    return calls


@pytest.mark.parametrize(
    "failure",
    ["not_compiled", "no_device", "wrong_index", "kernel", "cpu_fallback",
     "detector", "attributes", "empty_attributes"],
)
def test_gpu_configuration_does_not_advertise_unusable_acceleration(
    monkeypatch, failure,
):
    calls = _install_gpu_runtime(monkeypatch, failure)
    with TestClient(create_app()) as client:
        assert client.get("/health/live").status_code == 200
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json() == {"ready": False, "gpu_ready": False}
        assert client.post("/v1/labels", content=b"unused").status_code == 503
        assert "private" not in response.text
    if failure in {
        "not_compiled", "no_device", "wrong_index", "kernel", "cpu_fallback",
    }:
        assert "detector" not in calls
        assert "attributes" not in calls


def test_worker_advertises_gpu_only_after_kernel_and_both_models_execute(monkeypatch):
    calls = _install_gpu_runtime(monkeypatch, None)
    application = create_app()
    assert application.state.engine.gpu_ready is False
    with TestClient(application) as client:
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"ready": True, "gpu_ready": True}
        assert calls == ["tensor", "kernel", "detector", "attributes"]
        # Health polling reads the completed probe; it cannot start inference.
        client.get("/health/ready")
        assert calls == ["tensor", "kernel", "detector", "attributes"]
