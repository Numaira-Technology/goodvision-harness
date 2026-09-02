from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from workers.paddle_rich_label import gpu_smoke
from workers.paddle_rich_label.app import PaddleRichModels, RichLabelEngine, create_app


@pytest.fixture(autouse=True)
def _clean_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RICH_DEVICE",
        "RICH_DETECTOR_PROFILE",
        "RICH_ATTRIBUTE_PROFILE",
        "RICH_DETECTOR_MODEL",
        "RICH_ATTRIBUTE_MODEL",
        "RICH_DETECTOR_MODEL_DIR",
        "RICH_ATTRIBUTE_MODEL_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_profiles_preserve_the_existing_cpu_models() -> None:
    models = PaddleRichModels()

    assert models.device == "cpu"
    assert models.detector_profile == "balanced"
    assert models.detector_name == "PicoDet-S"
    assert models.attribute_profile == "standard"
    assert models.attribute_name == "PP-LCNet_x1_0_pedestrian_attribute"


@pytest.mark.parametrize(
    ("profile", "expected"),
    [("accurate", "PP-YOLOE_plus-L"), ("balanced", "PicoDet-S")],
)
def test_public_profiles_take_priority_over_legacy_model_names(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    expected: str,
) -> None:
    monkeypatch.setenv("RICH_DETECTOR_PROFILE", f" {profile.upper()} ")
    monkeypatch.setenv("RICH_ATTRIBUTE_PROFILE", "standard")
    monkeypatch.setenv("RICH_DETECTOR_MODEL", "legacy-private-detector")
    monkeypatch.setenv("RICH_ATTRIBUTE_MODEL", "legacy-private-attributes")

    models = PaddleRichModels()

    assert models.detector_name == expected
    assert models.detector_profile == profile
    assert models.attribute_name == "PP-LCNet_x1_0_pedestrian_attribute"
    assert models.attribute_profile == "standard"


def test_legacy_gpu_image_default_still_selects_the_accurate_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RICH_DEVICE", "gpu:0")
    monkeypatch.setenv("RICH_DETECTOR_MODEL", "PP-YOLOE_plus-L")
    monkeypatch.setenv("RICH_DETECTOR_PROFILE", " ")

    models = PaddleRichModels()

    assert models.detector_name == "PP-YOLOE_plus-L"
    assert models.detector_profile == "accurate"
    assert models.device == "gpu:0"


@pytest.mark.parametrize("use_profiles", [False, True])
def test_model_directories_remain_compatible_with_both_configuration_forms(
    monkeypatch: pytest.MonkeyPatch,
    use_profiles: bool,
) -> None:
    calls: list[dict[str, Any]] = []

    def create_model(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules, "paddlex", SimpleNamespace(create_model=create_model)
    )
    monkeypatch.setenv("RICH_DETECTOR_MODEL", "private-detector")
    monkeypatch.setenv("RICH_ATTRIBUTE_MODEL", "private-attributes")
    monkeypatch.setenv("RICH_DETECTOR_MODEL_DIR", "/models/local-detector")
    monkeypatch.setenv("RICH_ATTRIBUTE_MODEL_DIR", "/models/local-attributes")
    if use_profiles:
        monkeypatch.setenv("RICH_DETECTOR_PROFILE", "accurate")
        monkeypatch.setenv("RICH_ATTRIBUTE_PROFILE", "standard")

    models = PaddleRichModels()
    models.ensure_detector_loaded()
    models.ensure_attribute_loaded()

    assert calls == [
        {
            "model_name": "PP-YOLOE_plus-L" if use_profiles else "private-detector",
            "model_dir": "/models/local-detector",
            "device": "cpu",
        },
        {
            "model_name": (
                "PP-LCNet_x1_0_pedestrian_attribute"
                if use_profiles else "private-attributes"
            ),
            "model_dir": "/models/local-attributes",
            "device": "cpu",
        },
    ]
    if not use_profiles:
        assert models.detector_profile == models.attribute_profile == "custom"


@pytest.mark.parametrize(
    ("variable", "message"),
    [
        ("RICH_DETECTOR_PROFILE", "unsupported object detection profile"),
        ("RICH_ATTRIBUTE_PROFILE", "unsupported pedestrian attribute profile"),
    ],
)
def test_unknown_profile_fails_neutrally_without_falling_back(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    message: str,
) -> None:
    monkeypatch.setenv(variable, "Paddle-private-unknown-profile")
    monkeypatch.setenv("RICH_DETECTOR_MODEL", "PicoDet-S")
    monkeypatch.setenv("RICH_ATTRIBUTE_MODEL", "PP-LCNet_x1_0_pedestrian_attribute")

    with pytest.raises(ValueError) as error:
        PaddleRichModels()

    assert str(error.value) == message


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("ensure_detector_loaded", "object detection engine is unavailable"),
        ("ensure_attribute_loaded", "pedestrian attribute engine is unavailable"),
    ],
)
def test_missing_backend_reports_neutral_initialization_error(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    message: str,
) -> None:
    monkeypatch.setitem(sys.modules, "paddlex", None)
    models = PaddleRichModels()

    with pytest.raises(RuntimeError) as error:
        getattr(models, method)()

    assert str(error.value) == message
    assert error.value.__suppress_context__


class _Models:
    ready = True

    def detect(self, _image: Any) -> tuple[Any, ...]:
        return ()

    def attributes(self, _crops: Any) -> tuple[dict[str, str], ...]:
        return ({"estimated_age_range": "18-60"},)


def test_worker_public_metadata_and_health_are_neutral() -> None:
    with TestClient(create_app(RichLabelEngine(_Models()))) as client:
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert schema.json()["info"]["title"] == (
            "Goodvision Object and Attribute Engine"
        )
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/health/live").json() == {"live": True}
        assert client.get("/health/ready").json() == {"ready": True}
        assert "paddle" not in schema.text.lower()


def test_worker_inference_error_does_not_log_backend_name_or_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingModels(_Models):
        def detect(self, _image: Any) -> tuple[Any, ...]:
            raise RuntimeError("PaddleX private-model backend failure")

    _, encoded = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    with TestClient(create_app(RichLabelEngine(_FailingModels()))) as client:
        response = client.post(
            "/v1/labels",
            content=encoded.tobytes(),
            headers={
                "Content-Type": "image/jpeg",
                "X-Goodvision-Stream-Session-Id": "test-session",
                "X-Goodvision-Frame-Sequence": "1",
                "X-Goodvision-Timestamp-Seconds": "0.1",
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "inference unavailable"}
    assert "paddle" not in caplog.text.lower()
    assert "private-model" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_worker_warmup_failure_uses_a_neutral_error_boundary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingModels(_Models):
        def ensure_detector_loaded(self) -> None:
            raise RuntimeError("PaddleX private-model initialization failure")

    with pytest.raises(RuntimeError) as error:
        with TestClient(create_app(RichLabelEngine(_FailingModels()))):
            pass

    assert str(error.value) == "object and attribute engine initialization failed"
    assert error.value.__suppress_context__
    assert "paddle" not in caplog.text.lower()
    assert all(record.exc_info is None for record in caplog.records)


def _install_smoke_backends(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: str | None = None,
) -> list[str]:
    calls: list[str] = []

    def record(name: str, result: Any = None) -> Any:
        calls.append(name)
        if failure == name:
            raise RuntimeError("Paddle NVIDIA private-backend failure")
        return result

    class _Tensor:
        def __add__(self, _value: float) -> _Tensor:
            return self

        def numpy(self) -> np.ndarray[Any, Any]:
            return np.array([0.0 if failure == "tensor_value" else 2.0])

    class _SmokeModels:
        device = "cpu"

        def detect(self, _image: Any) -> tuple[Any, ...]:
            assert self.device == "gpu:0"
            return record("detect", ())

        def attributes(self, _crops: Any) -> tuple[dict[str, str], ...]:
            assert self.device == "gpu:0"
            result = ({},) if failure == "empty_attributes" else ({"age": "18-60"},)
            return record("attributes", result)

    backend = SimpleNamespace(
        device=SimpleNamespace(
            is_compiled_with_cuda=lambda: record(
                "compiled", failure != "compiled_false"
            ),
            cuda=SimpleNamespace(
                device_count=lambda: record(
                    "devices", 0 if failure == "no_device" else 1
                ),
                get_device_properties=lambda _device: record("properties"),
            ),
            set_device=lambda _device: record("set_device"),
        ),
        version=SimpleNamespace(
            cuda_archs=lambda: record("architecture", "private-arch")
        ),
        ones=lambda *_args, **_kwargs: record("tensor", _Tensor()),
        utils=SimpleNamespace(run_check=lambda: record("runtime_check")),
    )
    monkeypatch.setitem(sys.modules, "paddle", backend)
    monkeypatch.setitem(
        sys.modules, "app", SimpleNamespace(PaddleRichModels=_SmokeModels)
    )
    return calls


def test_gpu_smoke_checks_real_execution_paths_but_reports_neutral_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_smoke_backends(monkeypatch)

    assert gpu_smoke.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "acceleration": "ready",
        "object_detection": "ready",
        "pedestrian_attributes": "ready",
        "blank_detections": 0,
        "attribute_outputs": 1,
    }
    assert calls == [
        "compiled", "devices", "set_device", "properties", "architecture",
        "tensor", "runtime_check", "detect", "attributes",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "compiled_false", "no_device", "set_device", "tensor_value",
        "runtime_check", "detect", "attributes", "empty_attributes",
    ],
)
def test_gpu_smoke_never_reports_success_when_a_required_check_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    _install_smoke_backends(monkeypatch, failure=failure)

    assert gpu_smoke.main() == 1

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "ok": False,
        "error": "object and attribute engine validation failed",
    }
    assert captured.err == ""


def test_gpu_smoke_rejects_a_cpu_override_instead_of_claiming_acceleration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = _install_smoke_backends(monkeypatch)
    monkeypatch.setenv("RICH_DEVICE", "cpu")

    assert gpu_smoke.main() == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert calls == []


def test_gpu_smoke_reports_missing_backend_without_an_import_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(sys.modules, "paddle", None)

    assert gpu_smoke.main() == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] is False
    assert captured.err == ""
