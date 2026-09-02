from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DELIVERY_PATH = ROOT / "compose.full-gpu.delivery.yaml"


def _load_delivery() -> dict[str, Any]:
    return yaml.safe_load(DELIVERY_PATH.read_text(encoding="utf-8"))


def test_full_gpu_delivery_starts_both_prebuilt_services_by_default() -> None:
    services = _load_delivery()["services"]
    assert set(services) == {"harness", "rich-label-worker"}
    assert services["harness"]["image"] == (
        "goodvision-harness:2026.09.02-full-gpu.2"
    )
    assert services["rich-label-worker"]["image"] == (
        "goodvision-rich-label-worker:2026.09.02-full-gpu.2"
    )
    for service in services.values():
        assert "profiles" not in service
        assert "build" not in service
        assert service["pull_policy"] == "never"


def test_full_gpu_settings_are_fixed_despite_old_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RICH_LABEL_ENABLED", "false")
    monkeypatch.setenv("RICH_LABEL_ENDPOINT", "http://old-worker:8091/v1/labels")
    monkeypatch.setenv("FACE_PROVIDER", "CPUExecutionProvider")

    environment = _load_delivery()["services"]["harness"]["environment"]
    # Literal values cannot be substituted by old .env or shell settings.
    assert environment["RICH_LABEL_ENABLED"] == "true"
    assert environment["RICH_LABEL_ENDPOINT"] == (
        "http://rich-label-worker:8091/v1/labels"
    )
    assert environment["FACE_ACCELERATOR"] == "gpu"
    assert "FACE_PROVIDER" not in environment


def test_full_gpu_delivery_preserves_project_and_persistent_volumes() -> None:
    delivery = _load_delivery()
    assert delivery["name"] == "goodvision-harness"
    assert delivery["volumes"] == {
        "face-models": None,
        "face-data": None,
        "rich-models": None,
    }
    services = delivery["services"]
    assert "face-models:/models" in services["harness"]["volumes"]
    assert "face-data:/data" in services["harness"]["volumes"]
    assert "./config.delivery.yaml:/app/config.yaml:ro" in (
        services["harness"]["volumes"]
    )
    assert services["rich-label-worker"]["volumes"] == ["rich-models:/models"]


def test_full_gpu_worker_does_not_publish_host_ports() -> None:
    worker = _load_delivery()["services"]["rich-label-worker"]
    assert "ports" not in worker
    assert worker.get("network_mode") != "host"


def test_full_gpu_delivery_reserves_one_gpu_without_vendor_selector() -> None:
    for service in _load_delivery()["services"].values():
        devices = service["deploy"]["resources"]["reservations"]["devices"]
        assert devices == [{"count": 1, "capabilities": ["gpu"]}]


def test_full_gpu_delivery_uses_neutral_credentials_and_model_profiles() -> None:
    services = _load_delivery()["services"]
    harness = services["harness"]["environment"]
    for key in (
        "INFERENCE_PROVIDER",
        "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT",
        "INFERENCE_API_KEY",
    ):
        assert harness[key] == "${" + key + ":-}"
    assert not any(
        key in harness
        for key in ("AICLUB_API_KEY", "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY")
    )
    assert harness["VIDEO_CAPTURE_OPTIONS"] == (
        "${VIDEO_CAPTURE_OPTIONS:-rtsp_transport;tcp|timeout;5000000}"
    )
    worker = services["rich-label-worker"]["environment"]
    assert worker["RICH_DEVICE"] == "gpu:0"
    assert worker["RICH_DETECTOR_PROFILE"] == "${RICH_DETECTOR_PROFILE:-accurate}"
    assert worker["RICH_ATTRIBUTE_PROFILE"] == "${RICH_ATTRIBUTE_PROFILE:-standard}"
    assert "RICH_DETECTOR_MODEL" not in worker
    assert "RICH_ATTRIBUTE_MODEL" not in worker
    for key in ("RICH_DETECTOR_MODEL_DIR", "RICH_ATTRIBUTE_MODEL_DIR"):
        assert worker[key] == "${" + key + ":-}"


def test_neutral_delivery_configuration_preserves_analysis_settings() -> None:
    source = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    delivery = yaml.safe_load(
        (ROOT / "config.delivery.yaml").read_text(encoding="utf-8")
    )
    expected = deepcopy(source)
    expected["inference_service"].pop("provider")
    expected["inference_service"].pop("providers")
    expected["face"].pop("model_name")
    expected["face"].pop("provider")
    expected["face"].update(
        model_profile="standard", model_root="/models", accelerator="gpu"
    )
    assert delivery == expected


@pytest.mark.parametrize(
    "filename",
    [
        "API_DELIVERY.md",
        "FULL_DEPLOY.md",
        "compose.full-gpu.delivery.yaml",
        ".env.full-gpu.delivery.example",
        "config.delivery.yaml",
    ],
)
def test_customer_delivery_text_uses_neutral_implementation_names(filename: str) -> None:
    text = (ROOT / filename).read_text(encoding="utf-8")
    forbidden = re.compile(
        r"openrouter|qwen|insightface|paddle|picodet|buffalo|onnx|opencv|ffmpeg|"
        r"aiclub|nvidia|cuda|cudnn|tensorrt|huggingface|openai|anthropic|"
        r"deepseek|gemini|claude|pp[-_]yolo|pp[-_]lcnet",
        re.IGNORECASE,
    )
    assert forbidden.search(text) is None


def test_delivery_guide_only_references_neutral_runtime_check_entrypoints() -> None:
    text = (ROOT / "FULL_DEPLOY.md").read_text(encoding="utf-8")
    assert set(re.findall(r"tools/[a-z_]+\.py", text)) == {
        "tools/check_runtime.py",
        "tools/check_face.py",
        "tools/check_perception.py",
    }
