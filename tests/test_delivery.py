"""Contracts for the supported full delivery recipes and customer documents."""

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


def test_full_delivery_keeps_both_images_and_requires_gpu_profile_for_rich() -> None:
    services = _load_delivery()["services"]
    assert set(services) == {"harness", "rich-label-worker"}
    assert services["harness"]["image"] == (
        "goodvision-harness:20260923-full1"
    )
    assert services["rich-label-worker"]["image"] == (
        "goodvision-rich-label-worker:20260923-full1"
    )
    for service in services.values():
        assert "build" not in service
        assert service["pull_policy"] == "never"
    assert "profiles" not in services["harness"]
    assert services["rich-label-worker"]["profiles"] == ["gpu"]


def test_full_gpu_settings_are_fixed_despite_old_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RICH_LABEL_ENABLED", "false")
    monkeypatch.setenv("RICH_LABEL_ENDPOINT", "http://old-worker:8091/v1/labels")
    monkeypatch.setenv("FACE_PROVIDER", "CPUExecutionProvider")
    monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", "development")
    monkeypatch.setenv("INFERENCE_PROVIDER", "aiclub")
    monkeypatch.setenv("INFERENCE_MODEL", "old-model")
    monkeypatch.setenv("INFERENCE_ENDPOINT", "http://old-service/v1/chat/completions")

    environment = _load_delivery()["services"]["harness"]["environment"]
    # Literal values cannot be substituted by old .env or shell settings.
    assert environment["RICH_LABEL_ENABLED"] == "false"
    assert environment["RICH_LABEL_ENDPOINT"] == (
        "http://rich-label-worker:8091/v1/labels"
    )
    assert environment["FACE_ACCELERATOR"] == "cpu"
    accelerated = yaml.safe_load(
        (ROOT / "compose.full-gpu.accelerated.yaml").read_text(encoding="utf-8")
    )["services"]["harness"]["environment"]
    assert accelerated["FACE_ACCELERATOR"] == "gpu"
    assert accelerated["RICH_LABEL_ENABLED"] == "true"
    assert "FACE_PROVIDER" not in environment
    assert environment["HARNESS_INFERENCE_PROFILE"] == "customer-standard"
    assert not any(
        key in environment
        for key in ("INFERENCE_PROVIDER", "INFERENCE_MODEL", "INFERENCE_ENDPOINT")
    )


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


def test_full_delivery_reserves_devices_only_in_gpu_overlay() -> None:
    for service in _load_delivery()["services"].values():
        assert "deploy" not in service
    accelerated = yaml.safe_load(
        (ROOT / "compose.full-gpu.accelerated.yaml").read_text(encoding="utf-8")
    )
    for service in accelerated["services"].values():
        devices = service["deploy"]["resources"]["reservations"]["devices"]
        assert devices == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]


def test_full_gpu_delivery_uses_neutral_credentials_and_model_profiles() -> None:
    services = _load_delivery()["services"]
    harness = services["harness"]["environment"]
    assert harness["INFERENCE_API_KEY"] == "${INFERENCE_API_KEY:-}"
    assert harness["HARNESS_INFERENCE_PROFILE"] == "customer-standard"
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
    example = (ROOT / ".env.full-gpu.delivery.example").read_text(encoding="utf-8")
    assert not re.search(
        r"(?m)^(?:INFERENCE_PROVIDER|INFERENCE_MODEL|INFERENCE_ENDPOINT)=", example
    )


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
    expected["events"].pop("ffmpeg_path")
    expected["face"].update(
        model_profile="standard", model_root="/models", accelerator="gpu"
    )
    expected["display"]["tracking_delay_seconds"] = expected["display"].pop(
        "sam_live_delay_seconds"
    )
    assert delivery == expected


@pytest.mark.parametrize(
    "filename",
    [
        "API_DELIVERY.md",
        "DEPLOY.md",
        "compose.full-gpu.delivery.yaml",
        ".env.full-gpu.delivery.example",
        "config.delivery.yaml",
    ],
)
def test_customer_delivery_text_uses_neutral_implementation_names(
    filename: str,
) -> None:
    text = (ROOT / filename).read_text(encoding="utf-8")
    forbidden = re.compile(
        r"openrouter|qwen|insightface|paddle|picodet|buffalo|onnx|opencv|ffmpeg|"
        r"aiclub|nvidia|cuda|cudnn|tensorrt|huggingface|openai|anthropic|"
        r"deepseek|gemini|claude|pp[-_]yolo|pp[-_]lcnet",
        re.IGNORECASE,
    )
    assert forbidden.search(text) is None


def test_delivery_guide_only_references_neutral_runtime_check_entrypoints() -> None:
    text = (ROOT / "DEPLOY.md").read_text(encoding="utf-8")
    assert set(re.findall(r"tools/[a-z_]+\.py", text)) == {
        "tools/check_runtime.py",
        "tools/check_face.py",
        "tools/check_perception.py",
    }


def test_project_metadata_uses_a_neutral_inline_description() -> None:
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = project["project"]["readme"]
    assert isinstance(readme, dict)
    assert readme["content-type"] == "text/markdown"
    assert "text" in readme and "file" not in readme
    assert "GoodVision" in readme["text"]
    assert "license and attribution" in readme["text"]
    assert re.search(
        r"openrouter|qwen|insightface|paddle|aiclub|cuda|onnx|opencv",
        readme["text"],
        re.IGNORECASE,
    ) is None
