from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["compose.yaml", "compose.delivery.yaml"])
def test_compose_passes_route_overrides_and_independent_credentials(filename: str) -> None:
    compose = yaml.safe_load((ROOT / filename).read_text(encoding="utf-8"))
    environment = compose["services"]["harness"]["environment"]
    for key in (
        "INFERENCE_PROVIDER",
        "INFERENCE_MODEL",
        "INFERENCE_ENDPOINT",
        "INFERENCE_API_KEY",
        "AICLUB_API_KEY",
        "OPENROUTER_API_KEY",
        "DASHSCOPE_API_KEY",
    ):
        # Empty values must retain YAML/preset defaults, rather than fixing a route.
        assert environment[key] == "${" + key + ":-}"


def test_packaged_config_matches_repository_default() -> None:
    assert (ROOT / "config.yaml").read_bytes() == (
        ROOT / "src" / "goodvision_harness" / "config.yaml"
    ).read_bytes()
