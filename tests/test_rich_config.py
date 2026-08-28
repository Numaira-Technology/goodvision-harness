from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from goodvision_harness.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_rich_labeling_defaults_are_safe_and_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RICH_LABEL_ENABLED", raising=False)
    monkeypatch.delenv("RICH_LABEL_ENDPOINT", raising=False)

    settings = load_config(ROOT / "config.yaml").rich_labeling

    assert settings.enabled is False
    assert settings.endpoint == "http://rich-label-worker:8091/v1/labels"
    assert settings.max_fps == 7.5
    assert settings.max_interpolation_gap_seconds == 0.75


def test_rich_labeling_environment_overrides_connection_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RICH_LABEL_ENABLED", "true")
    monkeypatch.setenv("RICH_LABEL_ENDPOINT", "http://127.0.0.1:18091/v1/labels")

    settings = load_config(ROOT / "config.yaml").rich_labeling

    assert settings.enabled is True
    assert settings.endpoint == "http://127.0.0.1:18091/v1/labels"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("endpoint", "file:///tmp/worker", "must use http or https"),
        ("timeout_seconds", 0, "positive number"),
        ("max_fps", 0, "positive number"),
        ("history_size", 0, "positive integer"),
        ("history_size", 1, "at least 2"),
        ("max_interpolation_gap_seconds", -1, "positive number"),
    ],
)
def test_rich_labeling_config_validation_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.delenv("RICH_LABEL_ENABLED", raising=False)
    monkeypatch.delenv("RICH_LABEL_ENDPOINT", raising=False)
    payload = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    payload["rich_labeling"][field] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_config(config_path)
