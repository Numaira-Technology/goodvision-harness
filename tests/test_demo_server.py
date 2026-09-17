"""Local demo must enable the console API-key field without Docker volumes."""

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from goodvision_harness.config import load_config
from goodvision_harness.demo.server import create_demo_app

ROOT = Path(__file__).resolve().parents[1]


def test_demo_app_enables_console_api_key_without_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in (
        "HARNESS_DEPLOYMENT_SETTINGS_PATH",
        "INFERENCE_API_KEY",
        "OPENROUTER_API_KEY",
        "AICLUB_API_KEY",
        "MODEL_API_KEY",
        "DASHSCOPE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "false")
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    path = tmp_path / "config.yaml"
    path.write_bytes((ROOT / "config.yaml").read_bytes())
    config = load_config(path)
    config = replace(config, events=replace(config.events, directory=tmp_path / "events"))

    with TestClient(create_demo_app(config)) as client:
        settings = client.get("/v1/deployment/settings").json()
        assert settings["persistence_available"] is True
        saved = client.put("/v1/deployment/settings", json={"api_key": "offline-demo-key"})
        assert saved.status_code == 200
        assert saved.json()["api_key_present"] is True
        assert saved.json()["api_key_persisted"] is True
        assert "offline-demo-key" not in saved.text
    assert (tmp_path / ".goodvision" / "model-api-key").read_text(encoding="utf-8").strip() == (
        "offline-demo-key"
    )


def test_dashboard_requests_undelayed_live_overlay_when_tracking() -> None:
    html = (ROOT / "src/goodvision_harness/dashboard.html").read_text(encoding="utf-8")
    assert "/v1/video/live-overlay/latest.jpg" in html
    assert "person_tracking_enabled" in html
