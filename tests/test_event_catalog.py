"""The console learns presets from the detector contract, including future IDs."""

from fastapi.testclient import TestClient

from goodvision_harness import grounding
from goodvision_harness.event_catalog import EVENT_PRESENTATION, event_catalog
from goodvision_harness.event_sensitivity import (
    EVENT_SENSITIVITY_DEFAULTS,
    EVENT_SENSITIVITY_OPTIONS,
)
from goodvision_harness.service import HarnessRuntime, create_app


def test_catalog_matches_detectors_and_returns_independent_values():
    catalog = event_catalog()
    assert catalog["version"] == 1
    assert [item["id"] for item in catalog["events"]] == list(
        grounding.DETECTION_EVENT_LABELS
    )
    by_id = {item["id"]: item for item in catalog["events"]}
    assert by_id["ground_litter"]["scope"] == "scene"
    assert by_id["stranger"]["kind"] == "identity"
    assert "sensitivity" not in by_id["stranger"]
    assert by_id["smoking"]["sensitivity"]["default"] == "high"
    by_id["smoking"]["sensitivity"]["options"][0]["value"] = "mutated"
    by_id["fight"]["labels"]["zh"] = "mutated"
    assert event_catalog() != catalog
    assert EVENT_SENSITIVITY_OPTIONS["smoking"][0]["value"] == "standard"
    assert EVENT_PRESENTATION["fight"]["labels"]["zh"] == "打架"


def test_future_presets_and_removal_reach_both_console_contracts(monkeypatch):
    monkeypatch.setattr(
        grounding, "DETECTION_EVENT_LABELS", ("spill", "unmapped_event")
    )
    monkeypatch.setitem(
        EVENT_PRESENTATION,
        "spill",
        {
            "labels": {"zh": "地面积水", "en": "Floor spill"},
            "scope": "scene",
        },
    )
    monkeypatch.setitem(EVENT_SENSITIVITY_DEFAULTS, "spill", "strict")
    monkeypatch.setitem(
        EVENT_SENSITIVITY_OPTIONS,
        "spill",
        [
            {"value": "strict", "label": "严格", "description": "可见积水"},
        ],
    )
    runtime = HarnessRuntime(None, config_error="not configured")
    with TestClient(create_app(lambda: runtime)) as client:
        info = client.get("/v1/ui/info").json()
        # Catalog loading is independent of model credentials and video startup.
        assert info["configured"] is False
        assert info["event_catalog"] == runtime.analysis_settings()["event_catalog"]
        events = info["event_catalog"]["events"]
        assert [item["id"] for item in events] == ["spill", "unmapped_event"]
        assert events[0]["sensitivity"]["default"] == "strict"
        assert events[1]["labels"] == {"zh": "unmapped_event", "en": "unmapped_event"}
        asset = client.get("/assets/event_catalog.js")
        assert asset.status_code == 200
        assert "application/javascript" in asset.headers["content-type"]
        assert "GoodvisionEvents" in asset.text
        assert runtime._session is None
