"""Console choices must change evidence policy without changing JSON safety."""

import json
import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import replace
from itertools import product
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest
from fake_session_owner import FakeSessionOwner
from fastapi.testclient import TestClient

from goodvision_harness.config import load_config
from goodvision_harness.event_sensitivity import (
    normalize_event_sensitivities,
    sensitivity_policy_text,
)
from goodvision_harness.grounding import (
    SAFETY_EVENT_LABELS,
    AIClubMultiGroundingClient,
    build_grounding_messages,
    build_multi_grounding_messages,
)
from goodvision_harness.identity_review_worker import IdentityReviewWorker
from goodvision_harness.model_api import ModelAPIClient
from goodvision_harness.pipeline import StreamingSession
from goodvision_harness.reference_memory import MemoryContext
from goodvision_harness.route_probe import probe_window
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]


def _messages(level, *, multi=True, labels=("smoking",)):
    settings = load_config(ROOT / "config.delivery.yaml").inference
    kwargs = {
        "system_prompt": settings.system_prompt,
        "prompt": settings.prompt,
        "event_sensitivities": {label: level for label in labels},
    }
    if multi:
        return build_multi_grounding_messages(
            probe_window(), event_labels=labels, **kwargs,
        )
    return build_grounding_messages(
        probe_window(), event_label=labels[0], **kwargs,
    )


def _text(messages):
    return messages[0]["content"] + "\n" + messages[1]["content"][0]["text"]


@pytest.mark.parametrize("multi", [True, False])
def test_smoking_high_has_no_contradictory_item_evidence_or_box_rule(multi):
    text = _text(_messages("high", multi=multi))
    assert "an identifiable smoking item is NOT required" in text
    assert "smoking-like hand-to-mouth" in text
    assert "label smoking_action" in text
    assert "Never fabricate a cigarette box" in text
    assert "Lower confidence when uncertain" in text
    for obsolete in (
        "Mandatory smoking evidence rule", "detected=true requires BOTH",
        "return only the cigarette", "never the smoker's person box",
        "Keep smoking/weapon objects as item boxes", "For smoking and fall",
        "white object alone is not enough",
    ):
        assert obsolete not in text
    assert "0..1000" in text
    assert "objects=[]" in text
    if multi:
        assert "top-level events array" in text
        assert '{"no_events":0.95}' in text


@pytest.mark.parametrize("multi", [True, False])
def test_smoking_standard_retains_identifiable_item_floor(multi):
    text = _text(_messages("standard", multi=multi))
    assert "Mandatory smoking evidence rule" in text
    assert "detected=true requires BOTH an identifiable cigarette" in text
    assert "never the smoker's person box" in text
    assert "smoking_action" not in text
    assert "Smoking sensitivity HIGH" not in text


@pytest.mark.parametrize("label,standard,high", [
    ("fight", "raised fists", "plausible early confrontation"),
    ("fall", "require visible loss of balance/support", "plausible imminent fall"),
    ("weapon", "visible firearm", "partly obscured weapon-like item"),
])
def test_each_other_event_has_distinct_real_policy(label, standard, high):
    standard_text = _text(_messages("standard", labels=(label,)))
    high_text = _text(_messages("high", labels=(label,)))
    assert standard in standard_text
    assert high in high_text
    assert standard_text != high_text


@pytest.mark.parametrize("multi", [False, True])
def test_weapon_standard_prompt_preserves_visible_weapons_and_rejects_pose_only(multi):
    text = _text(_messages("standard", multi=multi, labels=("weapon",)))
    assert "weapon-specific features" in text
    assert "weapon does not require an attack to be positive" in text
    assert "object itself is being used to strike, stab" in text
    assert "punching or raised fists must not turn an unidentified handheld" in text
    assert "Carrying, placing, dropping, picking up" in text
    assert "high-recall policy" not in text


def test_temporal_weapon_policy_has_same_evidence_floor_without_box_instructions():
    text = sensitivity_policy_text(
        {"weapon": "standard"}, smoking_standard_rule="unused",
        include_output_guidance=False,
    )
    assert "a blade with its handle" in text
    assert "firearm's barrel and grip" in text
    assert "mere possession or movement is insufficient" in text
    assert "Box the actual item" not in text


@pytest.mark.parametrize("levels", product(("standard", "high"), repeat=4))
def test_evidence_only_policies_preserve_each_level_without_output_conflicts(levels):
    choices = dict(zip(("fight", "smoking", "fall", "weapon"), levels, strict=True))
    standard_smoking = (
        "Smoking STANDARD: require an identifiable smoking item and smoking action."
    )
    text = sensitivity_policy_text(
        choices, smoking_standard_rule=standard_smoking,
        include_output_guidance=False,
    )
    criteria = {
        ("fight", "standard"): "raised fists",
        ("fight", "high"): "plausible early confrontation",
        ("smoking", "standard"): standard_smoking,
        ("smoking", "high"): "an identifiable smoking item is NOT required",
        ("fall", "standard"): "require visible loss of balance/support",
        ("fall", "high"): "plausible imminent fall",
        ("weapon", "standard"): "visible firearm",
        ("weapon", "high"): "partly obscured weapon-like item",
    }
    for label, level in choices.items():
        assert criteria[(label, level)] in text
        other_level = "high" if level == "standard" else "standard"
        own_policy = sensitivity_policy_text(
            {label: level}, smoking_standard_rule=standard_smoking,
            include_output_guidance=False,
        )
        # Another event may mention the cue to reject a cross-label inference
        # (e.g. raised fists alone are not weapon evidence).
        assert criteria[(label, other_level)] not in own_policy
    for output_instruction in ("box", "describe", "description", "report suspected"):
        assert output_instruction not in text.lower()
    if choices["smoking"] == "high":
        assert "Lower confidence when uncertain" in text
        assert "do not claim an identifiable cigarette, smoke" in text
        assert "eating, drinking, or wiping the face remains negative" in text
    if choices["weapon"] == "high":
        assert "Lower confidence if uncertain" in text
        assert "Bare-hand gestures without a visible item" in text
        assert "Never invent a hidden weapon" in text


@pytest.mark.parametrize("label,level,instructions", [
    ("smoking", "high", (
        "describe a suspected smoking action",
        "box the visible hand/mouth action region (label smoking_action)",
        "Never fabricate a cigarette box",
    )),
    ("weapon", "standard", ("Box the actual item.",)),
    ("weapon", "high", (
        "Report suspected weapon and lower confidence if uncertain.",
        "Box only the visible item, never an invented hidden weapon.",
    )),
])
def test_grounding_policy_keeps_output_guidance_by_default(label, level, instructions):
    text = sensitivity_policy_text(
        {label: level}, smoking_standard_rule="Unused smoking standard policy.",
    )
    for instruction in instructions:
        assert instruction in text


def test_multi_client_uses_selected_policies_instead_of_legacy_fixed_prompt():
    settings = load_config(ROOT / "config.delivery.yaml").inference
    transport = Mock(spec=ModelAPIClient)
    transport.chat_completion.return_value = {
        "choices": [{"message": {"content": '{"no_events":0.9}'}}],
    }
    transport.assistant_text.side_effect = ModelAPIClient.assistant_text
    choices = {"smoking": "high", "fall": "standard"}
    client = AIClubMultiGroundingClient(
        settings, transport, SAFETY_EVENT_LABELS, event_sensitivities=choices,
    )
    choices["smoking"] = "standard"
    result = client.classify_window(probe_window())
    assert result.classification is not None
    text = _text(transport.chat_completion.call_args.kwargs["messages"])
    assert "Smoking sensitivity HIGH" in text
    assert "Falling sensitivity STANDARD" in text
    assert "Mandatory smoking evidence rule" not in text
    assert "top-level events array" in text
    assert "smoking_action" in text


@pytest.mark.parametrize("invalid", [
    [], "high", 1, True, {"stranger": "high"}, {"unknown": "standard"},
    {"smoking": "low"}, {"smoking": True}, {"smoking": None},
    {"smoking": "HIGH"},
])
def test_unknown_events_or_levels_are_rejected(invalid):
    with pytest.raises(ValueError):
        normalize_event_sensitivities(invalid, ("smoking",))


def test_defaults_are_snapshot_and_only_apply_to_selected_events():
    assert normalize_event_sensitivities(None, ("smoking", "stranger")) == {
        "smoking": "high",
    }
    assert normalize_event_sensitivities(None, ("fight", "fall", "weapon")) == {
        "fight": "standard", "fall": "standard", "weapon": "standard",
    }
    assert normalize_event_sensitivities({"smoking": "standard"}, ("stranger",)) == {}


@pytest.mark.parametrize("event_label,memory", [
    ("custom_event", None),
    ("smoking", MemoryContext("custom_memory", (), "Apply custom memory semantics.")),
    ("smoking", MemoryContext("face_recognition", (), "Match only reference faces.")),
])
def test_console_policy_does_not_replace_custom_or_memory_prompts(event_label, memory):
    kwargs = {
        "event_label": event_label, "memory_context": memory,
        "system_prompt": "Preserve this configured system policy.",
        "prompt": "Preserve this custom event policy.",
    }
    original = build_grounding_messages(probe_window(), **kwargs)
    selected = build_grounding_messages(
        probe_window(), event_sensitivities={"smoking": "high"}, **kwargs,
    )
    assert selected == original


@pytest.fixture
def runtime_owner(tmp_path, monkeypatch, request):
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "true")
    config = load_config(ROOT / "config.delivery.yaml")
    config = replace(
        config, stream=replace(config.stream, url="rtsp://unused.local/live"),
        events=replace(config.events, directory=tmp_path / "events"),
    )
    instance = HarnessRuntime(config)
    return FakeSessionOwner(instance, monkeypatch, request)


@pytest.fixture
def runtime(runtime_owner):
    return runtime_owner.runtime


def test_http_start_snapshots_choices_and_ui_exposes_options(runtime, runtime_owner):
    with TestClient(create_app(lambda: runtime)) as client:
        info = client.get("/v1/ui/info").json()
        assert info["event_sensitivity_defaults"]["smoking"] == "high"
        assert set(info["event_sensitivity_options"]) == set(SAFETY_EVENT_LABELS)
        assert "threshold" in info["face"]
        response = client.post("/v1/sessions/start", json={
            "features": ["smoking", "fall"], "dry_run": True,
            "event_sensitivities": {"smoking": "standard", "fall": "high"},
        })
        assert response.status_code == 200
        expected = {"smoking": "standard", "fall": "high"}
        assert response.json()["event_sensitivities"] == expected
        assert client.get("/v1/status").json()["event_sensitivities"] == expected
        assert info["event_sensitivity_defaults"]["smoking"] == "high"
        assert runtime_owner.workers == [runtime._identity_worker]
        assert runtime_owner.sessions == [runtime._session]


@pytest.mark.parametrize("mode", ["camera", "stream"])
def test_websocket_start_propagates_choices(runtime, runtime_owner, mode):
    with TestClient(create_app(lambda: runtime)) as client:
        with client.websocket_connect("/ws") as socket:
            assert socket.receive_json()["type"] == "hello"
            socket.send_json({
                "type": "start", "mode": mode, "features": ["smoking"],
                "dry_run": True, "event_sensitivities": {"smoking": "standard"},
            })
            response = socket.receive_json()
            assert response["type"] == "session"
            assert response["event_sensitivities"] == {"smoking": "standard"}
            assert runtime.status()["event_sensitivities"] == {"smoking": "standard"}
            assert runtime_owner.workers == [runtime._identity_worker]
            assert runtime_owner.sessions == [runtime._session]


@pytest.mark.parametrize("invalid", [{"smoking": "low"}, {"stranger": "high"}, []])
def test_http_and_ws_reject_invalid_settings_before_start(
    runtime, runtime_owner, invalid,
):
    with TestClient(create_app(lambda: runtime)) as client:
        response = client.post("/v1/sessions/start", json={
            "features": ["smoking"], "event_sensitivities": invalid,
        })
        assert response.status_code == 400
        assert runtime._session is None
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.send_json({
                "type": "start", "mode": "camera", "features": ["smoking"],
                "event_sensitivities": invalid,
            })
            assert socket.receive_json()["type"] == "error"
            assert runtime._session is None
    assert runtime_owner.workers == []
    assert runtime_owner.sessions == []


def test_active_session_cannot_change_sensitivity(runtime, runtime_owner, monkeypatch):
    with TestClient(create_app(lambda: runtime)) as client:
        first = client.post("/v1/sessions/start", json={
            "features": ["smoking"], "event_sensitivities": {"smoking": "standard"},
        })
        assert first.status_code == 200
        with monkeypatch.context() as patch:
            patch.setattr(StreamingSession, "running", property(lambda self: True))
            response = client.post("/v1/sessions/start", json={
                "features": ["smoking"], "event_sensitivities": {"smoking": "high"},
            })
        assert response.status_code == 409
        assert runtime.status()["event_sensitivities"] == {"smoking": "standard"}
        assert runtime_owner.workers == [runtime._identity_worker]
        assert runtime_owner.sessions == [runtime._session]


def test_fake_start_cleanup_is_repeatable_and_owner_scoped(runtime, runtime_owner):
    runtime.start_camera(owner="test", features=["smoking"], dry_run=True)
    worker = runtime._identity_worker
    assert runtime_owner.workers == [worker]
    assert runtime_owner.sessions == [runtime._session]
    assert runtime._session._thread is None
    reviewed = Event()

    def review():
        with runtime._control_lock, runtime._state_lock:
            reviewed.set()

    worker._review = review
    unrelated = IdentityReviewWorker(lambda: None)
    try:
        worker.wake()
        # The finalizer must retain this exact owner even after detachment.
        runtime._identity_worker = None
        runtime_owner.close()
        assert reviewed.is_set()
        assert worker._closing and not worker._thread.is_alive()
        runtime_owner.close()
        assert worker._closing and not worker._thread.is_alive()
        assert not unrelated._closing and unrelated._thread.is_alive()
    finally:
        unrelated.close(timeout_seconds=1.0)
        assert not unrelated._thread.is_alive()


@pytest.mark.parametrize("stop_fails,pythonpath", [(False, None), (True, "unrelated")])
def test_fake_start_finalizer_preserves_body_failure(
    tmp_path, monkeypatch, stop_fails, pythonpath,
):
    # Exercise real pytest call/teardown reporting, not a hand-written imitation
    # of request.addfinalizer. The child creates only one idle identity worker.
    monkeypatch.setenv("PYTEST_ADDOPTS", "-p nonexistent_outer_plugin")
    monkeypatch.setenv("PYTEST_PLUGINS", "nonexistent_outer_plugin")
    if pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", str(tmp_path / pythonpath))
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    case = tmp_path / "test_failure.py"
    case.write_text(textwrap.dedent(f"""
        import json
        from pathlib import Path
        from types import SimpleNamespace

        from fake_session_owner import FakeSessionOwner
        from goodvision_harness.identity_review_worker import IdentityReviewWorker
        from goodvision_harness.pipeline import StreamingSession

        def test_failure(request, monkeypatch):
            stops = []
            def stop():
                stops.append(True)
                if {stop_fails!r}:
                    raise RuntimeError("intentional stop failure")

            runtime = SimpleNamespace(
                stop=stop, _session=SimpleNamespace(_thread=None),
                _identity_worker=None,
            )
            def record_cleanup():
                Path("cleanup.json").write_text(json.dumps({{
                    "closing": worker._closing,
                    "alive": worker._thread.is_alive(), "stops": len(stops),
                    "captured": owner.workers == [worker],
                }}))

            request.addfinalizer(record_cleanup)
            owner = FakeSessionOwner(runtime, monkeypatch, request)
            worker = IdentityReviewWorker(lambda: None)
            runtime._identity_worker = worker
            StreamingSession.start(runtime._session)
            runtime._identity_worker = None
            assert False, "intentional body failure"
    """), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-c", str(config),
         f"--confcutdir={tmp_path}", str(case), "--junitxml=result.xml"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={
            # Keep OS execution necessities, not outer pytest plugins, sharding
            # options or PYTHONPATH. Import this checkout explicitly in the child.
            **{name: os.environ[name] for name in (
                "PATH", "SystemRoot", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR",
            ) if name in os.environ},
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": os.pathsep.join((str(ROOT / "tests"), str(ROOT / "src"))),
        },
    )
    assert result.returncode == 1, result.stdout + result.stderr
    report = ET.parse(tmp_path / "result.xml")
    failures = report.findall(".//failure")
    errors = report.findall(".//error")
    assert len(failures) == 1
    assert "intentional body failure" in failures[0].attrib["message"]
    assert len(errors) == int(stop_fails)
    if stop_fails:
        assert "intentional stop failure" in errors[0].attrib["message"]
    assert json.loads((tmp_path / "cleanup.json").read_text()) == {
        "closing": True, "alive": False, "stops": 1, "captured": True,
    }
