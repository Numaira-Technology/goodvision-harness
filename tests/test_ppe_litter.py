"""New event policies, person-free lifecycle, delivery and archive regressions."""

import json
from dataclasses import replace

import jsonschema
import pytest
from test_multi_event import ROOT, _window
from test_vlm_independent import BOX, config, job, observe_person, report

from goodvision_harness.contracts import FrameReference
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.grounding import (
    DETECTION_EVENT_LABELS,
    build_multi_grounding_messages,
    parse_multi_grounded_classification,
)
from goodvision_harness.scene_archive import SceneArchive
from goodvision_harness.scene_delivery import _display_events
from goodvision_harness.scene_results import SceneResultPublisher
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.vlm_fusion import VLMFusion
from goodvision_harness.vlm_independent import IndependentEventState
from goodvision_harness.vlm_independent_model import (
    IndependentVerdict,
    IndependentVLMInspector,
)

LABELS = ("ppe_violation", "ground_litter")


def test_new_events_parse_and_use_evidence_rules_without_custom_sensitivity():
    payload = {
        "events": [
            {
                "event_label": label,
                "detected": True,
                "confidence": 0.9,
                "description": "可见证据 / Visible evidence",
                "objects": [
                    {
                        "label": target,
                        "confidence": 0.9,
                        "bbox_2d": [100, 100, 400, 900],
                    }
                ],
            }
            for label, target in zip(LABELS, ("person", "litter"), strict=True)
        ]
    }
    parsed = parse_multi_grounded_classification(
        json.dumps(payload), event_labels=LABELS
    )
    assert all(j.detected for j in parsed.judgments)
    messages = build_multi_grounding_messages(
        _window(),
        system_prompt="Detect events",
        prompt="Evaluate {events}",
        event_labels=LABELS,
    )
    rules = messages[0]["content"]
    assert "missing EITHER" in rules
    assert "Occlusion, cropping, blur or distance" in rules
    assert "No person or observed act of littering is required" in rules
    assert "Box the actual litter" in rules
    assert set(LABELS) <= set(DETECTION_EVENT_LABELS)


@pytest.mark.parametrize("label", LABELS)
def test_independent_event_reaches_confirmation_and_clears(label):
    cfg = replace(config(), labels=(label,))
    state = IndependentEventState(cfg)
    runtime = SceneRuntime(
        cfg.stream_session_id,
        cfg.source_id,
        event_labels=(label,),
        independent_temporal_inference=True,
    )
    fusion = VLMFusion(runtime)
    for at, seq, detected in (
        (1.0, 10, True),
        (2.0, 20, True),
        (3.0, 30, False),
        (4.0, 40, False),
    ):
        if label == "ppe_violation":
            observe_person(fusion, at, seq)
        else:
            runtime.observe_frame(job(cfg, at, seq).frames[-1].jpeg_bytes, at, seq)
        ids = ("p1",) if label == "ppe_violation" else ()
        boxes = (("p1", BOX),) if ids else ()
        result = report(
            state, at, seq, (IndependentVerdict(label, ids, detected, 0.9, boxes),)
        )
        assert result["observations"][0]["accepted"]
        assert fusion._associate(result, job(cfg, at, seq).frames, set())
        event = runtime.snapshot()["scene"]["events"][0]
        assert (
            event["phase"]
            == {10: "suspected", 20: "confirmed", 30: "confirmed", 40: "ended"}[seq]
        )
        if label == "ground_litter":
            assert runtime.snapshot()["scene"]["entities"] == []
            assert event["subject_ids"] == []
            assert runtime.scene.observe_tracking_presence(
                (), timestamp_seconds=at, frame_sequence=seq
            )
    assert all(link["scene_event_ids"] for link in fusion.links())


def test_litter_survives_empty_tracking_and_projects_and_archives(tmp_path):
    runtime = SceneRuntime("session", "camera", event_labels=("ground_litter",))
    for seq in (1, 2):
        assert runtime.scene.observe_event(
            "ground_litter",
            (),
            timestamp_seconds=float(seq),
            frame_sequence=seq,
            detected=True,
            confidence=0.9,
            evidence_id=f"litter-{seq}",
        )
    for seq in range(3, 24):
        runtime.scene.observe_tracking_presence(
            (), timestamp_seconds=float(seq), frame_sequence=seq
        )
    runtime.scene.advance(23.0)
    snapshot = runtime.snapshot()
    event = snapshot["scene"]["events"][0]
    assert event["phase"] == "confirmed"
    assert event["observation_state"] == "uncertain"
    assert _display_events(snapshot, 23.0) == [event]
    publisher = SceneResultPublisher()
    publisher.reset("session")
    publisher.observe_reference(
        "session", FrameReference(23, 23.0, "2026-09-28T00:00:00Z", 64, 48)
    )
    result = publisher.project(
        snapshot, labels=("ground_litter",), emitted_at="2026-09-28T00:00:00Z"
    )
    assert result.judgments[0].detected
    assert result.objects[0].label == "scene"
    jsonschema.validate(
        result.as_dict(),
        json.loads(
            (ROOT / "schemas/analysis-result-v2.schema.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    assert SceneArchive._record(event, "session", "camera")["subject_ids"] == []
    archive = EventArchive(EventArchiveSettings(directory=tmp_path))
    try:
        assert archive.submit_scene(snapshot)
        assert archive.flush()
        events = archive.list_events()["events"]
        assert len(events) == 1
        assert events[0]["subject_type"] == "scene"
        assert events[0]["subject_ids"] == ["scene-ground-litter"]
        jsonschema.validate(
            events[0],
            json.loads(
                (ROOT / "schemas/event-v1.schema.json").read_text(encoding="utf-8")
            ),
        )
    finally:
        archive.close()


def test_litter_parser_does_not_blame_nearby_person():
    cfg = replace(config(), labels=LABELS)
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "subjects": [
                                {"subject_id": "p1", "bbox": [0.1, 0.1, 0.5, 0.9]}
                            ],
                            "verdicts": [
                                {
                                    "event_label": "ground_litter",
                                    "subject_ids": ["p1"],
                                    "detected": True,
                                    "confidence": 0.9,
                                },
                                {
                                    "event_label": "ppe_violation",
                                    "subject_ids": ["p1"],
                                    "detected": False,
                                    "confidence": 0.9,
                                },
                            ],
                        }
                    )
                },
            }
        ]
    }
    verdicts = IndependentVLMInspector._parse(job(cfg, 1.0, 10), response)
    assert verdicts[0].subject_ids == () and verdicts[0].subject_boxes == ()
    assert verdicts[0].grounding_issues == ()
    assert verdicts[1].subject_ids == ("p1",)


def test_temporal_litter_scan_without_people():
    from test_vlm_independent import jpeg

    from goodvision_harness.scene_inspection import (
        _validate_verdicts,
        build_temporal_messages,
    )
    from goodvision_harness.scene_runtime import TemporalVerdict

    runtime = SceneRuntime("session", "camera", event_labels=("ground_litter",))
    runtime.observe_frame(jpeg(), 1.0, 1)
    runtime.observe_frame(jpeg(), 2.0, 2)
    jobs = runtime.take_inspections()
    assert jobs and jobs[0].subject_boxes == ()
    verdicts = (TemporalVerdict("ground_litter", (), True, 0.9),)
    assert _validate_verdicts(jobs[0], verdicts, set()) == verdicts
    assert "ground_litter" in build_temporal_messages(jobs[0])[0]["content"]
    assert runtime.complete_inspection(jobs[0].request.request_id, verdicts)
    assert runtime.snapshot()["scene"]["events"][0]["phase"] == "suspected"


def test_scene_only_mode_retains_all_requested_policies():
    from goodvision_harness.vlm_compact_model import EventJSONInspector

    cfg = replace(config(), labels=("fight", *LABELS))
    inspector = EventJSONInspector(cfg.model_settings, model="test", api_key="test")
    try:
        rules = inspector.build_messages(job(cfg, 1.0, 10))[0]["content"]
        assert "Fighting sensitivity STANDARD" in rules
        assert "missing EITHER" in rules
        assert "No person or observed act of littering is required" in rules
    finally:
        inspector.close()


def test_event_routes_and_selected_features(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from goodvision_harness.config import load_config
    from goodvision_harness.pipeline import StreamingSession
    from goodvision_harness.service import HarnessRuntime, create_app

    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "true")
    monkeypatch.setattr(StreamingSession, "start", lambda self: None)
    settings = load_config(ROOT / "config.delivery.yaml")
    settings = replace(
        settings,
        stream=replace(settings.stream, url="rtsp://unused.local/live"),
        events=replace(settings.events, directory=tmp_path / "events"),
    )
    runtime = HarnessRuntime(settings)
    try:
        with TestClient(create_app(lambda: runtime)) as client:
            info = client.get("/v1/ui/info").json()
            assert set(LABELS) <= set(info["supported_features"])
            response = client.post(
                "/v1/sessions/start", json={"features": list(LABELS), "dry_run": True}
            )
            assert response.status_code == 200
            assert response.json()["features"] == list(LABELS)
            assert response.json()["event_sensitivities"] == dict.fromkeys(
                LABELS, "standard"
            )
            paths = client.get("/openapi.json").json()["paths"]
            for label in LABELS:
                assert f"/v1/results/{label}/latest" in paths
                assert f"/v1/results/{label}/events" in paths
    finally:
        runtime.stop()
