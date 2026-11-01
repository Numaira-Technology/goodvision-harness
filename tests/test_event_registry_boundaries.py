"""Custom events must survive real persistence, rendering and workflow boundaries."""

from __future__ import annotations

import copy
import io
import json
from dataclasses import replace
from threading import Event

import pytest
from PIL import Image
from test_event_injection import _injected
from test_scene_event_archive import RecordingEncoder, finish, frames
from test_workflows import graph

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.event_archive import EventArchive, EventArchiveSettings
from goodvision_harness.event_registry import (
    BUILTIN_VOCABULARY,
    MAX_INJECTED_EVENTS,
    vocabulary_from_json,
)
from goodvision_harness.scene_archive import SceneArchive
from goodvision_harness.scene_delivery import (
    SceneDisplayWorker,
    _display_events,
    _event_lines,
    render_scene_jpeg,
)
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.scene_state import SceneState
from goodvision_harness.workflows import (
    WorkflowStore,
    WorkflowValidationError,
    compile_document,
    compile_graph,
    validate_graph,
)


def test_events_lane_budget_and_response_cover_maximum_custom_vocabulary(monkeypatch):
    import httpx
    from test_vlm_independent import config, job

    from goodvision_harness import scene_model, vlm_independent

    vocabulary = vocabulary_from_json(
        [
            {
                "label": f"event_{index:02d}_" + "x" * 39,
                "zh_name": f"事件{index}",
                "definition": "A person opens a door.",
            }
            for index in range(MAX_INJECTED_EVENTS)
        ]
    )
    labels = tuple(d.label for d in vocabulary.definitions if d.branch == "safety")
    replies = {label: False for label in labels}
    response_text = json.dumps(replies, separators=(",", ":"))
    requests = []

    async def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": response_text,
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(
        scene_model,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    cfg = replace(
        config(),
        labels=labels,
        output_mode="events",
        vocabulary=vocabulary,
        api_key="TEST_KEY",
    )
    inspector = vlm_independent._inspector(cfg)
    try:
        votes = inspector.inspect(replace(job(cfg, 1.2, 18), vocabulary=vocabulary))
    finally:
        inspector.close()
    assert {vote.event_label for vote in votes} == set(labels)
    assert all(vote.detected is False for vote in votes)
    # Even one token per ASCII character fits, including all six built-ins.
    assert requests[0]["max_tokens"] >= len(response_text)

    builtin = replace(cfg, labels=("smoking", "fall"), vocabulary=BUILTIN_VOCABULARY)
    inspector = vlm_independent._inspector(builtin)
    try:
        assert inspector._parameters["max_tokens"] == 160
    finally:
        inspector.close()


def confirmed_snapshot():
    state = SceneState("session", "camera", vocabulary=_injected())
    for sequence in (1, 2):
        state.observe_person(
            "person-1",
            timestamp_seconds=float(sequence),
            frame_sequence=sequence,
            bbox=NormalizedBBox(0.2, 0.3, 0.7, 0.9),
            tracking_confirmation="confirmed",
        )
        for label in ("fight", "opens_door"):
            state.observe_event(
                label,
                ("person-1",),
                timestamp_seconds=float(sequence),
                frame_sequence=sequence,
                detected=True,
                confidence=0.95,
                evidence_id=f"{label}-{sequence}",
            )
    snapshot = state.snapshot()
    assert {event["phase"] for event in snapshot["events"]} == {"confirmed"}
    return snapshot


def test_mixed_events_persist_and_reload_with_only_allowlisted_metadata(tmp_path):
    snapshot = confirmed_snapshot()
    snapshot["events"][0]["secret"] = "must not persist"
    archive = SceneArchive(tmp_path, vocabulary=_injected())
    try:
        assert archive.submit(snapshot, raise_on_invalid=True)
        assert archive.flush(3)
        assert {item["event_label"] for item in archive.list_events()} == {
            "fight",
            "opens_door",
        }
    finally:
        archive.close()
    reopened = SceneArchive(tmp_path, vocabulary=_injected())
    try:
        rows = reopened.list_events()
        assert {item["event_label"] for item in rows} == {"fight", "opens_door"}
        assert all("secret" not in item and "history" not in item for item in rows)
        # A later correction to a custom event must survive the same validator.
        event = next(
            item for item in snapshot["events"] if item["event_label"] == "opens_door"
        )
        corrected = {
            **snapshot,
            "events": [],
            "transitions": [
                {
                    **event,
                    "type": "retract",
                    "timestamp_seconds": 3,
                }
            ],
        }
        assert reopened.submit(corrected, raise_on_invalid=True)
        assert reopened.flush(3)
        assert (
            next(
                item
                for item in reopened.list_events()
                if item["event_label"] == "opens_door"
            )["phase"]
            == "retracted"
        )
    finally:
        reopened.close()


def test_permanent_rejection_is_distinct_from_transient_queue_pressure(tmp_path):
    snapshot = confirmed_snapshot()
    archive = SceneArchive(tmp_path / "default")
    try:
        assert not archive.submit(snapshot)
        with pytest.raises(ValueError, match="invalid scene archive snapshot"):
            archive.submit(snapshot, raise_on_invalid=True)
        assert archive.stats()["invalid_snapshots"] == 2
    finally:
        archive.close()
    archive = SceneArchive(tmp_path / "custom", vocabulary=_injected(), queue_size=1)
    entered, release = Event(), Event()
    original = archive._write

    def blocked(connection, records):
        entered.set()
        assert release.wait(3)
        original(connection, records)

    archive._write = blocked
    try:
        assert archive.submit(snapshot, raise_on_invalid=True)
        assert entered.wait(2)
        assert archive.submit(snapshot, raise_on_invalid=True)
        assert not archive.submit(snapshot, raise_on_invalid=True)
        assert archive.stats()["invalid_snapshots"] == 0
    finally:
        release.set()
        archive.close(3)


def test_custom_and_builtin_video_evidence_survives_restart(tmp_path):
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path),
        encoder=RecordingEncoder(),
        clock=lambda: 0.0,
        vocabulary=_injected(),
    )
    try:
        frames(archive)
        assert archive.submit_scene(confirmed_snapshot())
        records = finish(archive)
        assert {item["event_label"] for item in records} == {"fight", "opens_door"}
        assert all(item["evidence"]["status"] == "ready" for item in records)
        assert all(archive.video_path(item["event_id"]).is_file() for item in records)
    finally:
        archive.close()
    reopened = EventArchive(
        EventArchiveSettings(directory=tmp_path), vocabulary=_injected()
    )
    try:
        assert {item["event_label"] for item in reopened.list_events()["events"]} == {
            "fight",
            "opens_door",
        }
    finally:
        reopened.close()


def test_custom_render_keeps_label_and_confirmed_box_color():
    snapshot = confirmed_snapshot()
    snapshot["events"] = [
        e for e in snapshot["events"] if e["event_label"] == "opens_door"
    ]
    assert _display_events(snapshot, 2) == []
    assert _display_events(snapshot, 2, _injected()) == snapshot["events"]
    assert _event_lines(snapshot, 2, {"person-1"}, _injected())[0][0].startswith(
        "opens_door: confirmed"
    )
    with Image.new("RGB", (400, 300), "black") as source, io.BytesIO() as buffer:
        source.save(buffer, "JPEG")
        encoded = render_scene_jpeg(
            buffer.getvalue(), snapshot, 2, vocabulary=_injected()
        )
    with Image.open(io.BytesIO(encoded)) as rendered:
        red, green, blue = rendered.convert("RGB").getpixel((80, 180))
        assert red > green + 50 and red > blue + 50
    # Payload text cannot add an undeclared label or control characters.
    snapshot["events"][0]["event_label"] = "opens_door\nforged"
    assert _display_events(snapshot, 2, _injected()) == []


def test_display_worker_passes_its_scene_vocabulary(monkeypatch):
    from goodvision_harness import scene_delivery
    from goodvision_harness.result_bus import FramePacket

    scene = SceneRuntime(
        "session", "camera", event_labels=("opens_door",), vocabulary=_injected()
    )
    monkeypatch.setattr(
        scene, "display_snapshot", lambda _timestamp: confirmed_snapshot()
    )
    rendered, published = [], Event()
    real = scene_delivery.render_scene_jpeg

    def capture(*args, **kwargs):
        rendered.append(kwargs.get("vocabulary"))
        return real(*args, **kwargs)

    monkeypatch.setattr(scene_delivery, "render_scene_jpeg", capture)
    with Image.new("RGB", (400, 300), "black") as source, io.BytesIO() as buffer:
        source.save(buffer, "JPEG")
        packet = FramePacket(buffer.getvalue(), 2, 2, "2026-11-01T00:00:00Z")
    worker = SceneDisplayWorker(scene, lambda _packet: published.set())
    try:
        worker.start()
        worker.submit(packet)
        assert published.wait(3), worker.stats()
        assert rendered == [scene.vocabulary]
    finally:
        worker.close()


def test_workflow_custom_selection_and_defaults_survive_store_reload(tmp_path):
    vocabulary = _injected()
    value = graph()
    value["nodes"][2]["config"] = {"features": ["opens_door", "fight"]}
    with pytest.raises(WorkflowValidationError, match="supported"):
        compile_graph(value)
    plan = compile_graph(value, vocabulary=vocabulary)
    assert plan["features"] == ["fight", "stranger", "opens_door"]
    defaults = copy.deepcopy(value)
    defaults["nodes"][2]["config"] = {}
    normalized = validate_graph(defaults, vocabulary=vocabulary)
    assert normalized["nodes"][2]["config"]["features"] == [
        "fight",
        "smoking",
        "fall",
        "weapon",
        "ppe_violation",
        "ground_litter",
    ]
    path = tmp_path / "workflow.json"
    store = WorkflowStore(path, vocabulary=vocabulary)
    candidate = store.candidate(
        "create_workflow", {"name": "Custom", "graph": value}, 0
    )
    saved = store.commit(candidate, 0)
    reloaded = WorkflowStore(path, vocabulary=vocabulary).load()
    assert reloaded == saved
    assert (
        compile_graph(reloaded["workflows"][0]["graph"], vocabulary=vocabulary) == plan
    )
    assert compile_document(reloaded, vocabulary=vocabulary) == {}


@pytest.mark.parametrize("model", ["videojev", "decision3"])
def test_native_inspector_templates_keep_custom_criteria(model):
    from test_vlm_independent import config

    from goodvision_harness.decision3_model import Decision3Inspector
    from goodvision_harness.videojev_model import VideoJevInspector

    inspector_type = VideoJevInspector if model == "videojev" else Decision3Inspector
    cfg = config()
    cfg = replace(
        cfg,
        labels=("fight", "opens_door"),
        vocabulary=_injected(),
        model=inspector_type.model_name,
        output_mode="events",
        model_settings=replace(
            cfg.model_settings, provider="local", endpoint="http://127.0.0.1:9000"
        ),
    )
    inspector = inspector_type(cfg)
    try:
        template = " ".join(str(row) for row in inspector._question_template)
        assert "swings a door open" in template
        assert "Standing beside a closed door" in template
        assert [row["id"] for row in inspector._question_template] == [
            "scene/fight",
            "scene/opens_door",
        ]
    finally:
        inspector.close()
