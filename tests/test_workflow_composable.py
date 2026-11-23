"""Composable Canvas modules: port contract, Bbox → object presence, templates."""

from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from goodvision_harness import object_presence, vlm_independent
from goodvision_harness.object_presence import (
    PRESETS,
    ObjectPresenceManager,
    person_crop_box,
    synthetic_post,
)
from goodvision_harness.workflow_modules import CONTRACT, module_catalog
from goodvision_harness.workflow_templates import (
    TemplateStore,
    builtin_templates,
    strip_deployment,
)
from goodvision_harness.workflows import (
    NODE_PORTS,
    WorkflowValidationError,
    compile_graph,
    validate_graph,
)


def edge(source, port, target, to_port=None):
    return {
        "id": f"{source}-{port}-{target}", "from_node": source, "from_port": port,
        "to_node": target, "to_port": to_port or port,
    }


def base_graph(*, tracking="bbox", wire_tracks=True, tracks_to_output=False):
    nodes = [
        {"id": "source", "type": "source", "config": {"camera_id": "camera01"}},
        {"id": "track", "type": "tracking", "config": {"engine": tracking}},
        {"id": "objects", "type": "objects",
         "config": {"objects": [PRESETS["knife"]]}},
        {"id": "out", "type": "output"},
    ]
    edges = [
        edge("source", "frames", "track"),
        edge("source", "frames", "objects"),
        edge("objects", "objects", "out"),
    ]
    if wire_tracks:
        edges.append(edge("track", "tracks", "objects"))
    if tracks_to_output:
        edges.append(edge("track", "tracks", "out"))
    return {"nodes": nodes, "edges": edges}


# -- contract ----------------------------------------------------------------
def test_existing_port_table_is_unchanged_except_the_optional_objects_input():
    assert NODE_PORTS["objects"]["inputs"] == ("frames", "tracks")
    legacy = {
        "source": ((), ("frames",)),
        "face": (("frames",), ("faces",)),
        "tracking": (("frames",), ("tracks",)),
        "events": (("frames",), ("events",)),
        "summary": (("frames",), ("summary",)),
        "output": (
            ("frames", "tracks", "faces", "events", "summary", "objects"), (),
        ),
    }
    for kind, (inputs, outputs) in legacy.items():
        assert NODE_PORTS[kind] == {"inputs": inputs, "outputs": outputs}
    assert set(CONTRACT) == set(NODE_PORTS)


def test_catalog_marks_required_optional_and_accepted_inputs():
    catalog = module_catalog()
    json.dumps(catalog)  # served as JSON
    inputs = {i["port"]: i for i in catalog["modules"]["objects"]["inputs"]}
    assert inputs["frames"]["required"] is True
    assert inputs["tracks"] == {
        "port": "tracks", "type": "tracks", "required": False,
        "accepts": [["tracking", "bbox"]],
    }
    assert catalog["modules"]["objects"]["debug"] == "records"


def test_unwired_object_plan_keeps_its_previous_shape():
    plain = compile_graph(base_graph(wire_tracks=False, tracks_to_output=True),
                          camera_id="camera01")
    assert set(plain["objects"]) == {
        "id", "objects", "sample_fps", "clear_after", "window_frames",
        "frame_step_seconds",
    }
    wired = compile_graph(base_graph(tracks_to_output=True), camera_id="camera01")
    extra = wired["objects"]
    assert (extra.pop("focus"), extra.pop("tracks_node_id")) == ("person_crop", "track")
    assert wired == plain


def test_contour_tracks_cannot_feed_object_presence():
    with pytest.raises(WorkflowValidationError, match="does not accept"):
        validate_graph(base_graph(tracking="contour"))


def test_bbox_feeding_only_object_presence_still_runs():
    plan = compile_graph(base_graph(), camera_id="camera01")
    assert plan["tracking_enabled"] is True
    assert plan["tracking_engine"] == "bbox"
    assert plan["tracking_node_id"] == "track"
    assert plan["objects"]["focus"] == "person_crop"
    # Tracks are consumed by the objects node, not published to the output.
    assert plan["outputs"] == [
        {"node_id": "out", "inputs": [{"node_id": "objects", "port": "objects"}]}
    ]


def test_gated_off_bbox_falls_back_to_full_frames():
    plan = compile_graph(base_graph(), camera_id="camera01",
                         block_enabled={"track": False})
    assert plan["tracking_enabled"] is False
    assert "focus" not in plan["objects"]


def test_apply_requires_frames_even_when_tracks_are_wired():
    value = base_graph()
    value["edges"] = [e for e in value["edges"] if e["id"] != "source-frames-objects"]
    validate_graph(value)  # a draft may be incomplete
    with pytest.raises(WorkflowValidationError, match="frames input"):
        validate_graph(value, for_apply=True)


# -- person crop -------------------------------------------------------------
def test_person_crop_box_pads_widens_and_clamps():
    assert person_crop_box([]) is None
    # Union of two people, padded by a quarter on each side; a padded side
    # beyond the frame shifts back and clamps.
    assert person_crop_box([(0.2, 0.2, 0.4, 0.8), (0.5, 0.3, 0.6, 0.9)]) == (
        0.1, 0.0, 0.7, 1.0,
    )
    assert person_crop_box([(0.25, 0.2, 0.5, 0.8)]) == (0.1875, 0.05, 0.5625, 0.95)
    # A small box is widened to the minimum side around its centre.
    x1, y1, x2, y2 = person_crop_box([(0.5, 0.5, 0.52, 0.52)])
    assert x2 - x1 == pytest.approx(0.35) and y2 - y1 == pytest.approx(0.35)
    # Near an edge the crop shifts inside the frame instead of shrinking.
    x1, _, x2, _ = person_crop_box([(0.0, 0.4, 0.02, 0.6)])
    assert x1 == 0.0 and x2 == pytest.approx(0.35)


def _jpeg(width=400, height=200):
    out = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 30)).save(out, format="JPEG")
    return out.getvalue()


def _manager(boxes, *, focus="person_crop"):
    frames = iter([(1, 1.0, _jpeg()), (2, 2.0, _jpeg())])
    bodies = []
    post = synthetic_post()

    def record(body):
        bodies.append(body)
        return post(body)

    spec = {"id": "objects", "objects": [PRESETS["knife"]], "focus": focus}
    manager = ObjectPresenceManager(
        spec, frame_provider=lambda: next(frames), post=record,
        stream_session_id="s", boxes_provider=boxes,
    )
    manager.tick()
    manager.tick()
    return manager, bodies


def test_person_crop_sends_the_same_crop_for_every_window_frame():
    seen = []

    def boxes(sequence, timestamp):
        seen.append((sequence, timestamp))
        return {"alignment": "exact", "boxes": [(0.25, 0.2, 0.5, 0.8)]}

    manager, bodies = _manager(boxes)
    assert seen == [(1, 1.0), (2, 2.0)]
    sizes = {
        Image.open(io.BytesIO(base64.b64decode(image))).size
        for image in bodies[0]["images"]
    }
    assert sizes == {(150, 180)}  # 0.375 x 0.9 of the 400 x 200 frame
    focus = manager.latest()["focus"]
    assert focus["mode"] == "person_crop" and focus["boxes"] == 2
    assert focus["alignment"] == ["exact", "exact"]
    assert manager.tick_log()[-1]["focus"] == focus
    assert manager.status()["focus"] == "person_crop"


def test_no_person_keeps_the_full_frame_and_says_so():
    manager, bodies = _manager(lambda *_: {"alignment": "none", "boxes": []})
    image = Image.open(io.BytesIO(base64.b64decode(bodies[0]["images"][0])))
    assert image.size == (400, 200)
    assert manager.latest()["focus"]["mode"] == "full_frame_no_person"


def test_box_lookup_failure_never_stops_the_judgment():
    def broken(*_):
        raise RuntimeError("coordinator gone")

    manager, bodies = _manager(broken)
    assert bodies and manager.latest()["error"] is None
    assert manager.latest()["focus"]["alignment"] == ["error", "error"]


def test_full_frame_mode_needs_no_provider_and_person_crop_does():
    manager, _ = _manager(None, focus="full_frame")
    assert manager.latest()["focus"]["mode"] == "full_frame"
    with pytest.raises(ValueError, match="box provider"):
        ObjectPresenceManager(
            {"id": "o", "objects": [PRESETS["knife"]], "focus": "person_crop"},
            frame_provider=lambda: None, post=synthetic_post(),
            stream_session_id="s",
        )


def test_rich_label_boxes_aligns_by_source_frame():
    from goodvision_harness.contracts import NormalizedBBox
    from goodvision_harness.rich_labeling import (
        RichLabelObject,
        RichLabelSelection,
        RichLabelSnapshot,
    )

    calls = []

    class Coordinator:
        def select(self, session, timestamp, *, frame_sequence=None):
            calls.append((session, timestamp, frame_sequence))
            snapshot = RichLabelSnapshot("sess", 7, 3.5, (
                RichLabelObject("1", "person", NormalizedBBox(0.1, 0.2, 0.3, 0.4)),
            ))
            return RichLabelSelection(snapshot, "exact")

    provider = object_presence.rich_label_boxes(lambda: Coordinator(), "sess")
    assert provider(7, 3.5) == {"alignment": "exact",
                                "boxes": [(0.1, 0.2, 0.3, 0.4)]}
    assert calls == [("sess", 3.5, 7)]
    missing = object_presence.rich_label_boxes(lambda: None, "sess")
    assert missing(7, 3.5) == {"alignment": "none", "boxes": []}


# -- templates ---------------------------------------------------------------
BUILTIN_IDS = [
    "builtin-person-events", "builtin-item-check", "builtin-summary-patrol",
]


def _bound(template):
    graph = copy.deepcopy(template["graph"])
    assert all("camera_id" not in n["config"] for n in graph["nodes"])
    graph["nodes"][0]["config"]["camera_id"] = "camera01"
    return compile_graph(graph, camera_id="camera01")


def test_builtin_templates_cover_project_types_and_compile_once_bound():
    templates = {t["id"]: t for t in builtin_templates()}
    assert list(templates) == BUILTIN_IDS
    assert all(t["builtin"] for t in templates.values())
    people = _bound(templates["builtin-person-events"])
    assert (people["tracking_engine"], people["events_engine"]) == ("bbox", "system1")
    assert "objects" not in people
    items = _bound(templates["builtin-item-check"])
    assert items["objects"]["focus"] == "person_crop"
    assert items["tracking_engine"] == "bbox" and "events_engine" not in items
    assert "实验性" in templates["builtin-item-check"]["description"]
    patrol = _bound(templates["builtin-summary-patrol"])
    assert len(patrol["summaries"]) == 1 and not patrol["tracking_enabled"]


def test_strip_deployment_removes_cameras_and_addresses():
    value = base_graph()
    value["nodes"][1]["config"].update(
        execution_mode="server", service_url="http://10.0.0.2:8092/v1/labels"
    )
    stripped = strip_deployment(value)
    assert "camera_id" not in stripped["nodes"][0]["config"]
    assert stripped["nodes"][1]["config"] == {"engine": "bbox"}
    assert value["nodes"][0]["config"]["camera_id"] == "camera01"  # not mutated


def test_saved_templates_persist_separately_and_builtins_are_protected(tmp_path):
    path = tmp_path / "workflow-templates.json"
    store = TemplateStore(path)
    saved = store.save({"name": " 我的基础盘 ", "graph": base_graph()})
    assert saved["name"] == "我的基础盘" and saved["builtin"] is False
    assert "camera_id" not in saved["graph"]["nodes"][0]["config"]
    again = TemplateStore(path)
    assert [t["id"] for t in again.list()] == [*BUILTIN_IDS, saved["id"]]
    with pytest.raises(WorkflowValidationError, match="built-in"):
        again.delete("builtin-item-check")
    with pytest.raises(WorkflowValidationError):
        again.save({"name": "bad", "graph": base_graph(tracking="contour")})
    again.delete(saved["id"])
    assert [t["id"] for t in TemplateStore(path).list()] == BUILTIN_IDS
    with pytest.raises(WorkflowValidationError, match="unknown"):
        again.delete(saved["id"])


# -- runtime: debug view and dry-run wiring -----------------------------------
from test_workflow_runtime import (  # noqa: E402
    configure,
    feed_until_judged,
    objects_graph,
    rig,  # noqa: F401  (fixture)
)
from test_workflow_runtime import graph as runtime_graph  # noqa: E402

RECORD_KEYS = {
    "at", "source_frame_sequence", "frame_times", "input", "output",
    "latency_seconds", "error", "reason", "synthetic",
}


def test_node_debug_reports_object_ticks_in_the_shared_shape(rig):  # noqa: F811
    from test_video_connection import wait_for

    runtime, connection, adapter, _clock = rig
    configure(runtime, objects_graph([PRESETS["knife"]]))
    assert runtime.workflow_node_debug("objects")["status"]["state"] == "idle"
    runtime.start_connected_analysis()
    feed_until_judged(connection, adapter, runtime._workflow_objects)
    value = runtime.workflow_node_debug("objects", limit=50)
    assert value["kind"] == "objects" and value["engine"] == "system1"
    assert value["synthetic"] is True
    assert value["status"]["state"] == "running"
    assert value["counts"]["succeeded"] >= 1
    assert all(set(item) == RECORD_KEYS for item in value["items"])
    judged = [item for item in value["items"] if item["reason"] == "valid"]
    assert judged and judged[-1]["output"]["objects"][0]["id"] == "knife"
    assert judged[-1]["input"]["focus"]["mode"] == "full_frame"
    assert runtime.workflow_node_debug("missing")["status"]["state"] == "not_in_plan"
    with pytest.raises(ValueError):
        runtime.workflow_node_debug("objects", limit=0)
    runtime.stop_session()
    wait_for(lambda: not runtime._processing_lifecycle_busy())


def test_node_debug_reports_summary_history(rig):  # noqa: F811
    from test_video_connection import wait_for

    runtime, _connection, _adapter, clock = rig
    configure(runtime, runtime_graph("summary"))
    runtime.start_connected_analysis()
    manager = runtime._workflow_summaries
    clock.advance(manager, 1.1)
    wait_for(lambda: manager.latest("summary") is not None)
    value = runtime.workflow_node_debug("summary")
    assert value["kind"] == "summary" and value["items"]
    item = value["items"][-1]
    assert set(item) == RECORD_KEYS and item["synthetic"] is True
    assert item["output"]["summary"]
    runtime.stop_session()
    wait_for(lambda: not runtime._processing_lifecycle_busy())


def test_dry_run_person_crop_uses_a_synthetic_person_box(rig, monkeypatch):  # noqa: F811
    runtime, _connection, _adapter, _clock = rig
    captured = {}

    class Recorder:
        def __init__(self, spec, **options):
            captured.update(spec=spec, **options)

        def start(self):
            pass

    monkeypatch.setattr(object_presence, "ObjectPresenceManager", Recorder)
    runtime._workflow_plan = {"objects": {
        "id": "objects", "objects": [PRESETS["knife"]], "focus": "person_crop",
        "tracks_node_id": "track",
    }, "objects_service_url": "http://127.0.0.1:8097"}
    runtime._start_workflow_objects(runtime._video_connection, session_id="s")
    assert captured["spec"]["focus"] == "person_crop"
    assert captured["boxes_provider"] is object_presence.synthetic_boxes
    runtime._workflow_objects = None
    runtime._workflow_plan = None


# -- routes ------------------------------------------------------------------
def test_template_routes_and_module_catalog(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from test_workflow_routes import workflow_app

    app = workflow_app(monkeypatch, tmp_path)
    with TestClient(app) as client:
        data = client.get("/v1/workflows").json()
        assert data["modules"] == module_catalog()
        listed = client.get("/v1/workflow-templates")
        assert listed.status_code == 200
        assert listed.headers["cache-control"] == "no-store"
        assert [t["id"] for t in listed.json()["templates"]] == BUILTIN_IDS
        saved = client.post(
            "/v1/workflow-templates", json={"name": "Store", "graph": base_graph()}
        )
        assert saved.status_code == 200
        template_id = saved.json()["id"]
        assert (tmp_path / "workflow-templates.json").exists()
        # Templates never enter the workflow document or its revision.
        assert client.get("/v1/workflows").json()["document"]["revision"] == 0
        bad = client.post("/v1/workflow-templates", json={"name": "x", "extra": 1})
        assert bad.status_code == 400
        assert client.delete(
            "/v1/workflow-templates/builtin-person-events"
        ).status_code == 400
        assert client.delete(f"/v1/workflow-templates/{template_id}").status_code == 204
        assert len(client.get("/v1/workflow-templates").json()["templates"]) == 3


def test_debug_route_is_on_the_fleet_allowlist():
    from goodvision_harness.fleet_outputs import _OUTPUT_RE, OUTPUT_PATHS

    assert "/workflow/nodes/{node_id}/debug" in OUTPUT_PATHS
    assert _OUTPUT_RE.match("/v1/workflow/nodes/objects-1/debug?limit=20")
    assert not _OUTPUT_RE.match("/v1/workflow/nodes/a/b/debug")


# -- engine catalog: Decision 3 ----------------------------------------------
def _events_graph(engine, **extra):
    return {
        "nodes": [
            {"id": "source", "type": "source", "config": {"camera_id": "camera01"}},
            {"id": "events", "type": "events",
             "config": {"engine": engine, "features": ["smoking"], **extra}},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            edge("source", "frames", "events"), edge("events", "events", "out"),
        ],
    }


def test_catalog_lists_switchable_engines_per_slot():
    modules = module_catalog()["modules"]
    events = {e["id"]: e for e in modules["events"]["engines"]}
    assert list(events) == ["vlm", "system1", "decision3", "neural"]
    assert events["decision3"] == {
        "id": "decision3", "name": "Decision 3 recognition",
        "model": "vllm-sr/d3-nano", "local": True,
    }
    # Decision 3 needs its own object request/response adapter first.
    assert [e["id"] for e in modules["objects"]["engines"]] == ["system1"]
    assert modules["summary"]["engines"] == []
    assert modules["objects"]["task"] is True and modules["events"]["task"] is False


def test_system1_and_decision3_are_separate_peer_modules():
    peers = {p["id"]: p for p in module_catalog()["peers"]}
    assert list(peers) == ["contour", "bbox", "vlm", "system1", "decision3", "neural"]
    assert peers["system1"]["model"] == "ZhuYuhan/VideoJev-4B"
    assert peers["decision3"]["model"] == "vllm-sr/d3-nano"
    assert (peers["system1"]["local_port"], peers["decision3"]["local_port"]) == (
        8097, 8099,
    )
    assert peers["decision3"]["output_contract"] == "native_option_probabilities"
    assert peers["neural"]["output_contract"] == "embedding_margin"
    assert peers["vlm"]["output_contract"] == "generative_json"
    assert peers["bbox"]["layer"] == "perception"
    # Both may sit on one canvas as separate blocks; Apply still runs one
    # enabled event module per camera (single native event lane).
    value = _events_graph("system1")
    value["nodes"].append({"id": "d3", "type": "events", "config": {
        "engine": "decision3", "features": ["smoking"]}})
    value["edges"] += [edge("source", "frames", "d3"), edge("d3", "events", "out")]
    validate_graph(value)
    with pytest.raises(WorkflowValidationError, match="only one events"):
        validate_graph(value, for_apply=True)


def test_decision3_module_compiles_like_system1_on_its_own_service():
    system1 = compile_graph(_events_graph("system1"), camera_id="camera01")
    decision3 = compile_graph(_events_graph("decision3"), camera_id="camera01")
    assert decision3["events_engine"] == "decision3"
    assert decision3["events_service_url"] == "http://127.0.0.1:8099"
    assert decision3["events_execution_mode"] == "local"
    for key in ("features", "event_sensitivities", "outputs", "events_node_id"):
        assert decision3[key] == system1[key]


def test_objects_node_does_not_offer_decision3_yet():
    value = base_graph()
    value["nodes"][2]["config"]["engine"] = "decision3"
    with pytest.raises(WorkflowValidationError, match="unsupported workflow engine"):
        validate_graph(value)


def _two_camera_graph():
    """Objects takes frames from camera01; Bbox runs on camera02's source."""
    value = base_graph()
    value["nodes"].insert(1, {"id": "source2", "type": "source",
                              "config": {"camera_id": "camera02"}})
    value["edges"] = [
        e for e in value["edges"] if e["id"] != "source-frames-track"
    ] + [edge("source2", "frames", "track"), edge("track", "tracks", "out")]
    return value


def test_optional_tracks_cannot_stand_in_for_this_cameras_frames():
    value = _two_camera_graph()
    validate_graph(value, for_apply=True)
    other = compile_graph(value, camera_id="camera02")
    assert "objects" not in other and other["tracking_engine"] == "bbox"
    own = compile_graph(value, camera_id="camera01")
    assert own["objects"]["id"] == "objects"
    assert "focus" not in own["objects"]  # camera02's boxes never crop camera01
    assert own["tracking_enabled"] is False


def test_multi_camera_graph_saves_as_a_template_with_unbound_slots(tmp_path):
    store = TemplateStore(tmp_path / "t.json")
    saved = store.save({"name": "Two cameras", "graph": _two_camera_graph()})
    sources = [n for n in saved["graph"]["nodes"] if n["type"] == "source"]
    assert len(sources) == 2
    assert all("camera_id" not in n["config"] for n in sources)
    assert TemplateStore(tmp_path / "t.json").list()[-1]["graph"] == saved["graph"]


def test_decision3_uses_the_native_event_lane_and_its_own_model(rig):  # noqa: F811
    runtime = rig[0]
    configure(runtime, _events_graph(
        "decision3", execution_mode="server", service_url="http://10.0.0.9:8099",
    ))
    assert runtime._workflow_plan["events_engine"] == "decision3"
    configuration = runtime._workflow_event_configuration()
    assert configuration.inference.model == "vllm-sr/d3-nano"
    assert configuration.aiclub.endpoint == "http://10.0.0.9:8099/predict"
    # The independent lane picks its inspector by this model ID.
    source = Path(vlm_independent.__file__).read_text(encoding="utf-8")
    assert 'config.model == "vllm-sr/d3-nano"' in source
    assert "Decision3Inspector" in source


def test_local_decision3_module_runs_the_videojev_service_runtime(
    monkeypatch, tmp_path
):
    import sys

    from goodvision_harness import module_service, videojev_service

    seen = {}
    monkeypatch.setattr(videojev_service, "main", lambda: seen.update(argv=sys.argv))
    monkeypatch.setenv("GOODVISION_MODULE_MODEL_DIR", str(tmp_path / "models"))
    module_service.serve("decision3", "127.0.0.1", 8099, tmp_path / "data")
    argv = seen["argv"]
    assert argv[1:3] == ["--runtime", "decision3"]
    assert argv[argv.index("--port") + 1] == "8099"
    assert argv[argv.index("--result-log") + 1].endswith("decision3-results.jsonl")
