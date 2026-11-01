"""Engine selection, service isolation and output composition contracts."""

import copy
import io
from types import SimpleNamespace

import pytest
from PIL import Image
from test_workflow_runtime import configure, graph
from test_workflow_runtime import rig as rig
from test_workflow_tracking import tracking_graph

from goodvision_harness.model_api import MissingAPIKeyError
from goodvision_harness.workflow_modules import MODELS
from goodvision_harness.workflow_render import render_output
from goodvision_harness.workflows import compile_graph, validate_graph


def event_graph(engine):
    value = graph("events")
    value["nodes"][-1]["config"].update(
        engine=engine, service_url="http://127.0.0.1:8094"
    )
    return value


def test_alternative_engines_keep_connections_but_only_selected_engine_compiles():
    value = event_graph("system1")
    alternative = copy.deepcopy(value["nodes"][-1])
    alternative.update(id="neural", enabled=False)
    alternative["config"]["engine"] = "neural"
    value["nodes"].append(alternative)
    for edge in list(value["edges"]):
        other = {
            k: ("neural" if v == "events" and k in {"from_node", "to_node"} else v)
            for k, v in edge.items()
        }
        other["id"] += "-neural"
        value["edges"].append(other)
    first = compile_graph(value, camera_id="camera01")
    assert first["events_engine"] == "system1"
    value["nodes"][-2]["enabled"] = False
    value["nodes"][-1]["enabled"] = True
    second = compile_graph(value, camera_id="camera01")
    assert second["events_engine"] == "neural"
    assert second["outputs"][0]["inputs"] == [{"port": "events", "node_id": "neural"}]
    value["nodes"][-2]["enabled"] = True
    with pytest.raises(ValueError, match="only one events"):
        compile_graph(value, camera_id="camera01")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://host",
        "http://u:p@host",
        "http://host?q=1",
        "http://host/#token",
        "http://host:99999",
        "http://host\\path",
    ],
)
def test_module_rejects_unsafe_or_malformed_endpoints(url):
    value = event_graph("system1")
    value["nodes"][-1]["config"]["service_url"] = url
    with pytest.raises(ValueError):
        validate_graph(value)


def test_draft_service_is_optional_but_apply_requires_it():
    value = event_graph("neural")
    value["nodes"][-1]["config"]["service_url"] = ""
    validate_graph(value)
    with pytest.raises(ValueError, match="service URL"):
        compile_graph(value, camera_id="camera01")
    value["nodes"][-1]["config"]["event_sensitivities"] = {"smoking": "high"}
    with pytest.raises(ValueError, match="fixed visual prompts"):
        validate_graph(value)


@pytest.mark.parametrize("engine", ["system1", "neural"])
def test_event_route_switch_restores_baseline_and_keeps_summary_service(rig, engine):
    runtime, _, _, _ = rig
    baseline = runtime.config
    configure(runtime, event_graph(engine))
    assert runtime._workflow_event_configuration().inference.model == MODELS[engine]
    assert runtime._workflow_event_configuration().model_api.provider == "local"
    assert (
        runtime._workflow_event_configuration().model_api.endpoint
        == "http://127.0.0.1:8094/predict"
    )
    assert runtime.config.inference == baseline.inference
    assert runtime.config.model_api == baseline.model_api
    assert runtime.config.runtime.scene_independent_vlm
    assert runtime.config.runtime.scene_vlm_output_mode == "events"
    assert runtime._workflow_summary_configuration().inference == baseline.inference
    assert runtime._workflow_summary_configuration().model_api == baseline.model_api
    configure(runtime, graph("events"))
    assert runtime.config.inference == baseline.inference
    assert runtime.config.model_api == baseline.model_api
    assert (
        runtime.config.runtime.scene_independent_vlm
        == baseline.runtime.scene_independent_vlm
    )
    runtime.configure_workflow({"binding": None})
    assert runtime.config == baseline


def test_custom_events_do_not_bypass_summary_credentials(rig):
    runtime, _, _, _ = rig
    value = graph("events", "summary")
    value["nodes"][2]["config"].update(
        engine="system1", service_url="http://local.test"
    )
    configure(runtime, value)
    runtime.dry_run = False
    runtime._aiclub_api_key = None
    with pytest.raises(MissingAPIKeyError):
        runtime.prepare_analysis()


def test_draft_native_events_probe_selected_route_without_vlm_inference(
    rig, monkeypatch
):
    runtime, _, _, _ = rig
    seen = []
    monkeypatch.setattr(
        runtime, "_workflow_event_probe", lambda plan: seen.append(plan)
    )
    monkeypatch.setattr(
        "goodvision_harness.grounding.build_selected_features_grounding_client",
        lambda *a, **kw: pytest.fail("wrong event adapter"),
    )
    baseline = runtime.config
    result = runtime.test_workflow(event_graph("neural"))["results"][0]
    assert result["state"] == "live_analysis_required"
    assert result["observations"] == []
    assert seen[0]["events_engine"] == "neural"
    assert runtime.config is baseline and runtime._workflow_plan is None


@pytest.mark.parametrize("engine", ["system1", "neural"])
def test_live_start_routes_native_events_without_changing_shared_model(
    rig, monkeypatch, engine
):
    from test_vlm_independent_service import LocalBranch

    from goodvision_harness import service

    runtime, _, _, _ = rig
    shared = runtime.config.inference
    monkeypatch.setattr(service, "IndependentVLMBranch", LocalBranch)
    monkeypatch.setattr(runtime, "_workflow_event_probe", lambda plan: None)
    configure(runtime, event_graph(engine))
    runtime.dry_run = False
    runtime._aiclub_api_key = None
    runtime.start_connected_analysis()
    assert runtime._vlm_branch.config.model == MODELS[engine]
    assert runtime._vlm_branch.config.model_settings.endpoint.endswith("/predict")
    assert runtime.config.inference == shared


def test_bbox_route_checks_its_detector_and_restores_contour_baseline(rig, monkeypatch):
    runtime, _, _, _ = rig
    seen = []
    probe = SimpleNamespace(gpu_ready=lambda: True, close=lambda: None)
    monkeypatch.setattr(
        runtime,
        "_create_rich_http_client",
        lambda rich, **kw: seen.append((rich, kw)) or probe,
    )
    # Coordinator construction is independent of this route-selection check.
    monkeypatch.setattr(runtime, "_try_enable_rich_labels", lambda: True)
    baseline = runtime.config
    value = tracking_graph()
    value["nodes"][-1]["config"] = {
        "engine": "bbox",
        "service_url": "http://detector.test/v1/labels",
    }
    configure(runtime, value)
    assert runtime.config.rich_labeling.backend == "paddle"
    assert (
        runtime._tracking_requested
        and runtime.config.rich_labeling.processing_method is None
    )
    assert runtime.prepare_analysis()["prepared"]
    assert seen[0][0].endpoint == "http://detector.test/v1/labels"
    assert list(seen[0][1]["headers"].values()) == ["yolo26-s"]
    runtime.configure_workflow({"binding": None})
    assert runtime.config == baseline


def jpeg():
    stream = io.BytesIO()
    Image.new("RGB", (640, 360), "white").save(stream, "JPEG")
    return stream.getvalue()


def pixels(output, timestamp=10):
    return Image.open(io.BytesIO(render_output(jpeg(), output, timestamp)))


def test_output_display_roundtrips_and_positions_wired_current_layers():
    value = graph("events", "summary")
    display = {
        "events": {"position": "bottom-right"},
        "summary": {"position": "top-left"},
        "tracks": {"visible": False},
    }
    value["nodes"][1]["config"] = {"display": display}
    plan = compile_graph(value, camera_id="camera01")
    assert plan["outputs"][0]["display"]["tracks"]["visible"] is False
    output = {
        "display": display,
        "inputs": [
            {
                "port": "events",
                "value": {
                    "results": [
                        {
                            "frame": {"timestamp_seconds": 10},
                            "judgments": [{"event_label": "smoking", "detected": True}],
                        }
                    ]
                },
            },
            {
                "port": "summary",
                "value": {
                    "timestamp_seconds": 10,
                    "summary": "Two people walking",
                    "summary_translations": {"zh": "两个人正在行走"},
                },
            },
        ],
    }
    image = pixels(output)
    red = image.getpixel((400, 320))
    navy = image.getpixel((30, 45))
    assert red[0] > 140 and red[1] < 90
    assert navy[0] < 90 and navy[2] < 140
    output["display"] = {"events": {"visible": False}, "summary": {"visible": False}}
    assert min(pixels(output).getpixel((400, 320))) > 240
    output["display"] = {}
    assert min(pixels(output, timestamp=200).getpixel((30, 25))) > 240
    output["inputs"] = []
    assert min(pixels(output).getpixel((30, 25))) > 240


def test_bbox_overlay_rejects_geometry_from_a_different_frame():
    output = {
        "inputs": [
            {
                "port": "tracks",
                "value": {
                    "timestamp_seconds": 10,
                    "objects": [
                        {"bbox": {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}}
                    ],
                },
            }
        ]
    }
    green = pixels(output).getpixel((160, 120))
    assert green[1] > green[0] + 50
    assert min(pixels(output, timestamp=10.2).getpixel((160, 120))) > 240


def test_output_preview_is_scoped_to_active_output_and_stops_with_session(rig):
    runtime, _, _, _ = rig
    configure(runtime, graph(frames=True))
    with pytest.raises(ValueError, match="Start"):
        runtime.workflow_output_preview("out")
    runtime.start_connected_analysis()
    assert runtime.workflow_output_preview("out").startswith(b"\xff\xd8")
    with pytest.raises(ValueError, match="not active"):
        runtime.workflow_output_preview("other")
    runtime.stop_session()
    with pytest.raises(ValueError):
        runtime.workflow_output_preview("out")


def test_summary_output_remains_live_after_native_analysis_ends(rig):
    from test_video_connection import wait_for

    runtime, _, _, _ = rig
    configure(runtime, graph("face", "summary"))
    runtime.start_connected_analysis()
    native = runtime._session
    native.stop(drain=True)
    native.join(1)
    wait_for(lambda: runtime._session is None)
    assert runtime._workflow_summaries.is_running
    assert runtime.workflow_output_preview("out").startswith(b"\xff\xd8")


def test_output_http_route_is_no_store_and_rejects_unknown_outputs(rig):
    from fastapi.testclient import TestClient

    from goodvision_harness.service import create_app

    runtime, _, _, _ = rig
    configure(runtime, graph(frames=True))
    runtime.start_connected_analysis()
    app = create_app(lambda: runtime)
    app.state.runtime = runtime
    client = TestClient(app)
    try:
        response = client.get("/v1/workflow/output.jpg?node_id=out&language=en")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "no-store"
        assert client.get("/v1/workflow/output.jpg?node_id=unknown").status_code == 404
        assert (
            client.get(
                "/v1/workflow/output.jpg?node_id=out&workflow_id=other"
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/v1/workflow/output.jpg?node_id=out&workflow_id=recipe"
            ).status_code
            == 200
        )
    finally:
        client.close()


@pytest.mark.parametrize(
    "session, timestamp, expected",
    [
        ("current", 10, (True, "ready")),
        ("current", 10.0004, (True, "ready")),
        ("previous", 10, (False, "waiting_for_tracking")),
        ("current", 1, (False, "tracking_stale")),
        ("current", 11, (False, "tracking_stale")),
        (None, 10, (False, "waiting_for_tracking")),
    ],
)
def test_bbox_only_readiness_uses_current_session_snapshots(
    session, timestamp, expected
):
    import time
    from threading import RLock

    from goodvision_harness.rich_labeling import RichLabelSnapshot
    from goodvision_harness.service import HarnessRuntime

    snapshot = RichLabelSnapshot(session, 1, timestamp, ()) if session else None
    runtime = SimpleNamespace(
        _state_lock=RLock(),
        _state={"pipeline_state": "running", "last_media_timestamp": 10},
        _last_frame_monotonic=time.monotonic(),
        _last_result_monotonic=None,
        _session=SimpleNamespace(running=True, stream_session_id="current"),
        _workflow_plan={
            "tracking_enabled": True,
            "tracking_engine": "bbox",
            "features": [],
        },
        _rich_labels=SimpleNamespace(latest=lambda session_id: snapshot),
        settings=SimpleNamespace(
            ready_max_frame_age_seconds=3, ready_max_result_age_seconds=5
        ),
    )
    assert HarnessRuntime.readiness(runtime) == expected
