"""Canvas adapters exercised against real HarnessRuntime and video subscriptions."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import pytest
from test_video_connection import ControlledAdapter, frame, publish, wait_for
from test_workflow_summary import Clock

from goodvision_harness import service
from goodvision_harness.config import load_config
from goodvision_harness.deployment_settings import DeploymentSettingsStore
from goodvision_harness.privacy import PUBLIC_FACE_ERROR
from goodvision_harness.workflow_summary import SummaryManager

ROOT = Path(__file__).resolve().parents[1]


def graph(*kinds, frames=False):
    nodes = [{"id": "source", "type": "source"}, {"id": "out", "type": "output"}]
    edges = []
    ports = {"face": "faces", "events": "events", "summary": "summary"}
    for kind in kinds:
        config = {"features": ["smoking"]} if kind == "events" else {}
        if kind == "summary":
            config = {"interval_seconds": 1, "prompt": "Describe this picture."}
        nodes.append({"id": kind, "type": kind, "config": config})
        edges.extend(
            [
                {
                    "id": "in-" + kind,
                    "from_node": "source",
                    "from_port": "frames",
                    "to_node": kind,
                    "to_port": "frames",
                },
                {
                    "id": "out-" + kind,
                    "from_node": kind,
                    "from_port": ports[kind],
                    "to_node": "out",
                    "to_port": ports[kind],
                },
            ]
        )
    if frames:
        edges.append(
            {
                "id": "raw",
                "from_node": "source",
                "from_port": "frames",
                "to_node": "out",
                "to_port": "frames",
            }
        )
    return {"nodes": nodes, "edges": edges}


def configure(runtime, value, workflow_id="recipe"):
    return runtime.configure_workflow(
        {
            "binding": {
                "workflow_id": workflow_id,
                "group_id": "entrances",
                "graph": value,
                "enabled": True,
                "block_enabled": {},
                "camera_override": {},
            }
        }
    )


def values(runtime):
    return {
        item["port"]: item["value"]
        for output in runtime.workflow_results()["outputs"]
        for item in output["inputs"]
    }


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    monkeypatch.setenv("HARNESS_DRY_RUN", "true")
    monkeypatch.delenv("PERCEPTION_METHOD", raising=False)
    base = load_config(ROOT / "config.yaml")
    config = replace(
        base,
        rich_labeling=replace(
            base.rich_labeling,
            enabled=False,
            tracking_enabled=False,
            processing_method=None,
        ),
        runtime=replace(base.runtime, scene_mode="disabled"),
        events=replace(base.events, directory=tmp_path / "events"),
        face=replace(base.face, gallery_path=tmp_path / "gallery.json"),
    )
    runtime = service.HarnessRuntime(
        config, deployment_store=DeploymentSettingsStore(tmp_path / "settings.json")
    )
    runtime.start()
    adapter = ControlledAdapter("rtsp://camera.test/video")
    runtime.connect_stream("rtsp://camera.test/video", _adapter=adapter)
    connection = runtime._video_connection
    publish(connection, adapter, frame(1))
    wait_for(lambda: connection.latest_jpeg() is not None)
    clock = Clock()
    original = runtime._make_summary_manager

    def manager(specs, source, **options):
        value = original(specs, source, **options)
        value._clock = clock
        return value

    monkeypatch.setattr(runtime, "_make_summary_manager", manager)
    yield runtime, connection, adapter, clock
    runtime.stop()
    for worker in (runtime._workflow_summaries, runtime._workflow_test_manager):
        if worker is not None:
            worker.stop(1)


def test_summary_only_runs_without_gpu_and_locks_configuration_until_stopped(
    rig, monkeypatch
):
    runtime, connection, _adapter, clock = rig
    monkeypatch.setattr(
        runtime,
        "_require_processing_ready",
        lambda: pytest.fail("summary must not require GPU readiness"),
    )
    configure(runtime, graph("summary", frames=True))
    legacy = copy.deepcopy(runtime.analysis_settings())
    result = runtime.start_connected_analysis()
    assert result["pipeline_state"] == "running"
    assert runtime._session is None
    assert runtime._processing_lifecycle_busy()
    worker = runtime._workflow_summaries
    clock.advance(worker, 1)
    wait_for(lambda: worker.latest() is not None)
    output = values(runtime)
    assert output["summary"]["synthetic"] is True
    assert output["summary"]["source_id"] == runtime.source_id
    assert output["frames"]["jpeg_url"] == "/v1/video/source/latest.jpg"
    assert output["frames"]["source_frame_sequence"] == 1
    with pytest.raises(RuntimeError, match="stop"):
        configure(runtime, graph(frames=True))
    with pytest.raises(RuntimeError, match="stop"):
        runtime.update_analysis_settings({"features": ["stranger"]})
    assert runtime.analysis_settings() == legacy
    runtime.stop_session()
    wait_for(lambda: not runtime._processing_lifecycle_busy())
    assert runtime.status()["pipeline_state"] == "idle"
    assert connection.status()["connected"]
    configure(runtime, graph("summary"), "different-recipe")
    assert runtime.workflow_summary_results()["items"] == []
    assert values(runtime)["summary"] is None


def test_raw_source_output_has_a_logical_run_and_no_model_calls(rig, monkeypatch):
    runtime, connection, _adapter, _clock = rig
    monkeypatch.setattr(
        runtime,
        "_require_processing_ready",
        lambda: pytest.fail("raw output requires no tracking"),
    )
    monkeypatch.setattr(
        runtime,
        "_make_summary_manager",
        lambda *a, **k: pytest.fail("raw output requires no inference"),
    )
    configure(runtime, graph(frames=True))
    status = runtime.start_connected_analysis()
    session_id = status["workflow"]["stream_session_id"]
    assert session_id and status["pipeline_state"] == "running"
    assert runtime._workflow_summaries is None and runtime._session is None
    assert runtime._processing_lifecycle_busy()
    assert values(runtime)["frames"]["connection_id"] == connection.connection_id
    with pytest.raises(RuntimeError):
        runtime.start_connected_analysis()
    runtime.stop_session()
    assert not runtime._processing_lifecycle_busy()
    status = runtime.start_connected_analysis()
    assert status["workflow"]["stream_session_id"] != session_id


def test_slow_face_does_not_delay_summary_on_the_same_real_connection(rig, monkeypatch):
    runtime, connection, adapter, clock = rig
    entered, release = Event(), Event()
    original = service.build_selected_features_grounding_client

    def client(*args, **kwargs):
        value = original(*args, **kwargs)
        face = value._stranger_client
        classify = face.classify_window

        def blocked(window):
            entered.set()
            assert release.wait(3)
            return classify(window)

        face.classify_window = blocked
        return value

    monkeypatch.setattr(service, "build_selected_features_grounding_client", client)
    configure(runtime, graph("face", "summary"))
    try:
        runtime.start_connected_analysis()
        publish(connection, adapter, frame(2))
        assert entered.wait(1)
        worker = runtime._workflow_summaries
        clock.advance(worker, 1)
        wait_for(lambda: worker.latest() is not None)
        assert values(runtime)["summary"]["status"] == "completed"
        assert runtime.results.latest(feature="stranger") is None
        assert worker.stream_session_id == runtime._session.stream_session_id
        release.set()
        wait_for(lambda: runtime.results.latest(feature="stranger") is not None)
        assert (
            values(runtime)["faces"]["results"][0]["judgments"][0]["event_label"]
            == "stranger"
        )
    finally:
        release.set()


def test_slow_summary_does_not_delay_native_face_results(rig, monkeypatch):
    runtime, connection, adapter, clock = rig
    entered, release = Event(), Event()

    async def infer(spec, snapshot):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return "Injected summary result."

    def manager(specs, source, *, session_id):
        from goodvision_harness.workflow_summary import snapshot_from_connection

        return SummaryManager(
            specs,
            source_id=runtime.source_id,
            stream_session_id=session_id,
            generation=runtime._generation,
            frame_provider=lambda: snapshot_from_connection(source),
            clock=clock,
            inference=infer,
        )

    monkeypatch.setattr(runtime, "_make_summary_manager", manager)
    configure(runtime, graph("face", "summary"))
    try:
        runtime.start_connected_analysis()
        worker = runtime._workflow_summaries
        clock.advance(worker, 1)
        assert entered.wait(1)
        publish(connection, adapter, frame(2))
        wait_for(lambda: runtime.results.latest(feature="stranger") is not None)
        assert worker.latest() is None
        assert values(runtime)["faces"] is not None
    finally:
        release.set()


def test_test_returns_real_single_frame_faces_events_summary_without_live_mutation(rig):
    runtime, _connection, _adapter, _clock = rig
    configure(runtime, graph("summary"))
    before_binding = copy.deepcopy(runtime._workflow_binding)
    before_epoch = runtime.results.epoch
    result = runtime.test_workflow(graph("face", "events", "summary"))
    by_type = {item["type"]: item for item in result["results"]}
    assert by_type["face"]["faces"][0]["attributes"]["status"] == "known"
    assert by_type["face"]["observations"][0]["event_label"] == "stranger"
    assert by_type["events"]["observations"][0]["event_label"] == "smoking"
    assert by_type["summary"]["result"]["test"] is True
    assert (
        by_type["summary"]["result"]["stream_session_id"] == result["stream_session_id"]
    )
    assert result["synthetic"] is True
    assert runtime._workflow_binding == before_binding
    assert runtime.results.epoch == before_epoch
    assert runtime.results.latest() is None
    assert runtime._session is None and runtime._workflow_summaries is None
    assert not runtime._processing_lifecycle_busy()


def test_test_face_failure_does_not_hide_event_or_summary_result(rig, monkeypatch):
    from goodvision_harness import grounding

    runtime, _connection, _adapter, _clock = rig
    original = grounding.build_selected_features_grounding_client

    def client(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs["features"] == ("stranger",):

            def fail(window):
                raise RuntimeError("PRIVATE_CREDENTIAL")

            value.classify_stranger_window = fail
        return value

    monkeypatch.setattr(grounding, "build_selected_features_grounding_client", client)
    result = runtime.test_workflow(graph("face", "events", "summary"))
    by_type = {item["type"]: item for item in result["results"]}
    assert by_type["face"]["error"] == PUBLIC_FACE_ERROR
    assert by_type["face"]["faces"] == []
    assert by_type["events"]["observations"]
    assert by_type["summary"]["result"]["status"] == "completed"
    assert "PRIVATE_CREDENTIAL" not in str(result)


def publish_result(runtime, label, sequence):
    runtime.results.publish(
        {
            "source_id": runtime.source_id,
            "stream_session_id": runtime._session.stream_session_id,
            "frame": {
                "source_frame_index": sequence,
                "timestamp_seconds": sequence / 10,
            },
            "judgments": [
                {
                    "event_label": label,
                    "detected": False,
                    "description": label + " observation",
                    "object_ids": [],
                }
            ],
            "description": label + " aggregate",
            "objects": [],
        }
    )


def test_output_uses_event_feature_payload_and_rejects_prior_graph_and_run(rig):
    runtime, _connection, _adapter, _clock = rig
    configure(runtime, graph("face", "events"))
    runtime.start_connected_analysis()
    publish_result(runtime, "smoking", 10)
    publish_result(runtime, "stranger", 20)
    assert runtime.results.latest().payload["judgments"][0]["event_label"] == "stranger"
    output = values(runtime)["events"]
    assert output["results"][0]["frame"]["source_frame_index"] == 10
    assert [j["event_label"] for j in output["results"][0]["judgments"]] == ["smoking"]
    runtime.stop_session()
    wait_for(lambda: not runtime._processing_lifecycle_busy())
    configure(runtime, graph("face", "events"), "replacement")
    assert values(runtime)["events"] is None and values(runtime)["faces"] is None
    runtime.start_connected_analysis()
    assert values(runtime)["events"] is None
    publish_result(runtime, "smoking", 3)
    assert values(runtime)["events"]["results"][0]["frame"]["source_frame_index"] == 3


@pytest.mark.parametrize("operation", ["stop", "replace"])
def test_summary_preflight_cannot_start_after_stop_or_new_apply(
    rig, monkeypatch, operation
):
    runtime, _connection, _adapter, _clock = rig
    configure(runtime, graph("summary"))
    prepared, release, errors = Event(), Event(), []
    original = runtime.prepare_analysis

    def prepare():
        original()
        prepared.set()
        assert release.wait(2)

    def start():
        try:
            runtime.start_connected_analysis()
        except RuntimeError as error:
            errors.append(str(error))

    monkeypatch.setattr(runtime, "prepare_analysis", prepare)
    thread = Thread(target=start)
    thread.start()
    assert prepared.wait(1)
    if operation == "stop":
        runtime.stop_session()
    else:
        configure(runtime, graph(frames=True), "replacement")
    release.set()
    thread.join(2)
    assert not thread.is_alive() and errors
    assert runtime._workflow_summaries is None and runtime._session is None
    assert not runtime._processing_lifecycle_busy()


def test_stop_at_native_admission_does_not_start_a_late_face_session(rig, monkeypatch):
    runtime, _connection, _adapter, _clock = rig
    configure(runtime, graph("face", "summary"))
    entered, release, errors = Event(), Event(), []
    original = runtime.start_stream

    def stream(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    def start():
        try:
            runtime.start_connected_analysis()
        except RuntimeError as error:
            errors.append(str(error))

    monkeypatch.setattr(runtime, "start_stream", stream)
    thread = Thread(target=start)
    thread.start()
    assert entered.wait(1)
    runtime.stop_session()
    release.set()
    thread.join(2)
    assert errors and not thread.is_alive()
    assert runtime._session is None and runtime._workflow_summaries is None
    assert not runtime._processing_lifecycle_busy()


def test_test_manager_remains_busy_until_cancel_resistant_inference_drains(
    rig, monkeypatch
):
    runtime, _connection, _adapter, _clock = rig
    entered, release, errors = Event(), Event(), []

    async def infer(spec, snapshot):
        entered.set()
        while not release.is_set():
            try:
                await asyncio.sleep(0.005)
            except asyncio.CancelledError:
                pass
        return "Late test result."

    def manager(specs, source, *, session_id):
        return SummaryManager(
            specs,
            source_id=runtime.source_id,
            stream_session_id=session_id,
            generation=runtime._generation,
            frame_provider=lambda: None,
            inference=infer,
        )

    def test():
        try:
            runtime.test_workflow(graph("summary"))
        except RuntimeError as error:
            errors.append(str(error))

    monkeypatch.setattr(runtime, "_make_summary_manager", manager)
    thread = Thread(target=test)
    thread.start()
    assert entered.wait(1)
    runtime.stop_session()
    try:
        assert runtime._workflow_test_manager is not None
        assert runtime._workflow_test_manager.state == "stopping"
        assert runtime._processing_lifecycle_busy()
        with pytest.raises(RuntimeError):
            configure(runtime, graph(frames=True))
    finally:
        release.set()
        thread.join(2)
    assert errors and not thread.is_alive()
    wait_for(lambda: not runtime._processing_lifecycle_busy())
    assert runtime.results.latest() is None


def test_summary_preparation_error_preserves_completed_face_test(rig, monkeypatch):
    from goodvision_harness.model_api import MissingAPIKeyError

    runtime, _connection, _adapter, _clock = rig

    def unavailable(*args, **kwargs):
        raise MissingAPIKeyError("PRIVATE_CREDENTIAL")

    monkeypatch.setattr(runtime, "_make_summary_manager", unavailable)
    result = runtime.test_workflow(graph("face", "summary"))
    by_type = {item["type"]: item for item in result["results"]}
    assert by_type["face"]["faces"]
    assert by_type["summary"]["result"]["status"] == "error"
    assert by_type["summary"]["result"]["error"]["code"] == "missing_api_key"
    assert by_type["summary"]["result"]["summary"] is None
    assert "PRIVATE_CREDENTIAL" not in str(result)
    assert not runtime._processing_lifecycle_busy()


def test_independent_summary_remains_visible_after_native_session_ends(rig):
    runtime, _connection, _adapter, clock = rig
    configure(runtime, graph("face", "summary"))
    runtime.start_connected_analysis()
    worker = runtime._workflow_summaries
    native = runtime._session
    original_generation = worker.generation
    native.stop(drain=True)
    native.join(1)
    wait_for(lambda: runtime._session is None)
    assert runtime._generation != original_generation
    assert worker.is_running
    clock.advance(worker, 1)
    wait_for(lambda: worker.latest() is not None)
    assert (
        runtime.workflow_summary_results()["items"][0]["stream_session_id"]
        == native.stream_session_id
    )
    assert values(runtime)["summary"]["generation"] == original_generation
    assert runtime.status()["workflow"]["generation"] == original_generation
    runtime.stop_session()
    wait_for(lambda: not runtime._processing_lifecycle_busy())
    assert runtime.workflow_summary_results()["items"]
    assert values(runtime)["summary"] is not None


def test_native_error_does_not_report_idle_while_independent_summary_is_busy(rig):
    from goodvision_harness.fleet import FleetController

    runtime, _connection, _adapter, _clock = rig
    configure(runtime, graph("summary"))
    runtime.start_connected_analysis()
    runtime._update_state(pipeline_state="error", last_error=PUBLIC_FACE_ERROR)
    for lightweight in (True, False):
        status = runtime.status(lightweight=lightweight)
        assert status["analysis_state"] == "running"
        assert status["native_pipeline_state"] == "error"
        assert status["last_error"] == PUBLIC_FACE_ERROR
        assert FleetController._analysis_state(status) == "running"
    with pytest.raises(RuntimeError):
        configure(runtime, graph(frames=True))


def test_summary_and_raw_run_generations_match_public_state(rig):
    runtime, _connection, _adapter, _clock = rig
    for recipe in (graph("summary"), graph(frames=True)):
        configure(runtime, recipe)
        runtime.start_connected_analysis()
        status = runtime.status()
        assert status["generation"] == runtime._generation
        assert status["workflow"]["generation"] == runtime._generation
        runtime.stop_session()
        wait_for(lambda: not runtime._processing_lifecycle_busy())


@pytest.mark.parametrize("managed", [False, True])
def test_scene_disabled_native_run_finishes_identity_and_evidence_without_scene(
    rig,
    caplog,
    managed,
):
    runtime, _connection, _adapter, _clock = rig
    if managed:
        configure(runtime, graph("face"))
    else:
        runtime.update_analysis_settings({"features": ["stranger"]})
    runtime.start_connected_analysis()
    session = runtime._session
    assert runtime._scene is None
    control = runtime._archive_run(session)
    runtime.stop_session()
    wait_for(lambda: runtime._session is None)
    wait_for(lambda: control.stage == "complete" or control.state == "failed")
    # The publisher follows the completion marker inside this same worker task.
    runtime._postprocess_pool.submit(lambda: None).result(timeout=2)
    assert control.state == "finalized"
    assert control.stage == "complete" and control.error is None
    status = runtime.status()
    assert status["scene_drain"]["state"] == "completed"
    assert runtime._last_scene_snapshot is None
    assert runtime._last_identity_review is not None
    assert "background run finalization failed" not in caplog.text
