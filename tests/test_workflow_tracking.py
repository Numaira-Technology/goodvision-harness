"""Canvas tracking controls a real native run without changing deployment defaults."""

from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from test_processing_method_service import rig as rig
from test_video_connection import ControlledAdapter, frame, publish, wait_for
from test_workflow_runtime import configure, graph, values

from goodvision_harness import service


def tracking_graph(*kinds):
    value = graph(*kinds)
    value["nodes"].append({"id": "tracking", "type": "tracking"})
    value["edges"].extend(
        [
            {
                "id": "in-tracking",
                "from_node": "source",
                "from_port": "frames",
                "to_node": "tracking",
                "to_port": "frames",
            },
            {
                "id": "out-tracking",
                "from_node": "tracking",
                "from_port": "tracks",
                "to_node": "out",
                "to_port": "tracks",
            },
        ]
    )
    return value


def connect(runtime):
    adapter = ControlledAdapter("rtsp://camera.test/video")
    runtime.connect_stream("rtsp://camera.test/video", _adapter=adapter)
    connection = runtime._video_connection
    publish(connection, adapter, frame(1))
    wait_for(lambda: connection.latest_jpeg() is not None)
    return connection


def stop(runtime):
    session = runtime._session
    runtime.stop_session()
    if session is not None:
        session.join(timeout=3)
    wait_for(lambda: not runtime._processing_lifecycle_busy())


def test_unconfigured_tracking_rejects_binding_without_partial_override(rig):
    config = replace(
        rig.config,
        rich_labeling=replace(
            rig.config.rich_labeling,
            backend="paddle",
            enabled=False,
            tracking_enabled=False,
            processing_method=None,
        ),
        runtime=replace(rig.config.runtime, scene_mode="disabled"),
    )
    runtime = rig.make(config)
    capability = runtime.workflow_tracking_capability()
    assert capability["configured"] is False
    assert (
        capability["reason"] == "Person tracking is not configured for this deployment."
    )
    with pytest.raises(ValueError, match="not configured"):
        configure(runtime, tracking_graph())
    assert runtime.config is config
    assert runtime._workflow_binding is None
    assert runtime._workflow_tracking_baseline is None


def test_tracking_only_uses_native_capture_without_event_features_or_model_key(rig):
    runtime = rig.make()
    runtime._aiclub_api_key = None
    connection = connect(runtime)
    deployment = rig.store.load()
    remote = runtime._remote_config
    configure(runtime, tracking_graph())
    assert runtime.config.rich_labeling.processing_method == "sam_live"
    assert runtime.config.rich_labeling.tracking_enabled is True
    assert runtime.config.rich_labeling.enabled is False
    assert runtime._rich_requested is False
    assert runtime._tracking_requested is True
    assert runtime._remote_config is remote
    assert runtime.prepare_analysis()["prepared"]
    result = runtime.start_connected_analysis()
    assert result["pipeline_state"] in {"starting", "running"}
    assert result["features"] == []
    assert result["event_sensitivities"] == {}
    assert runtime._session is not None
    assert runtime._vlm_branch is None
    assert runtime._workflow_raw_session_id is None
    assert rig.store.load() == deployment
    output = values(runtime)["tracks"]
    assert output["state"] == "waiting_for_tracks"
    assert output["stream_session_id"] == runtime._session.stream_session_id
    assert output["source_id"] == runtime.source_id
    assert output["generation"] == runtime._generation
    stop(runtime)
    assert connection.status()["connected"] is True
    assert values(runtime)["tracks"] is None


def test_tracking_output_requires_current_session_and_carries_native_anchors(rig):
    runtime = rig.make()
    connect(runtime)
    configure(runtime, tracking_graph())
    runtime.start_connected_analysis()
    session = runtime._session.stream_session_id
    metadata = runtime._camera_display_metadata
    batch = SimpleNamespace(
        stream_session_id=session,
        predictor_epoch="native-current",
        frames=(SimpleNamespace(frame_sequence=7, timestamp_seconds=0.5, objects=()),),
    )
    assert metadata.observe_batch(batch, native_token=metadata._native_token)
    output = values(runtime)["tracks"]
    assert output["state"] == "running"
    assert output["metadata"]["anchors"][0]["source_sequence"] == 7
    assert output["metadata"]["anchors"][0]["subjects"] == []
    metadata.reset("previous-run")
    assert values(runtime)["tracks"] is None
    stop(runtime)


def test_tracking_readiness_is_required_even_with_independent_events(rig):
    runtime = rig.make()
    runtime._aiclub_api_key = "TEST_KEY"
    connect(runtime)
    configure(runtime, tracking_graph("events"))
    rig.control.ready = False
    with pytest.raises(service.ProcessingMethodUnavailableError):
        runtime.prepare_analysis()
    assert runtime._session is None


def test_disabled_tracking_does_not_probe_and_unassign_restores_baseline(
    rig, monkeypatch
):
    runtime = rig.make()
    runtime._aiclub_api_key = "TEST_KEY"
    connect(runtime)
    base_rich = runtime.config.rich_labeling
    base_scene = runtime.config.runtime.scene_mode
    remote = runtime._remote_config
    saved = rig.store.load()
    value = tracking_graph("events")
    value["nodes"][-1]["enabled"] = False
    configure(runtime, value)
    assert runtime._rich_labels is None
    assert runtime.config.rich_labeling.processing_method is None
    assert not runtime._tracking_requested
    assert not runtime.video_display_config()["enabled"]
    monkeypatch.setattr(
        runtime,
        "_require_processing_ready",
        lambda: pytest.fail("an off workflow tracker must never be probed"),
    )
    assert runtime.prepare_analysis()["prepared"]
    runtime.config = replace(
        runtime.config, inference=replace(runtime.config.inference, model="new-route")
    )
    runtime.configure_workflow({"binding": None})
    assert runtime.config.rich_labeling == base_rich
    assert runtime.config.runtime.scene_mode == base_scene
    assert runtime.config.inference.model == "new-route"
    assert runtime._remote_config is remote
    assert runtime._workflow_tracking_baseline is None
    assert rig.store.load() == saved


@pytest.mark.parametrize("unassign", [False, True])
def test_delayed_retirement_finishes_before_binding_returns_and_first_start(
    rig,
    monkeypatch,
    unassign,
):
    runtime = rig.make()
    connect(runtime)
    configure(runtime, tracking_graph())
    runtime.start_connected_analysis()
    stop(runtime)
    old = runtime._rich_labels
    released, closing, finished = Event(), Event(), Event()
    errors = []
    original_close = old.close

    def close(**kwargs):
        original_close(**kwargs)
        closing.set()

    monkeypatch.setattr(old, "close", close)
    monkeypatch.setattr(old, "close_complete", released.is_set)

    def change():
        try:
            if unassign:
                runtime.configure_workflow({"binding": None})
            else:
                configure(runtime, graph("summary"))
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    worker = Thread(target=change)
    worker.start()
    try:
        assert closing.wait(1)
        assert not finished.wait(0.05)
        # Waiting for retirement releases both lifecycle locks.
        assert runtime.status(lightweight=True)["processing_cleanup"] == "closing"
        released.set()
        worker.join(2)
        assert not worker.is_alive() and not errors
        if unassign:
            runtime.dry_run = True
            runtime._analysis_settings = {
                "features": ["smoking"],
                "event_sensitivities": {},
            }
        else:
            runtime.dry_run = True
        assert runtime.start_connected_analysis()["pipeline_state"] in {
            "starting",
            "running",
        }
        stop(runtime)
    finally:
        released.set()
        worker.join(2)


def test_stop_cancels_binding_retirement_wait_without_installing_late_run(
    rig, monkeypatch
):
    runtime = rig.make()
    old = runtime._rich_labels
    released, closing = Event(), Event()
    monkeypatch.setattr(old, "close_complete", released.is_set)
    original_close = old.close

    def close(**kwargs):
        original_close(**kwargs)
        closing.set()

    monkeypatch.setattr(old, "close", close)
    errors = []

    def change():
        try:
            configure(runtime, graph("summary"))
        except Exception as error:
            errors.append(error)

    worker = Thread(target=change)
    worker.start()
    try:
        assert closing.wait(1)
        runtime.stop_session()
        worker.join(1)
        assert not worker.is_alive()
        assert errors and "cancelled" in str(errors[0])
        assert runtime._session is None
    finally:
        released.set()
        worker.join(2)


def test_external_processing_changes_update_baseline_without_losing_workflow_policy(
    rig,
):
    runtime = rig.make()
    configure(runtime, graph("summary"))
    baseline = runtime._workflow_tracking_baseline["rich_labeling"]
    changed = replace(
        runtime.config,
        rich_labeling=replace(baseline, max_fps=7, processing_method="sam_live"),
        inference=replace(runtime.config.inference, model="updated-model"),
    )
    runtime._replace_processing_method("sam_live", config=changed)
    assert runtime.config.rich_labeling.processing_method is None
    assert runtime._tracking_requested is False
    assert runtime._workflow_tracking_baseline["rich_labeling"].max_fps == 7
    runtime.configure_workflow({"binding": None})
    assert runtime.config.rich_labeling.processing_method == "sam_live"
    assert runtime.config.rich_labeling.max_fps == 7
    assert runtime.config.inference.model == "updated-model"


@pytest.mark.parametrize("independent", [False, True])
def test_global_processing_save_uses_baseline_capability_while_workflow_is_off(
    rig,
    independent,
):
    config = replace(
        rig.config,
        rich_labeling=replace(rig.config.rich_labeling, processing_method=None),
        runtime=replace(rig.config.runtime, scene_independent_vlm=independent),
    )
    runtime = rig.make(config)
    configure(runtime, graph("summary"))
    assert runtime._processing_method() is None
    assert runtime._scene_mode == ("active" if independent else "disabled")
    runtime.update_deployment_settings({"processing_method": "sam_live"})
    assert rig.store.load().processing_method == "sam_live"
    assert runtime._processing_method() is None
    assert not runtime._tracking_requested
    runtime.configure_workflow({"binding": None})
    assert runtime._processing_method() == "sam_live"
    assert runtime._scene_mode == "active"
    assert runtime._tracking_requested


def test_tracking_node_activates_configured_service_from_disabled_scene_defaults(rig):
    config = replace(
        rig.config,
        rich_labeling=replace(
            rig.config.rich_labeling,
            enabled=False,
            tracking_enabled=False,
            processing_method=None,
        ),
        runtime=replace(rig.config.runtime, scene_mode="disabled"),
    )
    runtime = rig.make(config)
    baseline = runtime.config.rich_labeling
    connect(runtime)
    assert runtime.workflow_tracking_capability()["configured"]
    configure(runtime, tracking_graph())
    assert runtime._scene_mode == "active"
    assert runtime._processing_method() == "sam_live"
    assert runtime.prepare_analysis()["prepared"]
    runtime.configure_workflow({"binding": None})
    assert runtime._scene_mode == "disabled"
    assert runtime._processing_method() is None
    assert runtime.config.rich_labeling == baseline


@pytest.mark.parametrize("combined_method", [False, True])
def test_new_tracking_server_updates_baseline_scene_without_enabling_off_node(
    rig,
    combined_method,
):
    config = replace(
        rig.config,
        rich_labeling=replace(
            rig.config.rich_labeling,
            enabled=False,
            tracking_enabled=False,
            processing_method=None,
        ),
        runtime=replace(rig.config.runtime, scene_mode="disabled"),
    )
    runtime = rig.make(config)
    configure(runtime, graph("summary"))
    payload = {
        "tracking_mode": "server",
        "tracking_server_url": "http://tracker.test:8092",
    }
    if combined_method:
        payload["processing_method"] = "sam_live"
    runtime.update_deployment_settings(payload)
    assert runtime._processing_method() is None
    assert runtime._tracking_requested is False
    assert runtime._workflow_tracking_baseline["scene_mode"] == "active"
    assert (
        runtime._workflow_tracking_baseline["rich_labeling"].processing_method
        == "sam_live"
    )
    runtime.configure_workflow({"binding": None})
    assert runtime._scene_mode == "active"
    assert runtime._processing_method() == "sam_live"
    assert runtime._tracking_requested is True
    assert (
        runtime.config.rich_labeling.endpoint
        == "http://tracker.test:8092/v1/video/sessions"
    )


@pytest.mark.parametrize("ready", [False, True])
def test_draft_tracking_test_checks_readiness_without_claiming_single_frame_tracks(
    rig, ready
):
    runtime = rig.make()
    connect(runtime)
    binding, configuration = runtime._workflow_binding, runtime.config
    rig.control.ready = ready
    result = runtime.test_workflow(tracking_graph())
    record = result["results"][0]
    assert record["type"] == "tracking"
    assert record["state"] == ("live_analysis_required" if ready else "unavailable")
    assert "tracks" not in record and "metadata" not in record
    assert runtime._workflow_binding is binding
    assert runtime.config is configuration
    assert runtime._session is None
