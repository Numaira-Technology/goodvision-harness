import json
import struct
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from goodvision_harness.camera_display import CameraDisplayMetadata
from goodvision_harness.network_display import MEDIA_TYPE, display_response
from goodvision_harness.result_bus import FrameBus, FramePacket


def runtime():
    bus = CameraDisplayMetadata()
    bus.reset("run")
    bus.accept_source(
        dict(
            stream_session_id="run",
            generation=2,
            crop_version=0,
            source_sequence=100,
            media_timestamp=10.0,
        )
    )
    connection = SimpleNamespace(
        connection_id="connection",
        status=lambda: {"connected": True},
        display_frames=lambda after, **options: ((101, 10.1, 1280, 720, b"jpeg"),),
    )
    return SimpleNamespace(
        _video_connection=connection,
        _camera_display_metadata=bus,
        _state={"pipeline_state": "running"},
        _session=SimpleNamespace(mode="stream"),
        _display_settings=SimpleNamespace(delay_seconds=6.0),
        _scene_mode="active",
        config=SimpleNamespace(rich_labeling=SimpleNamespace(backend="sam31_stream")),
        video_display_config=lambda: {"target_fps": 18, "max_propagation_seconds": 1.5},
    )


def unpack(response):
    assert response.media_type == MEDIA_TYPE
    assert response.body[:4] == b"GVD1"
    size = struct.unpack(">I", response.body[4:8])[0]
    return json.loads(response.body[8 : 8 + size]), response.body[8 + size :]


def test_display_uses_new_raw_source_without_any_sam_result_or_annotation():
    run = runtime()
    data, image = unpack(display_response(run))
    assert image == b"jpeg"
    assert data["frames"][0]["source_sequence"] == 101
    assert data["frames"][0]["media_timestamp"] == 10.1
    assert data["config"]["delay_seconds"] == 6
    assert data["config"]["generation"] == 2
    assert data["metadata"]["anchors"] == []
    assert data["config"]["enabled"] is True


def test_canvas_tracking_gate_controls_geometry_without_interrupting_video():
    run = runtime()
    run._workflow_plan = {"tracking_enabled": False}
    data, image = unpack(display_response(run))
    assert image == b"jpeg"
    assert data["frames"][0]["source_sequence"] == 101
    assert data["config"]["enabled"] is False
    assert data["metadata"] is None
    assert data["config"]["delay_seconds"] == 6

    run._workflow_plan = {"tracking_enabled": True}
    data, image = unpack(display_response(run))
    assert image == b"jpeg"
    assert data["config"]["enabled"] is True
    assert data["config"]["stream_session_id"] == "run"
    assert data["metadata"] is not None


def test_metadata_delta_and_connection_cursor_are_independent():
    run = runtime()
    afters = []
    run._video_connection.display_frames = lambda after, **options: (
        afters.append(after) or ()
    )
    full, _ = unpack(display_response(run, after=999, connection_id="old"))
    assert afters == [-1]
    delta, _ = unpack(
        display_response(
            run,
            after=101,
            connection_id="connection",
            revision=full["metadata"]["metadata_revision"],
            stream_session_id="run",
        )
    )
    assert delta["metadata"] is None
    assert afters == [-1, 101]
    restarted, _ = unpack(
        display_response(run, revision=99999, stream_session_id="old-run")
    )
    assert restarted["metadata"] is not None


def test_stopping_analysis_keeps_live_video_and_removes_old_run_geometry():
    run = runtime()
    run._state["pipeline_state"] = "idle"
    data, image = unpack(display_response(run))
    assert image == b"jpeg"
    assert data["config"]["enabled"] is False
    assert data["config"]["stream_session_id"] == "connection"
    assert data["config"]["delay_seconds"] == 6
    assert data["metadata"] is None
    run._video_connection.status = lambda: {"connected": False}
    with pytest.raises(HTTPException) as exc:
        display_response(run)
    assert exc.value.status_code == 404


def bbox_runtime():
    run = runtime()
    run.config.rich_labeling.backend = "paddle"
    run._workflow_plan = {"tracking_enabled": True, "tracking_engine": "bbox"}
    run.annotated_frames = FrameBus()
    run.annotated_frames.publish(FramePacket(
        jpeg=b"actual server-drawn boxes", media_timestamp=10.1,
        frame_sequence=101, observed_at="now", annotation_mode="held",
    ))
    return run


@pytest.mark.parametrize("managed", [True, False])
def test_bbox_network_playback_delivers_server_annotations_without_sam_geometry(
    managed,
):
    run = bbox_runtime()
    if not managed:
        run._workflow_plan = None
    data, image = unpack(display_response(run))
    assert image == b"actual server-drawn boxes"
    assert data["frames"][0] == {
        "source_sequence": 101, "media_timestamp": 10.1,
        "width": 1280, "height": 720, "bytes": len(image),
    }
    assert data["config"]["server_annotations"] is True
    assert data["config"]["enabled"] is False
    assert data["metadata"] is None


def test_bbox_annotation_cursor_and_retained_source_must_both_match():
    run = bbox_runtime()
    data, image = unpack(display_response(run, after=101, connection_id="connection"))
    assert data["frames"] == [] and image == b""
    run.annotated_frames.publish(FramePacket(
        jpeg=b"stale boxes", media_timestamp=9.0,
        frame_sequence=90, observed_at="now", annotation_mode="held",
    ))
    data, image = unpack(display_response(run))
    assert data["frames"] == [] and image == b""
    run._state["pipeline_state"] = "idle"
    data, image = unpack(display_response(run))
    assert image == b"jpeg" and data["config"]["server_annotations"] is False


def test_bbox_continuous_stream_delivers_boxes_then_returns_to_raw_after_stop():
    import asyncio

    from goodvision_harness.network_display import display_stream_response

    run = bbox_runtime()

    async def read():
        iterator = display_stream_response(run).body_iterator
        first = await anext(iterator)
        duplicate = await anext(iterator)
        run._state["pipeline_state"] = "idle"
        stopped = await anext(iterator)
        await iterator.aclose()
        return [unpack(SimpleNamespace(body=b[4:], media_type=MEDIA_TYPE))
                for b in (first, duplicate, stopped)]

    (first, image), (duplicate, _), (stopped, raw) = asyncio.run(read())
    assert image == b"actual server-drawn boxes"
    assert first["config"]["transport"] == "stream" and first["metadata"] is None
    assert duplicate["frames"] == []
    assert stopped["config"]["server_annotations"] is False and raw == b"jpeg"


def test_stream_never_requests_contours_and_recovers_inside_playback_budget():
    import asyncio

    from goodvision_harness.network_display import display_stream_response

    run = runtime()
    options = []

    def frames(after, **kwargs):
        options.append((after, kwargs))
        return ((101, 10.1, 1280, 720, b"jpeg"),)

    run._video_connection.display_frames = frames
    run._camera_display_metadata.delta = lambda *a, **k: pytest.fail(
        "video waited for contours"
    )
    response = display_stream_response(run)

    async def read():
        iterator = response.body_iterator
        body = await anext(iterator)
        await iterator.aclose()
        return body

    body = asyncio.run(read())
    assert struct.unpack(">I", body[:4])[0] == len(body) - 4
    packet = SimpleNamespace(body=body[4:], media_type=MEDIA_TYPE)
    data, image = unpack(packet)
    assert data["metadata"] is None and data["config"]["transport"] == "stream"
    assert image == b"jpeg"
    assert options == [(-1, {"max_age_seconds": 6.5, "limit": 64})]


def test_stream_keeps_cursor_and_observes_stop_without_a_model_result():
    import asyncio

    from goodvision_harness.network_display import display_stream_response

    run = runtime()
    cursors = []

    def frames(after, **kwargs):
        cursors.append(after)
        return ((101, 10.1, 1280, 720, b"jpeg"),) if after < 101 else ()

    run._video_connection.display_frames = frames

    async def read():
        iterator = display_stream_response(run).body_iterator
        first = await anext(iterator)
        run._state["pipeline_state"] = "idle"
        second = await anext(iterator)
        await iterator.aclose()
        return first, second

    _first, second = asyncio.run(read())
    data, _ = unpack(SimpleNamespace(body=second[4:], media_type=MEDIA_TYPE))
    assert cursors == [-1, 101]
    assert data["frames"] == [] and data["config"]["enabled"] is False
    assert data["config"]["delay_seconds"] == 6


@pytest.mark.parametrize("delay", [0, 2.5, 30])
@pytest.mark.parametrize("mode", ["idle", "no_tracking", "other_backend"])
def test_network_preview_buffer_is_independent_of_analysis(delay, mode):
    run = runtime()
    run._display_settings.delay_seconds = delay
    if mode == "idle":
        run._state["pipeline_state"] = "idle"
    elif mode == "no_tracking":
        run._workflow_plan = {"tracking_enabled": False}
    else:
        run.config.rich_labeling.backend = "none"
    options = []
    run._video_connection.display_frames = lambda after, **kwargs: (
        options.append(kwargs) or ((101, 10.1, 1280, 720, b"jpeg"),)
    )
    data, _ = unpack(display_response(run))
    assert data["config"]["enabled"] is False
    assert data["metadata"] is None
    assert data["config"]["delay_seconds"] == delay
    assert data["config"]["max_history_seconds"] >= delay + 4
    assert options[0]["max_age_seconds"] == delay + .5


def test_live_stream_observes_buffer_changes_even_without_new_frames():
    import asyncio

    from goodvision_harness.network_display import display_stream_response

    run = runtime()
    run._state["pipeline_state"] = "idle"
    run._video_connection.display_frames = lambda *a, **k: ()

    async def read():
        iterator = display_stream_response(run).body_iterator
        packets = []
        try:
            for delay in [6, 3, 0, 8]:
                run._display_settings.delay_seconds = delay
                body = await anext(iterator)
                data, _ = unpack(SimpleNamespace(body=body[4:], media_type=MEDIA_TYPE))
                packets.append(data["config"]["delay_seconds"])
        finally:
            await iterator.aclose()
        return packets

    assert asyncio.run(read()) == [6, 3, 0, 8]


def test_slow_contour_serialization_releases_source_identity_lock(monkeypatch):
    import threading

    import goodvision_harness.camera_display as module

    run = runtime()
    entered, release, accepted = threading.Event(), threading.Event(), threading.Event()
    original = module.public_safe_value

    def slow_copy(payload):
        entered.set()
        assert release.wait(2)
        return original(payload)

    monkeypatch.setattr(module, "public_safe_value", slow_copy)
    worker = threading.Thread(
        target=lambda: run._camera_display_metadata.delta(None, stream_session_id="run")
    )
    worker.start()
    assert entered.wait(1)

    def identity():
        run._camera_display_metadata.source_identity()
        accepted.set()

    reader = threading.Thread(target=identity)
    reader.start()
    try:
        assert accepted.wait(1), "mask copying must not block video source identity"
    finally:
        release.set()
        worker.join(2)
        reader.join(2)


def test_metadata_delta_filters_old_contours_before_copying(monkeypatch):
    import goodvision_harness.camera_display as module

    bus = runtime()._camera_display_metadata
    bus.anchors.extend(
        [
            {"source_sequence": i, "media_timestamp": 10 + i / 15, "subjects": []}
            for i in (101, 102)
        ]
    )
    bus._anchor_revisions.update({101: 8, 102: 9})
    bus.metadata_revision = 9
    selected = []
    original = module.public_safe_value

    def copied(payload):
        selected.extend(anchor["source_sequence"] for anchor in payload["anchors"])
        return original(payload)

    monkeypatch.setattr(module, "public_safe_value", copied)
    assert bus.delta(8, stream_session_id="run")["metadata_revision"] == 9
    assert selected == [102]
