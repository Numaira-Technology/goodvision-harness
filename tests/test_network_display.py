import json
import struct
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from goodvision_harness.camera_display import CameraDisplayMetadata
from goodvision_harness.network_display import MEDIA_TYPE, display_response


def runtime():
    bus = CameraDisplayMetadata()
    bus.reset("run")
    bus.accept_source(dict(stream_session_id="run", generation=2, crop_version=0,
                           source_sequence=100, media_timestamp=10.0))
    connection = SimpleNamespace(connection_id="connection", status=lambda: {"connected": True},
                                 display_frames=lambda after: ((101, 10.1, 1280, 720, b"jpeg"),))
    return SimpleNamespace(_video_connection=connection, _camera_display_metadata=bus,
                           _state={"pipeline_state": "running"}, _session=SimpleNamespace(mode="stream"),
                           _display_settings=SimpleNamespace(delay_seconds=6.0), _scene_mode="active",
                           config=SimpleNamespace(rich_labeling=SimpleNamespace(backend="sam31_stream")),
                           video_display_config=lambda: {"target_fps": 18, "max_propagation_seconds": 1.5})


def unpack(response):
    assert response.media_type == MEDIA_TYPE
    assert response.body[:4] == b"GVD1"
    size = struct.unpack(">I", response.body[4:8])[0]
    return json.loads(response.body[8:8+size]), response.body[8+size:]


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


def test_metadata_delta_and_connection_cursor_are_independent():
    run = runtime()
    afters = []
    run._video_connection.display_frames = lambda after: afters.append(after) or ()
    full, _ = unpack(display_response(run, after=999, connection_id="old"))
    assert afters == [-1]
    delta, _ = unpack(display_response(run, after=101, connection_id="connection",
                                       revision=full["metadata"]["metadata_revision"], stream_session_id="run"))
    assert delta["metadata"] is None
    assert afters == [-1, 101]
    restarted, _ = unpack(display_response(run, revision=99999, stream_session_id="old-run"))
    assert restarted["metadata"] is not None


def test_stopping_analysis_keeps_live_video_and_removes_old_run_geometry():
    run = runtime()
    run._state["pipeline_state"] = "idle"
    data, image = unpack(display_response(run))
    assert image == b"jpeg"
    assert data["config"]["enabled"] is False
    assert data["config"]["stream_session_id"] == "connection"
    assert data["config"]["delay_seconds"] == 0
    assert data["metadata"] is None
    run._video_connection.status = lambda: {"connected": False}
    with pytest.raises(HTTPException) as exc:
        display_response(run)
    assert exc.value.status_code == 404
