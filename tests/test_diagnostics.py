"""Request pictures and partial results are independent of accepted event facts."""

import base64
import json
from dataclasses import replace

import httpx
import pytest
import test_scene_model as model_tests
import test_scene_service as service_tests
from fastapi.testclient import TestClient
from test_scene_model import _job, _response

from goodvision_harness.diagnostics import SceneDiagnostics, failure_category
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_inspection import build_temporal_messages
from goodvision_harness.scene_runtime import SceneRuntime, TemporalVerdict
from goodvision_harness.service import create_app


@pytest.fixture
def make_inspector(monkeypatch):
    yield from model_tests.make_inspector.__wrapped__(monkeypatch)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    service_tests._isolate_environment.__wrapped__(monkeypatch)


@pytest.fixture
def runtime_factory(tmp_path, monkeypatch):
    yield from service_tests.runtime_factory.__wrapped__(tmp_path, monkeypatch)


def job(diagnostics):
    value = _job()
    return replace(
        value,
        diagnostics=diagnostics,
        request=replace(
            value.request,
            event_labels=("smoking", "weapon"),
        ),
    )


def test_actual_all_frame_payload_bytes_dimensions_and_source_times(make_inspector):
    diagnostics = SceneDiagnostics()
    value = job(diagnostics)
    diagnostics.claimed(value)
    inspector, seen, _ = make_inspector(max_image_side=24)
    inspector.inspect(value)
    row = diagnostics.snapshot(value.request.event_labels)["records"][0]
    parts = json.loads(seen[0].content)["messages"][1]["content"]
    sent = [
        base64.b64decode(p["image_url"]["url"].split(",")[1])
        for p in parts
        if p["type"] == "image_url"
    ]
    assert len(row["frames"]) == len(sent) == 2
    for frame, expected in zip(row["frames"], sent, strict=True):
        assert (
            diagnostics.jpeg(value.request.request_id, frame["frame_sequence"])
            == expected
        )
        assert (frame["width"], frame["height"]) == (24, 16)
    assert [f["timestamp_seconds"] for f in row["frames"]] == [0, 1]
    assert row["response_outcome"] == "validated"
    assert row["evidence_outcome"] == "pending"
    assert row["confirmed_event_labels"] == []
    encoded = json.dumps(diagnostics.snapshot(value.request.event_labels))
    assert "SECRET" not in encoded and "rtsp://" not in encoded


@pytest.mark.parametrize(
    "content,missing,unknown",
    [
        ("[]", ["smoking", "weapon"], []),
        (
            '[{"event_label":"smoking","subject_ids":["person-1"],'
            '"detected":false,"confidence":0.9}]',
            ["weapon"],
            [],
        ),
        (
            '[{"event_label":"smoking","subject_ids":["person-1"],'
            '"detected":null,"confidence":null}]',
            ["weapon"],
            ["smoking"],
        ),
    ],
)
def test_missing_or_null_label_is_partial_never_negative(
    make_inspector, content, missing, unknown
):
    diagnostics = SceneDiagnostics()
    value = job(diagnostics)
    inspector, _, _ = make_inspector(
        lambda _: httpx.Response(200, json=_response(content))
    )
    inspector.inspect(value)
    row = diagnostics.snapshot(value.request.event_labels)["records"][0]
    assert row["result"] == "partial"
    assert row["missing_event_labels"] == missing
    assert row["unknown_event_labels"] == unknown


def test_positive_labels_survive_partial_response(make_inspector):
    diagnostics = SceneDiagnostics()
    inspector, _, _ = make_inspector()
    inspector.inspect(job(diagnostics))
    row = diagnostics.snapshot(("smoking", "weapon"))["records"][0]
    assert row["result"] == "partial"
    assert row["detected_labels"] == ["smoking"]


def test_input_memory_eviction_and_late_response_do_not_revive_evidence():
    diagnostics = SceneDiagnostics(image_history=1)
    first = job(diagnostics)
    second = replace(first, request=replace(first.request, request_id="second"))
    for value in (first, second):
        diagnostics.claimed(value)
        encoded_images = []
        build_temporal_messages(value, encoded_images=encoded_images)
        diagnostics.prepared(value, encoded_images)
    assert diagnostics.jpeg(first.request.request_id, 0) is None
    assert diagnostics.jpeg(second.request.request_id, 0)
    diagnostics.completed(second, success=False, expired=True)
    diagnostics.responded(
        second, verdicts=(TemporalVerdict("smoking", ("person-1",), True),)
    )
    diagnostics.completed(second, success=True, accepted=(("smoking", 1, True),))
    row = diagnostics.snapshot(("smoking",))["records"][0]
    assert row["evidence_outcome"] == "expired"
    assert row["accepted_verdicts"] == 0
    assert (
        diagnostics.snapshot(("smoking",))["unanalyzed_gaps"]["smoking"][
            "accepted_definite_verdicts"
        ]
        == 0
    )


def test_reused_inspector_keeps_runs_isolated(make_inspector):
    first, second = SceneDiagnostics(), SceneDiagnostics()
    inspector, _, _ = make_inspector()
    inspector.inspect(job(first))
    before = first.snapshot(("smoking",))
    value = job(second)
    inspector.inspect(
        replace(
            value,
            stream_session_id="new-run",
            frames=tuple(
                replace(frame, stream_session_id="new-run") for frame in value.frames
            ),
        )
    )
    assert first.snapshot(("smoking",)) == before
    assert second.snapshot(("smoking",))["records"][0]["stream_session_id"] == "new-run"


def test_runtime_acceptance_confirmation_and_per_label_gaps():
    scene = SceneRuntime(
        "run",
        "source",
        tracking_mode="batch",
        native_memory_scope="session",
        temporal_observation_fps=1.25,
        event_labels=("smoking", "weapon"),
    )
    original = _job()
    box = original.subject_boxes[0][2]
    jobs = []
    for seq in range(7):
        at = seq / 2
        scene.observe_frame(original.frames[0].jpeg_bytes, at, seq)
        assert scene.observe_tracking_batch(
            (
                RichLabelSnapshot(
                    "run", seq, at, (RichLabelObject("person-1", "person", box),)
                ),
            ),
            completed_timestamp_seconds=at,
            completed_frame_sequence=seq,
        )
        for value in scene.take_inspections():
            verdicts = (TemporalVerdict("smoking", ("person-1",), True, 0.9),)
            scene.diagnostics.responded(value, verdicts=verdicts)
            assert scene.complete_inspection(value.request.request_id, verdicts)
            jobs.append(value)
    snapshot = scene.diagnostics.snapshot(scene.event_labels)
    assert len(jobs) >= 2
    assert snapshot["records"][0]["confirmed_event_labels"] == ["smoking"]
    assert snapshot["records"][-1]["confirmed_event_labels"] == []
    assert snapshot["records"][0]["missing_event_labels"] == ["weapon"]
    assert snapshot["unanalyzed_gaps"]["weapon"]["current_seconds"] == 3
    assert snapshot["unanalyzed_gaps"]["smoking"]["current_seconds"] < 1


def test_exact_input_api_and_eviction(runtime_factory):
    runtime = runtime_factory()
    scene = SceneRuntime("run", "source")
    runtime._scene = scene
    value = job(scene.diagnostics)
    encoded_images = []
    build_temporal_messages(value, max_image_side=24, encoded_images=encoded_images)
    scene.diagnostics.prepared(value, encoded_images)
    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        response = client.get("/v1/scene/diagnostics")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        image = client.get("/v1/scene/diagnostics/request/0.jpg")
        assert image.content == scene.diagnostics.jpeg("request", 0)
        scoped = client.get("/v1/fleet/cameras/camera01/scene/diagnostics")
        assert scoped.status_code == 200
        assert scoped.json()["camera_id"] == "camera01"
        scoped_image = client.get(
            "/v1/fleet/cameras/camera01/scene/diagnostics/request/0.jpg"
        )
        assert scoped_image.content == image.content
        assert scoped_image.headers["cache-control"] == "no-store"
        assert client.get("/v1/scene/diagnostics/other/0.jpg").status_code == 404
        assert client.get("/assets/scene_diagnostics.js").status_code == 200


@pytest.mark.parametrize(
    "code,status,expected",
    [
        ("http_error", 401, "auth"),
        ("http_error", 429, "rate_limited"),
        ("http_error", 503, "server_error"),
        ("timeout", None, "timeout"),
        ("invalid_response", None, "parse"),
        ("connection", None, "connection"),
    ],
)
def test_failure_categories(code, status, expected):
    assert failure_category({"code": code, "http_status": status}) == expected
