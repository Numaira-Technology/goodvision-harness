"""Detector preview event attribution, source boundaries and worker failures."""

import base64
import io
import json
import urllib.error
from types import SimpleNamespace

import pytest
from PIL import Image

from tools.vlm_benchmark.monitor import LiveMonitor, _event_subjects


def box(x1=0.1, x2=0.4):
    return {"x1": x1, "y1": 0.1, "x2": x2, "y2": 0.9}


class Frames(list):
    media_seconds = (0.0, 0.4, 0.8, 1.2)
    media_duration = 1.6


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    monkeypatch.setenv("PREVIEW_DETECTOR_URL", "http://detector")
    image = io.BytesIO()
    Image.new("RGB", (32, 24)).save(image, format="JPEG")
    frames = Frames([image.getvalue()] * 4)
    options = SimpleNamespace(
        model="simulated", real=False, backend="local", observation_fps=1.25
    )
    instance = LiveMonitor(tmp_path, [frames], options)
    instance.begin(
        {
            "id": "stage-one",
            "mode": "parallel",
            "phase": "measurement",
            "duration": 10,
            "repeat": 1,
            "streams": 1,
        },
        tmp_path,
    )
    yield instance
    instance.log.close()


@pytest.fixture
def worker(monkeypatch):
    state = SimpleNamespace(
        snapshot={
            "objects": [
                {"track_id": "actor", "bbox": box()},
                {"track_id": "bystander", "bbox": box(0.6, 0.9)},
            ]
        },
        requests=[],
        error=None,
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, size):
            return json.dumps(state.snapshot).encode()[:size]

    def urlopen(request, **_):
        state.requests.append(
            {key.lower(): value for key, value in request.header_items()}
        )
        if state.error:
            raise state.error
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return state


def report(tick=0, lane="grounding", *, subject=True, scene_label=None):
    observations = []
    objects = []
    if subject:
        objects.append({"object_id": "a1", "label": "person", "bbox": box()})
        observations.append(
            {
                "event_label": "smoking",
                "detected": True,
                "grounded": True,
                "subject_ids": ["a1"],
            }
        )
    labels = (["smoking"] if subject else []) + ([scene_label] if scene_label else [])
    return {
        "message_id": f"{lane}-{tick}",
        "vlm_lane": lane,
        "frame": {"source_frame_index": tick, "timestamp_seconds": tick * 0.4},
        "state": "completed",
        "analysis_complete": True,
        "timing": {},
        "objects": objects,
        "observations": observations,
        "judgments": [
            {"event_label": label, "observed_detected": True} for label in labels
        ],
    }


def preview(monitor, at=0.1, epoch=0, generation=0):
    return monitor.track_preview(
        monitor.state["stage"],
        0,
        {
            "client": "a" * 36,
            "time": at,
            "epoch": epoch,
            "generation": generation,
            "jpeg": base64.b64encode(monitor.framesets[0][0]).decode(),
        },
    )


def test_fresh_subject_labels_only_the_unambiguous_track(monitor, worker):
    # Even an overlapping bystander's IoU > 0.3 cannot claim the same subject.
    worker.snapshot["objects"][1]["bbox"] = box(0.2, 0.5)
    monitor.result(0, report())
    result = preview(monitor)
    assert result["objects"][0]["event"] == "smoking"
    assert result["objects"][0]["event_state"] == "alert"
    assert result["objects"][1]["event_state"] == "normal"
    assert "event" not in result["objects"][1]


@pytest.mark.parametrize("tick,at,epoch", [(0, 1.1, 0), (0, 1.7, 1), (2, 0.1, 0)])
def test_old_loop_delayed_and_future_subjects_fall_back_to_scene(
    monitor, worker, tick, at, epoch
):
    # The actor moved and a bystander occupies the old subject rectangle.
    worker.snapshot["objects"] = [
        {"track_id": "actor", "bbox": box(0.6, 0.9)},
        {"track_id": "bystander", "bbox": box()},
    ]
    monitor.result(0, report(tick))
    result = preview(monitor, at, epoch)
    assert result["event_state"] == "alert"
    assert all(obj["event_state"] == "alert" for obj in result["objects"])
    assert all("event" not in obj for obj in result["objects"])


def test_ambiguous_overlap_preserves_scene_alert_without_actor_label(monitor, worker):
    worker.snapshot["objects"][1]["bbox"] = box()
    monitor.result(0, report())
    result = preview(monitor)
    assert result["event_state"] == "alert"
    assert all("event" not in obj for obj in result["objects"])


@pytest.mark.parametrize(
    "scene_label,other_alert", [("smoking", False), ("fight", True)]
)
def test_scene_fallback_is_resolved_per_event_across_lanes(
    monitor, worker, scene_label, other_alert
):
    monitor.result(0, report())
    monitor.result(
        0, report(lane="information", subject=False, scene_label=scene_label)
    )
    result = preview(monitor)
    assert result["objects"][0]["event"] == "smoking"
    assert (result["objects"][1]["event_state"] == "alert") is other_alert
    assert "event" not in result["objects"][1]


def test_incomplete_report_cannot_assign_events(monitor, worker):
    row = report()
    row.update(state="failed", analysis_complete=False, error_code="incomplete")
    monitor.result(0, row)
    assert all(obj["event_state"] == "normal" for obj in preview(monitor)["objects"])


def test_predicted_box_cannot_claim_an_actor_event(monitor, worker):
    worker.snapshot["objects"][0]["observation_state"] = "predicted"
    monitor.result(0, report())
    result = preview(monitor)
    assert result["event_state"] == "alert"
    assert all("event" not in obj for obj in result["objects"])


def test_forward_seek_resets_detector_session(monitor, worker):
    preview(monitor, 0.1)
    preview(monitor, 0.8, generation=1)
    ids = [row["x-goodvision-stream-session-id"] for row in worker.requests]
    assert ids[0] != ids[1]
    assert [row["x-goodvision-frame-sequence"] for row in worker.requests] == ["1", "1"]


def test_unmatched_second_actor_keeps_scene_alert(monitor, worker):
    row = report()
    row["objects"].append(
        {"object_id": "a2", "label": "person", "bbox": box(0.45, 0.55)}
    )
    row["observations"][0]["subject_ids"].append("a2")
    monitor.result(0, row)
    result = preview(monitor)
    assert result["objects"][0]["event"] == "smoking"
    assert result["objects"][1]["event_state"] == "alert"
    assert "event" not in result["objects"][1]


def test_detector_sessions_reset_on_stage_loop_and_seek(monitor, worker):
    preview(monitor, 0.1)
    preview(monitor, 0.2)
    preview(monitor, 1.7, 1)
    preview(monitor, 1.65, 1)
    monitor.state["stage"] = "stage-two"
    preview(monitor, 1.75, 1)
    ids = [row["x-goodvision-stream-session-id"] for row in worker.requests]
    sequences = [row["x-goodvision-frame-sequence"] for row in worker.requests]
    assert ids[0] == ids[1]
    assert len(set(ids)) == 4
    assert sequences == ["1", "2", "1", "1", "1"]


@pytest.mark.parametrize(
    "snapshot",
    [
        [],
        None,
        {},
        {"objects": None},
        {"objects": [None]},
        {"objects": [{"track_id": "a", "bbox": box(-0.1, 0.4)}]},
        {"objects": [{"track_id": "a", "bbox": box(0.1, float("inf"))}]},
        {
            "objects": [
                {"track_id": "a", "bbox": box()},
                {"track_id": "a", "bbox": box()},
            ]
        },
    ],
)
def test_malformed_detector_snapshots_fail_closed(monitor, worker, snapshot):
    worker.snapshot = snapshot
    result = preview(monitor)
    assert result["status"] == "detector_unavailable"
    assert result["objects"] == []


def test_detector_transport_failure_falls_back_without_raising(monitor, worker):
    worker.error = urllib.error.URLError("offline")
    monitor.result(0, report(subject=False, scene_label="fight"))
    result = preview(monitor)
    assert result == {
        "status": "detector_unavailable",
        "event_state": "alert",
        "objects": [],
    }


def test_grounded_subject_extraction_uses_fired_valid_objects():
    row = report()
    assert _event_subjects(row) == [{"event": "smoking", "bbox": (0.1, 0.1, 0.4, 0.9)}]
    row["objects"][0]["bbox"] = box(0.1, float("inf"))
    assert _event_subjects(row) == []
