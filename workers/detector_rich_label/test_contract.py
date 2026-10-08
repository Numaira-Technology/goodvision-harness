"""The detector worker's /v1/labels output must satisfy the harness contract.

Uses DETECTOR_MODEL=dummy so no model weights / GPU are needed. Proves:
  1. the response parses into goodvision_harness.rich_labeling.RichLabelSnapshot;
  2. track_ids stay stable across frames (IoU tracker continuity);
  3. health endpoints respond.
Run: PYTHONPATH=src python -m pytest workers/detector_rich_label/test_contract.py
"""

import importlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))              # tracker / detectors / app
sys.path.insert(0, str(_ROOT / "src"))      # goodvision_harness

os.environ["DETECTOR_MODEL"] = "dummy"
import app as worker_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from goodvision_harness.rich_labeling import RichLabelSnapshot  # noqa: E402


def _jpeg(w=320, h=240) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _headers(seq: int, ts: float) -> dict:
    return {
        "x-goodvision-stream-session-id": "sess-1",
        "x-goodvision-frame-sequence": str(seq),
        "x-goodvision-timestamp-seconds": str(ts),
        "Content-Type": "image/jpeg",
    }


@pytest.fixture()
def client():
    importlib.reload(worker_app)  # rebuild app with DETECTOR_MODEL=dummy
    with TestClient(worker_app.app) as c:
        yield c


def test_response_parses_into_harness_snapshot(client):
    r = client.post("/v1/labels", headers=_headers(1, 0.10), content=_jpeg())
    assert r.status_code == 200, r.text
    snap = RichLabelSnapshot.from_dict(r.json())  # the real harness parser
    assert snap.stream_session_id == "sess-1"
    assert snap.frame_sequence == 1
    assert len(snap.objects) == 1
    obj = snap.objects[0]
    assert obj.label == "person"
    assert obj.track_id.startswith("P")
    assert 0.0 <= obj.bbox.x1 < obj.bbox.x2 <= 1.0


def test_track_id_is_stable_across_frames(client):
    ids = []
    for i in range(1, 4):
        r = client.post("/v1/labels", headers=_headers(i, i * 0.1), content=_jpeg())
        assert r.status_code == 200, r.text
        snap = RichLabelSnapshot.from_dict(r.json())
        ids.append(snap.objects[0].track_id)
    assert len(set(ids)) == 1  # same person -> one stable track_id


def test_tracks_route_carries_observation_metadata(client):
    # The harness RichLabelHTTPClient uses mode="tracks" -> POST /v1/tracks and
    # rejects any object missing observation_state. This worker must serve that
    # route and populate the tracking metadata (the /v1/labels-only gap that
    # made every harness rich-label call fail).
    r = client.post("/v1/tracks", headers=_headers(1, 0.10), content=_jpeg())
    assert r.status_code == 200, r.text
    snap = RichLabelSnapshot.from_dict(r.json())
    assert len(snap.objects) == 1
    for obj in snap.objects:
        assert obj.observation_state is not None
        assert obj.track_id.startswith("P")


def test_console_selection_switches_model_live(monkeypatch):
    # The harness sends the console-selected model per request (X-GoodVision-
    # Detector-Model). The worker must run that model, cache it, and reset the
    # tracker when the active model changes mid-session.
    importlib.reload(worker_app)
    from tracker import Detection

    built: list[str] = []
    used: list[str] = []

    class _Fake:
        def __init__(self, key: str) -> None:
            self.name = key
            self.gpu_ready = True

        def warmup(self) -> None:
            return None

        def detect(self, _img):
            used.append(self.name)
            return (Detection("person", 0.9, (10.0, 10.0, 50.0, 80.0)),)

    def _factory(key: str):
        built.append(key)
        return _Fake(key)

    monkeypatch.setattr(worker_app, "build_detector", _factory)
    eng = worker_app.DetectorEngine()  # default = yolo26-s
    img = np.zeros((100, 100, 3), np.uint8)

    def _infer(seq, ts, model):
        return eng.infer(img, stream_session_id="s", frame_sequence=seq,
                         timestamp_seconds=ts, model_key=model)

    r1 = _infer(1, 0.1, "yolov8-m")
    assert eng._active_model == "yolov8-m" and used[-1] == "yolov8-m"
    r2 = _infer(2, 0.2, "yolov8-m")
    assert r2["objects"][0]["track_id"] == r1["objects"][0]["track_id"]  # stable
    assert built.count("yolov8-m") == 1  # cached, not rebuilt

    r3 = _infer(3, 0.3, "rfdetr-l")  # switch -> tracker resets, ids restart
    assert eng._active_model == "rfdetr-l" and used[-1] == "rfdetr-l"
    assert r3["objects"][0]["track_id"] == r1["objects"][0]["track_id"] == "P0001"

    with pytest.raises(ValueError):
        _infer(4, 0.4, "no-such-model")  # unknown -> rejected (becomes HTTP 400)


def test_bad_metadata_and_health(client):
    # non-increasing frame metadata is rejected
    client.post("/v1/labels", headers=_headers(5, 0.5), content=_jpeg())
    r = client.post("/v1/labels", headers=_headers(5, 0.5), content=_jpeg())
    assert r.status_code == 400
    assert client.get("/health/ready").json()["ready"] is True
