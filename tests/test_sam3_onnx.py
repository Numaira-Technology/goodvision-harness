"""Offline SAM3 adapter/HTTP contracts: no weights, network or accelerator."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from goodvision_harness.rich_label_models import resolve_rich_label_model
from goodvision_harness.rich_labeling import (
    RichLabelHTTPClient,
    RichLabelSnapshot,
    validate_mask_contours,
)
from workers.detector_rich_label import app as worker
from workers.detector_rich_label import sam3_onnx as sam
from workers.detector_rich_label.detectors import build_detector


class Session:
    def __init__(self, provider):
        self.providers = [provider]
        self.fallback_disabled = False

    def get_providers(self):
        return self.providers

    def disable_fallback(self):
        self.fallback_disabled = True


def annotation(score=0.9):
    return SimpleNamespace(
        score=score, bounding_box=SimpleNamespace(xmin=20, ymin=10, xmax=59, ymax=39),
        mask=np.ones((30, 40), dtype=bool),
    )


@pytest.fixture()
def fake_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_ONNX_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("SAM3_ONNX_DEVICE", "coreml")
    sessions = {key: Session("CoreMLExecutionProvider") for key in sam.SESSION_KEYS}
    requests = []
    results = [annotation()]

    def generate(request):
        requests.append(request)
        return SimpleNamespace(annotations=results)

    model = SimpleNamespace(_inference_sessions=sessions, generate=generate)
    monkeypatch.setattr(sam, "_load_model", lambda *args: model)
    monkeypatch.setitem(sys.modules, "osam", SimpleNamespace(types=SimpleNamespace(
        GenerateRequest=SimpleNamespace, Prompt=SimpleNamespace,
    )))
    return SimpleNamespace(model=model, results=results, requests=requests,
                           sessions=sessions)


def image():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[0, 0] = [1, 2, 3]
    return frame


def infer(engine, seq=1, session="a", model="sam3-onnx", frame=None):
    return engine.infer(
        image() if frame is None else frame, stream_session_id=session,
        frame_sequence=seq, timestamp_seconds=seq / 10, model_key=model,
    )


def test_rgb_inclusive_bbox_and_source_mask_conversion(fake_model):
    detector = build_detector("sam3-onnx")
    detections = detector.detect(image())
    assert fake_model.requests[-1].image[0, 0].tolist() == [3, 2, 1]
    assert fake_model.requests[-1].image.flags.c_contiguous
    assert fake_model.requests[-1].prompt.texts == ["person"]
    assert detections[0].bbox == (20, 10, 60, 40)
    points = np.asarray(detections[0].mask_polygons[0])
    assert points.min(axis=0).tolist() == [0.1, 0.1]
    assert points.max(axis=0).tolist() == [59 / 200, 39 / 100]
    assert all(s.fallback_disabled for s in fake_model.sessions.values())
    assert detector.health()["ready"] is True
    assert detector.health()["gpu_ready"] is False


def test_masks_are_observed_only_and_sessions_stay_independent(fake_model):
    engine = worker.DetectorEngine()
    first = RichLabelSnapshot.from_dict(infer(engine))
    obj = first.objects[0]
    assert obj.bbox.x2 == 0.3
    assert obj.attributes["mask_contours"]["source_frame_sequence"] == 1
    other = RichLabelSnapshot.from_dict(infer(engine, session="b"))
    assert other.objects[0].track_id == obj.track_id
    fake_model.results.clear()
    predicted = RichLabelSnapshot.from_dict(infer(engine, seq=2))
    assert predicted.objects[0].observation_state == "predicted"
    assert "mask_contours" not in predicted.objects[0].attributes
    fake_model.results.append(annotation())
    current = RichLabelSnapshot.from_dict(infer(engine, seq=3))
    assert current.objects[0].track_id == obj.track_id
    assert current.objects[0].attributes["mask_contours"]["source_frame_sequence"] == 3
    with pytest.raises(ValueError, match="metadata must increase"):
        infer(engine, seq=3)


def test_failure_preserves_source_order_and_recovers_health(fake_model):
    engine = worker.DetectorEngine()
    infer(engine)
    fake_model.sessions["decoder"].providers = ["CPUExecutionProvider"]
    with pytest.raises(RuntimeError, match="provider mismatch"):
        infer(engine, seq=2)
    assert engine.health("sam3-onnx")["ready"] is False
    fake_model.sessions["decoder"].providers = ["CoreMLExecutionProvider"]
    assert engine.health("sam3-onnx")["ready"] is True
    assert RichLabelSnapshot.from_dict(infer(engine, seq=2)).frame_sequence == 2


def test_resolution_changes_drop_old_geometry(fake_model):
    engine = worker.DetectorEngine()
    infer(engine)
    fake_model.results.clear()
    result = RichLabelSnapshot.from_dict(infer(
        engine, seq=2, frame=np.zeros((50, 100, 3), np.uint8),
    ))
    assert result.objects == ()


def test_bbox_only_and_empty_masks(fake_model, monkeypatch):
    monkeypatch.setenv("SAM3_ONNX_MASKS", "false")
    assert not build_detector("sam3-onnx").detect(image())[0].mask_polygons
    monkeypatch.setenv("SAM3_ONNX_MASKS", "true")
    fake_model.results[0].mask[:] = False
    assert not build_detector("sam3-onnx").detect(image())[0].mask_polygons


@pytest.mark.parametrize("bad", ["nan", "box", "mask"])
def test_invalid_model_output_fails_closed(fake_model, bad):
    if bad == "nan":
        fake_model.results[0].score = float("nan")
    elif bad == "box":
        fake_model.results[0].bounding_box.xmax = 999
    else:
        fake_model.results[0].mask = np.zeros((1, 1), dtype=bool)
    detector = build_detector("sam3-onnx")
    with pytest.raises(RuntimeError):
        detector.detect(image())
    assert detector.health()["ready"] is False


def test_contour_complexity_obeys_harness_contract():
    mask = np.zeros((300, 500), np.uint8)
    for y in range(10, 290, 30):
        for x in range(10, 490, 30):
            cv2.circle(mask, (x, y), 9, 1, -1)
    polygons = sam._mask_polygons(mask, xmin=0, ymin=0, width=500, height=300)
    validate_mask_contours({
        "coordinate_system": "normalized", "source_frame_sequence": 1,
        "source_timestamp_seconds": 0.1, "polygons": polygons,
    })
    assert len(polygons) == 16


def test_http_routes_health_and_backpressure(fake_model, monkeypatch):
    monkeypatch.setenv("DETECTOR_MODEL", "sam3-onnx")
    engine = worker.DetectorEngine()
    monkeypatch.setattr(worker, "DetectorEngine", lambda: engine)
    ok, jpeg = cv2.imencode(".jpg", image())
    assert ok
    headers = {
        "Content-Type": "image/jpeg", "x-goodvision-stream-session-id": "http",
        "x-goodvision-frame-sequence": "1", "x-goodvision-timestamp-seconds": "0.1",
    }
    with TestClient(worker.create_app()) as client:
        assert client.get("/health/ready").json()["provider_ready"] is True
        for path in ("/v1/labels", "/v1/tracks"):
            headers["x-goodvision-stream-session-id"] = path
            response = client.post(path, headers=headers, content=jpeg.tobytes())
            assert response.status_code == 200
            RichLabelSnapshot.from_dict(response.json())
        with engine._lock:
            response = client.post(
                "/v1/tracks", headers=headers, content=jpeg.tobytes(),
            )
            assert response.status_code == 429
            assert response.headers["Retry-After"] == "1"
            assert client.get("/health/ready").json()["ready"] is False
        fake_model.sessions["decoder"].providers = ["CPUExecutionProvider"]
        headers["x-goodvision-frame-sequence"] = "2"
        response = client.post(
            "/v1/tracks", headers=headers, content=jpeg.tobytes(),
        )
        assert response.status_code == 503
        assert client.get("/health/tracking").json()["ready"] is False


def test_selected_health_cannot_borrow_default_detector_gpu(fake_model):
    engine = worker.DetectorEngine()
    engine.default_model = "dummy"
    engine.warmup()
    assert engine.health()["gpu_ready"] is True
    fake_model.sessions["decoder"].providers = ["CPUExecutionProvider"]
    assert engine.health("sam3-onnx")["ready"] is False


@pytest.mark.parametrize("device", ["cpu", "cuda", "coreml"])
def test_client_accepts_only_verified_selected_provider(fake_model, device):
    detector = build_detector("sam3-onnx")
    detector.device = device
    for session in fake_model.sessions.values():
        session.providers = [sam.PROVIDERS[device]]
    detector.detect(image())
    payload = detector.health()
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json=payload)

    with RichLabelHTTPClient(
        "http://worker/v1/labels", mode="tracks",
        headers={worker.DETECTOR_MODEL_HEADER: "sam3-onnx"},
        transport=httpx.MockTransport(handle),
    ) as client:
        assert client.gpu_ready()
        assert seen[-1].headers[worker.DETECTOR_MODEL_HEADER] == "sam3-onnx"
        for key, value in [("provider_ready", False), ("ready", False),
                           ("device", []), ("model", "dummy"),
                           ("execution_providers", {})]:
            original = payload[key]
            payload[key] = value
            assert not client.gpu_ready()
            payload[key] = original


@pytest.mark.parametrize("key,value", [
    ("SAM3_ONNX_DEVICE", "auto"), ("SAM3_ONNX_SCORE_THRESHOLD", "nan"),
    ("SAM3_ONNX_MAX_OBJECTS", "0"), ("SAM3_ONNX_MASKS", "yes"),
    ("SAM3_ONNX_MODEL_DIR", ""),
])
def test_configuration_rejected_before_model_load(fake_model, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="configuration"):
        build_detector("sam3-onnx")
    assert not fake_model.requests


def test_local_bundle_verified_without_downloader(tmp_path):
    @dataclass
    class Blob:
        url: str
        hash: str
        attachments: tuple = ()

        @property
        def filename(self):
            return self.url.rsplit("/", 1)[-1]

    content = b"synthetic test graph"
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    base = type("Base", (), {"_blobs": {
        key: Blob(f"https://example.invalid/{key}.onnx", digest)
        for key in sam.SESSION_KEYS
    }})
    local = sam._local_model_type(base, Blob, tmp_path, tmp_path / "cache")
    with pytest.raises(RuntimeError, match="missing local"):
        local.pull()
    for key in sam.SESSION_KEYS:
        (tmp_path / f"{key}.onnx").write_bytes(content)
    local.pull()
    with pytest.raises(RuntimeError, match="downloads are disabled"):
        local._blobs["decoder"].pull()
    (tmp_path / "decoder.onnx").write_bytes(b"wrong graph")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        local.pull()
    assert not (tmp_path / "cache").exists()


def test_registry_keeps_default_and_exposes_optional_component(monkeypatch):
    from goodvision_harness.rich_label_models import DEFAULT_RICH_LABEL_MODEL

    monkeypatch.delenv("DETECTOR_MODEL", raising=False)
    assert worker.DetectorEngine().default_model == "yolo26-s"
    assert DEFAULT_RICH_LABEL_MODEL == "contour"
    selected = resolve_rich_label_model("det-segment")
    assert (selected.backend, selected.detector_model) == ("paddle", "sam3-onnx")
