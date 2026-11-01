"""Exercise the pinned OSAM preprocessing/decoding with synthetic ORT outputs.

Install .[dev,sam3-onnx] to run. No real model weights or inference are involved.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from workers.detector_rich_label import sam3_onnx as adapter


def test_pinned_osam_contract_without_weights(monkeypatch, tmp_path):
    osam = pytest.importorskip("osam")
    from osam.types import _model as model_module

    def forbidden(*args, **kwargs):
        pytest.fail("model downloading must never be called")

    monkeypatch.setattr(osam.types.Blob, "pull", forbidden)
    monkeypatch.setattr(osam.types.Model, "pull", forbidden)
    base = osam.apis.get_model_type_by_name("sam3")
    bundle = {}
    for key in adapter.SESSION_KEYS:
        content = key.encode()
        filename = f"{key}.onnx"
        (tmp_path / filename).write_bytes(content)
        bundle[key] = osam.types.Blob(
            url=f"https://example.invalid/{filename}",
            hash="sha256:" + hashlib.sha256(content).hexdigest(),
        )
    monkeypatch.setattr(base, "_blobs", bundle)
    feeds = {}

    class Session:
        def __init__(self, key):
            self.key = key

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def disable_fallback(self):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="box_coords", shape=[1, 1, 4])]

        def run(self, names, feed):
            feeds[self.key] = feed
            if self.key == "image_encoder":
                return [np.zeros((1, 1, 1, 1), np.float32) for _ in range(9)]
            if self.key == "language_encoder":
                return [np.zeros((1, 32), bool), np.zeros((32, 1, 1), np.float32)]
            return [np.array([[0.1, 0.2, 0.5, 0.8]], np.float32),
                    np.array([0.9], np.float32), np.ones((1, 1, 8, 8), np.float32)]

    def create_session(blob, providers):
        assert providers == ["CPUExecutionProvider"]
        assert blob.path.startswith(str(tmp_path))
        return Session(blob.filename.removesuffix(".onnx"))

    monkeypatch.setattr(model_module, "_create_device_session", create_session)
    monkeypatch.setenv("SAM3_ONNX_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("SAM3_ONNX_DEVICE", "cpu")
    detector = adapter.Sam3OnnxDetector()
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:, :] = [1, 2, 3]
    detection, = detector.detect(frame)
    assert feeds["image_encoder"]["image"].shape == (3, 1008, 1008)
    assert feeds["image_encoder"]["image"][:, 0, 0].tolist() == [3, 2, 1]
    assert feeds["language_encoder"]["tokens"].shape == (1, 32)
    assert detection.bbox == (20, 20, 101, 81)
    assert detection.mask_polygons
    assert detector.health()["provider_ready"] is True
    assert set(feeds) == adapter.SESSION_KEYS


def test_real_package_missing_bundle_fails_without_network(monkeypatch, tmp_path):
    osam = pytest.importorskip("osam")

    def forbidden(*args, **kwargs):
        pytest.fail("model downloading must never be called")

    monkeypatch.setattr(osam.types.Blob, "pull", forbidden)
    monkeypatch.setattr(osam.types.Model, "pull", forbidden)
    with pytest.raises(RuntimeError, match="missing local SAM3 model"):
        adapter._load_model(tmp_path, tmp_path / "cache", "cpu")
    assert not list(tmp_path.iterdir())
