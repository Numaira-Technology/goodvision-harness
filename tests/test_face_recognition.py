from __future__ import annotations

import builtins
import importlib
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

import goodvision_harness.face_recognition as face_module
from goodvision_harness.face_recognition import (
    FaceGallery,
    FaceRecognitionRuntime,
    FaceRecognizerConfig,
)
from goodvision_harness.sliding_window import FrameWindow, SampledFrame


ROOT = Path(__file__).resolve().parents[1]


def _jpeg(width: int = 320, height: int = 160) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _face(
    bbox: tuple[float, float, float, float],
    embedding: tuple[float, ...],
    *,
    detection_score: float = 0.9,
) -> SimpleNamespace:
    vector = np.asarray(embedding, dtype=np.float32)
    return SimpleNamespace(
        bbox=np.asarray(bbox, dtype=np.float32),
        embedding=vector,
        normed_embedding=vector / np.linalg.norm(vector),
        det_score=detection_score,
    )


class _FakeFaceAnalysis:
    def __init__(self, responses: list[list[Any]]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.image_shapes: list[tuple[int, ...]] = []

    def get(self, image: np.ndarray[Any, Any]) -> list[Any]:
        self.calls += 1
        self.image_shapes.append(tuple(image.shape))
        if not self.responses:
            raise AssertionError("unexpected inference call")
        return self.responses.pop(0)


def _window(jpeg: bytes, *, index: int = 0) -> FrameWindow:
    frame = SampledFrame(
        sample_index=0,
        source_frame_index=7,
        timestamp_seconds=1.25,
        jpeg_bytes=jpeg,
        width=320,
        height=160,
    )
    return FrameWindow(index=index, frames=(frame,))


def _config(tmp_path: Path, **overrides: Any) -> FaceRecognizerConfig:
    values: dict[str, Any] = {
        "model_name": "buffalo_l",
        "model_root": tmp_path / "models",
        "provider": "CPUExecutionProvider",
        "det_size": 640,
        "threshold": 0.8,
        "min_face_size": 50,
        "gallery_path": tmp_path / "gallery.json",
    }
    values.update(overrides)
    return FaceRecognizerConfig(**values)


def test_gallery_is_normalized_persistent_and_model_bound(tmp_path: Path) -> None:
    path = tmp_path / "gallery.json"
    gallery = FaceGallery(path, model_name="buffalo_l")

    first = gallery.enroll("alice", "Alice", [3.0, 4.0])
    second = gallery.enroll("alice", "Alice", [1.0, 0.0])
    gallery.enroll("bob", "Bob", [0.0, 1.0])

    assert first.sample_count == 1
    assert second.sample_count == 2
    assert [person.person_id for person in gallery.list_people()] == ["alice", "bob"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["embedding_profile"] == "local-face-embedding-v1"
    assert "model_name" not in payload
    assert "buffalo" not in path.read_text(encoding="utf-8").lower()
    assert payload["embedding_dimension"] == 2
    assert payload["people"][0]["embeddings"][0] == pytest.approx([0.6, 0.8])

    reloaded = FaceGallery(path, model_name="buffalo_l")
    match = reloaded.match([0.99, 0.01])
    assert match is not None
    assert match.person_id == "alice"
    assert match.similarity > 0.99
    assert reloaded.delete("alice") is True
    assert reloaded.delete("alice") is False
    assert [person.person_id for person in reloaded.list_people()] == ["bob"]

    payload["embedding_profile"] = "incompatible-profile"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="face gallery could not be loaded"):
        FaceGallery(path, model_name="buffalo_l")


def test_legacy_gallery_is_migrated_without_model_name(tmp_path: Path) -> None:
    path = tmp_path / "gallery.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "buffalo_l",
                "embedding_dimension": 2,
                "people": [
                    {
                        "person_id": "alice",
                        "name": "Alice",
                        "embeddings": [[1.0, 0.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    gallery = FaceGallery(path, model_name="buffalo_l")

    assert gallery.list_people()[0].person_id == "alice"
    migrated = path.read_text(encoding="utf-8").lower()
    assert '"schema_version":2' in migrated
    assert "model_name" not in migrated
    assert "buffalo" not in migrated


def test_gallery_mutations_are_thread_safe(tmp_path: Path) -> None:
    gallery = FaceGallery(tmp_path / "gallery.json", model_name="buffalo_l")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(gallery.enroll, "alice", "Alice", [1.0, index + 1.0])
            for index in range(12)
        ]
        for future in futures:
            future.result()

    assert gallery.list_people()[0].sample_count == 12
    reloaded = FaceGallery(gallery.path, model_name="buffalo_l")
    assert reloaded.list_people()[0].sample_count == 12


def test_gallery_rolls_back_when_persistence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery = FaceGallery(tmp_path / "gallery.json", model_name="buffalo_l")
    gallery.enroll("alice", "Alice", [1.0, 0.0])

    def fail() -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(gallery, "_persist_locked", fail)
    with pytest.raises(RuntimeError, match="could not persist"):
        gallery.enroll("bob", "Bob", [0.0, 1.0])
    assert [person.person_id for person in gallery.list_people()] == ["alice"]

    with pytest.raises(RuntimeError, match="could not persist"):
        gallery.delete("alice")
    assert [person.person_id for person in gallery.list_people()] == ["alice"]


def test_runtime_enrolls_and_classifies_every_face(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis(
        [
            [_face((20, 20, 120, 130), (1.0, 0.0))],
            [
                _face((10, 10, 110, 130), (0.99, 0.01), detection_score=0.95),
                _face((120, 10, 220, 130), (0.0, 1.0), detection_score=0.88),
                _face((230, 10, 260, 40), (1.0, 0.0), detection_score=0.82),
            ],
        ]
    )
    loads: list[FaceRecognizerConfig] = []

    def load(config: FaceRecognizerConfig) -> _FakeFaceAnalysis:
        loads.append(config)
        return app

    monkeypatch.setattr(face_module, "_load_face_analysis", load)
    runtime = FaceRecognitionRuntime(_config(tmp_path))

    assert runtime.loaded is False
    enrolled = runtime.enroll("alice", "Alice", _jpeg())
    assert enrolled.as_dict() == {
        "person_id": "alice",
        "name": "Alice",
        "sample_count": 1,
    }
    assert runtime.loaded is True
    assert runtime.model == "numaira-beta-fast"
    assert runtime.event_label == "unknown_person"
    assert runtime.settings.event_label == "unknown_person"

    response = runtime.classify_window(_window(_jpeg()))
    assert response.ok
    classification = response.classification
    assert classification is not None
    assert classification.analysis_kind == "face_recognition"
    assert classification.detected is True
    assert classification.event_object_indexes == (1,)
    assert [item.label for item in classification.objects] == [
        "known_person",
        "unknown_person",
        "unusable_face",
    ]
    known, unknown, unusable = classification.objects
    assert known.attributes["status"] == "known"
    assert known.attributes["person_id"] == "alice"
    assert known.attributes["name"] == "Alice"
    assert known.attributes["similarity"] > 0.99
    assert unknown.attributes["status"] == "unknown"
    assert unknown.attributes["similarity"] == pytest.approx(0.0)
    assert unusable.attributes["status"] == "unusable"
    assert unusable.attributes["reason"] == "face_too_small"
    assert classification.confidence == pytest.approx(0.88)
    assert "1 known, 1 unknown, 1 unusable" in classification.description
    assert len(loads) == 1
    assert app.calls == 2


def test_runtime_warmup_loads_model_and_runs_detector_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis([[]])
    loads: list[FaceRecognizerConfig] = []

    def load(config: FaceRecognizerConfig) -> _FakeFaceAnalysis:
        loads.append(config)
        return app

    monkeypatch.setattr(face_module, "_load_face_analysis", load)
    runtime = FaceRecognitionRuntime(_config(tmp_path, det_size=(320, 240)))

    assert runtime.loaded is False
    assert runtime.warmed is False
    assert runtime.warmup() >= 0.0
    assert runtime.loaded is True
    assert runtime.warmed is True
    assert runtime.warmup() == 0.0
    assert len(loads) == 1
    assert app.calls == 1
    assert app.image_shapes == [(240, 320, 3)]


def test_empty_gallery_marks_usable_face_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis(
        [[_face((20, 20, 120, 130), (1.0, 0.0), detection_score=0.91)]]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path))

    response = runtime.classify_window(_window(_jpeg()))
    classification = response.classification
    assert classification is not None
    assert classification.detected is True
    assert classification.event_object_indexes == (0,)
    assert classification.objects[0].attributes["status"] == "unknown"
    assert "similarity" not in classification.objects[0].attributes


def test_unknown_face_capture_persists_and_reuses_stranger_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_model = _FakeFaceAnalysis(
        [
            [_face((20, 20, 120, 130), (1.0, 0.0), detection_score=0.91)],
            [_face((20, 20, 120, 130), (0.99, 0.01), detection_score=0.92)],
        ]
    )
    restarted_model = _FakeFaceAnalysis(
        [[_face((20, 20, 120, 130), (1.0, 0.0), detection_score=0.93)]]
    )
    models = iter((first_model, restarted_model))
    monkeypatch.setattr(
        face_module,
        "_load_face_analysis",
        lambda _config: next(models),
    )
    config = _config(tmp_path, min_face_size=20)
    runtime = FaceRecognitionRuntime(config)

    first = runtime.classify_window(_window(_jpeg(), index=0)).classification
    second = runtime.classify_window(_window(_jpeg(), index=1)).classification
    assert first is not None and second is not None
    first_id = first.objects[0].attributes["stranger_id"]
    assert second.objects[0].attributes["stranger_id"] == first_id
    assert first.objects[0].attributes["photo_url"] == (
        f"/v1/faces/strangers/{first_id}/photo"
    )
    captures = runtime.list_strangers()
    assert [capture.stranger_id for capture in captures] == [first_id]
    assert captures[0].photo_available is True
    photo = runtime.stranger_photo(first_id)
    assert photo is not None
    decoded = cv2.imdecode(np.frombuffer(photo, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (110, 100)
    assert len(list((tmp_path / "stranger-captures").glob("*.jpg"))) == 1

    restarted = FaceRecognitionRuntime(config)
    after_restart = restarted.classify_window(_window(_jpeg())).classification
    assert after_restart is not None
    assert after_restart.objects[0].attributes["stranger_id"] == first_id


def test_capture_write_failure_does_not_create_id_or_suppress_alert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeFaceAnalysis(
        [[_face((20, 20, 120, 130), (1.0, 0.0), detection_score=0.91)]]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: model)

    def fail_write(_path: Path, _payload: bytes) -> None:
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(face_module, "_atomic_write_bytes", fail_write)
    runtime = FaceRecognitionRuntime(_config(tmp_path, min_face_size=20))

    classification = runtime.classify_window(_window(_jpeg())).classification

    assert classification is not None
    assert classification.detected is True
    assert classification.objects[0].attributes["status"] == "unknown"
    assert "stranger_id" not in classification.objects[0].attributes
    assert runtime.list_strangers() == ()


@pytest.mark.parametrize(
    ("face_width", "expected_status", "expected_detected"),
    [
        (19, "unusable", False),
        (20, "unknown", True),
    ],
)
def test_live_face_size_boundary_controls_alert_and_bbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    face_width: int,
    expected_status: str,
    expected_detected: bool,
) -> None:
    app = _FakeFaceAnalysis(
        [
            [
                _face(
                    (100, 30, 100 + face_width, 50),
                    (1.0, 0.0),
                    detection_score=0.86,
                )
            ]
        ]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path, min_face_size=20))

    response = runtime.classify_window(_window(_jpeg()))
    classification = response.classification
    assert classification is not None
    assert classification.detected is expected_detected
    assert classification.event_object_indexes == (
        (0,) if expected_detected else ()
    )
    face = classification.objects[0]
    assert face.attributes["status"] == expected_status
    assert face.confidence == pytest.approx(0.86)
    assert face.bbox.to_pixels(320, 160) == (100, 30, 100 + face_width, 50)
    if expected_detected:
        assert face.label == "unknown_person"
    else:
        assert face.label == "unusable_face"
        assert face.attributes["reason"] == "face_too_small"


def test_enrollment_keeps_stricter_face_size_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis(
        [[_face((100, 30, 140, 70), (1.0, 0.0), detection_score=0.9)]]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(
        _config(
            tmp_path,
            min_face_size=20,
            enrollment_min_face_size=80,
        )
    )

    with pytest.raises(ValueError, match="enrollment_min_face_size=80"):
        runtime.enroll("alice", "Alice", _jpeg())
    assert runtime.list() == ()


def test_enrollment_rejects_multiple_usable_faces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis(
        [
            [
                _face((10, 10, 100, 120), (1.0, 0.0)),
                _face((120, 10, 220, 120), (0.0, 1.0)),
            ]
        ]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path))

    with pytest.raises(ValueError, match="exactly one usable face"):
        runtime.enroll("alice", "Alice", _jpeg())
    assert runtime.list() == ()


def test_missing_insightface_is_reported_on_first_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def missing(name: str, package: str | None = None) -> Any:
        if name == "insightface.app":
            raise ModuleNotFoundError(
                "No module named 'insightface'",
                name="insightface",
            )
        return real_import(name, package)

    monkeypatch.setattr(face_module.importlib, "import_module", missing)
    runtime = FaceRecognitionRuntime(_config(tmp_path))
    assert runtime.loaded is False

    with pytest.raises(
        RuntimeError,
        match="local face analysis is unavailable",
    ):
        runtime.classify_window(_window(_jpeg()))
    assert runtime.loaded is False


def test_missing_model_does_not_auto_download_without_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    class UnexpectedFaceAnalysis:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("FaceAnalysis must not run before the license gate")

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "insightface.app":
            return SimpleNamespace(FaceAnalysis=UnexpectedFaceAnalysis)
        if name == "onnxruntime":
            return SimpleNamespace(
                get_available_providers=lambda: ["CPUExecutionProvider"]
            )
        return real_import(name, package)

    monkeypatch.setattr(face_module.importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="automatic download is disabled"):
        face_module._load_face_analysis(_config(tmp_path))
    assert not (tmp_path / "models").exists()


def test_explicit_opt_in_allows_face_analysis_to_install_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module
    prepared: list[tuple[int, tuple[int, int]]] = []

    class FakeFaceAnalysis:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["name"] == "buffalo_l"
            assert kwargs["root"] == str(tmp_path / "models")

        def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
            prepared.append((ctx_id, det_size))

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "insightface.app":
            return SimpleNamespace(FaceAnalysis=FakeFaceAnalysis)
        if name == "onnxruntime":
            return SimpleNamespace(
                get_available_providers=lambda: ["CPUExecutionProvider"]
            )
        return real_import(name, package)

    monkeypatch.setattr(face_module.importlib, "import_module", fake_import)

    model = face_module._load_face_analysis(
        _config(tmp_path, allow_model_download=True)
    )
    assert isinstance(model, FakeFaceAnalysis)
    assert prepared == [(-1, (640, 640))]


def test_cuda_face_analysis_preloads_and_verifies_real_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module
    preloads: list[str] = []
    prepared: list[tuple[int, tuple[int, int]]] = []

    class FakeSession:
        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    class FakeFaceAnalysis:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["providers"] == ["CUDAExecutionProvider"]
            self.models = {
                "detection": SimpleNamespace(session=FakeSession()),
                "recognition": SimpleNamespace(session=FakeSession()),
            }

        def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
            prepared.append((ctx_id, det_size))

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "insightface.app":
            return SimpleNamespace(FaceAnalysis=FakeFaceAnalysis)
        if name == "onnxruntime":
            return SimpleNamespace(
                preload_dlls=lambda *, directory: preloads.append(directory),
                get_available_providers=lambda: [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
        return real_import(name, package)

    monkeypatch.setattr(face_module.importlib, "import_module", fake_import)

    model = face_module._load_face_analysis(
        _config(
            tmp_path,
            allow_model_download=True,
            provider="CUDAExecutionProvider",
        )
    )

    assert isinstance(model, FakeFaceAnalysis)
    assert preloads == [""]
    assert prepared == [(0, (640, 640))]


def test_cuda_face_analysis_rejects_silent_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    class FakeSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    class FakeFaceAnalysis:
        def __init__(self, **_kwargs: Any) -> None:
            self.models = {
                "detection": SimpleNamespace(session=FakeSession()),
                "recognition": SimpleNamespace(session=FakeSession()),
            }

        def prepare(self, **_kwargs: Any) -> None:
            pass

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "insightface.app":
            return SimpleNamespace(FaceAnalysis=FakeFaceAnalysis)
        if name == "onnxruntime":
            return SimpleNamespace(
                preload_dlls=lambda **_kwargs: None,
                get_available_providers=lambda: [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
        return real_import(name, package)

    monkeypatch.setattr(face_module.importlib, "import_module", fake_import)

    with pytest.raises(
        RuntimeError,
        match="local face analysis could not be initialized",
    ):
        face_module._load_face_analysis(
            _config(
                tmp_path,
                allow_model_download=True,
                provider="CUDAExecutionProvider",
            )
        )


def test_directml_face_analysis_uses_and_verifies_real_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module
    prepared: list[tuple[int, tuple[int, int]]] = []

    class FakeSession:
        def get_providers(self) -> list[str]:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    class FakeFaceAnalysis:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["providers"] == ["DmlExecutionProvider"]
            self.models = {
                "detection": SimpleNamespace(session=FakeSession()),
                "recognition": SimpleNamespace(session=FakeSession()),
            }

        def prepare(self, *, ctx_id: int, det_size: tuple[int, int]) -> None:
            prepared.append((ctx_id, det_size))

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "insightface.app":
            return SimpleNamespace(FaceAnalysis=FakeFaceAnalysis)
        if name == "onnxruntime":
            return SimpleNamespace(
                get_available_providers=lambda: [
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
        return real_import(name, package)

    monkeypatch.setattr(face_module.importlib, "import_module", fake_import)

    model = face_module._load_face_analysis(
        _config(
            tmp_path,
            allow_model_download=True,
            provider="DmlExecutionProvider",
        )
    )

    assert isinstance(model, FakeFaceAnalysis)
    assert prepared == [(0, (640, 640))]


def test_directml_face_analysis_rejects_silent_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    class FakeSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    class FakeFaceAnalysis:
        def __init__(self, **_kwargs: Any) -> None:
            self.models = {
                "detection": SimpleNamespace(session=FakeSession()),
                "recognition": SimpleNamespace(session=FakeSession()),
            }

        def prepare(self, **_kwargs: Any) -> None:
            pass

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "insightface.app":
            return SimpleNamespace(FaceAnalysis=FakeFaceAnalysis)
        if name == "onnxruntime":
            return SimpleNamespace(
                get_available_providers=lambda: [
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
        return real_import(name, package)

    monkeypatch.setattr(face_module.importlib, "import_module", fake_import)

    with pytest.raises(
        RuntimeError,
        match="local face analysis could not be initialized",
    ):
        face_module._load_face_analysis(
            _config(
                tmp_path,
                allow_model_download=True,
                provider="DmlExecutionProvider",
            )
        )


def test_face_gpu_image_and_compose_are_isolated_from_cpu_delivery() -> None:
    cpu_dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    gpu_dockerfile = (ROOT / "Dockerfile.face-gpu").read_text(encoding="utf-8")
    dev_overlay = (ROOT / "compose.face-gpu.yaml").read_text(encoding="utf-8")
    delivery_overlay = (ROOT / "compose.face-gpu.delivery.yaml").read_text(
        encoding="utf-8"
    )
    smoke = (ROOT / "tools" / "face_gpu_smoke.py").read_text(encoding="utf-8")

    assert "onnxruntime-gpu" not in cpu_dockerfile
    assert ".[face-gpu,webrtc]" in gpu_dockerfile
    assert "FACE_PROVIDER=CUDAExecutionProvider" in gpu_dockerfile
    assert "face_gpu_smoke.py" in gpu_dockerfile
    assert "FROM python:3.11-slim AS runtime" in gpu_dockerfile
    assert "FROM runtime AS customer" in gpu_dockerfile
    assert gpu_dockerfile.rstrip().endswith("FROM runtime AS development")
    assert "python tools/face_gpu_smoke.py --runtime" in gpu_dockerfile
    assert "/opt/goodvision/internal/face_check.py --runtime" in gpu_dockerfile
    assert "COPY tools/customer_check.py ./tools/check_runtime.py" in gpu_dockerfile
    assert "COPY tools/customer_check.py ./tools/check_face.py" in gpu_dockerfile
    assert "runtime_check.py" not in gpu_dockerfile
    for overlay in (dev_overlay, delivery_overlay):
        assert "FACE_PROVIDER: CUDAExecutionProvider" in overlay
        assert "capabilities: [gpu]" in overlay
    assert "runtime.warmup()" in smoke
    assert 'print("face_gpu_smoke=ready")' in smoke


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"threshold": 1.1}, "raw cosine"),
        ({"min_face_size": 0}, "positive integer"),
        ({"enrollment_min_face_size": 0}, "positive integer"),
        ({"det_size": (640, 0)}, "positive integers"),
        ({"provider": "UnknownExecutionProvider"}, "provider must be one of"),
    ],
)
def test_config_rejects_invalid_values(
    tmp_path: Path,
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, **overrides)


def test_config_rejects_non_boolean_model_download_opt_in(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="true or false"):
        _config(tmp_path, allow_model_download="yes")


# Runtime-only dependency audits never load model weights.
FACE_RUNTIME_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "face_gpu_smoke.py"
ORT_EXCEPTION = "insightface 1.0.1 requires onnxruntime, which is not installed."
CV_EXCEPTION = "insightface 1.0.1 requires opencv-python, which is not installed."
VALID_OUTPUT = f"{ORT_EXCEPTION}\n{CV_EXCEPTION}\n"


def _load_runtime_check() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "tested_face_gpu_smoke", FACE_RUNTIME_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime_check() -> ModuleType:
    return _load_runtime_check()


def test_runtime_check_import_does_not_import_inference_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    inference_modules = {"cv2", "insightface", "numpy", "onnxruntime"}

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", 1)[0] in inference_modules:
            raise AssertionError(
                f"unexpected import while loading audit helper: {name}"
            )
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", guarded_import)
        module = _load_runtime_check()
    assert callable(module.validate_dependency_audit)


@pytest.mark.parametrize(
    "output",
    [
        VALID_OUTPUT,
        f"{CV_EXCEPTION}\n{ORT_EXCEPTION}",
        f"\n  {ORT_EXCEPTION}\r\n\t{CV_EXCEPTION}  \r\n\n",
    ],
    ids=["declared-order", "reverse-order", "blank-lines-and-whitespace"],
)
def test_dependency_audit_accepts_only_the_two_declared_exceptions(
    runtime_check: ModuleType, output: str,
) -> None:
    assert runtime_check.validate_dependency_audit(1, output) is None


@pytest.mark.parametrize(
    "output",
    [
        "",
        "\n  \n",
        ORT_EXCEPTION,
        CV_EXCEPTION,
        f"{ORT_EXCEPTION}\n{ORT_EXCEPTION}",
        f"{CV_EXCEPTION}\n{CV_EXCEPTION}",
        VALID_OUTPUT + ORT_EXCEPTION,
        VALID_OUTPUT + CV_EXCEPTION,
        VALID_OUTPUT
        + "some-package 1.0 requires missing-package, which is not installed.",
        "unknown dependency error",
        "No broken requirements found.",
        f"{ORT_EXCEPTION}\n"
        "insightface 1.0.0 requires opencv-python, which is not installed.",
        f"{ORT_EXCEPTION}\n"
        "insightface 1.0.1 requires opencv-python>=6, which is not installed.",
    ],
    ids=[
        "empty", "whitespace-only", "missing-opencv", "missing-ort",
        "duplicate-ort-only", "duplicate-opencv-only", "extra-duplicate-ort",
        "extra-duplicate-opencv", "extra-unrelated-error", "unknown-error",
        "unexpected-clean-check", "changed-insightface-version",
        "changed-dependency-requirement",
    ],
)
def test_dependency_audit_rejects_missing_extra_duplicate_or_unknown_lines(
    runtime_check: ModuleType, output: str,
) -> None:
    with pytest.raises(RuntimeError, match="Unexpected dependency audit result"):
        runtime_check.validate_dependency_audit(1, output)


@pytest.mark.parametrize("returncode", [0, -1, 2, 127, 137])
def test_dependency_audit_rejects_wrong_returncode_even_with_expected_lines(
    runtime_check: ModuleType, returncode: int,
) -> None:
    with pytest.raises(RuntimeError, match="Unexpected dependency audit result"):
        runtime_check.validate_dependency_audit(returncode, VALID_OUTPUT)


def test_face_runtime_mode_audits_packages_without_starting_models(
    runtime_check: ModuleType, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runtime_check.metadata, "version", runtime_check.EXPECTED_VERSIONS.__getitem__
    )

    def absent(_name: str) -> None:
        raise runtime_check.metadata.PackageNotFoundError

    def no_models() -> None:
        raise AssertionError("runtime audit must not start face models")

    monkeypatch.setattr(runtime_check.metadata, "distribution", absent)
    monkeypatch.setattr(runtime_check, "check_models", no_models)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        __version__="4.14.0", IMREAD_COLOR=1,
        imencode=lambda *_args: (True, b"synthetic-jpeg"),
        imdecode=lambda *_args: SimpleNamespace(shape=(32, 32, 3)),
    ))
    monkeypatch.setitem(sys.modules, "insightface", SimpleNamespace(
        app=SimpleNamespace(FaceAnalysis=object),
    ))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        __version__="1.26.0",
        get_available_providers=lambda: ["CUDAExecutionProvider"],
    ))
    monkeypatch.setattr(runtime_check.subprocess, "run", lambda *_args, **_kwargs: (
        SimpleNamespace(returncode=1, stdout=VALID_OUTPUT, stderr="")
    ))

    assert runtime_check.main(["--runtime"]) == 0
    assert json.loads(capsys.readouterr().out)["real_gpu_inference_tested"] is False


def test_face_default_mode_still_runs_real_model_check(
    runtime_check: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(runtime_check, "check_models", lambda: calls.append("models"))
    monkeypatch.setattr(runtime_check, "check_runtime", lambda: calls.append("runtime"))
    assert runtime_check.main([]) == 0
    assert calls == ["models"]


def test_face_validation_failures_return_nonzero_without_raw_exception(
    runtime_check: ModuleType, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> None:
        raise RuntimeError("private provider failure")

    monkeypatch.setattr(runtime_check, "check_runtime", fail)
    assert runtime_check.main(["--runtime"]) == 1
    assert capsys.readouterr().out == "ERROR face validation failed\n"
    assert runtime_check.main(["--unexpected"]) == 2
