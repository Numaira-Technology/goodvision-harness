from __future__ import annotations

import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
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
