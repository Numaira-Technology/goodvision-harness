from __future__ import annotations

import builtins
import importlib
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

import goodvision_harness.face_recognition as face_module
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.face_recognition import (
    FaceGallery,
    FaceRecognitionRuntime,
    FaceRecognitionSession,
    FaceRecognizerConfig,
)
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
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


def test_unknown_faces_do_not_create_photos_or_persistent_stranger_gallery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeFaceAnalysis(
        [[_face((20, 20, 120, 130), (1.0, 0.0))] for _ in range(2)]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: model)
    config = _config(tmp_path)
    legacy = config.gallery_path.with_name(f"{config.gallery_path.stem}-strangers.json")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy malformed state must never be loaded", encoding="utf-8")
    runtime = FaceRecognitionRuntime(config)
    for index in range(2):
        result = runtime.classify_window(_window(_jpeg(), index=index)).classification
        assert result is not None and result.detected
        assert "stranger_id" not in result.objects[0].attributes
        assert "photo_url" not in result.objects[0].attributes
    assert not (tmp_path / "stranger-captures").exists()
    assert (
        legacy.read_text(encoding="utf-8")
        == "legacy malformed state must never be loaded"
    )
    assert not hasattr(runtime, "stranger_gallery")


def test_unknown_classification_does_not_write_any_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeFaceAnalysis([[_face((20, 20, 120, 130), (1.0, 0.0))]])
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: model)
    runtime = FaceRecognitionRuntime(_config(tmp_path))

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("recognition attempted persistent enrollment")

    monkeypatch.setattr(runtime.gallery, "enroll", forbidden)
    result = runtime.classify_window(_window(_jpeg())).classification
    assert result is not None and result.detected
    assert not list(tmp_path.rglob("*.jpg"))
    assert not list(tmp_path.rglob("*-strangers.json"))


@pytest.mark.parametrize(
    ("minimum", "face_width", "face_height", "expected_detected"),
    [
        (20, 19, 40, False),
        (20, 40, 19, False),
        (20, 20, 30, True),
        (20, 30, 20, True),
        (20, 55, 67, True),
        (20, 59, 59, True),
        (20, 60, 60, True),
        (32, 31, 40, False),
        (32, 32, 40, True),
        (64, 63, 80, False),
        (64, 64, 80, True),
    ],
)
def test_live_face_size_boundary_controls_alert_and_bbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    minimum: int,
    face_width: int,
    face_height: int,
    expected_detected: bool,
) -> None:
    app = _FakeFaceAnalysis(
        [
            [
                _face(
                    (100, 30, 100 + face_width, 30 + face_height),
                    (1.0, 0.0),
                    detection_score=0.86,
                )
            ]
        ]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path, min_face_size=minimum))

    response = FaceRecognitionSession(runtime).classify_window(_window(_jpeg()))
    classification = response.classification
    assert classification is not None
    assert classification.detected is expected_detected
    assert classification.event_object_indexes == ((0,) if expected_detected else ())
    face = classification.objects[0]
    assert face.attributes["status"] == ("unknown" if expected_detected else "unusable")
    assert face.confidence == pytest.approx(0.86)
    assert face.bbox.to_pixels(320, 160) == (
        100, 30, 100 + face_width, 30 + face_height,
    )
    if expected_detected:
        assert face.label == "unknown_person"
        assert face.attributes["unknown_confirmations"] == 1
    else:
        assert face.label == "unusable_face"
        assert face.attributes["reason"] == "face_too_small"


def _timed_window(index: int) -> FrameWindow:
    frame = replace(
        _window(_jpeg()).newest_frame,
        sample_index=index,
        source_frame_index=index * 10,
        timestamp_seconds=index * 0.4,
    )
    return FrameWindow(index=index, frames=(frame,))


def test_session_keeps_known_person_for_distant_and_low_quality_faces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis(
        [
            [_face((20, 20, 100, 100), (1.0, 0.0))],
            [_face((30, 30, 85, 85), (0.0, 1.0))],
            [_face((30, 30, 85, 85), (0.0, 1.0), detection_score=0.5)],
        ]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path, min_face_size=20))
    runtime.gallery.enroll("alice", "Alice", (1.0, 0.0))
    session = FaceRecognitionSession(runtime)
    for index in range(3):
        result = session.classify_window(_timed_window(index)).classification
        assert result is not None and not result.detected
        assert result.event_object_indexes == ()
        assert result.objects[0].attributes["person_id"] == "alice"
        assert result.objects[0].attributes["identity_source"] == (
            "face_match" if index == 0 else "track_history"
        )


def test_session_reports_each_reliable_stranger_frame_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis([[_face((20, 20, 100, 100), (0.0, 1.0))] for _ in range(3)])
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path))
    session = FaceRecognitionSession(runtime)
    for index in range(3):
        result = session.classify_window(_timed_window(index)).classification
        assert result is not None
        assert result.detected
        assert result.event_object_indexes == (0,)
        assert result.objects[0].label == "unknown_person"


def test_session_keeps_stranger_annotation_through_poor_and_missing_faces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis(
        [
            *[[_face((20, 20, 100, 100), (0.0, 1.0))] for _ in range(3)],
            [_face((30, 30, 85, 85), (0.0, 1.0), detection_score=0.5)],
            [],
        ]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path))
    session = FaceRecognitionSession(runtime)
    results = [
        session.classify_window(_timed_window(index)).classification
        for index in range(5)
    ]

    assert all(result is not None for result in results)
    confirmed = results[2]
    assert confirmed is not None
    face_track_id = confirmed.objects[0].attributes["face_track_id"]
    for index in (3, 4):
        result = results[index]
        assert result is not None and result.detected
        assert result.event_object_indexes == (0,)
        assert len(result.objects) == 1
        assert result.objects[0].label == "unknown_person"
        attributes = result.objects[0].attributes
        assert attributes["face_track_id"] == face_track_id
        assert attributes["identity_source"] == "track_history"
        assert attributes["identity_age_seconds"] == pytest.approx(index * 0.4 - 0.8)
    assert not (tmp_path / "stranger-captures").exists()


def test_session_retained_stranger_keeps_its_track_when_another_face_arrives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis(
        [
            *[[_face((20, 20, 100, 100), (1.0, 0.0))] for _ in range(3)],
            [_face((210, 20, 290, 100), (0.0, 1.0))],
        ]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path))
    session = FaceRecognitionSession(runtime)
    for index in range(3):
        confirmed = session.classify_window(_timed_window(index)).classification
    assert confirmed is not None
    original_id = confirmed.objects[0].attributes["face_track_id"]

    result = session.classify_window(_timed_window(3)).classification

    assert result is not None and result.detected
    assert len(result.objects) == 2
    assert result.event_object_indexes == (0, 1)
    newcomer, retained = result.objects
    assert newcomer.attributes["status"] == "unknown"
    assert newcomer.attributes["unknown_confirmations"] == 1
    assert "stranger_id" not in newcomer.attributes
    assert retained.attributes["status"] == "unknown"
    assert retained.attributes["face_track_id"] == original_id
    assert retained.attributes["identity_source"] == "track_history"
    assert retained.bbox == confirmed.objects[0].bbox


def test_session_stranger_annotation_with_body_expires_after_three_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis(
        [
            *[[_face((20, 20, 100, 100), (1.0, 0.0))] for _ in range(3)],
            *[[] for _ in range(8)],
        ]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path))
    body = RichLabelObject(
        "P0001",
        "person",
        NormalizedBBox(0, 0, 0.5, 1),
        confidence=0.9,
    )
    snapshots: list[RichLabelSnapshot] = []
    coordinator = SimpleNamespace(history=lambda: tuple(snapshots))
    session = FaceRecognitionSession(runtime)
    session.bind_rich_labels("session", coordinator)
    for index in range(11):
        snapshots.append(RichLabelSnapshot("session", index * 10, index * 0.4, (body,)))
        result = session.classify_window(_timed_window(index)).classification
        assert result is not None
        assert result.detected is (index <= 9)
        if 3 <= index <= 9:
            assert result.objects[0].attributes["identity_source"] == "track_history"
            assert result.objects[0].attributes["person_track_id"] == "P0001"
        if index == 10:
            assert result.objects == ()
            assert result.event_object_indexes == ()


@pytest.mark.parametrize("kind", ["low_score", "extreme_angle", "invalid_landmarks"])
def test_unreliable_face_never_creates_stranger_alert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    face = _face((20, 20, 100, 100), (0.0, 1.0))
    if kind == "low_score":
        face.det_score = 0.5
    elif kind == "extreme_angle":
        face.kps = [[50, 40], [55, 40], [75, 60], [50, 80], [55, 80]]
    else:
        face.kps = [[float("nan"), 40]] * 5
    app = _FakeFaceAnalysis([[face]])
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path))
    result = runtime.classify_window(_timed_window(0)).classification
    assert result is not None and not result.detected
    assert result.objects[0].attributes["status"] == "unusable"


def test_small_face_can_still_positively_match_enrolled_person(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis([[_face((20, 20, 45, 45), (1.0, 0.0))]])
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path, min_face_size=20))
    runtime.gallery.enroll("alice", "Alice", (1.0, 0.0))
    result = runtime.classify_window(_timed_window(0)).classification
    assert result is not None and not result.detected
    assert result.objects[0].attributes["person_id"] == "alice"


def test_session_uses_raw_body_history_and_tolerates_brief_disappearance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis(
        [
            [_face((20, 20, 100, 100), (1.0, 0.0))],
            [_face((20, 20, 100, 100), (0.0, 1.0), detection_score=0.5)],
        ]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path))
    runtime.gallery.enroll("alice", "Alice", (1.0, 0.0))
    body = RichLabelObject(
        "P0001", "person", NormalizedBBox(0, 0, 0.5, 1), confidence=0.9
    )
    snapshots = [RichLabelSnapshot("session", 0, 0.0, (body,))]
    coordinator = SimpleNamespace(history=lambda: tuple(snapshots))
    session = FaceRecognitionSession(runtime)
    session.bind_rich_labels("session", coordinator)
    first = session.classify_window(_timed_window(0)).classification
    assert first is not None
    assert first.objects[0].attributes["person_track_id"] == "P0001"
    snapshots.extend(
        [
            RichLabelSnapshot("session", 5, 0.2, ()),
            RichLabelSnapshot("session", 10, 0.4, (body,)),
        ]
    )
    result = session.classify_window(_timed_window(1)).classification
    assert result is not None and not result.detected
    assert result.objects[0].attributes["status"] == "known"
    assert result.objects[0].attributes["person_id"] == "alice"


def test_new_session_shares_gallery_but_never_inherits_identity_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _FakeFaceAnalysis(
        [
            [_face((20, 20, 100, 100), (1.0, 0.0))],
            [_face((20, 20, 100, 100), (0.0, 1.0), detection_score=0.5)],
        ]
    )
    monkeypatch.setattr(face_module, "_load_face_analysis", lambda _config: app)
    runtime = FaceRecognitionRuntime(_config(tmp_path))
    runtime.gallery.enroll("alice", "Alice", (1.0, 0.0))
    first = FaceRecognitionSession(runtime)
    first.classify_window(_timed_window(0))
    result = (
        FaceRecognitionSession(runtime).classify_window(_timed_window(0)).classification
    )
    assert result is not None and not result.detected
    assert result.objects[0].attributes["status"] == "unusable"
    assert "person_id" not in result.objects[0].attributes
    assert len(runtime.list_people()) == 1


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
    assert ".[face-gpu,webrtc,vlm]" in gpu_dockerfile
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
FACE_RUNTIME_SCRIPT = (
    Path(__file__).resolve().parents[1] / "tools" / "face_gpu_smoke.py"
)
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
    runtime_check: ModuleType,
    output: str,
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
        "empty",
        "whitespace-only",
        "missing-opencv",
        "missing-ort",
        "duplicate-ort-only",
        "duplicate-opencv-only",
        "extra-duplicate-ort",
        "extra-duplicate-opencv",
        "extra-unrelated-error",
        "unknown-error",
        "unexpected-clean-check",
        "changed-insightface-version",
        "changed-dependency-requirement",
    ],
)
def test_dependency_audit_rejects_missing_extra_duplicate_or_unknown_lines(
    runtime_check: ModuleType,
    output: str,
) -> None:
    with pytest.raises(RuntimeError, match="Unexpected dependency audit result"):
        runtime_check.validate_dependency_audit(1, output)


@pytest.mark.parametrize("returncode", [0, -1, 2, 127, 137])
def test_dependency_audit_rejects_wrong_returncode_even_with_expected_lines(
    runtime_check: ModuleType,
    returncode: int,
) -> None:
    with pytest.raises(RuntimeError, match="Unexpected dependency audit result"):
        runtime_check.validate_dependency_audit(returncode, VALID_OUTPUT)


def test_face_runtime_mode_audits_packages_without_starting_models(
    runtime_check: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            __version__="4.14.0",
            IMREAD_COLOR=1,
            imencode=lambda *_args: (True, b"synthetic-jpeg"),
            imdecode=lambda *_args: SimpleNamespace(shape=(32, 32, 3)),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "insightface",
        SimpleNamespace(
            app=SimpleNamespace(FaceAnalysis=object),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            __version__="1.26.0",
            get_available_providers=lambda: ["CUDAExecutionProvider"],
        ),
    )
    monkeypatch.setattr(
        runtime_check.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout=VALID_OUTPUT, stderr=""
        ),
    )

    assert runtime_check.main(["--runtime"]) == 0
    assert json.loads(capsys.readouterr().out)["real_gpu_inference_tested"] is False


def test_face_default_mode_still_runs_real_model_check(
    runtime_check: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(runtime_check, "check_models", lambda: calls.append("models"))
    monkeypatch.setattr(runtime_check, "check_runtime", lambda: calls.append("runtime"))
    assert runtime_check.main([]) == 0
    assert calls == ["models"]


def test_face_validation_failures_return_nonzero_without_raw_exception(
    runtime_check: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail() -> None:
        raise RuntimeError("private provider failure")

    monkeypatch.setattr(runtime_check, "check_runtime", fail)
    assert runtime_check.main(["--runtime"]) == 1
    assert capsys.readouterr().out == "ERROR face validation failed\n"
    assert runtime_check.main(["--unexpected"]) == 2
