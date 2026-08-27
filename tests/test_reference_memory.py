from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from goodvision_harness.config import load_config
from goodvision_harness.reference_memory import (
    FACE_RECOGNITION_PURPOSE,
    MemoryReference,
    build_face_memory_context,
    load_memory_context,
    load_memory_reference,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, *, width: int = 800, height: int = 400) -> None:
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def test_load_memory_reference_resizes_and_prepares_jpeg(tmp_path: Path) -> None:
    image_path = tmp_path / "wide.png"
    _write_image(image_path)

    reference = load_memory_reference(
        "person_01",
        image_path,
        max_image_side=200,
        jpeg_quality=80,
    )

    assert reference.memory_id == "person_01"
    assert (reference.width, reference.height) == (200, 100)
    prefix, encoded = reference.image_data_url.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    assert decoded.shape[:2] == (100, 200)


def test_load_config_resolves_enabled_memory_relative_to_config(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "memory" / "alice.png"
    _write_image(image_path, width=300, height=150)
    raw: dict[str, Any] = yaml.safe_load(
        (ROOT / "config.yaml").read_text(encoding="utf-8")
    )
    raw["memory"] = {
        "enabled": True,
        "purpose": "face_recognition",
        "newest_frame_only": True,
        "max_references": 2,
        "max_image_side": 96,
        "jpeg_quality": 75,
        "references": [{"id": "alice", "image": "memory/alice.png"}],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.memory is not None
    assert config.memory.purpose == FACE_RECOGNITION_PURPOSE
    assert config.memory.memory_ids == ("alice",)
    assert config.memory.references[0].width == 96
    assert config.memory.references[0].height == 48
    assert config.memory.prompt_prefix[1]["text"] == "MEMORY_ID: alice"
    assert "same stranger-detection semantics as InsightFace" in (
        config.memory.result_contract
    )
    assert "independently reports all known matches" in config.memory.result_contract


@pytest.mark.parametrize(
    "config_path",
    [ROOT / "config.yaml", ROOT / "src" / "goodvision_harness" / "config.yaml"],
)
def test_checked_in_config_disables_reference_memory(config_path: Path) -> None:
    assert load_config(config_path).memory is None


def test_disabled_memory_skips_reference_validation(tmp_path: Path) -> None:
    assert (
        load_memory_context(
            {
                "enabled": False,
                "references": "not-a-list",
                "max_image_side": 0,
            },
            config_directory=tmp_path,
        )
        is None
    )


def test_memory_config_validation_is_strict(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="memory.enabled"):
        load_memory_context({"enabled": "yes"}, config_directory=tmp_path)
    with pytest.raises(ValueError, match="must not be empty"):
        load_memory_context({"enabled": True}, config_directory=tmp_path)
    with pytest.raises(ValueError, match="instruction is required"):
        load_memory_context(
            {"enabled": True, "purpose": "custom", "references": [{}]},
            config_directory=tmp_path,
        )
    with pytest.raises(ValueError, match="1-64 ASCII"):
        load_memory_context(
            {
                "enabled": True,
                "references": [{"id": "not allowed", "image": "face.jpg"}],
            },
            config_directory=tmp_path,
        )
    with pytest.raises(ValueError, match="exceeds memory.max_references"):
        load_memory_context(
            {
                "enabled": True,
                "max_references": 1,
                "references": [
                    {"id": "one", "image": "one.jpg"},
                    {"id": "two", "image": "two.jpg"},
                ],
            },
            config_directory=tmp_path,
        )


def test_face_context_rejects_empty_and_duplicate_ids() -> None:
    reference = MemoryReference(
        memory_id="alice",
        image_data_url="data:image/jpeg;base64,AA==",
        width=1,
        height=1,
    )
    with pytest.raises(ValueError, match="at least one reference"):
        build_face_memory_context([])
    with pytest.raises(ValueError, match="IDs must be unique"):
        build_face_memory_context([reference, reference])


def test_memory_reference_rejects_missing_and_unreadable_images(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_memory_reference("missing", tmp_path / "missing.jpg")
    unreadable = tmp_path / "unreadable.jpg"
    unreadable.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        load_memory_reference("broken", unreadable)
