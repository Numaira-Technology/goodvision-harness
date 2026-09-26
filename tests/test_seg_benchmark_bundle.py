"""Portable sample identity and explicit source selection, without network/GPU."""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
bundle = importlib.import_module("tools.export_seg_benchmark")


def test_portable_dataset_rebinds_labels_and_preserves_media(tmp_path):
    source, stage = tmp_path / "dataset", tmp_path / "stage"
    (source / "frames").mkdir(parents=True)
    (source / "frames/a.jpg").write_bytes(b"fixed image")
    video = tmp_path / "original.mp4"
    video.write_bytes(b"fixed video")
    manifest = {
        "frames": [
            {
                "id": "a",
                "image": "frames/a.jpg",
                "sha256": bundle.sha256(source / "frames/a.jpg"),
            }
        ],
        "sources": [
            {"id": "clip-01", "path": str(video), "sha256": bundle.sha256(video)}
        ],
    }
    bundle.write_json(source / "manifest.json", manifest)
    labels = {
        "dataset_sha256": bundle.sha256(source / "manifest.json"),
        "frames": {"a": {"reviewed": True, "people": []}},
    }
    bundle.write_json(source / "annotations.json", labels)
    template = stage / "tools/seg_benchmark/annotate.html"
    template.parent.mkdir(parents=True)
    template.write_text("<script>const data=__DATASET_JSON__;</script>")
    bundle.stage_dataset(source, stage)
    target = stage / "outputs/seg-benchmark/dataset"
    rewritten = json.loads((target / "manifest.json").read_text())
    rebound = json.loads((target / "annotations.json").read_text())
    assert rebound["frames"] == labels["frames"]
    assert rebound["dataset_sha256"] == bundle.sha256(target / "manifest.json")
    assert str(tmp_path) not in (target / "manifest.json").read_text()
    assert (
        target / rewritten["sources"][0]["path"]
    ).resolve().read_bytes() == b"fixed video"
    assert (target / "frames/a.jpg").read_bytes() == b"fixed image"


def test_external_or_symbolic_source_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    private = tmp_path / "private"
    private.write_text("do not package")
    (root / "link.py").symlink_to(private)
    with pytest.raises(ValueError):
        bundle.checked_file(root / "link.py", root)
    with pytest.raises(ValueError):
        bundle.checked_file(private, root)


def test_source_selection_excludes_local_state_and_automatic_rental_controller():
    paths = [str(p.relative_to(bundle.ROOT)) for p in bundle.source_files(bundle.ROOT)]
    assert "tools/seg_benchmark/models.py" in paths
    assert "workers/sam_scene/sam_live.py" in paths
    assert "tools/start_sam.py" not in paths
    assert not any(
        p.startswith((".env", ".models", ".venv", ".goodvision", "outputs"))
        for p in paths
    )
    defaults = bundle.worker_defaults(bundle.ROOT)
    assert defaults["SAM31_LIVE_ENABLED"] == "true"
    assert "SAM31_CHECKPOINT_PATH" not in defaults
    assert all(k.startswith("SAM") for k in defaults)
