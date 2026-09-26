"""Offline accounting/regression checks; no model weights, GPU, or network required."""

import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
cli = importlib.import_module("tools.seg_benchmark.__main__")
metrics = importlib.import_module("tools.seg_benchmark.metrics")


def test_missing_and_duplicate_predictions_are_not_successes():
    mask = np.zeros((30, 30), bool)
    mask[4:20, 4:15] = True
    missed = metrics.score_frame([mask], [], 0.5)
    assert missed["matched"] == 0
    duplicate = metrics.score_frame([mask], [mask, mask], 0.5)
    assert duplicate["matched"] == 1
    assert duplicate["false_positives"] == 1
    assert duplicate["boundary_f1_sum"] == pytest.approx(1)


def test_empty_frames_and_incomplete_annotations_do_not_pass():
    row = {
        "pipeline_ms": 10,
        "simulated_ready_ms": 10,
        "predicted_people": 0,
        "quality": None,
    }
    result = metrics.summarize([row], 500, 0.7, 1)
    assert result["passes_target"] is None
    assert result["effective_coverage"] is None
    row["quality"] = metrics.score_frame([], [], 0.5)
    assert metrics.summarize([row], 500, 0.7, 1)["passes_target"] is None
    row["quality"] = metrics.score_frame(
        [np.ones((5, 5), bool)], [np.ones((5, 5), bool)], 0.5
    )
    assert metrics.summarize([row], 500, 0.7, 2)["passes_target"] is None


def test_queue_overload_and_deadline_affect_coverage():
    finish, delay = metrics.replay_delay(0, 800)
    assert delay == pytest.approx(800)
    finish, delay = metrics.replay_delay(0.5, 800, finish)
    assert delay == pytest.approx(1100)
    quality = metrics.score_frame([np.ones((5, 5), bool)], [np.ones((5, 5), bool)], 0.5)
    row = {
        "pipeline_ms": 800,
        "simulated_ready_ms": delay,
        "predicted_people": 1,
        "quality": quality,
    }
    result = metrics.summarize([row], 1000, 0.7, 1)
    assert result["mask_recall"] == 1
    assert result["effective_coverage"] == 0
    assert result["passes_target"] is False


def test_multipolygon_person_is_one_instance_and_bad_coordinates_rejected():
    a = [[1, 1], [5, 1], [5, 5], [1, 5]]
    b = [[10, 10], [15, 10], [15, 15], [10, 15]]
    masks = metrics.polygons_to_masks([[a, b]], 20, 20)
    assert len(masks) == 1 and masks[0][2, 2] and masks[0][12, 12]
    with pytest.raises(ValueError):
        metrics.polygons_to_masks([[[[0, 0], [99, 0], [2, 2]]]], 20, 20)


@pytest.fixture
def dataset(tmp_path):
    video = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4, (64, 48))
    assert writer.isOpened()
    for index in range(8):
        image = np.zeros((48, 64, 3), np.uint8)
        image[5:40, index + 4 : index + 20] = 220
        writer.write(image)
    writer.release()
    output = tmp_path / "dataset"
    cli.prepare(
        SimpleNamespace(
            output=output,
            video=[video],
            video_dir=None,
            max_videos=1,
            seconds=2,
            sample_fps=2,
            max_side=64,
        )
    )
    return output


def test_prepare_freezes_source_and_requires_correct_annotation_identity(dataset):
    manifest, _ = cli.load_dataset(dataset)
    assert [f["source_frame_index"] for f in manifest["frames"]] == [0, 2, 4, 6]
    assert "__DATASET_JSON__" not in (dataset / "annotate.html").read_text()
    annotations = json.loads((dataset / "annotations.json").read_text())
    annotations["dataset_sha256"] = "wrong"
    cli.write_json(dataset / "annotations.json", annotations)
    with pytest.raises(ValueError, match="different dataset"):
        cli.load_dataset(dataset)


def test_corrupted_frames_are_rejected(dataset):
    manifest, _ = cli.load_dataset(dataset)
    (dataset / manifest["frames"][0]["image"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="changed"):
        cli.load_dataset(dataset)


def test_worker_with_fake_model_and_report(dataset, tmp_path, monkeypatch):
    manifest, labels = cli.load_dataset(dataset)
    for frame in manifest["frames"]:
        labels["frames"][frame["id"]] = {
            "reviewed": True,
            "people": [[[[4, 5], [19, 5], [19, 39], [4, 39]]]],
        }
    cli.write_json(dataset / "annotations.json", labels)
    weight = tmp_path / "fake-weight"
    weight.write_bytes(b"synthetic fixture, not a model")

    class FakeAdapter:
        size = 64

        def __init__(self, *_args):
            self.weight = weight

        def predict(self, _image):
            return np.asarray(
                metrics.polygons_to_masks(
                    [[[[4, 5], [19, 5], [19, 39], [4, 39]]]], 64, 48
                )
            )

    monkeypatch.setattr(cli, "Adapter", FakeAdapter)
    monkeypatch.setattr(cli, "setup_cache", lambda: None)
    monkeypatch.setattr(cli, "environment", lambda: {"synthetic": True})
    monkeypatch.setattr(cli, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(cli, "synchronize", lambda _device: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(manual_seed=lambda _seed: None, inference_mode=nullcontext),
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))
        ),
    )
    output = tmp_path / "run" / "yolov8-n"
    job = {
        "model": "yolov8-n",
        "dataset": str(dataset),
        "output": str(output),
        "annotations": str(dataset / "annotations.json"),
        "device": "cpu",
        "seed": 42,
        "confidence": 0.25,
        "size": None,
        "warmup": 1,
        "repeats": 2,
        "iou": 0.5,
        "target": 0.7,
        "budget_ms": 1000,
    }
    cli.write_json(tmp_path / "job.json", job)
    assert cli.worker(SimpleNamespace(job=tmp_path / "job.json")) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["summary"]["frames"] == 8
    assert result["summary"]["effective_coverage"] == 1
    cli.build_report(output.parent, [result], manifest)
    report = json.loads((output.parent / "summary.json").read_text())
    assert report["fastest_qualified_in_this_run"] == "yolov8-n"
    assert len(list(output.glob("*.jpg"))) == 4
