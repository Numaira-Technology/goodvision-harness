"""Local source/measurement tests; fake CUDA methods never run inference."""

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from workers.sam_scene.continuous import Session

spec = importlib.util.spec_from_file_location(
    "postprocess_benchmark",
    Path(__file__).resolve().parents[1] / "tools/sam31_postprocess_benchmark.py",
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_source_load_is_exact_sixteen_bounded_files_and_preserves_hashes(tmp_path):
    for index in range(17):
        with Image.new("RGB", (320, 160), (index, 40, 90)) as image:
            image.save(tmp_path / f"frame{index:03}.jpg")
    images, sources = benchmark.load_images(tmp_path, start_index=1, max_side=128)
    try:
        assert len(images) == len(sources) == 16
        assert sources[0]["file"] == "frame001.jpg"
        assert sources[-1]["file"] == "frame016.jpg"
        assert all(image.size == (128, 64) for image in images)
        assert all(row["source_dimensions"] == (320, 160) for row in sources)
        assert all(len(row["sha256"]) == 64 for row in sources)
    finally:
        for image in images:
            image.close()


def test_short_directory_and_nonjpeg_contents_rejected(tmp_path):
    with pytest.raises(ValueError, match="sixteen"):
        benchmark.load_images(tmp_path)
    for index in range(16):
        with Image.new("RGB", (16, 16)) as image:
            image.save(tmp_path / f"frame{index:03}.jpg", format="PNG")
    with pytest.raises(ValueError, match="JPEG"):
        benchmark.load_images(tmp_path)


def native_output(index):
    masks = np.zeros((1, 6, 8), dtype=bool)
    masks[0, 1:4, index % 4 : index % 4 + 2] = True
    return {
        "out_obj_ids": np.array([2]),
        "out_probs": np.array([0.8]),
        "out_binary_masks": masks,
    }


def test_comparison_checks_ids_and_mask_hash_not_just_frame_count():
    first = {"frames": [benchmark.frame_signature(0, native_output(0), (8, 6))]}
    second = {"frames": [benchmark.frame_signature(0, native_output(1), (8, 6))]}
    assert benchmark.compare(first, first)["equal_ids_and_masks"]
    assert benchmark.compare(first, second) == {
        "equal_ids_and_masks": False,
        "different_frames": [0],
    }


def test_stages_cover_append_pruning_empty_eof_and_cleanup_before_hash(monkeypatch):
    calls = []
    cuda = SimpleNamespace(
        synchronize=lambda: calls.append("sync"),
        reset_peak_memory_stats=lambda: None,
        memory_allocated=lambda: 10,
        max_memory_allocated=lambda: 20,
    )

    class Backend:
        video = SimpleNamespace(
            backend=SimpleNamespace(_torch=SimpleNamespace(cuda=cuda))
        )

        def run(self, session, images, *, final):
            calls.append("eof" if final else "append")
            session.native_state = {}
            assert len(images) == (0 if final else 16)
            yield from (
                (index, native_output(index)) for index in range(12, 16) if final
            )
            if not final:
                yield from ((index, native_output(index)) for index in range(12))

        def prune(self, session):
            assert session.total_frames == 16 and session.emitted_index == 11
            calls.append("prune")

        def discard(self, session):
            calls.append("discard")
            session.native_state = None

    original = benchmark.frame_signature

    def signature(*args):
        assert "discard" in calls
        calls.append("hash")
        return original(*args)

    monkeypatch.setattr(benchmark, "frame_signature", signature)
    result = benchmark.measure_session(
        Backend(),
        Session,
        [SimpleNamespace(size=(8, 6)) for _ in range(16)],
        deadline=time.monotonic() + 5,
    )
    assert result["emitted_before_eof"] == 12
    assert result["emitted_at_empty_eof"] == 4
    assert [row["frame_sequence"] for row in result["frames"]] == list(range(16))
    assert set(result["stages"]) == {
        "append_16_seconds",
        "retention_seconds",
        "empty_eof_seconds",
        "release_seconds",
    }
    assert result["cuda_peak_allocated"] == 20
    assert calls.index("hash") > calls.index("discard")


def test_expired_budget_still_cleans_session():
    discarded = []
    cuda = SimpleNamespace(
        synchronize=lambda: None,
        reset_peak_memory_stats=lambda: None,
        memory_allocated=lambda: 0,
    )
    backend = SimpleNamespace(
        video=SimpleNamespace(
            backend=SimpleNamespace(_torch=SimpleNamespace(cuda=cuda))
        ),
        discard=lambda session: discarded.append(session),
    )
    with pytest.raises(TimeoutError, match="budget"):
        benchmark.measure_session(backend, Session, [], deadline=time.monotonic() - 1)
    assert len(discarded) == 1
