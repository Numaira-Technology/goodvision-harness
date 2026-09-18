"""Probe accounting and failure cleanup; these tests do not measure a GPU."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from workers.contour.benchmark_gpu import (
    load_images,
    measure_case,
    memory_stats,
    percentile,
)
from workers.sam_scene.continuous import Session


@pytest.mark.parametrize("fail", [False, True])
def test_probe_consumes_reused_masks_and_releases_lane_states(tmp_path, fail):
    path = tmp_path / "000001.jpg"
    Image.new("RGB", (16, 16)).save(path)
    released = []
    shared_mask = np.zeros((1, 16, 16), dtype=bool)
    runtime = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
    ))

    class Backend:
        video = SimpleNamespace(backend=SimpleNamespace(_torch=runtime))

        def run(self, session, images, *, final):
            if fail and session.total_frames >= 36:
                raise RuntimeError("simulated OOM")
            for offset, _image in enumerate(images):
                index = session.total_frames + offset
                shared_mask.fill(index % 2 == 0)
                yield index, {
                    "out_obj_ids": np.array([7]),
                    "out_binary_masks": shared_mask,
                    "out_probs": np.array([0.9]),
                }

        def prune(self, session):
            pass

        def discard(self, session):
            released.append(session.video_session_id)

        def memory_stats(self):
            return {"cuda_allocated_bytes": 123}

    case = {"ok": False}
    if fail:
        with pytest.raises(RuntimeError, match="OOM"):
            measure_case(Backend(), Session, [path] * 64, 2, 4, 512, case)
        assert not case["ok"]
    else:
        measure_case(Backend(), Session, [path] * 64, 2, 4, 512, case)
        assert case["ok"] and case["completed_frames"] == 128
        first = case["outputs"][0]
        assert first == case["outputs"][1]
        assert first[0]["mask_sha256"] != first[1]["mask_sha256"]
        assert first[0]["mask_sha256"] == first[2]["mask_sha256"]
        assert case["steady_saturated_native_fps"] > 0
    assert released == ["lane-0", "lane-1"]
    assert case["memory_after_state_release"] == {"cuda_allocated_bytes": 123}


def test_probe_rejects_non_jpeg_source(tmp_path):
    path = tmp_path / "wrong.jpg"
    Image.new("RGB", (16, 16)).save(path, format="PNG")
    with pytest.raises(ValueError, match="JPEG"):
        load_images([path], 512)


def test_percentiles_use_nearest_rank_and_do_not_invent_empty_samples():
    assert percentile([], 0.95) is None
    assert percentile([1, 2, 3, 4], 0.5) == 2
    assert percentile(list(range(1, 101)), 0.95) == 95


def test_baseline_allocator_probe_does_not_require_modified_worker():
    cuda = SimpleNamespace(
        mem_get_info=lambda: (100, 200), memory_allocated=lambda: 1,
        memory_reserved=lambda: 2, max_memory_allocated=lambda: 3,
        max_memory_reserved=lambda: 4,
    )
    original_native = SimpleNamespace(_torch=SimpleNamespace(cuda=cuda))
    original_live = SimpleNamespace(video=SimpleNamespace(backend=original_native))
    expected = {
        "cuda_allocated_bytes": 1, "cuda_reserved_bytes": 2,
        "cuda_peak_allocated_bytes": 3, "cuda_peak_reserved_bytes": 4,
        "cuda_device_free_bytes": 100, "cuda_device_total_bytes": 200,
    }
    assert memory_stats(original_native) == memory_stats(original_live) == expected
