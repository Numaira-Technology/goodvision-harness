"""Finite probe mechanics only; native model/GPU equivalence remains unmeasured."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[1] / "tools" / "sam31_stream_probe.py"
    spec = importlib.util.spec_from_file_location("sam31_stream_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state(probe, count):
    return {
        "num_frames": count,
        "orig_height": 2,
        "orig_width": 3,
        "input_batch": SimpleNamespace(
            img_batch=SimpleNamespace(tensors=np.arange(count), mask=None),
            find_inputs=[
                SimpleNamespace(img_ids=np.array([i]), img_ids_np=np.array([i]))
                for i in range(count)
            ],
            find_targets=[None] * count,
            find_metadatas=[None] * count,
        ),
        "sam2_inference_states": [{"num_frames": count, "preserved_memory": "native"}],
        "generator_state": {"hotstart_buffer": ["pending"]},
        "tracker_metadata": {"max_obj_id": 7},
        **{name: [None] * count for name in probe.FRAME_FIELDS},
    }


def test_append_changes_only_arrived_input_indices_and_preserves_native_memory(
    probe, monkeypatch
):
    torch = ModuleType("torch")
    torch.cat = lambda arrays, dim: np.concatenate(arrays, axis=dim)
    monkeypatch.setitem(sys.modules, "torch", torch)
    active = state(probe, 2)
    memory = active["sam2_inference_states"][0]
    pending = active["generator_state"]
    metadata = active["tracker_metadata"]
    supplied = []

    def initialize(*, resource_path, **_kwargs):
        supplied.append(resource_path)
        return state(probe, len(resource_path))

    model = SimpleNamespace(init_state=initialize)
    probe.append_arrived_frames(model, active, ["arrived2", "arrived3"], frame_limit=4)
    assert supplied == [["arrived2", "arrived3"]]
    assert active["num_frames"] == memory["num_frames"] == 4
    assert [int(item.img_ids[0]) for item in active["input_batch"].find_inputs] == [
        0,
        1,
        2,
        3,
    ]
    assert [int(item.img_ids_np[0]) for item in active["input_batch"].find_inputs] == [
        0,
        1,
        2,
        3,
    ]
    assert active["sam2_inference_states"][0] is memory
    assert memory["preserved_memory"] == "native"
    assert active["generator_state"] is pending
    assert active["tracker_metadata"] is metadata
    with pytest.raises(ValueError, match="frame limit"):
        probe.append_arrived_frames(model, active, ["future4"], frame_limit=4)
    assert len(supplied) == 1


def test_comparison_requires_exact_native_ids_and_masks(probe):
    reference = {"frames": {0: {"ids": [4], "mask_sha256": ["mask"]}}}
    assert probe.compare(reference, reference)["exact_ids_and_masks"]
    for value in (
        {"ids": [5], "mask_sha256": ["mask"]},
        {"ids": [4], "mask_sha256": ["different"]},
    ):
        assert probe.compare(reference, {"frames": {0: value}}) == {
            "exact_ids_and_masks": False,
            "mismatched_frame_indices": [0],
        }


def test_signature_rejects_invalid_masks_and_duplicate_ids(probe):
    output = {
        "out_obj_ids": np.array([1, 1]),
        "out_probs": np.array([0.9, 0.9]),
        "out_binary_masks": np.ones((2, 2, 3), dtype=bool),
    }
    with pytest.raises(RuntimeError, match="native output"):
        probe.signature(output, width=3, height=2, max_objects=16)


@pytest.mark.parametrize(
    "start,count,total",
    [(0, 16, 48), (16, 16, 48), (32, 16, 48), (0, 1, 1), (47, 1, 48)],
)
def test_pinned_processing_offset_and_detector_count_agree(probe, start, count, total):
    class OfficialProcessingOrder:
        # Exact forward index formula from pinned Tracking lines 295-297.
        def _get_processing_order(self, state, start, offset, reverse):
            assert reverse is False
            end = min(start + offset, state["num_frames"] - 1)
            return range(start, end + 1), end

    class Adapter(probe.CountBoundedProcessingOrder, OfficialProcessingOrder):
        pass

    frames, end = Adapter()._get_processing_order(
        {"num_frames": total}, start, count, False
    )
    # Pinned detector lines 399-402 independently uses this exclusive end.
    detector_end = min(total, start + count)
    assert list(frames) == list(range(start, detector_end))
    assert end == detector_end - 1


def test_boundary_adapter_rejects_future_frames_and_reverse(probe):
    adapter = probe.CountBoundedProcessingOrder()
    for start, count, reverse in [(16, 1, False), (0, 0, False), (0, 1, True)]:
        with pytest.raises(ValueError, match="arrived forward"):
            adapter._get_processing_order({"num_frames": 16}, start, count, reverse)
