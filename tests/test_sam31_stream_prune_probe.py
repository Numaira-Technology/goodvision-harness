"""Pruning mechanics; numerical native equivalence requires the GPU probe."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tools" / "sam31_stream_prune_probe.py"
    spec = importlib.util.spec_from_file_location("sam31_pruning_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class Frame:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return Frame(self.value)


def model():
    return SimpleNamespace(
        tracker=SimpleNamespace(
            model=SimpleNamespace(
                use_memory_selection=False,
                keep_first_cond_frame=False,
                max_cond_frames_in_attn=4,
                memory_temporal_stride_for_eval=1,
                num_maskmem=7,
                max_obj_ptrs_in_encoder=16,
            )
        ),
        use_batched_grounding=False,
        world_size=1,
        hotstart_delay=15,
    )


def test_ring_retains_only_arrived_slots_and_rejects_expired_absolute_index(module):
    source = [Frame(i) for i in range(16)]
    ring = module.RingFrames.from_tensor(source)
    assert ring[0] is not source[0]
    for index in range(16, 80):
        ring.put(index, Frame(index))
        assert ring[index].value == ring[index % 16].value == index
        assert len(ring.frames) == len(ring) == 16
    assert sorted(frame.value for frame in ring.frames) == list(range(64, 80))
    for invalid in (16, 80, -1, slice(0, 16)):
        with pytest.raises(IndexError):
            ring[invalid]


def state(module):
    cond = {i: {"anchor": i} for i in (0, 4, 8, 12, 16, 20)}
    outputs = {
        "cond_frame_outputs": cond,
        "non_cond_frame_outputs": {
            i: {"memory": i} for i in range(64) if i not in cond
        },
    }
    tracker = {
        "obj_ids": [7],
        "output_dict": outputs,
        "output_dict_per_obj": {0: deepcopy(outputs)},
        "temp_output_dict_per_obj": {
            0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        },
        "consolidated_frame_inds": {
            "cond_frame_outputs": set(cond),
            "non_cond_frame_outputs": set(),
        },
        "point_inputs_per_obj": {0: {}},
        "mask_inputs_per_obj": {0: {i: "prompt" for i in cond}},
        "frames_already_tracked": {i: {"reverse": False} for i in range(64)},
        "first_ann_frame_idx": 0,
    }
    return {
        "num_frames": 64,
        **{name: dict.fromkeys(range(64)) for name in module.probe.FRAME_FIELDS},
        "generator_state": {
            "hotstart_buffer": [
                [i, {"obj_id_to_mask": {7: "mask", 8: "mask"}}] for i in range(50, 64)
            ],
            "postprocess_yield_list": [],
            "unconfirmed_obj_ids_per_frame": {i: [] for i in range(64)},
            "hotstart_removed_obj_ids": {8, 999},
        },
        "cached_frame_outputs": dict.fromkeys(range(64), "mask"),
        "tracker_metadata": {
            "max_obj_id": 1000,
            "obj_id_to_score": {7: 0.9, 8: 0.8, 999: 0.1},
            "obj_id_to_sam2_score_frame_wise": {i: {} for i in range(64)},
            "rank0_metadata": {
                "suppressed_obj_ids": {i: set() for i in range(64)},
                "obj_first_frame_idx": {7: 0, 8: 50, 999: 1},
                "unmatched_frame_inds": {7: list(range(64)), 999: [1]},
                "trk_keep_alive": {7: 1, 8: 2, 999: 0},
                "removed_obj_ids": {8, 999},
                "overlap_pair_to_frame_inds": {(7, 8): [51, 52], (7, 999): [1]},
            },
        },
        "sam2_inference_states": [tracker],
    }


def test_pruning_retains_native_attention_anchors_pending_masks_and_id_counter(module):
    active = state(module)
    tracker = active["sam2_inference_states"][0]
    module.BoundedRetention(model()).prune(active, emitted_through=49)
    assert set(tracker["output_dict"]["cond_frame_outputs"]) == {8, 12, 16, 20}
    assert set(tracker["output_dict"]["non_cond_frame_outputs"]) == set(range(32, 64))
    assert tracker["output_dict_per_obj"][0] == tracker["output_dict"]
    assert set(tracker["mask_inputs_per_obj"][0]) == {8, 12, 16, 20}
    assert tracker["consolidated_frame_inds"]["cond_frame_outputs"] == {8, 12, 16, 20}
    assert tracker["first_ann_frame_idx"] == 8
    assert len(active["generator_state"]["hotstart_buffer"]) == 14
    assert set(active["cached_frame_outputs"]) == set(range(50, 64))
    assert set(active["previous_stages_out"]) == set(range(32, 64))
    assert active["tracker_metadata"]["max_obj_id"] == 1000
    assert active["tracker_metadata"]["obj_id_to_score"] == {7: 0.9, 8: 0.8}
    assert active["generator_state"]["hotstart_removed_obj_ids"] == {8}
    rank0 = active["tracker_metadata"]["rank0_metadata"]
    assert rank0["obj_first_frame_idx"] == {7: 0, 8: 50}
    assert rank0["unmatched_frame_inds"] == {7: list(range(32, 64))}
    assert rank0["overlap_pair_to_frame_inds"] == {(7, 8): [51, 52]}


def test_pruning_rejects_unverified_attention_settings_and_incomplete_handoff(module):
    selected = model()
    selected.tracker.model.use_memory_selection = True
    with pytest.raises(ValueError, match="audited memory"):
        module.BoundedRetention(selected)
    active = state(module)
    with pytest.raises(RuntimeError, match="pending native"):
        module.BoundedRetention(model()).prune(active, emitted_through=50)
    active = state(module)
    active["sam2_inference_states"][0]["temp_output_dict_per_obj"][0][
        "cond_frame_outputs"
    ][63] = "pending"
    with pytest.raises(RuntimeError, match="unconsolidated"):
        module.BoundedRetention(model()).prune(active, emitted_through=49)
