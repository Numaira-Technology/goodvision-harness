#!/usr/bin/env python3
"""Finite GPU experiment: arrived frames with and without bounded history.

This is an isolated, forward-only adapter for the pinned SAM source. It does not
change the serving worker. Exact equivalence on a clip does not measure accuracy
or prove unattended production reliability.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam31_stream_probe as probe

MAX_FRAMES = 256
INPUT_SLOTS = 16


@dataclass
class RingFrames:
    """Image IDs use physical slots; native tracking uses absolute indices.

    The pinned detector selects FindStage with absolute_index % len(find_inputs).
    Its FindStage.img_ids addresses a physical image slot (0..15). The native
    feature cache independently requests tensors[absolute_index]. Both access
    paths are checked against the currently installed slot content.
    """

    frames: list
    absolute_indices: list[int]

    @classmethod
    def from_tensor(cls, tensor):
        if len(tensor) != INPUT_SLOTS:
            raise ValueError("pruning probe requires 16 arrived startup frames")
        return cls([frame.clone() for frame in tensor], list(range(INPUT_SLOTS)))

    def __len__(self):
        return INPUT_SLOTS

    @property
    def shape(self):
        return (INPUT_SLOTS, *self.frames[0].shape)

    @property
    def device(self):
        return self.frames[0].device

    def __getitem__(self, index):
        if type(index) is not int or index < 0:
            raise IndexError("ring supports nonnegative scalar frame indices only")
        slot = index % INPUT_SLOTS
        if index >= INPUT_SLOTS and self.absolute_indices[slot] != index:
            raise IndexError("native code requested an expired or future source frame")
        return self.frames[slot]

    def put(self, absolute_index, tensor):
        slot = absolute_index % INPUT_SLOTS
        self.frames[slot] = tensor.clone()
        self.absolute_indices[slot] = absolute_index


def _prune_mapping(mapping, keep):
    for key in tuple(mapping):
        if key not in keep:
            del mapping[key]


class BoundedRetention:
    def __init__(self, model, history_frames=32):
        tracker = model.tracker.model
        if (
            tracker.use_memory_selection
            or tracker.keep_first_cond_frame
            or tracker.max_cond_frames_in_attn != 4
            or tracker.memory_temporal_stride_for_eval != 1
            or tracker.num_maskmem != 7
            or tracker.max_obj_ptrs_in_encoder != 16
            or model.use_batched_grounding
            or model.world_size != 1
        ):
            raise ValueError("pruning requires the explicitly audited memory settings")
        if not 32 <= history_frames <= 64:
            raise ValueError("history_frames must be within 32..64")
        if model.hotstart_delay > history_frames:
            raise ValueError("history must include the complete hotstart horizon")
        self.history_frames = history_frames
        self.cond_limit = tracker.max_cond_frames_in_attn

    def install(self, state):
        batch = state["input_batch"]
        batch.img_batch.tensors = RingFrames.from_tensor(batch.img_batch.tensors)
        if len(batch.find_inputs) != INPUT_SLOTS:
            raise ValueError("expected fixed 16 FindStage slots")
        for slot, stage in enumerate(batch.find_inputs):
            if (
                int(stage.img_ids.item()) != slot
                or int(stage.img_ids_np.item()) != slot
            ):
                raise ValueError("unexpected image IDs in initial source stages")
        for name in probe.FRAME_FIELDS:
            state[name] = dict(enumerate(state[name]))

    def append(self, model, state, images, *, frame_limit):
        previous = state["num_frames"]
        if (
            not images
            or len(images) > INPUT_SLOTS
            or previous + len(images) > frame_limit
        ):
            raise ValueError("append exceeds finite probe frame limit")
        delta = model.init_state(
            resource_path=list(images),
            offload_video_to_cpu=True,
            async_loading_frames=False,
        )
        try:
            if (state["orig_height"], state["orig_width"]) != (
                delta["orig_height"],
                delta["orig_width"],
            ):
                raise ValueError("source resolution changed")
            store = state["input_batch"].img_batch.tensors
            for offset, frame in enumerate(delta["input_batch"].img_batch.tensors):
                store.put(previous + offset, frame)
            for name in probe.FRAME_FIELDS:
                for offset, value in enumerate(delta[name]):
                    state[name][previous + offset] = value
            state["num_frames"] = previous + len(images)
            for tracker_state in state["sam2_inference_states"]:
                tracker_state["num_frames"] = state["num_frames"]
        finally:
            delta.clear()

    def prune(self, state, *, emitted_through):
        # Only called after normal generator exhaustion: Prod has persisted all
        # pending output and confirmation state. Failures destroy the whole state.
        end = state["num_frames"]
        recent = set(range(max(0, end - self.history_frames), end))
        generator = state["generator_state"]
        pending = {frame for frame, _ in generator["hotstart_buffer"]} | {
            item[0] for item in generator["postprocess_yield_list"]
        }
        if any(frame <= emitted_through for frame in pending):
            raise RuntimeError("pending native outputs precede emitted watermark")
        keep_frames = recent | pending
        for name in probe.FRAME_FIELDS:
            _prune_mapping(state[name], keep_frames)
        # Frame outputs already delivered have no role in forward propagation.
        _prune_mapping(state["cached_frame_outputs"], pending)
        _prune_mapping(generator["unconfirmed_obj_ids_per_frame"], keep_frames)

        metadata = state["tracker_metadata"]
        _prune_mapping(metadata["obj_id_to_sam2_score_frame_wise"], keep_frames)
        _prune_mapping(metadata["rank0_metadata"]["suppressed_obj_ids"], keep_frames)

        live_or_pending_ids = {
            int(obj_id)
            for item in state["sam2_inference_states"]
            for obj_id in item["obj_ids"]
        }
        for _, output in generator["hotstart_buffer"]:
            live_or_pending_ids.update(output["obj_id_to_mask"])
        for _, output, _ in generator["postprocess_yield_list"]:
            live_or_pending_ids.update(output["obj_id_to_mask"])
        # IDs are never recycled; max_obj_id is deliberately untouched.
        _prune_mapping(metadata["obj_id_to_score"], live_or_pending_ids)
        generator["hotstart_removed_obj_ids"].intersection_update(live_or_pending_ids)
        _prune_mapping(metadata.get("obj_id_to_last_occluded", {}), live_or_pending_ids)
        rank0 = metadata["rank0_metadata"]
        for name in ("obj_first_frame_idx", "unmatched_frame_inds", "trk_keep_alive"):
            _prune_mapping(rank0.get(name, {}), live_or_pending_ids)
        rank0.get("removed_obj_ids", set()).intersection_update(live_or_pending_ids)
        for values in rank0.get("unmatched_frame_inds", {}).values():
            values[:] = [frame for frame in values if frame in keep_frames]
        pairs = rank0.get("overlap_pair_to_frame_inds", {})
        for pair in tuple(pairs):
            if not all(obj_id in live_or_pending_ids for obj_id in pair):
                del pairs[pair]
            else:
                pairs[pair][:] = [
                    frame for frame in pairs[pair] if frame in keep_frames
                ]

        for tracker in state["sam2_inference_states"]:
            output_dict = tracker["output_dict"]
            # Forward attention chooses latest four past anchors. Recent anchors
            # must also remain available as non-conditioning pointer fallbacks.
            anchors = set(sorted(output_dict["cond_frame_outputs"])[-self.cond_limit :])
            keep = keep_frames | anchors
            for outputs in [output_dict, *tracker["output_dict_per_obj"].values()]:
                for kind in ("cond_frame_outputs", "non_cond_frame_outputs"):
                    _prune_mapping(outputs[kind], keep)
            for outputs in tracker["temp_output_dict_per_obj"].values():
                if any(outputs.values()):
                    raise RuntimeError(
                        "cannot prune unconsolidated native prompt outputs"
                    )
            for kind, indices in tracker["consolidated_frame_inds"].items():
                indices.intersection_update(output_dict[kind])
            for name in ("point_inputs_per_obj", "mask_inputs_per_obj"):
                for inputs in tracker[name].values():
                    _prune_mapping(inputs, keep)
            for indices in tracker.get("user_refined_frames_per_obj", {}).values():
                indices.intersection_update(keep)
            _prune_mapping(tracker["frames_already_tracked"], keep)
            if tracker["first_ann_frame_idx"] not in output_dict["cond_frame_outputs"]:
                tracker["first_ann_frame_idx"] = min(
                    output_dict["cond_frame_outputs"], default=None
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history-frames", type=int, default=32)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--budget-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--empty-final", action="store_true")
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if not 64 < len(args.images) <= MAX_FRAMES:
        parser.error("supply 65..256 chronological arrived JPEG frames")
    if not 32 <= args.history_frames <= 64 or not 1 <= args.max_side <= 640:
        parser.error("history must be 32..64, max side 1..640")
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 600:
        parser.error("cooperative budget must be within 0..600 seconds")
    import numpy as np
    import torch

    from workers.sam_scene.sam31 import SAM3_COMMIT, NativeSam31, VideoSettings
    from workers.sam_scene.streaming_state import finalize_pending

    native = NativeSam31(
        replace(VideoSettings(), enabled=True, checkpoint_path=args.checkpoint)
    )
    report = {
        "sam3_commit": SAM3_COMMIT,
        "gpu_ready": False,
        "accuracy_evaluated": False,
        "continuous_session_integrated": False,
        "retention_policy": "16 input slots; recent history plus conditioning anchors",
        "history_frames": args.history_frames,
        "hard_frame_limit": MAX_FRAMES,
        "jpeg_bytes_limit": probe.MAX_TOTAL_JPEG_BYTES,
        "arrival_simulation": "decode each 16-frame chunk only after its predecessor",
        "boundary_adapter": "exclusive detector count; inclusive processing offset",
        "empty_final_comparison": args.empty_final,
    }
    try:
        native.warmup()
        report["gpu_ready"] = native.gpu_ready
        model = probe.make_prod_model(native)
        retention = BoundedRetention(model, args.history_frames)
        rng_states = [
            (module, name, deepcopy(value.bit_generator.state))
            for module in model.modules()
            for name in ("rng", "rng2")
            if isinstance((value := getattr(module, name, None)), np.random.Generator)
        ]
        report["cases"] = []
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            for adapter in (None, retention):
                torch.manual_seed(42)
                for module, name, state in rng_states:
                    getattr(module, name).bit_generator.state = deepcopy(state)
                result = probe.run_case(
                    model,
                    args.images,
                    mode="arrived_append",
                    chunk=16,
                    side=args.max_side,
                    budget_seconds=args.budget_seconds,
                    frame_limit=MAX_FRAMES,
                    retention=adapter,
                    finalizer=finalize_pending
                    if adapter and args.empty_final
                    else None,
                )
                result["mode"] = "arrived_pruned" if adapter else "arrived_unpruned"
                report["cases"].append(result)
        report["unpruned_vs_pruned"] = probe.compare(*report["cases"])
        report["informative_person_evidence"] = (
            report["cases"][0]["visible_observations"] > 0
        )
        passed = (
            report["unpruned_vs_pruned"]["exact_ids_and_masks"]
            and report["informative_person_evidence"]
        )
        return_code = 0 if passed else 2
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc(limit=8)
        return_code = 1
    finally:
        native.close()
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
