"""Pinned, forward-only SAM3.1 continuation and bounded state adapter.

Derived from isolated native GPU equivalence experiments. No reverse tracking
or historical interactive edits are supported after expired frames are pruned.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass

INPUT_SLOTS = 16
FRAME_FIELDS = (
    "previous_stages_out",
    "per_frame_raw_point_input",
    "per_frame_raw_box_input",
    "per_frame_visual_prompt",
    "per_frame_geometric_prompt",
    "per_frame_cur_step",
)


def finalize_pending(model, state):
    """Flush arrived hotstart outputs without running a fabricated/repeated frame.

    This is the single-GPU, bounded postprocessing specialization of the
    pinned Prod final-batch postprocessing (tracking.py lines 1995-2059).
    The native detector/tracker memory is never advanced here.
    """
    batch_size = model.postprocess_batch_size
    if model.world_size != 1 or type(batch_size) is not int or batch_size not in {1, 4}:
        raise RuntimeError("unsupported native finalization settings")
    generator = state["generator_state"]
    if generator["postprocess_yield_list"]:
        raise RuntimeError("unflushed native postprocess batch")
    delay = model.masklet_confirmation_consecutive_det_thresh - 1
    last = state["num_frames"] - 1
    removed = set(generator["hotstart_removed_obj_ids"])
    pending = generator["hotstart_buffer"]
    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset : offset + batch_size]
        postprocess_inputs = []
        for index, output in chunk:
            unconfirmed = generator["unconfirmed_obj_ids_per_frame"].get(
                max(0, min(index + delay, last)), None
            )
            postprocess_inputs.append(
                (output, set(removed), output["suppressed_obj_ids"], unconfirmed)
            )
        if batch_size == 1:
            outputs = [model._postprocess_output(state, *postprocess_inputs[0])]
        else:
            # The pinned CPU mask buffer has capacity batch_size * max_objects.
            # Never pass all 14 hotstart frames to a four-frame buffer at once.
            outputs = model._postprocess_output_batched(
                state["orig_height"], state["orig_width"], postprocess_inputs
            )
        if len(outputs) != len(chunk):
            raise RuntimeError("native EOF postprocessing omitted source frames")
        for (index, _), output in zip(chunk, outputs, strict=True):
            yield index, output
    generator["hotstart_buffer"].clear()


class CountBoundedProcessingOrder:
    """Resolve the pinned source's conflicting meanings of the tracking limit.

    Tracking._get_processing_order uses an inclusive offset (lines 295-297),
    while MultiplexDetector uses an exclusive count (lines 399-402). Keep the
    public propagate argument a count for the detector, and translate only the
    processing-order call. This is a probe adapter, not an upstream patch.
    """

    def _get_processing_order(
        self, inference_state, start_frame_idx, max_frame_num_to_track, reverse
    ):
        if (
            reverse
            or type(start_frame_idx) is not int
            or type(max_frame_num_to_track) is not int
            or start_frame_idx < 0
            or max_frame_num_to_track < 1
            or start_frame_idx + max_frame_num_to_track > inference_state["num_frames"]
        ):
            raise ValueError("probe requires an explicit arrived forward frame count")
        return super()._get_processing_order(
            inference_state, start_frame_idx, max_frame_num_to_track - 1, False
        )


def hotstart_parameters(delay):
    """Explicit latency profiles; shorter early rejection is a quality tradeoff."""
    if type(delay) is not int or delay not in {3, 5, 15}:
        raise ValueError("hotstart_delay must be 3, 5, or 15")
    return {
        "hotstart_delay": delay,
        # The pinned constructor requires both thresholds <= hotstart_delay.
        # Keep the separate three-detection masklet confirmation unchanged.
        "hotstart_unmatch_thresh": min(8, delay),
        "hotstart_dup_thresh": min(8, delay),
    }


def make_prod_model(
    native, *, hotstart_delay=15, postprocess_batch_size=1, model_mixin=None
):
    if type(postprocess_batch_size) is not int or postprocess_batch_size not in {1, 4}:
        raise ValueError("postprocess_batch_size must be 1 or 4")
    from sam3.model.sam3_multiplex_base import Sam3MultiplexBase
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingProd

    bases = (CountBoundedProcessingOrder, Sam3MultiplexTrackingProd)
    if model_mixin is not None:
        bases = (model_mixin, *bases)
    CountBoundedProd = type("CountBoundedProd", bases, {})

    existing = native.predictor.model
    kwargs = {}
    for name, parameter in inspect.signature(
        Sam3MultiplexBase.__init__
    ).parameters.items():
        if name in {"self", "ckpt_path", "sam3_ckpt_path"}:
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        # Copy actual audited builder settings, never constructor defaults.
        kwargs[name] = getattr(existing, name)
    for name in ("image_size", "image_mean", "image_std"):
        kwargs[name] = getattr(existing, name)
    kwargs.update(
        compile_model=False,
        postprocess_batch_size=postprocess_batch_size,
        use_batched_grounding=False,
    )
    if kwargs["masklet_confirmation_consecutive_det_thresh"] != 3:
        raise RuntimeError("unsupported native masklet confirmation settings")
    kwargs.update(hotstart_parameters(hotstart_delay))
    model = CountBoundedProd(**kwargs).eval()
    expected, actual = existing.state_dict(), model.state_dict()
    if expected.keys() != actual.keys() or any(
        expected[key].data_ptr() != actual[key].data_ptr() for key in expected
    ):
        raise RuntimeError("Prod must reuse exactly the verified model tensors")
    if model.world_size != 1:
        raise RuntimeError("the incremental probe supports one GPU only")
    return model


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
    def from_tensor(cls, tensor, *, allow_partial=False):
        if len(tensor) != INPUT_SLOTS and not (
            allow_partial and 1 <= len(tensor) <= INPUT_SLOTS
        ):
            raise ValueError("pruning probe requires 16 arrived startup frames")
        count = len(tensor)
        return cls(
            [frame.clone() for frame in tensor] + [None] * (INPUT_SLOTS - count),
            list(range(count)) + [-1] * (INPUT_SLOTS - count),
        )

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
        if self.frames[slot] is None:
            raise IndexError("native code requested a source frame not yet arrived")
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

    def install(self, state, *, allow_partial=False):
        batch = state["input_batch"]
        batch.img_batch.tensors = RingFrames.from_tensor(
            batch.img_batch.tensors, allow_partial=allow_partial
        )
        if allow_partial and 1 <= len(batch.find_inputs) < INPUT_SLOTS:
            # These are prompt-address slots, not source images. Empty image slots
            # raise until a real append installs their exact source tensor.
            template = batch.find_inputs[0]
            for slot in range(len(batch.find_inputs), INPUT_SLOTS):
                stage = deepcopy(template)
                stage.img_ids[...] = slot
                stage.img_ids_np[...] = slot
                batch.find_inputs.append(stage)
            for name in ("find_targets", "find_metadatas"):
                if hasattr(batch, name):
                    values = getattr(batch, name)
                    values.extend([None] * (INPUT_SLOTS - len(values)))
        if len(batch.find_inputs) != INPUT_SLOTS:
            raise ValueError("expected fixed 16 FindStage slots")
        for slot, stage in enumerate(batch.find_inputs):
            if (
                int(stage.img_ids.item()) != slot
                or int(stage.img_ids_np.item()) != slot
            ):
                raise ValueError("unexpected image IDs in initial source stages")
        for name in FRAME_FIELDS:
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
            for name in FRAME_FIELDS:
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
        for name in FRAME_FIELDS:
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
