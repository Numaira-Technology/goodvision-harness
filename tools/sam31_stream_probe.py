#!/usr/bin/env python3
"""GPU-only finite experiment for Prod continuation and truly incremental input.

This does not change the HTTP worker or claim bounded, infinite camera memory.
Run separately from the serving process to measure its own CUDA memory footprint.
"""

from __future__ import annotations

import argparse
import base64
import gc
import inspect
import json
import math
import sys
import time
import zlib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FRAME_FIELDS = (
    "previous_stages_out",
    "per_frame_raw_point_input",
    "per_frame_raw_box_input",
    "per_frame_visual_prompt",
    "per_frame_geometric_prompt",
    "per_frame_cur_step",
)
MAX_FRAMES = 64
MAX_JPEG_BYTES = 8 * 1024 * 1024
MAX_TOTAL_JPEG_BYTES = 64 * 1024 * 1024


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


def append_arrived_frames(model, state, images, *, frame_limit=MAX_FRAMES):
    """Probe-only append of arrived PIL images, retaining native state unchanged.

    Retention grows with the prefix and is bounded only by frame_limit. No images
    are supplied for future frames, no directory rescanning or tracker reset.
    """
    previous = state["num_frames"]
    if not images or previous + len(images) > frame_limit:
        raise ValueError("append exceeds finite probe frame limit")
    import torch

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
            raise ValueError("all source images must have the same dimensions")
        current_batch, extra_batch = state["input_batch"], delta["input_batch"]
        # Convert both tensor and numpy image IDs to the absolute native index.
        for stage in extra_batch.find_inputs:
            stage.img_ids += previous
            stage.img_ids_np += previous
        current_batch.img_batch.tensors = torch.cat(
            (current_batch.img_batch.tensors, extra_batch.img_batch.tensors), dim=0
        )
        for name in ("find_inputs", "find_targets", "find_metadatas"):
            getattr(current_batch, name).extend(getattr(extra_batch, name))
        for name in FRAME_FIELDS:
            state[name].extend(delta[name])
        state["num_frames"] = previous + len(images)
        for tracker_state in state["sam2_inference_states"]:
            tracker_state["num_frames"] = state["num_frames"]
        assert len(current_batch.find_inputs) == state["num_frames"]
        assert len(current_batch.img_batch.tensors) == state["num_frames"]
    finally:
        # reset_state(delta) would mutate stages now owned by the live state.
        delta.clear()


def make_prod_model(native):
    from sam3.model.sam3_multiplex_base import Sam3MultiplexBase
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingProd

    class CountBoundedProd(CountBoundedProcessingOrder, Sam3MultiplexTrackingProd):
        pass

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
        compile_model=False, postprocess_batch_size=1, use_batched_grounding=False
    )
    model = CountBoundedProd(**kwargs).eval()
    expected, actual = existing.state_dict(), model.state_dict()
    if expected.keys() != actual.keys() or any(
        expected[key].data_ptr() != actual[key].data_ptr() for key in expected
    ):
        raise RuntimeError("Prod must reuse exactly the verified model tensors")
    if model.world_size != 1:
        raise RuntimeError("the incremental probe supports one GPU only")
    return model


def state_metrics(state):
    import torch

    seen_objects, seen_storage = set(), set()
    tensor_bytes = {"cpu": 0, "cuda": 0}

    def visit(value):
        if id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key not in seen_storage:
                seen_storage.add(key)
                tensor_bytes[value.device.type] += storage.nbytes()
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif hasattr(value, "tensors") and hasattr(value, "mask"):
            # Official NestedTensor is a regular class, not a dataclass.
            visit(value.tensors)
            visit(value.mask)

    visit(state)
    trackers = state["sam2_inference_states"]
    return {
        "input_frames_retained": len(state["input_batch"].img_batch.tensors),
        "cached_output_frames": len(state["cached_frame_outputs"]),
        "hotstart_frames": len(
            state.get("generator_state", {}).get("hotstart_buffer", [])
        ),
        "source_stage_slots": len(state["input_batch"].find_inputs),
        "per_frame_metadata_records": len(state["previous_stages_out"]),
        "historical_object_score_records": len(
            state["tracker_metadata"].get("obj_id_to_score", {})
        ),
        "legacy_unmatched_frame_records": sum(
            len(values)
            for values in state["tracker_metadata"]
            .get("rank0_metadata", {})
            .get("unmatched_frame_inds", {})
            .values()
        ),
        "legacy_overlap_frame_records": sum(
            len(values)
            for values in state["tracker_metadata"]
            .get("rank0_metadata", {})
            .get("overlap_pair_to_frame_inds", {})
            .values()
        ),
        "tracker_nonconditioning_frames": sum(
            len(item["output_dict"]["non_cond_frame_outputs"]) for item in trackers
        ),
        "tracker_conditioning_frames": sum(
            len(item["output_dict"]["cond_frame_outputs"]) for item in trackers
        ),
        "unique_state_tensor_bytes": tensor_bytes,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def signature(output, *, width, height, max_objects):
    import numpy as np

    ids = np.asarray(output["out_obj_ids"])
    scores = np.asarray(output["out_probs"])
    masks = np.asarray(output["out_binary_masks"])
    if (
        ids.ndim != 1
        or len(ids) > max_objects
        or ids.dtype.kind not in "iu"
        or len(set(ids.tolist())) != len(ids)
        or scores.shape != ids.shape
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
        or masks.dtype != np.bool_
        or masks.shape != (len(ids), height, width)
    ):
        raise RuntimeError("invalid native output")
    return {
        "ids": [int(value) for value in ids],
        "scores": [float(value) for value in scores],
        "mask_values": {
            "shape": list(masks.shape),
            "dtype": masks.dtype.str,
            "encoding": "packbits-little-zlib-base64",
            "data": base64.b64encode(
                zlib.compress(np.packbits(masks, bitorder="little").tobytes())
            ).decode("ascii"),
        },
        "mask_areas": [int(mask.sum()) for mask in masks],
    }


class Arrivals:
    def __init__(self, paths, side):
        self.paths, self.side = paths, side
        self.loaded = 0
        self.total_bytes = 0
        self.original_dimensions = None

    def receive(self, stop):
        from workers.sam_scene.app import decode_jpeg
        from workers.sam_scene.sam31 import VideoSettings

        images = []
        try:
            while self.loaded < stop:
                path = self.paths[self.loaded]
                if not 0 < path.stat().st_size <= MAX_JPEG_BYTES:
                    raise ValueError("source JPEG exceeds byte limit")
                with path.open("rb") as source:
                    data = source.read(MAX_JPEG_BYTES + 1)
                if len(data) > MAX_JPEG_BYTES:
                    raise ValueError("source JPEG grew beyond byte limit")
                self.total_bytes += len(data)
                if self.total_bytes > MAX_TOTAL_JPEG_BYTES:
                    raise ValueError("source JPEG set exceeds byte budget")
                image = decode_jpeg(data, VideoSettings())
                images.append(image)
                if self.original_dimensions not in (None, image.size):
                    raise ValueError("source dimensions changed")
                self.original_dimensions = image.size
                image.thumbnail((self.side, self.side))
                self.loaded += 1
            return images
        except Exception:
            for image in images:
                image.close()
            raise


def run_case(
    model,
    paths,
    *,
    mode,
    chunk,
    side,
    budget_seconds,
    frame_limit=MAX_FRAMES,
    retention=None,
    finalizer=None,
):
    import torch

    arrivals = Arrivals(paths, side)
    state = None
    outputs, checkpoints = {}, []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        initial = chunk if mode == "arrived_append" else len(paths)
        images = arrivals.receive(initial)
        try:
            width, height = images[0].size
            state = model.init_state(
                resource_path=images,
                offload_video_to_cpu=True,
                async_loading_frames=False,
            )
        finally:
            for image in images:
                image.close()
        model.add_prompt(inference_state=state, frame_idx=0, text_str="person")
        if retention is not None:
            retention.install(state)
        step = len(paths) if mode == "finite_full" else chunk
        for start in range(0, len(paths), step):
            end = min(start + step, len(paths))
            if mode == "arrived_append" and start:
                images = arrivals.receive(end)
                try:
                    if retention is None:
                        append_arrived_frames(
                            model, state, images, frame_limit=frame_limit
                        )
                    else:
                        retention.append(model, state, images, frame_limit=frame_limit)
                finally:
                    for image in images:
                        image.close()
            if mode == "arrived_append":
                assert arrivals.loaded == state["num_frames"] == end
            # The detector needs an exclusive count; the probe's explicit adapter
            # translates just the processing-order method to its inclusive offset.
            for index, output in model.propagate_in_video(
                inference_state=state,
                start_frame_idx=start,
                max_frame_num_to_track=end - start,
                reverse=False,
                is_last_batch=end == len(paths) and finalizer is None,
            ):
                if type(index) is not int or index in outputs or not 0 <= index < end:
                    raise RuntimeError("native output sequence violated source order")
                if outputs and index != max(outputs) + 1:
                    raise RuntimeError("native output skipped a source frame")
                outputs[index] = signature(
                    output,
                    width=width,
                    height=height,
                    max_objects=model.max_num_objects,
                )
                if time.monotonic() - started > budget_seconds:
                    raise RuntimeError("probe exceeded cooperative run budget")
            torch.cuda.synchronize()
            if retention is not None:
                retention.prune(state, emitted_through=max(outputs, default=-1))
            checkpoints.append(
                {
                    "accepted_prefix_frames": end,
                    "decoded_source_frames": arrivals.loaded,
                    "processed_through_native_index": end - 1,
                    "emitted_through_native_index": max(outputs, default=None),
                    **state_metrics(state),
                }
            )
        if finalizer is not None:
            for index, output in finalizer(model, state):
                if (
                    type(index) is not int
                    or index != len(outputs)
                    or not index < len(paths)
                ):
                    raise RuntimeError("empty EOF violated source prefix")
                outputs[index] = signature(
                    output,
                    width=width,
                    height=height,
                    max_objects=model.max_num_objects,
                )
                if time.monotonic() - started > budget_seconds:
                    raise RuntimeError("probe exceeded cooperative run budget")
            torch.cuda.synchronize()
            if retention is not None:
                retention.prune(state, emitted_through=max(outputs, default=-1))
            checkpoints.append(
                {
                    "accepted_prefix_frames": len(paths),
                    "decoded_source_frames": arrivals.loaded,
                    "processed_through_native_index": len(paths) - 1,
                    "emitted_through_native_index": max(outputs, default=None),
                    "empty_eof": True,
                    **state_metrics(state),
                }
            )
        if list(outputs) != list(range(len(paths))):
            raise RuntimeError("final flush did not return every source frame")
        return {
            "mode": mode,
            "elapsed_seconds": time.monotonic() - started,
            "visible_observations": sum(
                len(value["ids"]) for value in outputs.values()
            ),
            "frames": outputs,
            "checkpoints": checkpoints,
        }
    finally:
        if state is not None:
            if retention is None:
                model.reset_state(state)
            state.clear()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def compare(reference, candidate):
    mismatches = []
    for index, expected in reference["frames"].items():
        actual = candidate["frames"][index]
        if (
            expected["ids"] != actual["ids"]
            or expected["mask_values"] != actual["mask_values"]
        ):
            mismatches.append(index)
    return {
        "exact_ids_and_masks": not mismatches,
        "mismatched_frame_indices": mismatches,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--budget-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if not 1 <= args.chunk_frames <= 16 or not 1 <= args.max_side <= 1280:
        parser.error("chunk frames must be 1..16 and max side 1..1280")
    if not args.chunk_frames < len(args.images) <= MAX_FRAMES:
        parser.error("supply more than one chunk, at most 64 chronological JPEGs")
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 600:
        parser.error("cooperative budget must be within 0..600 seconds")
    import numpy as np
    import torch

    from workers.sam_scene.sam31 import SAM3_COMMIT, NativeSam31, VideoSettings

    native = NativeSam31(
        replace(VideoSettings(), enabled=True, checkpoint_path=args.checkpoint)
    )
    report = {
        "sam3_commit": SAM3_COMMIT,
        "gpu_ready": False,
        "accuracy_evaluated": False,
        "continuous_session_integrated": False,
        "retention_policy": "finite_probe_no_pruning",
        "hard_frame_limit": MAX_FRAMES,
        "arrival_simulation": "decode a chunk after processing its predecessor",
        "boundary_adapter": "exclusive detector count; inclusive processing offset",
    }
    try:
        native.warmup()
        report["gpu_ready"] = native.gpu_ready
        model = make_prod_model(native)
        rng_states = [
            (module, name, deepcopy(value.bit_generator.state))
            for module in model.modules()
            for name in ("rng", "rng2")
            if isinstance((value := getattr(module, name, None)), np.random.Generator)
        ]
        cases = []
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            for mode in ("finite_full", "finite_segmented", "arrived_append"):
                torch.manual_seed(42)
                for module, name, state in rng_states:
                    getattr(module, name).bit_generator.state = deepcopy(state)
                cases.append(
                    run_case(
                        model,
                        args.images,
                        mode=mode,
                        chunk=args.chunk_frames,
                        side=args.max_side,
                        budget_seconds=args.budget_seconds,
                    )
                )
        report.update(
            cases=cases,
            full_vs_segmented=compare(cases[0], cases[1]),
            full_vs_arrived_append=compare(cases[0], cases[2]),
            segmented_vs_arrived_append=compare(cases[1], cases[2]),
            informative_person_evidence=cases[0]["visible_observations"] > 0,
            startup_pointer_horizon_matches_full=args.chunk_frames
            >= min(16, len(args.images)),
        )
        # Exact matching is a regression check, not a calibrated accuracy score.
        passed = (
            all(
                report[key]["exact_ids_and_masks"]
                for key in ("full_vs_segmented", "full_vs_arrived_append")
            )
            and report["informative_person_evidence"]
        )
        return_code = 0 if passed else 2
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return_code = 1
    finally:
        native.close()
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
