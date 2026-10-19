#!/usr/bin/env python3
"""GPU-only comparison of native session postprocessing at one/four frames.

Run independently of active serving/inference. One strict-loaded model is shared
by both profiles. The same sixteen local JPEGs run through fresh sessions, then
empty EOF; file loading, observation encoding and network are outside timed stages.
This is a finite performance/equivalence diagnostic, not an accuracy benchmark.
"""

from __future__ import annotations

import argparse
import base64
import gc
import importlib
import io
import json
import math
import random
import sys
import time
import traceback
import zlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace


def worker_modules(worker_dir=None):
    if worker_dir:
        directory = Path(worker_dir).resolve()
        if not all(
            (directory / name).is_file() for name in ("sam31.py", "continuous.py")
        ):
            raise ValueError("worker-dir must contain sam31.py and continuous.py")
        sys.path.insert(0, str(directory))
        return importlib.import_module("sam31"), importlib.import_module("continuous")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return (
        importlib.import_module("workers.sam_scene.sam31"),
        importlib.import_module("workers.sam_scene.continuous"),
    )


def load_images(directory, *, start_index=0, max_side=512):
    from PIL import Image

    if type(start_index) is not int or not 0 <= start_index <= 4096:
        raise ValueError("start-index must be within 0..4096")
    if type(max_side) is not int or not 128 <= max_side <= 640:
        raise ValueError("max-side must be within 128..640")
    paths = sorted(
        path
        for path in Path(directory).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg"} and path.is_file()
    )
    paths = paths[start_index : start_index + 16]
    if len(paths) != 16:
        raise ValueError("exactly sixteen existing JPEGs are required")
    images, sources = [], []
    total_bytes = total_pixels = 0
    source_size = None
    try:
        for path in paths:
            size = path.stat().st_size
            if not 0 < size <= 8 * 1024 * 1024:
                raise ValueError("source JPEG exceeds eight MiB")
            total_bytes += size
            if total_bytes > 24 * 1024 * 1024:
                raise ValueError("sixteen source JPEGs exceed 24 MiB")
            with path.open("rb") as source:
                data = source.read(8 * 1024 * 1024 + 1)
            if len(data) != size or len(data) > 8 * 1024 * 1024:
                raise ValueError("source JPEG changed or exceeded byte budget")
            with Image.open(io.BytesIO(data)) as opened:
                if opened.format != "JPEG":
                    raise ValueError("source must be JPEG")
                pixels = opened.width * opened.height
                total_pixels += pixels
                if pixels > 3840 * 2160 or total_pixels > 64 * 1024 * 1024:
                    raise ValueError("source JPEG pixel budget exceeded")
                if source_size is not None and opened.size != source_size:
                    raise ValueError("all source JPEG dimensions must match")
                source_size = opened.size
                image = opened.convert("RGB")
            image.thumbnail((max_side, max_side))
            images.append(image)
            sources.append(
                {
                    "file": path.name,
                    "jpeg_bytes": len(data),
                    "source_dimensions": source_size,
                    "model_source_dimensions": image.size,
                }
            )
        return images, sources
    except BaseException:
        for image in images:
            image.close()
        raise


def frame_signature(index, output, image_size):
    import numpy as np

    width, height = image_size
    ids = np.asarray(output["out_obj_ids"])
    masks = np.asarray(output["out_binary_masks"])
    scores = np.asarray(output["out_probs"])
    if (
        type(index) is not int
        or ids.ndim != 1
        or len(ids) > 16
        or ids.dtype.kind not in "iu"
        or len(set(ids.tolist())) != len(ids)
        or masks.shape != (len(ids), height, width)
        or masks.dtype != np.bool_
        or scores.shape != ids.shape
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
        or (ids < 0).any()
    ):
        raise RuntimeError("native benchmark output violates source/mask contract")
    return {
        "frame_sequence": index,
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


def compare(first, second):
    a, b = first["frames"], second["frames"]
    differences = []
    if len(a) != len(b):
        return {"equal_ids_and_masks": False, "different_frames": "frame_count"}
    for one, four in zip(a, b, strict=True):
        if any(
            one[key] != four[key] for key in ("frame_sequence", "ids", "mask_values")
        ):
            differences.append(one["frame_sequence"])
    return {"equal_ids_and_masks": not differences, "different_frames": differences}


def measure_session(backend, session_type, images, *, deadline):
    """Time real adapter stages; observations are encoded after every CUDA stage."""
    torch = backend.video.backend._torch
    session = session_type("benchmark", "benchmark", "benchmark", time.monotonic())
    stages = {}
    outputs = []

    def stage(name, action):
        if time.monotonic() >= deadline:
            raise TimeoutError("cooperative benchmark budget exceeded")
        torch.cuda.synchronize()
        before = time.perf_counter()
        value = action()
        torch.cuda.synchronize()
        stages[name] = time.perf_counter() - before
        return value

    torch.cuda.reset_peak_memory_stats()
    before_allocated = torch.cuda.memory_allocated()
    started = time.perf_counter()
    try:
        outputs.extend(
            stage(
                "append_16_seconds",
                lambda: list(backend.run(session, images, final=False)),
            )
        )
        emitted_before_eof = len(outputs)
        session.total_frames = len(images)
        session.emitted_index = outputs[-1][0] if outputs else -1
        stage("retention_seconds", lambda: backend.prune(session))
        outputs.extend(
            stage(
                "empty_eof_seconds", lambda: list(backend.run(session, [], final=True))
            )
        )
        if [index for index, _ in outputs] != list(range(len(images))):
            raise RuntimeError("native session did not emit every source exactly once")
    finally:
        cleanup_started = time.perf_counter()
        backend.discard(session)
        torch.cuda.synchronize()
        stages["release_seconds"] = time.perf_counter() - cleanup_started
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    after_allocated = torch.cuda.memory_allocated()
    encoding_started = time.perf_counter()
    signatures = [
        frame_signature(index, output, images[index].size) for index, output in outputs
    ]
    return {
        "stages": stages,
        "native_session_seconds": elapsed,
        "observation_encoding_seconds_excluded": time.perf_counter() - encoding_started,
        "emitted_before_eof": emitted_before_eof,
        "emitted_at_empty_eof": len(outputs) - emitted_before_eof,
        "cuda_allocated_before": before_allocated,
        "cuda_peak_allocated": peak_allocated,
        "cuda_allocated_after_release": after_allocated,
        "frames": signatures,
    }


def run(args):
    if type(args.repeats) is not int or not 1 <= args.repeats <= 5:
        raise ValueError("repeats must be within 1..5")
    if type(args.hotstart_delay) is not int or args.hotstart_delay not in {3, 5, 15}:
        raise ValueError("hotstart-delay must be 3, 5, or 15")
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 600:
        raise ValueError("budget-seconds must be within (0,600]")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("output report must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    sam31, continuous = worker_modules(args.worker_dir)
    import numpy as np
    import torch

    loaded_at = time.perf_counter()
    images, sources = load_images(
        args.frame_dir, start_index=args.start_index, max_side=args.max_side
    )
    report = {
        "ok": False,
        "sam3_commit": sam31.SAM3_COMMIT,
        "network_measured": False,
        "accuracy_measured": False,
        "compile_mode": "off",
        "hotstart_delay": args.hotstart_delay,
        "sources": sources,
        "source_frames_per_session": 16,
        "input_loading_seconds_excluded": time.perf_counter() - loaded_at,
        "source_pixel_size_is_not_encoder_size": True,
        "encoder_side": 1008,
        "cases": [],
    }
    native, backends = None, {}
    deadline = time.monotonic() + args.budget_seconds
    try:
        started = time.perf_counter()
        native = sam31.NativeSam31(
            sam31.VideoSettings(
                enabled=True, checkpoint_path=args.checkpoint, compile_mode="off"
            )
        )
        native.warmup()
        report["model_load_and_cuda_warmup_seconds_excluded"] = (
            time.perf_counter() - started
        )
        report["gpu_ready"] = native.gpu_ready
        if not native.gpu_ready:
            raise RuntimeError("native CUDA warmup did not attest readiness")
        report["device"] = torch.cuda.get_device_name(0)
        report["torch_version"] = torch.__version__
        video = SimpleNamespace(backend=native, gpu_ready=native.gpu_ready)
        report["profile_warmups"] = []
        for size in (1, 4):
            backend = continuous.NativeContinuation(
                video,
                continuous.SessionSettings(
                    enabled=True,
                    hotstart_delay=args.hotstart_delay,
                    postprocess_batch_size=size,
                ),
            )
            backends[size] = backend
            backend.warmup()
            warm = measure_session(
                backend, continuous.Session, images, deadline=deadline
            )
            report["profile_warmups"].append(
                {
                    "postprocess_batch_size": size,
                    "native_session_seconds": warm["native_session_seconds"],
                }
            )
        rngs = {}
        for backend in backends.values():
            for module in backend.model.modules():
                for name in ("rng", "rng2"):
                    value = getattr(module, name, None)
                    if isinstance(value, np.random.Generator):
                        rngs[id(value)] = (value, deepcopy(value.bit_generator.state))
        for repeat in range(args.repeats):
            for size in (1, 4) if repeat % 2 == 0 else (4, 1):
                torch.manual_seed(42)
                np.random.seed(42)
                random.seed(42)
                for rng, state in rngs.values():
                    rng.bit_generator.state = deepcopy(state)
                gc.collect()
                case = measure_session(
                    backends[size], continuous.Session, images, deadline=deadline
                )
                case.update(repeat=repeat + 1, postprocess_batch_size=size)
                report["cases"].append(case)
                output.write_text(
                    json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
                )
        pairs = []
        for repeat in range(1, args.repeats + 1):
            pair = {
                row["postprocess_batch_size"]: row
                for row in report["cases"]
                if row["repeat"] == repeat
            }
            pairs.append({"repeat": repeat, **compare(pair[1], pair[4])})
        report["comparisons"] = pairs
        report["mean_native_session_seconds"] = {
            str(size): float(
                np.mean(
                    [
                        row["native_session_seconds"]
                        for row in report["cases"]
                        if row["postprocess_batch_size"] == size
                    ]
                )
            )
            for size in (1, 4)
        }
        report["informative_person_evidence"] = any(
            frame["ids"] for row in report["cases"] for frame in row["frames"]
        )
        report["ok"] = all(pair["equal_ids_and_masks"] for pair in pairs)
        report["caveat"] = (
            "Times include native state initialization, person prompt, propagation, "
            "mask postprocessing, pruning, EOF and CUDA sync; exclude JPEG loading, "
            "observation encoding and HTTP. Exact equality on this clip does not "
            "establish accuracy."
        )
        return report
    except BaseException as error:
        report["error_type"] = type(error).__name__
        raise
    finally:
        for backend in backends.values():
            backend.close()
        if native is not None:
            native.close()
        for image in images:
            image.close()
        output.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-dir")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--hotstart-delay", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-seconds", type=float, default=600)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception:
        traceback.print_exc()
        return 1
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "ok",
                    "gpu_ready",
                    "comparisons",
                    "mean_native_session_seconds",
                    "informative_person_evidence",
                )
            }
        )
    )
    return int(not result["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
