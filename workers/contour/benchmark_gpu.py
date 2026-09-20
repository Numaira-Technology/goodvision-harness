"""Isolated GPU capacity probe using real, sequential JPEGs and the live adapter.

Run with serving stopped. Every lane sees the same source clip with independent
state. This measures saturated native capacity, not camera/network/display lag
or segmentation accuracy. No model download or active-service mutation occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def worker_modules(directory=None):
    if directory:
        root = Path(directory).resolve()
        for name in ("sam31.py", "continuous.py", "sam_live.py"):
            if not (root / name).is_file():
                raise ValueError(f"worker-dir is missing {name}")
        sys.path.insert(0, str(root))
        prefix = ""
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        prefix = "workers.sam_scene."
    return [
        importlib.import_module(prefix + name)
        for name in ("sam31", "continuous", "sam_live")
    ]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def memory_stats(backend):
    """Read allocator counters even with the original, uninstrumented worker."""
    method = getattr(backend, "memory_stats", None)
    if method is not None:
        return method()
    torch = getattr(backend, "_torch", None)
    if torch is None:
        torch = backend.video.backend._torch
    cuda = torch.cuda
    free, total = cuda.mem_get_info()
    return {
        "cuda_allocated_bytes": cuda.memory_allocated(),
        "cuda_reserved_bytes": cuda.memory_reserved(),
        "cuda_peak_allocated_bytes": cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": cuda.max_memory_reserved(),
        "cuda_device_free_bytes": free,
        "cuda_device_total_bytes": total,
    }


def signature(index, output):
    """Consume shared CPU output buffers before the next generator advance."""
    import numpy as np

    ids = np.asarray(output["out_obj_ids"])
    masks = np.asarray(output["out_binary_masks"])
    scores = np.asarray(output["out_probs"])
    if len(ids) != len(masks) or len(ids) != len(scores):
        raise RuntimeError("native output object dimensions differ")
    return {
        "frame": index,
        "ids": ids.tolist(),
        "mask_sha256": [hashlib.sha256(mask.tobytes()).hexdigest() for mask in masks],
        "scores": scores.tolist(),
    }


def load_images(paths, max_side):
    from PIL import Image

    images = []
    try:
        for path in paths:
            if path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("source JPEG exceeds eight MiB")
            with Image.open(path) as source:
                if (source.format != "JPEG"
                        or source.width * source.height > 3840 * 2160):
                    raise ValueError("source must be a JPEG no larger than 3840x2160")
                image = source.convert("RGB")
            image.thumbnail((max_side, max_side))
            images.append(image)
        return images
    except BaseException:
        for image in images:
            image.close()
        raise


def measure_case(backend, session_type, paths, lanes, batch_frames, max_side, case):
    torch = backend.video.backend._torch
    sessions = [
        session_type(f"lane-{i}", f"source-{i}", f"epoch-{i}", time.monotonic())
        for i in range(lanes)
    ]
    case.update(
        lanes=lanes, appends=[], memory_samples=[], outputs=[[] for _ in sessions],
        peak_counter_scope="current_case",
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    case["memory_before"] = memory_stats(backend)
    size = None
    started = time.perf_counter()
    try:
        for offset in range(0, len(paths), batch_frames):
            images = load_images(paths[offset:offset + batch_frames], max_side)
            try:
                for image in images:
                    if size is not None and size != image.size:
                        raise ValueError("source dimensions must stay constant")
                    size = image.size
                for lane, session in enumerate(sessions):
                    began = time.perf_counter()
                    hash_seconds = 0.0
                    count = 0
                    outputs = backend.run(session, images, final=False)
                    try:
                        for index, output in outputs:
                            if index != offset + count:
                                raise RuntimeError("invalid native source prefix")
                            hash_started = time.perf_counter()
                            case["outputs"][lane].append(signature(index, output))
                            hash_seconds += time.perf_counter() - hash_started
                            count += 1
                    finally:
                        outputs.close()
                    if count != len(images):
                        raise RuntimeError("native live output omitted arrived frames")
                    session.total_frames += count
                    session.emitted_index = session.total_frames - 1
                    backend.prune(session)
                    torch.cuda.synchronize()
                    case["appends"].append({
                        "lane": lane + 1, "first_frame": offset, "frames": count,
                        "native_and_retention_seconds": (
                            time.perf_counter() - began - hash_seconds
                        ),
                        "output_hash_seconds_excluded": hash_seconds,
                    })
                case["memory_samples"].append({
                    "frames_per_lane": offset + len(images), **memory_stats(backend),
                })
            finally:
                for image in images:
                    image.close()
        for session in sessions:
            if list(backend.run(session, [], final=True)):
                raise RuntimeError("causal EOF invented frames")
        durations = [row["native_and_retention_seconds"] for row in case["appends"]]
        steady = [row for row in case["appends"] if row["first_frame"] >= 32]
        steady_seconds = sum(row["native_and_retention_seconds"] for row in steady)
        steady_frames = sum(row["frames"] for row in steady)
        case.update(
            completed_frames=lanes * len(paths),
            append_seconds_p50=percentile(durations, 0.5),
            append_seconds_p95=percentile(durations, 0.95),
            saturated_native_fps=lanes * len(paths) / sum(durations),
            steady_saturated_native_fps=steady_frames / steady_seconds,
            steady_native_fps_per_lane=steady_frames / steady_seconds / lanes,
            # No claim about end-to-end latency: IO and hashing are excluded above.
            probe_wall_seconds_including_io_hash_and_eof=time.perf_counter() - started,
            memory_before_release=memory_stats(backend),
            ok=True,
        )
    finally:
        # Retain partial evidence even for CUDA OOM or native failures.
        for session in sessions:
            backend.discard(session)
        torch.cuda.synchronize()
        case["memory_after_state_release"] = memory_stats(backend)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--frame-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--worker-dir")
    result.add_argument("--lanes", type=int, nargs="+", default=[1, 2, 6])
    result.add_argument("--frames", type=int, default=96)
    result.add_argument("--batch-frames", type=int, choices=(1, 4), default=1)
    result.add_argument("--max-side", type=int, default=512)
    result.add_argument("--fixed-person-text", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 64 <= args.frames <= 512 or not 128 <= args.max_side <= 1280:
        raise ValueError("frames must be 64..512 and max-side 128..1280")
    if not args.lanes or any(not 1 <= n <= 12 for n in args.lanes):
        raise ValueError("lanes must be within 1..12")
    paths = sorted(
        path for path in args.frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )[:args.frames]
    if len(paths) != args.frames:
        raise ValueError("not enough sequential JPEGs; use zero-padded filenames")
    if any(path.stat().st_size > 8 * 1024 * 1024 for path in paths):
        raise ValueError("source JPEG exceeds eight MiB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the report before loading a model; never overwrite an earlier run.
    with args.output.open("x", encoding="utf-8") as report_file:
        report = {
            "ok": False, "cases": [], "fixed_person_text": args.fixed_person_text,
            "frames_per_lane": args.frames, "batch_frames": args.batch_frames,
            "transport_max_side": args.max_side, "encoder_side": 1008,
            "network_or_realtime_latency_measured": False,
            "accuracy_measured": False, "all_lanes_use_same_clip": True,
        }
        native = backend = None
        try:
            sam31, continuous, live = worker_modules(args.worker_dir)
            import numpy as np
            import torch

            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            report.update(
                sam3_commit=sam31.SAM3_COMMIT, torch_version=torch.__version__,
                worker_sources={
                    module.__name__: hashlib.sha256(
                        Path(module.__file__).read_bytes()
                    ).hexdigest()
                    for module in (sam31, continuous, live)
                },
                sources=[{
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                } for path in paths],
            )
            settings = {"enabled": True, "checkpoint_path": args.checkpoint}
            fields = sam31.VideoSettings.__dataclass_fields__
            if "fixed_person_text" in fields:
                settings["fixed_person_text"] = args.fixed_person_text
            elif args.fixed_person_text:
                raise ValueError("selected baseline worker has no text-cache option")
            if "gpu_diagnostics" in fields:
                settings["gpu_diagnostics"] = True
            native = sam31.NativeSam31(sam31.VideoSettings(**settings))
            native.warmup()
            report["device"] = torch.cuda.get_device_name(0)
            video = SimpleNamespace(backend=native, gpu_ready=native.gpu_ready)
            backend = live.SamLiveContinuation(video, continuous.SessionSettings())
            backend.warmup()
            report["memory_after_warmup"] = memory_stats(native)
            for lanes in args.lanes:
                # Clear only between independent measurements, not every frame.
                torch.cuda.empty_cache()
                case = {"ok": False, "lanes": lanes}
                report["cases"].append(case)
                measure_case(
                    backend, continuous.Session, paths, lanes, args.batch_frames,
                    args.max_side, case,
                )
            report["ok"] = True
        except Exception as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            # Write useful failure evidence even if CUDA teardown itself fails.
            try:
                if backend is not None:
                    backend.close()
                if native is not None:
                    native.close()
            except Exception as exc:
                report["ok"] = False
                report["cleanup_error"] = type(exc).__name__
            json.dump(report, report_file, ensure_ascii=False, indent=2)
            report_file.write("\n")
    print(f"Report: {args.output} (ok={report['ok']})")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
