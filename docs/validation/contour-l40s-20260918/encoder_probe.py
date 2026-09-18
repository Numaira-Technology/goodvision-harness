"""Architecture-only CUDA probe; NO checkpoint, detections, or capacity claim."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from sam3.model_builder import _create_multiplex_tri_backbone
from sam31 import SAM3_COMMIT, verify_installed_source


def memory():
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "device_free": free,
        "device_total": total,
    }


def step(model, inputs):
    # Like the worker, each call has a bounded autocast context (no persistent
    # BF16 parameter cache across requests). Weights remain FP32.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_installed_source()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.35)
    report = {
        "ok": False,
        "scope": "randomly_initialized_architecture_only",
        "checkpoint_loaded": False,
        "real_video_inference": False,
        "full_pipeline_capacity_measured": False,
        "sam3_commit": SAM3_COMMIT,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "visible_devices": torch.cuda.device_count(),
        "cpu_threads": torch.get_num_threads(),
        "memory_fraction_limit": 0.35,
        "compile": False,
        "fa3": False,
        "rope_real": False,
        "weight_dtype": "float32",
        "autocast_dtype": "bfloat16",
        "input_dtype": "float16",
        "input_shape_per_image": [3, 1008, 1008],
        "autocast_context": "fresh_per_forward",
        "cases": [],
    }
    with args.output.open("x") as handle:
        try:
            model = _create_multiplex_tri_backbone(
                compile_mode=None, use_fa3=False, use_rope_real=False,
            ).cuda().eval()
            report["vision_parameter_bytes"] = sum(
                p.numel() * p.element_size() for p in model.parameters()
            )
            report["trunk_parameter_bytes"] = sum(
                p.numel() * p.element_size() for p in model.trunk.parameters()
            )
            for repeat in range(3):
                cases = [("vit_trunk", 1), ("vision_three_heads", 1),
                         ("vision_three_heads", 2), ("vision_three_heads", 4)]
                if repeat % 2:
                    cases.reverse()
                for component, batch in cases:
                    inputs = torch.rand(
                        batch, 3, 1008, 1008, device="cuda", dtype=torch.float16,
                    ) * 2 - 1
                    forward = model.trunk if component == "vit_trunk" else model
                    torch.cuda.empty_cache()
                    for _ in range(15):
                        output = step(forward, inputs)
                        del output
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    row = {"repeat": repeat + 1, "component": component,
                           "batch": batch, "warmup_calls": 15,
                           "before": memory(), "wall_ms": [], "cuda_event_ms": []}
                    report["cases"].append(row)
                    for _ in range(50):
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start = time.perf_counter()
                        begin.record()
                        output = step(forward, inputs)
                        end.record()
                        end.synchronize()
                        row["wall_ms"].append((time.perf_counter() - start) * 1000)
                        row["cuda_event_ms"].append(begin.elapsed_time(end))
                        del output
                    samples = row["wall_ms"]
                    row.update(
                        wall_mean_ms=statistics.mean(samples),
                        wall_p50_ms=statistics.median(samples),
                        wall_p95_ms=sorted(samples)[math.ceil(0.95 * len(samples))-1],
                        component_images_per_second=(
                            batch * 1000 / statistics.mean(samples)
                        ),
                        after=memory(),
                    )
                    print(json.dumps({k: row[k] for k in (
                        "repeat", "component", "batch", "wall_mean_ms",
                        "component_images_per_second",
                    )}), flush=True)
                    del inputs
            report["ok"] = True
        except Exception as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
