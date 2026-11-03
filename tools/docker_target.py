"""Detect the Docker engine's target, rather than the launcher's host OS."""

from __future__ import annotations

import copy
import json
import subprocess
from uuid import uuid4

if __package__:
    from . import docker_modules as spec
else:
    import docker_modules as spec

PLATFORMS = {"x86_64": "linux/amd64", "amd64": "linux/amd64",
             "aarch64": "linux/arm64", "arm64": "linux/arm64"}
DRIVER_PROBE = (
    "import ctypes; d=ctypes.CDLL('libcuda.so.1'); "
    "assert d.cuInit(0)==0; n=ctypes.c_int(); v=ctypes.c_int(); "
    "assert d.cuDeviceGetCount(ctypes.byref(n))==0 and n.value>0; "
    "assert d.cuDriverGetVersion(ctypes.byref(v))==0 and v.value>=12080"
)
TORCH_PROBE = (
    "import torch; assert torch.cuda.is_available(); "
    "x=torch.ones((4,4),device='cuda'); "
    "assert (x@x).sum().item()==64; torch.cuda.synchronize()"
)


def engine_platform(run):
    result = run(["docker", "info", "--format", "{{json .}}"],
                 capture_output=True, text=True, timeout=15)
    info = json.loads(result.stdout)
    platform = PLATFORMS.get(info.get("Architecture"))
    if info.get("OSType") != "linux" or platform is None:
        raise ValueError("Use a Linux Docker engine with amd64 or arm64 CPUs")
    return platform


def available_image(image, platform, run):
    try:
        result = run(["docker", "image", "inspect", image],
                     capture_output=True, text=True, timeout=15)
    except subprocess.CalledProcessError:
        return False
    info = json.loads(result.stdout)[0]
    return f"{info['Os']}/{info['Architecture']}" == platform


def gpu_available(image, gpu_id, run, *, torch=False):
    name = "goodvision-bbox-probe-" + uuid4().hex[:12]
    try:
        run(["docker", "run", "--rm", "--name", name, "--pull", "never",
             "--platform", "linux/amd64", "--network", "none",
             "--gpus", "device=" + gpu_id, "--entrypoint", "python", image,
             "-c", TORCH_PROBE if torch else DRIVER_PROBE],
            capture_output=True, text=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    finally:
        # A timed-out Docker CLI does not necessarily stop its container.
        try:
            run(["docker", "rm", "-f", name], capture_output=True,
                text=True, timeout=10)
        except subprocess.SubprocessError:
            pass


def select_bbox(settings, run, *, platform=None, building=False, probes=()):
    native = engine_platform(run)
    target = platform or native
    if target not in set(PLATFORMS.values()):
        raise ValueError("Unsupported Bbox target platform")
    policy = settings.get("bbox_device", "auto")
    if policy == "cuda" and target != "linux/amd64":
        raise ValueError("The supplied CUDA Bbox runtime requires linux/amd64")
    device, reason = "cpu", "CPU selected"
    if policy == "cuda" and building:
        device, reason = "cuda", "Explicit CUDA cross-build"
    elif policy != "cpu" and native == target == "linux/amd64":
        version = settings.get("module_versions", {}).get("bbox", settings["version"])
        candidates = [*probes, f"goodvision-bbox:{version}-cuda-amd64",
                      f"goodvision-bbox:{version}-cuda",
                      spec.runtime_image(settings, "2.10")]
        image = next((ref for ref in dict.fromkeys(candidates)
                      if available_image(ref, native, run)), None)
        torch = image is not None
        if image is None:
            cpu = settings.get("module_images", {}).get("bbox", {}).get("bbox")
            candidates = [ref for ref in [cpu, "python:3.12-slim-bookworm"] if ref]
            image = next((ref for ref in candidates
                          if available_image(ref, native, run)), None)
            if image is None and building:
                image = "python:3.12-slim-bookworm"
                run(["docker", "pull", "--platform", native, image])
        if image and gpu_available(image, settings["gpu_id"], run, torch=torch):
            device, reason = "cuda", "Container GPU probe passed"
        else:
            reason = "Container GPU unavailable; using CPU"
            if policy == "cuda":
                raise ValueError("CUDA requested but Docker GPU access is unavailable")
    elif policy == "cuda":
        raise ValueError("CUDA cannot run on this Docker engine")
    elif policy == "auto":
        reason = "Native CPU target; CUDA variant is not supported on this architecture"
    return {"platform": target, "device": device, "policy": policy, "reason": reason}


def resolve_bbox(settings, run, **options):
    selected = select_bbox(settings, run, **options)
    result = copy.deepcopy(settings)
    old_device = spec.bbox_device(settings)
    old_image = spec.bbox_image(settings)
    result.update(bbox_platform=selected["platform"],
                  bbox_runtime_device=selected["device"])
    if (old_device == selected["device"]
            and available_image(old_image, selected["platform"], run)):
        result.setdefault("module_images", {})["bbox"] = {"bbox": old_image}
    else:
        result.get("module_images", {}).pop("bbox", None)
    return result, selected
