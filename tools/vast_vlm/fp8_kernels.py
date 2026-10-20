"""Prepare persistent Triton tuning artifacts; GPU work needs --execute."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import fp8_profile as p
from . import runtime

SOURCE_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/"
    "benchmarks/kernels/benchmark_moe.py"
)
SOURCE_SHA256 = "cc78ddf704150b8585d6ffb8fba8fd2d0518629b0c3e6c511a3f39d217b640dd"
CONFIG_NAME = (
    "E=256,N=512,device_name=NVIDIA_GeForce_RTX_4090,"
    "dtype=fp8_w8a8,block_shape=[128,128].json"
)
BATCH_SIZES = (1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048)
IDENTITY = {
    "model": p.MODEL_REPO,
    "revision": p.MODEL_REVISION,
    "vllm_version": p.ENGINE_VERSION,
    "device_name": "NVIDIA GeForce RTX 4090",
    "compute_capability": [8, 9],
    "dtype": "fp8_w8a8",
    "weight_block_size": [128, 128],
    "tensor_parallel_size": 1,
    "tuner_sha256": SOURCE_SHA256,
}


def validate_config(data):
    if not isinstance(data, dict) or not isinstance(data.get("triton_version"), str):
        raise ValueError("missing Triton tuning metadata")
    if set(data) != {"triton_version", *(str(n) for n in BATCH_SIZES)}:
        raise ValueError("tuning must cover all decode/prefill batch sizes")
    required = {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
    }
    for n in BATCH_SIZES:
        row = data[str(n)]
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("invalid Triton kernel configuration")
        if any(type(v) is not int or v <= 0 for v in row.values()):
            raise ValueError("invalid Triton tile parameters")
        if row["num_warps"] not in (4, 8) or row["num_stages"] not in (2, 3, 4, 5):
            raise ValueError("kernel parameters outside pinned tuner search space")
        for key in ("BLOCK_SIZE_N", "BLOCK_SIZE_K"):
            if row[key] % 128 and 128 % row[key]:
                raise ValueError("kernel tile incompatible with FP8 block size")


def verify_bundle(directory):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("tuned directory cannot be a symlink")
    config = directory / CONFIG_NAME
    manifest = directory / "tuning.json"
    if config.is_symlink() or manifest.is_symlink():
        raise ValueError("tuned artifacts cannot be symlinks")
    meta = json.loads(manifest.read_text())
    if any(meta.get(k) != v for k, v in IDENTITY.items()):
        raise ValueError("tuned kernel model/GPU/version mismatch")
    if meta.get("config_sha256") != runtime.sha(config):
        raise ValueError("tuned kernel checksum mismatch")
    data = json.loads(config.read_text())
    validate_config(data)
    if meta.get("triton_version") != data["triton_version"]:
        raise ValueError("tuned kernel Triton metadata mismatch")
    if set(directory.glob("E=*.json")) != {config}:
        raise ValueError("unexpected kernel configuration in tuned directory")
    return {
        **IDENTITY,
        "config_sha256": meta["config_sha256"],
        "triton_version": data["triton_version"],
        "config_file": CONFIG_NAME,
    }


def command(root, script, output):
    return [
        sys.executable,
        str(script),
        "--model",
        str(runtime.model_dir(root, p=p)),
        "--tp-size",
        "1",
        "--dtype",
        "fp8_w8a8",
        "--batch-size",
        *(str(n) for n in BATCH_SIZES),
        "--save-dir",
        str(output),
        "--tune",
    ]


def plan(root, output):
    script = Path(root) / "tools/vllm-0.31.0/benchmark_moe.py"
    return {
        "execute": False,
        "source": SOURCE_URL,
        **IDENTITY,
        "command": command(root, script, output),
        "batch_sizes": BATCH_SIZES,
        "requirements": [
            "idle GPU; stop owned VLM/YOLO services explicitly",
            "pinned vLLM image, Ray and verified FP8 model",
            "new directory on the persistent volume",
        ],
        "scope": "synthetic MoE kernel tuning, not twelve-camera inference",
        "capacity_or_quality_validated": False,
    }


def tune(root, output):
    root = runtime.volume_root(root)
    output = Path(output).resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("choose a new tuning directory on the persistent volume")
    with (
        runtime.exclusive(root / ".server.lock"),
        runtime.exclusive(root / ".detector-server.lock"),
        runtime.exclusive(root / ".model.lock"),
    ):
        runtime.verify_model(runtime.model_dir(root, p=p), p=p)
        if importlib.metadata.version("vllm") != p.ENGINE_VERSION:
            raise ValueError("kernel tuning requires pinned vLLM version")
        importlib.metadata.version("ray")  # Fail before any download/GPU tuning.
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        if processes:
            raise ValueError("GPU is occupied; tuning never stops existing processes")
        import torch

        if torch.cuda.device_count() != 1:
            raise ValueError("tuning requires exactly one 48GB 4090")
        gpu = torch.cuda.get_device_properties(0)
        p.validate_gpu(gpu)
        if gpu.name != IDENTITY["device_name"]:
            raise ValueError("GPU name differs from tuning config filename")
        config = json.loads((runtime.model_dir(root, p=p) / "config.json").read_text())
        text = config["text_config"]
        if (
            text["num_experts"],
            text["moe_intermediate_size"],
            config["quantization_config"]["weight_block_size"],
        ) != (256, 512, [128, 128]):
            raise ValueError("unexpected model MoE shape")
        script = root / "tools/vllm-0.31.0/benchmark_moe.py"
        if script.is_symlink() or not script.resolve().is_relative_to(root):
            raise ValueError("unsafe tuner source path")
        if not script.exists():
            with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
                source = response.read(1024 * 1024)
            if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
                raise ValueError("upstream tuner checksum mismatch")
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_bytes(source)
        if runtime.sha(script) != SOURCE_SHA256:
            raise ValueError("cached tuner checksum mismatch")
        output.mkdir(parents=True)
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("VLLM_", "RAY_"))
            and not any(t in k.upper() for t in ("API_KEY", "HF_TOKEN"))
        }
        env.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TRITON_CACHE_DIR=str(root / "cache/moe-tuning/triton"),
        )
        # The benchmark source must use the installed pinned vLLM, not cwd imports.
        env.pop("PYTHONPATH", None)
        with (output / "tuning.log").open("w") as log:
            process = subprocess.Popen(
                command(root, script, output),
                env=env,
                cwd=output,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )

            def interrupt(_signum, _frame):
                raise KeyboardInterrupt

            previous = signal.signal(signal.SIGTERM, interrupt)
            try:
                if process.wait():
                    raise RuntimeError("kernel tuning failed; inspect tuning.log")
            finally:
                signal.signal(signal.SIGTERM, previous)
                runtime.stop_process_group(process)
        result = json.loads((output / CONFIG_NAME).read_text())
        validate_config(result)
        runtime.write_json(
            output / "tuning.json",
            {
                **IDENTITY,
                "triton_version": result["triton_version"],
                "config_sha256": runtime.sha(output / CONFIG_NAME),
                "capacity_or_quality_validated": False,
            },
        )
        return verify_bundle(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "tune", "verify"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--volume", type=Path, default=Path("/models"))
    parser.add_argument(
        "--output", type=Path, default=Path("/models/kernels/qwen35-fp8-031")
    )
    args = parser.parse_args(argv)
    if not args.execute or args.action == "plan":
        result = plan(args.volume, args.output)
    elif args.action == "verify":
        result = verify_bundle(args.output)
    else:
        result = tune(args.volume, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
