"""Detect local NVIDIA devices and start the isolated, offline model services.

Python 3.10+ standard library only. --dry-run never writes files or calls Docker.
Existing .env/cloud settings and saved console choices are not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIRECTORY = ".models/qwen3-vl-30b-a3b-instruct-fp8"
SETTINGS = {
    "HARNESS_PORT",
    "HARNESS_BIND_ADDRESS",
    "LOCAL_INFERENCE_ENABLED",
    "LOCAL_TRACKING_ENABLED",
    "LOCAL_TRACKING_FPS",
    "LOCAL_INFERENCE_MODEL",
    "LOCAL_INFERENCE_IMAGE",
    "LOCAL_INFERENCE_MODEL_DIRECTORY",
    "LOCAL_INFERENCE_MAX_MODEL_LEN",
    "LOCAL_INFERENCE_MAX_TOKENS",
    "LOCAL_INFERENCE_MAX_NUM_SEQS",
    "LOCAL_INFERENCE_MAX_BATCHED_TOKENS",
    "LOCAL_INFERENCE_EXECUTION_MODE",
    "LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION",
    "SAM31_CHECKPOINT_FILE",
}


class LaunchError(ValueError):
    """An actionable setup error before any container mutation."""


def parse_nvidia_devices(output: str) -> list[dict]:
    """Parse only NVIDIA's index, name and total VRAM query; fail closed."""
    devices = []
    for row in csv.reader(output.splitlines()):
        if not row or not any(part.strip() for part in row):
            continue
        if len(row) != 3:
            raise LaunchError("nvidia-smi returned an invalid GPU inventory")
        index, name, memory = (part.strip() for part in row)
        if not index.isdecimal() or not name or not memory.isdecimal():
            raise LaunchError("nvidia-smi returned an invalid GPU inventory")
        devices.append({"index": int(index), "name": name, "memory_mib": int(memory)})
    indices = [device["index"] for device in devices]
    if len(indices) != len(set(indices)):
        raise LaunchError("nvidia-smi returned duplicate GPU indices")
    if not devices:
        raise LaunchError(
            "No NVIDIA GPU detected. Non-NVIDIA cards are not supported; "
            "use the ordinary cloud deployment instead."
        )
    return sorted(devices, key=lambda device: device["index"])


def detect_nvidia_devices() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as error:
        raise LaunchError(
            "nvidia-smi is unavailable: no supported NVIDIA GPU/driver detected. "
            "Non-NVIDIA cards are treated as unavailable."
        ) from error
    except (subprocess.SubprocessError, OSError) as error:
        raise LaunchError(
            "NVIDIA GPU detection failed. Check the driver and nvidia-smi "
            "before enabling local GPU services."
        ) from error
    return parse_nvidia_devices(result.stdout)


def read_local_settings(path: Path) -> dict[str, str]:
    """Read the small launcher subset; let Compose handle all cloud credentials.

    Literal unquoted/single/double-quoted values are supported. Refuse .env
    expansion in this subset rather than silently disagreeing with Compose.
    """
    if not path.is_file():
        return {}
    settings = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in SETTINGS:
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            quote = value[0]
            closing = value.find(quote, 1)
            tail = value[closing + 1 :].strip() if closing >= 0 else ""
            if closing < 0 or (tail and not tail.startswith("#")):
                raise LaunchError(f"{key}: use a single-line literal .env value")
            value = value[1:closing]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if "$" in value:
            raise LaunchError(f"{key}: use a literal value, without .env interpolation")
        if value:
            settings[key] = value
    return settings


def plan_environment(
    devices: Sequence[Mapping], settings: Mapping[str, str], *, root: Path = ROOT
) -> dict[str, str]:
    if not devices:
        raise LaunchError("At least one NVIDIA GPU is required")
    indices = sorted(int(device["index"]) for device in devices)
    if len(indices) != len(set(indices)) or indices[0] < 0:
        raise LaunchError("GPU indices must be distinct nonnegative integers")
    shared = len(indices) == 1
    result = {
        key: value for key, value in settings.items() if key in SETTINGS and value
    }
    result.update(
        LOCAL_GPU_AVAILABLE="true",
        LOCAL_GPU_COUNT=str(len(indices)),
        LOCAL_INFERENCE_GPU_ID=str(indices[0]),
        LOCAL_TRACKING_GPU_ID=str(indices[0] if shared else indices[1]),
        LOCAL_INFERENCE_ENDPOINT="http://local-inference:8000/v1/chat/completions",
        LOCAL_TRACKING_ENDPOINT="http://local-tracking:8092/v1/video/sessions",
    )
    defaults = {
        "LOCAL_INFERENCE_ENABLED": "true",
        "LOCAL_TRACKING_ENABLED": "true",
        "LOCAL_TRACKING_FPS": "4",
        "LOCAL_INFERENCE_MODEL": "local-vision",
        "LOCAL_INFERENCE_IMAGE": "vllm/vllm-openai:v0.17.1",
        "LOCAL_INFERENCE_MAX_MODEL_LEN": "8192",
        "LOCAL_INFERENCE_MAX_TOKENS": "1600",
        "LOCAL_INFERENCE_MAX_NUM_SEQS": "4" if shared else "8",
        "LOCAL_INFERENCE_MAX_BATCHED_TOKENS": "4096",
        "LOCAL_INFERENCE_EXECUTION_MODE": (
            "--enforce-eager" if shared else "--no-enforce-eager"
        ),
        "LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION": "0.80" if shared else "0.90",
        "LOCAL_INFERENCE_MODEL_DIRECTORY": DEFAULT_MODEL_DIRECTORY,
        "SAM31_CHECKPOINT_FILE": ".models/sam3.1/sam3.1_multiplex.pt",
    }
    for key, value in defaults.items():
        result.setdefault(key, value)
    for key in ("LOCAL_INFERENCE_ENABLED", "LOCAL_TRACKING_ENABLED"):
        result[key] = result[key].lower()
        if result[key] not in {"true", "false"}:
            raise LaunchError(f"{key} must be true or false")
    try:
        utilization = float(result["LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION"])
        context = int(result["LOCAL_INFERENCE_MAX_MODEL_LEN"])
        tokens = int(result["LOCAL_INFERENCE_MAX_TOKENS"])
        sequences = int(result["LOCAL_INFERENCE_MAX_NUM_SEQS"])
        batched_tokens = int(result["LOCAL_INFERENCE_MAX_BATCHED_TOKENS"])
        tracking_fps = float(result["LOCAL_TRACKING_FPS"])
    except ValueError as error:
        raise LaunchError(
            "Local model memory/context/token settings must be numeric"
        ) from error
    if not math.isfinite(utilization) or not 0 < utilization < 1:
        raise LaunchError("LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION must be within (0,1)")
    if context < 4096 or not 1 <= tokens <= min(16384, context - 1):
        raise LaunchError(
            "Model context must be >=4096; output tokens must be 1..16384 "
            "and below context"
        )
    if not 1 <= sequences <= 128 or not 512 <= batched_tokens <= 65536:
        raise LaunchError("Sequence count must be 1..128 and batch tokens 512..65536")
    if not math.isfinite(tracking_fps) or not 0.5 <= tracking_fps <= 30:
        raise LaunchError("LOCAL_TRACKING_FPS must be within 0.5..30")
    if result["LOCAL_INFERENCE_EXECUTION_MODE"] not in {
        "--enforce-eager",
        "--no-enforce-eager",
    }:
        raise LaunchError(
            "Execution mode must be --enforce-eager or --no-enforce-eager"
        )
    for key in ("LOCAL_INFERENCE_MODEL_DIRECTORY", "SAM31_CHECKPOINT_FILE"):
        path = Path(result[key]).expanduser()
        result[key] = str(
            (root / path).resolve() if not path.is_absolute() else path.resolve()
        )
    return result


def _model_json(directory: Path, name: str) -> dict:
    try:
        data = json.loads((directory / name).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data:
            raise ValueError("empty model metadata")
        return data
    except (OSError, ValueError) as error:
        raise LaunchError(f"Missing or invalid local model metadata: {name}") from error


def validate_weights(
    environment: Mapping[str, str], devices: Sequence[Mapping] | None = None,
) -> None:
    directory = Path(environment["LOCAL_INFERENCE_MODEL_DIRECTORY"])
    checkpoint = Path(environment["SAM31_CHECKPOINT_FILE"])
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise LaunchError(
            f"Missing complete local vision model at {directory}. "
            "See docs/local-gpu.md; this launcher does not download model weights."
        )
    config = _model_json(directory, "config.json")
    _model_json(directory, "preprocessor_config.json")
    if config.get("model_type") == "qwen3_vl_moe":
        _model_json(directory, "video_preprocessor_config.json")
    tokenizer = _model_json(directory, "tokenizer_config.json")
    token_files = (
        ("tokenizer.json",), ("vocab.json", "merges.txt"), ("tokenizer.model",),
    )
    chosen_tokens = next((
        group for group in token_files
        if all((directory / name).is_file() and (directory / name).stat().st_size
               for name in group)
    ), None)
    if chosen_tokens is None:
        raise LaunchError("Local model tokenizer files are missing or empty")
    for name in chosen_tokens:
        if name.endswith(".json"):
            _model_json(directory, name)
    template = tokenizer.get("chat_template")
    for name in ("chat_template.json", "chat_template.jinja"):
        if (directory / name).is_file():
            template = (
                _model_json(directory, name).get("chat_template")
                if name.endswith(".json") else
                (directory / name).read_text(encoding="utf-8")
            )
            break
    if not isinstance(template, str) or not template.strip():
        raise LaunchError("Local Instruct model chat template is missing or empty")
    if directory.name == Path(DEFAULT_MODEL_DIRECTORY).name and (
        config.get("model_type") != "qwen3_vl_moe"
        or not isinstance(config.get("quantization_config"), dict)
        or config["quantization_config"].get("quant_method") != "fp8"
    ):
        raise LaunchError("Default model directory must contain Qwen3-VL MoE FP8")
    weights = set(directory.glob("*.safetensors"))
    if not weights:
        raise LaunchError(f"No safetensors model weights found in {directory}")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            shards = set(weight_map.values())
            if not shards or any(
                not isinstance(shard, str)
                or Path(shard).name != shard
                or not shard.endswith(".safetensors")
                or "/" in shard or "\\" in shard
                or not (directory / shard).is_file()
                or (directory / shard).stat().st_size == 0
                for shard in shards
            ):
                raise ValueError("missing model shards")
            weights = {directory / shard for shard in shards}
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise LaunchError(
                "Local model shard index is invalid or incomplete"
            ) from error
    elif len(weights) != 1 or re.search(r"-\d+-of-\d+", next(iter(weights)).name):
        raise LaunchError("Sharded model is missing model.safetensors.index.json")
    for weight in weights:
        with weight.open("rb") as stream:
            prefix = stream.read(64)
        if not prefix or prefix.startswith(b"version https://git-lfs.github.com/"):
            raise LaunchError(
                f"Model weight is empty or a Git LFS pointer: {weight.name}",
            )
    if devices is not None:
        device_id = int(environment["LOCAL_INFERENCE_GPU_ID"])
        device = next(item for item in devices if int(item["index"]) == device_id)
        if device.get("memory_mib"):
            budget = (
                int(device["memory_mib"]) * 1024**2
                * float(environment["LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION"])
            )
            # This rejects an impossible weight budget, not a capacity estimate:
            # KV cache, vision activations, CUDA graphs and other workers add more.
            if sum(weight.stat().st_size for weight in weights) >= budget:
                raise LaunchError(
                    "Model weights exceed the configured inference GPU memory budget. "
                    "Use more VRAM or revise LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION "
                    "while reserving space for tracking; GPU assignment is unchanged."
                )
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise LaunchError(f"Missing SAM3.1 checkpoint: {checkpoint}")


def compose_command(
    *,
    root: Path,
    env_file: Path | None,
    with_paddle: bool,
    build: bool,
    streams: int = 1,
    fleet_file: Path | None = None,
) -> list[str]:
    command = ["docker", "compose", "--project-directory", str(root)]
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    command.extend(
        ["-f", str(root / "compose.yaml"), "-f", str(root / "compose.local-gpu.yaml")]
    )
    if fleet_file is not None:
        command.extend(["-f", str(fleet_file)])
    if with_paddle:
        command.extend(["--profile", "rich"])
    command.extend(["up", "--detach", "--wait", "--wait-timeout", "900"])
    if build:
        command.append("--build")
    command.extend(["harness", "local-inference", "local-tracking"])
    command.extend(f"harness-{index:02d}" for index in range(2, streams + 1))
    if with_paddle:
        command.append("rich-label-worker")
    return command


def fleet_overlay(streams: int, *, root: Path = ROOT, port: int = 8080) -> str:
    """Each camera gets isolated runtime/data while VLM and SAM weights are shared.

    JSON flow collections are valid YAML. !override avoids inheriting the base
    host port and aliases, requiring Compose 2.24.4+. Only the primary console
    publishes a host port; its gateway reaches other cameras on the network.
    """
    if type(streams) is not int or streams < 2:
        raise LaunchError("Fleet requires at least two camera streams")
    if type(port) is not int or not 1 <= port <= 65535:
        raise LaunchError("Unified console port must be within 1..65535")
    lines = ["# Generated by tools/start_local_gpu.py; do not edit.", "services:"]
    for index in range(1, streams + 1):
        camera = f"camera-{index:02d}"
        service = "harness" if index == 1 else f"harness-{index:02d}"
        env = {
            "HARNESS_FLEET_CAPACITY": str(streams),
            "HARNESS_FLEET_CAMERA_INDEX": str(index),
            "HARNESS_FLEET_PRIMARY_URL": "http://harness:8080",
            "HARNESS_SOURCE_ID": camera,
            "HARNESS_STREAM_URL": f"${{LOCAL_CAMERA_{index:02d}_STREAM_URL:-}}",
            "RTSP_URL": "",
            # New cameras start idle until their stream has been configured.
            "HARNESS_AUTOSTART": f"${{LOCAL_CAMERA_{index:02d}_AUTOSTART:-false}}",
            "FACE_INFERENCE_INSTANCES": "${FACE_INFERENCE_INSTANCES:-1}",
            "LOCAL_TRACKING_ENDPOINT": (
                f"http://local-tracking:8092/v1/cameras/{camera}"
                "/perception/sam_live/sessions"
            ),
        }
        ports = (
            [
                {
                    "target": 8080,
                    "published": str(port),
                    "host_ip": "${HARNESS_BIND_ADDRESS:-127.0.0.1}",
                    "protocol": "tcp",
                }
            ]
            if index == 1
            else []
        )
        lines.append(f"  {service}:")
        if index > 1:
            lines.append(
                "    extends: "
                + json.dumps(
                    {
                        "file": str(root / "compose.local-gpu.yaml"),
                        "service": "harness",
                    }
                )
            )
        lines.append("    environment: " + json.dumps(env))
        lines.append("    ports: !override " + json.dumps(ports))
        # Compose replaces a volume by target, retaining config and model mounts.
        lines.append("    volumes: " + json.dumps([f"{camera}-data:/data"]))
        lines.append(
            "    networks: !override "
            + json.dumps(
                {
                    "default": {"aliases": [service]},
                }
            )
        )
    lines.extend(
        [
            "  local-tracking:",
            "    environment: " + json.dumps({"SAM31_LIVE_CAMERAS": str(streams)}),
            "    healthcheck: "
            + json.dumps(
                {
                    "test": [
                        "CMD",
                        "python",
                        "-c",
                        "import json,urllib.request; "
                        "r=json.load(urllib.request.urlopen("
                        "'http://127.0.0.1:8092/health/cameras/camera-01/"
                        "perception/sam_live', timeout=2)); "
                        "assert r['enabled'] and r['ready'] and r['gpu_ready']",
                    ],
                }
            ),
            "volumes:",
        ]
    )
    lines.extend(f"  camera-{index:02d}-data: {{}}" for index in range(1, streams + 1))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan; no Docker calls or writes",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        help="Simulate this many GPUs, allowed only with --dry-run",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Existing Compose .env file (default: repository .env)",
    )
    parser.add_argument(
        "--with-paddle",
        action="store_true",
        help="Also start the optional Paddle GPU worker",
    )
    parser.add_argument(
        "--no-build", action="store_true", help="Use prebuilt local images"
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=1,
        help="Actual camera count, any positive integer (default: 1)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        help="Unified console port (default: HARNESS_PORT, otherwise 8080)",
    )
    args = parser.parse_args(argv)
    try:
        if args.streams < 1:
            raise LaunchError("--streams must be a positive integer")
        if args.gpu_count is not None:
            if not args.dry_run or not 1 <= args.gpu_count <= 64:
                raise LaunchError("--gpu-count requires --dry-run and a value in 1..64")
            devices = [
                {"index": index, "name": "simulated NVIDIA GPU", "memory_mib": None}
                for index in range(args.gpu_count)
            ]
        else:
            devices = detect_nvidia_devices()
        env_file = args.env_file.resolve() if args.env_file else ROOT / ".env"
        if args.env_file and not env_file.is_file():
            raise LaunchError(f"Environment file does not exist: {env_file}")
        settings = read_local_settings(env_file)
        settings.update(
            {key: os.environ[key] for key in SETTINGS if os.environ.get(key)}
        )
        environment = plan_environment(devices, settings)
        try:
            port = int(
                args.base_port
                if args.base_port is not None
                else environment.get("HARNESS_PORT", "8080")
            )
        except ValueError as error:
            raise LaunchError("HARNESS_PORT must be an integer TCP port") from error
        if not 1 <= port <= 65535:
            raise LaunchError("Unified console port must be within 1..65535")
        environment.update(
            HARNESS_PORT=str(port),
            HARNESS_FLEET_CAPACITY=str(args.streams),
            HARNESS_FLEET_CAMERA_INDEX="1",
            HARNESS_FLEET_PRIMARY_URL="http://harness:8080",
        )
        host = environment.get("HARNESS_BIND_ADDRESS", "127.0.0.1")
        host = {"0.0.0.0": "127.0.0.1", "::": "::1", "[::]": "::1"}.get(host, host)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        console_url = f"http://{host}:{port}"
        fleet_file = ROOT / ".goodvision" / "local-gpu-compose.yaml"
        overlay = fleet_overlay(args.streams, port=port) if args.streams > 1 else None
        command = compose_command(
            root=ROOT,
            env_file=env_file if env_file.is_file() else None,
            with_paddle=args.with_paddle,
            build=not args.no_build,
            streams=args.streams,
            fleet_file=fleet_file if overlay is not None else None,
        )
        # Print only the local plan, never inherited API keys or full Compose config.
        print(
            json.dumps(
                {
                    "devices": devices,
                    "environment": environment,
                    "command": command,
                    "streams": args.streams,
                    "provisioned_lanes": args.streams,
                    "console_url": console_url,
                    "fleet_overlay": overlay,
                },
                indent=2,
            )
        )
        if args.dry_run:
            return 0
        validate_weights(environment, devices)
        if overlay is not None:
            fleet_file.parent.mkdir(parents=True, exist_ok=True)
            fleet_file.write_text(overlay, encoding="utf-8")
        process_environment = dict(os.environ)
        process_environment.update(environment)
        try:
            result = subprocess.run(
                command, cwd=ROOT, env=process_environment, check=False
            )
        except OSError as error:
            raise LaunchError(
                "Docker Compose could not start; check Docker and "
                "NVIDIA Container Toolkit"
            ) from error
        if result.returncode:
            print(
                "Local GPU startup did not become healthy. Check local-inference "
                "and local-tracking logs. A CUDA out-of-memory error requires a "
                "smaller model/context or more free VRAM; GPU allocation is never "
                "expanded onto the other card automatically.",
                file=sys.stderr,
            )
        return result.returncode
    except (LaunchError, OSError) as error:
        print(f"Local GPU setup: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
