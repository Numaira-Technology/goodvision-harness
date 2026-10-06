"""Start the CPU application, a standalone SAM server, or a standalone VLM server.

Python 3.10+ standard library only. Modules can run on different machines.
--dry-run prints a credential-free plan, without writing files or calling Docker.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

if __package__:
    from . import start_local_gpu as local
else:
    import start_local_gpu as local

ROOT = Path(__file__).resolve().parents[1]
QUEUE_SETTINGS = {
    "SAM31_QUEUE_POLICY",
    "SAM31_REALTIME_APPEND_FRAMES",
    "SAM31_REALTIME_PENDING_FRAMES",
    "SAM31_REALTIME_MAX_AGE_SECONDS",
    "SAM31_RESULT_DEADLINE_SECONDS",
}
SETTINGS = (
    local.SETTINGS
    | {
        "INFERENCE_EXECUTION_MODE",
        "TRACKING_EXECUTION_MODE",
        "INFERENCE_SERVER_URL",
        "INFERENCE_SERVER_MODEL",
        "INFERENCE_SERVER_PROTOCOL",
        "TRACKING_SERVER_URL",
        "TRACKING_SERVER_CAPACITY",
        "TRACKING_SERVER_PROTOCOL",
        "SAM_SERVER_GPU_ID",
        "SAM_SERVER_BIND_ADDRESS",
        "SAM_SERVER_PORT",
        "VLM_SERVER_BIND_ADDRESS",
        "VLM_SERVER_PORT",
        "LOCAL_INFERENCE_GPU_ID",
        "PERCEPTION_METHOD",
    }
    | QUEUE_SETTINGS
)
COMPOSE_FILES = {
    "main": "compose.main-server.yaml",
    "sam": "compose.sam-server.yaml",
    "vlm": "compose.vlm-server.yaml",
}
LaunchError = local.LaunchError


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise LaunchError(f"{name} must be a positive integer")
    try:
        number = int(str(value))
    except (ValueError, TypeError) as error:
        raise LaunchError(f"{name} must be a positive integer") from error
    if number < 1:
        raise LaunchError(f"{name} must be a positive integer")
    return number


def server_origin(value: str, name: str) -> str:
    # Match service_locations._service_origin without importing app dependencies
    # into this standard-library-only deployment entry point.
    try:
        if not isinstance(value, str) or len(value) > 2048:
            raise ValueError("invalid origin")
        rendered = value.strip()
        if not rendered:
            return ""
        if any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in rendered
        ) or any(char in rendered for char in "\\?#%"):
            raise ValueError("invalid origin")
        parsed = urlsplit(rendered)
        valid_port = parsed.port is None or 1 <= parsed.port <= 65535
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.netloc.endswith(":")
            or not valid_port
        ):
            raise ValueError("invalid origin")
        hostname = parsed.hostname.encode("idna").decode("ascii").rstrip(".")
        if ":" not in hostname and (
            not hostname
            or len(hostname) > 253
            or any(
                not re.fullmatch(
                    r"[a-zA-Z0-9_](?:[a-zA-Z0-9_-]{0,61}[a-zA-Z0-9_])?", label
                )
                for label in hostname.split(".")
            )
        ):
            raise ValueError("invalid origin")
    except (ValueError, UnicodeError) as error:
        raise LaunchError(
            f"{name} must be an HTTP(S) origin, for example http://gpu-host:8092; "
            "do not include credentials, API paths, queries or fragments"
        ) from error
    return urlunsplit((parsed.scheme, parsed.netloc.lower(), "", "", ""))


def listener(settings: Mapping[str, str], prefix: str, args) -> dict[str, str]:
    default_port = {"HARNESS": 8080, "SAM_SERVER": 8092, "VLM_SERVER": 8000}[prefix]
    port = positive_integer(
        args.port
        if args.port is not None
        else settings.get(f"{prefix}_PORT", default_port),
        "TCP port",
    )
    if port > 65535:
        raise LaunchError("TCP port must be within 1..65535")
    default_bind = "127.0.0.1" if prefix == "HARNESS" else "0.0.0.0"
    bind = args.bind or settings.get(f"{prefix}_BIND_ADDRESS", default_bind)
    try:
        address = ipaddress.ip_address(bind.strip("[]"))
    except ValueError as error:
        raise LaunchError("Bind address must be an IPv4 or IPv6 address") from error
    bind = f"[{address}]" if address.version == 6 else str(address)
    return {f"{prefix}_PORT": str(port), f"{prefix}_BIND_ADDRESS": bind}


def main_environment(settings: Mapping[str, str], args) -> dict[str, str]:
    """Plan only CPU application containers; never inspect this host's GPUs."""
    streams = positive_integer(args.streams, "--streams")
    tracking_url = server_origin(
        args.tracking_server
        if args.tracking_server is not None
        else settings.get("TRACKING_SERVER_URL", ""),
        "SAM server",
    )
    inference_url = server_origin(
        args.inference_server
        if args.inference_server is not None
        else settings.get("INFERENCE_SERVER_URL", ""),
        "VLM server",
    )
    tracking_mode = (
        "server"
        if args.tracking_server
        else settings.get(
            "TRACKING_EXECUTION_MODE", "server" if tracking_url else "legacy"
        )
    )
    inference_mode = (
        "server"
        if args.inference_server
        else settings.get(
            "INFERENCE_EXECUTION_MODE", "server" if inference_url else "cloud"
        )
    )
    if tracking_mode not in {"legacy", "server"}:
        raise LaunchError(
            "The CPU main module supports TRACKING_EXECUTION_MODE=legacy|server"
        )
    if inference_mode not in {"cloud", "server"}:
        raise LaunchError(
            "The CPU main module supports INFERENCE_EXECUTION_MODE=cloud|server"
        )
    if tracking_mode == "server" and not tracking_url:
        raise LaunchError(
            "Server tracking requires --tracking-server or TRACKING_SERVER_URL"
        )
    if inference_mode == "server" and not inference_url:
        raise LaunchError(
            "Server inference requires --inference-server or INFERENCE_SERVER_URL"
        )
    capacity = positive_integer(
        args.tracking_capacity
        if args.tracking_capacity is not None
        else settings.get("TRACKING_SERVER_CAPACITY", streams),
        "SAM server camera capacity",
    )
    model = args.inference_model or settings.get(
        "INFERENCE_SERVER_MODEL", "local-vision"
    )
    if (
        not model.strip()
        or len(model) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in model)
    ):
        raise LaunchError(
            "VLM served model name must be single-line and 1..512 characters"
        )
    protocol = settings.get("TRACKING_SERVER_PROTOCOL", "legacy")
    if protocol not in {"legacy", "contour"}:
        raise LaunchError("TRACKING_SERVER_PROTOCOL must be legacy or contour")
    inference_protocol = args.inference_protocol or settings.get(
        "INFERENCE_SERVER_PROTOCOL", "compatible"
    )
    if inference_protocol not in {"compatible", "goodvision"}:
        raise LaunchError("INFERENCE_SERVER_PROTOCOL must be compatible or goodvision")
    environment = {
        "HARNESS_FLEET_CAPACITY": str(streams),
        "HARNESS_FLEET_CAMERA_INDEX": "1",
        "HARNESS_FLEET_PRIMARY_URL": "http://harness:8080",
        "LOCAL_GPU_AVAILABLE": "false",
        "LOCAL_GPU_COUNT": "0",
        "LOCAL_INFERENCE_ENABLED": "false",
        "LOCAL_TRACKING_ENABLED": "false",
        "INFERENCE_EXECUTION_MODE": inference_mode,
        "TRACKING_EXECUTION_MODE": tracking_mode,
        "INFERENCE_SERVER_URL": inference_url,
        "INFERENCE_SERVER_MODEL": model.strip(),
        "INFERENCE_SERVER_PROTOCOL": inference_protocol,
        "TRACKING_SERVER_URL": tracking_url,
        "TRACKING_SERVER_CAPACITY": str(capacity),
        "TRACKING_SERVER_PROTOCOL": protocol,
    }
    environment.update(listener(settings, "HARNESS", args))
    environment.update(queue_environment(settings, args))
    if args.tracking_fps is not None:
        fps = args.tracking_fps
        if not math.isfinite(fps) or not 0.5 <= fps <= 30:
            raise LaunchError("Tracking FPS must be within 0.5..30")
        environment["LOCAL_TRACKING_FPS"] = str(fps)
    return environment


def queue_environment(settings: Mapping[str, str], args) -> dict[str, str]:
    """Expose explicit queue settings without changing YAML or sampling defaults."""
    environment = {
        key: settings[key]
        for key in QUEUE_SETTINGS | {"PERCEPTION_METHOD"}
        if settings.get(key)
    }
    if args.sam_queue_policy is not None:
        environment["SAM31_QUEUE_POLICY"] = args.sam_queue_policy
    policy = environment.get("SAM31_QUEUE_POLICY")
    if policy is not None and policy not in {
        "durable_ordered", "realtime_latest", "latest_only",
    }:
        raise LaunchError(
            "SAM31_QUEUE_POLICY must be latest_only|durable_ordered|realtime_latest"
        )
    if policy in {"realtime_latest", "latest_only"}:
        if environment.get("PERCEPTION_METHOD", "sam_live") != "sam_live":
            raise LaunchError(f"{policy} requires PERCEPTION_METHOD=sam_live")
        environment["PERCEPTION_METHOD"] = "sam_live"
    # Match the runtime's bounded settings; validate even when the policy is off,
    # because config loading also validates dormant experiment parameters.
    for key, minimum, maximum, integer in (
        ("SAM31_REALTIME_APPEND_FRAMES", 1, 4, True),
        ("SAM31_REALTIME_PENDING_FRAMES", 1, 8, True),
        ("SAM31_REALTIME_MAX_AGE_SECONDS", 0.01, 30, False),
        ("SAM31_RESULT_DEADLINE_SECONDS", 0.01, 30, False),
    ):
        if key not in environment:
            continue
        try:
            value = int(environment[key]) if integer else float(environment[key])
        except ValueError as error:
            raise LaunchError(f"{key} must be within {minimum}..{maximum}") from error
        if not minimum <= value <= maximum or not math.isfinite(value):
            raise LaunchError(f"{key} must be within {minimum}..{maximum}")
        if key == "SAM31_REALTIME_APPEND_FRAMES" and value not in {1, 2, 4}:
            raise LaunchError("SAM31_REALTIME_APPEND_FRAMES must be 1, 2, or 4")
    return environment


def application_fleet(streams: int, *, root: Path = ROOT) -> str:
    """CPU camera runtimes with isolated data and one published console port."""
    streams = positive_integer(streams, "--streams")
    lines = ["# Generated by tools/start_modular.py main; do not edit.", "services:"]
    for index in range(1, streams + 1):
        camera = f"camera-{index:02d}"
        service = "harness" if index == 1 else f"harness-{index:02d}"
        lines.append(f"  {service}:")
        if index > 1:
            lines.append(
                "    extends: "
                + json.dumps(
                    {
                        "file": str(root / COMPOSE_FILES["main"]),
                        "service": "harness",
                    }
                )
            )
        stream_fallback = "${HARNESS_STREAM_URL:-}" if index == 1 else ""
        environment = {
            "HARNESS_FLEET_CAPACITY": str(streams),
            "HARNESS_FLEET_CAMERA_INDEX": str(index),
            "HARNESS_FLEET_PRIMARY_URL": "http://harness:8080",
            "HARNESS_SOURCE_ID": camera,
            "HARNESS_STREAM_URL": (
                f"${{LOCAL_CAMERA_{index:02d}_STREAM_URL:-{stream_fallback}}}"
            ),
            "RTSP_URL": "${RTSP_URL:-}" if index == 1 else "",
            "HARNESS_AUTOSTART": f"${{LOCAL_CAMERA_{index:02d}_AUTOSTART:-false}}",
        }
        lines.append("    environment: " + json.dumps(environment))
        ports = (
            [
                {
                    "target": 8080,
                    "published": "${HARNESS_PORT:-8080}",
                    "host_ip": "${HARNESS_BIND_ADDRESS:-127.0.0.1}",
                    "protocol": "tcp",
                }
            ]
            if index == 1
            else []
        )
        lines.append("    ports: !override " + json.dumps(ports))
        lines.append("    volumes: " + json.dumps([f"{camera}-data:/data"]))
        lines.append(
            "    networks: !override "
            + json.dumps(
                {
                    "default": {"aliases": [service]},
                }
            )
        )
    lines.append("volumes:")
    lines.extend(f"  camera-{index:02d}-data: {{}}" for index in range(1, streams + 1))
    return "\n".join(lines) + "\n"


def worker_environment(
    module: str, settings: Mapping[str, str], args, devices
) -> dict[str, str]:
    gpu_key = "SAM_SERVER_GPU_ID" if module == "sam" else "LOCAL_INFERENCE_GPU_ID"
    try:
        gpu_id = int(
            args.gpu_id if args.gpu_id is not None else settings.get(gpu_key, "0")
        )
    except ValueError as error:
        raise LaunchError("GPU ID must be a nonnegative NVIDIA device index") from error
    selected = next(
        (device for device in devices if int(device["index"]) == gpu_id), None
    )
    if selected is None:
        raise LaunchError(f"NVIDIA GPU {gpu_id} is not present on this worker host")
    if module == "sam":
        checkpoint = Path(
            settings.get(
                "SAM31_CHECKPOINT_FILE",
                ".models/sam3.1/sam3.1_multiplex.pt",
            )
        ).expanduser()
        environment = {
            "SAM_SERVER_GPU_ID": str(gpu_id),
            "SAM31_LIVE_CAMERAS": str(positive_integer(args.streams, "--streams")),
            "SAM31_CHECKPOINT_FILE": str(
                (ROOT / checkpoint).resolve()
                if not checkpoint.is_absolute()
                else checkpoint.resolve()
            ),
        }
    else:
        # Dedicated VLM card; these are initial limits, not measured capacity.
        defaults = {
            "LOCAL_INFERENCE_GPU_MEMORY_UTILIZATION": "0.90",
            "LOCAL_INFERENCE_MAX_NUM_SEQS": "8",
            "LOCAL_INFERENCE_EXECUTION_MODE": "--no-enforce-eager",
        }
        defaults.update(
            {
                key: value
                for key, value in settings.items()
                if key.startswith("LOCAL_INFERENCE_")
            }
        )
        planned = local.plan_environment([selected], defaults, root=ROOT)
        environment = {
            key: value
            for key, value in planned.items()
            if key.startswith("LOCAL_INFERENCE_")
            and key not in {"LOCAL_INFERENCE_ENABLED", "LOCAL_INFERENCE_ENDPOINT"}
        }
    environment.update(listener(settings, f"{module.upper()}_SERVER", args))
    return environment


def validate_sam_weights(environment: Mapping[str, str]) -> None:
    checkpoint = Path(environment["SAM31_CHECKPOINT_FILE"])
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise LaunchError(f"Missing SAM3.1 checkpoint: {checkpoint}")
    with checkpoint.open("rb") as stream:
        if stream.read(64).startswith(b"version https://git-lfs.github.com/"):
            raise LaunchError(
                "SAM3.1 checkpoint is a Git LFS pointer; download the actual weights"
            )


def compose_command(
    module: str,
    *,
    env_file: Path | None,
    streams: int,
    fleet_file: Path | None,
    build: bool,
) -> list[str]:
    command = ["docker", "compose", "--project-directory", str(ROOT)]
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    command.extend(["-f", str(ROOT / COMPOSE_FILES[module])])
    if fleet_file is not None:
        command.extend(["-f", str(fleet_file)])
    command.extend(["up", "--detach", "--wait", "--wait-timeout", "900"])
    if build:
        command.append("--build")
    else:
        command.append("--no-build")
    command.extend(
        ["harness", *(f"harness-{index:02d}" for index in range(2, streams + 1))]
        if module == "main"
        else ["vlm-server", "vlm-processing"] if module == "vlm"
        else ["sam-server"]
    )
    return command


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    modules = result.add_subparsers(dest="module", required=True)
    for module in COMPOSE_FILES:
        command = modules.add_parser(module)
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--env-file", type=Path)
        command.add_argument("--no-build", action="store_true")
        command.add_argument("--bind", help="Host IP for the published service port")
        command.add_argument("--port", type=int)
        if module in {"main", "sam"}:
            command.add_argument("--streams", type=int, default=1)
        if module == "main":
            command.add_argument(
                "--sam-queue-policy",
                choices=("latest_only", "durable_ordered", "realtime_latest"),
                help="Queue override; latest_only keeps two replaceable candidates",
            )
            command.add_argument(
                "--tracking-fps",
                type=float,
                help="Initial tracking sample FPS (0.5..30); saved console values win",
            )
            command.add_argument(
                "--tracking-server", help="SAM origin, e.g. http://gpu-host:8092"
            )
            command.add_argument(
                "--tracking-capacity",
                type=int,
                help="Must match the SAM server's --streams",
            )
            command.add_argument(
                "--inference-server", help="VLM origin, e.g. http://vlm-host:8000"
            )
            command.add_argument(
                "--inference-model", help="Served model alias, default local-vision"
            )
            command.add_argument(
                "--inference-protocol", choices=("compatible", "goodvision"),
                help="compatible: direct API; goodvision: delegate VLM CPU processing",
            )
        else:
            command.add_argument(
                "--gpu-id", type=int, help="NVIDIA index on this host, default 0"
            )
            command.add_argument(
                "--gpu-count", type=int, help="Simulated GPU count, only with --dry-run"
            )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        env_file = args.env_file.resolve() if args.env_file else ROOT / ".env"
        if args.env_file and not env_file.is_file():
            raise LaunchError(f"Environment file does not exist: {env_file}")
        settings = local.read_local_settings(env_file, keys=SETTINGS)
        settings.update(
            {key: os.environ[key] for key in SETTINGS if os.environ.get(key)}
        )
        devices = []
        streams = getattr(args, "streams", 1)
        if args.module == "main":
            environment = main_environment(settings, args)
        else:
            if args.gpu_count is not None:
                if not args.dry_run or not 1 <= args.gpu_count <= 64:
                    raise LaunchError(
                        "--gpu-count requires --dry-run and a value in 1..64"
                    )
                devices = [
                    {"index": index, "name": "simulated NVIDIA GPU", "memory_mib": None}
                    for index in range(args.gpu_count)
                ]
            else:
                devices = local.detect_nvidia_devices()
            environment = worker_environment(args.module, settings, args, devices)
        fleet_file = (
            ROOT / ".goodvision" / "main-server-compose.yaml"
            if args.module == "main"
            else None
        )
        overlay = (
            application_fleet(streams, root=ROOT) if fleet_file is not None else None
        )
        command = compose_command(
            args.module,
            env_file=env_file if env_file.is_file() else None,
            streams=streams,
            fleet_file=fleet_file,
            build=not args.no_build,
        )
        checks = []
        if args.module == "vlm" and args.no_build:
            compose = command[:command.index("up")]
            checks.append([
                *compose, "run", "--rm", "--no-deps", "--pull", "never", "-T",
                "vlm-processing", "python", "-c",
                local.runtime_check("local-vlm-processing", root=ROOT),
            ])
        # Cloud and worker keys go directly from the existing environment to Compose.
        # They are never included in this plan or written to the generated overlay.
        print(
            json.dumps(
                {
                    "module": args.module,
                    "devices": devices,
                    "environment": environment,
                    "command": command,
                    "fleet_overlay": overlay,
                    "preflight_commands": checks,
                },
                indent=2,
            )
        )
        if args.dry_run:
            return 0
        if args.module == "sam":
            validate_sam_weights(environment)
        elif args.module == "vlm":
            local.validate_weights(environment, devices, require_tracking=False)
        if fleet_file is not None:
            fleet_file.parent.mkdir(parents=True, exist_ok=True)
            fleet_file.write_text(overlay, encoding="utf-8")
        process_environment = dict(os.environ)
        process_environment.update(environment)
        try:
            for check in checks:
                checked = subprocess.run(
                    check, cwd=ROOT, env=process_environment, check=False,
                    timeout=120,
                )
                if checked.returncode:
                    raise LaunchError(
                        "VLM CPU companion preflight failed; rebuild or import "
                        "the image matching this source."
                    )
            result = subprocess.run(
                command, cwd=ROOT, env=process_environment, check=False
            )
        except OSError as error:
            raise LaunchError(
                "Docker Compose could not start; check the Docker installation"
            ) from error
        if result.returncode:
            print(
                f"The {args.module} module did not become healthy. Check its logs; "
                "other modules are deployed independently and were not started "
                "by this command.",
                file=sys.stderr,
            )
        return result.returncode
    except (LaunchError, OSError) as error:
        print(f"Modular setup: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
