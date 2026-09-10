#!/usr/bin/env python3
"""Start the complete GoodVision delivery with verified CUDA or CPU execution.

The same latest main image is used in both modes. CPU mode stops the optional
rich worker while preserving its data. No image is pulled or replaced by an
older build. This launcher requires only Python's standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MAIN_IMAGE = "goodvision-harness:2026.09.10-full.1"
RICH_IMAGE = "goodvision-rich-label-worker:2026.09.10-full.1"
PROBE_MARKER = "GOODVISION_CUDA_PROBE="
PCI_DEVICE_DIRECTORY = Path("/sys/bus/pci/devices")
CUDA_PROBE = """import ctypes, json
available = False
reason = 'no_cuda_driver'
try:
    cuda = ctypes.CDLL('libcuda.so.1')
except OSError as exc:
    detail = str(exc)
    if not (detail.startswith('libcuda.so.1:')
            and 'No such file or directory' in detail):
        reason = 'cuda_driver_load_failed'
else:
    result = cuda.cuInit(0)
    if result == 100:
        reason = 'no_cuda_device'
    elif result != 0:
        reason = 'cuda_initialization_failed'
    else:
        count = ctypes.c_int()
        result = cuda.cuDeviceGetCount(ctypes.byref(count))
        if result != 0:
            reason = 'cuda_device_query_failed'
        elif count.value == 0:
            reason = 'no_cuda_device'
        else:
            available = True
            reason = 'ready'
print('GOODVISION_CUDA_PROBE=' + json.dumps({'available': available, 'reason': reason}))
raise SystemExit(0 if available else 3)
"""


class DeliveryError(RuntimeError):
    """A failed preflight or launch; never interpreted as proof of no GPU."""


@dataclass(frozen=True)
class LaunchPlan:
    mode: str
    reason: str
    commands: tuple[tuple[str, ...], ...]


def default_directory() -> Path:
    script = Path(__file__).resolve()
    return script.parent.parent if script.parent.name == "tools" else script.parent


def find_file(directory: Path, explicit: str | None, names: tuple[str, ...]) -> Path:
    candidates = [Path(explicit)] if explicit else [Path(name) for name in names]
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else directory / candidate
        if path.is_file():
            return path.resolve()
    raise DeliveryError("Delivery file not found: " + (explicit or " / ".join(names)))


def docker_check(
    arguments: list[str], directory: Path, purpose: str
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", *arguments],
            cwd=directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DeliveryError("Docker is not installed or is not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise DeliveryError(f"{purpose} timed out; no services were started.") from exc
    except OSError as exc:
        raise DeliveryError(f"Could not run Docker for {purpose}.") from exc


def require_daemon(directory: Path) -> None:
    result = docker_check(
        ["info", "--format", "{{.ServerVersion}}"], directory, "Docker daemon check"
    )
    if result.returncode:
        raise DeliveryError(
            "Docker daemon is unavailable. Start Docker and retry; "
            "GPU availability is unknown."
        )


def preflight(directory: Path) -> None:
    require_daemon(directory)
    if docker_check(
        ["compose", "version"], directory, "Docker Compose check"
    ).returncode:
        raise DeliveryError(
            "Docker Compose v2 is unavailable. "
            "Install or enable the Docker Compose plugin."
        )
    images = docker_check(
        ["image", "inspect", MAIN_IMAGE, RICH_IMAGE, "--format", "{{.Id}}"],
        directory,
        "Delivery image check",
    )
    if images.returncode:
        require_daemon(directory)
        raise DeliveryError(
            f"Both current delivery images must be loaded locally: {MAIN_IMAGE} "
            f"and {RICH_IMAGE}. "
            "Load the supplied image archives before starting."
        )


def docker_uses_local_host(directory: Path) -> bool:
    context_name = os.environ.get("DOCKER_CONTEXT")
    endpoint = None if context_name else os.environ.get("DOCKER_HOST")
    if not endpoint:
        arguments = ["context", "inspect"]
        if context_name:
            arguments.append(context_name)
        arguments.extend(["--format", "{{json .Endpoints.docker.Host}}"])
        result = docker_check(
            arguments,
            directory,
            "Docker host check",
        )
        if result.returncode:
            return False
        try:
            endpoint = json.loads(result.stdout)
        except (ValueError, TypeError):
            return False
    return isinstance(endpoint, str) and endpoint.startswith(("unix://", "npipe://"))


def host_has_nvidia_gpu() -> bool | None:
    """Read local hardware; None means unknown, never proof of a CPU-only host."""
    if sys.platform.startswith("linux"):
        try:
            vendors = [
                (device / "vendor").read_text().strip().lower()
                for device in PCI_DEVICE_DIRECTORY.iterdir()
                if (device / "vendor").is_file()
            ]
        except OSError:
            return None
        return "0x10de" in vendors if vendors else None
    if sys.platform == "win32":
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell is None:
            return None
        script = (
            "$ErrorActionPreference='Stop'; "
            "$cards=@(Get-CimInstance Win32_VideoController -ErrorAction Stop); "
            "if($cards.Count -eq 0){throw 'Hardware inventory unavailable'}; "
            "$found=@($cards | Where-Object { "
            "$_.PNPDeviceID -match 'VEN_10DE' -or "
            "$_.Name -match 'NVIDIA' }).Count -gt 0; "
            "ConvertTo-Json -Compress -InputObject $found"
        )
        try:
            result = subprocess.run(
                [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                check=False,
            )
            present = json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
            return None
        return present if result.returncode == 0 and isinstance(present, bool) else None
    return None


def probe_cuda(directory: Path) -> bool:
    result = docker_check(
        [
            "run",
            "--rm",
            "--pull=never",
            "--gpus",
            "all",
            "--network",
            "none",
            "--entrypoint",
            "python",
            MAIN_IMAGE,
            "-c",
            CUDA_PROBE,
        ],
        directory,
        "CUDA availability probe",
    )
    for line in result.stdout.splitlines():
        if not line.startswith(PROBE_MARKER):
            continue
        try:
            payload = json.loads(line[len(PROBE_MARKER) :])
        except (ValueError, TypeError) as exc:
            raise DeliveryError("CUDA probe returned an invalid response.") from exc
        if not isinstance(payload, dict):
            raise DeliveryError("CUDA probe returned an invalid response.")
        if result.returncode == 0 and payload.get("available") is True:
            return True
        if result.returncode == 3 and payload.get("available") is False:
            if payload.get("reason") in {"no_cuda_driver", "no_cuda_device"}:
                return False
            raise DeliveryError(
                "CUDA is present but initialization failed. "
                "Repair the GPU runtime; CPU mode was not selected."
            )
        raise DeliveryError("CUDA probe returned an inconsistent response.")

    # Docker itself may report that no GPU devices are present.
    # Recheck the daemon first so connection failures cannot become CPU starts.
    require_daemon(directory)
    error = result.stderr.lower()
    if result.returncode == 125 and (
        "wsl environment detected but no adapters were found" in error
        or ("no devices were found" in error and "nvidia-container-cli" in error)
    ):
        return False
    if (
        result.returncode == 125
        and "could not select device driver" in error
        and "capabilities: [[gpu]]" in error
        and docker_uses_local_host(directory)
        and host_has_nvidia_gpu() is False
    ):
        return False
    raise DeliveryError(
        "Docker could not verify CUDA availability. Check the GPU container runtime; "
        "CPU mode was not selected."
    )


def make_plan(
    directory: Path,
    *,
    mode: str = "auto",
    compose_file: str | None = None,
    gpu_compose_file: str | None = None,
    dry_run: bool = False,
) -> LaunchPlan:
    if mode not in {"auto", "cpu", "gpu"}:
        raise DeliveryError("Mode must be auto, cpu, or gpu.")
    base = find_file(
        directory, compose_file, ("compose.full-gpu.delivery.yaml", "compose.yaml")
    )
    find_file(directory, None, ("config.delivery.yaml",))
    selected = mode
    reason = "Explicit CPU execution; rich annotation is disabled."
    # Explicit dry runs are useful on a packaging machine without Docker.
    # Automatic dry runs still need the real availability probe to select a mode.
    if not dry_run or mode == "auto":
        preflight(directory)
        if mode != "cpu":
            available = probe_cuda(directory)
            if mode == "gpu" and not available:
                raise DeliveryError(
                    "GPU mode was requested but no CUDA GPU is available to Docker."
                )
            selected = "gpu" if available else "cpu"
            reason = (
                "CUDA GPU verified."
                if available
                else "No CUDA GPU is available; rich annotation is disabled."
            )
    elif mode == "gpu":
        reason = "Explicit GPU dry run; CUDA will be verified on startup."

    common = (
        "docker",
        "compose",
        "--project-directory",
        str(directory),
        "-f",
        str(base),
    )
    if selected == "gpu":
        overlay = find_file(
            directory,
            gpu_compose_file,
            ("compose.full-gpu.accelerated.yaml", "compose.gpu.yaml"),
        )
        commands = (
            (
                *common,
                "-f",
                str(overlay),
                "--profile",
                "gpu",
                "up",
                "-d",
                "harness",
                "rich-label-worker",
            ),
        )
    else:
        commands = (
            (*common, "--profile", "gpu", "stop", "rich-label-worker"),
            (*common, "up", "-d", "harness"),
        )
    return LaunchPlan(selected, reason, commands)


def command_text(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--directory", "--project-directory", type=Path, default=default_directory()
    )
    parser.add_argument("--compose", "--compose-file", dest="compose_file")
    parser.add_argument("--gpu-compose", "--gpu-compose-file", dest="gpu_compose_file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print launch commands without starting services; "
            "auto mode still probes Docker."
        ),
    )
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    try:
        plan = make_plan(
            directory,
            mode=args.mode,
            compose_file=args.compose_file,
            gpu_compose_file=args.gpu_compose_file,
            dry_run=args.dry_run,
        )
        print(f"Selected mode: {plan.mode}. {plan.reason}", flush=True)
        for command in plan.commands:
            print(command_text(command), flush=True)
            if not args.dry_run:
                completed = subprocess.run(command, cwd=directory, check=False)
                if completed.returncode:
                    raise DeliveryError(
                        "Docker Compose failed. No alternative image "
                        "or execution mode was attempted."
                    )
        return 0
    except (DeliveryError, OSError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
