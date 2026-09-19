"""Run only this node's engine with a continuous host-memory reserve guard.

Run as root on a cgroup-v2 Jetson host, outside the container. Never downloads
or builds images. Stopping the guard also stops its exact engine cgroup.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

RESERVE_MIB = 1024
ENGINE = "goodvision-contour-engine-1"


def validate_container(info):
    labels = info["Config"].get("Labels") or {}
    host = info["HostConfig"]
    if (
        labels.get("com.docker.compose.project") != "goodvision-contour"
        or labels.get("com.docker.compose.service") != "engine"
        or host["RestartPolicy"]["Name"] != "no"
        or host["Memory"] != 4500 * 1024 * 1024
        or host["MemorySwap"] != host["Memory"]
        or not re.fullmatch(r"[0-9a-f]{64}", info["Id"])
    ):
        raise RuntimeError("Engine identity or required memory limits differ")


def cgroup_path(group: str, container_id: str) -> Path:
    root = Path("/sys/fs/cgroup")
    path = root / group.lstrip("/")
    if (
        ".." in path.parts
        or not re.fullmatch(r"[0-9a-f]{64}", container_id)
        or not any(
            part in {container_id, f"docker-{container_id}.scope"}
            for part in path.parts
        )
    ):
        raise RuntimeError("Refusing a cgroup outside the identified container")
    return path


def available_mib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("Host memory availability is unavailable")


def process_identity(pid):
    # /proc/stat's process name may contain spaces or parentheses.
    try:
        return (
            (Path("/proc") / str(pid) / "stat")
            .read_text()
            .rsplit(") ", 1)[1]
            .split()[19]
        )
    except FileNotFoundError:
        return None


def inspect():
    return json.loads(
        subprocess.check_output(["docker", "inspect", ENGINE], timeout=10)
    )[0]


def main():
    if os.geteuid() != 0:
        raise RuntimeError("Run this host-memory guard with sudo")
    if available_mib() < RESERVE_MIB:
        raise RuntimeError("Insufficient host memory reserve to start the engine")
    node = Path(__file__).resolve().parent
    subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            "goodvision-contour",
            "-f",
            "compose.yaml",
            "-f",
            "compose.jetson.yaml",
            "create",
            "--no-build",
            "--pull",
            "never",
            "engine",
        ],
        cwd=node,
        check=True,
        timeout=60,
    )
    info = inspect()
    validate_container(info)
    container_id = info["Id"]
    subprocess.run(["docker", "start", container_id], check=True, timeout=30)
    kill_file = None
    pid = 0
    identity = None
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        info = inspect()
        if info["Id"] != container_id or not info["State"]["Running"]:
            raise RuntimeError("Engine changed or exited during startup")
        pid = info["State"]["Pid"]
        identity = process_identity(pid)
        if pid <= 0 or identity is None:
            raise RuntimeError("Engine process exited before guard attachment")
        group = next(
            line.split(":", 2)[2]
            for line in (Path("/proc") / str(pid) / "cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        kill_file = cgroup_path(group, container_id) / "cgroup.kill"
        if not kill_file.is_file():
            raise RuntimeError("This host requires cgroup-v2 cgroup.kill support")
        Path("/proc/self/oom_score_adj").write_text("-900")
        os.nice(-5)
        print(
            f"Guard attached to {container_id}; reserve={RESERVE_MIB} MiB", flush=True
        )
        last_log = 0.0
        while not stopping and process_identity(pid) == identity:
            available = available_mib()
            if available < RESERVE_MIB:
                print(f"Stopping engine: host reserve {available:.0f} MiB", flush=True)
                break
            if time.monotonic() - last_log >= 60:
                print(f"Host available memory {available:.0f} MiB", flush=True)
                last_log = time.monotonic()
            time.sleep(0.2)
    finally:
        if (
            kill_file is not None
            and kill_file.is_file()
            and identity
            and process_identity(pid) == identity
        ):
            # No Docker-daemon round trip is needed under memory pressure.
            kill_file.write_text("1")
        else:
            # Includes startup failures before the cgroup could be identified.
            subprocess.run(
                ["docker", "stop", "--timeout", "5", container_id], timeout=15
            )
        print("Guard finished; engine stopped", flush=True)


if __name__ == "__main__":
    main()
