"""Client-side facts the hardware run log does not already record.

Already recorded by ``tools/vast_run_log.py`` and not repeated here: the Vast
offer, CPU model, host core count, host RAM, CUDA version, cgroup limits, and
per-GPU PCIe, clocks, power, temperature, memory, and utilization (probe and
2-second samples).

This module measures, from the machine that sends requests:

- TCP connect time to the SSH host (client RTT)
- TCP connect time to the HTTP endpoint
- whether that endpoint is the SSH local forward or a direct port

CPU governor, turbo, NUMA, the GPU's NUMA node, and other GPUs' compute
processes are collected on the rented host by the existing probe. See
``INVESTIGATION_NOTE``.
"""

from __future__ import annotations

import socket
import time

INVESTIGATION_NOTE = (
    "cpu_governor, cpu_turbo, numa_node_count, gpu_numa_nodes, and "
    "gpu_compute_apps are on the probe EXTRA line. gpu_compute_apps is the "
    "process list; per-GPU utilization is already in probe.gpus and "
    "gpu-samples.jsonl. client_access is measured on the benchmark client."
)


def measure_tcp_rtt(host: str, port: int, timeout: float = 2.0):
    if not host or not isinstance(port, int) or isinstance(port, bool):
        return "unknown"
    if not 1 <= port <= 65535:
        return "unknown"
    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError:
        return "unknown"
    return round(time.monotonic() - started, 6)


def client_access(
    *,
    endpoint_host: str,
    endpoint_port: int,
    ssh_host: str | None = None,
    ssh_port: int | None = None,
    mode: str = "direct",
) -> dict:
    """Describe how the client reaches vLLM, plus connect RTTs.

    ``mode`` is ``ssh-local-forward`` when the HTTP port is an SSH ``-L``
    tunnel and ``direct`` when the client dials the server port itself.
    """
    if mode not in {"ssh-local-forward", "direct"}:
        mode = "unknown"
    ssh_rtt = "unknown"
    if ssh_host and isinstance(ssh_port, int) and not isinstance(ssh_port, bool):
        ssh_rtt = measure_tcp_rtt(ssh_host, ssh_port)
    return {
        "access": mode,
        "endpoint_host": endpoint_host or "unknown",
        "endpoint_port": endpoint_port if isinstance(endpoint_port, int) else "unknown",
        "endpoint_tcp_rtt_seconds": measure_tcp_rtt(endpoint_host, endpoint_port)
        if isinstance(endpoint_port, int) and not isinstance(endpoint_port, bool)
        else "unknown",
        "ssh_host": ssh_host or "unknown",
        "ssh_port": ssh_port if isinstance(ssh_port, int) else "unknown",
        "ssh_tcp_rtt_seconds": ssh_rtt,
        "note": (
            "ssh-local-forward means vLLM is reached through an SSH -L tunnel "
            "to 127.0.0.1 on the instance. ssh_tcp_rtt_seconds is the client "
            "TCP handshake to the SSH port. endpoint_tcp_rtt_seconds is the "
            "handshake to the HTTP port the benchmark dials, which is loopback "
            "when the tunnel is local."
        ),
    }
