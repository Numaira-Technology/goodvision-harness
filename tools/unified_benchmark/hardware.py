"""Read-only host identity; run on the GPU host, not the API controller."""
import json
import os
import platform
from pathlib import Path
import subprocess


def snapshot():
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version,power.limit",
        "--format=csv,noheader,nounits"], text=True, timeout=15).strip().splitlines()
    if not gpu:
        raise ValueError("no GPU identity available")
    host_id = Path("/etc/machine-id")
    return {"host": platform.node(), "machine_id": host_id.read_text().strip() if host_id.exists() else None,
            "system": platform.platform(), "cpu": platform.processor(), "cpu_count": os.cpu_count(),
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "gpus": sorted(gpu)}


if __name__ == "__main__":
    print(json.dumps(snapshot(), sort_keys=True))
