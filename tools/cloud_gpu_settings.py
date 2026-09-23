"""Validated rental parameters shared by the console controller and CLI."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CloudGPUSettings:
    gpu_name: str = "RTX_4090"
    gpu_count: int = 1
    streams: int = 1
    disk_gb: int = 50

    def __post_init__(self):
        if not isinstance(self.gpu_name, str):
            raise ValueError("GPU 型号必须是字符串")
        name = self.gpu_name.strip().replace(" ", "_")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
            raise ValueError("GPU 型号仅支持字母、数字、空格、下划线、点和连字符")
        object.__setattr__(self, "gpu_name", name)
        for field in ("gpu_count", "streams", "disk_gb"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} 必须是正整数")
        # Each worker needs a distinct TCP port, not a product GPU quota.
        if self.gpu_count > 65535 - 8100 + 1:
            raise ValueError("GPU 数量超出独立进程端口范围")

    @classmethod
    def from_environment(cls):
        return cls(
            gpu_name=os.environ.get("SAM_GPU_NAME", "RTX_4090"),
            gpu_count=int(os.environ.get("SAM_GPU_COUNT", "1")),
            streams=int(os.environ.get("SAM31_LIVE_CAMERAS", "1")),
            disk_gb=int(os.environ.get("SAM_DISK_GB", "50")),
        )

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict) or set(payload) != set(asdict(cls())):
            raise ValueError("租用配置必须包含 gpu_name、gpu_count、streams、disk_gb")
        return cls(**payload)

    def public(self):
        return asdict(self)

    def arguments(self):
        return [
            part
            for name, value in self.public().items()
            for part in ("--" + name.replace("_", "-"), str(value))
        ]

    def query(self):
        # A100 is a family shortcut; an exact marketplace name stays exact.
        name = (
            "gpu_name in [A100_SXM4,A100_PCIE,A100_SXM4_80GB,A100_PCIE_80GB]"
            if self.gpu_name.upper() == "A100"
            else f"gpu_name={self.gpu_name}"
        )
        return (
            f"{name} num_gpus={self.gpu_count} rentable=true direct_port_count>=1 "
            f"cuda_max_good>=12.6 disk_space>={self.disk_gb} geolocation!=CN"
        )
