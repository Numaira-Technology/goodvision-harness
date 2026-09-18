"""Small, explicit deployment configuration; never downloads model assets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
PRODUCT = "GoodVision Contour"
VERSION = "0.1.0"
MODEL_ID = "goodvision-contour-1"


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    engine_url: str = ""
    checkpoint_path: Path = ROOT / "models" / "contour.pt"
    request_timeout: float = 70.0
    max_request_bytes: int = 32 * 1024 * 1024
    max_response_bytes: int = 16 * 1024 * 1024
    camera_capacity: int = 1

    def __post_init__(self):
        if type(self.camera_capacity) is not int or self.camera_capacity < 1:
            raise ValueError("Contour camera capacity must be a positive integer")
        if self.api_key and (
            len(self.api_key) < 32
            or len(self.api_key) > 256
            or not self.api_key.isascii()
            or not self.api_key.isprintable()
            or any(char.isspace() for char in self.api_key)
        ):
            raise ValueError("Contour access key must be 32-256 ASCII characters")
        if self.engine_url:
            url = urlsplit(self.engine_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
                or url.path not in {"", "/"}
            ):
                raise ValueError("Contour engine URL must be a plain HTTP(S) origin")
        if not 0 < self.request_timeout <= 180:
            raise ValueError("Contour request timeout must be in (0, 180]")
        if not 1 <= self.max_request_bytes <= 32 * 1024 * 1024:
            raise ValueError("Invalid Contour request bound")
        if not 1 <= self.max_response_bytes <= 32 * 1024 * 1024:
            raise ValueError("Invalid Contour response bound")

    @classmethod
    def from_env(cls):
        key = os.getenv("CONTOUR_API_KEY", "")
        key_file = Path(os.getenv("CONTOUR_API_KEY_FILE", ROOT / "secrets/node-key"))
        if not key and key_file.is_file():
            key = key_file.read_text(encoding="utf-8").strip()
        return cls(
            api_key=key,
            engine_url=os.getenv("CONTOUR_ENGINE_URL", "").rstrip("/"),
            checkpoint_path=Path(
                os.getenv("CONTOUR_CHECKPOINT_PATH", ROOT / "models/contour.pt")
            ),
            request_timeout=float(os.getenv("CONTOUR_REQUEST_TIMEOUT", "70")),
            camera_capacity=int(os.getenv("CONTOUR_CAPACITY", "1")),
        )
