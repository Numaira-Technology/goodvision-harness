"""Apply the manually selected camera view, fixed for one session."""

from __future__ import annotations
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any
from .video_to_stream import LiveFrame, crop_frame

CROP_MODES = frozenset({"none", "left_half", "right_half"})


class FrameCropSelector:
    """No visual inference or startup buffering; old auto means full-frame."""

    def __init__(self, mode: str) -> None:
        if mode not in CROP_MODES and mode != "auto":
            raise ValueError("frame crop mode must be none, left_half, or right_half")
        self.mode = mode
        self.effective = "none" if mode == "auto" else mode
        self.reason = "legacy_auto_disabled" if mode == "auto" else "manual"
        self.probes = 0

    def status(self) -> dict[str, Any]:
        return {
            "requested": self.mode, "effective": self.effective,
            "reason": self.reason, "probed_frames": 0,
        }

    def frames(
        self, frames: Iterator[LiveFrame], *, stopped: Callable[[], bool],
    ) -> Iterator[LiveFrame]:
        for frame in frames:
            if stopped():
                return
            yield replace(frame, bgr=crop_frame(frame.bgr, mode=self.effective))
