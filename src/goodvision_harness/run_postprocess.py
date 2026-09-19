"""Run-owned archive controls; a retired run cannot mutate a new live run."""

from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class RunPostprocess:
    stream_session_id: str
    lock: Any = field(default_factory=RLock)
    pending_merges: dict = field(default_factory=dict)
    pending_scene: dict | None = None
    state: str = "capturing"
    stage: str = "capture"
    error: str | None = None
    session: Any = None
    scene: Any = None

    def status(self) -> dict:
        with self.lock:
            return {
                "stream_session_id": self.stream_session_id,
                "state": self.state, "stage": self.stage, "error": self.error,
            }
