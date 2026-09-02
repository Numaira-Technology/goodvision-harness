"""Thread-safe live result and frame fan-out."""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .privacy import public_safe_value


@dataclass(frozen=True, slots=True)
class ResultRecord:
    cursor: int
    payload: dict[str, Any]


class ResultBus:
    """Bounded result history with condition-based multi-client fan-out."""

    def __init__(self, history_size: int = 100) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self._condition = threading.Condition()
        self._history: deque[ResultRecord] = deque(maxlen=history_size)
        self._cursor = 0
        self.epoch = str(uuid4())

    def publish(self, payload: Mapping[str, Any]) -> ResultRecord:
        with self._condition:
            self._cursor += 1
            record = ResultRecord(
                cursor=self._cursor,
                payload=public_safe_value(copy.deepcopy(dict(payload))),
            )
            self._history.append(record)
            self._condition.notify_all()
            return _copy_result(record)

    def reset(self) -> None:
        """Clear one session's history while keeping cursors monotonic."""

        with self._condition:
            self._history.clear()
            self.epoch = str(uuid4())
            self._condition.notify_all()

    def latest(self) -> ResultRecord | None:
        with self._condition:
            if not self._history:
                return None
            return _copy_result(self._history[-1])

    def records_after(
        self,
        cursor: int,
        *,
        timeout_seconds: float = 0.0,
    ) -> tuple[ResultRecord, ...]:
        if cursor < 0:
            raise ValueError("cursor must be non-negative")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                records = tuple(
                    _copy_result(item) for item in self._history if item.cursor > cursor
                )
                if records or timeout_seconds == 0:
                    return records
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ()
                self._condition.wait(remaining)


@dataclass(frozen=True, slots=True)
class FramePacket:
    jpeg: bytes
    media_timestamp: float
    frame_sequence: int
    observed_at: str
    analysis_message_id: str | None = None
    next_analysis_message_id: str | None = None
    annotation_mode: str | None = None
    stream_delay_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.jpeg:
            raise ValueError("jpeg must not be empty")
        if self.media_timestamp < 0:
            raise ValueError("media_timestamp must be non-negative")
        if self.frame_sequence < 0:
            raise ValueError("frame_sequence must be non-negative")
        if not self.observed_at.strip():
            raise ValueError("observed_at must not be empty")
        if (
            self.analysis_message_id is not None
            and not self.analysis_message_id.strip()
        ):
            raise ValueError("analysis_message_id must not be empty")
        if (
            self.next_analysis_message_id is not None
            and not self.next_analysis_message_id.strip()
        ):
            raise ValueError("next_analysis_message_id must not be empty")
        if self.annotation_mode is not None and self.annotation_mode not in {
            "exact",
            "interpolated",
            "held",
            "none",
        }:
            raise ValueError("annotation_mode is invalid")
        if self.stream_delay_seconds is not None and self.stream_delay_seconds < 0:
            raise ValueError("stream_delay_seconds must be non-negative")


class FrameBus:
    """Latest-frame broadcaster for snapshot and MJPEG consumers."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._version = 0
        self._latest: FramePacket | None = None

    def publish(self, packet: FramePacket) -> int:
        with self._condition:
            self._version += 1
            self._latest = packet
            self._condition.notify_all()
            return self._version

    def reset(self) -> int:
        """Clear the latest frame and wake clients waiting on the old session."""

        with self._condition:
            self._version += 1
            self._latest = None
            self._condition.notify_all()
            return self._version

    def latest(self) -> tuple[int, FramePacket] | None:
        with self._condition:
            if self._latest is None:
                return None
            return self._version, self._latest

    def wait_after(
        self,
        version: int,
        *,
        timeout_seconds: float = 0.0,
    ) -> tuple[int, FramePacket] | None:
        if version < 0:
            raise ValueError("version must be non-negative")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._version <= version or self._latest is None:
                if timeout_seconds == 0:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._version, self._latest


def _copy_result(record: ResultRecord) -> ResultRecord:
    return ResultRecord(record.cursor, copy.deepcopy(record.payload))
