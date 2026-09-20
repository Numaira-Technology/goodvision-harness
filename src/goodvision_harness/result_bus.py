"""Thread-safe live result and frame fan-out."""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict, deque
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
    """Arrival-ordered delivery with source-ordered global and feature snapshots."""

    def __init__(self, history_size: int = 100) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self._condition = threading.Condition()
        self._history: deque[ResultRecord] = deque(maxlen=history_size)
        self._latest: ResultRecord | None = None
        self._latest_by_feature: OrderedDict[str, ResultRecord] = OrderedDict()
        # Public features must survive history eviction by a faster branch.
        # Bound auxiliary/custom labels as well as the arrival log.
        self._feature_limit = max(16, history_size)
        self._cursor = 0
        self.epoch = str(uuid4())

    def publish(
        self, payload: Mapping[str, Any], *, expected_epoch: str | None = None
    ) -> ResultRecord | None:
        safe = public_safe_value(copy.deepcopy(dict(payload)))
        with self._condition:
            if expected_epoch is not None and expected_epoch != self.epoch:
                return None
            self._cursor += 1
            record = ResultRecord(cursor=self._cursor, payload=safe)
            self._history.append(record)
            if self._latest is None or _frame_order(record) >= _frame_order(
                self._latest
            ):
                self._latest = record
            judgments = record.payload.get("judgments", ())
            for judgment in judgments if isinstance(judgments, (list, tuple)) else ():
                if not isinstance(judgment, Mapping):
                    continue
                label = judgment.get("event_label")
                if not isinstance(label, str) or not label.strip():
                    continue
                feature = _feature_label(label)
                previous = self._latest_by_feature.get(feature)
                if previous is None or _frame_order(record) >= _frame_order(previous):
                    self._latest_by_feature[feature] = record
                    self._latest_by_feature.move_to_end(feature)
            while len(self._latest_by_feature) > self._feature_limit:
                self._latest_by_feature.popitem(last=False)
            self._condition.notify_all()
        return _copy_result(record)

    def reset(self) -> None:
        """Clear one session's history while keeping cursors monotonic."""

        with self._condition:
            self._history.clear()
            self._latest = None
            self._latest_by_feature.clear()
            self.epoch = str(uuid4())
            self._condition.notify_all()

    def latest(self, *, feature: str | None = None) -> ResultRecord | None:
        """Return an unchanged observation, never a merge of different frames."""

        with self._condition:
            record = (
                self._latest
                if feature is None
                else self._latest_by_feature.get(_feature_label(feature))
            )
        return _copy_result(record) if record is not None else None

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
                records = tuple(item for item in self._history if item.cursor > cursor)
                if records or timeout_seconds == 0:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ()
                self._condition.wait(remaining)
        return tuple(_copy_result(item) for item in records)


def _feature_label(label: str) -> str:
    return "stranger" if label == "unknown_person" else label


def _frame_order(record: ResultRecord) -> tuple[float, int]:
    frame = record.payload.get("frame")
    if not isinstance(frame, Mapping):
        # Legacy unframed publications retain their arrival-order behavior.
        return (0.0, 0)
    return (
        float(frame.get("timestamp_seconds", 0.0)),
        int(frame.get("source_frame_index", 0)),
    )


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
            "face_aligned",
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
