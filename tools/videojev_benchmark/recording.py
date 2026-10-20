"""Persist streamed inputs, native reports, and measured publication boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from threading import RLock
import time
from typing import Any, Mapping, TextIO

from goodvision_harness.result_bus import ResultRecord


def publish_once(path, value):
    """Publish one JSON file atomically without replacing an existing result."""
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".publishing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return str(path.resolve())


class EventStreamRecorder:
    """Write one new stream recording without retaining its frames in memory.

    Input and report rows are saved unchanged. Publication rows link each raw
    report to its ResultBus cursor and actual readable-time bounds. The caller
    owns source completion, original-media EOF, and runtime drain accounting.
    """

    def __init__(self, directory: str | Path, manifest: Mapping[str, Any]) -> None:
        self.root = Path(directory).resolve()
        self._lock = RLock()
        self._files: dict[str, TextIO] = {}
        self._counts = {name: 0 for name in ("inputs", "reports", "publication", "failures")}
        self._pending_inputs = 0
        self._last_input_flush = time.monotonic()
        self._closed = False
        self._finished = False
        self.failed = False
        self.root.mkdir(parents=True, exist_ok=False)
        try:
            publish_once(self.root / "manifest.json", manifest)
            for name in self._counts:
                self._files[name] = (self.root / f"{name}.jsonl").open(
                    "x", encoding="utf-8", buffering=65536, newline="\n"
                )
        except BaseException:
            self.failed = True
            for stream in self._files.values():
                try:
                    stream.close()
                except OSError:
                    pass
            raise

    def _writable(self) -> None:
        if self.failed:
            raise RuntimeError("stream recording failed")
        if self._closed or self._finished:
            raise RuntimeError("stream recording is closed or finished")

    def _append(self, name: str, row: Any, *, flush: bool) -> int:
        with self._lock:
            self._writable()
            try:
                raw = json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ) + "\n"
                stream = self._files[name]
                stream.write(raw)
                self._counts[name] += 1
                if name == "inputs":
                    self._pending_inputs += 1
                    now = time.monotonic()
                    flush = (
                        self._pending_inputs >= 128
                        or now - self._last_input_flush >= 1.0
                    )
                if flush:
                    stream.flush()
                    if name == "inputs":
                        self._pending_inputs = 0
                        self._last_input_flush = now
                return self._counts[name]
            except BaseException:
                self.failed = True
                raise

    def source_input(self, row: Mapping[str, Any]) -> None:
        """Buffer the original per-frame input row; never infer missing inputs."""
        self._append("inputs", row, flush=False)

    def independent_report(self, report: Mapping[str, Any]) -> str:
        """Save the original report before returning its source-local reference."""
        line = self._append("reports", report, flush=True)
        return f"reports.jsonl:{line}"

    def publication(self, record: ResultRecord, *, report_reference: str) -> None:
        """Save ResultBus timing; file-write time is not a readable boundary."""
        bounds = record.readable_monotonic_bounds
        lower, upper = bounds if bounds is not None else (None, None)
        self._append(
            "publication",
            {
                "lane": record.payload.get("vlm_lane", "combined"),
                "cursor": record.cursor,
                "request_id": record.payload.get("request_id"),
                "report_reference": report_reference,
                "state": record.payload.get("state"),
                "readable_monotonic_lower": lower,
                "readable_monotonic_upper": upper,
            },
            flush=True,
        )

    def rejected_report(self, report: Mapping[str, Any], *, reason: str) -> None:
        """Retain an identity mismatch and make it visible to the caller."""
        self._append("failures", {"reason": reason, "report": report}, flush=True)
        raise ValueError(reason)

    def failure(self, reason: str, **details: Any) -> None:
        """Record a source or window failure without discarding other rows."""
        self._append("failures", {"reason": reason, **details}, flush=True)

    def finish(self, completion: Mapping[str, Any]) -> None:
        """Durably save rows, then publish the caller's exact completion states."""
        with self._lock:
            self._writable()
            try:
                for stream in self._files.values():
                    stream.flush()
                    os.fsync(stream.fileno())
                publish_once(self.root / "completion.json", completion)
                self._finished = True
            except BaseException:
                self.failed = True
                raise

    def summary(self) -> dict[str, int]:
        """Return only the row counts accepted by this recorder."""
        with self._lock:
            return dict(self._counts)

    def close(self) -> None:
        """Close every handle without manufacturing a completion record."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            error = None
            for stream in self._files.values():
                try:
                    stream.close()
                except BaseException as exc:
                    self.failed = True
                    if error is None:
                        error = exc
            if error is not None:
                raise error
