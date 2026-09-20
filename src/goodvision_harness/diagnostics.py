"""Bounded, per-run observations of the actual temporal inspection boundary.

Response validity, evidence acceptance and event confirmation are separate facts.
Only exact request JPEGs are retained, in memory; provider text and credentials
never enter diagnostics. Missing/unknown verdicts never count as negative.
"""

from __future__ import annotations

import base64
import copy
import io
import itertools
import math
import time
from collections import Counter, OrderedDict, deque
from threading import RLock

from PIL import Image


def note(job, method, *args, **kwargs):
    """Observability must not change inference or evidence acceptance."""
    diagnostics = getattr(job, "diagnostics", None)
    if diagnostics is not None:
        try:
            getattr(diagnostics, method)(job, *args, **kwargs)
        except Exception:
            pass


def failure_category(failure) -> str:
    code, status = failure.get("code"), failure.get("http_status")
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "rate_limited"
    if isinstance(status, int) and status >= 500:
        return "server_error"
    if code in {"timeout", "connection", "budget_exhausted", "cancelled"}:
        return code
    if code in {"invalid_response", "empty_response", "parse"}:
        return "parse"
    if isinstance(status, int) and status >= 400:
        return "http_error"
    return "input" if code == "input" else "internal"


def _summary(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "count": len(values),
        "p50": values[len(values) // 2],
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "max": values[-1],
    }


def verdict_summary(labels, verdicts):
    """Summarize returned labels, not completeness of all people in the frame."""
    returned = {v.event_label for v in verdicts}
    missing = sorted(set(labels) - returned)
    unknown = sorted({v.event_label for v in verdicts if v.detected is None})
    positives = sorted({v.event_label for v in verdicts if v.detected is True})
    return {
        "result": (
            "partial" if missing or unknown else "positive" if positives else "negative"
        ),
        "detected_labels": positives,
        "missing_event_labels": missing,
        "unknown_event_labels": unknown,
    }


class SceneDiagnostics:
    def __init__(
        self,
        *,
        history=256,
        image_history=8,
        max_image_bytes=16 * 1024 * 1024,
        clock=time.monotonic,
    ):
        if any(
            type(v) is not int or v < 1
            for v in (
                history,
                image_history,
                max_image_bytes,
            )
        ):
            raise ValueError("diagnostic limits must be positive integers")
        self._clock = clock
        self._lock = RLock()
        self._history = history
        self._image_history = image_history
        self._max_image_bytes = max_image_bytes
        self._records = OrderedDict()
        self._images = OrderedDict()
        self._image_bytes = 0
        self._counts = Counter()
        self._failures = Counter()
        self._first_source = self._latest_source = None
        self._captures = OrderedDict()
        self._accepted_times = {}
        self._accepted_totals = Counter()
        self._latencies = {
            name: deque(maxlen=history)
            for name in (
                "source_arrival_to_completion",
                "claim_to_response",
            )
        }

    def observe_source(self, sequence, timestamp):
        with self._lock:
            if self._first_source is None:
                self._first_source = timestamp
            self._latest_source = max(timestamp, self._latest_source or 0)
            self._captures.setdefault(sequence, self._clock())
            while len(self._captures) > self._history * 4:
                self._captures.popitem(last=False)

    def _record(self, job):
        key = job.request.request_id
        if key not in self._records:
            self._records[key] = {
                "request_id": key,
                "stream_session_id": job.stream_session_id,
                "source_start_seconds": job.request.start_seconds,
                "source_end_seconds": job.request.end_seconds,
                "event_labels": list(job.request.event_labels),
                "event_sensitivities": dict(job.event_sensitivities),
                "response_outcome": "pending",
                "evidence_outcome": "pending",
                "result": "pending",
                "frames": [],
                "detected_labels": [],
                "missing_event_labels": [],
                "unknown_event_labels": [],
                "accepted_verdicts": 0,
                "confirmed_event_labels": [],
                "request_attempted": False,
            }
            while len(self._records) > self._history:
                old, _ = self._records.popitem(last=False)
                self._drop_images(old)
        return self._records[key]

    def claimed(self, job):
        with self._lock:
            row = self._record(job)
            row["_claimed_at"] = self._clock()
            row["_source_arrived_at"] = self._captures.get(
                job.frames[-1].frame_sequence
            )
            self._counts["claimed"] += 1

    def prepared(self, job, messages):
        """Read the JPEG bytes from the final multimodal payload, without resizing."""
        frames, images = [], {}
        parts = [p for p in messages[1]["content"] if p["type"] == "image_url"]
        if len(parts) != len(job.frames):
            raise ValueError("diagnostic payload does not match temporal sources")
        for source, part in zip(job.frames, parts, strict=True):
            jpeg = base64.b64decode(
                part["image_url"]["url"].split(",", 1)[1], validate=True
            )
            with Image.open(io.BytesIO(jpeg)) as image:
                width, height = image.size
            frames.append(
                {
                    "frame_sequence": source.frame_sequence,
                    "timestamp_seconds": source.timestamp_seconds,
                    "width": width,
                    "height": height,
                    "bytes": len(jpeg),
                }
            )
            images[source.frame_sequence] = jpeg
        with self._lock:
            row = self._record(job)
            row["frames"] = frames
            key = job.request.request_id
            self._drop_images(key)
            size = sum(map(len, images.values()))
            if size <= self._max_image_bytes:
                self._images[key] = images
                self._image_bytes += size
            while (
                len(self._images) > self._image_history
                or self._image_bytes > self._max_image_bytes
            ):
                self._drop_images(next(iter(self._images)))

    def attempted(self, job):
        with self._lock:
            self._record(job)["request_attempted"] = True
            self._counts["request_attempts"] += 1

    def responded(self, job, *, verdicts=(), error=None, transport=None):
        with self._lock:
            row = self._record(job)
            # A late response can update the model fact, never runtime acceptance.
            if row["response_outcome"] != "pending":
                return
            now = self._clock()
            started = row.get("_claimed_at")
            if started is not None:
                self._latencies["claim_to_response"].append(max(0, now - started))
            if transport:
                for name in (
                    "http_status",
                    "request_body_completed",
                    "request_bytes",
                    "encoded_image_bytes",
                ):
                    if name in transport:
                        row[name] = transport[name]
            if error is not None:
                failure = {
                    "code": getattr(
                        error,
                        "code",
                        "input" if isinstance(error, ValueError) else "internal",
                    ),
                    "http_status": getattr(error, "status_code", None),
                }
                category = failure_category(failure)
                row.update(
                    response_outcome="failed",
                    result="unavailable",
                    failure_category=category,
                )
                self._failures[category] += 1
                self._counts["response_failed"] += 1
            else:
                row.update(
                    response_outcome="validated",
                    **verdict_summary(job.request.event_labels, verdicts),
                )
                self._counts["response_validated"] += 1

    def completed(self, job, *, success, accepted=(), confirmed=(), expired=False):
        with self._lock:
            row = self._record(job)
            if row["evidence_outcome"] != "pending":
                return
            row["evidence_outcome"] = (
                "expired"
                if expired
                else "failed"
                if not success
                else "accepted"
                if accepted
                else "no_accepted_evidence"
            )
            row["accepted_verdicts"] = len(accepted)
            row["confirmed_event_labels"] = sorted(set(confirmed))
            self._counts[row["evidence_outcome"]] += 1
            if accepted:
                captured = row.get("_source_arrived_at")
                if captured is not None:
                    self._latencies["source_arrival_to_completion"].append(
                        max(0, self._clock() - captured)
                    )
                # Accepted nulls are lifecycle evidence, not analyzed negatives.
                for label, timestamp, detected in accepted:
                    if detected is not None:
                        self._accepted_times.setdefault(
                            label, deque(maxlen=self._history)
                        ).append(timestamp)
                        self._accepted_totals[label] += 1

    def _drop_images(self, key):
        old = self._images.pop(key, {})
        self._image_bytes -= sum(map(len, old.values()))

    def jpeg(self, request_id, sequence):
        with self._lock:
            return self._images.get(request_id, {}).get(sequence)

    def snapshot(self, labels):
        with self._lock:
            gaps = {}
            for label in labels:
                times = sorted(set(self._accepted_times.get(label, ())))
                truncated = self._accepted_totals[label] > self._history
                points = times[:]
                if not truncated and self._first_source is not None:
                    points.insert(0, self._first_source)
                if self._latest_source is not None:
                    points.append(self._latest_source)
                gaps[label] = {
                    "current_seconds": (
                        max(0, points[-1] - (times[-1] if times else points[0]))
                        if points
                        else None
                    ),
                    "longest_seconds": max(
                        (max(0, b - a) for a, b in itertools.pairwise(points)),
                        default=None,
                    ),
                    "scope": "recent" if truncated else "session",
                    "accepted_definite_verdicts": self._accepted_totals[label],
                }
            records = []
            for key, row in reversed(self._records.items()):
                item = {k: v for k, v in row.items() if not k.startswith("_")}
                item["images_available"] = key in self._images
                records.append(item)
            return copy.deepcopy(
                {
                    "counts": dict(self._counts),
                    "failures": dict(self._failures),
                    "unanalyzed_gaps": gaps,
                    "gap_definition": "between accepted non-null label observations; "
                    "not proof every person was inspected",
                    "latency_seconds": {
                        k: _summary(v) for k, v in self._latencies.items()
                    },
                    "image_bytes": self._image_bytes,
                    "records": records,
                }
            )
