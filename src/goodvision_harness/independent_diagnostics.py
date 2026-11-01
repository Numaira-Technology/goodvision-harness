"""Best-effort model-boundary diagnostics across the independent VLM process.

Inference never waits for diagnostic IPC. Each of at most eight active calls
retains at most 2 MiB of JPEGs; the separate IPC queue holds two such packets.
The parent retains eight image sets / 16 MiB and 256 metadata records per run.
These observations do not measure downstream tracking/event acceptance.
"""

from __future__ import annotations

import copy
import time
from collections import deque
from queue import Full

from .diagnostics import SceneDiagnostics

DIAGNOSTIC_QUEUE_CAPACITY = 2
REQUEST_IMAGE_BYTES = 2 * 1024 * 1024


class IndependentRequestDiagnostics(SceneDiagnostics):
    def __init__(self, job, lane, sample, generation):
        super().__init__(
            history=1, image_history=1, max_image_bytes=REQUEST_IMAGE_BYTES
        )
        self.job = job
        self.lane = lane
        self.sample = sample
        self.generation = generation
        self.claimed(self.job)

    def publish(self, outgoing, *, verdicts=(), error=None):
        # The adapter normally recorded the response already. This fallback
        # covers failures before inspect() and test/custom inspector adapters.
        self.responded(self.job, verdicts=verdicts, error=error)
        snapshot = self.snapshot(self.job.request.event_labels)
        row = snapshot["records"][0]
        original_id = row["request_id"]
        row.update(
            request_id=original_id + ":" + self.lane,
            window_id=original_id,
            output_mode=self.lane,
            evidence_outcome="not_observed",
            accepted_verdicts=None,
            confirmed_event_labels=None,
            claim_to_response_seconds=snapshot["latency_seconds"]["claim_to_response"][
                "p50"
            ],
            source_arrival_to_response_seconds=max(
                0, time.monotonic() - self.sample.received_monotonic
            ),
        )
        images = {
            frame["frame_sequence"]: self.jpeg(original_id, frame["frame_sequence"])
            for frame in row["frames"]
            if row["images_available"]
        }
        packet = {
            "stream_session_id": self.job.stream_session_id,
            "generation": self.generation,
            "record": row,
            "images": images,
        }
        try:
            outgoing.put_nowait(packet)
            return True
        except Full:
            return False


class IndependentDiagnostics(SceneDiagnostics):
    """Run-scoped parent cache, accessed only by diagnostics reader and HTTP."""

    def __init__(self, session_id, generation, **kwargs):
        super().__init__(**kwargs)
        self.session_id, self.generation = session_id, generation
        self._latencies = {
            "claim_to_response": self._latencies["claim_to_response"],
            "source_arrival_to_response": deque(maxlen=self._history),
        }

    def receive(self, packet):
        if (
            packet["stream_session_id"] != self.session_id
            or packet["generation"] != self.generation
        ):
            return
        row = copy.deepcopy(packet["record"])
        key = row["request_id"]
        with self._lock:
            if key in self._records:
                return
            self._records[key] = row
            while len(self._records) > self._history:
                old, _ = self._records.popitem(last=False)
                self._drop_images(old)
            images = dict(packet["images"])
            size = sum(map(len, images.values()))
            if images and size <= self._max_image_bytes:
                self._images[key] = images
                self._image_bytes += size
            while (
                len(self._images) > self._image_history
                or self._image_bytes > self._max_image_bytes
            ):
                self._drop_images(next(iter(self._images)))
            self._counts["received_requests"] += 1
            self._counts["response_" + row["response_outcome"]] += 1
            if row.get("request_attempted"):
                self._counts["request_attempts"] += 1
            if row.get("failure_category"):
                self._failures[row["failure_category"]] += 1
            for name, values in self._latencies.items():
                values.append(row[name + "_seconds"])

    def snapshot(self, labels):
        snapshot = super().snapshot(())
        snapshot.update(
            scope="independent_model_response",
            stream_session_id=self.session_id,
            generation=self.generation,
            delivery="best_effort_completed_requests",
            gap_definition="Not measured: tracking acceptance and event "
            "confirmation are outside this model-boundary collector.",
        )
        return snapshot
