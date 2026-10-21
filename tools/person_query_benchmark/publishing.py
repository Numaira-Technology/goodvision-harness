"""Publish actual person snapshots and read previously published query answers."""

from threading import RLock
import time

from goodvision_harness.result_bus import ResultBus


class PersonQueryPublisher:
    """Keep source order and real publication time separate."""

    def __init__(self, video_id, recorder):
        self.video_id, self.recorder = video_id, recorder
        self.results = ResultBus(128)
        self._lock = RLock()
        self._counts = {}
        self._published = []

    def publish(self, snapshot, *, source_frame_index, source_size):
        count, reason = 0, None
        for person in snapshot.objects:
            if person.label.casefold() not in {"person", "pedestrian"}:
                continue
            if person.observation_state == "predicted":
                continue
            if (person.observation_state != "observed"
                    or person.last_observed_frame_sequence != snapshot.frame_sequence
                    or person.last_observed_timestamp_seconds != snapshot.timestamp_seconds):
                reason = "person_observation_metadata_unknown"
            else:
                count += 1
        if reason is not None:
            count = None
        with self._lock:
            if count is not None:
                self._counts[source_frame_index] = count
            prefix = [(frame, value) for frame, value in self._counts.items()
                      if frame <= source_frame_index]
            peak_frame, peak = (min(prefix, key=lambda item: (-item[1], item[0]))
                                if prefix else (None, None))
            report = {
                "video_id": self.video_id,
                "frame": {"source_frame_index": source_frame_index,
                          "timestamp_seconds": snapshot.timestamp_seconds},
                "source_size": dict(source_size), "state": "completed",
                "snapshot": snapshot.as_dict(),
                "answers": {"current_count": count,
                            "sparse_observed_peak_estimate": peak},
                "peak_source_frame_index": peak_frame, "count_unknown_reason": reason,
            }
            reference = self.recorder.independent_report(report)
            record = self.results.publish(report, measure_publication=True)
            if record is None or record.readable_monotonic_bounds is None:
                raise RuntimeError("person report publication has no readable boundary")
            self.recorder.publication(record, report_reference=reference)
            self._published.append((source_frame_index, record.cursor,
                                    record.readable_monotonic_bounds, reference))
            return reference

    def read_query(self, query, ingress_monotonic):
        lower = time.monotonic()
        with self._lock:
            possible = [item for item in self._published
                        if item[0] <= query["query_frame_index"]]
            readable = [item for item in possible if item[2][1] <= ingress_monotonic]
            selected = max(readable, key=lambda item: (item[0], item[1]), default=None)
            overlap = any(lo <= ingress_monotonic < hi for _, _, (lo, hi), _ in possible)
            newer_uncertain = any(
                lo <= ingress_monotonic < hi
                and (selected is None or (frame, cursor) > selected[:2])
                for frame, cursor, (lo, hi), _ in possible)
            reason = (None if selected is not None else
                      "publication_boundary_overlaps_query" if overlap else
                      "no_report_readable_at_query_ingress")
        return {**query, "ingress_monotonic": ingress_monotonic,
                "read_monotonic_lower": lower, "read_monotonic_upper": time.monotonic(),
                "selected_report_reference": selected[3] if selected else None,
                "newer_publication_boundary_uncertain": newer_uncertain,
                "unknown_reason": reason}
