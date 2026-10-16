"""Archive native streaming callbacks before adapting them to answer evidence."""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
import time

from .evidence import EvidenceCapture
from .privacy import reject_private
from .recording import publish_once


class StreamingRecorder:
    """Keep the delivered source plan, including frames replaced by native queues.

    Unlike offline capture, windows are selected by the native scheduler at run
    time. Its admission statistics are retained separately from observation
    completeness. Missing person frames never disappear from the denominator.
    """

    def __init__(self, directory, *, video_id, stream_session_id, source_id,
                 source, provenance, profile):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "records").mkdir()
        self.clock_domain = f"{stream_session_id}:pid-{os.getpid()}:monotonic"
        self.manifest = dict(video_id=video_id, stream_session_id=stream_session_id,
                             source_id=source_id, source=source, provenance=provenance)
        reject_private(self.manifest, observation=True)
        self.profile = profile
        self._lock = RLock()
        self._sequence = 0
        self._previous = None
        self._poisoned = False
        self._closed = False
        self.frames = {}
        self._references = []
        publish_once(self.root / "manifest.json", {
            "schema_version": "native-stream-recording-v1", **self.manifest,
            "clock_domain": self.clock_domain, "face_enabled": False,
            "question_conditioned": False,
            "window_plan_origin": "native_runtime_admission_not_offline_preregistration"})

    def record(self, kind, payload):
        with self._lock:
            if self._poisoned or self._closed:
                raise RuntimeError("stream recording is closed or failed")
            try:
                reject_private(payload, observation=True)
                self._sequence += 1
                path = self.root / "records" / f"{self._sequence:08d}-{kind}.json"
                row = {"sequence": self._sequence, "kind": kind,
                       "recorded_monotonic": time.monotonic(),
                       "clock_domain": self.clock_domain,
                       "previous_record_sha256": self._previous, "payload": payload}
                self._previous = publish_once(path, row)
                self._references.append((kind, path, self._sequence))
                return str(path)
            except BaseException:
                self._poisoned = True
                raise

    def source_frame(self, jpeg, timestamp, sequence):
        with self._lock:
            if self.frames and (sequence <= max(self.frames)
                                or timestamp <= self.frames[max(self.frames)]["t"]):
                raise ValueError("native source frame order changed")
            row = {"frame": sequence, "t": timestamp,
                   "jpeg_sha256": hashlib.sha256(jpeg).hexdigest()}
            self.record("source", row)
            self.frames[sequence] = row

    def tracking_frame(self, **payload):
        return self.record("tracking", payload)

    def independent_report(self, report):
        return self.record("independent", report)

    def rejected_report(self, report, *, reason):
        self.record("rejected", {"report": report, "reason": reason})
        raise ValueError("native independent report identity mismatch")

    def scene_snapshot(self, snapshot, *, association_links):
        self.record("scene", {"snapshot": snapshot, "association_links": association_links})

    def timing(self, payload):
        self.record("timing", payload)

    def request_failure(self, *, stage, reason, detail=None):
        self.record("failure", {"stage": stage, "reason": reason, "detail": detail})

    def finish(self, *, source_eof, tracking_drained, events_drained, fusion_drained,
               runtime_statistics):
        with self._lock:
            self.record("runtime-final", runtime_statistics)
            windows, report_windows = [], {}
            archived_lanes = Counter()
            for kind, path, cursor in self._references:
                if kind != "independent":
                    continue
                report = json.loads(path.read_bytes())["payload"]
                stamps = report["evidence"]
                if not stamps:
                    raise ValueError("native report lacks source evidence")
                for item in stamps:
                    source = self.frames.get(item["frame_sequence"])
                    if source is None or source["t"] != item["timestamp_seconds"]:
                        raise ValueError("native report refers to an undelivered frame")
                key = (report["request_id"], report.get("vlm_lane", "combined"))
                if key in report_windows:
                    raise ValueError("duplicate terminal native report")
                window_id = f"native-{cursor:08d}"
                report_windows[key] = window_id
                archived_lanes[key[1]] += 1
                windows.append({"window_id": window_id,
                    "start": min(item["timestamp_seconds"] for item in stamps),
                    "end": max(item["timestamp_seconds"] for item in stamps),
                    "event_labels": list(self.profile.labels),
                    "report_lanes": [key[1]], "request_ids": {key[1]: key[0]}})
            if self.manifest["provenance"]["vlm_enabled"]:
                worker = runtime_statistics["vlm"]
                accounting = worker.get("request_accounting", {})
                if worker.get("state") != "drained" or set(accounting) != {"events", "grounding"}:
                    raise ValueError("native VLM drain accounting missing")
                for lane, report_lane in (("events", "information"), ("grounding", "grounding")):
                    counts = accounting[lane]
                    count = archived_lanes[report_lane]
                    if count == 0 or any(counts.get(key) != count for key in ("submitted", "finished", "published")):
                        raise ValueError("native VLM dispatch, completion and archive counts differ")
                if (worker.get("late_information_reports_discarded", 0)
                        or worker.get("published_reports") != sum(archived_lanes.values())):
                    raise ValueError("native VLM publication coverage incomplete")
            capture = EvidenceCapture(**copy.deepcopy(self.manifest),
                                      frame_plan=list(self.frames.values()), window_plan=windows)
            for kind, path, cursor in self._references:
                row = json.loads(path.read_bytes())["payload"]
                reference = str(path)
                if kind == "tracking":
                    capture.tracking_frame(**{key: value for key, value in row.items()
                                              if key != "raw_response"}, raw_reference=reference)
                elif kind == "independent":
                    key = (row["request_id"], row.get("vlm_lane", "combined"))
                    capture.independent_report(row, window_id=report_windows[key],
                                               raw_reference=reference, publication_cursor=cursor)
                elif kind == "scene":
                    capture.scene_snapshot(row["snapshot"], association_links=row["association_links"],
                                           raw_reference=reference)
                elif kind == "failure":
                    capture.request_failure(stage=row["stage"], reason=row["reason"], reference=reference)
            if self.manifest["provenance"]["vlm_enabled"] and not any(
                    row["report_lanes"] == ["information"] for row in windows):
                capture.request_failure(stage="events", reason="no_native_information_window",
                                        reference=str(self.root / "manifest.json"))
            evidence = capture.seal(source_eof=source_eof, tracking_drained=tracking_drained,
                                    events_drained=events_drained, fusion_drained=fusion_drained)
            digest = publish_once(self.root / "capture.json", evidence)
            publish_once(self.root / "completion.json", {
                "capture_file_sha256": digest, "evidence_sha256": evidence["evidence_sha256"],
                "record_count": self._sequence, "last_record_sha256": self._previous,
                "runtime_statistics": runtime_statistics,
                "complete": evidence["observation_manifest"]["complete"]})
            self._closed = True
            return evidence
