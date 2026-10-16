"""Adapt published production evidence for unified QA without models, questions, or ground truth.

Callers must preserve raw reports and feed them after freezing the source plan.
The bounded ResultBus snapshot cannot reconstruct a complete video. Keep missing
windows and frames incomplete instead of filling them with negative verdicts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from .privacy import reject_private


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative source time required")
    return float(value)


class EvidenceCapture:
    """Collect evidence lanes for one video observation, excluding face processing.

Plans contain no questions, options, or ground truth. Keep independent raw events
separate from native SceneState confirmations; reuse its association and lifecycle logic.
"""

    def __init__(self, *, video_id, stream_session_id, source_id, source,
                 frame_plan, window_plan, provenance):
        self.video_id, self.session_id, self.source_id = video_id, stream_session_id, source_id
        if not all(isinstance(x, str) and x for x in (video_id, stream_session_id, source_id)):
            raise ValueError("scoped video/session/source required")
        reject_private(dict(source=source, frame_plan=frame_plan, window_plan=window_plan,
                            provenance=provenance), observation=True)
        if any(type(source.get(k)) is not int or source[k] <= 0 for k in ("width", "height")):
            raise ValueError("original media dimensions required")
        self.source = copy.deepcopy(source)
        if "duration_seconds" in source:
            _time(source["duration_seconds"])
        self.frame_plan = copy.deepcopy(list(frame_plan))
        previous = (-1, -1.0)
        for row in self.frame_plan:
            stamp = (row["frame"], _time(row["t"]))
            if type(stamp[0]) is not int or stamp[0] <= previous[0] or stamp[1] <= previous[1]:
                raise ValueError("unique increasing source frame plan required")
            previous = stamp
        self.window_plan = copy.deepcopy(list(window_plan))
        self._windows = {}
        for row in self.window_plan:
            key = row["window_id"]
            if not isinstance(key, str) or not key or key in self._windows:
                raise ValueError("unique window ID required")
            if _time(row["end"]) < _time(row["start"]):
                raise ValueError("window ends before it starts")
            if not row["event_labels"] or len(set(row["event_labels"])) != len(row["event_labels"]):
                raise ValueError("explicit fixed window vocabulary required")
            lanes = row.setdefault("report_lanes", ["combined"])
            if lanes not in (["combined"], ["information"], ["grounding"], ["information", "grounding"]):
                raise ValueError("explicit independent report lane plan required")
            self._windows[key] = row
        self.provenance = copy.deepcopy(provenance)
        self.frames, self.reports, self.events, self.episodes = {}, {}, {}, {}
        self.failures, self.links = [], {}
        self._closed = False
        self._completion = {}

    def _open(self):
        if self._closed:
            raise RuntimeError("evidence capture is already sealed")

    def _scope(self, row):
        if row.get("stream_session_id") != self.session_id or row.get("source_id") != self.source_id:
            raise ValueError("cross-session evidence rejected")

    def tracking_frame(self, *, frame, timestamp_seconds, subjects, raw_reference,
                       jpeg_sha256):
        """Record tracking responses, including empty frames and predicted-box states."""
        self._open()
        plan = next((x for x in self.frame_plan if x["frame"] == frame), None)
        if plan is None or _time(timestamp_seconds) != plan["t"]:
            raise ValueError("tracking response not in fixed source plan")
        if plan.get("jpeg_sha256") != jpeg_sha256:
            raise ValueError("tracking input hash mismatch")
        if frame in self.frames or not raw_reference:
            raise ValueError("duplicate frame or missing raw evidence reference")
        self.frames[frame] = {"frame": frame, "t": float(timestamp_seconds),
                              "subjects": copy.deepcopy(subjects),
                              "jpeg_sha256": jpeg_sha256, "raw_reference": raw_reference}

    def independent_report(self, report, *, window_id, raw_reference, publication_cursor):
        """Accept published records without filtering raw scene verdicts by accepted/grounded."""
        self._open()
        self._scope(report)
        lane = report.get("vlm_lane", "combined")
        if type(publication_cursor) is not int or publication_cursor < 1 or not raw_reference:
            raise ValueError("published report cursor and saved reference required")
        plan = self._windows[window_id]
        if lane not in plan["report_lanes"]:
            raise ValueError("report lane is not in fixed window plan")
        report_key = (window_id, lane)
        if report_key in self.reports:
            raise ValueError("one terminal report per planned window/lane; archive retries separately")
        source_frames = report.get("evidence", [])
        if not source_frames:
            raise ValueError("independent report has no source window")
        times = [_time(x["timestamp_seconds"]) for x in source_frames]
        if min(times) != plan["start"] or max(times) != plan["end"]:
            raise ValueError("independent report window mismatch")
        expected_request = plan.get("request_ids", {}).get(lane, plan.get("request_id"))
        if expected_request and expected_request != report.get("request_id"):
            raise ValueError("independent request identity mismatch")
        # Keep raw reports outside the bounded live cache to preserve observation coverage.
        self.reports[report_key] = {"report": copy.deepcopy(report),
                                  "reference": raw_reference,
                                  "publication_cursor": publication_cursor}
        if report.get("state") != "completed" or report.get("error_code"):
            self.failures.append({"stage": "fusion" if lane == "grounding" else "independent_vlm",
                                  "window_id": window_id, "vlm_lane": lane,
                                  "error": report.get("error_code"), "reference": raw_reference})
            return
        judgments = {x["event_label"]: x for x in report.get("judgments", [])}
        if len(judgments) != len(report.get("judgments", [])):
            raise ValueError("duplicate label judgment")
        observations = report.get("observations", [])
        observation_ids = {}
        for observation in observations:
            label, raw_eid = observation["event_label"], observation["evidence_id"]
            # Parallel lanes share request_id and may reuse the same raw index.
            # Keep grounding IDs for fusion; namespace information IDs within evaluation evidence.
            eid = f"information:{raw_eid}" if lane == "information" else raw_eid
            if raw_eid in observation_ids:
                raise ValueError("duplicate raw evidence ID within one lane")
            observation_ids[raw_eid] = eid
            if label not in plan["event_labels"] or eid in self.events:
                raise ValueError("unexpected label or duplicate raw evidence ID")
            detected = observation.get("detected")
            if detected is not None and type(detected) is not bool:
                raise ValueError("explicit tri-state observation required")
            issues = set(observation.get("grounding_issues", []))
            whole_scene = (report.get("event_scope") == "scene"
                           or observation.get("scope") == "scene"
                           or "scene_only" in issues
                           or (detected is False and "scene_only_negative" in issues))
            self.events[eid] = {
                "evidence_id": eid, "raw_evidence_id": raw_eid,
                "window_id": window_id, "event_label": label,
                "start": plan["start"], "end": plan["end"], "detected": detected,
                "scope": "scene" if whole_scene else "subject", "vlm_lane": lane,
                "publication_status": "published", "native_confirmation_status": "not_evaluated",
                "subject_ids": list(observation.get("subject_ids", [])),
                "associated_subject_ids": [], "association_status": "unresolved",
                "raw_reference": raw_reference,
            }
        for label in plan["event_labels"]:
            judgment = judgments.get(label)
            if judgment is None:
                continue
            detected = judgment.get("observed_detected")
            if detected is not None and type(detected) is not bool:
                raise ValueError("explicit tri-state event judgment required")
            matching = [x for x in observations if x.get("event_label") == label]
            eid = f"published:{report['message_id']}:{label}"
            if eid in self.events:
                raise ValueError("duplicate published evidence ID")
            whole_scene = (report.get("event_scope") == "scene" or bool(matching) and all(
                self.events[observation_ids[x["evidence_id"]]]["scope"] == "scene" for x in matching))
            self.events[eid] = {
                "evidence_id": eid, "window_id": window_id, "event_label": label,
                "start": plan["start"], "end": plan["end"], "detected": detected,
                "scope": "scene" if whole_scene else "subject", "vlm_lane": lane,
                "publication_status": "published",
                "native_confirmation_status": judgment.get("confirmation_state"),
                "subject_ids": [], "association_status": "not_required_for_scene" if whole_scene else "unresolved",
                "raw_observation_ids": [observation_ids[x["evidence_id"]] for x in matching],
                "raw_reference": raw_reference,
            }

    def scene_snapshot(self, snapshot, *, association_links, raw_reference):
        """Collect native confirmed episodes; require observed negatives rather than disappearance."""
        self._open()
        scene = snapshot.get("scene", snapshot)
        self._scope(scene)
        if not raw_reference:
            raise ValueError("saved scene reference required")
        for link in association_links:
            self.links[link["evidence_id"]] = copy.deepcopy(link)
            if link["evidence_id"] in self.events:
                row = self.events[link["evidence_id"]]
                row["associated_subject_ids"] = [x["subject_id"] for x in link.get("subjects", [])]
                row["association_status"] = "matched" if row["associated_subject_ids"] else "scene"
        for event in scene.get("events", []):
            key = event["event_id"]
            history = event.get("history", [])
            refs = list(dict.fromkeys(eid for item in history
                                      for eid in item.get("evidence_ids", [item.get("evidence_id")])
                                      if eid))
            old = self.episodes.get(key, {})
            refs = list(dict.fromkeys([*old.get("evidence_ids", []), *refs]))
            confirmed = event.get("confirmed_since_timestamp_seconds") is not None
            ended = event.get("ended_since_timestamp_seconds")
            explicit_end = (event.get("phase") == "ended" and ended is not None
                            and not event.get("ended_reason")
                            and any(x.get("phase") == "ended" and x.get("detected") is False
                                    and x.get("timestamp_seconds") == ended for x in history))
            end_refs = [eid for item in history
                        if item.get("phase") == "ended" and item.get("detected") is False
                        and item.get("timestamp_seconds") == ended
                        for eid in item.get("evidence_ids", [item.get("evidence_id")]) if eid]
            ids = event.get("subject_ids", [])
            linked = [self.links[eid] for eid in refs if eid in self.links]
            # Reuse production fusion links without extra nearest-center matching.
            matched = bool(ids) and any(
                set(ids) == {x["subject_id"] for x in link.get("subjects", [])}
                for link in linked)
            retained_end = (event.get("phase") == "ended" and not event.get("ended_reason")
                            and ended is not None and old.get("end") == ended)
            self.episodes[key] = {
                "episode_id": key, "event_label": event["event_label"],
                "subject_ids": list(ids), "confirmed": confirmed,
                "association_status": "matched" if matched else "scene" if not ids else "unresolved",
                "start": event["first_evidence_timestamp_seconds"],
                "end": ended if explicit_end or retained_end else None,
                "end_reason": "explicit_negative" if explicit_end else old.get("end_reason") if retained_end else None,
                "end_evidence_ids": end_refs if explicit_end else old.get("end_evidence_ids", []) if retained_end else [],
                "phase": event["phase"], "evidence_ids": refs,
                "native_ended_reason": event.get("ended_reason"), "raw_reference": raw_reference,
            }
            if event.get("phase") == "retracted":
                self.episodes[key]["confirmed"] = False

    def request_failure(self, *, stage, reference, reason):
        self._open()
        self.failures.append({"stage": stage, "reference": reference, "reason": reason})

    def seal(self, *, source_eof, tracking_drained, events_drained, fusion_drained):
        self._open()
        values = (source_eof, tracking_drained, events_drained, fusion_drained)
        if any(type(x) is not bool for x in values):
            raise ValueError("explicit EOF and drain states required")
        self._completion = dict(zip(("source_eof", "tracking_drained", "events_drained",
                                     "fusion_drained"), values))
        for window in self.window_plan:
            for lane in window["report_lanes"]:
                if (window["window_id"], lane) not in self.reports:
                    self.failures.append({"stage": "fusion" if lane == "grounding" else "independent_vlm",
                                          "window_id": window["window_id"], "vlm_lane": lane,
                                          "reason": "planned_terminal_report_missing"})
        self._closed = True
        return self.snapshot()

    def snapshot(self):
        complete_frames = len(self.frames) == len(self.frame_plan) and bool(self.frame_plan)
        complete_windows = all(
            all((row["window_id"], lane) in self.reports for lane in row["report_lanes"] if lane != "grounding") and all(
                any(x["window_id"] == row["window_id"] and x["event_label"] == label and x["vlm_lane"] != "grounding"
                    for x in self.events.values()) for label in row["event_labels"])
            for row in self.window_plan if row["report_lanes"] != ["grounding"])
        complete = (self._closed and all(self._completion.values()) and complete_frames
                    and complete_windows and not self.failures)
        evidence = {
            "schema_version": "unified-evidence-v2", "video_id": self.video_id,
            "stream_session_id": self.session_id, "source_id": self.source_id,
            "source": self.source, "provenance": self.provenance,
            "face_enabled": False, "request_failure": bool(self.failures),
            "failures": self.failures, **self._completion,
            "frames": [self.frames[x] for x in sorted(self.frames)],
            "independent_events": list(self.events.values()),
            "fused_episodes": list(self.episodes.values()),
            "observation_manifest": {"complete": complete,
                "person_frames_complete": complete_frames,
                "event_windows_complete": complete_windows,
                "frame_plan": self.frame_plan, "window_plan": self.window_plan},
        }
        # Both answer branches record this hash and reuse this observation.
        return copy.deepcopy({**evidence, "evidence_sha256": _digest(evidence)})
