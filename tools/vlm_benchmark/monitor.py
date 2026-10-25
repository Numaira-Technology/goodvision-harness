"""Loopback-only live benchmark view with optional detector preview calls."""

from __future__ import annotations

import base64
import bisect
import copy
import json
import math
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from tools.vlm_benchmark.recording import save_recording
from tools.vlm_benchmark.report import write_json

ASSETS = Path(__file__).resolve().parent
# Spatial overlap alone is only a preview estimate near the evidence time.
EVENT_SUBJECT_MAX_AGE = 0.30


def lane_name(report):
    return (
        "grounding" if report.get("vlm_lane") in {"grounding", "combined"} else "events"
    )


def _normalized_xyxy(bbox):
    """A report bbox dict -> (x1,y1,x2,y2) in [0,1], or None if not usable."""
    if not isinstance(bbox, dict):
        return None
    try:
        box = (
            float(bbox["x1"]),
            float(bbox["y1"]),
            float(bbox["x2"]),
            float(bbox["y2"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not (
        all(math.isfinite(value) and 0 <= value <= 1 for value in box)
        and box[0] < box[2]
        and box[1] < box[3]
    ):
        return None
    return box


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _event_subjects(report):
    """Fired, grounded events mapped to the subject box doing them.

    Returns [{"event": label, "bbox": (x1,y1,x2,y2)}]. Only observations that
    the model both detected and grounded to a subject are included; the subject
    box is looked up from report.objects by id. Empty for scene-only verdicts.
    """
    boxes_by_id = {}
    for obj in report.get("objects", []):
        box = _normalized_xyxy(obj.get("bbox"))
        if box is not None and obj.get("object_id"):
            boxes_by_id[obj["object_id"]] = box
    subjects = []
    for obs in report.get("observations", []):
        if not (obs.get("detected") and obs.get("grounded")):
            continue
        for sid in obs.get("subject_ids", []):
            box = boxes_by_id.get(sid)
            if box is not None:
                subjects.append({"event": obs.get("event_label"), "bbox": box})
    return subjects


def _event_matches(objects, subjects):
    """Conservative one-to-one overlap estimates, never person identities."""
    labels_by_box = {}
    for subject in subjects:
        labels_by_box.setdefault(tuple(subject["bbox"]), set()).add(subject["event"])
    scores = {
        box: {
            index: _iou(box, _normalized_xyxy(obj["bbox"]))
            for index, obj in enumerate(objects)
            if obj.get("observation_state", "observed") == "observed"
        }
        for box in labels_by_box
    }

    def winner(candidates):
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        if (
            ranked
            and ranked[0][0] >= 0.3
            and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.15)
        ):
            return ranked[0][1]
        return None

    matches = {}
    unmatched = set(labels_by_box)
    for box, values in scores.items():
        target = winner((score, index) for index, score in values.items())
        if (
            target is not None
            and winner((others[target], other) for other, others in scores.items())
            == box
        ):
            matches[target] = sorted(label for label in labels_by_box[box] if label)
            unmatched.discard(box)
    return matches, bool(unmatched)


def media_time(frames, tick, fps):
    index = tick % len(frames)
    times = getattr(frames, "media_seconds", None)
    return times[index] if times else index / fps


def _positive_env(name, default, convert=float):
    try:
        value = convert(os.environ.get(name, default))
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if value <= 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f"{name} must be positive and finite")
    return value


class LiveMonitor:
    def __init__(self, output, framesets, args):
        self.output, self.framesets, self.args = Path(output), framesets, args
        self.layout = getattr(args, "monitor_layout", "evidence")
        self.lock = threading.RLock()
        self.state = {"stage": None, "state": "preparing"}
        self.log = None
        self.started = None
        self.latencies = {}
        self.seen = set()
        self.previews = []
        self.token = secrets.token_urlsafe(24)
        self.server = None
        self.recorder = None
        self.preview_trackers = {}
        self.preview_viewers = {}
        # Optional: draw real YOLO detector boxes (colored by event state) in the
        # preview instead of the VLM-seeded optical-flow estimate. Points at a
        # running detector worker (tools/start_detector.py).
        self.detector_url = (
            os.environ.get("PREVIEW_DETECTOR_URL", "").strip().rstrip("/")
        )
        self.detector_model = os.environ.get(
            "PREVIEW_DETECTOR_MODEL", "yolo26-s"
        ).strip()
        self._detector_sessions = {}
        self._detector_lock = threading.Lock()
        # Optional numbered-person selection supplements the benchmark evidence.
        # Its source-frame lifetime and detector-session identity bound every tag.
        self._select_events = os.environ.get(
            "PREVIEW_SELECT_EVENTS", ""
        ).strip().lower() not in {"", "0", "false", "no", "off"}
        self.backend_person_tracks = getattr(
            args, "backend_person_tracks", getattr(args, "person_ids_only", False)
        )
        self.person_display = getattr(args, "person_display", "ids")
        if self.person_display not in {"ids", "boxes", "off"}:
            raise ValueError("invalid person display")
        if self.backend_person_tracks:
            # Backend broker owns detection; preview must not add API load or boxes.
            self.detector_url = ""
            self._select_events = False
        self._select_key = (
            os.environ.get("OPENROUTER_API_KEY", "").strip()
            if self._select_events else ""
        )
        self._select_model = os.environ.get(
            "PREVIEW_SELECT_MODEL", "qwen/qwen3-vl-30b-a3b-instruct"
        ).strip()
        self._select_labels = (
            "fight",
            "smoking",
            "fall",
            "weapon",
            "ppe_violation",
            "ground_litter",
        )
        # Tunable for dense coverage on many streams: with N cameras sharing
        # _select_max_inflight slots and each select ~3s, a camera refreshes every
        # ~N/max_inflight*3s; TTL must exceed that gap or the overlay flickers.
        self._select_interval = _positive_env("PREVIEW_SELECT_INTERVAL", "4")
        self._select_ttl = _positive_env("PREVIEW_SELECT_TTL", "6")
        self._select_states = {}
        self._select_waiters = {}
        self._select_inflight = set()
        self._select_max_inflight = _positive_env(
            "PREVIEW_SELECT_MAX_INFLIGHT", "4", int
        )
        self._select_closed = False
        self._select_state_lock = threading.Lock()

    def __enter__(self):
        from tools.vlm_benchmark.recording import BrowserRecorder, make_preview

        media_dir = self.output / "monitor-media"
        media_dir.mkdir()
        for i, frames in enumerate(self.framesets):
            path = getattr(frames, "source_path", None)
            preview = media_dir / f"video-{i + 1}.mp4"
            if path:
                print(
                    f"Preparing browser preview {i + 1} (original playback speed)...",
                    flush=True,
                )
                make_preview(path, preview)
                self.previews.append(preview)
            else:
                if self.layout == "single-video":
                    raise ValueError("single-video monitor requires an actual video")
                self.previews.append(None)
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", self.args.monitor_port), self.handler()
        )
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/{self.token}/"
        print(f"Live monitor: {self.url}", flush=True)
        write_json(
            self.output / "monitor.json",
            {
                "url": self.url,
                "recording": self.args.record,
                "scope": "live benchmark browser viewport; no retrospective replay",
                "preview": "15 FPS H.264 proxy; media PTS/inference JPEGs unchanged",
                "layout": self.layout,
            },
        )
        try:
            if self.args.record:
                # Tracking overlays are opt-in (?tracking=on). The YOLO detector
                # preview exists to show boxes, so record it with tracking on;
                # plain benchmark recordings stay overlay-free as before.
                record_url = self.url + ("?tracking=on" if self.detector_url else "")
                self.recorder = BrowserRecorder(record_url, self.args.headed)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        with self._select_state_lock:
            self._select_closed = True
            self._select_states.clear()
            self._select_waiters.clear()
        try:
            if self.recorder:
                self.recorder.close()
        finally:
            if self.log:
                self.log.close()
                self.log = None
            if self.server:
                self.server.shutdown()
                self.server.server_close()
                self.thread.join(timeout=5)

    def emit(self, kind, **payload):
        if self.log:
            self.log.write(
                json.dumps({"kind": kind, "monotonic": time.monotonic(), **payload})
                + "\n"
            )

    def begin(self, job, location):
        from tools.vlm_multistream_benchmark import planned_observations

        with self.lock:
            self.location = location
            self.preview_trackers = {}
            self.preview_viewers = {}
            with self._select_state_lock:
                self._select_states.clear()
                self._select_waiters.clear()
            if self.log:
                self.log.close()
            self.log = (location / "monitor.jsonl").open("w", encoding="utf-8")
            self.started, self.seen = None, set()
            self.latencies = {
                lane: []
                for lane in (
                    ("events", "grounding")
                    if job["mode"] == "parallel"
                    else ("events",)
                    if job["mode"] == "events"
                    else ("grounding",)
                )
            }
            self.state = {
                "stage": job["id"],
                "state": "preparing",
                "mode": job["mode"],
                "phase": job["phase"],
                "duration": job["duration"],
                "repeat": job["repeat"],
                "model": self.args.model,
                "real": self.args.real,
                "backend": self.args.backend,
                "person_ids_only": getattr(self.args, "person_ids_only", False),
                "backend_person_tracks": self.backend_person_tracks,
                "person_display": self.person_display,
                "monitor_layout": self.layout,
                "preview_fps": 15,
                "sample_fps": getattr(self.args, "source_sample_fps", None) or max(
                    2.0, self.args.observation_fps * 2
                ),
                "observation_fps": self.args.observation_fps,
                "planned": planned_observations(
                    job["duration"], self.args.observation_fps,
                    getattr(self.args, "source_sample_fps", None),
                )
                * job["streams"],
                "offsets": [0] * job["streams"],
                "summary": None,
                "source_count": len(self.framesets),
                "lanes": list(self.latencies),
                # The browser enables the YOLO box preview on this flag even when
                # there is no VLM grounding lane (events-only runs).
                "detector": bool(self.detector_url),
                "cameras": [
                    {
                        "camera": i + 1,
                        "tick": -1,
                        "rejected": 0,
                        "out_of_order": 0,
                        "lanes": {},
                        "tracking_anchor": None,
                    }
                    for i in range(job["streams"])
                ],
            }
            self.emit("stage", **self.state)
        self.recording_active = bool(
            self.recorder and job["phase"] in {"measurement", "soak"}
        )
        if self.recording_active:
            self.recorder.begin(location, job["id"])

    def source_started(self, started, offsets):
        with self.lock:
            self.started = started
            self.state.update(state="playing", offsets=offsets)
            self.emit("source_started", started=started, offsets=offsets)

    def sample(self, camera, tick, accepted):
        if self.recorder:
            self.recorder.check()
        with self.lock:
            item = self.state["cameras"][camera]
            item["tick"] = tick
            item["rejected"] += int(not accepted)
            frames = self.framesets[camera % len(self.framesets)]
            self.emit(
                "sample",
                camera=camera + 1,
                tick=tick,
                accepted=accepted,
                media_seconds=media_time(frames, tick, self.state["sample_fps"]),
            )

    def result(self, camera, report):
        with self.lock:
            key = (
                camera,
                report.get("message_id")
                or (report.get("window_id"), lane_name(report)),
            )
            if key in self.seen:
                return
            self.seen.add(key)
            lane = lane_name(report)
            if lane not in self.latencies:
                return
            tick = report.get("frame", {}).get("source_frame_index")
            frames = self.framesets[camera % len(self.framesets)]
            valid_tick = type(tick) is int and 0 <= tick < math.ceil(
                self.state["duration"] * self.state["sample_fps"]
            )
            seconds = report.get("timing", {}).get("source_to_publish_seconds")
            complete = (
                report.get("state") == "completed"
                and report.get("analysis_complete") is True
            )
            if (
                complete
                and isinstance(seconds, (float, int))
                and math.isfinite(seconds)
            ):
                bisect.insort(self.latencies[lane], seconds)
            view = {
                "id": report.get("message_id"),
                "window": report.get("window_id"),
                "tick": tick if valid_tick else None,
                "source_seconds": report.get("frame", {}).get("timestamp_seconds"),
                "media_seconds": media_time(frames, tick, self.state["sample_fps"])
                if valid_tick
                else None,
                "received_elapsed": time.monotonic() - self.started
                if self.started
                else None,
                "latency": seconds,
                "complete": complete,
                "error": report.get("error_code"),
                "judgments": [
                    {
                        "label": j.get("event_label"),
                        "observed": j.get("observed_detected"),
                    }
                    for j in report.get("judgments", [])
                ],
                "objects": [
                    {"bbox": o.get("bbox"), "label": o.get("label")}
                    for o in report.get("objects", [])
                ],
                # Per-subject event grounding: each fired+grounded observation
                # links an event to a subject whose box is in report.objects.
                # Near the evidence time, the preview can estimate which
                # detector box overlaps the actor. Older or ambiguous evidence
                # retains a generic scene alert without assigning an actor.
                "event_subjects": _event_subjects(report),
            }
            current = self.state["cameras"][camera]
            previous = current["lanes"].get(lane)
            stale = bool(
                report.get("historical_only")
                or (previous and (tick if valid_tick else -1) < (previous["tick"] or 0))
            )
            if stale:
                current["out_of_order"] += 1
            else:
                current["lanes"][lane] = view
            anchor = current["tracking_anchor"]
            if (
                lane == "grounding"
                and not stale
                and complete
                and not view["error"]
                and valid_tick
                and (anchor is None or tick > anchor["tick"])
            ):
                current["tracking_anchor"] = view
            self.emit("result", camera=camera + 1, lane=lane, stale=stale, result=view)

    def draining(self):
        with self.lock:
            self.state["state"] = "draining"
            self.emit("draining")

    def finish(self, result=None, error=None):
        with self.lock:
            self.state.update(
                state="failed" if error else "finished",
                summary=result,
                error=type(error).__name__ if error else None,
            )
            if self.started:
                self.state["finished_elapsed"] = time.monotonic() - self.started
            self.emit("finish", result=result, error=self.state["error"])
            self.log.flush()
        if self.recording_active:
            try:
                return self.recorder.finish(
                    self.state["stage"], result is not None and not error
                )
            except Exception as exc:
                record = {
                    "state": "failed",
                    "complete": False,
                    "error_type": type(exc).__name__,
                }
                folder = self.location / "recording"
                folder.mkdir(exist_ok=True)
                save_recording(folder / "recording.json", record)
                return record
            finally:
                self.recording_active = False
        return None

    def snapshot(self, client=None):
        with self.lock:
            data = copy.deepcopy(self.state)
            if not data.get("stage"):
                return data
            data["elapsed"] = (time.monotonic() - self.started) if self.started else 0
            data["counters"] = {
                lane: {
                    "complete": len(values),
                    "p50": values[max(0, math.ceil(len(values) * 0.5) - 1)]
                    if values
                    else None,
                    "p95": values[max(0, math.ceil(len(values) * 0.95) - 1)]
                    if values
                    else None,
                }
                for lane, values in self.latencies.items()
            }
            data["media"] = [
                {
                    "url": f"media/{i}" if self.previews[i] else None,
                    "timestamps": getattr(
                        frames,
                        "media_seconds",
                        [j / data["sample_fps"] for j in range(len(frames))],
                    ),
                    "duration": getattr(
                        frames, "media_duration", len(frames) / data["sample_fps"]
                    ),
                    "sample_count": len(frames),
                }
                for i, frames in enumerate(self.framesets)
            ]
            # Surface the real per-person select events (YOLO boxes + cheap VLM
            # "which number" query) onto each camera so the chips reflect them,
            # not just the box colour, and flag that a real model is in use.
            if self._select_events:
                data["select_active"] = bool(self._select_key and self.detector_url)
                now = time.monotonic()
                with self._select_state_lock:
                    live = (
                        isinstance(client, str)
                        and re.fullmatch(r"[a-f0-9-]{36}", client)
                        and now - self.preview_viewers.get(client, -math.inf) <= 30
                        and data["state"] not in {"draining", "finished", "failed"}
                    )
                    for i, cam in enumerate(data.get("cameras", [])):
                        key = (client, i)
                        state = self._select_states.get(key) if live else None
                        session = self._detector_sessions.get(key)
                        events = set()
                        cam["select_epoch"] = cam["select_generation"] = None
                        if (
                            key in self.preview_trackers
                            and state is not None
                            and session is not None
                            and state["stage"] == data["stage"]
                            and session["stage"] == data["stage"]
                            and state["session"] == session["id"]
                            and state["expires"] > now
                        ):
                            for labels in state["tags"].values():
                                events.update(labels)
                            cam["select_epoch"] = session["epoch"]
                            cam["select_generation"] = session["generation"]
                        cam["select_events"] = sorted(events)
            return data

    def track_preview(self, stage, camera, payload):
        if self.layout == "single-video":
            raise ValueError("tracking disabled in single-video layout")
        from tools.vlm_benchmark.preview_tracking import PreviewTracker, gray_frame

        client = payload.get("client", "")
        at, epoch = payload.get("time"), payload.get("epoch")
        generation = payload.get("generation", 0)
        if (
            not isinstance(client, str)
            or not re.fullmatch(r"[a-f0-9-]{36}", client)
            or type(at) not in (int, float)
            or not math.isfinite(at)
            or at < 0
            or type(epoch) is not int
            or not 0 <= epoch <= 100000
            or type(generation) is not int
            or not 0 <= generation <= 1000000
        ):
            raise ValueError("invalid preview metadata")
        with self.lock:
            # The optical-flow preview needs VLM grounding boxes as seeds; the
            # YOLO detector preview supplies its own boxes, so it only needs a
            # live stage (works in events-only mode, no grounding lane).
            if (
                stage != self.state.get("stage")
                or (
                    not self.detector_url
                    and "grounding" not in self.state.get("lanes", [])
                )
                or not 0 <= camera < len(self.state["cameras"])
            ):
                raise ValueError("unknown preview")
            now = time.monotonic()
            expired = {
                c for c, touched in self.preview_viewers.items() if now - touched > 30
            }
            self.preview_trackers = {
                k: v for k, v in self.preview_trackers.items() if k[0] not in expired
            }
            self.preview_viewers = {
                c: t for c, t in self.preview_viewers.items() if c not in expired
            }
            key = (client, camera)
            # At most two simultaneous viewers; never an unbounded image cache.
            if key not in self.preview_trackers:
                clients = {k[0] for k in self.preview_trackers}
                if client not in clients and len(clients) >= 2:
                    raise ValueError("too many tracking viewers")
                self.preview_trackers[key] = PreviewTracker()
            self.preview_viewers[client] = now
            tracker = self.preview_trackers[key]
            view = self.state["cameras"][camera]["tracking_anchor"]
            frames = self.framesets[camera % len(self.framesets)]
            duration = getattr(
                frames, "media_duration", len(frames) / self.state["sample_fps"]
            )
            lanes = self.state["cameras"][camera].get("lanes", {})
            event_subjects = []
            positive_labels, grounded_labels = set(), set()
            for report in lanes.values():
                if not report or not report["complete"] or report["error"]:
                    continue
                subjects = report.get("event_subjects") or []
                positive_labels.update(
                    judgment.get("label")
                    for judgment in report.get("judgments", [])
                    if judgment.get("observed") is True
                )
                tick = report["tick"]
                if tick is None:
                    continue
                source_epoch = tick // len(frames)
                source_time = source_epoch * duration + report["media_seconds"]
                # Never match a delayed/future/previous-loop rectangle against
                # somebody now occupying that location. No identity is carried
                # beyond this short evidence-aligned spatial estimate.
                if epoch == source_epoch and (
                    0 <= at - source_time <= EVENT_SUBJECT_MAX_AGE
                ):
                    event_subjects.extend(subjects)
                    grounded_labels.update(subject["event"] for subject in subjects)
            # Resolve fallback per event across both lanes: grounding smoking
            # cannot hide a scene-only fight, or make scene-only smoking turn
            # every person red when its grounding lane supplied a subject.
            scene_alert = bool(positive_labels - grounded_labels)
            if not epoch * duration <= at <= (epoch + 1) * duration + 0.05:
                raise ValueError("preview time does not match loop")
            anchor = None
            if view is not None:
                tick = view["tick"]
                anchor = {
                    "id": f"{tick}:{view['id']}",
                    "epoch": tick // len(frames),
                    "time": (tick // len(frames)) * duration + view["media_seconds"],
                    "objects": view["objects"],
                    "jpeg": frames[tick % len(frames)],
                }
        if not tracker.lock.acquire(blocking=False):
            return {"status": "busy", "objects": []}
        try:
            jpeg = base64.b64decode(payload.get("jpeg", ""), validate=True)
            # Check compressed dimensions before allocating an OpenCV image.
            from io import BytesIO

            from PIL import Image

            with Image.open(BytesIO(jpeg)) as image:
                if (
                    image.format != "JPEG"
                    or not 8 <= min(image.size)
                    or max(image.size) > 640
                ):
                    raise ValueError("invalid preview dimensions")
            if self.detector_url:
                return self._yolo_track(
                    key,
                    at,
                    jpeg,
                    event_subjects,
                    scene_alert,
                    stage=stage,
                    epoch=epoch,
                    generation=generation,
                    evidence_at=now,
                )
            return tracker.advance(gray_frame(jpeg), at, epoch=epoch, anchor=anchor)
        finally:
            tracker.lock.release()

    def _yolo_track(
        self,
        key,
        at,
        jpeg,
        event_subjects,
        scene_alert,
        *,
        stage=None,
        epoch=0,
        generation=0,
        evidence_at=None,
    ):
        """Detect live preview boxes and apply bounded spatial event estimates.

        Worker requests are serialized. Sessions are isolated across stages,
        loops and seeks, so unrelated source frames cannot share tracker state.
        """
        evidence_at = time.monotonic() if evidence_at is None else evidence_at
        scene_state = "alert" if scene_alert else "normal"
        unavailable = {
            "status": "detector_unavailable",
            "event_state": "alert" if event_subjects else scene_state,
            "objects": [],
        }
        with self._detector_lock:
            now = time.monotonic()
            self._detector_sessions = {
                k: value
                for k, value in self._detector_sessions.items()
                if now - value["touched"] <= 30
            }
            session = self._detector_sessions.get(key)
            if (
                session is None
                or session["stage"] != stage
                or session["epoch"] != epoch
                or session["generation"] != generation
                or at <= session["at"]
            ):
                session = {
                    "id": "preview-" + secrets.token_hex(16),
                    "sequence": 0,
                    "stage": stage,
                    "epoch": epoch,
                    "generation": generation,
                }
                self._detector_sessions[key] = session
                with self._select_state_lock:
                    self._select_states.pop(key, None)
                    self._select_waiters.pop(key, None)
            session.update(sequence=session["sequence"] + 1, at=at, touched=now)
            # Wall time remains monotonic even when the clip wraps or seeks.
            request = urllib.request.Request(
                f"{self.detector_url}/v1/tracks",
                data=jpeg,
                method="POST",
                headers={
                    "Content-Type": "image/jpeg",
                    "x-goodvision-stream-session-id": session["id"],
                    "x-goodvision-frame-sequence": str(session["sequence"]),
                    "x-goodvision-timestamp-seconds": repr(now),
                    "X-GoodVision-Detector-Model": self.detector_model,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=2.0) as response:
                    payload = response.read(1024 * 1024 + 1)
                    if len(payload) > 1024 * 1024:
                        return unavailable
                    snapshot = json.loads(payload)
                if not isinstance(snapshot, dict) or not isinstance(
                    snapshot.get("objects"), list
                ):
                    return unavailable
                objects = snapshot["objects"]
                ids = set()
                for obj in objects:
                    if not isinstance(obj, dict):
                        return unavailable
                    track_id = obj.get("track_id")
                    if (
                        not isinstance(track_id, str)
                        or not track_id.strip()
                        or track_id in ids
                        or _normalized_xyxy(obj.get("bbox")) is None
                    ):
                        return unavailable
                    ids.add(track_id)
            except (urllib.error.URLError, OSError, ValueError, TimeoutError):
                return unavailable
        # Selection supplements existing evidence, including when its key is
        # absent, a request is pending, or the provider fails validation.
        matches, unmatched = _event_matches(objects, event_subjects)
        tags, selected_scene_alert = {}, False
        if self._select_events:
            raw = [
                (obj["track_id"], _normalized_xyxy(obj["bbox"]))
                for obj in objects
                if obj.get("observation_state", "observed") == "observed"
            ]
            # Feed the full-res source frame (not the 320px capture) to the VLM
            # select query; 320px is too small to read actions (fight/fall).
            self._maybe_select(
                key, session["id"], stage, evidence_at,
                self._hires_frame(key, at, epoch) or jpeg, raw,
            )
            tags, selected_scene_alert = self._active_tags(
                key, session["id"], {track_id for track_id, _ in raw}
            )
        if unmatched or selected_scene_alert:
            scene_state = "alert"
        result = []
        for index, obj in enumerate(objects):
            events = sorted(
                set(matches.get(index, ())) | set(tags.get(obj["track_id"], ()))
            )
            entry = {
                "track_id": obj["track_id"],
                "bbox": obj["bbox"],
                "event_state": "alert" if events else scene_state,
            }
            if events:
                entry["event"] = ", ".join(events)
            result.append(entry)
        return {
            "status": "tracking" if result else "no_person_boxes",
            "event_state": "alert"
            if any(obj["event_state"] == "alert" for obj in result)
            else scene_state,
            "objects": result,
        }

    def _active_tags(self, key, session_id, observed):
        """Return only fresh actor tags from this detector session."""
        with self._select_state_lock:
            state = self._select_states.get(key)
            if (
                self._select_closed
                or state is None
                or state["session"] != session_id
                or state["expires"] <= time.monotonic()
            ):
                return {}, False
            tags = state["tags"]
            return (
                {
                    track_id: labels
                    for track_id, labels in tags.items()
                    if track_id in observed
                },
                state["scene_alert"] or bool(set(tags) - observed),
            )

    def _hires_frame(self, key, at, epoch):
        """Use a source JPEG only when its exact media PTS matches the capture.

        Preview playback is denser than the stored source samples. A nearby
        sample can show different people, so unmatched captures must keep their
        uploaded JPEG and its detector boxes together.
        """
        try:
            camera = key[1]
            frames = self.framesets[camera % len(self.framesets)]
            timestamps = getattr(frames, "media_seconds", ())
            duration = getattr(frames, "media_duration", None)
            if (
                len(timestamps) != len(frames)
                or type(duration) not in (int, float)
                or not math.isfinite(duration)
                or duration <= 0
                or type(at) not in (int, float)
                or not math.isfinite(at)
                or type(epoch) is not int
                or epoch < 0
            ):
                return None
            local = at - epoch * duration
            if not 0 <= local < duration:
                return None
            # Cover timestamp serialization and the proxy's 90 kHz rounding,
            # without substituting a different sparse evidence frame.
            index = bisect.bisect_left(timestamps, local - 1e-4)
            if index < len(frames) and math.isclose(
                timestamps[index], local, rel_tol=0, abs_tol=1e-4
            ):
                return frames[index]
        except (IndexError, KeyError, TypeError, ZeroDivisionError, AttributeError):
            pass
        return None

    def _maybe_select(self, key, session_id, stage, evidence_at, jpeg, raw):
        """Reserve a bounded asynchronous selection for a current source frame."""
        with self._detector_lock:
            now = time.monotonic()
            session = self._detector_sessions.get(key)
            if session is None or session["id"] != session_id:
                return
            with self.lock:
                live = {
                    candidate
                    for candidate in self.preview_trackers
                    if now - self.preview_viewers.get(candidate[0], -math.inf) <= 30
                }
                with self._select_state_lock:
                    self._select_states = {
                        candidate: state
                        for candidate, state in self._select_states.items()
                        if candidate in live and now - state["touched"] <= 30
                    }
                    self._select_waiters = {
                        candidate: (identity, touched)
                        for candidate, (
                            identity,
                            touched,
                        ) in self._select_waiters.items()
                        if candidate in self._select_states
                        and self._select_states[candidate]["session"] == identity
                        and self._select_states[candidate]["stage"]
                        == self.state.get("stage")
                        and now - touched < self._select_ttl
                    }
                    if (
                        self._select_closed
                        or key not in live
                        or stage != self.state.get("stage")
                        or self.state.get("state") in {"draining", "finished", "failed"}
                    ):
                        self._select_waiters.pop(key, None)
                        return
                    state = self._select_states.get(key)
                    if state is None or state["session"] != session_id:
                        state = {
                            "session": session_id,
                            "stage": stage,
                            "touched": now,
                            "last": -math.inf,
                            "pending": None,
                            "tags": {},
                            "scene_alert": False,
                            "expires": 0.0,
                        }
                        self._select_states[key] = state
                    state["touched"] = now
                    if (
                        not self._select_key
                        or not raw
                        or now >= evidence_at + self._select_ttl
                        or state["pending"] is not None
                        or now - state["last"] < self._select_interval
                    ):
                        self._select_waiters.pop(key, None)
                        return
                    # Metadata only: each lane supplies a fresh frame when its
                    # FIFO turn arrives. Recently completed lanes join the tail.
                    self._select_waiters[key] = (session_id, now)
                    if (
                        len(self._select_inflight) >= self._select_max_inflight
                        or next(iter(self._select_waiters)) != key
                    ):
                        return
                    self._select_waiters.pop(key)
                    request_id = secrets.token_hex(16)
                    state.update(last=now, pending=request_id)
                    self._select_inflight.add(request_id)
        numbered = [
            (number, track_id, box) for number, (track_id, box) in enumerate(raw, 1)
        ]
        try:
            threading.Thread(
                target=self._run_select,
                args=(key, session_id, request_id, evidence_at, jpeg, numbered),
                daemon=True,
            ).start()
        except Exception:
            self._finish_select(key, session_id, request_id)

    def _finish_select(self, key, session_id, request_id):
        with self._select_state_lock:
            self._select_inflight.discard(request_id)
            state = self._select_states.get(key)
            if (
                state is not None
                and state["session"] == session_id
                and state["pending"] == request_id
            ):
                state["pending"] = None

    def _current_select(self, key, session_id, request_id, evidence_at):
        """Caller holds the monitor and selection locks in that order."""
        now = time.monotonic()
        state = self._select_states.get(key)
        if (
            not self._select_closed
            and key in self.preview_trackers
            and now - self.preview_viewers.get(key[0], -math.inf) <= 30
            and state is not None
            and state["stage"] == self.state.get("stage")
            and self.state.get("state") not in {"draining", "finished", "failed"}
            and state["session"] == session_id
            and state["pending"] == request_id
            and now < evidence_at + self._select_ttl
        ):
            return state
        return None

    def _run_select(self, key, session_id, request_id, evidence_at, jpeg, numbered):
        try:
            with self.lock, self._select_state_lock:
                if (
                    self._current_select(key, session_id, request_id, evidence_at)
                    is None
                ):
                    return
            tags, scene_alert = self._select_query(jpeg, numbered)
            with self.lock, self._select_state_lock:
                state = self._current_select(key, session_id, request_id, evidence_at)
                if state is not None:
                    # A complete valid negative is a new snapshot too.
                    state.update(
                        tags=tags,
                        scene_alert=scene_alert,
                        expires=evidence_at + self._select_ttl,
                    )
        except Exception:
            pass
        finally:
            self._finish_select(key, session_id, request_id)

    def _select_query(self, jpeg, numbered):
        """Validate a complete provider verdict before binding actor labels."""
        from io import BytesIO

        from PIL import Image, ImageDraw

        with Image.open(BytesIO(jpeg)) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        num_to_track = {}
        for number, track_id, box in numbered:
            num_to_track[number] = track_id
            x1, y1, x2, y2 = (
                box[0] * width,
                box[1] * height,
                box[2] * width,
                box[3] * height,
            )
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            draw.text((x1 + 2, max(0, y1 - 12)), f"#{number}", fill=(255, 255, 0))
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=90)
        data = base64.b64encode(buffer.getvalue()).decode()
        prompt = (
            "People detected by a separate detector are drawn as numbered green "
            "boxes (#1, #2, ...). For each of these events say whether it is "
            "happening and WHICH numbered person is doing it (or null): "
            + ", ".join(self._select_labels)
            + ". Do NOT output coordinates; only use the given numbers. Return "
            "compact JSON {event: {happening: bool, person: number or null}}. "
            "Include every listed event. Use person:null when happening is false."
        )
        body = json.dumps(
            {
                "model": self._select_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
                "temperature": 0,
            }
        ).encode()
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._select_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self._select_ttl) as response:
            encoded = response.read(64 * 1024 + 1)
        if len(encoded) > 64 * 1024:
            raise ValueError("oversized selection response")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate selection field")
                result[key] = value
            return result

        payload = json.loads(encoded, object_pairs_hook=unique)
        if not isinstance(payload, dict):
            raise ValueError("invalid selection completion")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("invalid selection choices")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise ValueError("incomplete selection completion")
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("invalid selection message")
        # Models sometimes wrap the JSON in markdown fences or prose; extract the
        # outermost object so a cosmetic wrapper does not drop the whole verdict.
        content = message["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        verdicts = json.loads(
            match.group(0) if match else content, object_pairs_hook=unique
        )
        if not isinstance(verdicts, dict) or set(verdicts) != set(self._select_labels):
            raise ValueError("incomplete selection verdicts")
        tags, scene_alert = {}, False
        for event, info in verdicts.items():
            if (
                not isinstance(info, dict)
                or set(info) != {"happening", "person"}
                or type(info["happening"]) is not bool
            ):
                raise ValueError("invalid selection verdict")
            person = info["person"]
            if person is not None and (
                type(person) is not int or person not in num_to_track
            ):
                raise ValueError("invalid selection person")
            if not info["happening"]:
                if person is not None:
                    raise ValueError("negative selection has a person")
                continue
            if person is None:
                scene_alert = True
            else:
                tags.setdefault(num_to_track[person], []).append(event)
        return {
            track_id: tuple(sorted(labels)) for track_id, labels in tags.items()
        }, scene_alert

    def handler(self):
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, data, content_type, status=200, extra=None, head=False):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                for key, value in (extra or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                if not head:
                    self.wfile.write(data)

            def do_POST(self):
                path = urlsplit(self.path).path
                prefix = f"/{monitor.token}/tracking/"
                try:
                    if not path.startswith(prefix):
                        raise ValueError("unknown route")
                    match = re.fullmatch(r"([^/]+)/(\d+)", path[len(prefix) :])
                    size = int(self.headers.get("Content-Length", "0"))
                    if (
                        not match
                        or not 0 < size <= 192 * 1024
                        or self.headers.get("Content-Type") != "application/json"
                    ):
                        raise ValueError("invalid preview request")
                    self.connection.settimeout(5)
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        raise ValueError("invalid preview payload")
                    result = monitor.track_preview(match[1], int(match[2]), payload)
                    self.reply(json.dumps(result).encode(), "application/json")
                except (ValueError, TypeError, OSError):
                    self.reply(
                        b'{"error":"invalid tracking request"}', "application/json", 400
                    )

            def do_HEAD(self):
                self.do_GET(head=True)

            def do_GET(self, head=False):
                path = urlsplit(self.path).path
                prefix = f"/{monitor.token}/"
                if not path.startswith(prefix):
                    self.reply(b"Not found", "text/plain", 404, head=head)
                    return
                path = path[len(prefix) :]
                try:
                    if path in {"", "monitor.js", "preview_tracking.js"}:
                        name = "monitor.html" if not path else path
                        self.reply(
                            (ASSETS / name).read_bytes(),
                            "text/html; charset=utf-8"
                            if not path
                            else "text/javascript; charset=utf-8",
                            head=head,
                        )
                    elif path == "state":
                        self.reply(
                            json.dumps(
                                monitor.snapshot(
                                    self.headers.get("X-GoodVision-Preview-Client")
                                )
                            ).encode(),
                            "application/json",
                            head=head,
                        )
                    elif re.fullmatch(r"media/\d+", path):
                        index = int(path.split("/")[1])
                        if (
                            index >= len(monitor.previews)
                            or not monitor.previews[index]
                        ):
                            raise ValueError("media unavailable")
                        file = monitor.previews[index]
                        size = file.stat().st_size
                        start, end, status = 0, size - 1, 200
                        headers = {"Accept-Ranges": "bytes"}
                        if value := self.headers.get("Range"):
                            match = re.fullmatch(r"bytes=(\d+)-(\d*)", value)
                            if not match:
                                self.reply(
                                    b"",
                                    "video/mp4",
                                    416,
                                    {"Content-Range": f"bytes */{size}"},
                                    head=head,
                                )
                                return
                            start, end = (
                                int(match[1]),
                                min(int(match[2]) if match[2] else size - 1, size - 1),
                            )
                            if start > end:
                                self.reply(
                                    b"",
                                    "video/mp4",
                                    416,
                                    {"Content-Range": f"bytes */{size}"},
                                    head=head,
                                )
                                return
                            status = 206
                            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
                        self.send_response(status)
                        for key, value in {
                            "Content-Type": "video/mp4",
                            "Content-Length": str(end - start + 1),
                            **headers,
                        }.items():
                            self.send_header(key, value)
                        self.end_headers()
                        if not head:
                            with file.open("rb") as stream:
                                stream.seek(start)
                                remaining = end - start + 1
                                while remaining:
                                    chunk = stream.read(min(remaining, 256 * 1024))
                                    self.wfile.write(chunk)
                                    remaining -= len(chunk)
                    elif match := re.fullmatch(r"frame/([^/]+)/(\d+)/(\d+)\.jpg", path):
                        stage, camera, tick = match[1], int(match[2]), int(match[3])
                        with monitor.lock:
                            if (
                                stage != monitor.state.get("stage")
                                or not 0 <= camera < len(monitor.state["cameras"])
                                or tick
                                >= math.ceil(
                                    monitor.state["duration"]
                                    * monitor.state["sample_fps"]
                                )
                            ):
                                raise ValueError("unknown evidence frame")
                            frames = monitor.framesets[camera % len(monitor.framesets)]
                            data = frames[tick % len(frames)]
                        self.reply(data, "image/jpeg", head=head)
                    else:
                        self.reply(b"Not found", "text/plain", 404, head=head)
                except (ValueError, IndexError):
                    self.reply(b"Not found", "text/plain", 404, head=head)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        return Handler
