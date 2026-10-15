"""Loopback-only live view of benchmark input/results, with no inference calls."""

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


def lane_name(report):
    return (
        "grounding" if report.get("vlm_lane") in {"grounding", "combined"} else "events"
    )


def _normalized_xyxy(bbox):
    """A report bbox dict -> (x1,y1,x2,y2) in [0,1], or None if not usable."""
    if not isinstance(bbox, dict):
        return None
    try:
        box = (float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not (box[0] < box[2] and box[1] < box[3]):
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


def media_time(frames, tick, fps):
    index = tick % len(frames)
    times = getattr(frames, "media_seconds", None)
    return times[index] if times else index / fps


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
        self.detector_model = (
            os.environ.get("PREVIEW_DETECTOR_MODEL", "yolo26-s").strip()
        )
        self._detector_seq = {}
        self._detector_lock = threading.Lock()

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
                self.recorder = BrowserRecorder(self.url, self.args.headed)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
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
                "monitor_layout": self.layout,
                "preview_fps": 15,
                "sample_fps": max(2.0, self.args.observation_fps * 2),
                "observation_fps": self.args.observation_fps,
                "planned": planned_observations(
                    job["duration"], self.args.observation_fps
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
                # The preview matches these to YOLO tracks so only the actor's
                # box turns red (tight VLM<->detector binding), not the whole
                # lane. Empty when the model only judged the scene (scene_only).
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

    def snapshot(self):
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
            return data

    def track_preview(self, stage, camera, payload):
        if self.layout == "single-video":
            raise ValueError("tracking disabled in single-video layout")
        from tools.vlm_benchmark.preview_tracking import PreviewTracker, gray_frame

        client = payload.get("client", "")
        at, epoch = payload.get("time"), payload.get("epoch")
        if (
            not isinstance(client, str)
            or not re.fullmatch(r"[a-f0-9-]{36}", client)
            or type(at) not in (int, float)
            or not math.isfinite(at)
            or at < 0
            or type(epoch) is not int
            or not 0 <= epoch <= 100000
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
            # Box color follows the camera's latest event verdicts. Prefer
            # per-subject binding: if the model grounded the event to a subject
            # box, only the YOLO track overlapping that subject turns red (tight
            # VLM<->detector binding). Fall back to scene-level (whole lane red)
            # when an event fired but the model only judged the scene.
            lanes = self.state["cameras"][camera].get("lanes", {})
            event_subjects = []
            scene_alert = False
            for report in lanes.values():
                if not report:
                    continue
                event_subjects.extend(report.get("event_subjects") or [])
                if any(j.get("observed") for j in report.get("judgments", [])):
                    scene_alert = True
            # Scene-level alert only matters as a fallback when nothing grounded.
            scene_alert = scene_alert and not event_subjects
            frames = self.framesets[camera % len(self.framesets)]
            duration = getattr(
                frames, "media_duration", len(frames) / self.state["sample_fps"]
            )
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
                return self._yolo_track(key, at, jpeg, event_subjects, scene_alert)
            return tracker.advance(gray_frame(jpeg), at, epoch=epoch, anchor=anchor)
        finally:
            tracker.lock.release()

    def _yolo_track(self, key, at, jpeg, event_subjects, scene_alert):
        """Per-frame YOLO detector boxes, colored by per-subject event binding.

        Calls the detector worker's /v1/tracks with this camera's own stable
        session id and a monotonic sequence, so each preview stream keeps stable
        track_ids. Calls are serialized across cameras so a single-GPU worker is
        never flooded with concurrent requests (which it rejects with 429); each
        stream simply gets boxes in round-robin. Each returned track is matched
        by IoU against the VLM's grounded event subjects: a matched track carries
        event_state="alert" plus the event label, so only the actor turns red.
        When the model only judged the scene (scene_alert, no grounded subject)
        every track turns red as a fallback. Any hiccup degrades to an empty
        frame, never raises.
        """
        scene_state = "alert" if scene_alert else "normal"
        with self._detector_lock:
            seq = self._detector_seq.get(key, 0) + 1
            self._detector_seq[key] = seq
            session_id = f"preview-{key[0]}-{key[1]}"
            # The detector rejects non-increasing per-session frame metadata.
            # The preview loops each clip, so replay media time (`at`) rewinds on
            # every wrap; send a monotonic wall clock instead so the session
            # keeps accepting frames across loops.
            timestamp = time.monotonic()
            request = urllib.request.Request(
                f"{self.detector_url}/v1/tracks",
                data=jpeg,
                method="POST",
                headers={
                    "Content-Type": "image/jpeg",
                    "x-goodvision-stream-session-id": session_id,
                    "x-goodvision-frame-sequence": str(seq),
                    "x-goodvision-timestamp-seconds": repr(timestamp),
                    "X-GoodVision-Detector-Model": self.detector_model,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=2.0) as response:
                    snapshot = json.loads(response.read())
            except (urllib.error.URLError, OSError, ValueError, TimeoutError):
                return {
                    "status": "detector_unavailable",
                    "event_state": scene_state,
                    "objects": [],
                }
        objects = []
        for o in snapshot.get("objects", []):
            bbox = o.get("bbox")
            if not (isinstance(bbox, dict) and o.get("track_id")):
                continue
            track_box = _normalized_xyxy(bbox)
            state, event = scene_state, None
            # Tight binding: a track doing a grounded event (best IoU >= 0.3)
            # turns red and carries that event's label; others stay green even
            # when another person in the same lane triggered the event.
            if track_box is not None and event_subjects:
                best = max(
                    event_subjects,
                    key=lambda s: _iou(track_box, s["bbox"]),
                )
                if _iou(track_box, best["bbox"]) >= 0.3:
                    state, event = "alert", best.get("event")
            entry = {"track_id": o["track_id"], "bbox": bbox, "event_state": state}
            if event:
                entry["event"] = event
            objects.append(entry)
        return {
            "status": "tracking" if objects else "no_person_boxes",
            "event_state": "alert" if any(o["event_state"] == "alert" for o in objects) else scene_state,
            "objects": objects,
        }

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
                            json.dumps(monitor.snapshot()).encode(),
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
