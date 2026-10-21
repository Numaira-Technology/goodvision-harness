"""Paced real-file input to the Harness event branch, sharing one VideoJev service."""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from urllib.parse import urlparse

from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.video_to_stream import VideoFileStream, encode_jpeg
from goodvision_harness.videojev_contract import LABELS, MODEL, REVISION
from goodvision_harness.videojev_model import NATIVE_SHARED_ERRORS
from goodvision_harness.vlm_independent import IndependentVLMBranch, VLMProcessConfig

from .recording import EventStreamRecorder, publish_once

_DEFAULTS = {
    "endpoint": "http://127.0.0.1:8093/predict",
    "labels": list(LABELS),
    "candidate_fps": 2.5,
    "window_frames": 4,
    "frame_delta_seconds": None,
    "observation_fps": 1.25,
    "max_in_flight": 1,
    "http_timeout_seconds": 20.0,
    "drain_timeout_seconds": 30.0,
    "event_policy": "research-v1",
    "event_definitions": {},
    "videojev_question_layout": "repeated",
    "max_image_side": 720,
    "jpeg_quality": 90,
    "start_delay_seconds": 0.5,
}
_STREAM_FIELDS = frozenset(("id", "video", "start_seconds", "end_seconds", "offset_seconds"))


def _finite(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(name + " must be a finite " + ("positive" if positive else "nonnegative") + " number")
    return value


def load_run_config(path):
    """Read observation settings only; benchmark labels/answers have no input field."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) - (_DEFAULTS.keys() | {"streams"}):
        raise ValueError("unknown VideoJev run settings")
    config = {**_DEFAULTS, **raw}
    streams = config.get("streams")
    if not isinstance(streams, list) or not 1 <= len(streams) <= 12:
        raise ValueError("provide 1..12 explicitly listed video streams")
    endpoint = urlparse(config["endpoint"])
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
            or endpoint.path != "/predict"):
        raise ValueError("provide the explicit local scoring /predict endpoint")
    labels = config["labels"]
    if (not isinstance(labels, list) or not labels or len(set(labels)) != len(labels)
            or not set(labels) <= set(LABELS)):
        raise ValueError("select distinct existing Harness event labels")
    for key in ("candidate_fps", "observation_fps", "http_timeout_seconds",
                "drain_timeout_seconds"):
        _finite(config[key], key, positive=True)
    _finite(config["start_delay_seconds"], "start_delay_seconds")
    if config["frame_delta_seconds"] is not None:
        _finite(config["frame_delta_seconds"], "frame_delta_seconds", positive=True)
    for key, low, high in (("max_image_side", 32, 4096), ("jpeg_quality", 1, 100)):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError("invalid " + key)
    definitions = config["event_definitions"]
    if (not isinstance(definitions, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                   for k, v in definitions.items())
            or (definitions and set(definitions) != set(labels))):
        raise ValueError("fixed definitions must exactly cover the selected events")
    normalized, names = [], set()
    for source in streams:
        if (not isinstance(source, dict) or set(source) - _STREAM_FIELDS
                or not isinstance(source.get("id"), str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", source["id"]) is None
                or source["id"] in names
                or not isinstance(source.get("video"), str) or not source["video"]):
            raise ValueError("each stream needs a distinct short ID and video path")
        start = _finite(source.get("start_seconds", 0), "start_seconds")
        end = source.get("end_seconds")
        if end is not None and _finite(end, "end_seconds") <= start:
            raise ValueError("end_seconds must be after start_seconds")
        offset = _finite(source.get("offset_seconds", 0), "offset_seconds")
        video = Path(source["video"]).expanduser()
        if not video.is_absolute():
            video = path.parent / video
        normalized.append(dict(id=source["id"], video=str(video.resolve()),
                               start_seconds=start, end_seconds=end, offset_seconds=offset))
        names.add(source["id"])
    config["streams"] = normalized
    # Validate native scheduling fields without creating a process or HTTP client.
    _process_config(config, normalized[0], "configuration-only")
    return config


def _process_config(config, source, run_id):
    return VLMProcessConfig(
        stream_session_id=run_id + "-" + source["id"], source_id=source["id"],
        generation=1, labels=tuple(config["labels"]),
        model_settings=ModelAPISettings(
            provider="local", endpoint=config["endpoint"],
            timeout_seconds=config["http_timeout_seconds"], max_retries=0,
        ),
        model=MODEL, output_mode="events", max_in_flight=config["max_in_flight"],
        observation_fps=config["observation_fps"], candidate_fps=config["candidate_fps"],
        observation_window_frames=config["window_frames"],
        observation_frame_delta_seconds=config["frame_delta_seconds"],
        event_policy=config["event_policy"], retain_historical_results=True,
        event_definitions=tuple(config["event_definitions"].items()),
        videojev_question_layout=config["videojev_question_layout"],
    )


class _SourceStop:
    """End one clip locally while also observing a shared run interruption."""

    def __init__(self, shared):
        self.local, self.shared = Event(), shared

    def is_set(self):
        return self.local.is_set() or self.shared.is_set()

    def set(self):
        self.local.set()

    def wait(self, timeout):
        return self.local.wait(timeout) or self.shared.is_set()


class _StreamRun:
    def __init__(self, config, source, directory, run_id, stop):
        self.config, self.source, self.stop = config, source, stop
        self.process_config = _process_config(config, source, run_id)
        self.recorder = EventStreamRecorder(directory, {
            "schema_version": "videojev-real-file-v1", "stream": source,
            "stream_session_id": self.process_config.stream_session_id,
            "clock": "parent_process_time.monotonic", "model": MODEL,
            "revision": REVISION,
            "settings": {key: value for key, value in config.items()
                         if key not in {"streams", "endpoint"}},
            "event_policy": config["event_policy"], "ground_truth_input": False,
            "tracking_enabled": False, "fusion_enabled": False,
        })
        self.branch = None
        self.stream = None
        self.error = None
        self.shared_failure = False
        self.range_end_reached = False
        self.original_eof = False
        self.first_ingress = self.last_ingress = None
        self.first_source_time = self.last_source_time = None
        self.last_raw_index = None
        self.encoded = self.offered = self.admitted = 0

    def _input(self, row):
        end = self.source["end_seconds"]
        if end is not None and row["timestamp_seconds"] >= end:
            self.range_end_reached = True
            self.stream.stop()
            return
        try:
            self.recorder.source_input(row)
        except Exception:
            self.stop.set()
            raise
        self.first_ingress = row["ingress_monotonic"] if self.first_ingress is None else self.first_ingress
        self.first_source_time = row["timestamp_seconds"] if self.first_source_time is None else self.first_source_time
        self.last_ingress = row["ingress_monotonic"]
        self.last_source_time = row["timestamp_seconds"]
        self.last_raw_index = row["raw_frame_index"]

    def _report(self, report):
        try:
            reference = self.recorder.independent_report(report)
        except Exception:
            self.stop.set()
            raise
        error = report.get("timing", {}).get("response_diagnostics", {}).get("native_service_error")
        if error in NATIVE_SHARED_ERRORS:
            self.shared_failure = True
            self.error = error
            self.stop.set()
        return reference

    def _publication(self, record, *, report_reference):
        try:
            self.recorder.publication(record, report_reference=report_reference)
        except Exception:
            self.stop.set()
            raise

    def prepare(self):
        try:
            self.stream = VideoFileStream(
                self.source["video"], display_fps=120, start_seconds=self.source["start_seconds"],
                stop_event=_SourceStop(self.stop), input_observer=self._input,
            )
            info = self.stream.info
            if (not math.isfinite(info.source_fps) or info.source_fps <= 0
                    or self.source["start_seconds"] >= info.duration_seconds):
                raise ValueError("video timing cannot cover the requested range")
            # Candidate selection below is the sole pre-encoding sampler. The
            # paced source still exposes every raw frame to the input observer.
            self.stream.display_fps = info.source_fps
            self.branch = IndependentVLMBranch(
                self.process_config, report_observer=self._report,
                rejected_report_observer=self.recorder.rejected_report,
                publication_observer=self._publication,
            )
            self.branch.start()
        except Exception as exc:
            self.error = "source_prepare_failed:" + type(exc).__name__
            try:
                self.recorder.failure(self.error)
            except Exception:
                self.stop.set()
                self.error = "recording_failed"

    def execute(self, origin):
        runtime = {}
        drained = False
        sample_complete = False
        started = time.monotonic()
        try:
            if self.error is None:
                if not self.stop.wait(max(0, origin + self.source["offset_seconds"] - time.monotonic())):
                    next_candidate = -math.inf
                    iterator = self.stream.frames()
                    try:
                        for frame in iterator:
                            if self.stop.is_set() or self.range_end_reached:
                                self.stream.stop()
                                break
                            timestamp = frame.timestamp_seconds
                            if timestamp + 1e-9 < next_candidate:
                                continue
                            # Encode once at the same cadence consumed by offer();
                            # another independently phased sampler would halve FPS.
                            next_candidate = timestamp + 1.0 / self.config["candidate_fps"]
                            jpeg, _width, _height = encode_jpeg(
                                frame.bgr, max_image_side=self.config["max_image_side"],
                                quality=self.config["jpeg_quality"],
                            )
                            self.encoded += 1
                            self.offered += 1
                            self.admitted += int(self.branch.offer(jpeg, timestamp, frame.index))
                            state = self.branch.stats()
                            if state.get("state") == "failed":
                                self.error = "event_branch_failed:" + str(state.get("error_code", "unknown"))
                                break
                        else:
                            if not self.stream.stopped:
                                self.original_eof = (
                                    self.last_raw_index is not None
                                    and self.last_raw_index >= self.stream.info.frame_count - 1
                                )
                                if not self.original_eof:
                                    self.error = "decoder_ended_before_metadata_frame_count"
                    finally:
                        iterator.close()
                    if self.range_end_reached and not self.stop.is_set():
                        sample_complete = True
                    elif self.original_eof and not self.stop.is_set():
                        end = self.source["end_seconds"]
                        # A requested range beyond the actual media end is partial.
                        sample_complete = (end is None or end <= self.stream.info.duration_seconds + 1e-9)
                        if not sample_complete:
                            self.error = "media_ended_before_requested_range"
                    self.branch.finish_source()
                    deadline = time.monotonic() + self.config["drain_timeout_seconds"]
                    while time.monotonic() < deadline:
                        runtime = self.branch.stats()
                        if runtime["state"] in {"drained", "failed"}:
                            drained = runtime["state"] == "drained"
                            if not drained and self.error is None:
                                self.error = "event_branch_failed:" + str(runtime.get("error_code", "unknown"))
                            break
                        time.sleep(0.02)
                    if not drained and self.error is None:
                        self.error = "event_branch_drain_timeout"
        except Exception as exc:
            self.error = "source_failed:" + type(exc).__name__
            if self.recorder.failed:
                self.stop.set()
            else:
                try:
                    self.recorder.failure(self.error)
                except Exception:
                    self.stop.set()
                    self.error = "recording_failed"
        finally:
            if self.branch is not None:
                if not runtime:
                    runtime = self.branch.stats()
                self.branch.close(timeout_seconds=12.0)
            cleanup_complete = self.branch is None or self.branch.close_complete()
            if not cleanup_complete and self.error is None:
                self.error = "event_branch_cleanup_incomplete"
            result = {
                "stream_id": self.source["id"], "sample_complete": sample_complete,
                "original_media_eof": self.original_eof, "range_end_reached": self.range_end_reached,
                "events_drained": drained,
                "cleanup_complete": cleanup_complete,
                "execution_complete": sample_complete and drained and self.error is None,
                "error": self.error, "shared_failure": self.shared_failure,
                "interrupted": self.stop.is_set(), "wall_seconds": time.monotonic() - started,
                "first_ingress_monotonic": self.first_ingress,
                "last_ingress_monotonic": self.last_ingress,
                "first_source_seconds": self.first_source_time, "last_source_seconds": self.last_source_time,
                "encoded_frames": self.encoded, "offered_frames": self.offered,
                "admitted_frames": self.admitted, "runtime_statistics": runtime,
                "input_statistics": self.stream.input_timing_statistics if self.stream else None,
                "media": asdict(self.stream.info) if self.stream else None,
                "record_counts": self.recorder.summary(),
            }
            try:
                if self.recorder.failed:
                    raise OSError("stream recording previously failed")
                self.recorder.finish(result)
            except Exception as exc:
                self.stop.set()
                result["execution_complete"] = False
                result["error"] = "recording_failed:" + type(exc).__name__
            finally:
                try:
                    self.recorder.close()
                except Exception as exc:
                    self.stop.set()
                    result["execution_complete"] = False
                    result["error"] = "recording_close_failed:" + type(exc).__name__
        return result


def run(config, output):
    """Run explicitly listed real streams once; no automatic retry or warmup."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    stop = Event()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    publish_once(root / "run.json", {
        "schema_version": "videojev-real-multistream-v1", "run_id": run_id,
        "config": {key: value for key, value in config.items() if key != "endpoint"},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "revision": REVISION,
        "ground_truth_input": False, "realtime": True, "playback_speed": 1.0,
    })
    owners, results, cleanup_errors = [], [], []
    previous_handlers = {}

    def interrupt(_signum, _frame):
        stop.set()
        for owner in owners:
            if owner.stream is not None:
                owner.stream.stop()

    for name in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[name] = signal.signal(name, interrupt)
    try:
        for source in config["streams"]:
            owners.append(_StreamRun(config, source, root / source["id"], run_id, stop))
        with ThreadPoolExecutor(max_workers=len(owners), thread_name_prefix="videojev-source") as pool:
            list(pool.map(lambda owner: owner.prepare(), owners))
            origin = time.monotonic() + config["start_delay_seconds"]
            tasks = {pool.submit(owner.execute, origin): owner for owner in owners}
            for future in as_completed(tasks):
                try:
                    result = future.result()
                except Exception as exc:
                    interrupt(None, None)
                    result = {"stream_id": tasks[future].source["id"],
                              "execution_complete": False,
                              "error": "source_worker_failed:" + type(exc).__name__}
                results.append(result)
                print(json.dumps({"stream_id": result["stream_id"],
                                  "execution_complete": result["execution_complete"],
                                  "error": result["error"]}), flush=True)
    finally:
        interrupt(None, None)
        try:
            for owner in owners:
                try:
                    if owner.branch is not None:
                        owner.branch.close(timeout_seconds=12.0)
                except Exception as exc:
                    cleanup_errors.append({"stream_id": owner.source["id"],
                                           "stage": "event_branch",
                                           "error_type": type(exc).__name__})
                try:
                    owner.recorder.close()
                except Exception as exc:
                    cleanup_errors.append({"stream_id": owner.source["id"],
                                           "stage": "recording",
                                           "error_type": type(exc).__name__})
        finally:
            for name, handler in previous_handlers.items():
                signal.signal(name, handler)
    results.sort(key=lambda item: item["stream_id"])
    outcome = {"run_id": run_id, "stream_count": len(owners), "streams": results,
               "cleanup_errors": cleanup_errors,
               "execution_complete": not cleanup_errors and len(results) == len(owners) and all(
                   item["execution_complete"] for item in results),
               "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    publish_once(root / "completion.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_run_config(args.config)
    if not args.execute:
        print(json.dumps({"status": "not_started", "config": config}, indent=2))
        return
    if args.output is None:
        parser.error("--execute requires a new --output directory")
    outcome = run(config, args.output)
    raise SystemExit(0 if outcome["execution_complete"] else 1)


if __name__ == "__main__":
    main()
