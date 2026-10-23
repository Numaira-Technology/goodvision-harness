"""Score explicit original-frame windows through the existing Harness adapter.

This is fixed-input classification, not paced-stream reaction timing. Reference
labels are intentionally separate from both input files and model requests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
from threading import Event
import time

import cv2

from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.runtime.scene import TemporalInspection
from goodvision_harness.runtime.scheduling import EvidenceFrame, InspectionRequest
from goodvision_harness.video_to_stream import encode_jpeg
from goodvision_harness.videojev_model import NATIVE_SHARED_ERRORS, VideoJevInspector

from .recording import publish_once
from .run import _process_config, load_run_config


def load_windows(path, config):
    windows = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = {source["id"] for source in config["streams"]}
    names = set()
    for window in windows:
        if set(window) != {"id", "stream_id", "frame_indices", "timestamps_seconds"}:
            raise ValueError("windows contain observation fields only")
        indices, times = window["frame_indices"], window["timestamps_seconds"]
        if (not isinstance(window["id"], str) or not window["id"]
                or window["id"] in names or window["stream_id"] not in sources
                or len(indices) != config["window_frames"] or len(times) != len(indices)
                or any(type(index) is not int or index < 0 for index in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))):
            raise ValueError("provide distinct ordered source-frame windows")
        names.add(window["id"])
    return windows


def original_frames(source, window, config, session):
    capture = cv2.VideoCapture(source["video"])
    selected = []
    wanted = dict(zip(window["frame_indices"], window["timestamps_seconds"]))
    try:
        if not capture.isOpened():
            raise ValueError("source video could not be opened")
        for index in range(window["frame_indices"][-1] + 1):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError("source ended before the requested frame")
            if index in wanted:
                jpeg, _width, _height = encode_jpeg(
                    bgr, max_image_side=config["max_image_side"],
                    quality=config["jpeg_quality"],
                )
                selected.append(EvidenceFrame(session, index, wanted[index], jpeg))
    finally:
        capture.release()
    return tuple(selected)


def run(config, windows, output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    sources = {source["id"]: source for source in config["streams"]}
    process_config = _process_config(config, {"id": "points"}, stamp)
    session = process_config.stream_session_id
    inspector = VideoJevInspector(process_config)
    stop = Event()
    handlers = {}
    rows = []
    shared_failure = None
    publish_once(root / "run.json", {
        "run_id": stamp, "mode": "fixed_original_frames", "realtime": False,
        "ground_truth_input": False, "session": session, "windows": windows,
        "config": {key: value for key, value in config.items() if key != "endpoint"},
    })
    for name in (signal.SIGINT, signal.SIGTERM):
        handlers[name] = signal.signal(name, lambda _signum, _frame: stop.set())
    try:
        with (root / "reports.jsonl").open("x", encoding="utf-8") as stream:
            for ordinal, window in enumerate(windows, 1):
                if stop.is_set():
                    break
                request_id = stamp + "-" + str(ordinal)
                row = {"window_id": window["id"], "source_id": window["stream_id"],
                       "request_id": request_id, "frame_indices": window["frame_indices"],
                       "timestamps_seconds": window["timestamps_seconds"],
                       "state": "failed", "judgments": []}
                attempted = False
                cooldown = None
                started = time.monotonic()
                try:
                    frames = original_frames(sources[window["stream_id"]], window, config, session)
                    row["image_prepare_seconds"] = time.monotonic() - started
                    request = InspectionRequest(
                        request_id, (), "independent_source", tuple(config["labels"]),
                        frames[0].timestamp_seconds, frames[-1].timestamp_seconds,
                    )
                    job = TemporalInspection(
                        session, window["stream_id"], request, frames,
                        event_sensitivities=process_config.sensitivities,
                        allowed_event_labels=tuple(config["labels"]),
                        scene_only_event_labels=tuple(config["labels"]),
                        event_definitions=process_config.event_definitions,
                    )
                    attempted = True
                    verdicts = inspector.inspect(job)
                    row["state"] = "completed"
                    row["judgments"] = [{"event_label": item.event_label,
                                         "observed_detected": item.detected}
                                        for item in verdicts]
                except Exception as error:
                    row["error_code"] = getattr(error, "code", type(error).__name__)
                    if attempted:
                        service_error = inspector.last_response_diagnostics.get("native_service_error")
                        if service_error in NATIVE_SHARED_ERRORS:
                            shared_failure = service_error
                        if isinstance(error, ModelAPIError) and error.status_code in {401, 403}:
                            shared_failure = "authorization_unavailable"
                        if isinstance(error, ModelAPIError) and error.status_code == 429:
                            try:
                                retry_after = float(getattr(error, "retry_after", None))
                            except (ValueError, TypeError, OverflowError):
                                retry_after = 60.0
                            cooldown = max(60.0, retry_after) if math.isfinite(retry_after) else 60.0
                row["client_elapsed_seconds"] = time.monotonic() - started
                if attempted:
                    row["http_timing"] = dict(inspector.last_http_timing)
                    row["inspection_timing"] = dict(inspector.last_inspection_timing)
                    row["response_diagnostics"] = dict(inspector.last_response_diagnostics)
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                stream.flush()
                rows.append(row)
                if cooldown is not None:
                    stop.wait(cooldown)
                if shared_failure is not None:
                    break
    finally:
        inspector.close()
        for name, previous in handlers.items():
            signal.signal(name, previous)
    outcome = {
        "run_id": stamp, "mode": "fixed_original_frames", "realtime": False,
        "execution_complete": len(rows) == len(windows) and not stop.is_set()
                              and shared_failure is None,
        "planned_windows": len(windows), "processed_windows": len(rows),
        "completed_windows": sum(row["state"] == "completed" for row in rows),
        "failed_windows": sum(row["state"] == "failed" for row in rows),
        "shared_failure": shared_failure,
        "streams": [],
    }
    publish_once(root / "completion.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"state": "not_started", "mode": "fixed_original_frames"}))
        return 0
    config = load_run_config(args.config)
    outcome = run(config, load_windows(args.windows, config), args.output)
    print(json.dumps(outcome), flush=True)
    return 0 if outcome["execution_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
